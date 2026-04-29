/**
 * map_runner_persistent.cpp — MapRunnerPersistent 实现
 *
 * 负责：
 *   1. fork/exec maptr_server.py（conda run -n maptr）
 *   2. 通过 Unix domain socket 与服务器通信
 *   3. 后台发送线程（最新帧语义）
 *   4. read_result() 通过 MapRunner 实现（读 map_result.json）
 */

#include "map_runner_persistent.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <sstream>
#include <chrono>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>

namespace fastbev {
namespace perception {

// ─── 内部工具 ────────────────────────────────────────────────────────────────

static bool _exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

static bool _send_line(int fd, const std::string& msg) {
    std::string line = msg + "\n";
    ssize_t n = send(fd, line.c_str(), line.size(), MSG_NOSIGNAL);
    return n == static_cast<ssize_t>(line.size());
}

static bool _recv_line(int fd, std::string& out, int timeout_s = 5) {
    // 设置非阻塞超时（使用 select）
    fd_set rdset;
    FD_ZERO(&rdset);
    FD_SET(fd, &rdset);
    struct timeval tv = { static_cast<time_t>(timeout_s), 0 };
    int ret = select(fd + 1, &rdset, nullptr, nullptr, &tv);
    if (ret <= 0) return false;

    out.clear();
    char ch;
    while (true) {
        ssize_t n = recv(fd, &ch, 1, 0);
        if (n <= 0) return !out.empty();
        if (ch == '\n') return true;
        out += ch;
    }
}

// ─── 构造 / 析构 ─────────────────────────────────────────────────────────────

MapRunnerPersistent::MapRunnerPersistent(const MapRunnerPersistentConfig& cfg)
    : cfg_(cfg)
{
    // 生成 socket 路径
    if (!cfg_.socket_path.empty()) {
        sock_path_ = cfg_.socket_path;
    } else {
        char buf[128];
        snprintf(buf, sizeof(buf), "/tmp/fastbev_maptr_%d.sock",
                 static_cast<int>(getpid()));
        sock_path_ = buf;
    }

    // reader_ 仅用于 read_result（score_thr 不影响读取结果）
    MapRunnerConfig rcfg;
    rcfg.verbose = false;
    reader_      = MapRunner(rcfg);
}

MapRunnerPersistent::~MapRunnerPersistent() {
    if (server_alive_.load())
        shutdown();
}

// ─── init ─────────────────────────────────────────────────────────────────────

bool MapRunnerPersistent::init()
{
    // 清理旧 socket 文件
    if (_exists(sock_path_))
        ::unlink(sock_path_.c_str());

    // 构造 Python 服务器命令（conda run -n maptr）
    // 绝对路径：cwd + relative 或 已经是绝对路径
    std::string ckpt = cfg_.ckpt_path;
    std::string cfgp = cfg_.cfg_path;

    char cwd[4096] = {};
    if (getcwd(cwd, sizeof(cwd)) == nullptr) cwd[0] = '\0';

    if (!ckpt.empty() && ckpt[0] != '/')
        ckpt = std::string(cwd) + "/" + ckpt;
    if (!cfgp.empty() && cfgp[0] != '/')
        cfgp = std::string(cwd) + "/" + cfgp;

    // 找到 maptr_server.py
    std::string script =
        std::string(cwd) + "/tools/maptr/maptr_server.py";
    if (!_exists(script)) {
        fprintf(stderr, "[MapRunnerPersistent] 找不到: %s\n", script.c_str());
        return false;
    }

    char score_str[32];
    snprintf(score_str, sizeof(score_str), "%.4f", cfg_.score_thr);

    // 构造命令字符串（conda run 会新建 shell，使用 system() 最简单）
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "conda run -n %s python %s"
        " --socket %s"
        " --config %s"
        " --checkpoint %s"
        " --score-thr %s"
        " --device %s"
        "%s"
        " >> /tmp/fastbev_maptr_server.log 2>&1 &",
        cfg_.maptr_env.c_str(),
        script.c_str(),
        sock_path_.c_str(),
        cfgp.c_str(),
        ckpt.c_str(),
        score_str,
        cfg_.device.c_str(),
        cfg_.verbose ? " --verbose" : "");

    printf("[MapRunnerPersistent] 启动 MapTR GPU 服务器:\n  %s\n", cmd);
    printf("[MapRunnerPersistent] 日志: /tmp/fastbev_maptr_server.log\n");
    fflush(stdout);

    if (std::system(cmd) != 0) {
        fprintf(stderr, "[MapRunnerPersistent] 启动命令失败\n");
        return false;
    }

    // ── 等待 socket 文件出现（服务器进程启动并绑定） ─────────────────────────
    printf("[MapRunnerPersistent] 等待模型加载（最多 %ds）...\n",
           cfg_.startup_timeout_s);
    fflush(stdout);

    if (!_wait_socket(cfg_.startup_timeout_s)) {
        fprintf(stderr, "[MapRunnerPersistent] 超时：socket 未出现 %s\n",
                sock_path_.c_str());
        fprintf(stderr, "  请检查日志: /tmp/fastbev_maptr_server.log\n");
        return false;
    }

    // ── 连接 socket ───────────────────────────────────────────────────────────
    sock_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd_ < 0) {
        perror("[MapRunnerPersistent] socket()");
        return false;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(sock_fd_, reinterpret_cast<struct sockaddr*>(&addr),
                  sizeof(addr)) < 0) {
        perror("[MapRunnerPersistent] connect()");
        ::close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }

    // ── 等待服务器就绪（收到 "READY"） ───────────────────────────────────────
    std::string resp;
    if (!_recv_line(sock_fd_, resp, cfg_.startup_timeout_s)) {
        fprintf(stderr, "[MapRunnerPersistent] 等待 READY 超时\n");
        ::close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }
    if (resp != "READY") {
        fprintf(stderr, "[MapRunnerPersistent] 意外响应: %s\n", resp.c_str());
        ::close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }

    printf("[MapRunnerPersistent] MapTR GPU 服务器已就绪\n");
    fflush(stdout);

    server_alive_.store(true);

    // ── 启动后台发送线程 ──────────────────────────────────────────────────────
    sender_thread_ = std::thread(&MapRunnerPersistent::_sender_thread, this);
    return true;
}

// ─── _wait_socket ────────────────────────────────────────────────────────────

bool MapRunnerPersistent::_wait_socket(int timeout_s) const
{
    auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        if (_exists(sock_path_)) return true;
        usleep(200000);  // 轮询间隔 200ms
    }
    return false;
}

// ─── submit_frame ────────────────────────────────────────────────────────────

void MapRunnerPersistent::submit_frame(const std::string& frame_dir)
{
    if (!server_alive_.load()) return;
    {
        std::unique_lock<std::mutex> lk(pending_mtx_);
        pending_dir_   = frame_dir;
        pending_fresh_ = true;
    }
    pending_cv_.notify_one();
}

// ─── _sender_thread ──────────────────────────────────────────────────────────
//
// 后台线程：等待 pending_dir 更新 → 发送给服务器 → 重复
// 最新帧语义：若发送期间又有新帧提交，下次循环直接发送最新帧

void MapRunnerPersistent::_sender_thread()
{
    while (!stop_sender_.load()) {
        std::string dir;
        {
            std::unique_lock<std::mutex> lk(pending_mtx_);
            pending_cv_.wait_for(lk, std::chrono::milliseconds(200),
                [this]{ return pending_fresh_ || stop_sender_.load(); });
            if (!pending_fresh_) continue;
            dir            = pending_dir_;
            pending_fresh_ = false;
        }
        if (dir.empty() || sock_fd_ < 0) continue;

        std::string msg = "INFER " + dir;
        if (!_send_line(sock_fd_, msg)) {
            if (!stop_sender_.load()) {
                fprintf(stderr, "[MapRunnerPersistent] 发送失败，服务器可能已退出\n");
                server_alive_.store(false);
            }
            break;
        }
    }
}

// ─── read_result ──────────────────────────────────────────────────────────────

bool MapRunnerPersistent::read_result(const std::string& frame_dir,
                                       MapResult&         result) const
{
    return reader_.read_result(frame_dir, result);
}

// ─── shutdown ────────────────────────────────────────────────────────────────

void MapRunnerPersistent::shutdown()
{
    server_alive_.store(false);

    // 停止发送线程
    stop_sender_.store(true);
    pending_cv_.notify_all();
    if (sender_thread_.joinable())
        sender_thread_.join();

    // 发送 QUIT 并等待 BYE
    if (sock_fd_ >= 0) {
        _send_line(sock_fd_, "QUIT");
        std::string resp;
        _recv_line(sock_fd_, resp, 3);   // 最多等待 3s
        ::close(sock_fd_);
        sock_fd_ = -1;
    }

    printf("[MapRunnerPersistent] 已关闭\n");
    fflush(stdout);
}

}  // namespace perception
}  // namespace fastbev
