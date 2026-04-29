#pragma once
/**
 * map_runner_persistent.hpp — MapTRv2 持久化 GPU 服务器 C++ 客户端
 *
 * 在 run_pipeline 启动时启动 maptr_server.py（maptr conda 环境），
 * 通过 Unix domain socket 向服务器提交推理请求（fire-and-forget）。
 *
 * 特性：
 *   - 模型只加载一次（Python 服务器端），消除每帧 subprocess 启动开销
 *   - GPU 推理（maptr_server.py 使用 cuda:0）
 *   - 最新帧语义：C++ 异步提交，服务器丢弃积压旧帧
 *   - submit_frame() 完全非阻塞（后台线程发送）
 *   - read_result()  与 MapRunner 接口兼容（读 map_result.json）
 *
 * 使用方式：
 * @code
 *   MapRunnerPersistentConfig cfg;
 *   cfg.ckpt_path   = "model/maptr/maptr_nano_r18_110e.pth";
 *   cfg.score_thr   = 0.3f;
 *   MapRunnerPersistent runner(cfg);
 *
 *   if (!runner.init()) { fprintf(stderr, "服务器启动失败\n"); }
 *
 *   // 每帧：提交（非阻塞）→ BEV 推理 → 读取结果
 *   runner.submit_frame("outputs/frames/frame_00000");
 *   // ... BEV 推理 ...
 *   MapResult map;
 *   runner.read_result("outputs/frames/frame_00000", map);  // 非阻塞
 *
 *   runner.shutdown();
 * @endcode
 */

#include "perception_types.hpp"
#include "map_runner.hpp"  // for MapResult, read_result() 实现

#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <memory>

namespace fastbev {
namespace perception {

// ─── 配置 ──────────────────────────────────────────────────────────────────

struct MapRunnerPersistentConfig {
    // MapTRv2 权重（model 模式）
    std::string ckpt_path    = "model/maptr/maptr_nano_r18_110e.pth";
    std::string cfg_path     = "model/maptr/config/maptr_nano_r18_110e.py";

    // 置信度阈值
    float score_thr          = 0.3f;

    // conda 环境名（maptr 环境含 mmcv 1.x）
    std::string maptr_env    = "maptr";

    // Unix socket 路径（空 = 自动生成 /tmp/fastbev_maptr_<pid>.sock）
    std::string socket_path;

    // 等待服务器启动的超时时间（秒）
    int startup_timeout_s    = 120;

    // 推理设备
    std::string device       = "cuda:0";

    // 是否打印详细日志
    bool verbose             = false;
};

// ─── 客户端 ────────────────────────────────────────────────────────────────

class MapRunnerPersistent {
public:
    explicit MapRunnerPersistent(const MapRunnerPersistentConfig& cfg
                                 = MapRunnerPersistentConfig{});
    ~MapRunnerPersistent();

    /**
     * 启动 maptr_server.py 服务器进程并等待连接就绪。
     *
     * @return true = 成功连接并收到 "READY"；false = 超时或启动失败
     */
    bool init();

    /**
     * 向服务器提交推理请求（非阻塞，fire-and-forget）。
     * 由后台线程异步发送，不阻塞 BEV 关键路径。
     *
     * @param frame_dir  帧目录路径（需含 meta.json + 6 路图像）
     */
    void submit_frame(const std::string& frame_dir);

    /**
     * 读取已有的 map_result.json（不重新推理）。
     * 与 MapRunner::read_result 接口兼容。
     *
     * @return true = 成功读取并解析
     */
    bool read_result(const std::string& frame_dir, MapResult& result) const;

    /**
     * 优雅关闭：发送 QUIT，等待服务器响应 BYE，再等待子进程退出。
     */
    void shutdown();

    bool is_alive() const { return server_alive_.load(); }

private:
    // 后台发送线程（将 pending_dir 发送给服务器）
    void _sender_thread();

    // 等待 socket 文件出现
    bool _wait_socket(int timeout_s) const;

    MapRunnerPersistentConfig  cfg_;
    std::string                sock_path_;
    int                        sock_fd_  = -1;
    pid_t                      server_pid_ = -1;

    // 最新帧缓冲（最新帧语义：始终只保留一个待发送的 frame_dir）
    std::string                pending_dir_;
    bool                       pending_fresh_ = false;
    std::mutex                 pending_mtx_;
    std::condition_variable    pending_cv_;
    std::atomic<bool>          stop_sender_{false};
    std::thread                sender_thread_;
    std::atomic<bool>          server_alive_{false};

    // 复用 MapRunner 的 read_result 和 _parse_json 实现
    MapRunner                  reader_;
};

}  // namespace perception
}  // namespace fastbev
