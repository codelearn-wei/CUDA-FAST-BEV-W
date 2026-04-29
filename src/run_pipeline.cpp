/**
 * run_pipeline.cpp — 全感知流水线（流式 / 实时部署版）
 *
 * 架构（模拟 ROS 话题 pub/sub，待传感器接入后替换 FrameSource 即可）：
 *
 *   ┌────────────────────────────────────────────────────────────┐
 *   │  FrameSource（模拟 ROS /camera_array 话题）                  │
 *   │    读取 NuScenes 帧目录 → 按仿真帧率写入 FrameBuffer         │
 *   │    接入真实传感器时：替换为 camera::CameraDriver::subscribe() │
 *   └──────────────────────┬─────────────────────────────────────┘
 *                          │ 最新帧（非阻塞 pop）
 *         ┌────────────────▼────────────────┐    ┌──────────────────────────────┐
 *         │  BEV Worker（主线程 / GPU）       │    │  MapTR Worker（后台线程/GPU）  │
 *         │  ImagePreprocessor::from_frame_  │    │  MapRunner::run_batch()       │
 *         │  dir() → TensorRT 推理           │    │  Python subprocess（并行）     │
 *         │  → 多目标跟踪（含 ego 补偿）       │    │  写入 map_result.json          │
 *         └────────────────┬────────────────┘    └──────────────────────────────┘
 *                          │                              ↓ 非阻塞 read_result()
 *         ┌────────────────▼────────────────────────────────────┐
 *         │  ResultWriter                                        │
 *         │  tracks.json / result.json / joint_result.json       │
 *         │  兼容 video_demo.py --joint 和 video_visualize_tracking.py │
 *         └─────────────────────────────────────────────────────┘
 *
 * 编译：在 CMakeLists.txt 添加 run_pipeline target（参见文末说明）
 *
 * 用法：
 *   ./build/run_pipeline outputs/frames resnet18int8 int8 [选项]
 *
 * 必选参数：
 *   frames_dir   帧目录（frame_XXXXX 子目录，由 nuscenes_adapter.py 生成）
 *   model        模型名称（resnet18 / resnet18int8）
 *
 * 可选参数：
 *   precision         fp16 / int8（默认 fp16）
 *   --fps N           仿真帧率（默认 20，影响帧源发布速度）
 *   --score-thr F     置信度阈值（默认 0.35）
 *   --map-mode S      auto|gt|model（默认 auto）
 *   --ckpt PATH       MapTR checkpoint（默认 model/maptr/maptr_nano_r18_110e.pth）
 *   --nuscenes-dir S  NuScenes 数据目录（默认 data/nuscenes）
 *   --skip-map        跳过 MapTR
 *   --save-json       保存 JSON 结果（默认：开启）
 *   --no-save-json    关闭 JSON 输出
 *   --no-warmup       跳过 TRT warmup
 *   --verbose         打印每帧调试信息
 *
 * 示例：
 *   # 标准运行（BEV + MapTR 并行，保存 JSON）
 *   ./build/run_pipeline outputs/frames resnet18int8 int8 \
 *       --map-mode model --ckpt model/maptr/maptr_nano_r18_110e.pth
 *
 *   # 仅 BEV（跳过 MapTR）
 *   ./build/run_pipeline outputs/frames resnet18int8 int8 --skip-map
 *
 *   # 生成可视化视频（运行后）
 *   python tools/video_demo.py --frames-dir outputs/frames \
 *       --out-dir outputs/video --joint --fps 6
 */

#include "perception/perception_types.hpp"
#include "perception/image_preprocessor.hpp"
#include "perception/map_runner.hpp"
#include "perception/map_runner_persistent.hpp"
#include "perception/prefetch_loader.hpp"
#include "pipeline/perception_pipeline.hpp"
#include "camera/camera_frame.hpp"
#include "tracking/tracker.hpp"
#include "tracking/track.hpp"
#include "fastbev/fastbev.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>   // usleep

using namespace fastbev;
using namespace fastbev::perception;
using namespace fastbev::pipeline;

// ══════════════════════════════════════════════════════════════════════════════
// 工具函数
// ══════════════════════════════════════════════════════════════════════════════

static bool str_eq(const char* a, const char* b) { return strcmp(a, b) == 0; }

static bool path_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

/// 从 JSON 字符串中提取 "key": <number>
static double json_get_double(const std::string& json,
                              const std::string& key,
                              double def = 0.0)
{
    std::string pat = "\"" + key + "\"";
    size_t kp = json.find(pat);
    if (kp == std::string::npos) return def;
    size_t vp = json.find_first_of("-0123456789", kp + pat.size());
    if (vp == std::string::npos) return def;
    try { return std::stod(json.substr(vp)); } catch (...) { return def; }
}

/// 从 JSON 字符串中提取 "key": [v0, v1, v2]
static bool json_get_array3(const std::string& json,
                             const std::string& key,
                             double out[3])
{
    std::string pat = "\"" + key + "\"";
    size_t kp = json.find(pat);
    if (kp == std::string::npos) return false;
    size_t lb = json.find('[', kp + pat.size());
    size_t rb = json.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return false;
    std::string arr = json.substr(lb + 1, rb - lb - 1);
    int idx = 0;
    size_t pos = 0;
    while (idx < 3 && pos < arr.size()) {
        size_t sp = arr.find_first_of("-0123456789", pos);
        if (sp == std::string::npos) break;
        try {
            size_t end = 0;
            out[idx++] = std::stod(arr.substr(sp), &end);
            pos = sp + end;
        } catch (...) { break; }
    }
    return idx == 3;
}

/**
 * 将 MapResult 的地图点从 src_ego 坐标系变换到 dst_ego 坐标系。
 *
 * 变换路径：src_ego → global → dst_ego
 *   p_global = R(src_yaw) * p_src + src_trans
 *   p_dst    = R(-dst_yaw) * (p_global - dst_trans)
 *
 * 即：  cx =  cos(dst)*dx + sin(dst)*dy
 *        cy = -sin(dst)*dx + cos(dst)*dy
 *   其中 dx = p_global_x - dst_x
 */
static MapResult compensate_map_ego(const MapResult& src,
                                    const tracking::EgoPose& src_ego,
                                    const tracking::EgoPose& dst_ego)
{
    MapResult dst = src;
    dst.source = "stale";   // 标记为复用帧

    if (!src_ego.valid || !dst_ego.valid)
        return dst;   // 无 ego 信息，原样返回

    const double cs = std::cos(src_ego.yaw), ss = std::sin(src_ego.yaw);
    const double cd = std::cos(dst_ego.yaw), sd = std::sin(dst_ego.yaw);

    for (auto& elem : dst.elements) {
        for (auto& pt : elem.points) {
            // src_ego → global
            double gx = src_ego.x + cs * pt[0] - ss * pt[1];
            double gy = src_ego.y + ss * pt[0] + cs * pt[1];
            // global → dst_ego  (multiply by R(-dst_yaw) = [[cd, sd],[-sd, cd]])
            double dx = gx - dst_ego.x;
            double dy = gy - dst_ego.y;
            pt[0] = static_cast<float>( cd * dx + sd * dy);
            pt[1] = static_cast<float>(-sd * dx + cd * dy);
        }
    }
    return dst;
}

/// 从 frame_dir/meta.json 读取 ego 全局位姿
static tracking::EgoPose read_ego_pose(const std::string& frame_dir)
{
    tracking::EgoPose pose;
    std::string meta_path = frame_dir + "/meta.json";
    std::ifstream f(meta_path);
    if (!f.is_open()) return pose;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();
    double trans[3] = {0, 0, 0};
    if (json_get_array3(content, "ego_translation_global", trans)) {
        pose.x = trans[0];
        pose.y = trans[1];
    }
    pose.yaw   = json_get_double(content, "ego_yaw_global", 0.0);
    pose.valid = true;
    return pose;
}

/// 收集帧目录列表（按名称升序排列）
static std::vector<std::string> collect_frame_dirs(const std::string& base)
{
    std::vector<std::string> dirs;
    DIR* d = opendir(base.c_str());
    if (!d) return dirs;
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        std::string name(entry->d_name);
        if (name.rfind("frame_", 0) == 0) {
            std::string full = base + "/" + name;
            struct stat st;
            if (stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                dirs.push_back(full);
        }
    }
    closedir(d);
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

// ══════════════════════════════════════════════════════════════════════════════
// FrameBuffer — 最新帧缓冲区（单槽 lock-free 语义，模拟 ROS 话题 latest()）
//
// 生产者（FrameSource 线程）写入最新帧；
// 消费者（BEV 线程）取走最新帧（若无新帧则返回 false，不阻塞）。
// 当传感器接入后，FrameSource 替换为真实相机驱动，此接口保持不变。
// ══════════════════════════════════════════════════════════════════════════════

struct FrameBuffer {
    // 帧的所有权用 frame_dir 路径表示（CameraFrame 的内存由 BEV Worker 分配）
    std::string     latest_dir;
    uint64_t        latest_seq = 0;     // 递增序号
    uint64_t        consumed_seq = 0;   // BEV 消费的最新序号
    std::mutex      mtx;
    std::condition_variable cv;
    bool            done = false;       // FrameSource 已发布完所有帧

    /// 生产者：发布新帧（非阻塞，直接覆盖旧帧）
    void publish(const std::string& frame_dir) {
        std::unique_lock<std::mutex> lk(mtx);
        latest_dir = frame_dir;
        ++latest_seq;
        cv.notify_one();
    }

    /// 消费者：等待新帧（timeout_us 微秒内返回）
    /// 返回 true 表示拿到新帧，frame_dir_out 已填入；false 表示超时
    bool wait_pop(std::string& frame_dir_out, uint64_t& seq_out,
                  int timeout_us = 100000 /* 100ms */)
    {
        std::unique_lock<std::mutex> lk(mtx);
        auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::microseconds(timeout_us);
        bool ok = cv.wait_until(lk, deadline,
            [this]{ return latest_seq > consumed_seq || done; });
        if (!ok || (done && latest_seq <= consumed_seq)) return false;
        frame_dir_out = latest_dir;
        seq_out       = latest_seq;
        consumed_seq  = latest_seq;
        return true;
    }

    /// 生产者：通知已发布完毕
    void finish() {
        std::unique_lock<std::mutex> lk(mtx);
        done = true;
        cv.notify_all();
    }
};

// ══════════════════════════════════════════════════════════════════════════════
// FrameSource — 模拟 ROS 相机话题发布者
//
// 遍历 NuScenes 帧目录，按仿真帧率逐帧发布到 FrameBuffer。
//
// TODO（接入真实传感器后替换）：
//   class FrameSource {
//     void run(FrameBuffer& buf) {
//       camera::CameraDriver driver;
//       driver.open("calib.json");
//       while (running_) {
//         auto frame = driver.capture();      // 从 SDK 获取 6 路图像
//         buf.publish(frame.frame_dir);       // 传入帧目录或直接存内存
//       }
//     }
//   };
// ══════════════════════════════════════════════════════════════════════════════

class FrameSource {
public:
    FrameSource(const std::vector<std::string>& frame_dirs,
                int fps)
        : dirs_(frame_dirs), fps_(fps)
    {}

    /// 在独立线程中运行：按仿真帧率逐帧写入 FrameBuffer
    void run(FrameBuffer& buf)
    {
        int interval_us = fps_ > 0 ? 1000000 / fps_ : 0;
        printf("[FrameSource] 开始仿真发布，共 %zu 帧，仿真帧率 %d Hz\n",
               dirs_.size(), fps_);
        for (size_t i = 0; i < dirs_.size(); ++i) {
            buf.publish(dirs_[i]);
            if (interval_us > 0)
                usleep(static_cast<useconds_t>(interval_us));
        }
        buf.finish();
        printf("[FrameSource] 所有帧已发布完毕\n");
    }

    size_t size() const { return dirs_.size(); }

private:
    const std::vector<std::string>& dirs_;
    int fps_;
};

// ══════════════════════════════════════════════════════════════════════════════
// 结果 JSON 序列化（与 joint_inference.cpp 格式完全一致）
// ══════════════════════════════════════════════════════════════════════════════

/// 将 BEV 检测框序列化为 result.json
static std::string dets_to_json(const std::vector<tracking::Detection>& dets)
{
    std::ostringstream o;
    o << "[\n";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        o << "  {\"x\":" << d.x
          << ",\"y\":"   << d.y
          << ",\"z\":"   << d.z
          << ",\"yaw\":" << d.yaw
          << ",\"l\":"   << d.l
          << ",\"w\":"   << d.w
          << ",\"h\":"   << d.h
          << ",\"score\":" << d.score
          << ",\"class_id\":" << d.class_id
          << "}";
        if (i + 1 < dets.size()) o << ",";
        o << "\n";
    }
    o << "]\n";
    return o.str();
}

/// 将跟踪结果序列化为 tracks.json（与 tracking_demo / joint_inference 格式一致）
static std::string tracks_to_json(const std::vector<tracking::Track>& tracks)
{
    std::ostringstream o;
    o << "[\n";
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = tracks[i];
        o << "  {\"track_id\": " << t.track_id
          << ", \"position\": [" << t.x << ", " << t.y << ", " << t.z << "]"
          << ", \"size\": ["     << t.w << ", " << t.l << ", " << t.h << "]"
          << ", \"yaw\": "       << t.yaw
          << ", \"velocity\": [" << t.vx << ", " << t.vy << "]"
          << ", \"score\": "     << t.score
          << ", \"class_id\": "  << t.class_id
          << "}";
        if (i + 1 < tracks.size()) o << ",";
        o << "\n";
    }
    o << "]\n";
    return o.str();
}

/// 联合结果序列化为 joint_result.json（兼容 video_demo.py --joint）
struct JointFrame {
    uint64_t frame_id     = 0;
    uint64_t timestamp    = 0;
    double   bev_lat_ms   = 0.0;
    double   map_lat_ms   = 0.0;
    std::vector<tracking::Detection> detections;
    std::vector<tracking::Track>     tracks;
    MapResult map;
};

static std::string joint_to_json(const JointFrame& jf)
{
    std::ostringstream o;
    o << "{\n";
    o << "  \"frame_id\": "      << jf.frame_id    << ",\n";
    o << "  \"timestamp\": "     << jf.timestamp   << ",\n";
    o << "  \"bev_latency_ms\": " << jf.bev_lat_ms << ",\n";
    o << "  \"map_latency_ms\": " << jf.map_lat_ms << ",\n";

    // tracks
    o << "  \"tracks\": [\n";
    for (size_t i = 0; i < jf.tracks.size(); ++i) {
        const auto& t = jf.tracks[i];
        o << "    {\"track_id\": " << t.track_id
          << ", \"position\": [" << t.x << ", " << t.y << ", " << t.z << "]"
          << ", \"size\": ["     << t.w << ", " << t.l << ", " << t.h << "]"
          << ", \"yaw\": "       << t.yaw
          << ", \"velocity\": [" << t.vx << ", " << t.vy << "]"
          << ", \"score\": "     << t.score
          << ", \"class_id\": "  << t.class_id << "}";
        if (i + 1 < jf.tracks.size()) o << ",";
        o << "\n";
    }
    o << "  ],\n";

    // detections
    o << "  \"detections\": [\n";
    for (size_t i = 0; i < jf.detections.size(); ++i) {
        const auto& d = jf.detections[i];
        o << "    {\"x\":"    << d.x
          << ",\"y\":"        << d.y
          << ",\"z\":"        << d.z
          << ",\"yaw\":"      << d.yaw
          << ",\"l\":"        << d.l
          << ",\"w\":"        << d.w
          << ",\"h\":"        << d.h
          << ",\"score\":"    << d.score
          << ",\"class_id\":" << d.class_id << "}";
        if (i + 1 < jf.detections.size()) o << ",";
        o << "\n";
    }
    o << "  ],\n";

    // map
    if (jf.map.is_valid() && !jf.map.raw_json.empty()) {
        o << "  \"map\": " << jf.map.raw_json << "\n";
    } else {
        o << "  \"map\": {\"source\": \"none\", \"elements\": []}\n";
    }
    o << "}\n";
    return o.str();
}

// ══════════════════════════════════════════════════════════════════════════════
// 每帧延迟统计
// ══════════════════════════════════════════════════════════════════════════════

struct FrameLatency {
    double preproc_ms = 0.0;   // 图像预处理（tensor 加载 / 在线计算几何）
    double bev_ms     = 0.0;   // BEV TensorRT 推理（含几何张量更新）
    double track_ms   = 0.0;   // 跟踪器更新
    double map_ms     = 0.0;   // 地图结果读取（本地文件 IO，非 MapTR 推理）
    double write_ms   = 0.0;   // JSON 写入磁盘

    double total_bev() const { return preproc_ms + bev_ms + track_ms; }
};

struct PipelineStats {
    size_t total_frames = 0;
    size_t bev_ok       = 0;
    double sum_preproc  = 0.0;
    double sum_bev      = 0.0;
    double sum_track    = 0.0;
    double sum_write    = 0.0;
    double min_bev      = 1e9;
    double max_bev      = 0.0;
    double wall_ms      = 0.0;    // BEV 线程总墙钟时间
    double maptr_ms     = 0.0;    // MapTR 线程墙钟时间
    int    map_hits     = 0;      // 使用当帧 map 结果的帧数
    int    map_stale    = 0;      // 复用上一帧 map 结果的帧数

    void add(const FrameLatency& lat) {
        ++bev_ok;
        sum_preproc += lat.preproc_ms;
        sum_bev     += lat.bev_ms;
        sum_track   += lat.track_ms;
        sum_write   += lat.write_ms;
        double tb = lat.bev_ms;
        if (tb < min_bev) min_bev = tb;
        if (tb > max_bev) max_bev = tb;
    }

    double avg_bev()     const { return bev_ok > 0 ? sum_bev / bev_ok : 0.0; }
    double avg_preproc() const { return bev_ok > 0 ? sum_preproc / bev_ok : 0.0; }
    double avg_track()   const { return bev_ok > 0 ? sum_track / bev_ok : 0.0; }
    double avg_total()   const { return bev_ok > 0 ? (sum_preproc + sum_bev + sum_track) / bev_ok : 0.0; }
    double fps()         const { return wall_ms > 0 ? bev_ok * 1000.0 / wall_ms : 0.0; }
};

static void print_stats(const PipelineStats& s,
                        const std::string& map_mode,
                        bool skip_map)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║              run_pipeline 推理耗时统计                   ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  处理帧数      : %4zu / %4zu                             ║\n",
           s.bev_ok, s.total_frames);
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  [BEV 流水线各阶段均值（ms/帧）]                         ║\n");
    printf("║   预处理        : %8.2f ms                              ║\n",
           s.avg_preproc());
    printf("║   TensorRT 推理 : %8.2f ms  (min=%.1f  max=%.1f)        ║\n",
           s.avg_bev(), s.min_bev, s.max_bev);
    printf("║   跟踪器更新    : %8.2f ms                              ║\n",
           s.avg_track());
    printf("║   JSON 写入     : %8.2f ms                              ║\n",
           s.avg_total() > 0 ? s.sum_write / s.bev_ok : 0.0);
    printf("║  ─────────────────────────────────────────────────────  ║\n");
    printf("║   BEV 单帧合计  : %8.2f ms                              ║\n",
           s.avg_total());
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  BEV 墙钟总计   : %8.1f ms                              ║\n",
           s.wall_ms);
    printf("║  BEV 实测 FPS   : %8.1f fps                             ║\n",
           s.fps());
    if (!skip_map) {
        printf("╠══════════════════════════════════════════════════════════╣\n");
        printf("║  [MapTR 后台线程（与 BEV 并行）]                         ║\n");
        printf("║   MapTR 墙钟    : %8.1f ms（模式: %-10s）          ║\n",
               s.maptr_ms, map_mode.c_str());
        printf("║   当帧命中      : %4d 帧                                ║\n",
               s.map_hits);
        printf("║   复用上帧      : %4d 帧                                ║\n",
               s.map_stale);
        double parallel_wall = std::max(s.wall_ms, s.maptr_ms);
        double serial_wall   = s.wall_ms + s.maptr_ms;
        double saved         = serial_wall - parallel_wall;
        printf("║   并行节省      : %8.1f ms（串行需 %.0f ms）             ║\n",
               saved, serial_wall);
    }
    printf("╠══════════════════════════════════════════════════════════╣\n");

    // ── 实时性分析 ────────────────────────────────────────────────────────
    double fps_val = s.fps();
    printf("║  [实时性评估]                                            ║\n");
    printf("║   实测 FPS      : %6.1f fps                              ║\n",
           fps_val);
    const double targets[] = {30.0, 20.0, 10.0, 6.0};
    const char* labels[]   = {"30 Hz（自动驾驶）",
                               "20 Hz（高速场景）",
                               "10 Hz（城市场景）",
                               " 6 Hz（低速场景）"};
    for (int i = 0; i < 4; ++i) {
        const char* mark = fps_val >= targets[i] ? "✓" : "✗";
        printf("║   %s  %s %s                        ║\n",
               mark, labels[i],
               fps_val >= targets[i] ? "（满足）   " : "（不满足） ");
    }
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");

    if (fps_val < 6.0) {
        printf("[提示] BEV FPS < 6，建议：\n");
        printf("  1. 使用 int8 量化模型（resnet18int8）\n");
        printf("  2. 减小 --score-thr 以降低后处理量\n");
        printf("  3. 检查 GPU 是否被其他任务占用（nvidia-smi）\n");
    } else if (fps_val < 10.0) {
        printf("[提示] BEV FPS 约 %.1f，满足低速场景实时要求。\n", fps_val);
        printf("  城市/高速场景建议升级到 int8 量化或更高显卡。\n");
    } else {
        printf("[提示] BEV FPS %.1f，满足城市场景实时要求。\n", fps_val);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 用法说明
// ══════════════════════════════════════════════════════════════════════════════

static void print_usage(const char* prog)
{
    printf(
        "用法: %s <frames_dir> <model> [precision] [options]\n"
        "\n"
        "必选参数:\n"
        "  frames_dir      帧目录（frame_XXXXX 子目录）\n"
        "  model           模型名（resnet18 / resnet18int8）\n"
        "\n"
        "可选参数:\n"
        "  precision       fp16 / int8（默认 fp16）\n"
        "  --fps N         仿真帧率（默认 20 Hz，0=不限速）\n"
        "  --score-thr F   置信度阈值（默认 0.35）\n"
        "  --map-mode S    auto|gt|model（默认 auto）\n"
        "  --ckpt PATH     MapTR checkpoint（默认 model/maptr/maptr_nano_r18_110e.pth）\n"
        "  --nuscenes-dir S  NuScenes 数据目录（默认 data/nuscenes）\n"
        "  --skip-map      跳过 MapTR\n"
        "  --save-json     保存 JSON 结果（默认开启）\n"
        "  --no-save-json  关闭 JSON 输出\n"
        "  --no-warmup     跳过 TRT warmup\n"
        "  --verbose       打印每帧详细信息\n"
        "\n"
        "说明:\n"
        "  FrameSource 模拟 ROS 相机话题：按仿真帧率逐帧发布。\n"
        "  BEV 在主线程执行 GPU 推理 + 跟踪；MapTR 在后台线程并行运行。\n"
        "  输出 joint_result.json 可直接用于 video_demo.py --joint。\n"
        "\n"
        "示例:\n"
        "  # 标准推理（BEV + MapTR model 模式）\n"
        "  %s outputs/frames resnet18int8 int8 \\\n"
        "      --map-mode model --ckpt model/maptr/maptr_nano_r18_110e.pth\n"
        "\n"
        "  # 仅 BEV（跳过 MapTR）\n"
        "  %s outputs/frames resnet18int8 int8 --skip-map\n"
        "\n"
        "  # 推理完成后生成视频\n"
        "  python tools/video_demo.py \\\n"
        "      --frames-dir outputs/frames --out-dir outputs/video --joint --fps 6\n",
        prog, prog, prog);
}

// ══════════════════════════════════════════════════════════════════════════════
// main
// ══════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv)
{
    if (argc < 3) { print_usage(argv[0]); return 1; }

    // ── 参数解析 ──────────────────────────────────────────────────────────
    std::string frames_dir   = argv[1];
    std::string model_name   = argv[2];
    std::string precision    = "fp16";

    int         sim_fps      = 20;
    float       score_thr    = 0.35f;
    std::string map_mode     = "auto";
    std::string ckpt_path    = "model/maptr/maptr_nano_r18_110e.pth";
    std::string nuscenes_dir = "data/nuscenes";
    bool        skip_map     = false;
    bool        save_json    = true;
    bool        no_warmup    = false;
    bool        use_prefetch = true;
    bool        verbose      = false;

    for (int i = 3; i < argc; ++i) {
        if (!argv[i]) continue;
        if (str_eq(argv[i], "--fps") && i + 1 < argc) {
            sim_fps = atoi(argv[++i]);
        } else if (str_eq(argv[i], "--score-thr") && i + 1 < argc) {
            score_thr = static_cast<float>(atof(argv[++i]));
        } else if (str_eq(argv[i], "--map-mode") && i + 1 < argc) {
            map_mode = argv[++i];
        } else if (str_eq(argv[i], "--ckpt") && i + 1 < argc) {
            ckpt_path = argv[++i];
        } else if (str_eq(argv[i], "--nuscenes-dir") && i + 1 < argc) {
            nuscenes_dir = argv[++i];
        } else if (str_eq(argv[i], "--skip-map")) {
            skip_map = true;
        } else if (str_eq(argv[i], "--no-save-json")) {
            save_json = false;
        } else if (str_eq(argv[i], "--save-json")) {
            save_json = true;
        } else if (str_eq(argv[i], "--no-warmup")) {
            no_warmup = true;
        } else if (str_eq(argv[i], "--no-prefetch")) {
            use_prefetch = false;
        } else if (str_eq(argv[i], "--offline")) {
            sim_fps = 0;    // fps=0 → 离线全帧处理
        } else if (str_eq(argv[i], "--verbose")) {
            verbose = true;
        } else if (argv[i][0] != '-') {
            precision = argv[i];
        }
    }

    if (!path_exists(frames_dir)) {
        fprintf(stderr, "[run_pipeline] frames_dir 不存在: %s\n",
                frames_dir.c_str());
        return 1;
    }

    // ── 收集帧目录 ────────────────────────────────────────────────────────
    auto frame_dirs = collect_frame_dirs(frames_dir);
    if (frame_dirs.empty()) {
        fprintf(stderr, "[run_pipeline] 未找到 frame_* 子目录: %s\n",
                frames_dir.c_str());
        return 1;
    }

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║         CUDA-FastBEV  run_pipeline（流式管线）            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("  帧目录   : %s（%zu 帧）\n", frames_dir.c_str(), frame_dirs.size());
    printf("  BEV 模型 : %s (%s)\n", model_name.c_str(), precision.c_str());
    printf("  置信度   : %.2f\n", score_thr);
    printf("  仿真帧率 : %s\n",
           sim_fps == 0 ? "离线模式（处理所有帧）" :
           (std::to_string(sim_fps) + " Hz").c_str());
    // --map-mode gpu 等同于持久化 GPU 服务器；skip 等同于 --skip-map
    if (map_mode == "skip") skip_map = true;
    if (skip_map) map_mode = "skip";
    const bool offline_mode = (sim_fps <= 0);

    if (!skip_map)
        printf("  地图模式 : %s  ckpt=%s\n", map_mode.c_str(), ckpt_path.c_str());
    else
        printf("  地图模式 : 跳过 MapTR\n");
    printf("  帧预取   : %s\n", use_prefetch ? "开启" : "关闭");
    printf("  保存 JSON: %s\n\n", save_json ? "是" : "否");

    // ══════════════════════════════════════════════════════════════════════
    // Step 1：初始化 BEV 感知管线（主线程 / GPU）
    //   使用 PerceptionPipeline 但关闭内置跟踪，改为外部手动管理
    //   原因：外部管理可在每帧传入 ego 位姿（ego 运动补偿），
    //         而 PerceptionPipeline::process() 暂不暴露 ego 接口。
    // ══════════════════════════════════════════════════════════════════════
    PipelineConfig pcfg;
    pcfg.model_name       = model_name;
    pcfg.precision        = precision;
    pcfg.score_thr        = score_thr;
    pcfg.enable_tracking  = false;   // 关闭内置跟踪；手动管理以支持 ego 补偿
    pcfg.verbose          = verbose;

    PerceptionPipeline bev_pipeline(pcfg);
    if (!bev_pipeline.init()) {
        fprintf(stderr, "[run_pipeline] BEV 管线初始化失败\n");
        return 1;
    }
    printf("[BEV] 推理核心就绪 (%s/%s)\n\n", model_name.c_str(), precision.c_str());

    // ── 外部跟踪器（含 ego 运动补偿） ────────────────────────────────────
    tracking::TrackerConfig tcfg;
    tcfg.threshold       = -0.4f;
    tcfg.metric          = tracking::MetricType::GIOU_3D;
    tcfg.max_lost_frames = 3;
    tcfg.dt              = (sim_fps > 0) ? 1.0f / sim_fps : 0.05f;
    tcfg.algo            = tracking::AlgoType::GREEDY;
    tcfg.min_hits        = 3;
    tcfg.enable_ego_comp = true;
    tracking::Tracker tracker(tcfg);

    // ── BEV Warmup ────────────────────────────────────────────────────────
    if (!no_warmup && !frame_dirs.empty()) {
        printf("[BEV] Warmup...\n");
        camera::CameraFrame wf;
        RawImageInput       wi;
        if (ImagePreprocessor::from_frame_dir(frame_dirs[0], 0, wf, wi)) {
            bev_pipeline.process(wf);
            camera::FrameLoader::free_frame(wf);
        }
        tracker.reset();
        printf("[BEV] Warmup 完成\n\n");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Step 2：后台启动 MapTR（与 BEV 资源隔离，异步并行）
    //   MapTR 使用 Python subprocess（CPU 为主），BEV 使用 TensorRT（GPU）。
    //   两者通过文件系统（map_result.json）解耦，无直接数据竞争。
    // ══════════════════════════════════════════════════════════════════════
    // ── MapTR：持久化 GPU 服务器 或 批量模式 ────────────────────────────
    std::unique_ptr<MapRunnerPersistent> map_gpu_server;
    std::unique_ptr<MapRunner>           map_runner;
    std::thread      map_thread;
    std::atomic<int> map_result_count{-99};
    double           maptr_wall_ms = 0.0;

    if (!skip_map && map_mode == "gpu") {
        MapRunnerPersistentConfig mcfg;
        mcfg.ckpt_path       = ckpt_path;
        mcfg.score_thr       = score_thr;
        mcfg.device          = "cuda:0";
        mcfg.verbose         = verbose;
        map_gpu_server = std::make_unique<MapRunnerPersistent>(mcfg);
        printf("[MapTR] 启动持久化 GPU 服务器（模型加载约 30-60s）...\n");
        if (!map_gpu_server->init()) {
            fprintf(stderr, "[run_pipeline] MapTR GPU 服务器启动失败，降级为 skip\n");
            map_gpu_server.reset();
            skip_map = true;
            map_mode = "skip";
        } else {
            printf("[MapTR] GPU 服务器已就绪，与 BEV 并行运行\n\n");
        }
    } else if (!skip_map) {
        MapRunnerConfig mcfg;
        mcfg.mode         = map_mode;
        mcfg.frames_dir   = frames_dir;
        mcfg.nuscenes_dir = nuscenes_dir;
        mcfg.ckpt_path    = ckpt_path;
        mcfg.score_thr    = score_thr;
        mcfg.verbose      = verbose;

        map_runner = std::make_unique<MapRunner>(mcfg);

        if (offline_mode) {
            // 离线模式：BEV 启动前同步跑完所有帧，保证 100% map 覆盖率
            printf("[MapTR] 离线模式：先批量推理所有 %zu 帧（BEV 等待）...\n",
                   frame_dirs.size());
            auto t0 = std::chrono::high_resolution_clock::now();
            int cnt = map_runner->run_batch(/*overwrite=*/false);
            auto t1 = std::chrono::high_resolution_clock::now();
            maptr_wall_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            map_result_count.store(cnt);
            printf("[MapTR] 批量推理完成（%d 帧，%.1f ms），启动 BEV\n\n",
                   cnt, maptr_wall_ms);
        } else {
            // 流式模式：后台线程与 BEV 并行
            printf("[MapTR] 后台线程启动（模式: %s），与 BEV 并行...\n",
                   map_mode.c_str());
            printf("[MapTR] BEV 每帧非阻塞读取 map_result.json，无结果时复用上一帧。\n\n");
            map_thread = std::thread(
                [&map_runner, &map_result_count, &maptr_wall_ms]() {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    map_result_count.store(map_runner->run_batch(/*overwrite=*/false));
                    auto t1 = std::chrono::high_resolution_clock::now();
                    maptr_wall_ms =
                        std::chrono::duration<double, std::milli>(t1 - t0).count();
                    printf("\n[MapTR] 后台线程完成，耗时 %.1f ms\n", maptr_wall_ms);
                    fflush(stdout);
                });
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // Step 3：初始化预取加载器 + 选择运行模式
    //   offline_mode (fps==0 / --offline): 直接遍历所有帧，不丢帧
    //   streaming_mode (fps>0): FrameBuffer 最新帧语义，模拟实时订阅
    // ══════════════════════════════════════════════════════════════════════
    // offline_mode 已在参数解析后定义，此处直接使用
    PrefetchLoader prefetch;

    // 离线/流式 共用的主循环体（lambda，避免代码重复）
    // ── 帧变量（在 lambda 外声明，供两种模式复用） ──────────────────────
    PipelineStats stats;
    stats.total_frames = frame_dirs.size();
    MapResult          last_map;
    tracking::EgoPose  last_map_ego;   // ego 位姿（上一次 map_ok 时）
    size_t    processed = 0;
    auto bev_wall_t0 = std::chrono::high_resolution_clock::now();

    // 预取第一帧（两种模式均适用）
    if (use_prefetch && !frame_dirs.empty())
        prefetch.submit(frame_dirs[0], 0);

    // ─── 共用帧处理函数（内联 lambda，避免代码重复） ──────────────────────
    auto process_one_frame = [&](const std::string& cur_dir) {
        camera::CameraFrame bev_frame;
        RawImageInput       map_input;
        FrameLatency        lat;

        // ── 预处理：先消费预取结果，再提交下一帧 ─────────────────────────
        // 正确的双缓冲顺序：consume(当前帧) → submit(下一帧)
        // 错误顺序（原来）：submit(下一帧) → consume() 拿到的是下一帧！
        auto t_prep0 = std::chrono::high_resolution_clock::now();
        bool prep_ok = false;
        if (use_prefetch) {
            // 1. 消费当前帧的预取结果（当前帧应已在后台加载完成或接近完成）
            prep_ok = prefetch.consume(bev_frame, map_input);
            if (!prep_ok) {
                // 预取失败（极少），回退到同步加载
                prep_ok = ImagePreprocessor::from_frame_dir(
                    cur_dir, processed, bev_frame, map_input);
            }
            // 2. 提交下一帧预取（与当前帧 BEV 推理并行执行）
            if (processed + 1 < frame_dirs.size())
                prefetch.submit(frame_dirs[processed + 1], processed + 1);
        } else {
            prep_ok = ImagePreprocessor::from_frame_dir(
                cur_dir, processed, bev_frame, map_input);
        }
        auto t_prep1 = std::chrono::high_resolution_clock::now();
        lat.preproc_ms =
            std::chrono::duration<double, std::milli>(t_prep1 - t_prep0).count();

        if (!prep_ok) {
            fprintf(stderr, "[run_pipeline] 预处理失败，跳过帧: %s\n", cur_dir.c_str());
            return;
        }

        // ── 向 MapTR GPU 服务器提交当前帧（在 BEV 推理前提交，最大化并行）
        if (!skip_map && map_gpu_server && map_gpu_server->is_alive())
            map_gpu_server->submit_frame(cur_dir);

        // ── BEV TensorRT 推理 ────────────────────────────────────────────
        auto t_bev0 = std::chrono::high_resolution_clock::now();
        PerceptionResult pr = bev_pipeline.process(bev_frame);
        auto t_bev1 = std::chrono::high_resolution_clock::now();
        lat.bev_ms = std::chrono::duration<double, std::milli>(t_bev1 - t_bev0).count();

        // ── 多目标跟踪（含 ego 运动补偿） ───────────────────────────────
        tracking::EgoPose ego = read_ego_pose(cur_dir);
        auto t_trk0 = std::chrono::high_resolution_clock::now();
        std::vector<tracking::Track> tracks =
            tracker.update(pr.detections, bev_frame.timestamp, ego);
        auto t_trk1 = std::chrono::high_resolution_clock::now();
        lat.track_ms =
            std::chrono::duration<double, std::milli>(t_trk1 - t_trk0).count();

        // ── 非阻塞读取 MapTR 结果 ────────────────────────────────────────
        JointFrame jf;
        jf.frame_id   = processed;
        jf.timestamp  = bev_frame.timestamp;
        jf.bev_lat_ms = lat.bev_ms;
        jf.detections = pr.detections;
        jf.tracks     = tracks;

        if (!skip_map) {
            auto t_map0 = std::chrono::high_resolution_clock::now();
            MapResult cur_map;
            bool map_ok = false;
            if (map_gpu_server) {
                if (offline_mode) {
                    // 离线模式：GPU 服务器可能还在推理，最多等 2000ms
                    auto t_dl = std::chrono::steady_clock::now()
                                + std::chrono::milliseconds(2000);
                    do {
                        map_ok = map_gpu_server->read_result(cur_dir, cur_map);
                        if (!map_ok) usleep(20000);
                    } while (!map_ok && std::chrono::steady_clock::now() < t_dl);
                } else {
                    map_ok = map_gpu_server->read_result(cur_dir, cur_map);
                }
            } else if (map_runner)
                map_ok = map_runner->read_result(cur_dir, cur_map);
            auto t_map1 = std::chrono::high_resolution_clock::now();
            lat.map_ms =
                std::chrono::duration<double, std::milli>(t_map1 - t_map0).count();
            if (map_ok) {
                last_map = cur_map;  last_map_ego = ego;
                jf.map = cur_map;  ++stats.map_hits;
            } else if (last_map.is_valid()) {
                // ego 运动补偿：将上一帧地图从 last_map_ego 坐标系变换到当前 ego 坐标系
                jf.map = compensate_map_ego(last_map, last_map_ego, ego);
                ++stats.map_stale;
            }
            jf.map_lat_ms = lat.map_ms;
        }

        // ── 写入 JSON 结果 ────────────────────────────────────────────────
        if (save_json) {
            auto t_wr0 = std::chrono::high_resolution_clock::now();
            std::ofstream(cur_dir + "/joint_result.json") << joint_to_json(jf);
            std::ofstream(cur_dir + "/tracks.json")       << tracks_to_json(tracks);
            std::ofstream(cur_dir + "/result.json")       << dets_to_json(pr.detections);
            auto t_wr1 = std::chrono::high_resolution_clock::now();
            lat.write_ms =
                std::chrono::duration<double, std::milli>(t_wr1 - t_wr0).count();
        }

        stats.add(lat);
        ++processed;

        if (verbose) {
            printf("[frame %04zu] prep=%.1fms  bev=%.1fms  trk=%.1fms"
                   "  det=%zu  trk=%zu\n",
                   processed - 1, lat.preproc_ms, lat.bev_ms, lat.track_ms,
                   pr.detections.size(), tracks.size());
        } else if (processed % 10 == 0 || processed == frame_dirs.size()) {
            printf("\r[Progress] %zu / %zu 帧完成  prep=%.1fms  BEV=%.1fms  trk=%zu",
                   processed, frame_dirs.size(), lat.preproc_ms, lat.bev_ms,
                   tracks.size());
            fflush(stdout);
        }

        camera::FrameLoader::free_frame(bev_frame);
    }; // end process_one_frame lambda

    // ══════════════════════════════════════════════════════════════════════
    // Step 4：主循环（离线模式 OR 流式模式）
    // ══════════════════════════════════════════════════════════════════════
    std::thread src_thread;  // 流式模式才用

    if (offline_mode) {
        // ─── 离线模式：直接遍历所有帧，保证每帧都被处理 ─────────────────
        printf("[run_pipeline] 离线模式：顺序处理所有 %zu 帧\n",
               frame_dirs.size());
        for (size_t fi = 0; fi < frame_dirs.size(); ++fi)
            process_one_frame(frame_dirs[fi]);
    } else {
        // ─── 流式模式：订阅 FrameBuffer 最新帧（模拟实时） ──────────────
        // 若 BEV 慢于 FrameSource，自动丢弃中间帧，贴近实时语义
        FrameBuffer frame_buf;
        FrameSource frame_src(frame_dirs, sim_fps);
        src_thread = std::thread([&frame_src, &frame_buf]() {
            frame_src.run(frame_buf);
        });

        while (true) {
            std::string cur_dir;
            uint64_t    cur_seq;
            if (!frame_buf.wait_pop(cur_dir, cur_seq, 500000)) {
                if (frame_buf.done) break;
                continue;
            }
            process_one_frame(cur_dir);
        }
    }

    auto bev_wall_t1 = std::chrono::high_resolution_clock::now();
    stats.wall_ms =
        std::chrono::duration<double, std::milli>(bev_wall_t1 - bev_wall_t0).count();
    printf("\n");

    // ── 等待帧源线程 ──────────────────────────────────────────────────────
    if (src_thread.joinable()) src_thread.join();

    // ── 关闭 MapTR GPU 服务器 ─────────────────────────────────────────────
    if (map_gpu_server)
        map_gpu_server->shutdown();

    // ══════════════════════════════════════════════════════════════════════
    // Step 5：等待 MapTR 后台线程完成
    // ══════════════════════════════════════════════════════════════════════
    if (map_thread.joinable()) {
        if (map_result_count.load() == -99) {
            printf("[BEV 已完成，等待 MapTR 后台线程...]\n");
            fflush(stdout);
        }
        map_thread.join();
    }

    stats.maptr_ms = maptr_wall_ms;

    // ══════════════════════════════════════════════════════════════════════
    // Step 6：打印耗时统计与实时性分析
    // ══════════════════════════════════════════════════════════════════════
    print_stats(stats, map_mode, skip_map);

    if (save_json) {
        printf("[输出] JSON 已写入各帧目录：\n");
        printf("  joint_result.json — 联合结果（BEV + 地图，video_demo.py --joint）\n");
        printf("  tracks.json       — 跟踪轨迹（video_visualize_tracking.py）\n");
        printf("  result.json       — BEV 检测框（video_demo.py 默认模式）\n");
        printf("\n[下一步] 生成可视化视频：\n");
        printf("  python tools/video_demo.py \\\n");
        printf("      --frames-dir %s --out-dir outputs/video \\\n",
               frames_dir.c_str());
        printf("      --joint --fps 6 --bev-size 800 --cam-width 480\n");
    }

    return 0;
}
