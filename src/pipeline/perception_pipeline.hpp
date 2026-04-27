#pragma once
/**
 * perception_pipeline.hpp — 完整感知管线
 *
 * 链路：多相机帧 → BEV 检测 → 多目标跟踪 → 轨迹输出
 *
 * 设计原则：
 *   - 不直接依赖 ROS / 具体输入源
 *   - 每帧调用 process(frame) 即可获得 PerceptionResult
 *   - 可选地注册回调（output_callback），用于与 ROS 节点等集成
 */

#include "../fastbev/fastbev.hpp"
#include "../tracking/tracker.hpp"
#include "../camera/camera_frame.hpp"
#include <memory>
#include <functional>
#include <vector>
#include <string>

namespace fastbev {
namespace pipeline {

// ─── 管线配置 ──────────────────────────────────────────────────────────────

struct PipelineConfig {
    // 推理相关
    std::string model_name = "resnet18";
    std::string precision  = "fp16";

    // 过滤
    float  score_thr    = 0.4f;
    std::vector<int> class_filter;  // 空=全部
    float  nms_bev_dist = 0.8f;     // BEV 中心距离 NMS（米），0=不启用

    // 跟踪
    bool   enable_tracking   = true;
    float  track_max_dist    = 3.0f;   // 匹配距离阈值（米）
    int    track_max_lost    = 3;      // 最大丢失帧数
    int    track_min_hits    = 2;      // 确认所需最小匹配次数
    double track_dt          = 0.05;   // 帧间隔（秒）

    // 输出
    std::string output_dir   = "outputs/results";
    std::string output_format= "json";  // "txt" 或 "json"
    bool   save_output       = false;
    bool   verbose           = false;
};

// ─── 单帧感知结果 ──────────────────────────────────────────────────────────

struct PerceptionResult {
    uint64_t frame_id  = 0;
    double   timestamp = 0.0;
    double   latency_ms= 0.0;       // 推理耗时

    // 原始检测结果
    std::vector<tracking::Detection> detections;

    // 跟踪轨迹（若 enable_tracking=true）
    std::vector<tracking::Track>     tracks;

    bool is_valid() const { return frame_id > 0 || timestamp > 0; }
};

// ─── 完整感知管线 ──────────────────────────────────────────────────────────

class PerceptionPipeline {
public:
    using ResultCallback = std::function<void(const PerceptionResult&)>;

    explicit PerceptionPipeline(const PipelineConfig& cfg = PipelineConfig{});
    ~PerceptionPipeline() = default;

    /**
     * 初始化推理核心（加载 TRT engine）
     * @return true 表示成功
     */
    bool init();

    /**
     * 处理一帧（同步）
     * @param frame     多相机帧（含图像和几何张量）
     * @param stream    CUDA stream
     * @return          本帧感知结果
     */
    PerceptionResult process(const camera::CameraFrame& frame,
                              void* stream = nullptr);

    /**
     * 注册结果回调（每帧处理完后触发）
     */
    void set_callback(ResultCallback cb) { callback_ = std::move(cb); }

    /**
     * 重置跟踪器（切换场景时调用）
     */
    void reset_tracker();

    /**
     * 获取当前配置
     */
    const PipelineConfig& config() const { return cfg_; }

    bool is_initialized() const { return core_ != nullptr; }

private:
    PipelineConfig cfg_;
    std::shared_ptr<Core> core_;
    std::unique_ptr<tracking::Tracker> tracker_;
    ResultCallback callback_;
    uint64_t frame_counter_ = 0;

    // 过滤并转换 BoundingBox → Detection
    std::vector<tracking::Detection> filter_and_convert(
        const std::vector<post::transbbox::BoundingBox>& boxes);
};

}  // namespace pipeline
}  // namespace fastbev
