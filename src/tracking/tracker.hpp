#pragma once
/**
 * tracker.hpp — 多目标跟踪器接口
 *
 * 采用简化的 Sort-like 算法：
 *   1. 预测所有现有轨迹
 *   2. 计算检测框与轨迹的代价矩阵（BEV 中心距离）
 *   3. 匈牙利匹配
 *   4. 更新匹配轨迹；为未匹配检测创建新轨迹；标记未匹配轨迹为 Lost/Removed
 *   5. 输出确认轨迹列表
 */

#include "track.hpp"
#include <memory>
#include <functional>

namespace fastbev {
namespace tracking {

// ─── 跟踪器参数 ────────────────────────────────────────────────────────────

struct TrackerConfig {
    float  max_dist          = 3.0f;  // 匹配距离阈值（米）
    int    max_lost_frames   = 3;     // 允许丢失的最大帧数，超过则移除
    int    min_hits_confirm  = 2;     // 轨迹确认所需最小匹配次数
    double dt                = 0.05;  // 帧间时间间隔（秒）
};

// ─── 多目标跟踪器 ──────────────────────────────────────────────────────────

class Tracker {
public:
    explicit Tracker(const TrackerConfig& cfg = TrackerConfig{});
    ~Tracker() = default;

    /**
     * 更新跟踪器（每帧调用一次）
     * @param detections  当帧 BEV 检测结果列表
     * @param timestamp   当前帧时间戳（秒）
     * @return            所有确认轨迹（state == Active 或 Lost）
     */
    std::vector<Track> update(const std::vector<Detection>& detections,
                              double timestamp = 0.0);

    /**
     * 获取当前所有有效轨迹（不更新状态）
     */
    const std::vector<Track>& tracks() const { return tracks_; }

    /**
     * 重置跟踪器（清空所有轨迹）
     */
    void reset();

    int  next_id() const { return static_cast<int>(next_id_); }

private:
    TrackerConfig cfg_;
    uint64_t      next_id_ = 1;
    std::vector<Track> tracks_;

    // 匈牙利算法（最小代价匹配）
    // cost[i][j] = 轨迹 i 与检测 j 的距离，超出阈值视为不可匹配
    static std::vector<std::pair<int,int>> hungarian_match(
        const std::vector<std::vector<float>>& cost,
        float max_cost);
};

}  // namespace tracking
}  // namespace fastbev
