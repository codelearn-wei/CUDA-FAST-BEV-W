#pragma once
/**
 * tracker.hpp — 多目标跟踪器接口（对齐 AB3DMOT 框架）
 */

#include "track.hpp"
#include <memory>
#include <functional>
#include <vector>

namespace fastbev {
namespace tracking {

// ─── 匹配度量类型 ────────────────────────────────────────────
enum class MetricType {
    DIST_3D,   // 欧氏距离（中心点）
    GIOU_3D,   // 3D 广义交并比（后续实现）
    IOU_3D,    // 3D 交并比
    MAHALANOBIS // 马氏距离（需协方差）
};

// ─── 匹配算法 ────────────────────────────────────────────────
enum class AlgoType {
    HUNGARIAN,   // 匈牙利算法（全局最优）
    GREEDY       // 贪心近似（当前实现）
};

// ─── 跟踪器配置（扩展）────────────────────────────────────────
struct TrackerConfig {
    // 基础参数
    float max_dist          = 3.0f;   // 距离/代价阈值（注意：距离度量时阈值为负）
    int   min_hits          = 3;      // 轨迹确认所需的最小匹配次数
    int   max_lost_frames   = 2;      // 允许丢失的最大帧数（超过则移除）
    double dt               = 0.5;   // 帧间时间间隔（秒）

    // 匹配相关
    MetricType metric       = MetricType::DIST_3D;
    AlgoType   algo         = AlgoType::GREEDY;
    float      threshold    = 0.5f;   // 统一阈值（距离取负，IoU/GIoU 直接使用）

    // 功能开关
    bool enable_ego_comp    = true;   // 启用自车运动补偿（修复航向角跳变）
    bool enable_prediction = true;
};

// ─── 自车全局位姿（用于 ego 运动补偿）──────────────────────────
struct EgoPose {
    double x    = 0.0;    // 全局 x（米）
    double y    = 0.0;    // 全局 y（米）
    double yaw  = 0.0;    // 全局航向角（弧度）
    double vx   = 0.0;    // 自车全局速度 x 分量（米/秒）
    double vy   = 0.0;    // 自车全局速度 y 分量（米/秒）
    bool   valid = false; // 是否有效
};

// ─── 多目标跟踪器（接口对齐 AB3DMOT）────────────────────────────
class Tracker {
public:
    explicit Tracker(const TrackerConfig& cfg = TrackerConfig{});
    ~Tracker() = default;

    /**
     * 每帧更新（核心方法）
     * @param detections    当前帧检测结果（lidar-local 坐标系）
     * @param timestamp     时间戳（秒），用于预测和补偿
     * @param current_ego   当前帧自车全局位姿（用于 ego 运动补偿）
     * @return              确认的轨迹列表（is_confirmed() == true）
     */
    std::vector<Track> update(const std::vector<Detection>& detections,
                              double timestamp,
                              const EgoPose& current_ego = EgoPose{});

    /**
     * 重置跟踪器（清空所有轨迹）
     */
    void reset();

    const std::vector<Track>& tracks() const { return tracks_; }
    int next_id() const { return static_cast<int>(next_id_); }

private:
    // ── 数据关联子步骤 ──────────────────────────────────────
    struct MatchResult {
        std::vector<std::pair<int,int>> matches;         // (det_idx, trk_idx)
        std::vector<int> unmatched_dets;
        std::vector<int> unmatched_trks;
        std::vector<float> costs;                        // 匹配成功的代价
        std::vector<std::vector<float>> affinity;        // 原始代价/相似度矩阵
    };

    // 构建代价矩阵（支持不同 metric）
    std::vector<std::vector<float>> buildCostMatrix(
        const std::vector<Detection>& dets,
        const std::vector<Track>& trks) const;
    
    // GIoU 3D 计算
    float computeGiou3D(const Track& trk, const Detection& det) const;

    // 辅助函数：计算两个 BEV 旋转矩形的 I_2D 和 C_2D（面积）
    void computeBEVIntersectionAndConvexArea(float cx1, float cy1, float l1, float w1, float yaw1,
                                             float cx2, float cy2, float l2, float w2, float yaw2,
                                             float& I_2D, float& C_2D) const;

    // 执行匹配（匈牙利或贪心）
    MatchResult dataAssociation(
        const std::vector<Detection>& dets,
        const std::vector<Track>& trks,
        const std::vector<std::vector<float>>& cost) const;

    // 贪心匹配（当前 hungarian_match 实现）
    std::vector<std::pair<int,int>> greedyMatch(
        const std::vector<std::vector<float>>& cost) const;

    std::vector<std::pair<int,int>> hungarianMatch(
        const std::vector<std::vector<float>>& cost) const;

    // 自车运动补偿：将所有轨迹预测状态从上一帧local坐标转到当前帧local坐标
    void egoMotionCompensation(
        std::vector<Track>& trks,
        const EgoPose& prev_ego,
        const EgoPose& curr_ego) const;

    // 更新匹配的轨迹（含卡尔曼更新）
    void updateMatchedTracks(
        const std::vector<std::pair<int,int>>& matches,
        const std::vector<Detection>& dets,
        double timestamp);

    // 创建新轨迹（birth）
    void createNewTracks(
        const std::vector<int>& unmatched_det_indices,
        const std::vector<Detection>& dets,
        double timestamp);

    // 删除过期轨迹并返回确认轨迹
    std::vector<Track> outputConfirmedTracks();

private:
    TrackerConfig cfg_;
    uint64_t next_id_ = 1;
    std::vector<Track> tracks_;
    EgoPose prev_ego_pose_;   // 上一帧自车全局位姿（用于运动补偿）
    EgoPose prev_ego_pose_for_velocity_;     // 用于速度差分
    double  prev_timestamp_ = -1.0;          // 上一帧时间戳
};

}  // namespace tracking
}  // namespace fastbev