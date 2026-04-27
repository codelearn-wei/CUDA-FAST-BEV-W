/**
 * tracker.cpp — 多目标跟踪器实现（对齐 AB3DMOT 框架）
 */

#include "tracker.hpp"
#include <algorithm>
#include <limits>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace fastbev {
namespace tracking {

// ─── 辅助转换：距离度量阈值处理（Python 中对距离取负）─────────────
static float normalizeThreshold(float thres, MetricType metric) {
    if (metric == MetricType::DIST_3D) {
        return std::abs(thres);     // 欧氏距离越小越好 → 代价取负
    } else if (metric == MetricType::MAHALANOBIS) {
        return std::abs(thres);       // 马氏距离越小越好，阈值直接为正
    } else { // GIOU_3D, IOU_3D
        return thres;                 // IoU/GIoU 阈值原样
    }
}

// ─── 构建代价矩阵（支持 DIST_3D, GIOU_3D，其余预留）─────────────────
std::vector<std::vector<float>> Tracker::buildCostMatrix(
    const std::vector<Detection>& dets,
    const std::vector<Track>& trks) const
{
    const int n_trk = static_cast<int>(trks.size());
    const int n_det = static_cast<int>(dets.size());
    std::vector<std::vector<float>> cost(n_trk, std::vector<float>(n_det, 1e9f));

    for (int i = 0; i < n_trk; ++i) {
        for (int j = 0; j < n_det; ++j) {
            // 类别过滤（可配置是否严格相同，此处保持严格匹配）
            if (trks[i].class_id != dets[j].class_id) continue;

            float val = 0.0f;
            switch (cfg_.metric) {
                case MetricType::DIST_3D:
                    // 3D 中心欧氏距离（需要在 Track 类中实现 dist3d）
                    val = trks[i].dist3d(dets[j]);
                    break;

                case MetricType::GIOU_3D: {
                    // 计算 GIoU 3D，代价 = 1 - GIoU
                    float giou = computeGiou3D(trks[i], dets[j]);
                    val = 1.0f - giou;   // 范围 [0,2]，越小表示匹配越好
                    break;
                }

                // 以下为预留分支，尚未实现，返回极大值表示不可匹配
                case MetricType::IOU_3D: {
                    // TODO: 实现 computeIou3D
                    // float iou = computeIou3D(trks[i], dets[j]);
                    // val = 1.0f - iou;
                    val = 1e9f;
                    break;
                }
                case MetricType::MAHALANOBIS: {
                    // 仅当轨迹在上一帧被更新过（协方差有效）时才计算马氏距离
                    if (trks[i].time_since_update == 0) {
                        val = trks[i].mahalanobisToDetection(dets[j]);
                    } else {
                        val = 1e9f;   // 无效匹配
                    }
                    break;
                }
                default:
                    val = 1e9f;
                    break;
            }
            cost[i][j] = val;
        }
    }
    return cost;
}

void Tracker::computeBEVIntersectionAndConvexArea(
    float cx1, float cy1, float l1, float w1, float yaw1,
    float cx2, float cy2, float l2, float w2, float yaw2,
    float& I_2D, float& C_2D) const
{
    // 将弧度转为度数（OpenCV 使用度数）
    double angle1 = yaw1 * 180.0 / M_PI;
    double angle2 = yaw2 * 180.0 / M_PI;

    cv::RotatedRect rect1(cv::Point2f(cx1, cy1), cv::Size2f(l1, w1), angle1);
    cv::RotatedRect rect2(cv::Point2f(cx2, cy2), cv::Size2f(l2, w2), angle2);

    cv::Point2f vertices1[4], vertices2[4];
    rect1.points(vertices1);
    rect2.points(vertices2);

    std::vector<cv::Point2f> poly1(vertices1, vertices1 + 4);
    std::vector<cv::Point2f> poly2(vertices2, vertices2 + 4);

    std::vector<cv::Point2f> intersection;
    if (cv::intersectConvexConvex(poly1, poly2, intersection)) {
        I_2D = cv::contourArea(intersection);
    } else {
        I_2D = 0.0f;
    }

    std::vector<cv::Point2f> all_points;
    all_points.insert(all_points.end(), poly1.begin(), poly1.end());
    all_points.insert(all_points.end(), poly2.begin(), poly2.end());
    std::vector<cv::Point2f> convex_hull;
    cv::convexHull(all_points, convex_hull);
    C_2D = cv::contourArea(convex_hull);
}

float Tracker::computeGiou3D(const Track& trk, const Detection& det) const
{
    // 获取参数（根据你的数据结构，确保成员名正确）
    float cx1 = trk.x, cy1 = trk.y, cz1 = trk.z;
    float l1 = trk.l, w1 = trk.w, h1 = trk.h;
    float yaw1 = trk.yaw;

    float cx2 = det.x, cy2 = det.y, cz2 = det.z;
    float l2 = det.l, w2 = det.w, h2 = det.h;
    float yaw2 = det.yaw;

    // BEV 投影
    float I_2D = 0.0f, C_2D = 0.0f;
    computeBEVIntersectionAndConvexArea(cx1, cy1, l1, w1, yaw1,
                                        cx2, cy2, l2, w2, yaw2,
                                        I_2D, C_2D);

    // 高度方向
    float zmin1 = cz1 - h1 / 2.0f;
    float zmax1 = cz1 + h1 / 2.0f;
    float zmin2 = cz2 - h2 / 2.0f;
    float zmax2 = cz2 + h2 / 2.0f;

    float overlap_h = std::max(0.0f, std::min(zmax1, zmax2) - std::max(zmin1, zmin2));
    float union_h   = std::max(zmax1, zmax2) - std::min(zmin1, zmin2);

    float I_3D = I_2D * overlap_h;
    float U_3D = (l1 * w1 * h1) + (l2 * w2 * h2) - I_3D;
    float C_3D = C_2D * union_h;

    if (U_3D < 1e-6f) return -1.0f;
    if (C_3D < 1e-6f) return -1.0f;

    float iou_3d = I_3D / U_3D;
    float giou_3d = iou_3d - (C_3D - U_3D) / C_3D;
    return giou_3d;
}

// ─── 贪心匹配（保留原有代码）──────────────────────────────────────
std::vector<std::pair<int,int>> Tracker::greedyMatch(
    const std::vector<std::vector<float>>& cost,
    float max_cost) const
{
    int n_rows = static_cast<int>(cost.size());
    if (n_rows == 0) return {};
    int n_cols = static_cast<int>(cost[0].size());

    struct Item { float c; int r, c_; };
    std::vector<Item> items;
    for (int i = 0; i < n_rows; ++i)
        for (int j = 0; j < n_cols; ++j)
            if (cost[i][j] <= max_cost)
                items.push_back({cost[i][j], i, j});

    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b) { return a.c < b.c; });

    std::vector<bool> used_row(n_rows, false);
    std::vector<bool> used_col(n_cols, false);
    std::vector<std::pair<int,int>> matches;
    for (const auto& it : items) {
        if (!used_row[it.r] && !used_col[it.c_]) {
            matches.push_back({it.r, it.c_});
            used_row[it.r] = true;
            used_col[it.c_] = true;
        }
    }
    return matches;
}

// ─── 匈牙利匹配（仅声明，后续实现）─────────────────────────────────
std::vector<std::pair<int,int>> Tracker::hungarianMatch(
    const std::vector<std::vector<float>>& /*cost*/,
    float /*max_cost*/) const
{
    // TODO: 实现标准匈牙利算法（可调用外部库，如 dlib/hungarian）
    return {};
}

// ─── 数据关联入口 ────────────────────────────────────────────────
Tracker::MatchResult Tracker::dataAssociation(
    const std::vector<Detection>& dets,
    const std::vector<Track>& trks,
    const std::vector<std::vector<float>>& cost) const
{
    float thr = normalizeThreshold(cfg_.threshold, cfg_.metric);

    std::vector<std::pair<int,int>> matches;
    if (cfg_.algo == AlgoType::HUNGARIAN) {
        matches = hungarianMatch(cost, thr);
    } else { // GREEDY
        matches = greedyMatch(cost, thr);
    }

    // 标记已匹配
    std::vector<bool> trk_matched(trks.size(), false);
    std::vector<bool> det_matched(dets.size(), false);
    for (const auto& m : matches) {
        trk_matched[m.first] = true;
        det_matched[m.second] = true;
    }

    // 收集未匹配
    std::vector<int> unmatched_trks, unmatched_dets;
    for (size_t i = 0; i < trks.size(); ++i)
        if (!trk_matched[i]) unmatched_trks.push_back(i);
    for (size_t j = 0; j < dets.size(); ++j)
        if (!det_matched[j]) unmatched_dets.push_back(j);

    // 收集匹配的代价
    std::vector<float> match_costs;
    for (const auto& m : matches)
        match_costs.push_back(cost[m.first][m.second]);

    // affinity 可以返回原始 cost 矩阵（可选）
    return {matches, unmatched_dets, unmatched_trks, match_costs, cost};
}

// ─── 自车运动补偿（空实现，仅框架）─────────────────────────────────
void Tracker::egoMotionCompensation(std::vector<Track>& /*trks*/,
                                    double /*timestamp*/) const
{
    // TODO: 根据 cfg_.enable_ego_comp 实现
    // 需要外部提供 pose 查询接口
}

// ─── 更新已匹配的轨迹 ────────────────────────────────────────────
void Tracker::updateMatchedTracks(
    const std::vector<std::pair<int,int>>& matches,
    const std::vector<Detection>& dets,
    double timestamp)
{
    for (const auto& m : matches) {
        int trk_idx = m.first;
        int det_idx = m.second;
        Track& trk = tracks_[trk_idx];
        // 更新统计
        trk.time_since_update = 0;
        trk.hits++;
        // 调用 Track 的更新方法（目前是简单低通，后续可替换为卡尔曼）
        trk.update(dets[det_idx], timestamp);
        // 可选：保存 info（如置信度）
        // trk.info = info[det_idx];
    }
}

// ─── 创建新轨迹（birth）─────────────────────────────────────────
void Tracker::createNewTracks(
    const std::vector<int>& unmatched_det_indices,
    const std::vector<Detection>& dets,
    double timestamp)
{
    for (int idx : unmatched_det_indices) {
        tracks_.emplace_back(next_id_++, dets[idx], timestamp);
    }
}

// ─── 输出确认的轨迹（过滤 min_hits / max_age）─────────────────────
std::vector<Track> Tracker::outputConfirmedTracks()
{
    std::vector<Track> keep;
    keep.reserve(tracks_.size());
    for (auto& trk : tracks_) {
        if (trk.state == TrackState::Removed) continue;
        // 移除超时轨迹
        if (trk.time_since_update > cfg_.max_lost_frames) {
            trk.state = TrackState::Removed;
            continue;
        }
        // 保留未超时轨迹（包括未确认的）
        keep.push_back(trk);
    }
    tracks_ = std::move(keep);

    // 返回确认的轨迹
    std::vector<Track> confirmed;
    for (const auto& trk : tracks_) {
        if (trk.is_confirmed())   // is_confirmed() 内部检查 hits >= min_hits? 实际上需结合 cfg_.min_hits
            confirmed.push_back(trk);
    }
    return confirmed;
}

// ─── 构造函数 ──────────────────────────────────────────────────
Tracker::Tracker(const TrackerConfig& cfg) : cfg_(cfg) {}

void Tracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}

// ─── 核心更新函数（完全对齐 AB3DMOT 流程）─────────────────────────
std::vector<Track> Tracker::update(const std::vector<Detection>& detections,
                                   double timestamp)
{
    // 1. 预测所有现有轨迹（匀速模型或卡尔曼）
    for (auto& trk : tracks_) {
        trk.predict(cfg_.dt);
    }

    // 2. 自车运动补偿（可选）
    if (cfg_.enable_ego_comp) {
        // 注意：补偿应该在预测之后，匹配之前，将轨迹变换到当前坐标系
        // 由于 tracks_ 是当前成员，直接传递引用
        egoMotionCompensation(tracks_, timestamp);
    }

    // 3. 构建代价矩阵
    auto cost = buildCostMatrix(detections, tracks_);

    // 4. 数据关联（匈牙利/贪心）
    auto matchRes = dataAssociation(detections, tracks_, cost);

    // 5. 更新匹配的轨迹
    updateMatchedTracks(matchRes.matches, detections, timestamp);

    // 6. 创建新轨迹（birth）
    createNewTracks(matchRes.unmatched_dets, detections, timestamp);

    // 7. 输出确认轨迹（根据 min_hits 和 max_age）
    return outputConfirmedTracks();
}

}  // namespace tracking
}  // namespace fastbev