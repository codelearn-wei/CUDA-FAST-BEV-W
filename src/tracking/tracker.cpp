/**
 * tracker.cpp — 多目标跟踪器实现（调试版，增加速度诊断）
 */

#include "tracker.hpp"
#include <algorithm>
#include <limits>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace fastbev {
namespace tracking {

// ─── 辅助转换：距离度量阈值处理 ─────────────────────────────────
static float normalizeThreshold(float thres, MetricType metric) {
    if (metric == MetricType::DIST_3D) {
        return thres;
    } else if (metric == MetricType::MAHALANOBIS) {
        return std::abs(thres);
    } else if (metric == MetricType::GIOU_3D) {
        return 1.0f - thres;
    } else {
        return thres;
    }
}

// 局部 → 全局（加入自车速度补偿）
void localToWorld(float x_local, float y_local, float yaw_local,
                  float vx_local, float vy_local,
                  const EgoPose& ego,
                  float& x_world, float& y_world, float& yaw_world,
                  float& vx_world, float& vy_world) {
    float cos_ego = std::cos(ego.yaw);
    float sin_ego = std::sin(ego.yaw);
    x_world = ego.x + (-sin_ego) * x_local + cos_ego * y_local;
    y_world = ego.y +  cos_ego * x_local + sin_ego * y_local;
    yaw_world = KalmanFilter::normalizeAngle(ego.yaw + static_cast<float>(M_PI) - yaw_local);
    // 相对速度旋转到全局
    float vx_rel_rot = (-sin_ego) * vx_local + cos_ego * vy_local;
    float vy_rel_rot =  cos_ego * vx_local + sin_ego * vy_local;
    // 加上自车速度得到绝对速度
    vx_world = vx_rel_rot + static_cast<float>(ego.vx);
    vy_world = vy_rel_rot + static_cast<float>(ego.vy);
}

// 全局 → 局部（减去自车速度）
void worldToLocal(float x_world, float y_world, float yaw_world,
                  float vx_world, float vy_world,
                  const EgoPose& ego,
                  float& x_local, float& y_local, float& yaw_local,
                  float& vx_local, float& vy_local) {
    float dx = x_world - ego.x;
    float dy = y_world - ego.y;
    float cos_ego = std::cos(ego.yaw);
    float sin_ego = std::sin(ego.yaw);
    x_local = (-sin_ego) * dx + cos_ego * dy;
    y_local =  cos_ego * dx + sin_ego * dy;
    yaw_local = KalmanFilter::normalizeAngle(ego.yaw + static_cast<float>(M_PI) - yaw_world);
    // 全局绝对速度减去自车速度得到相对速度
    float vx_abs_rel = vx_world - static_cast<float>(ego.vx);
    float vy_abs_rel = vy_world - static_cast<float>(ego.vy);
    // 旋转回局部
    vx_local = (-sin_ego) * vx_abs_rel + cos_ego * vy_abs_rel;
    vy_local =  cos_ego * vx_abs_rel + sin_ego * vy_abs_rel;
}

// ─── 构建代价矩阵（不变）────────────────────────────────────────
std::vector<std::vector<float>> Tracker::buildCostMatrix(
    const std::vector<Detection>& dets,
    const std::vector<Track>& trks) const
{
    const int n_trk = static_cast<int>(trks.size());
    const int n_det = static_cast<int>(dets.size());
    std::vector<std::vector<float>> cost(n_trk, std::vector<float>(n_det, 1e9f));
    const float max_angle_diff = static_cast<float>(M_PI) / 2.0f;
    for (int i = 0; i < n_trk; ++i) {
        for (int j = 0; j < n_det; ++j) {
            if (trks[i].class_id != dets[j].class_id) continue;
            float yaw_trk = trks[i].yaw;
            float yaw_det = dets[j].yaw;
            float diff = std::abs(yaw_trk - yaw_det);
            diff = std::fmod(diff, 2.0f * static_cast<float>(M_PI));
            if (diff > static_cast<float>(M_PI)) diff = 2.0f * static_cast<float>(M_PI) - diff;
            // 可选：放宽角度限制到 180°（暂时注释掉或保留）
            // const float max_angle_diff = static_cast<float>(M_PI);  // 180度
            if (diff > max_angle_diff) continue;  // max_angle_diff 仍使用原值（M_PI/2 或 M_PI）
            float val = 0.0f;
            switch (cfg_.metric) {
                case MetricType::DIST_3D:
                    val = trks[i].dist3d(dets[j]);
                    break;
                case MetricType::GIOU_3D: {
                    float giou = computeGiou3D(trks[i], dets[j]);
                    val = 1.0f - giou;
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

// ─── BEV 辅助计算（不变）────────────────────────────────────────
void Tracker::computeBEVIntersectionAndConvexArea(
    float cx1, float cy1, float l1, float w1, float yaw1,
    float cx2, float cy2, float l2, float w2, float yaw2,
    float& I_2D, float& C_2D) const
{
    double angle1 = -yaw1 * 180.0 / M_PI;
    double angle2 = -yaw2 * 180.0 / M_PI;
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

// ─── GIoU 3D（不变）─────────────────────────────────────────────
float Tracker::computeGiou3D(const Track& trk, const Detection& det) const
{
    float cx1 = trk.x, cy1 = trk.y, cz1 = trk.z;
    float l1 = trk.l, w1 = trk.w, h1 = trk.h;
    float yaw1 = trk.yaw;
    float cx2 = det.x, cy2 = det.y, cz2 = det.z;
    float l2 = det.l, w2 = det.w, h2 = det.h;
    float yaw2 = det.yaw;
    float I_2D = 0.0f, C_2D = 0.0f;
    computeBEVIntersectionAndConvexArea(cx1, cy1, l1, w1, yaw1,
                                        cx2, cy2, l2, w2, yaw2,
                                        I_2D, C_2D);
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

// ─── 贪心匹配（不变）────────────────────────────────────────────
std::vector<std::pair<int,int>> Tracker::greedyMatch(
    const std::vector<std::vector<float>>& cost) const
{
    int n_rows = static_cast<int>(cost.size());
    if (n_rows == 0) return {};
    int n_cols = static_cast<int>(cost[0].size());
    struct Item { float c; int r; int c_; };
    std::vector<Item> items;
    items.reserve(n_rows * n_cols);
    for (int i = 0; i < n_rows; ++i)
        for (int j = 0; j < n_cols; ++j)
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

// ─── 匈牙利算法（不变）──────────────────────────────────────────
std::vector<std::pair<int,int>> Tracker::hungarianMatch(
    const std::vector<std::vector<float>>& cost) const
{
    int n_rows = static_cast<int>(cost.size());
    if (n_rows == 0) return {};
    int n_cols = static_cast<int>(cost[0].size());
    bool transposed = false;
    std::vector<std::vector<double>> cost_double;
    if (n_rows > n_cols) {
        transposed = true;
        cost_double.resize(n_cols, std::vector<double>(n_rows));
        for (int i = 0; i < n_rows; ++i)
            for (int j = 0; j < n_cols; ++j)
                cost_double[j][i] = static_cast<double>(cost[i][j]);
        std::swap(n_rows, n_cols);
    } else {
        cost_double.resize(n_rows, std::vector<double>(n_cols));
        for (int i = 0; i < n_rows; ++i)
            for (int j = 0; j < n_cols; ++j)
                cost_double[i][j] = static_cast<double>(cost[i][j]);
    }
    const double INF = 1e18;
    std::vector<double> u(n_rows + 1, 0.0);
    std::vector<double> v(n_cols + 1, 0.0);
    std::vector<int> p(n_cols + 1, 0);
    std::vector<int> way(n_cols + 1, 0);
    for (int i = 1; i <= n_rows; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n_cols + 1, INF);
        std::vector<bool> used(n_cols + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0];
            double delta = INF;
            int j1 = 0;
            for (int j = 1; j <= n_cols; ++j) {
                if (!used[j]) {
                    double cur = cost_double[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n_cols; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<std::pair<int,int>> matches;
    if (!transposed) {
        for (int j = 1; j <= n_cols; ++j) {
            if (p[j] != 0) {
                matches.emplace_back(p[j] - 1, j - 1);
            }
        }
    } else {
        for (int j = 1; j <= n_cols; ++j) {
            if (p[j] != 0) {
                matches.emplace_back(j - 1, p[j] - 1);
            }
        }
    }
    return matches;
}

// ─── 数据关联入口（不变）────────────────────────────────
Tracker::MatchResult Tracker::dataAssociation(
    const std::vector<Detection>& dets,
    const std::vector<Track>& trks,
    const std::vector<std::vector<float>>& cost) const
{
    float thr = normalizeThreshold(cfg_.threshold, cfg_.metric);
    std::vector<std::pair<int,int>> raw_matches;
    if (cfg_.algo == AlgoType::HUNGARIAN) raw_matches = hungarianMatch(cost);
    else raw_matches = greedyMatch(cost);
    std::vector<bool> trk_matched(trks.size(), false);
    std::vector<bool> det_matched(dets.size(), false);
    std::vector<std::pair<int,int>> final_matches;
    for (const auto& m : raw_matches) {
        float c = cost[m.first][m.second];
        if (c <= thr) {
            final_matches.push_back(m);
            trk_matched[m.first] = true;
            det_matched[m.second] = true;
        }
    }
    std::vector<int> unmatched_trks, unmatched_dets;
    for (size_t i = 0; i < trks.size(); ++i)
        if (!trk_matched[i]) unmatched_trks.push_back(i);
    for (size_t j = 0; j < dets.size(); ++j)
        if (!det_matched[j]) unmatched_dets.push_back(j);
    return {final_matches, unmatched_dets, unmatched_trks, {}, cost};
}

// ─── 自车运动补偿（保留，但未使用）──────────────────────────────
void Tracker::egoMotionCompensation(
    std::vector<Track>& trks,
    const EgoPose& prev_ego,
    const EgoPose& curr_ego) const
{
    for (auto& trk : trks) {
        trk.applyEgoMotionTransform(
            prev_ego.x, prev_ego.y, prev_ego.yaw,
            curr_ego.x, curr_ego.y, curr_ego.yaw);
    }
}

// ─── 更新已匹配的轨迹（不变）────────────────────────────────────
void Tracker::updateMatchedTracks(
    const std::vector<std::pair<int,int>>& matches,
    const std::vector<Detection>& dets,
    double timestamp)
{
    for (const auto& m : matches) {
        int trk_idx = m.first;
        int det_idx = m.second;
        Track& trk = tracks_[trk_idx];
        trk.time_since_update = 0;
        trk.hits++;
        trk.update(dets[det_idx], timestamp);
    }
}

// ─── 创建新轨迹（不变）──────────────────────────────────────────
void Tracker::createNewTracks(
    const std::vector<int>& unmatched_det_indices,
    const std::vector<Detection>& dets,
    double timestamp)
{
    for (int idx : unmatched_det_indices) {
        tracks_.emplace_back(next_id_++, dets[idx], timestamp);
    }
}

// ─── 输出确认的轨迹（不变）──────────────────────────────────────
std::vector<Track> Tracker::outputConfirmedTracks()
{
    std::vector<Track> keep;
    keep.reserve(tracks_.size());
    for (auto& trk : tracks_) {
        if (trk.state == TrackState::Removed) continue;
        if (trk.time_since_update > cfg_.max_lost_frames) {
            trk.state = TrackState::Removed;
            continue;
        }
        keep.push_back(trk);
    }
    tracks_ = std::move(keep);
    std::vector<Track> confirmed;
    for (const auto& trk : tracks_) {
        if (trk.is_confirmed())
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

std::vector<Track> Tracker::update(const std::vector<Detection>& detections,
                                   double timestamp,
                                   const EgoPose& current_ego) {
    // 跳变检测
    if (current_ego.valid && prev_ego_pose_.valid) {
        double dx = current_ego.x - prev_ego_pose_.x;
        double dy = current_ego.y - prev_ego_pose_.y;
        double dpos = std::sqrt(dx*dx + dy*dy);
        float dyaw = std::abs(current_ego.yaw - prev_ego_pose_.yaw);
        dyaw = std::min(dyaw, 2.0f * float(M_PI) - dyaw);
        if (dpos > 10.0 || dyaw > 0.5f) {
            tracks_.clear();
            next_id_ = 1;
        }
    }
    if (current_ego.valid) prev_ego_pose_ = current_ego;

    // 计算自车全局速度（通过位姿差分）
    EgoPose ego_with_vel = current_ego;
    if (current_ego.valid && prev_timestamp_ > 0 && prev_ego_pose_for_velocity_.valid) {
        double dt = timestamp - prev_timestamp_;
        if (dt > 1e-6 && dt < 1.0) {
            ego_with_vel.vx = (current_ego.x - prev_ego_pose_for_velocity_.x) / dt;
            ego_with_vel.vy = (current_ego.y - prev_ego_pose_for_velocity_.y) / dt;
        } else {
            ego_with_vel.vx = 0.0; ego_with_vel.vy = 0.0;
        }
    } else {
        ego_with_vel.vx = 0.0; ego_with_vel.vy = 0.0;
    }
    prev_ego_pose_for_velocity_ = current_ego;
    prev_timestamp_ = timestamp;

    std::vector<Detection> world_dets;
    if (cfg_.enable_ego_comp && ego_with_vel.valid) {
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            Detection det_world = det;
            localToWorld(det.x, det.y, det.yaw, det.vx, det.vy, ego_with_vel, det_world.x, det_world.y, det_world.yaw, det_world.vx, det_world.vy);
            world_dets.push_back(det_world);
        }
    } else {
        world_dets = detections;
    }

    double dt = cfg_.dt;
    for (auto& trk : tracks_) {
        trk.predict(dt);
    }

    auto cost = buildCostMatrix(world_dets, tracks_);
    auto matchRes = dataAssociation(world_dets, tracks_, cost);

    for (const auto& m : matchRes.matches) {
        Track& trk = tracks_[m.first];
        const Detection& det = world_dets[m.second];
        trk.time_since_update = 0;
        trk.hits++;
        trk.update(det, timestamp);
    }

    for (int idx : matchRes.unmatched_dets) {
        tracks_.emplace_back(next_id_++, world_dets[idx], timestamp);
    }

    std::vector<Track> keep;
    for (auto& trk : tracks_) {
        if (trk.state == TrackState::Removed) continue;
        if (trk.time_since_update > cfg_.max_lost_frames) {
            trk.state = TrackState::Removed;
            continue;
        }
        keep.push_back(trk);
    }
    tracks_ = std::move(keep);

    std::vector<Track> confirmed;
    for (auto& trk : tracks_) {
        if (!trk.is_confirmed()) continue;
        Track out_trk = trk;
        if (cfg_.enable_ego_comp && ego_with_vel.valid) {
            float local_x, local_y, local_yaw, local_vx, local_vy;
            worldToLocal(trk.x, trk.y, trk.yaw, trk.vx, trk.vy, ego_with_vel, local_x, local_y, local_yaw, local_vx, local_vy);
            out_trk.global_vx = trk.vx;
            out_trk.global_vy = trk.vy;
            out_trk.vx = local_vx;
            out_trk.vy = local_vy;
            out_trk.x = local_x;
            out_trk.y = local_y;
            out_trk.yaw = local_yaw;
        } else {
            out_trk.global_vx = trk.vx;
            out_trk.global_vy = trk.vy;
        }
        confirmed.push_back(out_trk);
    }
    return confirmed;
}


}  // namespace tracking
}  // namespace fastbev