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

// 局部 → 全局
void localToWorld(float x_local, float y_local, float yaw_local,
                  float vx_local, float vy_local,
                  const EgoPose& ego,
                  float& x_world, float& y_world, float& yaw_world,
                  float& vx_world, float& vy_world) {
    float cos_ego = std::cos(ego.yaw);
    float sin_ego = std::sin(ego.yaw);
    // 位置变换（不变，因为局部轴方向已由该矩阵正确映射）
    x_world = ego.x + (-sin_ego) * x_local + cos_ego * y_local;
    y_world = ego.y +  cos_ego * x_local + sin_ego * y_local;
    // 角度变换：局部朝向 α -> 全局朝向 β = θ + π - α
    yaw_world = KalmanFilter::normalizeAngle(ego.yaw + static_cast<float>(M_PI) - yaw_local);
    // 速度变换（同位置旋转）
    vx_world = (-sin_ego) * vx_local + cos_ego * vy_local;
    vy_world =  cos_ego * vx_local + sin_ego * vy_local;
}

// 全局 → 局部
void worldToLocal(float x_world, float y_world, float yaw_world,
                  float vx_world, float vy_world,
                  const EgoPose& ego,
                  float& x_local, float& y_local, float& yaw_local,
                  float& vx_local, float& vy_local) {
    float dx = x_world - ego.x;
    float dy = y_world - ego.y;
    float cos_ego = std::cos(ego.yaw);
    float sin_ego = std::sin(ego.yaw);
    // 位置逆变换
    x_local = (-sin_ego) * dx + cos_ego * dy;
    y_local =  cos_ego * dx + sin_ego * dy;
    // 角度逆变换：全局 β -> 局部 α = θ + π - β
    yaw_local = KalmanFilter::normalizeAngle(ego.yaw + static_cast<float>(M_PI) - yaw_world);
    // 速度逆变换
    vx_local = (-sin_ego) * vx_world + cos_ego * vy_world;
    vy_local =  cos_ego * vx_world + sin_ego * vy_world;
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
            printf("[Tracker] 位姿跳变，清空轨迹\n");
            tracks_.clear();
            next_id_ = 1;
        }
    }
    if (current_ego.valid) prev_ego_pose_ = current_ego;

    // 保存原始检测局部坐标（用于对比）
    std::vector<Detection> local_dets = detections;

    // 1. 检测局部 -> 全局，并打印速度
    std::vector<Detection> world_dets;
    printf("\n[Frame t=%.3f] detections=%zu\n", timestamp, detections.size());
    if (cfg_.enable_ego_comp && current_ego.valid) {
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            printf("  det%d: LOCAL  pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f\n",
                   (int)i, det.x, det.y, det.yaw, det.vx, det.vy);
            Detection det_world = det;
            localToWorld(det.x, det.y, det.yaw, det.vx, det.vy,
                         current_ego,
                         det_world.x, det_world.y, det_world.yaw,
                         det_world.vx, det_world.vy);
            world_dets.push_back(det_world);
            printf("  det%d: GLOBAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f  ego=(%6.2f,%6.2f) yaw=%6.3f\n",
                   (int)i, det_world.x, det_world.y, det_world.yaw, det_world.vx, det_world.vy,
                   current_ego.x, current_ego.y, current_ego.yaw);
        }
    } else {
        world_dets = detections;
        printf("[Warning] 未启用 ego 补偿\n");
    }

    // 2. 预测世界坐标系下的轨迹，并打印预测前后的局部和全局
    double dt = cfg_.dt;
    printf("预测 dt=%.3f, 已有轨迹数=%zu\n", dt, tracks_.size());
    static int print_cnt = 0;
    for (auto& trk : tracks_) {
        // 预测前局部坐标（用当前 ego 转换）
        float local_before_x, local_before_y, local_before_yaw, local_before_vx, local_before_vy;
        worldToLocal(trk.x, trk.y, trk.yaw,
                     trk.vx, trk.vy,
                     current_ego,
                     local_before_x, local_before_y, local_before_yaw,
                     local_before_vx, local_before_vy);
        float old_vx = trk.vx, old_vy = trk.vy;
        trk.predict(dt);
        // 预测后局部坐标
        float local_after_x, local_after_y, local_after_yaw, local_after_vx, local_after_vy;
        worldToLocal(trk.x, trk.y, trk.yaw,
                     trk.vx, trk.vy,
                     current_ego,
                     local_after_x, local_after_y, local_after_yaw,
                     local_after_vx, local_after_vy);
        if (print_cnt < 5) {
            printf("  trk%llu predict BEFORE: GLOBAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f -> LOCAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f\n",
                   trk.track_id, trk.x, trk.y, trk.yaw, old_vx, old_vy,
                   local_before_x, local_before_y, local_before_yaw, local_before_vx, local_before_vy);
            printf("           AFTER : GLOBAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f -> LOCAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f\n",
                   trk.x, trk.y, trk.yaw, trk.vx, trk.vy,
                   local_after_x, local_after_y, local_after_yaw, local_after_vx, local_after_vy);
            print_cnt++;
        }
    }

    // 3. 构建代价矩阵并关联（注意：buildCostMatrix 内部已修复角度差计算）
    auto cost = buildCostMatrix(world_dets, tracks_);
    auto matchRes = dataAssociation(world_dets, tracks_, cost);
    printf("关联结果: matches=%zu, unmatched_dets=%zu, unmatched_trks=%zu\n",
           matchRes.matches.size(), matchRes.unmatched_dets.size(), matchRes.unmatched_trks.size());

    // 打印匹配的详细信息（全局 + 局部）
    for (const auto& m : matchRes.matches) {
        const Track& trk = tracks_[m.first];
        const Detection& det = world_dets[m.second];
        // 轨迹转换到当前局部
        float trk_local_x, trk_local_y, trk_local_yaw, trk_local_vx, trk_local_vy;
        worldToLocal(trk.x, trk.y, trk.yaw,
                     trk.vx, trk.vy,
                     current_ego,
                     trk_local_x, trk_local_y, trk_local_yaw,
                     trk_local_vx, trk_local_vy);
        // 原始检测局部坐标（从 local_dets 获取）
        const Detection& local_det = local_dets[m.second];
        printf("  Match trk%llu <-> det%d: cost=%.3f\n"
               "        trk GLOBAL: pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f\n"
               "        trk LOCAL : pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f\n"
               "        det GLOBAL: pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f\n"
               "        det LOCAL : pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f\n",
               trk.track_id, m.second, cost[m.first][m.second],
               trk.x, trk.y, trk.yaw, trk.vx, trk.vy,
               trk_local_x, trk_local_y, trk_local_yaw, trk_local_vx, trk_local_vy,
               det.x, det.y, det.yaw, det.vx, det.vy,
               local_det.x, local_det.y, local_det.yaw, local_det.vx, local_det.vy);
    }

    // 4. 更新匹配的轨迹（Track::update 中已移除 π 歧义修正）
    for (const auto& m : matchRes.matches) {
        Track& trk = tracks_[m.first];
        const Detection& det = world_dets[m.second];
        float old_vx = trk.vx, old_vy = trk.vy;
        float old_x = trk.x, old_y = trk.y, old_yaw = trk.yaw;
        // 更新前局部
        float local_before_x, local_before_y, local_before_yaw, local_before_vx, local_before_vy;
        worldToLocal(old_x, old_y, old_yaw, old_vx, old_vy,
                     current_ego,
                     local_before_x, local_before_y, local_before_yaw,
                     local_before_vx, local_before_vy);
        trk.time_since_update = 0;
        trk.hits++;
        trk.update(det, timestamp);   // 此处 update 已不使用 π 翻转
        // 更新后局部
        float local_after_x, local_after_y, local_after_yaw, local_after_vx, local_after_vy;
        worldToLocal(trk.x, trk.y, trk.yaw, trk.vx, trk.vy,
                     current_ego,
                     local_after_x, local_after_y, local_after_yaw,
                     local_after_vx, local_after_vy);
        printf("    Update trk%llu:\n"
               "        BEFORE: GLOBAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f -> LOCAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f\n"
               "        AFTER : GLOBAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f -> LOCAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f\n",
               trk.track_id,
               old_x, old_y, old_yaw, old_vx, old_vy,
               local_before_x, local_before_y, local_before_yaw, local_before_vx, local_before_vy,
               trk.x, trk.y, trk.yaw, trk.vx, trk.vy,
               local_after_x, local_after_y, local_after_yaw, local_after_vx, local_after_vy);
    }

    // 5. 创建新轨迹
    for (int idx : matchRes.unmatched_dets) {
        tracks_.emplace_back(next_id_++, world_dets[idx], timestamp);
        const auto& new_trk = tracks_.back();
        float local_x, local_y, local_yaw, local_vx, local_vy;
        worldToLocal(new_trk.x, new_trk.y, new_trk.yaw,
                     new_trk.vx, new_trk.vy,
                     current_ego,
                     local_x, local_y, local_yaw,
                     local_vx, local_vy);
        printf("    New trk%llu from det%d: GLOBAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f -> LOCAL pos=(%6.2f,%6.2f) yaw=%6.3f vx=%6.3f vy=%6.3f\n",
               new_trk.track_id, idx,
               new_trk.x, new_trk.y, new_trk.yaw, new_trk.vx, new_trk.vy,
               local_x, local_y, local_yaw, local_vx, local_vy);
    }

    // 6. 清理丢失过久的轨迹
    std::vector<Track> keep;
    for (auto& trk : tracks_) {
        if (trk.state == TrackState::Removed) continue;
        if (trk.time_since_update > cfg_.max_lost_frames) {
            trk.state = TrackState::Removed;
            printf("  Remove trk%llu (lost %d frames)\n", trk.track_id, trk.time_since_update);
            continue;
        }
        keep.push_back(trk);
    }
    tracks_ = std::move(keep);

    // 7. 输出确认轨迹并转回局部坐标，同时打印局部速度
    std::vector<Track> confirmed;
    for (auto& trk : tracks_) {
        if (!trk.is_confirmed()) continue;
        Track out_trk = trk;
        if (cfg_.enable_ego_comp && current_ego.valid) {
            float local_x, local_y, local_yaw, local_vx, local_vy;
            worldToLocal(trk.x, trk.y, trk.yaw,
                         trk.vx, trk.vy,
                         current_ego,
                         local_x, local_y, local_yaw,
                         local_vx, local_vy);
            printf("  Out trk%llu: GLOBAL vx=%6.3f vy=%6.3f -> LOCAL vx=%6.3f vy=%6.3f\n",
                   trk.track_id, trk.vx, trk.vy, local_vx, local_vy);
            printf("  Out trk%llu: GLOBAL pos=(%6.2f,%6.2f) yaw=%6.3f -> LOCAL pos=(%6.2f,%6.2f) yaw=%6.3f\n",
                   trk.track_id, trk.x, trk.y, trk.yaw, local_x, local_y, local_yaw);
            out_trk.x = local_x; out_trk.y = local_y; out_trk.yaw = local_yaw;
            out_trk.vx = local_vx; out_trk.vy = local_vy;
        }
        confirmed.push_back(out_trk);
    }
    printf("最终输出轨迹数: %zu\n\n", confirmed.size());
    return confirmed;
}


}  // namespace tracking
}  // namespace fastbev