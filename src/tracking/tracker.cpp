/**
 * tracker.cpp — 多目标跟踪器实现
 */

#include "tracker.hpp"
#include <algorithm>
#include <limits>
#include <numeric>

namespace fastbev {
namespace tracking {

// ─── 简化匈牙利算法（基于贪心近似，足够用于低目标数场景）──────────────────

std::vector<std::pair<int,int>> Tracker::hungarian_match(
    const std::vector<std::vector<float>>& cost,
    float max_cost)
{
    int n_tracks = static_cast<int>(cost.size());
    int n_dets   = n_tracks > 0 ? static_cast<int>(cost[0].size()) : 0;
    if (n_tracks == 0 || n_dets == 0) return {};

    // 构造代价列表并排序
    struct Item { float c; int ti, di; };
    std::vector<Item> items;
    items.reserve(n_tracks * n_dets);
    for (int i = 0; i < n_tracks; ++i)
        for (int j = 0; j < n_dets; ++j)
            if (cost[i][j] <= max_cost)
                items.push_back({cost[i][j], i, j});

    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b){
        return a.c < b.c;
    });

    std::vector<bool> used_t(n_tracks, false);
    std::vector<bool> used_d(n_dets,   false);
    std::vector<std::pair<int,int>> matches;

    for (const auto& it : items) {
        if (!used_t[it.ti] && !used_d[it.di]) {
            matches.push_back({it.ti, it.di});
            used_t[it.ti] = true;
            used_d[it.di] = true;
        }
    }
    return matches;
}

// ─── 构造 ─────────────────────────────────────────────────────────────────

Tracker::Tracker(const TrackerConfig& cfg) : cfg_(cfg) {}

// ─── 重置 ─────────────────────────────────────────────────────────────────

void Tracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}

// ─── 核心更新逻辑 ──────────────────────────────────────────────────────────

std::vector<Track> Tracker::update(const std::vector<Detection>& detections,
                                    double timestamp)
{
    int n_tracks = static_cast<int>(tracks_.size());
    int n_dets   = static_cast<int>(detections.size());

    // 1. 预测所有现有轨迹
    for (auto& t : tracks_) t.predict(cfg_.dt);

    // 2. 构造代价矩阵（仅同类别间匹配）
    std::vector<std::vector<float>> cost(n_tracks, std::vector<float>(n_dets, cfg_.max_dist + 1.f));
    for (int i = 0; i < n_tracks; ++i) {
        for (int j = 0; j < n_dets; ++j) {
            if (tracks_[i].class_id != detections[j].class_id) continue;
            cost[i][j] = tracks_[i].dist2d(detections[j]);
        }
    }

    // 3. 匈牙利匹配
    auto matches = hungarian_match(cost, cfg_.max_dist);

    // 4. 更新匹配轨迹
    std::vector<bool> matched_t(n_tracks, false);
    std::vector<bool> matched_d(n_dets,   false);

    for (const auto& [ti, di] : matches) {
        tracks_[ti].update(detections[di], timestamp);
        matched_t[ti] = true;
        matched_d[di] = true;
    }

    // 5. 未匹配检测 → 新轨迹
    for (int j = 0; j < n_dets; ++j) {
        if (!matched_d[j]) {
            tracks_.emplace_back(next_id_++, detections[j], timestamp);
        }
    }

    // 6. 未匹配轨迹 → 检查是否超时
    std::vector<Track> keep;
    keep.reserve(tracks_.size());
    for (auto& t : tracks_) {
        if (t.state == TrackState::Removed) continue;
        if (t.time_since_update > cfg_.max_lost_frames) {
            t.state = TrackState::Removed;
        } else {
            keep.push_back(t);
        }
    }
    tracks_ = std::move(keep);

    // 7. 返回确认轨迹
    std::vector<Track> output;
    for (const auto& t : tracks_) {
        if (t.is_confirmed()) output.push_back(t);
    }
    return output;
}

}  // namespace tracking
}  // namespace fastbev
