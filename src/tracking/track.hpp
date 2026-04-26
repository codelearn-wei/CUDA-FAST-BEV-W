#pragma once
/**
 * track.hpp — 单目标轨迹数据结构
 *
 * 定义跟踪目标的状态、历史轨迹和生命周期管理接口。
 * 与 BEV 检测后处理 (BoundingBox) 解耦，通过 Detection 中间结构进行交互。
 */

#include <cstdint>
#include <vector>
#include <array>
#include <cmath>

namespace fastbev {
namespace tracking {

// ─── 检测结果中间结构（由 BoundingBox 转换而来）────────────────────────────

struct Detection {
    float x, y, z;          // BEV 中心坐标（lidar 坐标系，米）
    float w, l, h;          // 尺寸：宽/长/高
    float yaw;              // 偏航角（弧度）
    float vx, vy;           // 速度（m/s，可能来自检测器输出）
    float score;            // 置信度
    int   class_id;         // 类别 id（0~9）
};

// ─── 轨迹状态机 ────────────────────────────────────────────────────────────

enum class TrackState : uint8_t {
    New     = 0,   // 刚创建，尚未确认
    Active  = 1,   // 稳定跟踪中
    Lost    = 2,   // 暂时丢失（仍在预测）
    Removed = 3,   // 已移除
};

// ─── 轨迹历史节点 ──────────────────────────────────────────────────────────

struct TrackPoint {
    float x, y, z;          // 中心位置
    float yaw;
    float vx, vy;
    double timestamp;       // UNIX 时间戳（秒）或帧号
};

// ─── 单目标轨迹 ────────────────────────────────────────────────────────────

class Track {
public:
    // ── 标识 ────────────────────────────────────────────────────
    uint64_t   track_id;
    int        class_id;
    TrackState state;

    // ── 当前状态估计（卡尔曼滤波或直接更新）──────────────────────
    float x, y, z;
    float vx, vy;
    float yaw;
    float w, l, h;
    float score;

    // ── 生命周期计数器 ──────────────────────────────────────────
    int hits;            // 连续匹配次数
    int age;             // 总帧数
    int time_since_update; // 上次匹配后经过的帧数

    // ── 历史轨迹（最多保留 N 帧）────────────────────────────────
    std::vector<TrackPoint> history;
    static constexpr int MAX_HISTORY = 30;

    // ── 构造 ─────────────────────────────────────────────────────
    Track() = default;
    explicit Track(uint64_t id, const Detection& det, double timestamp = 0.0)
        : track_id(id), class_id(det.class_id), state(TrackState::New),
          x(det.x), y(det.y), z(det.z),
          vx(det.vx), vy(det.vy), yaw(det.yaw),
          w(det.w), l(det.l), h(det.h),
          score(det.score),
          hits(1), age(1), time_since_update(0)
    {
        _push_history(timestamp);
    }

    // ── 更新（匹配到新检测）──────────────────────────────────────
    void update(const Detection& det, double timestamp = 0.0) {
        // 简单低通滤波更新（也可换为卡尔曼滤波）
        constexpr float alpha = 0.7f;
        x     = alpha * det.x   + (1.f - alpha) * (x + vx * 0.05f);
        y     = alpha * det.y   + (1.f - alpha) * (y + vy * 0.05f);
        z     = alpha * det.z   + (1.f - alpha) * z;
        vx    = alpha * det.vx  + (1.f - alpha) * vx;
        vy    = alpha * det.vy  + (1.f - alpha) * vy;
        yaw   = _lerp_angle(yaw, det.yaw, alpha);
        w     = alpha * det.w   + (1.f - alpha) * w;
        l     = alpha * det.l   + (1.f - alpha) * l;
        h     = alpha * det.h   + (1.f - alpha) * h;
        score = det.score;

        hits++;
        age++;
        time_since_update = 0;
        state = TrackState::Active;
        _push_history(timestamp);
    }

    // ── 预测（无匹配时）──────────────────────────────────────────
    void predict(double dt = 0.05) {
        x += vx * static_cast<float>(dt);
        y += vy * static_cast<float>(dt);
        age++;
        time_since_update++;
        if (time_since_update >= 1) state = TrackState::Lost;
    }

    // ── BEV 中心距离（用于关联）────────────────────────────────
    float dist2d(const Detection& det) const {
        float dx = x - det.x;
        float dy = y - det.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    bool is_confirmed() const { return hits >= 2 && state != TrackState::Removed; }
    bool is_active()    const { return state == TrackState::Active; }

private:
    void _push_history(double ts) {
        if (static_cast<int>(history.size()) >= MAX_HISTORY)
            history.erase(history.begin());
        history.push_back({x, y, z, yaw, vx, vy, ts});
    }

    static float _lerp_angle(float a, float b, float t) {
        // 角度插值：处理 ±π 跳变
        float diff = b - a;
        while (diff >  M_PI) diff -= 2.f * M_PI;
        while (diff < -M_PI) diff += 2.f * M_PI;
        return a + t * diff;
    }
};

}  // namespace tracking
}  // namespace fastbev
