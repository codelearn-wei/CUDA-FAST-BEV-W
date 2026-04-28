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
#include "KalmanFilter.hpp"

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
    // 现有成员变量保持不变
    uint64_t   track_id;
    int        class_id;
    TrackState state;
    float x, y, z;
    float vx, vy;
    float yaw;
    float w, l, h;
    float score;
    int hits;
    int age;
    int time_since_update;
    std::vector<TrackPoint> history;
    static constexpr int MAX_HISTORY = 30;

    // 构造函数：使用 KF 初始化
    explicit Track(uint64_t id, const Detection& det, double timestamp = 0.0)
        : track_id(id), class_id(det.class_id), state(TrackState::New),
          x(det.x), y(det.y), z(det.z),
          vx(det.vx), vy(det.vy), yaw(det.yaw),
          w(det.w), l(det.l), h(det.h),
          score(det.score),
          hits(1), age(1), time_since_update(0)
    {
        // 构造测量向量 [x, y, z, yaw, l, w, h]
        Eigen::VectorXd meas(7);
        meas << det.x, det.y, det.z, det.yaw, det.l, det.w, det.h;
        kf_.init(meas);
        _push_history(timestamp);
    }

    // 更新（匹配到新检测）
    void update(const Detection& det, double timestamp = 0.0) {
        // ── π 歧义修正 ────────────────────────────────────────────────────
        // BEV 检测器存在车头/车尾方向歧义，相邻帧同一目标 yaw 可能相差约 π。
        // 将检测 yaw 调整到与当前 track yaw 最接近的等效方向后再做 KF 更新。
        float det_yaw = det.yaw;
        float diff = det_yaw - yaw;
        while (diff >  static_cast<float>(M_PI)) diff -= 2.0f * static_cast<float>(M_PI);
        while (diff < -static_cast<float>(M_PI)) diff += 2.0f * static_cast<float>(M_PI);
        if (std::abs(diff) > static_cast<float>(M_PI) / 2.0f) {
            // 翻转 π 后更接近 track 当前 yaw
            det_yaw += static_cast<float>(M_PI);
            while (det_yaw >  static_cast<float>(M_PI)) det_yaw -= 2.0f * static_cast<float>(M_PI);
            while (det_yaw < -static_cast<float>(M_PI)) det_yaw += 2.0f * static_cast<float>(M_PI);
        }

        Eigen::VectorXd z(7);
        z << det.x, det.y, det.z, det_yaw, det.l, det.w, det.h;
        kf_.update(z);
        syncFromKF();
        hits++;
        age++;
        time_since_update = 0;
        state = TrackState::Active;
        score = det.score;
        _push_history(timestamp);
    }

    // 预测（无匹配）
    void predict(double dt = 0.05) {
        kf_.predict(dt);
        syncFromKF();               // 同步 x, y, z, vx, vy, yaw, w, l, h
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

    // ========== 获取 8 个角点 ==========
    std::vector<std::array<float,3>> getCorners() const {
        float cos_yaw = std::cos(yaw);
        float sin_yaw = std::sin(yaw);
        float l2 = l * 0.5f;
        float w2 = w * 0.5f;
        float h2 = h * 0.5f;

        // 局部坐标（x: 向前, y: 向左, z: 向上）
        std::vector<std::array<float,3>> local = {
            { l2,  w2,  h2}, { l2,  w2, -h2}, { l2, -w2, -h2}, { l2, -w2,  h2},
            {-l2,  w2,  h2}, {-l2,  w2, -h2}, {-l2, -w2, -h2}, {-l2, -w2,  h2}
        };

        std::vector<std::array<float,3>> world(8);
        for (int i = 0; i < 8; ++i) {
            float x_rot = cos_yaw * local[i][0] + sin_yaw * local[i][2];
            float z_rot = -sin_yaw * local[i][0] + cos_yaw * local[i][2];
            world[i] = { x + x_rot, y + local[i][1], z + z_rot };
        }
        return world;
    }

    // 3D 几何中心距离（基于角点平均）
    float dist3d(const Detection& other) const {
        auto corners1 = getCorners();
        // 计算 other 的角点
        float cos_yaw2 = std::cos(other.yaw);
        float sin_yaw2 = std::sin(other.yaw);
        float l2 = other.l * 0.5f;
        float w2 = other.w * 0.5f;
        float h2 = other.h * 0.5f;
        std::vector<std::array<float,3>> local2 = {
            { l2,  w2,  h2}, { l2,  w2, -h2}, { l2, -w2, -h2}, { l2, -w2,  h2},
            {-l2,  w2,  h2}, {-l2,  w2, -h2}, {-l2, -w2, -h2}, {-l2, -w2,  h2}
        };
        std::array<float,3> c2 = {0.0f, 0.0f, 0.0f};
        for (int i = 0; i < 8; ++i) {
            float x_rot = cos_yaw2 * local2[i][0] + sin_yaw2 * local2[i][2];
            float z_rot = -sin_yaw2 * local2[i][0] + cos_yaw2 * local2[i][2];
            c2[0] += other.x + x_rot;
            c2[1] += other.y + local2[i][1];
            c2[2] += other.z + z_rot;
        }
        c2[0] /= 8.0f; c2[1] /= 8.0f; c2[2] /= 8.0f;

        std::array<float,3> c1 = {0.0f, 0.0f, 0.0f};
        for (const auto& p : corners1) {
            c1[0] += p[0];
            c1[1] += p[1];
            c1[2] += p[2];
        }
        c1[0] /= 8.0f; c1[1] /= 8.0f; c1[2] /= 8.0f;

        float dx = c1[0] - c2[0];
        float dy = c1[1] - c2[1];
        float dz = c1[2] - c2[2];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    // 马氏距离（需卡尔曼协方差）
    float mahalanobisToDetection(const Detection& det) const {
        Eigen::VectorXd z(7);
        z << det.x, det.y, det.z, det.yaw, det.l, det.w, det.h;
        return static_cast<float>(kf_.mahalanobisDistance(z));
    }

    bool is_confirmed() const { return hits >= 2 && state != TrackState::Removed; }
    bool is_active()    const { return state == TrackState::Active; }

    /**
     * 自车运动补偿：将轨迹状态从上一帧 lidar-local 坐标系变换到当前帧 lidar-local 坐标系。
     * 需在 predict() 之后、数据关联之前调用。
     */
    void applyEgoMotionTransform(
        double prev_ego_x, double prev_ego_y, double prev_ego_yaw,
        double curr_ego_x, double curr_ego_y, double curr_ego_yaw)
    {
        kf_.applyEgoTransform(prev_ego_x, prev_ego_y, prev_ego_yaw,
                              curr_ego_x, curr_ego_y, curr_ego_yaw);
        syncFromKF();
    }

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

    KalmanFilter kf_;   // 卡尔曼滤波器实例

    // 从 KF 同步成员变量
    void syncFromKF() {
        Eigen::VectorXd state = kf_.getState();      // 7 维: x,y,z,theta,l,w,h
        x   = static_cast<float>(state(0));
        y   = static_cast<float>(state(1));
        z   = static_cast<float>(state(2));
        // 角度归一化到 (-π, π]，避免 KF 内部线性融合使 yaw 漂移到 ±π 边界两侧
        {
            float raw = static_cast<float>(state(3));
            while (raw >  static_cast<float>(M_PI)) raw -= 2.0f * static_cast<float>(M_PI);
            while (raw < -static_cast<float>(M_PI)) raw += 2.0f * static_cast<float>(M_PI);
            yaw = raw;
        }
        l   = static_cast<float>(state(4));
        w   = static_cast<float>(state(5));
        h   = static_cast<float>(state(6));

        Eigen::VectorXd vel = kf_.getVelocity();     // 3 维: dx,dy,dz
        vx  = static_cast<float>(vel(0));
        vy  = static_cast<float>(vel(1));
        // vz 忽略（Track 中无此字段）
    }

};

}  // namespace tracking
}  // namespace fastbev
