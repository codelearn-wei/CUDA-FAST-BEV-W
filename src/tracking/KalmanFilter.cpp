#include "KalmanFilter.hpp"
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

namespace fastbev {
namespace tracking {

namespace {
    double smallestAngleDiff(double target, double source) {
        double diff = std::fmod(target - source + M_PI, 2.0 * M_PI);
        if (diff < 0) diff += 2.0 * M_PI;
        return diff - M_PI;
    }
    double normalizeAngle(double theta) {
        while (theta > M_PI) theta -= 2.0 * M_PI;
        while (theta < -M_PI) theta += 2.0 * M_PI;
        return theta;
    }
}

KalmanFilter::KalmanFilter() : last_x_(0.0), last_y_(0.0),
                               has_prev_(false), prev_measurement_(dim_z), prev_timestamp_(0.0),
                               min_speed_for_motion_(0.5), init_frame_count_(0) {
    x_ = Eigen::VectorXd::Zero(dim_x);
    P_ = Eigen::MatrixXd::Identity(dim_x, dim_x);
    
    P_.block<7,7>(0,0).diagonal() << 1.0, 1.0, 1.0, 0.1, 0.2, 0.2, 0.2;
    P_(7,7) = 25.0;
    P_(8,8) = 1.0;
    P_(9,9) = 25.0;

    H_ = Eigen::MatrixXd::Zero(dim_z, dim_x);
    H_.block<7,7>(0,0) = Eigen::MatrixXd::Identity(7,7);

    R_ = Eigen::MatrixXd::Identity(dim_z, dim_z);
    R_(0,0) = 0.0001;  // x 标准差 0.05m (更信任)
    R_(1,1) = 0.0001;  // y 标准差 0.05m
    R_(2,2) = 0.01;    // z 保持 0.1m
    R_(3,3) = 0.005;   // yaw 标准差 ~0.07 rad (约4°，更信任)
    // 尺寸观测噪声可以保留或稍增大，因为尺寸不是重点
    R_(4,4) = 0.1;     // 放宽尺寸，因为不信任速度但尺寸关联不大
    R_(5,5) = 0.1;
    R_(6,6) = 0.1;

    Q_base_ = Eigen::MatrixXd::Zero(dim_x, dim_x);
    double q_pos  = 0.04;
    double q_yaw  = 0.01;
    double q_size = 0.005;
    double q_v    = 2.0;
    double q_omega= 1.0;
    double q_vz   = 1.0;
    Q_base_.diagonal() << q_pos, q_pos, q_pos, q_yaw, q_size, q_size, q_size,
                          q_v, q_omega, q_vz;
}

void KalmanFilter::init(const Eigen::VectorXd& bbox3d) {
    x_.head(7) = bbox3d;
    x_(7) = 0.0;
    x_(8) = 0.0;
    x_(9) = 0.0;
    last_x_ = x_(0);
    last_y_ = x_(1);
    has_prev_ = false;
    init_frame_count_ = 0;
    applyVehicleKinematics();
}

void KalmanFilter::predict(double dt) {
    double x   = x_(0);
    double y   = x_(1);
    double z   = x_(2);
    double yaw = x_(3);
    double v   = x_(7);
    double omega = x_(8);
    double vz  = x_(9);

    bool is_moving = (v >= min_speed_for_motion_);
    double effective_omega = omega;
    if (!is_moving) {
        effective_omega = 0.0;
        vz = 0.0;
        x_(8) = 0.0;
        x_(9) = 0.0;
    } else {
        // 物理限制：最大角速度 1.0 rad/s
        const double max_omega = 1.0;
        effective_omega = std::max(-max_omega, std::min(omega, max_omega));
        // 如果速度很低但大于阈值，也限制 omega 更严
        if (v < 1.0) effective_omega *= (v / 1.0);
        x_(8) = effective_omega;
    }
    
    // 预测新位置
    double new_x, new_y, new_z, new_yaw;
    new_z = z + vz * dt;
    new_yaw = normalizeAngle(yaw + effective_omega * dt);
    
    if (std::fabs(effective_omega) > 1e-6) {
        double v_over_omega = v / effective_omega;
        new_x = x + v_over_omega * (std::sin(new_yaw) - std::sin(yaw));
        new_y = y + v_over_omega * (std::cos(yaw) - std::cos(new_yaw));
    } else {
        new_x = x + v * std::cos(yaw) * dt;
        new_y = y + v * std::sin(yaw) * dt;
    }
    
    x_(0) = new_x;
    x_(1) = new_y;
    x_(2) = new_z;
    x_(3) = new_yaw;
    
    // 雅可比矩阵（与之前相同，略写）
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(dim_x, dim_x);
    if (std::fabs(effective_omega) > 1e-6) {
        double s_yaw  = std::sin(yaw);
        double c_yaw  = std::cos(yaw);
        double s_yawp = std::sin(new_yaw);
        double c_yawp = std::cos(new_yaw);
        double v_o_w  = v / effective_omega;
        double v_o_w2 = v / (effective_omega * effective_omega);
        F(0,3) = v_o_w * (c_yawp - c_yaw);
        F(0,7) = (s_yawp - s_yaw) / effective_omega;
        F(0,8) = -v_o_w2 * (s_yawp - s_yaw) + v_o_w * (dt * c_yawp);
        F(1,3) = v_o_w * (s_yawp - s_yaw);
        F(1,7) = (c_yaw - c_yawp) / effective_omega;
        F(1,8) = -v_o_w2 * (c_yaw - c_yawp) + v_o_w * (dt * s_yawp);
    } else {
        double s_yaw = std::sin(yaw);
        double c_yaw = std::cos(yaw);
        F(0,3) = -v * s_yaw * dt;
        F(0,7) = c_yaw * dt;
        F(0,8) = -0.5 * v * s_yaw * dt * dt;
        F(1,3) =  v * c_yaw * dt;
        F(1,7) = s_yaw * dt;
        F(1,8) =  0.5 * v * c_yaw * dt * dt;
    }
    F(2,9) = dt;
    F(3,8) = dt;
    
    // 自适应过程噪声
    Eigen::MatrixXd Q_adaptive = Q_base_;
    if (!is_moving) {
        Q_adaptive(8,8) = 1e-6;   // omega 几乎不变
        Q_adaptive(9,9) = 1e-6;
        Q_adaptive(7,7) = 0.1;
    } else {
        double speed_factor = std::min(2.0, v / 5.0);
        Q_adaptive(8,8) = 0.5 * (1.0 + speed_factor);
    }
    Eigen::MatrixXd Q_scaled = Q_adaptive * dt;
    P_ = F * P_ * F.transpose() + Q_scaled;
    
    applyVehicleKinematics();
}

void KalmanFilter::update(const Eigen::VectorXd& z_full, double timestamp) {
    Eigen::VectorXd z = z_full.head(7);
    
    // 两帧差分初始化速度（保留原有代码）
    if (!has_prev_) {
        prev_measurement_ = z;
        prev_timestamp_ = timestamp;
        has_prev_ = true;
    } else {
        double dt = timestamp - prev_timestamp_;
        if (dt > 0.01 && dt < 0.5) {
            double dx = z(0) - prev_measurement_(0);
            double dy = z(1) - prev_measurement_(1);
            double dz = z(2) - prev_measurement_(2);
            double dyaw = smallestAngleDiff(z(3), prev_measurement_(3));
            double v_est = std::sqrt(dx*dx + dy*dy) / dt;
            double omega_est = dyaw / dt;
            double vz_est = dz / dt;
            if (v_est < min_speed_for_motion_) {
                omega_est = 0.0;
                vz_est = 0.0;
                v_est = 0.0;
            }
            x_(7) = v_est;
            x_(8) = omega_est;
            x_(9) = vz_est;
            if (v_est < min_speed_for_motion_) {
                P_(7,7) = 0.01;
                P_(8,8) = 1e-6;
                P_(9,9) = 0.01;
            } else {
                P_(7,7) = 4.0;
                P_(8,8) = 0.5;
                P_(9,9) = 4.0;
            }
        }
        has_prev_ = false;
    }
    
    // ========== 核心：自适应 yaw 残差（解决 π 歧义和跳变） ==========
    Eigen::VectorXd z_pred = x_.head(7);
    Eigen::VectorXd y = z - z_pred;
    
    // 原始残差
    double raw_diff = smallestAngleDiff(z(3), z_pred(3));
    double best_diff = raw_diff;
    double best_yaw_obs = z(3);
    
    // 尝试观测 yaw 翻转 π 后的残差
    double z_yaw_flipped = normalizeAngle(z(3) + M_PI);
    double flipped_diff = smallestAngleDiff(z_yaw_flipped, z_pred(3));
    if (std::abs(flipped_diff) < std::abs(best_diff)) {
        best_diff = flipped_diff;
        best_yaw_obs = z_yaw_flipped;
    }
    // 再尝试观测 yaw 减去 π
    double z_yaw_flipped2 = normalizeAngle(z(3) - M_PI);
    double flipped_diff2 = smallestAngleDiff(z_yaw_flipped2, z_pred(3));
    if (std::abs(flipped_diff2) < std::abs(best_diff)) {
        best_diff = flipped_diff2;
        best_yaw_obs = z_yaw_flipped2;
    }
    
    y(3) = best_diff;
    // 如果选择了翻转后的观测，需要调整 z 向量的 yaw 值用于后续协方差计算？不需要，因为我们只改了残差。
    // 但为了保持一致性，可以临时替换 z(3) = best_yaw_obs，但会影响其他部分？不推荐。
    // 这里仅修改变换残差，足够。
    // ---------------------------------------------------------------
    
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(dim_z, dim_z));
    
    x_ += K * y;
    P_ = (Eigen::MatrixXd::Identity(dim_x, dim_x) - K * H_) * P_;
    
    // 更新后强制状态一致性
    applyVehicleKinematics();
    
    // 增加初始化稳定期：前3帧不更新角速度（可选）
    if (init_frame_count_ < 3) {
        x_(8) = 0.0;
        P_(8,8) = 1e-4;
        init_frame_count_++;
    }
}

void KalmanFilter::applyVehicleKinematics() {
    double v = x_(7);
    if (v < min_speed_for_motion_) {
        x_(7) = 0.0;
        x_(8) = 0.0;
        x_(9) = 0.0;
        P_(7,7) = 0.01;
        P_(8,8) = 1e-6;
        P_(9,9) = 0.01;
    } else {
        // 限制角速度
        const double max_omega = 1.0;
        if (std::abs(x_(8)) > max_omega) {
            x_(8) = (x_(8) > 0) ? max_omega : -max_omega;
            P_(8,8) = std::max(P_(8,8), 0.1);
        }
        // 限制垂直速度
        const double max_vz = 2.0;
        if (std::abs(x_(9)) > max_vz) {
            x_(9) = (x_(9) > 0) ? max_vz : -max_vz;
            P_(9,9) = std::max(P_(9,9), 0.1);
        }
    }
    x_(3) = normalizeAngle(x_(3));
}

// 其余原有方法保持不变（computeInnovationMatrix, mahalanobisDistance, applyEgoTransform）
Eigen::MatrixXd KalmanFilter::computeInnovationMatrix() const {
    return H_ * P_ * H_.transpose() + R_;
}

double KalmanFilter::normalizeAngle(double theta) {
    while (theta > M_PI) theta -= 2.0 * M_PI;
    while (theta < -M_PI) theta += 2.0 * M_PI;
    return theta;
}

double KalmanFilter::mahalanobisDistance(const Eigen::VectorXd& z_full) const {
    Eigen::VectorXd z = z_full.head(7);
    Eigen::VectorXd z_pred = x_.head(7);
    Eigen::VectorXd y = z - z_pred;
    y(3) = smallestAngleDiff(z(3), z_pred(3));
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    double dist2 = y.transpose() * S.ldlt().solve(y);
    return std::sqrt(std::max(0.0, dist2));
}

} // namespace tracking
} // namespace fastbev