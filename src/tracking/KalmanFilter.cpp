#include "KalmanFilter.hpp"
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

namespace fastbev {
namespace tracking {

// 匿名命名空间：内部辅助函数
namespace {
    // 计算 target - source 的最小弧差，范围 (-π, π]
    double smallestAngleDiff(double target, double source) {
        double diff = std::fmod(target - source + M_PI, 2.0 * M_PI);
        if (diff < 0) diff += 2.0 * M_PI;
        return diff - M_PI;
    }
}

KalmanFilter::KalmanFilter() : last_x_(0.0), last_y_(0.0) {
    // 状态向量: [x, y, z, yaw, l, w, h, v, omega, vz] 共10维
    x_ = Eigen::VectorXd::Zero(dim_x);
    P_ = Eigen::MatrixXd::Identity(dim_x, dim_x);
    
    // 位置、尺寸初始不确定度
    P_.block<7,7>(0,0).diagonal() << 1.0, 1.0, 1.0, 0.1, 0.2, 0.2, 0.2;
    // 速率 v、角速度 omega、垂直速度 vz 初始不确定度
    P_(7,7) = 25.0;   // v 标准差 5 m/s
    P_(8,8) = 1.0;    // omega 标准差 1 rad/s
    P_(9,9) = 25.0;   // vz 标准差 5 m/s

    // 测量矩阵 H 固定：只观测前7维 [x,y,z,yaw,l,w,h]
    H_ = Eigen::MatrixXd::Zero(dim_z, dim_x);
    H_.block<7,7>(0,0) = Eigen::MatrixXd::Identity(7,7);

    // 测量噪声协方差 R （7x7）
    R_ = Eigen::MatrixXd::Identity(dim_z, dim_z);
    R_(0,0) = 0.01;   // x 0.1m
    R_(1,1) = 0.01;   // y 0.1m
    R_(2,2) = 0.01;   // z 0.1m
    R_(3,3) = 0.02;   // yaw ~0.14 rad
    R_(4,4) = 0.05; R_(5,5) = 0.05; R_(6,6) = 0.05; // 尺寸

    // 过程噪声协方差 Q 基准值（预测时会乘 dt）
    Q_ = Eigen::MatrixXd::Zero(dim_x, dim_x);
    double q_pos  = 0.04;   // 位置随机游走 (m²/s)
    double q_yaw  = 0.01;   // 航向过程噪声
    double q_size = 0.005;  // 尺寸缓慢变化
    double q_v    = 0.5;    // 速率噪声
    double q_omega= 0.2;    // 角速度噪声
    double q_vz   = 0.25;   // 垂直速度噪声
    Q_.diagonal() << q_pos, q_pos, q_pos, q_yaw, q_size, q_size, q_size,
                     q_v, q_omega, q_vz;
}

void KalmanFilter::init(const Eigen::VectorXd& bbox3d) {
    // bbox3d: [x, y, z, yaw, l, w, h] 共7维
    x_.head(7) = bbox3d;
    x_(7) = 0.0;      // v 初始为0
    x_(8) = 0.0;      // omega 初始为0
    x_(9) = 0.0;      // vz 初始为0
    last_x_ = x_(0);
    last_y_ = x_(1);
}

void KalmanFilter::predict(double dt) {
    double x   = x_(0);
    double y   = x_(1);
    double z   = x_(2);
    double yaw = x_(3);
    double v   = x_(7);
    double omega = x_(8);
    double vz  = x_(9);

    // CTRV 状态转移
    double new_x, new_y, new_z, new_yaw;
    new_z = z + vz * dt;
    new_yaw = normalizeAngle(yaw + omega * dt);
    
    if (std::fabs(omega) > 1e-6) {
        double yaw_final = new_yaw;
        new_x = x + (v / omega) * (std::sin(yaw_final) - std::sin(yaw));
        new_y = y + (v / omega) * (std::cos(yaw) - std::cos(yaw_final));
    } else {
        // 直线运动
        new_x = x + v * std::cos(yaw) * dt;
        new_y = y + v * std::sin(yaw) * dt;
    }

    // 更新状态（尺寸和速度本身不变，由过程噪声驱动变化）
    x_(0) = new_x;
    x_(1) = new_y;
    x_(2) = new_z;
    x_(3) = new_yaw;
    // x_[4..6] 尺寸保持不变
    // x_[7..9] 速度、角速度保持不变

    // 状态转移雅可比 F (10x10)
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(dim_x, dim_x);
    if (std::fabs(omega) > 1e-6) {
        double yaw_final = new_yaw;
        double s_yaw  = std::sin(yaw);
        double c_yaw  = std::cos(yaw);
        double s_yawp = std::sin(yaw_final);
        double c_yawp = std::cos(yaw_final);
        double v_o_w  = v / omega;
        double v_o_w2 = v / (omega * omega);

        F(0,3) = v_o_w * (c_yawp - c_yaw);
        F(0,7) = (s_yawp - s_yaw) / omega;
        F(0,8) = -v_o_w2 * (s_yawp - s_yaw) + v_o_w * (dt * c_yawp);

        F(1,3) = v_o_w * (s_yawp - s_yaw);
        F(1,7) = (c_yaw - c_yawp) / omega;
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
    F(2,9) = dt;      // dz/dvz
    F(3,8) = dt;      // dyaw/domega

    // 过程噪声缩放（离散化近似：Q_k = Q * dt）
    Eigen::MatrixXd Q_scaled = Q_ * dt;
    P_ = F * P_ * F.transpose() + Q_scaled;
}

void KalmanFilter::update(const Eigen::VectorXd& z_full) {
    Eigen::VectorXd z = z_full.head(7);
    Eigen::VectorXd z_pred = x_.head(7);
    Eigen::VectorXd y = z - z_pred;
    y(3) = smallestAngleDiff(z(3), z_pred(3));

    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(dim_z, dim_z));

    x_ += K * y;
    P_ = (Eigen::MatrixXd::Identity(dim_x, dim_x) - K * H_) * P_;

    // ========== 强制 yaw 对齐（完全信任观测） ==========
    x_(3) = normalizeAngle(z(3));           // 状态 yaw 直接覆写
    P_.row(3).setZero();                    // 清除 yaw 与其它状态的协方差增益
    P_.col(3).setZero();
    P_(3,3) = 0.1;                          // 保持适当的初始不确定度，避免后续卡死
}

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

void KalmanFilter::applyEgoTransform(
    double prev_ego_x, double prev_ego_y, double prev_ego_yaw,
    double curr_ego_x, double curr_ego_y, double curr_ego_yaw)
{
    // 自车运动增量
    double dx_ego = curr_ego_x - prev_ego_x;
    double dy_ego = curr_ego_y - prev_ego_y;
    double dyaw_ego = normalizeAngle(curr_ego_yaw - prev_ego_yaw);

    double cos_dyaw = std::cos(dyaw_ego);
    double sin_dyaw = std::sin(dyaw_ego);

    // 目标位置从上一帧局部坐标系转到当前帧局部坐标系
    double x_prev = x_(0);
    double y_prev = x_(1);
    // 旋转后再平移（坐标系定义：x右，y前）
    x_(0) = cos_dyaw * x_prev - sin_dyaw * y_prev - dx_ego;
    x_(1) = sin_dyaw * x_prev + cos_dyaw * y_prev - dy_ego;

    // 航向角：全局 yaw 不变，局部 yaw 减去自车 yaw 变化
    x_(3) = normalizeAngle(x_(3) - dyaw_ego);

    // 协方差位置部分旋转
    Eigen::Matrix2d R_ego;
    R_ego << cos_dyaw, -sin_dyaw, sin_dyaw, cos_dyaw;
    P_.block<2,2>(0,0) = R_ego * P_.block<2,2>(0,0) * R_ego.transpose();
    // 交叉项简化处理（可进一步扩展）
}

} // namespace tracking
} // namespace fastbev