#include "KalmanFilter.hpp"
#include <cmath>
#include <iostream>

namespace fastbev {
namespace tracking {

KalmanFilter::KalmanFilter() : last_x_(0.0), last_y_(0.0) {
    x_ = Eigen::VectorXd::Zero(dim_x);
    P_ = Eigen::MatrixXd::Identity(dim_x, dim_x);
    P_.block<7,7>(0,0) *= 1.0;                    // 位置起始不确定度1m
    for (int i = 7; i < dim_x; ++i) P_(i,i) = 5.0; // 速度起始不确定度增大（让滤波器一开始就信任测量）

    H_ = Eigen::MatrixXd::Zero(dim_z, dim_x);
    for (int i = 0; i < 7; ++i) H_(i,i) = 1.0;
    H_(7,7) = 1.0; H_(8,8) = 1.0;

    // 测量噪声：减小以更信任观测
    R_ = Eigen::MatrixXd::Identity(dim_z, dim_z);
    R_(0,0)=0.1; R_(1,1)=0.1; R_(2,2)=0.1;       // 位置 0.32m 标准差
    R_(3,3)=0.02;                                 // 朝向 ≈0.14 rad
    R_(4,4)=0.05; R_(5,5)=0.05; R_(6,6)=0.05;    // 尺寸 0.22m
    R_(7,7)=0.2; R_(8,8)=0.2;                    // 速度 0.45m/s 标准差（信任速度测量）

    // 过程噪声：根据实际运动幅度设定
    Q_ = Eigen::MatrixXd::Zero(dim_x, dim_x);
    Q_(3,3) = 0.001;                              // 角度过程噪声
    // 速度过程噪声：目标加速度噪声。静止目标希望速度稳定，用较小值；动态目标要适应变化，用适中值。
    for (int i = 7; i < dim_x; ++i) Q_(i,i) = 0.1;  // 原0.5偏高，降为0.1
}

void KalmanFilter::init(const Eigen::VectorXd& bbox3d) {
    // bbox3d 应为 7 维: [x, y, z, yaw, l, w, h]
    x_.head(7) = bbox3d;
    x_.tail(3).setZero();                // 初始速度为0
    last_x_ = x_(0);
    last_y_ = x_(1);
}

void KalmanFilter::predict(double dt) {
    // 状态转移矩阵（匀速模型）
    F_ = Eigen::MatrixXd::Identity(dim_x, dim_x);
    F_(0, 7) = dt;   // x += vx * dt
    F_(1, 8) = dt;   // y += vy * dt
    F_(2, 9) = dt;   // z += vz * dt

    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
    x_(3) = normalizeAngle(x_(3));
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    // 创新向量：z 为 9 维 [x,y,z,yaw,l,w,h,vx,vy]
    Eigen::VectorXd y = z - H_ * x_;
    y(3) = normalizeAngle(y(3));          // 角度残差归一化

    // 创新协方差
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    // 卡尔曼增益
    Eigen::MatrixXd K = P_ * H_.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(dim_z, dim_z));

    // 状态更新
    x_ = x_ + K * y;
    // 协方差更新
    P_ = (Eigen::MatrixXd::Identity(dim_x, dim_x) - K * H_) * P_;

    // 角度归一化
    x_(3) = normalizeAngle(x_(3));
}

Eigen::MatrixXd KalmanFilter::computeInnovationMatrix() const {
    return H_ * P_ * H_.transpose() + R_;
}

double KalmanFilter::normalizeAngle(double theta) {
    while (theta >= M_PI) theta -= 2.0 * M_PI;
    while (theta < -M_PI) theta += 2.0 * M_PI;
    return theta;
}

double KalmanFilter::mahalanobisDistance(const Eigen::VectorXd& z) const {
    // 注意：z 应为 9 维
    Eigen::VectorXd y = z - H_ * x_;
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    double dist2 = y.transpose() * S.ldlt().solve(y);
    return std::sqrt(dist2);
}

void KalmanFilter::applyEgoTransform(
    double prev_ego_x, double prev_ego_y, double prev_ego_yaw,
    double curr_ego_x, double curr_ego_y, double curr_ego_yaw)
{
    // 此函数保持不变（局部坐标系专用，当前全局方案不会调用）
    double prev_yaw = -prev_ego_yaw;
    double curr_yaw = -curr_ego_yaw;

    double cos_prev = std::cos(prev_yaw);
    double sin_prev = std::sin(prev_yaw);
    double x_world = prev_ego_x + cos_prev * x_(0) - sin_prev * x_(1);
    double y_world = prev_ego_y + sin_prev * x_(0) + cos_prev * x_(1);

    double cos_curr = std::cos(curr_yaw);
    double sin_curr = std::sin(curr_yaw);
    double dx = x_world - curr_ego_x;
    double dy = y_world - curr_ego_y;
    x_(0) =  cos_curr * dx + sin_curr * dy;
    x_(1) = -sin_curr * dx + cos_curr * dy;

    double delta = prev_yaw - curr_yaw;
    x_(3) = normalizeAngle(x_(3) + delta);

    double cos_d = std::cos(delta);
    double sin_d = std::sin(delta);
    double vx_new = cos_d * x_(7) - sin_d * x_(8);
    double vy_new = sin_d * x_(7) + cos_d * x_(8);
    x_(7) = vx_new;
    x_(8) = vy_new;

    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(dim_x, dim_x);
    J(0,0) = cos_d; J(0,1) = -sin_d;
    J(1,0) = sin_d; J(1,1) =  cos_d;
    J(7,7) = cos_d; J(7,8) = -sin_d;
    J(8,7) = sin_d; J(8,8) =  cos_d;
    P_ = J * P_ * J.transpose();
}

} // namespace tracking
} // namespace fastbev