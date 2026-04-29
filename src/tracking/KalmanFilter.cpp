#include "KalmanFilter.hpp"
#include <cmath>

namespace fastbev {
namespace tracking {

KalmanFilter::KalmanFilter() {
    // 状态向量
    x_ = Eigen::VectorXd::Zero(dim_x);

    // 协方差矩阵
    P_ = Eigen::MatrixXd::Identity(dim_x, dim_x);
    P_ *= 10.0;                     // 位置部分不确定性
    for (int i = 7; i < dim_x; ++i) {
        P_(i, i) = 1000.0;          // 速度部分初始不确定性更大
    }

    // 测量矩阵 H: 取前7维
    H_ = Eigen::MatrixXd::Zero(dim_z, dim_x);
    for (int i = 0; i < dim_z; ++i) {
        H_(i, i) = 1.0;
    }

    // 测量噪声 R（可根据检测噪声调优，Python 代码中曾注释放大10倍）
    R_ = Eigen::MatrixXd::Identity(dim_z, dim_z);
    // R_ *= 10.0;   // 取消注释可降低对测量的信任

    // 过程噪声 Q：只对速度有微小扰动
    Q_ = Eigen::MatrixXd::Zero(dim_x, dim_x);
    Q_(3, 3) = 0.1;
    for (int i = 7; i < dim_x; ++i) {
        Q_(i, i) = 0.01;            // 与 python 代码一致
    }
}

void KalmanFilter::init(const Eigen::VectorXd& bbox3d) {
    // bbox3d 必须是 7 维: [x, y, z, theta, l, w, h]
    x_.head(7) = bbox3d;
    x_.tail(3).setZero();           // 初始速度为0
}

void KalmanFilter::predict(double dt) {
    // 构建状态转移矩阵 F（匀速模型）
    F_ = Eigen::MatrixXd::Identity(dim_x, dim_x);
    F_(0, 7) = dt;   // x += dx * dt
    F_(1, 8) = dt;   // y += dy * dt
    F_(2, 9) = dt;   // z += dz * dt

    // 预测状态
    x_ = F_ * x_;
    // 预测协方差
    P_ = F_ * P_ * F_.transpose() + Q_;

    // 角度归一化
    x_(3) = normalizeAngle(x_(3));
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    // 创新
    Eigen::VectorXd y = z - H_ * x_;
    // 【关键】对角度残差归一化到 [-π, π)
    y(3) = normalizeAngle(y(3));
    
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(dim_z, dim_z));

    // 更新状态
    x_ = x_ + K * y;
    P_ = (Eigen::MatrixXd::Identity(dim_x, dim_x) - K * H_) * P_;

    // 角度归一化（保证最终状态角度在范围内）
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
    Eigen::VectorXd y = z - H_ * x_;                                     // 创新向量
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;                  // 创新矩阵
    double dist2 = y.transpose() * S.ldlt().solve(y);                   // 马氏距离平方
    return std::sqrt(dist2);
}

void KalmanFilter::applyEgoTransform(
    double prev_ego_x, double prev_ego_y, double prev_ego_yaw,
    double curr_ego_x, double curr_ego_y, double curr_ego_yaw)
{
    // 计算自车位移（从 prev 到 curr）和自车旋转变化（delta_yaw）
    double dx_ego = curr_ego_x - prev_ego_x;
    double dy_ego = curr_ego_y - prev_ego_y;
    double delta_yaw = curr_ego_yaw - prev_ego_yaw;   // 自车逆时针旋转为正（需根据实际数据调整符号）
    
    // 旋转矩阵（将上一帧局部坐标系中的向量旋转到当前帧局部坐标系）
    double cos_d = std::cos(delta_yaw);
    double sin_d = std::sin(delta_yaw);
    
    // 1. 位置补偿：先旋转，再平移（加上自车位移）
    double x_new = cos_d * x_(0) - sin_d * x_(1) + dx_ego;
    double y_new = sin_d * x_(0) + cos_d * x_(1) + dy_ego;
    x_(0) = x_new;
    x_(1) = y_new;
    
    // 2. 航向角补偿：目标朝向应减去自车旋转（因为自车原地旋转了delta_yaw）
    x_(3) = normalizeAngle(x_(3) - delta_yaw);
    
    // 3. 速度补偿：速度向量同样旋转（不涉及平移）
    double vx_new = cos_d * x_(7) - sin_d * x_(8);
    double vy_new = sin_d * x_(7) + cos_d * x_(8);
    x_(7) = vx_new;
    x_(8) = vy_new;
    
    // 4. 协方差更新（雅可比矩阵）
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(dim_x, dim_x);
    J(0,0) = cos_d; J(0,1) = -sin_d;
    J(1,0) = sin_d; J(1,1) =  cos_d;
    J(7,7) = cos_d; J(7,8) = -sin_d;
    J(8,7) = sin_d; J(8,8) =  cos_d;
    P_ = J * P_ * J.transpose();
}

} // namespace tracking
} // namespace fastbev