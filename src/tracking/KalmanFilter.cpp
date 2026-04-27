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
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

    // 更新状态
    x_ = x_ + K * y;
    // 更新协方差（标准公式）
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
    Eigen::VectorXd y = z - H_ * x_;                                     // 创新向量
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;                  // 创新矩阵
    double dist2 = y.transpose() * S.ldlt().solve(y);                   // 马氏距离平方
    return std::sqrt(dist2);
}

} // namespace tracking
} // namespace fastbev