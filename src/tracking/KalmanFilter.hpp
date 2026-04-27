#pragma once
#include <Eigen/Dense>
#include <cmath>

namespace fastbev {
namespace tracking {

class KalmanFilter {
public:
    static const int dim_x = 10;   // x,y,z,theta,l,w,h,dx,dy,dz
    static const int dim_z = 7;    // x,y,z,theta,l,w,h

    KalmanFilter();

    // 初始化状态（首次检测，7维测量）
    void init(const Eigen::VectorXd& bbox3d);

    // 预测（dt: 时间间隔秒）
    void predict(double dt);

    // 更新（测量值 z: 7维）
    void update(const Eigen::VectorXd& z);

    // 计算创新矩阵 S = H*P*H' + R（用于马氏距离关联）
    Eigen::MatrixXd computeInnovationMatrix() const;

    // 获取当前状态（前7维）
    Eigen::VectorXd getState() const { return x_.head(7); }

    // 获取速度（dx, dy, dz）
    Eigen::VectorXd getVelocity() const { return x_.tail(3); }

    // 获取完整状态向量（调试用）
    Eigen::VectorXd getFullState() const { return x_; }

    // 角度归一化到 [-π, π)
    static double normalizeAngle(double theta);

    // 计算马氏距离（给定测量值 z）
    double mahalanobisDistance(const Eigen::VectorXd& z) const;

private:
    Eigen::VectorXd x_;      // 状态 (10x1)
    Eigen::MatrixXd P_;      // 协方差 (10x10)
    Eigen::MatrixXd F_;      // 状态转移矩阵 (10x10) – 随 dt 变化
    Eigen::MatrixXd H_;      // 测量矩阵 (7x10)
    Eigen::MatrixXd Q_;      // 过程噪声 (10x10)
    Eigen::MatrixXd R_;      // 测量噪声 (7x7)
};

} // namespace tracking
} // namespace fastbev