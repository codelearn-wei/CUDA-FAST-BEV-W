#pragma once
#include <Eigen/Dense>
#include <cmath>

namespace fastbev {
namespace tracking {

class KalmanFilter {
public:
    static const int dim_x = 10;   // x,y,z,theta,l,w,h,dx,dy,dz
    static const int dim_z = 9;    // x,y,z,theta,l,w,h

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

    /**
     * 自车运动补偿：将 KF 状态从上一帧 lidar-local 坐标系变换到当前帧 lidar-local 坐标系。
     * 参数为上一帧和当前帧的全局位姿 (x, y, yaw)。
     * 变换包含：位置旋转+平移、航向角偏置、速度旋转、协方差旋转。
     */
    void applyEgoTransform(double prev_ego_x, double prev_ego_y, double prev_ego_yaw,
                           double curr_ego_x, double curr_ego_y, double curr_ego_yaw);

private:
    Eigen::VectorXd x_;      // 状态 (10x1)
    Eigen::MatrixXd P_;      // 协方差 (10x10)
    Eigen::MatrixXd F_;      // 状态转移矩阵 (10x10) – 随 dt 变化
    Eigen::MatrixXd H_;      // 测量矩阵 (7x10)
    Eigen::MatrixXd Q_;      // 过程噪声 (10x10)
    Eigen::MatrixXd R_;      // 测量噪声 (7x7)

    // 用于静止抑制的成员变量（每个KalmanFilter实例独立记录）
    double last_x_ = 0.0;
    double last_y_ = 0.0;
};

} // namespace tracking
} // namespace fastbev