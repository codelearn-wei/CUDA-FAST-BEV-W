#pragma once
/**
 * image_preprocessor.hpp — 图像预处理接口
 *
 * 将原始相机图像 + 标定参数转换为两个下游任务所需的输入格式：
 *   • BEV 推理    → camera::CameraFrame（含几何张量 valid_c_idx / x / y）
 *   • MapTR 推理  → frame_dir 下的 meta.json（已含 ego 位姿和标定信息）
 *
 * 三种数据来源：
 *   1. from_frame_dir() — 从 nuscenes_adapter.py 生成的帧目录加载（最常用）
 *   2. from_raw_input() — 从 RawImageInput + 标定动态计算几何张量（在线场景）
 *   3. from_camera()    — 从在线相机驱动获取（接口预留，待具体实现）
 *
 * 几何张量计算（BEV Pooling 所需）：
 *   对于每个相机，将体素网格 [200×200×4] 投影到特征图 [64×176]，
 *   输出 valid_c_idx / valid_x / valid_y 三个张量。
 *   算法与 tools/nuscenes_adapter.py::compute_geometry_tensors() 完全对应。
 */

#include "perception_types.hpp"
#include "../camera/camera_frame.hpp"

#include <string>
#include <memory>

namespace fastbev {
namespace perception {

// ─── 几何张量参数（与 Python nuscenes_adapter.py 保持一致） ────────────────

struct GeometryParams {
    // 体素网格大小（数量）
    int   nx          = 200;
    int   ny          = 200;
    int   nz          = 4;
    // 单个体素边长（米）
    float voxel_size_x = 0.5f;
    float voxel_size_y = 0.5f;
    float voxel_size_z = 1.5f;
    // 特征图尺寸（原始图像 256×704 下采样 4×）
    int   feat_height  = 64;
    int   feat_width   = 176;
    int   feat_stride  = 4;

    int num_voxels() const { return nx * ny * nz; }
};

// ─── 图像预处理器 ──────────────────────────────────────────────────────────

class ImagePreprocessor {
public:
    /**
     * 从帧目录加载完整帧（预计算张量 + 图像 + meta.json）。
     *
     * 适用场景：
     *   • 离线推理（nuscenes_adapter.py 已生成帧目录）
     *   • 快速验证
     *
     * @param frame_dir   帧目录路径（如 outputs/frames/frame_00000）
     * @param frame_id    帧 ID（调试用）
     * @param bev_frame   [out] BEV 推理所需的 CameraFrame（含几何张量）
     * @param map_input   [out] MapTR 推理所需的 RawImageInput（含 ego 位姿）
     * @return true 表示成功
     *
     * 注意：bev_frame.owns_tensors = true，需调用
     *       camera::FrameLoader::free_frame(bev_frame) 释放内存。
     */
    static bool from_frame_dir(const std::string& frame_dir,
                                uint64_t           frame_id,
                                camera::CameraFrame& bev_frame,
                                RawImageInput&       map_input);

    /**
     * 从 RawImageInput（含图像 + 标定）动态计算几何张量，构建 BEV 帧。
     *
     * 适用场景：
     *   • 在线推理（相机驱动直接提供图像 + 实时标定）
     *   • 标定更新频繁的场景
     *
     * 会在 frame_dir 下写入临时张量文件（若 frame_dir 非空），
     * 同时返回内存中的 bev_frame（owns_tensors = true）。
     *
     * @param input       含有 images_ptr, intrinsics, extrinsics, ego_pose 的输入
     * @param geo_params  体素网格参数（默认与训练时一致）
     * @param bev_frame   [out] 填充后的 BEV CameraFrame
     * @param voxel_origin 体素网格中心（lidar 坐标系，默认 {0,0,0}）
     * @return true 表示成功
     */
    static bool from_raw_input(const RawImageInput&  input,
                                const GeometryParams& geo_params,
                                camera::CameraFrame&  bev_frame,
                                const float           voxel_origin[3] = nullptr);

    /**
     * 【接口预留】从在线相机驱动获取图像并填充 RawImageInput。
     *
     * 目前仅声明接口，待与具体相机 SDK 集成后实现。
     * 典型实现需要：
     *   - 从 SDK 获取 6 路同步帧（或近似同步）
     *   - 将 YUV/BGR 格式转换为 RGB
     *   - 从相机配置文件或实时标定系统读取内外参
     *   - 从 GNSS/IMU 读取 ego 位姿
     *
     * @param camera_handle  相机驱动句柄（void* 保持平台无关）
     * @param calib_file     标定文件路径（JSON 格式）
     * @param output         [out] 填充后的 RawImageInput
     * @return true 表示成功
     */
    static bool from_camera(const void*    camera_handle,
                             const std::string& calib_file,
                             RawImageInput& output);

private:
    /**
     * 内部：计算单帧几何张量
     * extrinsics[i]  : 6 个相机的 lidar→camera 4×4 矩阵（行主序）
     * intrinsic_3x3  : 公共内参（3×3，图像坐标系）
     * voxel_origin   : 体素网格中心（lidar 坐标系）
     * geo            : 网格参数
     * out_cidx       : [6, num_voxels] float32（调用方分配）
     * out_x / out_y  : [6, num_voxels] int64  （调用方分配）
     */
    static void compute_geometry(
        const float     extrinsics[camera::NUM_CAMERAS][4][4],
        const float     intrinsic_3x3[3][3],
        const float     voxel_origin[3],
        const GeometryParams& geo,
        float*          out_cidx,
        int64_t*        out_x,
        int64_t*        out_y);

    /**
     * 内部：计算每相机独立内参的几何张量（NuScenes 原始数据路径）
     * intrinsics[i]  : 每个相机已缩放到模型输入坐标系（704×256）的 3×3 内参
     */
    static void compute_geometry_per_cam(
        const float     extrinsics[camera::NUM_CAMERAS][4][4],
        const float     intrinsics[camera::NUM_CAMERAS][3][3],
        const float     voxel_origin[3],
        const GeometryParams& geo,
        float*          out_cidx,
        int64_t*        out_x,
        int64_t*        out_y);

    /**
     * 回退路径：仅有图像 + meta.json（无 .tensor 文件）时
     * 从 meta.json 读取标定并在线计算几何张量。
     * meta.json 必须含有 lidar2cam_extrinsics / cam_intrinsics_raw / origin 字段。
     */
    static bool _from_images_and_meta(const std::string& frame_dir,
                                       uint64_t           frame_id,
                                       camera::CameraFrame& bev_frame,
                                       RawImageInput&       map_input);
};

}  // namespace perception
}  // namespace fastbev
