#pragma once
/**
 * calibration.hpp — 相机标定参数加载
 *
 * 从 NuScenes meta.json 或通用 JSON 配置文件加载相机内参和外参。
 * 外参格式：lidar → camera 4×4 变换矩阵（行主序）
 */

#include <string>
#include <array>
#include <vector>
#include "camera_frame.hpp"

namespace fastbev {
namespace camera {

// ─── 外参（lidar → camera）────────────────────────────────────────────────

struct CameraExtrinsic {
    // lidar → camera 4×4 变换矩阵（行主序，列是 4 列）
    float mat[4][4] = {};
};

// ─── 完整标定参数 ──────────────────────────────────────────────────────────

struct CalibrationData {
    std::array<CameraIntrinsic,  NUM_CAMERAS> intrinsics;
    std::array<CameraExtrinsic,  NUM_CAMERAS> extrinsics;

    // 原始图像尺寸（输入到模型前的图像）
    int orig_width  = 1600;
    int orig_height =  900;
    // 模型输入尺寸
    int model_width  = 704;
    int model_height = 256;
};

// ─── 标定加载器 ────────────────────────────────────────────────────────────

class CalibrationLoader {
public:
    /**
     * 从帧目录下的 meta.json 中读取标定参数
     * （目前 meta.json 不含完整标定；此函数保留扩展接口，返回占位数据）
     */
    static bool load_from_meta_json(const std::string& frame_dir,
                                     CalibrationData& out);

    /**
     * 从通用标定 JSON 文件读取（格式见 docs/calibration_format.md）
     */
    static bool load_from_json(const std::string& json_path,
                                CalibrationData& out);

    /**
     * 从 NuScenes calibrated_sensor.json 直接解析
     * nuscenes_dir: NuScenes 根目录（含 v1.0-mini/calibrated_sensor.json）
     * sample_token: 目标样本 token
     */
    static bool load_from_nuscenes(const std::string& nuscenes_dir,
                                    const std::string& version,
                                    const std::string& sample_token,
                                    CalibrationData& out);
};

}  // namespace camera
}  // namespace fastbev
