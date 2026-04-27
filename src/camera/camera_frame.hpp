#pragma once
/**
 * camera_frame.hpp — 单帧多相机数据容器
 *
 * 封装一帧来自多路相机的图像和时间戳信息，
 * 以及对应的几何张量（BEV Pooling 所需）。
 */

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <array>
#include <functional>

namespace fastbev {
namespace camera {

// ─── 相机枚举（NuScenes 6 路顺序）────────────────────────────────────────

enum class CameraId : int {
    FRONT       = 0,
    FRONT_RIGHT = 1,
    FRONT_LEFT  = 2,
    BACK        = 3,
    BACK_LEFT   = 4,
    BACK_RIGHT  = 5,
    NUM_CAMERAS = 6,
};

static constexpr int NUM_CAMERAS = 6;

inline const char* camera_name(CameraId id) {
    static const char* names[] = {
        "FRONT", "FRONT_RIGHT", "FRONT_LEFT",
        "BACK",  "BACK_LEFT",   "BACK_RIGHT",
    };
    return names[static_cast<int>(id)];
}

// ─── 相机内参 ──────────────────────────────────────────────────────────────

struct CameraIntrinsic {
    float fx = 0.f, fy = 0.f;
    float cx = 0.f, cy = 0.f;
    int   width  = 0;
    int   height = 0;
};

// ─── 单帧多相机容器 ─────────────────────────────────────────────────────────

struct CameraFrame {
    uint64_t frame_id  = 0;
    double   timestamp = 0.0;   // UNIX 时间戳（秒）

    // 每路相机的原始图像指针和尺寸（外部管理内存）
    // images[i] 对应 CameraId(i)
    std::array<unsigned char*, NUM_CAMERAS> images{};
    int image_width  = 0;
    int image_height = 0;
    int image_channels = 3;

    // 标定：从文件或 NuScenes JSON 加载
    std::array<CameraIntrinsic, NUM_CAMERAS> intrinsics{};

    // 几何张量：由 nuscenes_adapter 或 camera 模块预计算
    // valid_c_idx: [NUM_CAMERAS, num_voxels] float32
    // valid_x/y:   [NUM_CAMERAS, num_voxels] int64
    float*   valid_c_idx = nullptr;  // 非 const：FrameLoader 可持有所有权
    int64_t* valid_x     = nullptr;
    int64_t* valid_y     = nullptr;
    int      num_voxels  = 160000;
    bool     owns_tensors = false;   // true = FrameLoader 分配，free_frame() 负责释放

    // 来自哪个数据源（调试用）
    std::string source_tag;

    bool is_valid() const {
        for (int i = 0; i < NUM_CAMERAS; ++i)
            if (!images[i]) return false;
        return valid_c_idx && valid_x && valid_y;
    }
};

// ─── 帧工厂函数声明 ─────────────────────────────────────────────────────────

/**
 * 从磁盘目录加载帧（由 nuscenes_adapter.py 生成的 frame_XXXXX 目录）
 *
 * 目录结构：
 *   frame_dir/
 *     0-FRONT.jpg ... 5-BACK_RIGHT.jpg
 *     valid_c_idx.tensor
 *     x.tensor
 *     y.tensor
 *     meta.json
 */
class FrameLoader {
public:
    /**
     * 加载一帧。返回 nullptr 表示失败。
     * 调用方负责 delete 或使用 unique_ptr。
     */
    static std::unique_ptr<CameraFrame> load_from_dir(
        const std::string& frame_dir,
        uint64_t           frame_id = 0);

    /**
     * 释放帧所持有的图像内存（stbi 分配的数据）
     */
    static void free_frame(CameraFrame& frame);
};

}  // namespace camera
}  // namespace fastbev
