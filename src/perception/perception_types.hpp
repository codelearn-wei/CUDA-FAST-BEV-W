#pragma once
/**
 * perception_types.hpp — 联合感知公共数据类型
 *
 * 定义 BEV 检测 + MapTR 地图推理共享的输入/输出数据结构。
 *
 * 模块归属：src/perception/
 */

#include "../tracking/tracker.hpp"   // Detection, Track, EgoPose
#include "../camera/camera_frame.hpp" // CameraIntrinsic, NUM_CAMERAS

#include <array>
#include <vector>
#include <string>
#include <cstdint>

namespace fastbev {
namespace perception {

// ─── 相机外参（lidar → camera 4×4） ──────────────────────────────────────

struct Extrinsic4x4 {
    float mat[4][4] = {};
};

// ─── 单帧原始相机输入 ─────────────────────────────────────────────────────
//
// 封装一帧所有相机数据，作为图像预处理模块的统一输入。
// 同时适配两个下游任务：
//   • BEV 推理  — 需要 images + 几何张量（valid_c_idx / x / y）
//   • MapTR 推理 — 需要 images + intrinsics + extrinsics + ego_pose
//
// 来源可以是：
//   1. NuScenes 帧目录（pre-computed tensors）
//   2. 原始相机图像 + 标定文件（动态计算几何张量）
//   3. 在线相机驱动（接口预留，待实现）

struct RawImageInput {
    uint64_t frame_id  = 0;
    double   timestamp = 0.0;   // UNIX 时间（秒）

    // ── 图像数据（host 内存，RGB 格式，由 owner 管理生命周期） ──────────────
    // images_ptr[i] 指向 image_heights[i] × image_widths[i] × 3 字节缓冲
    std::array<unsigned char*, camera::NUM_CAMERAS> images_ptr{};
    std::array<int, camera::NUM_CAMERAS> image_widths{};
    std::array<int, camera::NUM_CAMERAS> image_heights{};

    // ── 相机标定（每帧固定或随实时标定更新） ──────────────────────────────
    std::array<camera::CameraIntrinsic, camera::NUM_CAMERAS> intrinsics{};
    std::array<Extrinsic4x4,            camera::NUM_CAMERAS> extrinsics{};

    // ── Ego 位姿（全局坐标系，来自 meta.json 或 GNSS/IMU） ─────────────────
    tracking::EgoPose ego_pose;

    // ── 来源标识（调试用） ─────────────────────────────────────────────────
    // "nuscenes" | "camera" | "file"
    std::string source_tag;
    // 若来自文件目录，此字段指向帧目录路径（MapTR 需要读取 meta.json）
    std::string frame_dir;

    bool has_images() const {
        for (int i = 0; i < camera::NUM_CAMERAS; ++i)
            if (!images_ptr[i]) return false;
        return true;
    }
};

// ─── 地图元素（MapTR 输出单元） ───────────────────────────────────────────

enum class MapElementType : int {
    Divider      = 0,   // 车道分隔线
    Boundary     = 1,   // 道路边界
    PedCrossing  = 2,   // 人行横道
};

struct MapElement {
    MapElementType type = MapElementType::Divider;
    float score        = 0.0f;
    // 折线点集（ego 坐标系，单位米）
    std::vector<std::array<float, 2>> points;
};

// ─── 单帧地图推理结果 ─────────────────────────────────────────────────────

struct MapResult {
    uint64_t frame_id   = 0;
    double   timestamp  = 0.0;

    std::vector<MapElement> elements;

    // 结果来源："model" | "gt" | "trajectory" | "none"
    std::string source;

    // 原始 JSON 字符串（原样透传，便于与 Python 可视化对接）
    std::string raw_json;

    bool is_valid() const { return !source.empty() && source != "none"; }
};

// ─── 联合感知结果（BEV + MapTR 合并输出） ─────────────────────────────────

struct JointPerceptionResult {
    uint64_t frame_id   = 0;
    double   timestamp  = 0.0;

    // ── BEV 检测 + 跟踪 ────────────────────────────────────────────────────
    std::vector<tracking::Detection> detections;  // 过滤后的原始检测
    std::vector<tracking::Track>     tracks;       // 带 track_id 的轨迹
    double bev_latency_ms = 0.0;

    // ── MapTR 地图 ─────────────────────────────────────────────────────────
    MapResult map;
    double map_latency_ms = 0.0;

    bool has_bev()  const { return !detections.empty() || !tracks.empty(); }
    bool has_map()  const { return map.is_valid(); }
};

}  // namespace perception
}  // namespace fastbev
