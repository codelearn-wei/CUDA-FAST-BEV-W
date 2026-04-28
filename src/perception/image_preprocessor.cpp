/**
 * image_preprocessor.cpp — 图像预处理接口实现
 */

#include "image_preprocessor.hpp"
#include "../camera/camera_frame.hpp"
#include "../common/tensor.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

// stb_image 读取图像（实现由 src/common/stb_image_impl.cpp 提供）
#include <stb_image.h>

namespace fastbev {
namespace perception {

// ─── 内部辅助：极简 JSON 单值提取 ────────────────────────────────────────

static double _json_double(const std::string& s, const std::string& key,
                            double def = 0.0)
{
    std::string pat = "\"" + key + "\"";
    size_t kp = s.find(pat);
    if (kp == std::string::npos) return def;
    size_t vp = s.find_first_of("-0123456789.", kp + pat.size());
    if (vp == std::string::npos) return def;
    try { return std::stod(s.substr(vp)); }
    catch (...) { return def; }
}

// ─── 内部辅助：JSON 嵌套数组数值提取 ────────────────────────────────────
//
// 找到 key 对应的顶层 [ ... ] 范围，提取其中所有浮点数（按顺序）。
// 用于解析 lidar2cam_extrinsics / cam_intrinsics_raw / origin 这类嵌套数组。

static std::pair<size_t, size_t> _json_find_array(const std::string& s,
                                                    const std::string& key)
{
    std::string pat = "\"" + key + "\"";
    size_t kp = s.find(pat);
    if (kp == std::string::npos)
        return {std::string::npos, std::string::npos};
    size_t bp = s.find('[', kp + pat.size());
    if (bp == std::string::npos)
        return {std::string::npos, std::string::npos};
    // 匹配括号
    int depth = 1;
    size_t ep = bp + 1;
    while (ep < s.size() && depth > 0) {
        if      (s[ep] == '[') ++depth;
        else if (s[ep] == ']') --depth;
        ++ep;
    }
    return {bp, ep};
}

static std::vector<double> _extract_numbers(const std::string& s,
                                              size_t start, size_t end)
{
    std::vector<double> nums;
    size_t pos = start;
    while (pos < end) {
        // 跳过非数字字符（但 '-' 可能是负号）
        while (pos < end && s[pos] != '-' && (s[pos] < '0' || s[pos] > '9'))
            ++pos;
        if (pos >= end) break;
        char* endptr = nullptr;
        double val = std::strtod(s.c_str() + pos, &endptr);
        if (endptr == s.c_str() + pos) { ++pos; continue; }
        nums.push_back(val);
        pos = static_cast<size_t>(endptr - s.c_str());
    }
    return nums;
}

static bool _path_exists(const std::string& p)
{
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// ─── from_frame_dir ───────────────────────────────────────────────────────
//
// 优先加载预计算的 .tensor 文件（fast path）；
// 若 .tensor 文件不存在，从 meta.json 读取标定并在线计算几何张量（fallback）。

bool ImagePreprocessor::from_frame_dir(const std::string& frame_dir,
                                        uint64_t           frame_id,
                                        camera::CameraFrame& bev_frame,
                                        RawImageInput&       map_input)
{
    // ── Fast path：使用预计算 .tensor 文件 ──────────────────────────────────
    bool tensors_exist =
        _path_exists(frame_dir + "/valid_c_idx.tensor") &&
        _path_exists(frame_dir + "/x.tensor") &&
        _path_exists(frame_dir + "/y.tensor");

    if (tensors_exist) {
        auto loaded = camera::FrameLoader::load_from_dir(frame_dir, frame_id);
        if (loaded) {
            bev_frame = std::move(*loaded);
            // 填充 map_input
            map_input.frame_id   = bev_frame.frame_id;
            map_input.timestamp  = bev_frame.timestamp;
            map_input.source_tag = "file";
            map_input.frame_dir  = frame_dir;
            for (int i = 0; i < camera::NUM_CAMERAS; ++i) {
                map_input.images_ptr[i]    = bev_frame.images[i];
                map_input.image_widths[i]  = bev_frame.image_width;
                map_input.image_heights[i] = bev_frame.image_height;
            }
            // 读取 ego 位姿
            std::string meta_path = frame_dir + "/meta.json";
            if (_path_exists(meta_path)) {
                std::ifstream mf(meta_path);
                if (mf) {
                    std::ostringstream ss; ss << mf.rdbuf();
                    std::string meta = ss.str();
                    std::pair<size_t,size_t> ego_range =
                        _json_find_array(meta, "ego_translation_global");
                    if (ego_range.first != std::string::npos) {
                        auto nums = _extract_numbers(meta,
                                        ego_range.first, ego_range.second);
                        if (nums.size() >= 2) {
                            map_input.ego_pose.x = nums[0];
                            map_input.ego_pose.y = nums[1];
                        }
                    }
                    map_input.ego_pose.yaw   = _json_double(meta, "ego_yaw_global", 0.0);
                    map_input.ego_pose.valid = true;
                }
            }
            return true;
        }
    }

    // ── Fallback：仅有图像 + meta.json，在线计算几何张量 ─────────────────────
    // 这条路径适用于"只有 6 张图 + meta.json"的场景，无需预运行 nuscenes_adapter.py
    return _from_images_and_meta(frame_dir, frame_id, bev_frame, map_input);
}

// ─── compute_geometry（内部，纯 C++ 实现） ────────────────────────────────
//
// 对应 Python tools/nuscenes_adapter.py::compute_geometry_tensors()
// 体素网格原点 = 网格中心（由 voxel_origin 传入）

void ImagePreprocessor::compute_geometry(
    const float     extrinsics[camera::NUM_CAMERAS][4][4],
    const float     intrinsic_3x3[3][3],
    const float     voxel_origin[3],
    const GeometryParams& g,
    float*          out_cidx,
    int64_t*        out_x,
    int64_t*        out_y)
{
    const int NX = g.nx, NY = g.ny, NZ = g.nz;
    const int NVOX = NX * NY * NZ;

    // 特征图尺度内参：除以 feat_stride
    const float fx = intrinsic_3x3[0][0] / g.feat_stride;
    const float fy = intrinsic_3x3[1][1] / g.feat_stride;
    const float cx = intrinsic_3x3[0][2] / g.feat_stride;
    const float cy = intrinsic_3x3[1][2] / g.feat_stride;

    // 体素左下角（与 Python get_voxel_points 中的 new_origin 一致）
    const float ox = voxel_origin[0] - NX * 0.5f * g.voxel_size_x;
    const float oy = voxel_origin[1] - NY * 0.5f * g.voxel_size_y;
    const float oz = voxel_origin[2] - NZ * 0.5f * g.voxel_size_z;

    // 生成体素点（x 变化最快，与 Python indexing 一致）
    // pts[v] = {px, py, pz}
    struct Pt3 { float x, y, z; };
    std::vector<Pt3> pts(NVOX);
    int idx = 0;
    for (int ix = 0; ix < NX; ++ix)
        for (int iy = 0; iy < NY; ++iy)
            for (int iz = 0; iz < NZ; ++iz, ++idx) {
                pts[idx].x = ox + ix * g.voxel_size_x;
                pts[idx].y = oy + iy * g.voxel_size_y;
                pts[idx].z = oz + iz * g.voxel_size_z;
            }

    // 零初始化输出
    memset(out_cidx, 0, sizeof(float)   * camera::NUM_CAMERAS * NVOX);
    memset(out_x,    0, sizeof(int64_t) * camera::NUM_CAMERAS * NVOX);
    memset(out_y,    0, sizeof(int64_t) * camera::NUM_CAMERAS * NVOX);

    for (int cam = 0; cam < camera::NUM_CAMERAS; ++cam) {
        const float (*E)[4] = extrinsics[cam];  // 4×4 行主序
        // 构建投影矩阵 P = K_feat @ E[:3, :]
        // P[row][col] = K_feat[row] · E[:3, col]
        // K_feat 为对角矩阵，简化：
        //   P[0][j] = fx * E[0][j] + cx * E[2][j]  (对于 j=0..3)
        //   P[1][j] = fy * E[1][j] + cy * E[2][j]
        //   P[2][j] = E[2][j]
        float P[3][4];
        for (int j = 0; j < 4; ++j) {
            P[0][j] = fx * E[0][j] + cx * E[2][j];
            P[1][j] = fy * E[1][j] + cy * E[2][j];
            P[2][j] = E[2][j];
        }

        float*   cidx_row = out_cidx + cam * NVOX;
        int64_t* x_row    = out_x    + cam * NVOX;
        int64_t* y_row    = out_y    + cam * NVOX;

        for (int v = 0; v < NVOX; ++v) {
            const float px = pts[v].x, py = pts[v].y, pz = pts[v].z;
            // 齐次投影：P @ [px, py, pz, 1]
            const float u2 = P[0][0]*px + P[0][1]*py + P[0][2]*pz + P[0][3];
            const float v2 = P[1][0]*px + P[1][1]*py + P[1][2]*pz + P[1][3];
            const float w2 = P[2][0]*px + P[2][1]*py + P[2][2]*pz + P[2][3];

            if (w2 <= 1e-6f) continue;  // 在相机后方或零深度

            const float inv_w = 1.0f / w2;
            const int64_t xi = static_cast<int64_t>(std::roundf(u2 * inv_w));
            const int64_t yi = static_cast<int64_t>(std::roundf(v2 * inv_w));

            if (xi < 0 || xi >= g.feat_width ||
                yi < 0 || yi >= g.feat_height)
                continue;

            cidx_row[v] = 1.0f;
            x_row[v]    = xi;
            y_row[v]    = yi;
        }
    }
}

// ─── from_raw_input ───────────────────────────────────────────────────────

bool ImagePreprocessor::from_raw_input(const RawImageInput&  input,
                                        const GeometryParams& geo,
                                        camera::CameraFrame&  bev_frame,
                                        const float           voxel_origin_in[3])
{
    if (!input.has_images()) {
        fprintf(stderr, "[ImagePreprocessor] from_raw_input: 图像指针为空\n");
        return false;
    }

    // 默认体素网格中心（与训练时 origin=[0,0,0] 一致）
    const float zero_origin[3] = {0.f, 0.f, 0.f};
    const float* vox_origin = voxel_origin_in ? voxel_origin_in : zero_origin;

    // 构造公共内参（取 cam0 的 fx/fy/cx/cy）
    // 注：FastBEV 使用公共内参（各相机缩放到同一图像尺寸后共享一个 K）
    const auto& K0 = input.intrinsics[0];
    const float K[3][3] = {
        { K0.fx, 0.f,   K0.cx },
        { 0.f,   K0.fy, K0.cy },
        { 0.f,   0.f,   1.f   },
    };

    // 从 Extrinsic4x4 转为 float[4][4] 数组
    float ext[camera::NUM_CAMERAS][4][4];
    for (int i = 0; i < camera::NUM_CAMERAS; ++i)
        memcpy(ext[i], input.extrinsics[i].mat, sizeof(float) * 16);

    const int NVOX = geo.num_voxels();

    float*   cidx = new float  [camera::NUM_CAMERAS * NVOX]();
    int64_t* xbuf = new int64_t[camera::NUM_CAMERAS * NVOX]();
    int64_t* ybuf = new int64_t[camera::NUM_CAMERAS * NVOX]();

    compute_geometry(ext, K, vox_origin, geo, cidx, xbuf, ybuf);

    // 填充 CameraFrame
    bev_frame.frame_id      = input.frame_id;
    bev_frame.timestamp     = input.timestamp;
    bev_frame.image_width   = input.image_widths[0];
    bev_frame.image_height  = input.image_heights[0];
    bev_frame.image_channels = 3;
    bev_frame.source_tag    = input.source_tag;
    bev_frame.num_voxels    = NVOX;
    bev_frame.owns_tensors  = true;
    bev_frame.valid_c_idx   = cidx;
    bev_frame.valid_x       = xbuf;
    bev_frame.valid_y       = ybuf;

    for (int i = 0; i < camera::NUM_CAMERAS; ++i)
        bev_frame.images[i] = input.images_ptr[i];

    return true;
}

// ─── compute_geometry_per_cam ─────────────────────────────────────────────
//
// 与 compute_geometry 相同，但每个相机使用各自独立的内参矩阵。
// 对应 Python tools/nuscenes_adapter.py::compute_geometry_tensors_per_cam()。

void ImagePreprocessor::compute_geometry_per_cam(
    const float     extrinsics[camera::NUM_CAMERAS][4][4],
    const float     intrinsics[camera::NUM_CAMERAS][3][3],
    const float     voxel_origin[3],
    const GeometryParams& g,
    float*          out_cidx,
    int64_t*        out_x,
    int64_t*        out_y)
{
    const int NX = g.nx, NY = g.ny, NZ = g.nz;
    const int NVOX = NX * NY * NZ;

    // 体素左下角（与 Python get_voxel_points 一致）
    const float ox = voxel_origin[0] - NX * 0.5f * g.voxel_size_x;
    const float oy = voxel_origin[1] - NY * 0.5f * g.voxel_size_y;
    const float oz = voxel_origin[2] - NZ * 0.5f * g.voxel_size_z;

    struct Pt3 { float x, y, z; };
    std::vector<Pt3> pts(NVOX);
    int idx = 0;
    for (int ix = 0; ix < NX; ++ix)
        for (int iy = 0; iy < NY; ++iy)
            for (int iz = 0; iz < NZ; ++iz, ++idx) {
                pts[idx].x = ox + ix * g.voxel_size_x;
                pts[idx].y = oy + iy * g.voxel_size_y;
                pts[idx].z = oz + iz * g.voxel_size_z;
            }

    memset(out_cidx, 0, sizeof(float)   * camera::NUM_CAMERAS * NVOX);
    memset(out_x,    0, sizeof(int64_t) * camera::NUM_CAMERAS * NVOX);
    memset(out_y,    0, sizeof(int64_t) * camera::NUM_CAMERAS * NVOX);

    for (int cam = 0; cam < camera::NUM_CAMERAS; ++cam) {
        const float (*E)[4] = extrinsics[cam];
        const float (*K)[3] = intrinsics[cam];

        // 特征图尺度内参
        const float fx = K[0][0] / g.feat_stride;
        const float fy = K[1][1] / g.feat_stride;
        const float cx = K[0][2] / g.feat_stride;
        const float cy = K[1][2] / g.feat_stride;

        // 投影矩阵 P = K_feat @ E[:3,:]
        float P[3][4];
        for (int j = 0; j < 4; ++j) {
            P[0][j] = fx * E[0][j] + cx * E[2][j];
            P[1][j] = fy * E[1][j] + cy * E[2][j];
            P[2][j] = E[2][j];
        }

        float*   cidx_row = out_cidx + cam * NVOX;
        int64_t* x_row    = out_x    + cam * NVOX;
        int64_t* y_row    = out_y    + cam * NVOX;

        for (int v = 0; v < NVOX; ++v) {
            const float px = pts[v].x, py = pts[v].y, pz = pts[v].z;
            const float u2 = P[0][0]*px + P[0][1]*py + P[0][2]*pz + P[0][3];
            const float v2 = P[1][0]*px + P[1][1]*py + P[1][2]*pz + P[1][3];
            const float w2 = P[2][0]*px + P[2][1]*py + P[2][2]*pz + P[2][3];

            if (w2 <= 1e-6f) continue;

            const float inv_w = 1.0f / w2;
            const int64_t xi = static_cast<int64_t>(std::roundf(u2 * inv_w));
            const int64_t yi = static_cast<int64_t>(std::roundf(v2 * inv_w));

            if (xi < 0 || xi >= g.feat_width || yi < 0 || yi >= g.feat_height)
                continue;

            cidx_row[v] = 1.0f;
            x_row[v]    = xi;
            y_row[v]    = yi;
        }
    }
}

// ─── _from_images_and_meta ────────────────────────────────────────────────
//
// 回退路径：仅有 6 张图像 + meta.json（无预计算 .tensor 文件）时，
// 从 meta.json 读取标定并在线计算几何张量。
//
// meta.json 格式（由 nuscenes_adapter.py 写入）：
//   lidar2cam_extrinsics: [6][4][4]（lidar→camera，直接可用）
//   cam_intrinsics_raw:   [6][3][3]（原始 1600×900 分辨率内参，需缩放）
//   origin:               [x, y, z]（体素网格中心）
//   timestamp:            number
//   ego_translation_global: [x, y, z]
//   ego_yaw_global:       number（弧度）
//
// 内参缩放公式（匹配 nuscenes_adapter.py::prepare_frame_data）：
//   ORIG_W=1600, ORIG_H=900, MODEL_W=704, MODEL_H=256
//   resize_lim  = MODEL_W / ORIG_W = 0.44
//   resized_h   = int(ORIG_H * resize_lim) = 396
//   crop_y_off  = (resized_h - MODEL_H) / 2 = 70
//   K_scaled[0] *= resize_lim    (fx, cx)
//   K_scaled[1] *= resize_lim    (fy, cy)
//   K_scaled[1][2] -= crop_y_off (cy)

bool ImagePreprocessor::_from_images_and_meta(const std::string& frame_dir,
                                               uint64_t           frame_id,
                                               camera::CameraFrame& bev_frame,
                                               RawImageInput&       map_input)
{
    // NuScenes 6 路图像文件名（与 FrameLoader 一致）
    static const char* IMAGE_FILES[camera::NUM_CAMERAS] = {
        "0-FRONT.jpg",
        "1-FRONT_RIGHT.jpg",
        "2-FRONT_LEFT.jpg",
        "3-BACK.jpg",
        "4-BACK_LEFT.jpg",
        "5-BACK_RIGHT.jpg",
    };

    // ── 读取 meta.json ────────────────────────────────────────────────────
    std::string meta_path = frame_dir + "/meta.json";
    if (!_path_exists(meta_path)) {
        fprintf(stderr,
                "[ImagePreprocessor] 无 .tensor 文件且找不到 meta.json: %s\n",
                meta_path.c_str());
        return false;
    }

    std::ifstream mf(meta_path);
    if (!mf) {
        fprintf(stderr, "[ImagePreprocessor] meta.json 读取失败: %s\n",
                meta_path.c_str());
        return false;
    }
    std::ostringstream ss; ss << mf.rdbuf();
    std::string meta = ss.str();

    // ── 解析 lidar2cam_extrinsics：[6][4][4] → 96 个数 ────────────────────
    float ext_raw[camera::NUM_CAMERAS][4][4] = {};
    {
        std::pair<size_t,size_t> rng =
            _json_find_array(meta, "lidar2cam_extrinsics");
        if (rng.first == std::string::npos) {
            fprintf(stderr, "[ImagePreprocessor] meta.json 缺少 lidar2cam_extrinsics\n");
            return false;
        }
        auto nums = _extract_numbers(meta, rng.first, rng.second);
        if (static_cast<int>(nums.size()) < camera::NUM_CAMERAS * 16) {
            fprintf(stderr,
                    "[ImagePreprocessor] lidar2cam_extrinsics 数值不足: "
                    "期望 %d，实际 %zu\n",
                    camera::NUM_CAMERAS * 16, nums.size());
            return false;
        }
        for (int c = 0; c < camera::NUM_CAMERAS; ++c)
            for (int r = 0; r < 4; ++r)
                for (int col = 0; col < 4; ++col)
                    ext_raw[c][r][col] =
                        static_cast<float>(nums[c * 16 + r * 4 + col]);
    }

    // ── 解析 cam_intrinsics_raw：[6][3][3] → 54 个数 ──────────────────────
    float K_raw[camera::NUM_CAMERAS][3][3] = {};
    {
        std::pair<size_t,size_t> rng =
            _json_find_array(meta, "cam_intrinsics_raw");
        if (rng.first == std::string::npos) {
            fprintf(stderr, "[ImagePreprocessor] meta.json 缺少 cam_intrinsics_raw\n");
            return false;
        }
        auto nums = _extract_numbers(meta, rng.first, rng.second);
        if (static_cast<int>(nums.size()) < camera::NUM_CAMERAS * 9) {
            fprintf(stderr,
                    "[ImagePreprocessor] cam_intrinsics_raw 数值不足: "
                    "期望 %d，实际 %zu\n",
                    camera::NUM_CAMERAS * 9, nums.size());
            return false;
        }
        for (int c = 0; c < camera::NUM_CAMERAS; ++c)
            for (int r = 0; r < 3; ++r)
                for (int col = 0; col < 3; ++col)
                    K_raw[c][r][col] =
                        static_cast<float>(nums[c * 9 + r * 3 + col]);
    }

    // ── 解析 origin：[x, y, z] ─────────────────────────────────────────────
    float vox_origin[3] = {0.f, 0.f, -1.f};  // NuScenes 默认
    {
        std::pair<size_t,size_t> rng = _json_find_array(meta, "origin");
        if (rng.first != std::string::npos) {
            auto nums = _extract_numbers(meta, rng.first, rng.second);
            if (nums.size() >= 3) {
                vox_origin[0] = static_cast<float>(nums[0]);
                vox_origin[1] = static_cast<float>(nums[1]);
                vox_origin[2] = static_cast<float>(nums[2]);
            }
        }
    }

    // ── 内参缩放：1600×900 → 704×256（匹配 nuscenes_adapter.py） ──────────
    // 仅对原始分辨率 1600×900 有效；若其他分辨率请更新此处常量
    constexpr float ORIG_W    = 1600.f;
    constexpr float ORIG_H    = 900.f;
    constexpr float MODEL_W   = 704.f;
    constexpr float MODEL_H   = 256.f;
    const float     resize_lim = MODEL_W / ORIG_W;          // 0.44
    const int       resized_h  = static_cast<int>(ORIG_H * resize_lim); // 396
    const int       crop_y_off = (resized_h - static_cast<int>(MODEL_H)) / 2; // 70

    float K_scaled[camera::NUM_CAMERAS][3][3];
    for (int c = 0; c < camera::NUM_CAMERAS; ++c) {
        memcpy(K_scaled[c], K_raw[c], sizeof(float) * 9);
        // row 0: fx, _, cx
        K_scaled[c][0][0] *= resize_lim;
        K_scaled[c][0][2] *= resize_lim;
        // row 1: _, fy, cy
        K_scaled[c][1][1] *= resize_lim;
        K_scaled[c][1][2] = K_scaled[c][1][2] * resize_lim
                             - static_cast<float>(crop_y_off);
        // row 2: 0, 0, 1（不变）
    }

    // ── 加载 6 路图像 ─────────────────────────────────────────────────────
    unsigned char* imgs[camera::NUM_CAMERAS] = {};
    int img_w = -1, img_h = -1;
    for (int i = 0; i < camera::NUM_CAMERAS; ++i) {
        std::string img_path = frame_dir + "/" + IMAGE_FILES[i];
        int w = 0, h = 0, ch = 0;
        imgs[i] = stbi_load(img_path.c_str(), &w, &h, &ch, 3);
        if (!imgs[i]) {
            fprintf(stderr,
                    "[ImagePreprocessor] 图像加载失败: %s\n",
                    img_path.c_str());
            // 释放已加载的图像
            for (int j = 0; j < i; ++j)
                if (imgs[j]) stbi_image_free(imgs[j]);
            return false;
        }
        if (img_w < 0) { img_w = w; img_h = h; }
    }

    // ── 计算每相机几何张量 ────────────────────────────────────────────────
    GeometryParams geo;  // 使用默认参数（200×200×4 体素，feat 64×176）
    const int NVOX = geo.num_voxels();

    float*   cidx = new float  [camera::NUM_CAMERAS * NVOX]();
    int64_t* xbuf = new int64_t[camera::NUM_CAMERAS * NVOX]();
    int64_t* ybuf = new int64_t[camera::NUM_CAMERAS * NVOX]();

    compute_geometry_per_cam(ext_raw, K_scaled, vox_origin, geo, cidx, xbuf, ybuf);

    // ── 填充 bev_frame ────────────────────────────────────────────────────
    bev_frame.frame_id       = frame_id;
    bev_frame.timestamp      = static_cast<uint64_t>(
        _json_double(meta, "timestamp", 0.0));
    bev_frame.image_width    = img_w;
    bev_frame.image_height   = img_h;
    bev_frame.image_channels = 3;
    bev_frame.source_tag     = "meta_fallback";
    bev_frame.num_voxels     = NVOX;
    bev_frame.owns_tensors   = true;
    bev_frame.valid_c_idx    = cidx;
    bev_frame.valid_x        = xbuf;
    bev_frame.valid_y        = ybuf;
    for (int i = 0; i < camera::NUM_CAMERAS; ++i)
        bev_frame.images[i] = imgs[i];

    // ── 填充 map_input ────────────────────────────────────────────────────
    map_input.frame_id   = frame_id;
    map_input.timestamp  = bev_frame.timestamp;
    map_input.source_tag = "meta_fallback";
    map_input.frame_dir  = frame_dir;
    for (int i = 0; i < camera::NUM_CAMERAS; ++i) {
        map_input.images_ptr[i]    = imgs[i];
        map_input.image_widths[i]  = img_w;
        map_input.image_heights[i] = img_h;
    }
    // ego 位姿
    {
        std::pair<size_t,size_t> ego_rng =
            _json_find_array(meta, "ego_translation_global");
        if (ego_rng.first != std::string::npos) {
            auto nums = _extract_numbers(meta,
                            ego_rng.first, ego_rng.second);
            if (nums.size() >= 2) {
                map_input.ego_pose.x = nums[0];
                map_input.ego_pose.y = nums[1];
            }
        }
        map_input.ego_pose.yaw   = _json_double(meta, "ego_yaw_global", 0.0);
        map_input.ego_pose.valid = true;
    }

    return true;
}

// ─── from_camera（接口预留，待与具体相机 SDK 集成后实现） ──────────────────

bool ImagePreprocessor::from_camera(const void* /*camera_handle*/,
                                     const std::string& /*calib_file*/,
                                     RawImageInput& /*output*/)
{
    fprintf(stderr,
            "[ImagePreprocessor] from_camera: 接口预留，尚未实现。\n"
            "  请在与具体相机 SDK 集成后补全此函数：\n"
            "  1. 从 SDK 获取 6 路同步帧（或近似同步）\n"
            "  2. 将 YUV/BGR 转换为 RGB\n"
            "  3. 从 calib_file 或实时标定系统加载内外参\n"
            "  4. 从 GNSS/IMU 读取 ego 位姿\n");
    return false;
}

}  // namespace perception
}  // namespace fastbev
