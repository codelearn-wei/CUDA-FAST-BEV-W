/**
 * camera_frame.cpp — FrameLoader 实现
 *
 * 从 nuscenes_adapter.py 生成的 frame_XXXXX 目录加载完整帧数据：
 *   - 6 路相机图像（stb_image，RGB 格式与模型输入一致）
 *   - 三个几何张量（valid_c_idx / x / y）
 *   - meta.json 中的时间戳
 *
 * 内存归属：
 *   - 图像由 stb_image_free() 释放
 *   - 张量数据由 new[] 分配；FrameLoader::free_frame() 负责 delete[]
 *   - 调用方通过 FrameLoader::free_frame(*frame) 或直接析构智能指针时清理
 */

#include "camera/camera_frame.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <thread>     // 并行图像加载
#include <array>

// stb_image 声明（实现由 src/common/stb_image_impl.cpp 提供）
#include <stb_image.h>

// nv::Tensor —— tensor 文件加载
#include "common/tensor.hpp"

namespace fastbev {
namespace camera {

// ─── 内部辅助 ──────────────────────────────────────────────────────────────────

static bool path_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

/**
 * 极简 JSON 数值提取（仅支持 "key": number 格式，避免引入第三方 JSON 库依赖）。
 */
static double json_get_double(const std::string& json,
                               const std::string& key,
                               double default_val = 0.0)
{
    std::string pat = "\"" + key + "\"";
    size_t kp = json.find(pat);
    if (kp == std::string::npos) return default_val;
    size_t vp = json.find_first_of("-0123456789", kp + pat.size());
    if (vp == std::string::npos) return default_val;
    try { return std::stod(json.substr(vp)); }
    catch (...) { return default_val; }
}

// ─── FrameLoader::load_from_dir ────────────────────────────────────────────────

std::unique_ptr<CameraFrame> FrameLoader::load_from_dir(
    const std::string& frame_dir,
    uint64_t           frame_id)
{
    // NuScenes 6 路相机文件名（与 nuscenes_adapter.py 输出对应）
    static const char* IMAGE_FILES[NUM_CAMERAS] = {
        "0-FRONT.jpg",
        "1-FRONT_RIGHT.jpg",
        "2-FRONT_LEFT.jpg",
        "3-BACK.jpg",
        "4-BACK_LEFT.jpg",
        "5-BACK_RIGHT.jpg",
    };

    auto frame = std::make_unique<CameraFrame>();
    frame->frame_id = frame_id;

    // ── 1. 并行加载 6 路相机图像 ──────────────────────────────────────────────
    // 每路图像在独立线程中解码，总耗时约等于单张最慢解码时间（而非 6 倍串行）
    struct ImgTask {
        std::string path;
        unsigned char* data = nullptr;
        int w = 0, h = 0;
    };
    std::array<ImgTask, NUM_CAMERAS> img_tasks;
    for (int i = 0; i < NUM_CAMERAS; ++i)
        img_tasks[i].path = frame_dir + "/" + IMAGE_FILES[i];

    std::array<std::thread, NUM_CAMERAS> img_threads;
    for (int i = 0; i < NUM_CAMERAS; ++i) {
        img_threads[i] = std::thread([&img_tasks, i]() {
            int w = 0, h = 0, ch = 0;
            img_tasks[i].data = stbi_load(
                img_tasks[i].path.c_str(), &w, &h, &ch, 3);
            img_tasks[i].w = w;
            img_tasks[i].h = h;
        });
    }
    for (int i = 0; i < NUM_CAMERAS; ++i)
        img_threads[i].join();

    bool any_loaded = false;
    for (int i = 0; i < NUM_CAMERAS; ++i) {
        frame->images[i] = img_tasks[i].data;
        if (img_tasks[i].data) {
            frame->image_width    = img_tasks[i].w;
            frame->image_height   = img_tasks[i].h;
            frame->image_channels = 3;
            any_loaded = true;
        } else {
            fprintf(stderr, "[FrameLoader] 无法加载图像: %s\n",
                    img_tasks[i].path.c_str());
        }
    }

    if (!any_loaded) {
        fprintf(stderr, "[FrameLoader] 目录内无任何可用图像: %s\n",
                frame_dir.c_str());
        return nullptr;
    }

    // ── 2. 加载几何张量 ────────────────────────────────────────────────────────
    // false = 加载到 CPU（host）内存；C++ 推理管线会在 update() 时上传 GPU
    auto t_cidx = nv::Tensor::load(frame_dir + "/valid_c_idx.tensor", false);
    auto t_x    = nv::Tensor::load(frame_dir + "/x.tensor",           false);
    auto t_y    = nv::Tensor::load(frame_dir + "/y.tensor",           false);

    if (t_cidx.empty() || t_x.empty() || t_y.empty()) {
        fprintf(stderr, "[FrameLoader] 无法加载几何张量: %s\n", frame_dir.c_str());
        free_frame(*frame);
        return nullptr;
    }

    // 从 nv::Tensor 拷贝到 heap 缓冲区（Tensor 本身为栈对象，离开作用域即析构）
    const size_t cidx_elems = t_cidx.numel;
    const size_t xy_elems   = t_x.numel;

    float*   cidx_buf = new float  [cidx_elems];
    int64_t* x_buf    = new int64_t[xy_elems];
    int64_t* y_buf    = new int64_t[xy_elems];

    memcpy(cidx_buf, t_cidx.ptr<float>(),   cidx_elems * sizeof(float));
    memcpy(x_buf,    t_x.ptr<int64_t>(),    xy_elems   * sizeof(int64_t));
    memcpy(y_buf,    t_y.ptr<int64_t>(),    xy_elems   * sizeof(int64_t));

    frame->valid_c_idx  = cidx_buf;
    frame->valid_x      = x_buf;
    frame->valid_y      = y_buf;
    frame->owns_tensors = true;
    frame->num_voxels   = static_cast<int>(cidx_elems / NUM_CAMERAS);

    // ── 3. 解析 meta.json（时间戳）────────────────────────────────────────────
    std::string meta_path = frame_dir + "/meta.json";
    if (path_exists(meta_path)) {
        std::ifstream ifs(meta_path);
        std::ostringstream oss;
        oss << ifs.rdbuf();
        std::string json_str = oss.str();

        // NuScenes timestamp 单位为微秒，转换为秒
        double ts_us = json_get_double(json_str, "timestamp", 0.0);
        frame->timestamp  = ts_us * 1e-6;
        frame->source_tag = frame_dir;
    }

    return frame;
}

// ─── FrameLoader::free_frame ───────────────────────────────────────────────────

void FrameLoader::free_frame(CameraFrame& frame)
{
    // 释放 stb_image 分配的图像内存
    for (int i = 0; i < NUM_CAMERAS; ++i) {
        if (frame.images[i]) {
            stbi_image_free(frame.images[i]);
            frame.images[i] = nullptr;
        }
    }

    // 释放几何张量缓冲区（仅当由 FrameLoader 分配时）
    if (frame.owns_tensors) {
        delete[] frame.valid_c_idx;  frame.valid_c_idx = nullptr;
        delete[] frame.valid_x;      frame.valid_x     = nullptr;
        delete[] frame.valid_y;      frame.valid_y     = nullptr;
        frame.owns_tensors = false;
    }
}

}  // namespace camera
}  // namespace fastbev
