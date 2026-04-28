/**
 * joint_inference.cpp — BEV + MapTR 联合感知主程序
 *
 * 功能：
 *   1. 图像预处理（从帧目录 / 原始图像）
 *   2. BEV 3D 目标检测 + 多目标跟踪（C++ / TensorRT）
 *   3. MapTR HD 地图推理（通过 Python subprocess）
 *   4. 联合结果保存（JSON）
 *
 * 用法：
 *   ./build/joint_inference <frames_dir> <model> [precision] [options]
 *
 * 必选参数：
 *   frames_dir    帧目录（frame_XXXXX 子目录，由 nuscenes_adapter.py 生成）
 *   model         模型名称（resnet18 / resnet18int8）
 *
 * 可选参数：
 *   precision         fp16 / int8（默认 fp16）
 *   --score-thr F     置信度阈值（默认 0.35）
 *   --map-mode S      地图模式：auto / gt / model / trajectory（默认 auto）
 *   --nuscenes-dir S  NuScenes 数据目录（GT 模式需要，默认 data/nuscenes）
 *   --skip-map        仅运行 BEV，跳过 MapTR
 *   --skip-bev        仅运行 MapTR，跳过 BEV（结合 --map-mode 使用）
 *   --save-json       将联合结果写入 <frame_dir>/joint_result.json
 *   --batch           批量处理所有帧（默认为批量模式）
 *   --no-warmup       跳过 TRT warmup
 *   --verbose         打印详细调试信息
 *
 * 示例：
 *   # 批量推理（BEV + GT 地图）
 *   ./build/joint_inference outputs/frames resnet18int8 int8 \
 *       --score-thr 0.3 --map-mode gt --save-json
 *
 *   # 仅 BEV（不运行 MapTR）
 *   ./build/joint_inference outputs/frames resnet18int8 int8 \
 *       --score-thr 0.3 --skip-map
 */

#include "perception/perception_types.hpp"
#include "perception/image_preprocessor.hpp"
#include "perception/map_runner.hpp"
#include "pipeline/perception_pipeline.hpp"
#include "camera/camera_frame.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <dirent.h>
#include <sys/stat.h>

using namespace fastbev;
using namespace fastbev::perception;
using namespace fastbev::pipeline;

// ─── 辅助函数 ──────────────────────────────────────────────────────────────

static bool str_eq(const char* a, const char* b) { return strcmp(a, b) == 0; }

static bool path_exists(const std::string& p)
{
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// 收集 frames_dir 下所有 frame_XXXXX 目录（按名称排序）
static std::vector<std::string> collect_frame_dirs(const std::string& base)
{
    std::vector<std::string> dirs;
    DIR* d = opendir(base.c_str());
    if (!d) return dirs;
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        std::string name(entry->d_name);
        if (name.rfind("frame_", 0) == 0) {
            std::string full = base + "/" + name;
            struct stat st;
            if (stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                dirs.push_back(full);
        }
    }
    closedir(d);
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

// 将 JointPerceptionResult 序列化为 JSON 字符串
static std::string result_to_json(const JointPerceptionResult& r)
{
    std::ostringstream o;
    o << "{\n";
    o << "  \"frame_id\": " << r.frame_id << ",\n";
    o << "  \"timestamp\": " << r.timestamp << ",\n";
    o << "  \"bev_latency_ms\": " << r.bev_latency_ms << ",\n";
    o << "  \"map_latency_ms\": " << r.map_latency_ms << ",\n";

    // tracks
    o << "  \"tracks\": [\n";
    for (size_t i = 0; i < r.tracks.size(); ++i) {
        const auto& t = r.tracks[i];
        o << "    {\"track_id\": " << t.track_id
          << ", \"position\": [" << t.x << ", " << t.y << ", " << t.z << "]"
          << ", \"size\": [" << t.w << ", " << t.l << ", " << t.h << "]"
          << ", \"yaw\": " << t.yaw
          << ", \"velocity\": [" << t.vx << ", " << t.vy << "]"
          << ", \"score\": " << t.score
          << ", \"class_id\": " << t.class_id
          << "}";
        if (i + 1 < r.tracks.size()) o << ",";
        o << "\n";
    }
    o << "  ],\n";

    // detections
    o << "  \"detections\": [\n";
    for (size_t i = 0; i < r.detections.size(); ++i) {
        const auto& d = r.detections[i];
        o << "    {\"x\":"    << d.x
          << ",\"y\":"        << d.y
          << ",\"z\":"        << d.z
          << ",\"yaw\":"      << d.yaw
          << ",\"l\":"        << d.l
          << ",\"w\":"        << d.w
          << ",\"h\":"        << d.h
          << ",\"score\":"    << d.score
          << ",\"class_id\":" << d.class_id
          << "}";
        if (i + 1 < r.detections.size()) o << ",";
        o << "\n";
    }
    o << "  ],\n";

    // map（直接嵌入原始 JSON）
    if (r.map.is_valid() && !r.map.raw_json.empty()) {
        o << "  \"map\": " << r.map.raw_json << "\n";
    } else {
        o << "  \"map\": {\"source\": \"none\", \"elements\": []}\n";
    }

    o << "}\n";
    return o.str();
}

// ─── 用法说明 ──────────────────────────────────────────────────────────────

static void print_usage(const char* prog)
{
    printf(
        "用法: %s <frames_dir> <model> [precision] [options]\n"
        "\n"
        "必选参数:\n"
        "  frames_dir           帧目录（frame_XXXXX 子目录）\n"
        "  model                模型名（resnet18 / resnet18int8）\n"
        "\n"
        "可选参数:\n"
        "  precision            fp16 / int8（默认 fp16）\n"
        "  --score-thr F        置信度阈值（默认 0.35）\n"
        "  --map-mode S         auto|gt|model（默认 auto）\n"
        "  --ckpt PATH          MapTR checkpoint 路径\n"
        "                       （默认 model/maptr/maptr_nano_r18_110e.pth）\n"
        "  --nuscenes-dir S     NuScenes 数据目录（默认 data/nuscenes）\n"
        "  --skip-map           仅运行 BEV，跳过 MapTR\n"
        "  --skip-bev           仅运行 MapTR，跳过 BEV\n"
        "  --save-json          保存联合结果到 <frame_dir>/joint_result.json\n"
        "  --no-warmup          跳过 TRT warmup\n"
        "  --verbose            打印详细信息\n"
        "\n"
        "说明（并行模式）:\n"
        "  BEV 在主线程逐帧推理；MapTR 在后台线程同时处理所有帧。\n"
        "  BEV 每帧读取 map_result.json（若已写入则用当帧，否则复用最近一帧结果）。\n"
        "  MapTR 不阻塞 BEV，二者互不等待。\n"
        "\n"
        "示例:\n"
        "  # 使用 MapTR 神经网络推理（推荐）\n"
        "  %s outputs/frames resnet18int8 int8 \\\n"
        "      --score-thr 0.3 --map-mode model --save-json\n"
        "\n"
        "  # 使用 NuScenes GT 地图（无需 checkpoint）\n"
        "  %s outputs/frames resnet18int8 int8 \\\n"
        "      --score-thr 0.3 --map-mode gt --save-json\n"
        "\n"
        "  # 仅 BEV 推理（不运行 MapTR）\n"
        "  %s outputs/frames resnet18int8 int8 --skip-map\n",
        prog, prog, prog, prog);
}

// ─── main ──────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    // ── 解析必选参数 ──────────────────────────────────────────────────────
    std::string frames_dir = argv[1];
    std::string model_name = argv[2];
    std::string precision  = "fp16";

    // ── 解析可选参数 ──────────────────────────────────────────────────────
    float       score_thr    = 0.35f;
    std::string map_mode     = "auto";
    std::string nuscenes_dir = "data/nuscenes";
    std::string ckpt_path    = "model/maptr/maptr_nano_r18_110e.pth";
    bool        skip_map     = false;
    bool        skip_bev     = false;
    bool        save_json    = false;
    bool        no_warmup    = false;
    bool        verbose      = false;

    for (int i = 3; i < argc; ++i) {
        if (!argv[i]) continue;
        if (str_eq(argv[i], "--score-thr") && i + 1 < argc) {
            score_thr = static_cast<float>(atof(argv[++i]));
        } else if (str_eq(argv[i], "--map-mode") && i + 1 < argc) {
            map_mode = argv[++i];
        } else if (str_eq(argv[i], "--ckpt") && i + 1 < argc) {
            ckpt_path = argv[++i];
        } else if (str_eq(argv[i], "--nuscenes-dir") && i + 1 < argc) {
            nuscenes_dir = argv[++i];
        } else if (str_eq(argv[i], "--skip-map")) {
            skip_map = true;
        } else if (str_eq(argv[i], "--skip-bev")) {
            skip_bev = true;
        } else if (str_eq(argv[i], "--save-json")) {
            save_json = true;
        } else if (str_eq(argv[i], "--no-warmup")) {
            no_warmup = true;
        } else if (str_eq(argv[i], "--verbose")) {
            verbose = true;
        } else if (!str_eq(argv[i], "--batch")) {
            // precision 参数（fp16 / int8）
            if (argv[i][0] != '-')
                precision = argv[i];
        }
    }

    if (!path_exists(frames_dir)) {
        fprintf(stderr, "[joint_inference] frames_dir 不存在: %s\n",
                frames_dir.c_str());
        return 1;
    }

    printf("=== CUDA-FastBEV 联合感知推理 ===\n");
    printf("  帧目录  : %s\n", frames_dir.c_str());
    printf("  BEV 模型: %s (%s)\n", model_name.c_str(), precision.c_str());
    printf("  置信度  : %.2f\n", score_thr);
    if (!skip_map) {
        printf("  地图模式: %s\n", map_mode.c_str());
        if (map_mode == "model" || map_mode == "auto")
            printf("  MapTR ckpt: %s\n", ckpt_path.c_str());
    } else {
        printf("  地图模式: 跳过\n");
    }
    printf("  保存 JSON: %s\n", save_json ? "是" : "否");
    printf("\n");

    // ── Step 1: 收集帧目录列表 ────────────────────────────────────────────
    auto frame_dirs = collect_frame_dirs(frames_dir);
    if (frame_dirs.empty()) {
        fprintf(stderr, "[joint_inference] 未找到 frame_* 子目录: %s\n",
                frames_dir.c_str());
        return 1;
    }
    printf("共找到 %zu 帧\n\n", frame_dirs.size());

    // ── Step 2: 初始化 BEV 感知管线 ──────────────────────────────────────
    std::unique_ptr<PerceptionPipeline> bev_pipeline;
    if (!skip_bev) {
        PipelineConfig cfg;
        cfg.model_name      = model_name;
        cfg.precision       = precision;
        cfg.score_thr       = score_thr;
        cfg.enable_tracking = true;
        cfg.save_output     = save_json;
        cfg.output_dir      = frames_dir;
        cfg.output_format   = "json";
        cfg.verbose         = verbose;

        bev_pipeline = std::make_unique<PerceptionPipeline>(cfg);
        if (!bev_pipeline->init()) {
            fprintf(stderr, "[joint_inference] BEV 管线初始化失败\n");
            return 1;
        }
        printf("[BEV] 推理核心初始化成功 (%s / %s)\n",
               model_name.c_str(), precision.c_str());

        // warmup
        if (!no_warmup) {
            printf("[BEV] Warmup...\n");
            auto warm_frame_ptr = camera::FrameLoader::load_from_dir(
                frame_dirs[0], 0);
            if (warm_frame_ptr) {
                bev_pipeline->process(*warm_frame_ptr);
                camera::FrameLoader::free_frame(*warm_frame_ptr);
            }
            // 重置跟踪器，warmup 帧不计入跟踪
            bev_pipeline->reset_tracker();
            printf("[BEV] Warmup 完成\n");
        }
    }

    // ── Step 3: 初始化 MapRunner，后台线程并行运行 MapTR ─────────────────
    // MapTR 在独立线程里通过 Python subprocess 执行，与主线程的 BEV 推理同时进行。
    // 两者资源分离（MapTR 主要占用 Python 进程 / CPU，BEV 使用 TensorRT / GPU），
    // 实际墙钟耗时约等于 max(BEV 总耗时, MapTR 总耗时)。
    std::unique_ptr<MapRunner> map_runner;
    std::thread      map_thread;
    std::atomic<int> map_batch_count{-99};   // -99 = 未启动
    double           map_thread_ms = 0.0;    // MapTR 线程墙钟耗时（ms）

    if (!skip_map) {
        MapRunnerConfig mcfg;
        mcfg.mode          = map_mode;
        mcfg.frames_dir    = frames_dir;
        mcfg.nuscenes_dir  = nuscenes_dir;
        mcfg.ckpt_path     = ckpt_path;   // model 模式使用
        mcfg.score_thr     = score_thr;
        mcfg.verbose       = verbose;

        map_runner = std::make_unique<MapRunner>(mcfg);

        printf("[MapTR] 后台启动批量地图推理（模式: %s），与 BEV 并行...\n",
               map_mode.c_str());
        printf("[MapTR] BEV 每帧取最新可用结果（无结果时复用上一帧），不等待 MapTR。\n");

        // 启动后台线程：run_batch() 内部调用 Python subprocess（阻塞直到完成）
        map_thread = std::thread([&map_runner, &map_batch_count, &map_thread_ms]() {
            auto t0 = std::chrono::high_resolution_clock::now();
            map_batch_count.store(map_runner->run_batch(/*overwrite=*/false));
            auto t1 = std::chrono::high_resolution_clock::now();
            map_thread_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("\n[MapTR] 线程完成，耗时 %.1f ms\n", map_thread_ms);
            fflush(stdout);
        });
    }

    // ── Step 4: BEV 逐帧推理（与 MapTR 线程同时进行） ────────────────────
    // BEV 不等待 MapTR：每帧尝试读取 map_result.json，
    // 若当帧尚未写入（MapTR 还没处理到），则复用上一帧的 map 结果。
    size_t total_frames  = frame_dirs.size();
    size_t bev_ok        = 0;
    double bev_total_ms  = 0.0;
    auto   bev_wall_t0   = std::chrono::high_resolution_clock::now();
    MapResult last_map;   // 最近一次成功读取的 map 结果（供回退使用）
    int       map_hits   = 0;  // 当帧有新结果的帧数
    int       map_stale  = 0;  // 使用上一帧结果（MapTR 未就绪）的帧数

    for (size_t fi = 0; fi < total_frames; ++fi) {
        const std::string& fdir = frame_dirs[fi];

        // ── 4a. 图像预处理（加载帧数据） ─────────────────────────────────
        camera::CameraFrame bev_frame;
        RawImageInput       map_input;

        bool prep_ok = ImagePreprocessor::from_frame_dir(
            fdir, static_cast<uint64_t>(fi), bev_frame, map_input);

        if (!prep_ok) {
            fprintf(stderr, "[joint_inference] 跳过帧（预处理失败）: %s\n",
                    fdir.c_str());
            continue;
        }

        JointPerceptionResult joint;
        joint.frame_id  = bev_frame.frame_id;
        joint.timestamp = bev_frame.timestamp;

        // ── 4b. BEV 推理 + 跟踪 ──────────────────────────────────────────
        if (!skip_bev && bev_pipeline) {
            auto t0 = std::chrono::high_resolution_clock::now();
            PerceptionResult pr = bev_pipeline->process(bev_frame);
            auto t1 = std::chrono::high_resolution_clock::now();
            double lat = std::chrono::duration<double, std::milli>(t1 - t0).count();

            joint.detections   = pr.detections;
            joint.tracks       = pr.tracks;
            joint.bev_latency_ms = lat;
            bev_total_ms += lat;
            ++bev_ok;

            if (verbose) {
                printf("[BEV] frame %04zu  det=%zu  track=%zu  lat=%.1fms\n",
                       fi, pr.detections.size(), pr.tracks.size(), lat);
            }

            // 将 tracks.json 写入帧目录（兼容 tracking_demo 的现有格式）
            if (save_json) {
                std::string track_path = fdir + "/tracks.json";
                std::ofstream tf(track_path);
                tf << "[\n";
                for (size_t ti = 0; ti < pr.tracks.size(); ++ti) {
                    const auto& t = pr.tracks[ti];
                    tf << "  {\"track_id\": " << t.track_id
                       << ", \"position\": [" << t.x << ", " << t.y << ", " << t.z << "]"
                       << ", \"size\": [" << t.w << ", " << t.l << ", " << t.h << "]"
                       << ", \"yaw\": " << t.yaw
                       << ", \"velocity\": [" << t.vx << ", " << t.vy << "]"
                       << ", \"score\": " << t.score
                       << ", \"class_id\": " << t.class_id
                       << "}";
                    if (ti + 1 < pr.tracks.size()) tf << ",";
                    tf << "\n";
                }
                tf << "]\n";
            }
        }

        // ── 4c. 读取 MapTR 最新可用结果 ──────────────────────────────────
        // 非阻塞：read_result 直接读文件，若文件不存在返回 false；
        // 此时使用 last_map（上一帧或更早写入的结果），BEV 绝不等待。
        if (!skip_map && map_runner) {
            MapResult cur_map;
            auto t0 = std::chrono::high_resolution_clock::now();
            bool got = map_runner->read_result(fdir, cur_map);
            auto t1 = std::chrono::high_resolution_clock::now();

            if (got) {
                last_map = cur_map;       // 更新最新缓存
                joint.map = cur_map;
                ++map_hits;
            } else if (last_map.is_valid()) {
                joint.map = last_map;     // 复用上一帧结果
                ++map_stale;
            }
            joint.map_latency_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (verbose) {
                printf("[MAP] frame %04zu  %s  elems=%zu\n",
                       fi,
                       got ? "新结果" : "复用缓存",
                       joint.map.elements.size());
            }
        }

        // ── 4d. 保存联合结果 ──────────────────────────────────────────────
        if (save_json) {
            std::string jpath = fdir + "/joint_result.json";
            std::ofstream jf(jpath);
            jf << result_to_json(joint);
        }

        // 打印进度
        if ((fi + 1) % 10 == 0 || fi + 1 == total_frames) {
            printf("\r[Progress] %zu / %zu 帧完成", fi + 1, total_frames);
            fflush(stdout);
        }

        // 释放 BEV 帧内存（图像由 stb_image 分配，张量由 FrameLoader 分配）
        camera::FrameLoader::free_frame(bev_frame);
    }
    auto bev_wall_t1 = std::chrono::high_resolution_clock::now();
    double bev_wall_ms =
        std::chrono::duration<double, std::milli>(bev_wall_t1 - bev_wall_t0).count();
    printf("\n");

    // ── Step 5: 等待 MapTR 线程完成 ───────────────────────────────────────
    if (map_thread.joinable()) {
        if (map_batch_count.load() == -99) {
            // MapTR 仍在运行，BEV 已经结束
            printf("[BEV 已完成，等待 MapTR 线程结束...]\n");
            fflush(stdout);
        }
        map_thread.join();
    }

    // ── Step 6: 并行耗时统计 ──────────────────────────────────────────────
    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║         并行推理耗时统计                 ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    if (!skip_bev) {
        printf("║  BEV 推理帧数 : %4zu / %4zu              ║\n",
               bev_ok, total_frames);
        printf("║  BEV 逐帧均值 : %8.1f ms / 帧          ║\n",
               bev_ok > 0 ? bev_total_ms / bev_ok : 0.0);
        printf("║  BEV 墙钟总计 : %8.1f ms               ║\n",
               bev_wall_ms);
    }
    if (!skip_map) {
        int mc = map_batch_count.load();
        if (mc < 0)
            printf("║  MapTR        : 推理失败 (code=%d)        ║\n", mc);
        else {
            printf("║  MapTR 推理帧 : %4d 帧（模式: %-9s）║\n",
                   mc, map_mode.c_str());
            printf("║  MapTR 墙钟   : %8.1f ms               ║\n",
                   map_thread_ms);
        }
        // 最新结果统计：当帧命中 vs 使用上帧缓存
        printf("║  BEV 命中新图 : %4d 帧 / 复用缓存: %3d 帧  ║\n",
               map_hits, map_stale);
    }
    if (!skip_bev && !skip_map) {
        double parallel_total = std::max(bev_wall_ms, map_thread_ms);
        double sequential_est = bev_wall_ms + map_thread_ms;
        double saved_ms       = sequential_est - parallel_total;
        printf("╠══════════════════════════════════════════╣\n");
        printf("║  并行墙钟总计 : %8.1f ms               ║\n", parallel_total);
        printf("║  顺序估算总计 : %8.1f ms               ║\n", sequential_est);
        printf("║  并行节省时间 : %8.1f ms  (%.0f%%)         ║\n",
               saved_ms,
               sequential_est > 0 ? 100.0 * saved_ms / sequential_est : 0.0);
    }
    printf("╚══════════════════════════════════════════╝\n");

    if (save_json)
        printf("\n结果已保存到各帧目录的 joint_result.json / tracks.json\n");

    return 0;
}
