/**
 * joint_inference.cpp — BEV + MapTR 联合感知主程序
 * (修改版：使用新版 tracking 模块)
 *
 * 用法及参数说明与原版相同，新增以下跟踪选项：
 *   --track-threshold F  关联阈值（默认 -0.4）
 *   --track-metric S     度量方式：giou_3d / dist_3d / iou_3d / mahalanobis（默认 giou_3d）
 *   --track-max-lost N   最大丢失帧数（默认 3）
 *   --track-dt F         帧间预期时间差（秒，默认 0.5）
 *   --track-algo S       匹配算法：greedy / hungarian（默认 greedy）
 *   --track-min-hits N   最小命中数（默认 3）
 *   --track-ego-comp N   是否启用 ego 运动补偿（1/0，默认 1）
 */

#include "perception/perception_types.hpp"
#include "perception/image_preprocessor.hpp"
#include "perception/map_runner.hpp"
#include "pipeline/perception_pipeline.hpp"
#include "camera/camera_frame.hpp"
#include "tracking/tracker.hpp"   // 新版跟踪器
#include "tracking/track.hpp"

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

// 从 meta.json 读取 ego 位姿（全局坐标）和时间戳
static fastbev::tracking::EgoPose read_ego_pose(const std::string& frame_dir, double& timestamp_sec)
{
    fastbev::tracking::EgoPose pose;
    timestamp_sec = 0.0;
    std::string meta_path = frame_dir + "/meta.json";
    std::ifstream f(meta_path);
    if (!f.is_open()) return pose;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    // 辅助 lambda：提取 "key": value
    auto get_double = [&](const std::string& key, double def=0.0) -> double {
        std::string pat = "\"" + key + "\"";
        size_t kp = content.find(pat);
        if (kp == std::string::npos) return def;
        size_t vp = content.find_first_of("-0123456789", kp + pat.size());
        if (vp == std::string::npos) return def;
        try { return std::stod(content.substr(vp)); } catch(...) { return def; }
    };
    auto get_array3 = [&](const std::string& key, double out[3]) -> bool {
        std::string pat = "\"" + key + "\"";
        size_t kp = content.find(pat);
        if (kp == std::string::npos) return false;
        size_t lb = content.find('[', kp + pat.size());
        size_t rb = content.find(']', lb);
        if (lb == std::string::npos || rb == std::string::npos) return false;
        std::string arr = content.substr(lb+1, rb-lb-1);
        int idx = 0;
        size_t pos = 0;
        while (idx < 3 && pos < arr.size()) {
            size_t sp = arr.find_first_of("-0123456789", pos);
            if (sp == std::string::npos) break;
            try {
                size_t end = 0;
                out[idx++] = std::stod(arr.substr(sp), &end);
                pos = sp + end;
            } catch(...) { break; }
        }
        return idx == 3;
    };

    double trans[3] = {0,0,0};
    if (get_array3("ego_translation_global", trans)) {
        pose.x = trans[0];
        pose.y = trans[1];
    }
    pose.yaw   = get_double("ego_yaw_global", 0.0);
    pose.valid = true;

    // 读取时间戳（微秒），转换为秒
    int64_t ts_us = static_cast<int64_t>(get_double("timestamp_us", 0.0));
    if (ts_us == 0) ts_us = static_cast<int64_t>(get_double("timestamp", 0.0));
    timestamp_sec = static_cast<double>(ts_us) / 1e6;
    return pose;
}

// 将 JointPerceptionResult 序列化为 JSON 字符串（格式与原版完全相同）
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

    // map
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
        "  --nuscenes-dir S     NuScenes 数据目录（默认 data/nuscenes）\n"
        "  --skip-map           仅运行 BEV，跳过 MapTR\n"
        "  --skip-bev           仅运行 MapTR，跳过 BEV\n"
        "  --save-json          保存联合结果到 <frame_dir>/joint_result.json\n"
        "  --no-warmup          跳过 TRT warmup\n"
        "  --verbose            打印详细信息\n"
        "  --track-threshold F  跟踪关联阈值（默认 -0.4）\n"
        "  --track-metric S     度量方式：giou_3d/dist_3d/iou_3d/mahalanobis（默认 giou_3d）\n"
        "  --track-max-lost N   最大丢失帧数（默认 3）\n"
        "  --track-dt F         帧间预期时间差（秒，默认 0.5）\n"
        "  --track-algo S       匹配算法：greedy/hungarian（默认 greedy）\n"
        "  --track-min-hits N   最小命中数（默认 3）\n"
        "  --track-ego-comp N   启用 ego 运动补偿（1/0，默认 1）\n"
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

    // ── 解析可选参数（包含跟踪配置）────────────────────────────────────────
    float       score_thr    = 0.35f;
    std::string map_mode     = "auto";
    std::string nuscenes_dir = "data/nuscenes";
    std::string ckpt_path    = "model/maptr/maptr_nano_r18_110e.pth";
    bool        skip_map     = false;
    bool        skip_bev     = false;
    bool        save_json    = false;
    bool        no_warmup    = false;
    bool        verbose      = false;

    // 跟踪参数（新版tracking使用）
    float       track_threshold  = -0.4f;
    std::string track_metric_str = "giou_3d";
    int         track_max_lost   = 3;
    float       track_dt         = 0.5f;
    std::string track_algo_str   = "greedy";
    int         track_min_hits   = 3;
    bool        track_ego_comp   = true;

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
        } else if (str_eq(argv[i], "--track-threshold") && i + 1 < argc) {
            track_threshold = static_cast<float>(atof(argv[++i]));
        } else if (str_eq(argv[i], "--track-metric") && i + 1 < argc) {
            track_metric_str = argv[++i];
        } else if (str_eq(argv[i], "--track-max-lost") && i + 1 < argc) {
            track_max_lost = atoi(argv[++i]);
        } else if (str_eq(argv[i], "--track-dt") && i + 1 < argc) {
            track_dt = static_cast<float>(atof(argv[++i]));
        } else if (str_eq(argv[i], "--track-algo") && i + 1 < argc) {
            track_algo_str = argv[++i];
        } else if (str_eq(argv[i], "--track-min-hits") && i + 1 < argc) {
            track_min_hits = atoi(argv[++i]);
        } else if (str_eq(argv[i], "--track-ego-comp") && i + 1 < argc) {
            track_ego_comp = (atoi(argv[++i]) != 0);
        } else if (!str_eq(argv[i], "--batch")) {
            // precision 参数
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
    printf("  跟踪参数: thr=%.2f, metric=%s, max_lost=%d, dt=%.2f, algo=%s, min_hits=%d, ego_comp=%d\n",
           track_threshold, track_metric_str.c_str(), track_max_lost, track_dt,
           track_algo_str.c_str(), track_min_hits, track_ego_comp);
    printf("\n");

    // ── Step 1: 收集帧目录列表 ────────────────────────────────────────────
    auto frame_dirs = collect_frame_dirs(frames_dir);
    if (frame_dirs.empty()) {
        fprintf(stderr, "[joint_inference] 未找到 frame_* 子目录: %s\n",
                frames_dir.c_str());
        return 1;
    }
    printf("共找到 %zu 帧\n\n", frame_dirs.size());

    // ── Step 2: 初始化 BEV 感知管线（禁用其内部跟踪）──────────────────────
    std::unique_ptr<PerceptionPipeline> bev_pipeline;
    if (!skip_bev) {
        PipelineConfig cfg;
        cfg.model_name      = model_name;
        cfg.precision       = precision;
        cfg.score_thr       = score_thr;
        cfg.enable_tracking = false;   // 禁用旧版跟踪，使用新版
        cfg.save_output     = false;   // 我们手动保存
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
            // 重置跟踪器（新版tracker在后续循环中单独创建，无需重置）
            printf("[BEV] Warmup 完成\n");
        }
    }

    // ── Step 3: 初始化 MapRunner，后台线程并行运行 MapTR ─────────────────
    std::unique_ptr<MapRunner> map_runner;
    std::thread      map_thread;
    std::atomic<int> map_batch_count{-99};
    double           map_thread_ms = 0.0;

    if (!skip_map) {
        MapRunnerConfig mcfg;
        mcfg.mode          = map_mode;
        mcfg.frames_dir    = frames_dir;
        mcfg.nuscenes_dir  = nuscenes_dir;
        mcfg.ckpt_path     = ckpt_path;
        mcfg.score_thr     = score_thr;
        mcfg.verbose       = verbose;

        map_runner = std::make_unique<MapRunner>(mcfg);

        printf("[MapTR] 后台启动批量地图推理（模式: %s），与 BEV 并行...\n",
               map_mode.c_str());
        printf("[MapTR] BEV 每帧取最新可用结果（无结果时复用上一帧），不等待 MapTR。\n");

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

    // ── Step 4: 初始化新版跟踪器 ──────────────────────────────────────────
    fastbev::tracking::MetricType metric = fastbev::tracking::MetricType::GIOU_3D;
    if (track_metric_str == "dist_3d") metric = fastbev::tracking::MetricType::DIST_3D;
    else if (track_metric_str == "iou_3d") metric = fastbev::tracking::MetricType::IOU_3D;
    else if (track_metric_str == "mahalanobis") metric = fastbev::tracking::MetricType::MAHALANOBIS;

    fastbev::tracking::AlgoType algo;
    if (track_algo_str == "hungarian") algo = fastbev::tracking::AlgoType::HUNGARIAN;
    else algo = fastbev::tracking::AlgoType::GREEDY;

    fastbev::tracking::TrackerConfig tracker_cfg;
    tracker_cfg.threshold       = track_threshold;
    tracker_cfg.metric          = metric;
    tracker_cfg.max_lost_frames = track_max_lost;
    tracker_cfg.dt              = track_dt;
    tracker_cfg.algo            = algo;
    tracker_cfg.min_hits        = track_min_hits;
    tracker_cfg.enable_ego_comp = track_ego_comp;

    fastbev::tracking::Tracker tracker(tracker_cfg);

    // ── Step 5: BEV 逐帧推理（与 MapTR 线程同时进行） ────────────────────
    size_t total_frames  = frame_dirs.size();
    size_t bev_ok        = 0;
    double bev_total_ms  = 0.0;
    auto   bev_wall_t0   = std::chrono::high_resolution_clock::now();
    MapResult last_map;   // 最近一次成功读取的 map 结果
    int       map_hits   = 0;
    int       map_stale  = 0;

    for (size_t fi = 0; fi < total_frames; ++fi) {
        const std::string& fdir = frame_dirs[fi];

        // ── 5a. 图像预处理（加载帧数据） ─────────────────────────────────
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

        // ── 5b. BEV 推理（仅检测，无跟踪） ───────────────────────────────
        PerceptionResult pr;
        if (!skip_bev && bev_pipeline) {
            auto t0 = std::chrono::high_resolution_clock::now();
            pr = bev_pipeline->process(bev_frame);
            auto t1 = std::chrono::high_resolution_clock::now();
            double lat = std::chrono::duration<double, std::milli>(t1 - t0).count();

            joint.detections   = pr.detections;
            bev_total_ms += lat;
            ++bev_ok;

            if (verbose) {
                printf("[BEV] frame %04zu  det=%zu  lat=%.1fms\n",
                       fi, pr.detections.size(), lat);
            }
        }

        // ── 5c. 新版跟踪：将检测框转换为 Detection 并更新 tracker ────────
        std::vector<fastbev::tracking::Detection> detections;
        for (const auto& box : joint.detections) {
            fastbev::tracking::Detection det;
            det.x      = box.x;
            det.y      = box.y;
            det.z      = box.z;
            det.w      = box.w;
            det.l      = box.l;
            det.h      = box.h;
            det.yaw    = box.yaw;
            det.vx     = box.vx;   // 检测框速度可能为0，后续由卡尔曼滤波估计
            det.vy     = box.vy;
            det.score  = box.score;
            det.class_id = box.class_id;
            detections.push_back(det);
        }

        // 读取 ego 位姿和时间戳
        double timestamp_sec = 0.0;
        fastbev::tracking::EgoPose ego_pose = read_ego_pose(fdir, timestamp_sec);
        // 如果时间戳无效，回退到 bev_frame.timestamp（微秒转秒）
        if (timestamp_sec < 1e-5 && bev_frame.timestamp > 0) {
            timestamp_sec = static_cast<double>(bev_frame.timestamp) / 1e6;
        }

        // 更新跟踪器，获得当前帧轨迹
        auto tracks = tracker.update(detections, timestamp_sec, ego_pose);
        joint.tracks = tracks;   // 直接赋值，joint.tracks 类型已是 std::vector<fastbev::tracking::Track>

        // ── 5d. 读取 MapTR 最新可用结果（同原逻辑）────────────────────────
        if (!skip_map && map_runner) {
            MapResult cur_map;
            auto t0 = std::chrono::high_resolution_clock::now();
            bool got = map_runner->read_result(fdir, cur_map);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (got) {
                last_map = cur_map;
                joint.map = cur_map;
                ++map_hits;
            } else if (last_map.is_valid()) {
                joint.map = last_map;
                ++map_stale;
            }
            joint.map_latency_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (verbose) {
                printf("[MAP] frame %04zu  %s  elems=%zu\n",
                       fi, got ? "新结果" : "复用缓存", joint.map.elements.size());
            }
        }

        // ── 5e. 保存联合结果及单独的轨迹文件（JSON 格式与原版一致）────────
        if (save_json) {
            std::string jpath = fdir + "/joint_result.json";
            std::ofstream jf(jpath);
            jf << result_to_json(joint);

            // 保存单独的 tracks.json，格式与原版相同
            std::string track_path = fdir + "/tracks.json";
            std::ofstream tf(track_path);
            tf << "[\n";
            for (size_t ti = 0; ti < joint.tracks.size(); ++ti) {
                const auto& t = joint.tracks[ti];
                tf << "  {\"track_id\": " << t.track_id
                   << ", \"position\": [" << t.x << ", " << t.y << ", " << t.z << "]"
                   << ", \"size\": [" << t.w << ", " << t.l << ", " << t.h << "]"
                   << ", \"yaw\": " << t.yaw
                   << ", \"velocity\": [" << t.vx << ", " << t.vy << "]"
                   << ", \"score\": " << t.score
                   << ", \"class_id\": " << t.class_id
                   << "}";
                if (ti + 1 < joint.tracks.size()) tf << ",";
                tf << "\n";
            }
            tf << "]\n";
        }

        // 打印进度
        if ((fi + 1) % 10 == 0 || fi + 1 == total_frames) {
            printf("\r[Progress] %zu / %zu 帧完成", fi + 1, total_frames);
            fflush(stdout);
        }

        // 释放 BEV 帧内存
        camera::FrameLoader::free_frame(bev_frame);
    }
    auto bev_wall_t1 = std::chrono::high_resolution_clock::now();
    double bev_wall_ms =
        std::chrono::duration<double, std::milli>(bev_wall_t1 - bev_wall_t0).count();
    printf("\n");

    // ── Step 6: 等待 MapTR 线程完成 ───────────────────────────────────────
    if (map_thread.joinable()) {
        if (map_batch_count.load() == -99) {
            printf("[BEV 已完成，等待 MapTR 线程结束...]\n");
            fflush(stdout);
        }
        map_thread.join();
    }

    // ── Step 7: 并行耗时统计 ──────────────────────────────────────────────
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