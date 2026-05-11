/*
 * CUDA-FastBEV 推理入口
 *
 * 功能特性：
 *   - 支持单帧或多帧目录批量推理
 *   - 可配置置信度阈值和类别过滤
 *   - 支持 TXT / JSON 两种输出格式
 *   - 适配 NuScenes 数据（通过 tools/nuscenes_adapter.py 预处理）
 *
 * 用法：
 *   ./fastbev [data_dir] [model] [precision] [options]
 *
 * 选项：
 *   --score-thr <float>          置信度阈值（默认 0.5）
 *   --classes <idx,...>          保留的类别 id，逗号分隔（默认 all）
 *                                  0=car 1=truck 2=construction_vehicle
 *                                  3=bus 4=trailer 5=barrier
 *                                  6=motorcycle 7=bicycle 8=pedestrian 9=traffic_cone
 *   --output-format <txt|json>   输出格式（默认 txt）
 *   --no-warmup                  跳过 warmup 推理
 *   --batch                      批量模式：data_dir 为包含 frame_* 子目录的根目录
 */

#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "fastbev/fastbev.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"
#include "tracking/tracker.hpp"
#include "tracking/track.hpp"

// ─── NuScenes 10 类名称 ────────────────────────────────────────────────────
static const char* CLASS_NAMES[] = {
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
};
static const int NUM_CLASSES = 10;

// ─── 轻量 JSON 辅助（避免引入第三方库）────────────────────────────────────

/// 从 JSON 字符串中提取 "key": <scalar_number>
static double json_get_double(const std::string& json,
                              const std::string& key,
                              double default_val = 0.0)
{
    std::string pat = "\"" + key + "\"";
    size_t kp = json.find(pat);
    if (kp == std::string::npos) return default_val;
    size_t vp = json.find_first_of("-0123456789", kp + pat.size());
    if (vp == std::string::npos) return default_val;
    try { return std::stod(json.substr(vp)); } catch (...) { return default_val; }
}

/// 从 JSON 字符串中提取 "key": [v0, v1, v2] 并填入 out[0..2]
static bool json_get_array3(const std::string& json,
                            const std::string& key,
                            double out[3])
{
    std::string pat = "\"" + key + "\"";
    size_t kp = json.find(pat);
    if (kp == std::string::npos) return false;
    size_t lb = json.find('[', kp + pat.size());
    size_t rb = json.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return false;
    std::string arr = json.substr(lb + 1, rb - lb - 1);
    // parse comma-separated numbers
    int idx = 0;
    size_t pos = 0;
    while (idx < 3 && pos < arr.size()) {
        size_t sp = arr.find_first_of("-0123456789", pos);
        if (sp == std::string::npos) break;
        try {
            size_t end = 0;
            out[idx++] = std::stod(arr.substr(sp), &end);
            pos = sp + end;
        } catch (...) { break; }
    }
    return idx == 3;
}

/// 从 frame_dir/meta.json 中读取 ego 位姿（全局坐标）
static fastbev::tracking::EgoPose read_ego_pose(const std::string& frame_dir)
{
    fastbev::tracking::EgoPose pose;
    std::string meta_path = frame_dir + "/meta.json";
    std::ifstream f(meta_path);
    if (!f.is_open()) return pose;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    double trans[3] = {0, 0, 0};
    if (json_get_array3(content, "ego_translation_global", trans)) {
        pose.x = trans[0];
        pose.y = trans[1];
    }
    pose.yaw   = json_get_double(content, "ego_yaw_global", 0.0);
    pose.valid = true;  // meta.json 存在即视为有效
    return pose;
}

/// 从 frame_dir/meta.json 中读取 timestamp 字段，并转换为秒
static double read_timestamp(const std::string& frame_dir)
{
    std::string meta_path = frame_dir + "/meta.json";
    std::ifstream f(meta_path);
    if (!f.is_open()) return 0.0;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();
    // 查找 "timestamp":
    std::string key = "\"timestamp\"";
    size_t pos = content.find(key);
    if (pos == std::string::npos) return 0.0;
    // 找到数字起始位置
    pos = content.find_first_of("0123456789", pos + key.size());
    if (pos == std::string::npos) return 0.0;
    char* end = nullptr;
    double ts = std::strtod(content.c_str() + pos, &end);
    // 原始单位是微秒，转换为秒
    return ts / 1e6;
}

// ─── 命令行参数结构 ────────────────────────────────────────────────────────
struct InferConfig {
  std::string data_dir      = "example-data";
  std::string model         = "resnet18";
  std::string precision     = "fp16";
  float       score_thr     = 0.5f;
  std::string output_format = "txt";  // "txt" 或 "json"
  bool        batch_mode    = false;  // true: data_dir 下有 frame_* 子目录
  bool        do_warmup     = true;
  std::set<int> class_filter;         // 空 = 保留全部类别
  // 跟踪参数
  float       track_threshold  = -0.4f;
  std::string track_metric_str = "giou_3d";
  int         track_max_lost   = 3;
  float       track_dt         = 0.5f;
  std::string track_algo_str   = "greedy";
  int         track_min_hits   = 3;
  bool        track_ego_comp   = true;
};

// ─── 工具函数 ──────────────────────────────────────────────────────────────

static bool file_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static std::string resolve_data_root(const std::string& root) {
  if (file_exists(root + "/valid_c_idx.tensor")) return root;
  std::string nested = root + "/example-data";
  if (file_exists(nested + "/valid_c_idx.tensor")) return nested;
  return root;
}

static std::vector<unsigned char*> load_images(const std::string& root) {
  const char* file_names[] = {
    "0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
    "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"
  };
  std::vector<unsigned char*> images;
  for (int i = 0; i < 6; ++i) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", root.c_str(), file_names[i]);
    int width, height, channels;
    unsigned char* img = stbi_load(path, &width, &height, &channels, 0);
    if (!img) {
      fprintf(stderr, "[警告] 无法加载图像: %s\n", path);
    }
    images.push_back(img);
  }
  return images;
}

static void free_images(std::vector<unsigned char*>& images) {
  for (size_t i = 0; i < images.size(); ++i) {
    if (images[i]) stbi_image_free(images[i]);
  }
  images.clear();
}

// ─── 过滤函数：置信度阈值、类别过滤 + 跨类别 IoU NMS 去重 ────
static std::vector<fastbev::post::transbbox::BoundingBox> filter_boxes(
    const std::vector<fastbev::post::transbbox::BoundingBox>& boxes,
    float score_thr,
    const std::set<int>& class_filter,
    float iou_threshold = 0.3f)   // 更宽松的 IoU 阈值，适应不同尺寸重叠
{
    // 第一步：置信度 + 类别过滤
    std::vector<fastbev::post::transbbox::BoundingBox> filtered;
    for (const auto& box : boxes) {
        if (box.score < score_thr) continue;
        if (!class_filter.empty() && class_filter.find(box.id) == class_filter.end()) continue;
        filtered.push_back(box);
    }
    if (filtered.empty()) return {};

    // 第二步：计算两个框的 BEV IoU 的辅助函数（修正角度符号）
    auto compute_bev_iou = [](const fastbev::post::transbbox::BoundingBox& a,
                              const fastbev::post::transbbox::BoundingBox& b) -> float {
        // OpenCV 角度：顺时针为正，检测输出通常逆时针，故取负
        cv::RotatedRect rect_a(
            cv::Point2f(a.position.x, a.position.y),
            cv::Size2f(a.size.l, a.size.w),
            static_cast<float>(-a.z_rotation * 180.0 / M_PI)
        );
        cv::RotatedRect rect_b(
            cv::Point2f(b.position.x, b.position.y),
            cv::Size2f(b.size.l, b.size.w),
            static_cast<float>(-b.z_rotation * 180.0 / M_PI)
        );
        // 获取四个顶点
        cv::Point2f points_a[4], points_b[4];
        rect_a.points(points_a);
        rect_b.points(points_b);
        std::vector<cv::Point2f> poly_a(points_a, points_a + 4);
        std::vector<cv::Point2f> poly_b(points_b, points_b + 4);
        // 计算交集面积
        std::vector<cv::Point2f> intersection;
        double inter_area = 0.0;
        if (cv::intersectConvexConvex(poly_a, poly_b, intersection)) {
            inter_area = cv::contourArea(intersection);
        }
        double area_a = rect_a.size.area();
        double area_b = rect_b.size.area();
        double union_area = area_a + area_b - inter_area;
        if (union_area <= 0) return 0.0f;
        return static_cast<float>(inter_area / union_area);
    };

    // 第三步：按置信度降序排序（高分优先）
    std::sort(filtered.begin(), filtered.end(),
              [](const auto& x, const auto& y) { return x.score > y.score; });

    // 第四步：跨类别 NMS（不再区分类别）
    std::vector<fastbev::post::transbbox::BoundingBox> result;
    std::vector<bool> kept(filtered.size(), false);
    
    // 可选：预先计算每个框的中心，用于快速距离过滤
    std::vector<float> cx(filtered.size()), cy(filtered.size());
    for (size_t i = 0; i < filtered.size(); ++i) {
        cx[i] = filtered[i].position.x;
        cy[i] = filtered[i].position.y;
    }

    for (size_t i = 0; i < filtered.size(); ++i) {
        if (kept[i]) continue;
        const auto& box_i = filtered[i];
        result.push_back(box_i);
        kept[i] = true;
        // 快速过滤：若中心距离 > 两个框的最大对角线的一半，则不可能有高 IoU，跳过 IoU 计算
        float max_diag_i = std::hypot(box_i.size.l, box_i.size.w);
        for (size_t j = i + 1; j < filtered.size(); ++j) {
            if (kept[j]) continue;
            float dx = cx[i] - cx[j];
            float dy = cy[i] - cy[j];
            float dist_centers = std::hypot(dx, dy);
            const auto& box_j = filtered[j];
            float max_diag_j = std::hypot(box_j.size.l, box_j.size.w);
            // 只有中心距离小于两框最大对角线之和的一半时才计算 IoU
            if (dist_centers > (max_diag_i + max_diag_j) * 0.5f) continue;
            float iou = compute_bev_iou(box_i, box_j);
            if (iou > iou_threshold) {
                kept[j] = true;
            }
        }
    }

    return result;
}

// ─── 输出函数：TXT 格式 ────────────────────────────────────────────────────

static void save_boxes_txt(
    const std::vector<fastbev::post::transbbox::BoundingBox>& boxes,
    const std::string& file_path)
{
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    std::cerr << "无法打开输出文件: " << file_path << std::endl;
    return;
  }
  // 格式: x y z w l h yaw class_id score
  for (const auto& box : boxes) {
    ofs << box.position.x  << " "
        << box.position.y  << " "
        << box.position.z  << " "
        << box.size.w      << " "
        << box.size.l      << " "
        << box.size.h      << " "
        << box.z_rotation  << " "
        << box.id          << " "
        << box.score       << "\n";
  }
  ofs.close();
  printf("  [TXT] 已保存 %zu 个检测框 → %s\n", boxes.size(), file_path.c_str());
}

// ─── 输出函数：JSON 格式 ───────────────────────────────────────────────────

static void save_boxes_json(
    const std::vector<fastbev::post::transbbox::BoundingBox>& boxes,
    const std::string& file_path)
{
  std::ofstream ofs(file_path);
  if (!ofs.is_open()) {
    std::cerr << "无法打开输出文件: " << file_path << std::endl;
    return;
  }
  ofs << "[\n";
  for (size_t i = 0; i < boxes.size(); ++i) {
    const auto& box = boxes[i];
    const char* class_name = (box.id >= 0 && box.id < NUM_CLASSES)
                              ? CLASS_NAMES[box.id] : "unknown";
    ofs << "  {\n"
        << "    \"label\": "      << box.id             << ",\n"
        << "    \"label_name\": \"" << class_name << "\",\n"
        << "    \"score\": "      << box.score           << ",\n"
        << "    \"center_xyz\": [" << box.position.x << ", "
                                   << box.position.y << ", "
                                   << box.position.z << "],\n"
        << "    \"size_xyz\": ["  << box.size.w << ", "
                                   << box.size.l << ", "
                                   << box.size.h << "],\n"
        << "    \"yaw\": "        << box.z_rotation      << ",\n"
        << "    \"velocity_xy\": [" << box.velocity.vx << ", "
                                     << box.velocity.vy << "]\n"
        << "  }";
    if (i + 1 < boxes.size()) ofs << ",";
    ofs << "\n";
  }
  ofs << "]\n";
  ofs.close();
}

static void save_boxes(
    const std::vector<fastbev::post::transbbox::BoundingBox>& boxes,
    const std::string& file_path,
    const std::string& format)
{
  if (format == "json") {
    save_boxes_json(boxes, file_path);
  } else {
    save_boxes_txt(boxes, file_path);
  }
}

// ─── 模型初始化 ────────────────────────────────────────────────────────────

static std::shared_ptr<fastbev::Core> create_core(
    const std::string& model, const std::string& /*precision*/)
{
  printf("创建推理核心: %s\n", model.c_str());
  fastbev::pre::NormalizationParameter normalization;
  normalization.image_width  = 1600;
  normalization.image_height = 900;
  normalization.output_width = 704;
  normalization.output_height= 256;
  normalization.num_camera   = 6;
  normalization.resize_lim   = 0.44f;
  normalization.interpolation= fastbev::pre::Interpolation::Nearest;

  float mean[3] = {123.675f, 116.28f, 103.53f};
  float std_v[3] = {58.395f,  57.12f,  57.375f};
  normalization.method = fastbev::pre::NormMethod::mean_std(mean, std_v, 1.0f, 0.0f);

  fastbev::pre::GeometryParameter geo_param;
  geo_param.feat_height  = 64;
  geo_param.feat_width   = 176;
  geo_param.num_camera   = 6;
  geo_param.valid_points = 160000;
  geo_param.volum_x      = 200;
  geo_param.volum_y      = 200;
  geo_param.volum_z      = 4;

  fastbev::CoreParameter param;
  param.pre_model  = nv::format("model/%s/build/fastbev_pre_trt.plan",           model.c_str());
  param.post_model = nv::format("model/%s/build/fastbev_post_trt_decode.plan",   model.c_str());
  param.normalize  = normalization;
  param.geo_param  = geo_param;
  return fastbev::create_core(param);
}

// ─── 单帧推理 ──────────────────────────────────────────────────────────────

static int run_single_frame(
    std::shared_ptr<fastbev::Core> core,
    const std::string& data_root,
    const std::string& output_path,
    cudaStream_t stream,
    const InferConfig& cfg)
{
  auto images = load_images(data_root);
  bool any_valid = false;
  for (auto* img : images) if (img) { any_valid = true; break; }
  if (!any_valid) {
    fprintf(stderr, "[错误] 无法加载任何图像，跳过: %s\n", data_root.c_str());
    free_images(images);
    return -1;
  }

  auto valid_c_idx = nv::Tensor::load(nv::format("%s/valid_c_idx.tensor", data_root.c_str()), false);
  auto valid_x     = nv::Tensor::load(nv::format("%s/x.tensor",           data_root.c_str()), false);
  auto valid_y     = nv::Tensor::load(nv::format("%s/y.tensor",           data_root.c_str()), false);

  if (valid_c_idx.empty() || valid_x.empty() || valid_y.empty()) {
    fprintf(stderr, "[错误] 无法加载几何张量: %s\n", data_root.c_str());
    free_images(images);
    return -1;
  }

  core->update(valid_c_idx.ptr<float>(), valid_x.ptr<int64_t>(), valid_y.ptr<int64_t>(), stream);

  auto all_boxes = core->forward((const unsigned char**)images.data(), stream);
  auto boxes     = filter_boxes(all_boxes, cfg.score_thr, cfg.class_filter);
  save_boxes(boxes, output_path, cfg.output_format);

  free_images(images);
  return static_cast<int>(boxes.size());
}

// ─── 批量模式：遍历 frame_* 子目录 ────────────────────────────────────────

static std::vector<std::string> list_frame_dirs(const std::string& root) {
  std::vector<std::string> dirs;
  DIR* dir = opendir(root.c_str());
  if (!dir) return dirs;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;
    if (name.find("frame_") == 0) {
      std::string full = root + "/" + name;
      struct stat st;
      if (stat(full.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        dirs.push_back(full);
      }
    }
  }
  closedir(dir);
  std::sort(dirs.begin(), dirs.end());
  return dirs;
}

// ─── 参数解析 ──────────────────────────────────────────────────────────────

static InferConfig parse_args(int argc, char** argv) {
  InferConfig cfg;
  // 位置参数
  if (argc > 1) cfg.data_dir    = argv[1];
  if (argc > 2) cfg.model       = argv[2];
  if (argc > 3) cfg.precision   = argv[3];

  // 可选参数
  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--score-thr" && i + 1 < argc) {
      cfg.score_thr = static_cast<float>(atof(argv[++i]));
    } else if (arg == "--classes" && i + 1 < argc) {
      std::string cls_str = argv[++i];
      std::stringstream ss(cls_str);
      std::string token;
      while (std::getline(ss, token, ',')) {
        cfg.class_filter.insert(atoi(token.c_str()));
      }
    } else if (arg == "--output-format" && i + 1 < argc) {
      cfg.output_format = argv[++i];
    } else if (arg == "--batch") {
      cfg.batch_mode = true;
    } else if (arg == "--no-warmup") {
      cfg.do_warmup = false;
    }
    else if (arg == "--track-threshold" && i + 1 < argc) {
      cfg.track_threshold = atof(argv[++i]);
    } else if (arg == "--track-metric" && i + 1 < argc) {
      cfg.track_metric_str = argv[++i];
    } else if (arg == "--track-max-lost" && i + 1 < argc) {
      cfg.track_max_lost = atoi(argv[++i]);
    } else if (arg == "--track-dt" && i + 1 < argc) {
      cfg.track_dt = atof(argv[++i]);
    } else if (arg == "--track-algo" && i + 1 < argc) {
      cfg.track_algo_str = argv[++i];
    } else if (arg == "--track-min-hits" && i + 1 < argc) {
      cfg.track_min_hits = atoi(argv[++i]);
    } else if (arg == "--track-ego-comp" && i + 1 < argc) {
      cfg.track_ego_comp = (atoi(argv[++i]) != 0);
    }
  }
  return cfg;
}

// ─── main ─────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  InferConfig cfg = parse_args(argc, argv);

  printf("========================================\n");
  printf("  CUDA-FastBEV 推理\n");
  printf("  模型:     %s\n",  cfg.model.c_str());
  printf("  数据:     %s\n",  cfg.data_dir.c_str());
  printf("  阈值:     %.2f\n", cfg.score_thr);
  printf("  输出格式: %s\n",  cfg.output_format.c_str());
  if (!cfg.class_filter.empty()) {
    printf("  类别过滤: ");
    for (int id : cfg.class_filter) printf("%d(%s) ", id,
        (id >= 0 && id < NUM_CLASSES) ? CLASS_NAMES[id] : "?");
    printf("\n");
  } else {
    printf("  类别过滤: 全部保留\n");
  }
  printf("  批量模式: %s\n", cfg.batch_mode ? "是" : "否");
  printf("========================================\n");

  auto core = create_core(cfg.model, cfg.precision);
  if (!core) {
    fprintf(stderr, "[错误] 模型初始化失败\n");
    return -1;
  }

  cudaStream_t stream;
  checkRuntime(cudaStreamCreate(&stream));
  core->print();

  std::string ext = (cfg.output_format == "json") ? ".json" : ".txt";

  if (cfg.batch_mode) {
    // ── 批量模式 ──
    auto frame_dirs = list_frame_dirs(cfg.data_dir);
    if (frame_dirs.empty()) {
      fprintf(stderr, "[错误] 在 %s 下未找到 frame_* 目录\n", cfg.data_dir.c_str());
      return -1;
    }
    printf("批量推理 %zu 帧...\n", frame_dirs.size());

    // 初始化跟踪器
    fastbev::tracking::MetricType metric = fastbev::tracking::MetricType::GIOU_3D;
    if (cfg.track_metric_str == "dist_3d") metric = fastbev::tracking::MetricType::DIST_3D;
    else if (cfg.track_metric_str == "iou_3d") metric = fastbev::tracking::MetricType::IOU_3D;
    else if (cfg.track_metric_str == "mahalanobis") metric = fastbev::tracking::MetricType::MAHALANOBIS;

    fastbev::tracking::AlgoType algo;
    if (cfg.track_algo_str == "hungarian") algo = fastbev::tracking::AlgoType::HUNGARIAN;
    else algo = fastbev::tracking::AlgoType::GREEDY;

    fastbev::tracking::TrackerConfig tracker_cfg;
    tracker_cfg.threshold       = cfg.track_threshold;
    tracker_cfg.metric          = metric;
    tracker_cfg.max_lost_frames = cfg.track_max_lost;
    tracker_cfg.dt              = cfg.track_dt;
    tracker_cfg.algo            = algo;
    tracker_cfg.min_hits        = cfg.track_min_hits;
    tracker_cfg.enable_ego_comp = cfg.track_ego_comp;

    fastbev::tracking::Tracker tracker(tracker_cfg);

    // warmup
    if (cfg.do_warmup) {
      std::string first_root = resolve_data_root(frame_dirs[0]);
      auto imgs = load_images(first_root);
      auto vc = nv::Tensor::load(nv::format("%s/valid_c_idx.tensor", first_root.c_str()), false);
      auto vx = nv::Tensor::load(nv::format("%s/x.tensor",           first_root.c_str()), false);
      auto vy = nv::Tensor::load(nv::format("%s/y.tensor",           first_root.c_str()), false);
      if (!vc.empty()) core->update(vc.ptr<float>(), vx.ptr<int64_t>(), vy.ptr<int64_t>(), stream);
      core->forward((const unsigned char**)imgs.data(), stream);
      free_images(imgs);
      printf("warmup 完成\n");
    }

    int total_boxes = 0;

    for (const auto& fdir : frame_dirs) {
      std::string data_root = resolve_data_root(fdir);
      std::string frame_name = fdir.substr(fdir.rfind('/') + 1);
      std::string out_path   = fdir + "/result" + ext;

      // 读取当前帧的 ego 全局位姿（用于 ego 运动补偿）
      fastbev::tracking::EgoPose ego_pose = read_ego_pose(fdir);
      // 读取真实时间戳（秒）
      double timestamp = read_timestamp(fdir);

      // 执行推理
      auto images = load_images(data_root);
      bool any_valid = false;
      for (auto* img : images) if (img) { any_valid = true; break; }
      if (!any_valid) {
        fprintf(stderr, "[错误] 无法加载图像，跳过: %s\n", data_root.c_str());
        free_images(images);
        continue;
      }

      auto valid_c_idx = nv::Tensor::load(nv::format("%s/valid_c_idx.tensor", data_root.c_str()), false);
      auto valid_x     = nv::Tensor::load(nv::format("%s/x.tensor",           data_root.c_str()), false);
      auto valid_y     = nv::Tensor::load(nv::format("%s/y.tensor",           data_root.c_str()), false);
      if (valid_c_idx.empty() || valid_x.empty() || valid_y.empty()) {
        fprintf(stderr, "[错误] 无法加载几何张量: %s\n", data_root.c_str());
        free_images(images);
        continue;
      }

      core->update(valid_c_idx.ptr<float>(), valid_x.ptr<int64_t>(), valid_y.ptr<int64_t>(), stream);
      auto all_boxes = core->forward((const unsigned char**)images.data(), stream);
      auto boxes     = filter_boxes(all_boxes, cfg.score_thr, cfg.class_filter);
      free_images(images);

      // 保存检测结果（与原代码相同）
      save_boxes(boxes, out_path, cfg.output_format);
      total_boxes += boxes.size();

      // ========== tracking：将检测框转换为跟踪模块需要的 Detection 格式 ==========
      std::vector<fastbev::tracking::Detection> detections;
      for (const auto& box : boxes) {
          fastbev::tracking::Detection det;
          det.x      = box.position.x;
          det.y      = box.position.y;
          det.z      = box.position.z;
          det.w      = box.size.w;
          det.l      = box.size.l;
          det.h      = box.size.h;
          det.yaw    = box.z_rotation;
          det.vx     = box.velocity.vx;
          det.vy     = box.velocity.vy;
          det.score  = box.score;
          det.class_id = box.id;
          detections.push_back(det);
      }

      // 更新跟踪器，得到轨迹（传入真实时间戳和 ego 位姿）
      auto tracks = tracker.update(detections, timestamp, ego_pose);

      // 为每一帧保存轨迹到单独文件（例如 result_tracks.json）
      if (cfg.output_format == "json") {
          std::string track_path = fdir + "/tracks.json";
          std::ofstream ofs(track_path);
          ofs << "[\n";
          for (size_t i = 0; i < tracks.size(); ++i) {
              const auto& trk = tracks[i];
              ofs << "  {"
                  << "\"track_id\": " << trk.track_id << ", "
                  << "\"position\": [" << trk.x << ", " << trk.y << ", " << trk.z << "], "
                  << "\"global_position\": [" << trk.global_x << ", " << trk.global_y << ", " << trk.z << "], "
                  << "\"size\": [" << trk.w << ", " << trk.l << ", " << trk.h << "], "
                  << "\"yaw\": " << trk.yaw << ", "
                  << "\"global_yaw\": " << trk.global_yaw << ", "
                  << "\"velocity\": [" << trk.vx << ", " << trk.vy << "], "
                  << "\"global_velocity\": [" << trk.global_vx << ", " << trk.global_vy << "]"
                  << "}";
              if (i != tracks.size() - 1) ofs << ",";
              ofs << "\n";
          }
          ofs << "]\n";
          ofs.close();
      }
    }

    printf("批量完成: 共 %zu 帧，总检测框 %d 个\n", frame_dirs.size(), total_boxes);
  } else {
    // ── 单帧模式 ──
    std::string data_root = resolve_data_root(cfg.data_dir);
    std::string out_dir   = nv::format("model/%s", cfg.model.c_str());
    std::string out_path  = out_dir + "/result" + ext;

    // 创建输出目录
    mkdir(out_dir.c_str(), 0755);

    // warmup
    if (cfg.do_warmup) {
      core->set_timer(false);
      auto imgs = load_images(data_root);
      auto vc = nv::Tensor::load(nv::format("%s/valid_c_idx.tensor", data_root.c_str()), false);
      auto vx = nv::Tensor::load(nv::format("%s/x.tensor",           data_root.c_str()), false);
      auto vy = nv::Tensor::load(nv::format("%s/y.tensor",           data_root.c_str()), false);
      if (!vc.empty()) core->update(vc.ptr<float>(), vx.ptr<int64_t>(), vy.ptr<int64_t>(), stream);
      for (int i = 0; i < 3; ++i)
        core->forward((const unsigned char**)imgs.data(), stream);
      free_images(imgs);
      printf("warmup 完成\n");
    }

    core->set_timer(true);
    run_single_frame(core, data_root, out_path, stream, cfg);
  }

  checkRuntime(cudaStreamDestroy(stream));
  return 0;
}