
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

// ─── 过滤函数 ──────────────────────────────────────────────────────────────

static std::vector<fastbev::post::transbbox::BoundingBox> filter_boxes(
    const std::vector<fastbev::post::transbbox::BoundingBox>& boxes,
    float score_thr,
    const std::set<int>& class_filter)
{
  std::vector<fastbev::post::transbbox::BoundingBox> result;
  for (const auto& box : boxes) {
    if (box.score < score_thr) continue;
    if (!class_filter.empty() && class_filter.find(box.id) == class_filter.end()) continue;
    result.push_back(box);
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
  printf("  [JSON] 已保存 %zu 个检测框 → %s\n", boxes.size(), file_path.c_str());
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
    fastbev::tracking::TrackerConfig tracker_cfg;
    tracker_cfg.max_dist = 3.0f;          // 匹配距离阈值（米）
    tracker_cfg.max_lost_frames = 3;      // 丢失帧数上限
    // 根据你实际 tracking 模块的配置补充其他参数（例如 min_hits 等）
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
    double timestamp = 0.0;
    double fps = 6.0;   // 可以根据实际帧率调整，或从数据中读取

    for (const auto& fdir : frame_dirs) {
      std::string data_root = resolve_data_root(fdir);
      std::string frame_name = fdir.substr(fdir.rfind('/') + 1);
      std::string out_path   = fdir + "/result" + ext;

      // 执行推理（与原代码相同）
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
        // 根据你的 tracking 模块实际定义赋值（请参考 track.hpp / tracker.hpp）
        det.position = {box.position.x, box.position.y, box.position.z};
        det.size     = {box.size.w, box.size.l, box.size.h};
        det.yaw      = box.z_rotation;
        det.label    = box.id;
        det.score    = box.score;
        det.velocity = {box.velocity.vx, box.velocity.vy};
        // 如果 Detection 需要 timestamp，则添加
        // det.timestamp = timestamp;
        detections.push_back(det);
      }

      // 更新跟踪器，得到轨迹
      auto tracks = tracker.update(detections, timestamp);

      // 可选：输出跟踪结果（打印或保存到文件）
      printf("帧 %s: 检测框 %zu, 跟踪轨迹 %zu\n", frame_name.c_str(), boxes.size(), tracks.size());

      // 为每一帧保存轨迹到单独文件（例如 result_tracks.json）
      if (cfg.output_format == "json") {
        std::string track_path = fdir + "/tracks.json";
        std::ofstream ofs(track_path);
        // 简单输出，实际应按 JSON 格式
        ofs << "[\n";
        for (size_t i = 0; i < tracks.size(); ++i) {
          const auto& trk = tracks[i];
          ofs << "  {\"track_id\": " << trk.id << ", \"position\": [" 
              << trk.position.x << "," << trk.position.y << "," << trk.position.z << "]}\n";
          if (i != tracks.size()-1) ofs << ",";
        }
        ofs << "]\n";
        ofs.close();
      }

      timestamp += 1.0 / fps;
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

