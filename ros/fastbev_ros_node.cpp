/*
 * CUDA-FastBEV ROS Noetic 节点
 *
 * 订阅 6 路相机 Image 话题，同步后调用 FastBEV 推理，
 * 发布检测结果（MarkerArray + 自定义 BoundingBox3DArray 消息）。
 *
 * 话题（可通过参数重映射）:
 *   订阅:
 *     /cam_front/image_raw           (sensor_msgs/Image)
 *     /cam_front_right/image_raw
 *     /cam_front_left/image_raw
 *     /cam_back/image_raw
 *     /cam_back_left/image_raw
 *     /cam_back_right/image_raw
 *     /fastbev/geometry              (std_msgs/String, JSON 格式的几何张量路径)
 *
 *   发布:
 *     /fastbev/detections            (visualization_msgs/MarkerArray)
 *     /fastbev/detections_info       (std_msgs/String, JSON 格式)
 *
 * 参数:
 *   ~model         (string, default: "resnet18")
 *   ~score_thr     (double, default: 0.5)
 *   ~class_filter  (string, default: "", 逗号分隔 id)
 *   ~geometry_dir  (string, 包含 valid_c_idx.tensor 等的目录)
 *   ~output_format (string, default: "json")
 *
 * 编译: 参考 ros/CMakeLists.txt
 */

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <sstream>
#include <fstream>
#include <set>
#include <cmath>

// FastBEV 核心头文件（通过 include 路径引入）
#include "fastbev/fastbev.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"

namespace fastbev_ros {

// ─── NuScenes 类别颜色（RGB） ──────────────────────────────────────────────
static const char* CLASS_NAMES[] = {
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
};
static const float CLASS_COLORS[][3] = {
    {1.0f, 0.62f, 0.00f},  // car
    {0.27f, 0.67f, 1.00f}, // truck
    {0.00f, 0.78f, 1.00f}, // construction_vehicle
    {0.16f, 0.47f, 1.00f}, // bus
    {0.71f, 0.43f, 1.00f}, // trailer
    {0.71f, 0.71f, 0.71f}, // barrier
    {0.00f, 0.86f, 0.47f}, // motorcycle
    {0.16f, 1.00f, 0.78f}, // bicycle
    {1.00f, 0.35f, 0.71f}, // pedestrian
    {0.63f, 0.78f, 0.31f}, // traffic_cone
};
static const int NUM_CLASSES = 10;

// ─── 同步策略（6 路相机，近似时间同步） ────────────────────────────────────
using ImageMsg  = sensor_msgs::Image;
using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    ImageMsg, ImageMsg, ImageMsg, ImageMsg, ImageMsg, ImageMsg>;

// ─── ROS 节点类 ────────────────────────────────────────────────────────────
class FastBEVNode {
 public:
  FastBEVNode() : nh_("~"), geometry_loaded_(false) {
    // 读取参数
    nh_.param<std::string>("model",        model_,        "resnet18");
    nh_.param<double>("score_thr",         score_thr_,     0.5);
    nh_.param<std::string>("class_filter", class_filter_str_, "");
    nh_.param<std::string>("geometry_dir", geometry_dir_,  "");
    nh_.param<std::string>("output_format",output_format_, "json");
    nh_.param<int>("queue_size",           queue_size_,    5);
    nh_.param<double>("sync_slop",         sync_slop_,     0.1);

    // 解析类别过滤
    if (!class_filter_str_.empty()) {
      std::stringstream ss(class_filter_str_);
      std::string token;
      while (std::getline(ss, token, ',')) {
        try { class_filter_.insert(std::stoi(token)); }
        catch (...) {}
      }
    }

    ROS_INFO("[FastBEV] 模型: %s  阈值: %.2f", model_.c_str(), score_thr_);

    // 初始化推理核心
    if (!init_core()) {
      ROS_FATAL("[FastBEV] 推理核心初始化失败，退出");
      ros::shutdown();
      return;
    }

    // 预加载几何张量（若已指定目录）
    if (!geometry_dir_.empty()) {
      load_geometry(geometry_dir_);
    }

    // 发布者
    pub_markers_ = nh_.advertise<visualization_msgs::MarkerArray>(
        "/fastbev/detections", 1);
    pub_json_    = nh_.advertise<std_msgs::String>(
        "/fastbev/detections_info", 1);

    // 相机话题
    std::vector<std::string> cam_topics = {
        "/cam_front/image_raw",
        "/cam_front_right/image_raw",
        "/cam_front_left/image_raw",
        "/cam_back/image_raw",
        "/cam_back_left/image_raw",
        "/cam_back_right/image_raw",
    };
    // 允许通过参数覆盖话题名
    for (int i = 0; i < 6; ++i) {
      std::string key = std::string("cam_topic_") + std::to_string(i);
      nh_.param<std::string>(key, cam_topics[i], cam_topics[i]);
    }

    for (int i = 0; i < 6; ++i) {
      subs_[i].reset(new message_filters::Subscriber<ImageMsg>(
          nh_, cam_topics[i], queue_size_));
      ROS_INFO("[FastBEV] 订阅相机 %d: %s", i, cam_topics[i].c_str());
    }

    sync_.reset(new message_filters::Synchronizer<SyncPolicy>(
        SyncPolicy(queue_size_), *subs_[0], *subs_[1], *subs_[2],
                                 *subs_[3], *subs_[4], *subs_[5]));
    sync_->setMaxIntervalDuration(ros::Duration(sync_slop_));
    sync_->registerCallback(
        boost::bind(&FastBEVNode::camera_callback, this,
                    _1, _2, _3, _4, _5, _6));

    // 可选：接收动态几何更新
    geo_sub_ = nh_.subscribe("/fastbev/geometry_dir", 1,
                              &FastBEVNode::geometry_callback, this);

    ROS_INFO("[FastBEV] 节点就绪，等待相机数据...");
  }

  ~FastBEVNode() {
    if (stream_) cudaStreamDestroy(stream_);
  }

 private:
  // ── 模型初始化 ──────────────────────────────────────────────────────────
  bool init_core() {
    checkRuntime(cudaStreamCreate(&stream_));

    fastbev::pre::NormalizationParameter norm;
    norm.image_width   = 1600;
    norm.image_height  = 900;
    norm.output_width  = 704;
    norm.output_height = 256;
    norm.num_camera    = 6;
    norm.resize_lim    = 0.44f;
    norm.interpolation = fastbev::pre::Interpolation::Nearest;

    float mean[3] = {123.675f, 116.28f, 103.53f};
    float std_v[3] = { 58.395f,  57.12f,  57.375f};
    norm.method = fastbev::pre::NormMethod::mean_std(mean, std_v, 1.0f, 0.0f);

    fastbev::pre::GeometryParameter geo;
    geo.feat_height  = 64;
    geo.feat_width   = 176;
    geo.num_camera   = 6;
    geo.valid_points = 160000;
    geo.volum_x      = 200;
    geo.volum_y      = 200;
    geo.volum_z      = 4;

    fastbev::CoreParameter param;
    param.pre_model  = "model/" + model_ + "/build/fastbev_pre_trt.plan";
    param.post_model = "model/" + model_ + "/build/fastbev_post_trt_decode.plan";
    param.normalize  = norm;
    param.geo_param  = geo;

    core_ = fastbev::create_core(param);
    if (!core_) {
      ROS_ERROR("[FastBEV] 无法创建推理核心（检查 TRT plan 文件路径）");
      return false;
    }
    core_->print();
    core_->set_timer(false);

    // warmup（使用零图像）
    std::vector<unsigned char*> dummy(6, nullptr);
    std::vector<std::vector<unsigned char>> dummy_bufs(6,
        std::vector<unsigned char>(704 * 256 * 3, 0));
    for (int i = 0; i < 6; ++i) dummy[i] = dummy_bufs[i].data();
    // 注意：只有在几何张量已加载后才能 forward，这里仅做内存预热
    ROS_INFO("[FastBEV] 推理核心已创建");
    return true;
  }

  // ── 几何张量加载 ─────────────────────────────────────────────────────────
  bool load_geometry(const std::string& dir) {
    auto vc = nv::Tensor::load(dir + "/valid_c_idx.tensor", false);
    auto vx = nv::Tensor::load(dir + "/x.tensor",           false);
    auto vy = nv::Tensor::load(dir + "/y.tensor",           false);

    if (vc.empty() || vx.empty() || vy.empty()) {
      ROS_WARN("[FastBEV] 几何张量加载失败: %s", dir.c_str());
      return false;
    }
    core_->update(vc.ptr<float>(), vx.ptr<int64_t>(), vy.ptr<int64_t>(), stream_);
    geometry_loaded_ = true;
    ROS_INFO("[FastBEV] 几何张量已加载: %s", dir.c_str());
    return true;
  }

  // ── 几何目录话题回调 ─────────────────────────────────────────────────────
  void geometry_callback(const std_msgs::String::ConstPtr& msg) {
    load_geometry(msg->data);
  }

  // ── 相机同步回调 ─────────────────────────────────────────────────────────
  void camera_callback(
      const ImageMsg::ConstPtr& m0, const ImageMsg::ConstPtr& m1,
      const ImageMsg::ConstPtr& m2, const ImageMsg::ConstPtr& m3,
      const ImageMsg::ConstPtr& m4, const ImageMsg::ConstPtr& m5)
  {
    if (!geometry_loaded_) {
      ROS_WARN_THROTTLE(5.0, "[FastBEV] 几何张量未加载，跳过推理");
      return;
    }

    ros::Time t_start = ros::Time::now();

    // 解码图像
    const ImageMsg::ConstPtr msgs[6] = {m0, m1, m2, m3, m4, m5};
    std::vector<cv::Mat> imgs(6);
    std::vector<unsigned char*> raw_ptrs(6, nullptr);
    bool ok = true;

    for (int i = 0; i < 6; ++i) {
      try {
        cv_bridge::CvImageConstPtr cv_ptr =
            cv_bridge::toCvShare(msgs[i], sensor_msgs::image_encodings::RGB8);
        imgs[i] = cv_ptr->image.clone();
        raw_ptrs[i] = imgs[i].data;
      } catch (cv_bridge::Exception& e) {
        ROS_WARN("[FastBEV] 图像解码错误 cam%d: %s", i, e.what());
        ok = false;
        break;
      }
    }
    if (!ok) return;

    // 推理
    auto all_boxes = core_->forward(
        const_cast<const unsigned char**>(raw_ptrs.data()), stream_);

    // 过滤
    std::vector<fastbev::post::transbbox::BoundingBox> boxes;
    for (const auto& b : all_boxes) {
      if (b.score < static_cast<float>(score_thr_)) continue;
      if (!class_filter_.empty() &&
          class_filter_.find(b.id) == class_filter_.end()) continue;
      boxes.push_back(b);
    }

    double latency_ms = (ros::Time::now() - t_start).toSec() * 1000.0;
    ROS_INFO_THROTTLE(1.0, "[FastBEV] 检测框: %zu  延迟: %.1f ms",
                      boxes.size(), latency_ms);

    // 发布 MarkerArray
    if (pub_markers_.getNumSubscribers() > 0) {
      publish_markers(boxes, msgs[0]->header.stamp);
    }

    // 发布 JSON
    if (pub_json_.getNumSubscribers() > 0) {
      publish_json(boxes, msgs[0]->header.stamp, latency_ms);
    }
  }

  // ── 发布 MarkerArray ─────────────────────────────────────────────────────
  void publish_markers(
      const std::vector<fastbev::post::transbbox::BoundingBox>& boxes,
      const ros::Time& stamp)
  {
    visualization_msgs::MarkerArray marker_arr;

    // 清除旧标记
    visualization_msgs::Marker del;
    del.action = visualization_msgs::Marker::DELETEALL;
    del.header.stamp = stamp;
    del.header.frame_id = "base_link";
    marker_arr.markers.push_back(del);

    for (size_t i = 0; i < boxes.size(); ++i) {
      const auto& b  = boxes[i];
      int  label_id  = (b.id >= 0 && b.id < NUM_CLASSES) ? b.id : 0;
      const float* c = CLASS_COLORS[label_id];

      // 3D bbox 线框
      visualization_msgs::Marker m;
      m.header.stamp    = stamp;
      m.header.frame_id = "base_link";
      m.ns      = "fastbev_boxes";
      m.id      = static_cast<int>(i);
      m.type    = visualization_msgs::Marker::LINE_LIST;
      m.action  = visualization_msgs::Marker::ADD;
      m.lifetime= ros::Duration(0.2);
      m.scale.x = 0.08;
      m.color.r = c[0]; m.color.g = c[1]; m.color.b = c[2]; m.color.a = 1.0f;

      // 8 个角点 (lidar 坐标系)
      float w = b.size.w, l = b.size.l, h = b.size.h;
      float px = b.position.x, py = b.position.y, pz = b.position.z;
      float cy = std::cos(b.z_rotation), sy = std::sin(b.z_rotation);

      float dx[4] = { w/2, -w/2, -w/2,  w/2};
      float dy[4] = { l/2,  l/2, -l/2, -l/2};

      geometry_msgs::Point corners[8];
      for (int k = 0; k < 4; ++k) {
        float rx = dx[k] * cy + dy[k] * sy;
        float ry = dx[k] * -sy + dy[k] * cy;
        corners[k].x   = px + rx; corners[k].y   = py + ry; corners[k].z   = pz;
        corners[k+4].x = px + rx; corners[k+4].y = py + ry; corners[k+4].z = pz + h;
      }
      // 底面 + 顶面 + 竖边
      int edges[12][2] = {
          {0,1},{1,2},{2,3},{3,0},
          {4,5},{5,6},{6,7},{7,4},
          {0,4},{1,5},{2,6},{3,7}};
      for (auto& e : edges) {
        m.points.push_back(corners[e[0]]);
        m.points.push_back(corners[e[1]]);
      }
      marker_arr.markers.push_back(m);

      // 文字标签
      visualization_msgs::Marker text_m;
      text_m.header    = m.header;
      text_m.ns        = "fastbev_labels";
      text_m.id        = static_cast<int>(i) + 10000;
      text_m.type      = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text_m.action    = visualization_msgs::Marker::ADD;
      text_m.lifetime  = ros::Duration(0.2);
      text_m.scale.z   = 0.4;
      text_m.color.r   = c[0]; text_m.color.g = c[1];
      text_m.color.b   = c[2]; text_m.color.a = 1.0f;
      text_m.pose.position.x = px;
      text_m.pose.position.y = py;
      text_m.pose.position.z = pz + h + 0.3f;
      text_m.pose.orientation.w = 1.0;
      char buf[64];
      snprintf(buf, sizeof(buf), "%s %.2f", CLASS_NAMES[label_id], b.score);
      text_m.text = buf;
      marker_arr.markers.push_back(text_m);
    }

    pub_markers_.publish(marker_arr);
  }

  // ── 发布 JSON 字符串 ──────────────────────────────────────────────────────
  void publish_json(
      const std::vector<fastbev::post::transbbox::BoundingBox>& boxes,
      const ros::Time& stamp, double latency_ms)
  {
    std::ostringstream oss;
    oss << "{\"timestamp\":" << stamp.toSec()
        << ",\"latency_ms\":" << latency_ms
        << ",\"detections\":[\n";
    for (size_t i = 0; i < boxes.size(); ++i) {
      const auto& b = boxes[i];
      int label_id  = (b.id >= 0 && b.id < NUM_CLASSES) ? b.id : 0;
      oss << "  {\"label\":" << b.id
          << ",\"label_name\":\"" << CLASS_NAMES[label_id] << "\""
          << ",\"score\":"   << b.score
          << ",\"center_xyz\":[" << b.position.x << "," << b.position.y << "," << b.position.z << "]"
          << ",\"size_xyz\":["  << b.size.w      << "," << b.size.l      << "," << b.size.h      << "]"
          << ",\"yaw\":"        << b.z_rotation
          << ",\"velocity_xy\":[" << b.velocity.vx << "," << b.velocity.vy << "]}";
      if (i + 1 < boxes.size()) oss << ",";
      oss << "\n";
    }
    oss << "]}";
    std_msgs::String msg;
    msg.data = oss.str();
    pub_json_.publish(msg);
  }

  // ── 成员变量 ──────────────────────────────────────────────────────────────
  ros::NodeHandle nh_;
  std::string     model_;
  double          score_thr_;
  std::string     class_filter_str_;
  std::string     geometry_dir_;
  std::string     output_format_;
  int             queue_size_;
  double          sync_slop_;
  std::set<int>   class_filter_;

  std::shared_ptr<fastbev::Core> core_;
  cudaStream_t stream_ = nullptr;
  bool geometry_loaded_;

  std::unique_ptr<message_filters::Subscriber<ImageMsg>> subs_[6];
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  ros::Publisher  pub_markers_;
  ros::Publisher  pub_json_;
  ros::Subscriber geo_sub_;
};

}  // namespace fastbev_ros

// ─── main ─────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
  ros::init(argc, argv, "fastbev_ros_node");
  fastbev_ros::FastBEVNode node;
  ros::spin();
  return 0;
}
