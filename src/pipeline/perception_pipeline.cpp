/**
 * perception_pipeline.cpp — 完整感知管线实现
 */

#include "perception_pipeline.hpp"
#include "../common/timer.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace fastbev {
namespace pipeline {

// ─── 构造 ─────────────────────────────────────────────────────────────────

PerceptionPipeline::PerceptionPipeline(const PipelineConfig& cfg)
    : cfg_(cfg)
{
    if (cfg_.enable_tracking) {
        tracking::TrackerConfig tc;
        tc.max_dist        = cfg_.track_max_dist;
        tc.max_lost_frames = cfg_.track_max_lost;
        tc.min_hits = cfg_.track_min_hits;
        tc.dt              = cfg_.track_dt;
        tracker_ = std::make_unique<tracking::Tracker>(tc);
    }
}

// ─── 初始化 ───────────────────────────────────────────────────────────────

bool PerceptionPipeline::init() {
    if (core_) return true;  // 已初始化

    // 复用 main.cpp 中的初始化逻辑
    pre::NormalizationParameter normalization;
    normalization.image_width   = 1600;
    normalization.image_height  = 900;
    normalization.output_width  = 704;
    normalization.output_height = 256;
    normalization.num_camera    = 6;
    normalization.resize_lim    = 0.44f;
  // 双线性插值与训练预处理对齐
  normalization.interpolation = pre::Interpolation::Bilinear;
    float mean[3]  = {123.675f, 116.28f, 103.53f};
    float std_v[3] = { 58.395f,  57.12f,  57.375f};
    normalization.method = pre::NormMethod::mean_std(mean, std_v, 1.0f, 0.0f);

    pre::GeometryParameter geo_param;
    geo_param.feat_height  = 64;
    geo_param.feat_width   = 176;
    geo_param.num_camera   = 6;
    geo_param.valid_points = 160000;
    geo_param.volum_x      = 200;
    geo_param.volum_y      = 200;
    geo_param.volum_z      = 4;

    CoreParameter param;
    param.pre_model  = nv::format("model/%s/build/fastbev_pre_trt.plan",
                                  cfg_.model_name.c_str());
    param.post_model = nv::format("model/%s/build/fastbev_post_trt_decode.plan",
                                  cfg_.model_name.c_str());
    param.normalize  = normalization;
    param.geo_param  = geo_param;

    core_ = fastbev::create_core(param);
    if (!core_) {
        fprintf(stderr, "[Pipeline] 初始化失败：无法创建推理核心 (%s)\n",
                cfg_.model_name.c_str());
        return false;
    }
    if (cfg_.verbose)
        printf("[Pipeline] 推理核心初始化成功 (%s)\n", cfg_.model_name.c_str());
    return true;
}

// ─── 过滤与转换 ───────────────────────────────────────────────────────────

std::vector<tracking::Detection> PerceptionPipeline::filter_and_convert(
    const std::vector<post::transbbox::BoundingBox>& boxes)
{
    std::vector<tracking::Detection> out;
    for (const auto& box : boxes) {
        if (box.score < cfg_.score_thr) continue;
        if (!cfg_.class_filter.empty()) {
            bool found = false;
            for (int cid : cfg_.class_filter)
                if (cid == box.id) { found = true; break; }
            if (!found) continue;
        }
        tracking::Detection det;
        det.x        = box.position.x;
        det.y        = box.position.y;
        det.z        = box.position.z;
        det.w        = box.size.w;
        det.l        = box.size.l;
        det.h        = box.size.h;
        det.yaw      = box.z_rotation;
        det.vx       = box.velocity.vx;
        det.vy       = box.velocity.vy;
        det.score    = box.score;
        det.class_id = box.id;
        out.push_back(det);
    }
    return out;
}

// ─── 主处理函数 ───────────────────────────────────────────────────────────

PerceptionResult PerceptionPipeline::process(const camera::CameraFrame& frame,
                                              void* stream_ptr)
{
    PerceptionResult result;
    result.frame_id  = frame_counter_++;
    result.timestamp = frame.timestamp;

    if (!core_) {
        fprintf(stderr, "[Pipeline] 未初始化，请先调用 init()\n");
        return result;
    }

    cudaStream_t stream = stream_ptr
        ? static_cast<cudaStream_t>(stream_ptr) : nullptr;
    if (!stream) cudaStreamCreate(&stream);

    // 更新几何张量
    if (frame.valid_c_idx) {
        core_->update(frame.valid_c_idx, frame.valid_x, frame.valid_y, stream);
    }

    // 推理计时
    auto t0 = std::chrono::high_resolution_clock::now();
    auto raw_boxes = core_->forward(
        (const unsigned char**)frame.images.data(), stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 过滤 + 转换
    result.detections = filter_and_convert(raw_boxes);

    // BEV 中心距离 NMS
    if (cfg_.nms_bev_dist > 0.0f && result.detections.size() > 1) {
        float dist_sq_thr = cfg_.nms_bev_dist * cfg_.nms_bev_dist;
        std::sort(result.detections.begin(), result.detections.end(),
                  [](const tracking::Detection& a, const tracking::Detection& b) {
                      return a.score > b.score;
                  });
        std::vector<bool> suppressed(result.detections.size(), false);
        for (size_t i = 0; i < result.detections.size(); ++i) {
            if (suppressed[i]) continue;
            for (size_t j = i + 1; j < result.detections.size(); ++j) {
                if (suppressed[j]) continue;
                if (result.detections[i].class_id != result.detections[j].class_id) continue;
                float dx = result.detections[i].x - result.detections[j].x;
                float dy = result.detections[i].y - result.detections[j].y;
                if (dx * dx + dy * dy < dist_sq_thr)
                    suppressed[j] = true;
            }
        }
        std::vector<tracking::Detection> kept;
        for (size_t i = 0; i < result.detections.size(); ++i)
            if (!suppressed[i]) kept.push_back(result.detections[i]);
        result.detections = std::move(kept);
    }

    // 多目标跟踪
    if (cfg_.enable_tracking && tracker_) {
        result.tracks = tracker_->update(result.detections, frame.timestamp);
    }

    if (cfg_.verbose) {
        printf("[Pipeline] frame=%lu  det=%zu  track=%zu  latency=%.1fms\n",
               result.frame_id, result.detections.size(),
               result.tracks.size(), result.latency_ms);
    }

    // 回调
    if (callback_) callback_(result);

    if (!stream_ptr) cudaStreamDestroy(stream);
    return result;
}

// ─── 重置跟踪器 ───────────────────────────────────────────────────────────

void PerceptionPipeline::reset_tracker() {
    if (tracker_) tracker_->reset();
}

}  // namespace pipeline
}  // namespace fastbev
