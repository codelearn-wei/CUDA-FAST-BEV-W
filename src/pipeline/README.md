# Pipeline 模块

> 路径：`src/pipeline/`

---

## 概述

本模块将检测、跟踪、输出整合为一条**完整感知管线**，对上层（main.cpp / ROS 节点）暴露单一的 `PerceptionPipeline::process()` 接口。

---

## 文件说明

| 文件 | 职责 |
|------|------|
| `perception_pipeline.hpp` | 管线接口声明、配置结构体、感知结果定义 |
| `perception_pipeline.cpp` | 管线实现：TRT 初始化、推理、过滤、跟踪、回调 |

---

## 数据流

```
CameraFrame（来自 camera 模块或 ROS）
        ↓
PerceptionPipeline::process(frame)
        ├── Core::update(valid_c_idx, x, y)    ← 几何张量
        ├── Core::forward(images)              ← TRT 推理
        ├── filter_and_convert()               ← 阈值 + 类别过滤
        ├── Tracker::update(detections)        ← 多目标跟踪
        └── callback(result)                   ← 结果回调
                ↓
        PerceptionResult {
            detections[]   ← 过滤后的 BEV 检测
            tracks[]       ← 带 track_id 的轨迹
            latency_ms     ← 端到端耗时
        }
```

---

## 用法

### 最简使用

```cpp
#include "pipeline/perception_pipeline.hpp"

fastbev::pipeline::PipelineConfig cfg;
cfg.model_name       = "resnet18";
cfg.score_thr        = 0.4f;
cfg.enable_tracking  = true;

fastbev::pipeline::PerceptionPipeline pipeline(cfg);
pipeline.init();

// 注册结果回调（可选）
pipeline.set_callback([](const fastbev::pipeline::PerceptionResult& r) {
    printf("frame=%lu  dets=%zu  tracks=%zu\n",
           r.frame_id, r.detections.size(), r.tracks.size());
});

// 每帧处理
auto frame = fastbev::camera::FrameLoader::load_from_dir("outputs/frames/frame_00000");
auto result = pipeline.process(*frame);
```

### 跨场景使用

```cpp
// 切换场景时重置跟踪器，防止 track_id 跨场景污染
pipeline.reset_tracker();
```

---

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | string | resnet18 | TRT 模型名 |
| `score_thr` | float | 0.4 | 置信度阈值 |
| `class_filter` | vector\<int\> | 空（全部） | 类别过滤 |
| `enable_tracking` | bool | true | 是否启用跟踪 |
| `track_max_dist` | float | 3.0m | 匹配距离阈值 |
| `track_max_lost` | int | 3 | 最大丢失帧数 |
| `track_min_hits` | int | 2 | 轨迹确认所需匹配次数 |
| `track_dt` | double | 0.05s | 帧间隔 |
| `verbose` | bool | false | 打印每帧日志 |

---

## PerceptionResult 结构

```cpp
struct PerceptionResult {
    uint64_t frame_id;       // 单调递增帧序号
    double   timestamp;      // UNIX 时间戳（秒）
    double   latency_ms;     // 推理耗时

    // 过滤后的 BEV 检测
    std::vector<Detection> detections;

    // 跟踪轨迹（含历史）
    std::vector<Track>     tracks;
};
```

---

## 轨迹输出格式（JSON）

```json
{
  "frame_id": 42,
  "timestamp": 1234567890.123,
  "latency_ms": 15.3,
  "tracks": [
    {
      "track_id": 7,
      "class_id": 0,
      "class_name": "car",
      "state": "active",
      "x": 12.3, "y": -1.5, "z": 0.8,
      "vx": 3.2, "vy": 0.1,
      "yaw": -0.12,
      "w": 4.2, "l": 1.9, "h": 1.6,
      "score": 0.87,
      "history": [[12.1,-1.4,0.8], [12.2,-1.45,0.8], ...]
    }
  ]
}
```

---

## 与 ROS 节点集成

在 `ros/fastbev_ros_node.cpp` 中使用管线：

```cpp
PerceptionPipeline pipeline(cfg);
pipeline.set_callback([this](const PerceptionResult& r) {
    publish_markers(r.tracks);
    publish_json(r);
});

// 在相机回调中：
pipeline.process(frame, stream_);
```

---

## 扩展方向

1. **多线程流水线**：将数据加载、推理、跟踪分为独立线程，通过 `MultiCameraBuffer` 解耦
2. **性能统计**：记录 P50/P99 延迟，帧率统计
3. **结果持久化**：将 `PerceptionResult` 序列化为 JSON 文件（配合 `video_demo.py` 使用）
4. **参数热更新**：通过 ROS `dynamic_reconfigure` 或文件监听动态调整阈值
