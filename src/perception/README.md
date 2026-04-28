# Perception 模块

> 路径：`src/perception/`

---

## 概述

本模块提供**联合感知**的公共接口层，将 BEV 目标检测（C++/TensorRT）和 MapTR HD 地图推理（Python subprocess）统一纳入同一数据流，对上层（`joint_inference.cpp` / ROS 节点）暴露简洁的输入/输出类型。

---

## 文件说明

| 文件 | 职责 |
|------|------|
| `perception_types.hpp` | 公共数据类型：`RawImageInput`、`MapResult`、`JointPerceptionResult` |
| `image_preprocessor.hpp/cpp` | 图像预处理接口：帧目录加载 / 动态几何张量计算 / 相机驱动预留接口 |
| `map_runner.hpp/cpp` | MapTR C++ 包装器：通过 subprocess 调用 Python 推理，解析 `map_result.json` |

---

## 数据流

```
数据来源
  ├── 帧目录（from_frame_dir）          ← nuscenes_adapter.py 生成
  ├── 原始图像 + 标定（from_raw_input） ← 在线场景
  └── 相机驱动（from_camera）           ← 预留接口，待实现
          ↓
    ImagePreprocessor
          ├── camera::CameraFrame ──────→ PerceptionPipeline（BEV + 跟踪）
          │       valid_c_idx / x / y          ↓
          │                          tracking::Track[]
          │
          └── RawImageInput ────────→ MapRunner（Python subprocess）
                  frame_dir                    ↓
                  ego_pose              map_result.json
                                               ↓
                                        MapResult
          ↓                                    ↓
                    JointPerceptionResult
                        detections[]
                        tracks[]
                        map.elements[]
```

---

## 核心类型

### `RawImageInput`

```cpp
#include "perception/perception_types.hpp"

fastbev::perception::RawImageInput input;
input.frame_id   = 42;
input.timestamp  = 1.623e9;
input.source_tag = "nuscenes";
input.frame_dir  = "outputs/frames/frame_00042";
// images_ptr[0..5] 由调用方管理生命周期
// intrinsics / extrinsics / ego_pose 由标定系统填充
```

### `MapResult`

```cpp
fastbev::perception::MapResult map;
// map.source    — "model" | "gt" | "trajectory"
// map.elements  — vector<MapElement>
// map.raw_json  — 原始 JSON 字符串（透传给 Python 可视化）
```

### `JointPerceptionResult`

```cpp
fastbev::perception::JointPerceptionResult result;
// result.detections     — 过滤后 BEV 检测框
// result.tracks         — 带 track_id 的跟踪结果
// result.bev_latency_ms — BEV 推理延迟
// result.map            — MapResult
// result.map_latency_ms — 地图推理延迟（读取 JSON）
```

---

## 图像预处理接口

### 1. 从帧目录加载（最常用）

```cpp
#include "perception/image_preprocessor.hpp"

camera::CameraFrame bev_frame;
fastbev::perception::RawImageInput map_input;

bool ok = fastbev::perception::ImagePreprocessor::from_frame_dir(
    "outputs/frames/frame_00000", /*frame_id=*/0,
    bev_frame, map_input);

// bev_frame.owns_tensors == true → 需要释放
camera::FrameLoader::free_frame(bev_frame);
```

### 2. 从原始图像 + 标定动态计算（在线场景）

```cpp
fastbev::perception::RawImageInput input;
input.images_ptr[0] = rgb_ptr_front;   // unsigned char* RGB
// 填充 intrinsics / extrinsics ...
input.ego_pose = { .x=1.2, .y=3.4, .yaw=0.5, .valid=true };

fastbev::perception::GeometryParams geo;  // 默认参数与训练一致
camera::CameraFrame bev_frame;
bool ok = fastbev::perception::ImagePreprocessor::from_raw_input(
    input, geo, bev_frame);
```

### 3. 相机驱动接口（预留）

```cpp
// 接口已声明，待与具体相机 SDK 集成后实现
// bool ok = fastbev::perception::ImagePreprocessor::from_camera(
//     camera_handle, "config/calibration.json", input);
```

---

## MapRunner

```cpp
#include "perception/map_runner.hpp"

fastbev::perception::MapRunnerConfig mcfg;
mcfg.mode         = "gt";           // "auto" | "gt" | "model" | "trajectory"
mcfg.frames_dir   = "outputs/frames";
mcfg.nuscenes_dir = "data/nuscenes";

fastbev::perception::MapRunner runner(mcfg);

// 批量推理（将 map_result.json 写入每帧目录）
int n = runner.run_batch(/*overwrite=*/false);

// 读取单帧结果
fastbev::perception::MapResult result;
runner.read_result("outputs/frames/frame_00000", result);
printf("地图来源: %s，元素数: %zu\n",
       result.source.c_str(), result.elements.size());
```

---

## 几何张量计算（`from_raw_input` 内部）

对应 Python `tools/nuscenes_adapter.py::compute_geometry_tensors()`，纯 C++ 实现，无 Python 依赖：

1. 生成体素网格 [200×200×4] 的 3D 坐标（与训练时体素参数完全一致）
2. 对每个相机构造特征图尺度投影矩阵 `P = K_feat @ E[:3, :]`
3. 将体素点投影到特征图，保留 `z > 0` 且在边界内的点
4. 输出 `valid_c_idx / valid_x / valid_y` 三个张量

参数（`GeometryParams`）：

| 参数 | 值 | 说明 |
|------|----|------|
| `nx/ny/nz` | 200/200/4 | 体素网格尺寸 |
| `voxel_size` | 0.5/0.5/1.5 m | 体素边长 |
| `feat_height/width` | 64/176 | 特征图尺寸（256×704 ÷ 4）|
| `feat_stride` | 4 | 下采样倍率 |

---

## 相机接口扩展指南

实现 `from_camera()` 时需要补全以下逻辑（参考注释）：

```cpp
// src/perception/image_preprocessor.cpp
bool ImagePreprocessor::from_camera(
    const void* camera_handle,
    const std::string& calib_file,
    RawImageInput& output)
{
    // TODO 1: 从 camera_handle（SDK 句柄）获取 6 路同步帧
    //         典型: SDK::grab_sync(handle, &frames)
    
    // TODO 2: 将 YUV/BGR → RGB 格式转换
    //         cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB)
    
    // TODO 3: 从 calib_file（JSON）加载内外参
    //         camera::CalibrationLoader::load_from_json(calib_file, calib)
    
    // TODO 4: 从 GNSS/IMU 读取 ego 位姿
    //         output.ego_pose = imu_driver->get_pose()
    
    return true;
}
```
