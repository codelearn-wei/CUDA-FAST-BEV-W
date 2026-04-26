# Camera 模块

> 路径：`src/camera/`

---

## 概述

本模块负责**多相机输入管理**，为检测推理提供统一的帧数据接口，解耦数据来源（离线目录 / ROS 话题 / 在线相机流）与推理链路。

---

## 文件说明

| 文件 | 职责 |
|------|------|
| `camera_frame.hpp` | 单帧多相机数据容器（图像指针 + 几何张量 + 标定）；`FrameLoader` 从磁盘目录加载 |
| `multi_camera_buffer.hpp` | 线程安全的多相机帧队列（支持在线和离线场景） |
| `calibration.hpp` | 相机内参/外参加载接口（NuScenes / JSON / 自定义格式） |

---

## 数据流

```
磁盘 frame_dir/      ROS /cam_*/image_raw     在线相机 SDK
         ↓                    ↓                      ↓
   FrameLoader::         ROS 节点转换            驱动回调
   load_from_dir()             ↓                      ↓
         └──────────────►  CameraFrame  ◄──────────────┘
                               ↓
                   MultiCameraBuffer::push()
                               ↓
                   MultiCameraBuffer::pop()
                               ↓
                     Pipeline / Tracker
```

---

## CameraFrame 接口

```cpp
CameraFrame frame;
frame.frame_id  = 42;
frame.timestamp = 1234567890.123;
frame.images    = {img0, img1, img2, img3, img4, img5};  // 6 路
frame.valid_c_idx = ptr_float32;  // [6, 160000]
frame.valid_x     = ptr_int64;
frame.valid_y     = ptr_int64;
bool ok = frame.is_valid();
```

### 从磁盘加载

```cpp
auto frame = FrameLoader::load_from_dir("outputs/frames/frame_00000", 0);
if (frame) {
    // 使用 frame->images 等字段
    FrameLoader::free_frame(*frame);
}
```

### 帧目录结构

```
frame_00000/
├── 0-FRONT.jpg
├── 1-FRONT_RIGHT.jpg
├── 2-FRONT_LEFT.jpg
├── 3-BACK.jpg
├── 4-BACK_LEFT.jpg
├── 5-BACK_RIGHT.jpg
├── valid_c_idx.tensor   # [6, 160000] float32
├── x.tensor             # [6, 160000] int64
├── y.tensor             # [6, 160000] int64
└── meta.json
```

---

## MultiCameraBuffer 用法

```cpp
BufferConfig buf_cfg;
buf_cfg.max_frames = 10;
MultiCameraBuffer buffer(buf_cfg);

// 生产端（数据加载线程）
buffer.push(std::move(frame_ptr));

// 消费端（推理线程）
auto frame = buffer.pop();  // nullptr 表示队列为空
```

---

## 相机顺序约定

| Index | 枚举 | NuScenes 通道 |
|-------|------|--------------|
| 0 | FRONT       | CAM_FRONT |
| 1 | FRONT_RIGHT | CAM_FRONT_RIGHT |
| 2 | FRONT_LEFT  | CAM_FRONT_LEFT |
| 3 | BACK        | CAM_BACK |
| 4 | BACK_LEFT   | CAM_BACK_LEFT |
| 5 | BACK_RIGHT  | CAM_BACK_RIGHT |

---

## 扩展方向

1. **在线相机流**：实现 `CameraDriver` 基类，通过 USB/GMSL 驱动填充 `CameraFrame`
2. **时间戳同步**：在 `MultiCameraBuffer` 中实现近似时间同步（按 `sync_slop_s` 容差对齐 6 路时间戳）
3. **图像去畸变**：在 `FrameLoader` 加载时调用 OpenCV `undistort`，需要扩展 `CalibrationData` 加入畸变系数
4. **ROI 裁剪**：支持根据 BEV 感兴趣区域裁剪相机图像以减少计算量
