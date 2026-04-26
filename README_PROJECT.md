# CUDA-FastBEV-W 项目架构与工程化指南

本文档面向后续工程化开发，描述整条感知链路的架构、各阶段实现状态和可运行命令。  
它不替代 [README.md](README.md)（快速上手），而是作为更完整的项目级说明。

---

## 项目目标

```
多相机图像 → 标定/同步 → BEV 检测 → 多目标跟踪 → 轨迹输出 → ROS/系统集成
```

从单帧视觉 BEV 检测起步，逐步形成可工程化部署的车端多相机感知系统。

---

## 当前整体状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| A | 检测推理链（TRT + C++ 单帧/批量） | ✅ 已实现、可运行 |
| B | 多相机管理、标定读取、NuScenes 适配 | ✅ 已实现、可运行 |
| C | 多目标跟踪模块（SORT 风格） | ✅ 已实现（接口完备） |
| D | 感知管线（检测+跟踪端到端流程） | ✅ 已实现（接口完备） |
| E | ROS 系统集成与部署 | 🚧 框架完成，待系统联调 |

---


---

## 目录结构

```text
CUDA-FastBEV/
├── src/
│   ├── common/                  # Tensor、日志、CUDA 工具、可视化
│   ├── fastbev/                 # BEV 前处理、TRT 推理、后处理
│   ├── tracking/                # 多目标跟踪（Track + Tracker）
│   ├── camera/                  # 相机帧管理、多路缓存、标定读取
│   ├── pipeline/                # 感知总管线（检测 + 跟踪 + 输出）
│   └── main.cpp                 # CLI 推理入口
├── tool/
│   ├── environment.sh           # 环境变量（TRT/CUDA 路径）
│   ├── build_trt_engine.sh      # ONNX → TRT engine
│   ├── run.sh                   # CMake 编译 + 运行
│   └── draw.py                  # 单帧结果可视化
├── tools/
│   ├── nuscenes_adapter.py      # NuScenes → 帧目录格式适配
│   └── video_demo.py            # 多帧结果渲染为视频
├── ros/
│   ├── fastbev_ros_node.cpp     # ROS 节点（检测 + 轨迹发布）
│   └── README.md
├── configs/                     # 模型配置
├── model/                       # TRT plan / ONNX / 权重
├── example-data/                # 轻量示例输入
├── data/nuscenes/               # NuScenes v1.0-mini 数据集
└── outputs/                     # 推理结果 / 视频（运行时生成）
```

---

## 阶段 A：单帧/批量 BEV 检测链

### 描述

C++ TensorRT 推理主程序，支持单帧和批量两种模式，输出 TXT/JSON 检测结果。

### 关键文件

- `src/main.cpp` — 推理入口、参数解析、结果保存
- `src/fastbev/` — 前处理 CUDA kernel、TRT 推理封装、框解码

### 如何运行

**1. 编译（如尚未编译）**

```bash
source tool/environment.sh
mkdir -p build && cd build && cmake .. && make -j4 && cd ..
```

**2. 单帧推理（example-data）**

```bash
source tool/environment.sh
./build/fastbev example-data/example-data resnet18int8 int8 --score-thr 0.3
cat model/resnet18int8/result.txt
```

**3. JSON 输出 + 可视化**

```bash
./build/fastbev example-data/example-data resnet18int8 int8 \
    --score-thr 0.3 --output-format json
python tool/draw.py --pred-path model/resnet18int8/result.txt \
    --vis-path outputs/vis.png
```

**4. 批量推理（多帧目录）**

```bash
./build/fastbev outputs/frames resnet18int8 int8 \
    --score-thr 0.3 --output-format json --batch --no-warmup
# 每帧目录下生成 result.json
```

### 推理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--score-thr` | 置信度阈值 | 0.5 |
| `--output-format` | `txt` 或 `json` | txt |
| `--batch` | 批量模式（对 `frame_*/` 目录迭代） | 否 |
| `--no-warmup` | 关闭 warmup（调试用） | 否 |
| `--classes` | 类别 ID 过滤（逗号分隔） | 全部 |

---

## 阶段 B：多相机管理与 NuScenes 数据适配

### 描述

- `src/camera/` — C++ 多相机帧数据结构与缓冲队列
- `tools/nuscenes_adapter.py` — 将 NuScenes v1.0-mini 原始数据转为推理帧目录格式

### 关键文件

| 文件 | 功能 |
|------|------|
| `src/camera/camera_frame.hpp` | 单帧数据结构（6路图像 + 几何张量 + 时间戳） |
| `src/camera/multi_camera_buffer.hpp` | 线程安全多帧缓冲队列 |
| `src/camera/calibration.hpp` | 相机标定参数读取接口 |
| `tools/nuscenes_adapter.py` | NuScenes → 帧目录（图像 + 几何张量 + meta.json） |

### 几何张量说明

每个帧目录包含三个预计算张量：

| 文件 | 形状 | 含义 |
|------|------|------|
| `valid_c_idx.tensor` | `[6, 160000]` float32 | 各体素是否落在该相机有效视野内 |
| `x.tensor` | `[6, 160000]` int64 | 特征图 x 坐标（0–175） |
| `y.tensor` | `[6, 160000]` int64 | 特征图 y 坐标（0–63） |

**重要**：C++ 硬编码期望 `1600×900` 输入图像，内部 CUDA kernel 做等比 resize（×0.44）和 center crop（crop_y=70）。
适配器保存原始分辨率图像，K 矩阵也相应做 resize_lim + crop 修正后再用于几何计算。

### 如何运行

**从 NuScenes v1.0-mini 准备帧数据**

```bash
conda activate bev
python tools/nuscenes_adapter.py \
    --nuscenes-dir data/nuscenes \
    --version v1.0-mini \
    --out-dir outputs/frames \
    --num-frames 40
```

指定场景（可选）：

```bash
python tools/nuscenes_adapter.py \
    --nuscenes-dir data/nuscenes \
    --scene-names scene-0061,scene-0103 \
    --num-frames 40 \
    --out-dir outputs/frames
```

输出目录结构：

```text
outputs/frames/
├── frame_00000/
│   ├── 0-FRONT.jpg           # 1600×900, RGB（C++ 期望尺寸）
│   ├── 1-FRONT_RIGHT.jpg
│   ├── ...
│   ├── 5-BACK_RIGHT.jpg
│   ├── valid_c_idx.tensor
│   ├── x.tensor
│   ├── y.tensor
│   └── meta.json             # 帧元信息（时间戳、sample token 等）
├── frame_00001/
│   └── ...
└── manifest.json             # 所有帧的索引
```

---

## 阶段 C：多目标跟踪模块

### 描述

基于 BEV 检测结果的在线多目标跟踪，采用贪心匈牙利匹配（center distance）+ 低通滤波运动模型。

### 关键文件

| 文件 | 功能 |
|------|------|
| `src/tracking/track.hpp` | Track 数据结构、状态机、低通滤波预测 |
| `src/tracking/tracker.hpp` + `.cpp` | 多目标跟踪器（创建/更新/删除轨迹） |

### Track 状态机

```
New → Active → Lost → Removed
           ↑_____|  （重新匹配）
```

### 数据接口

```cpp
// 输入：BEV 检测框
struct Detection {
    float x, y, z;        // 3D 中心坐标
    float w, l, h;        // 尺寸
    float yaw;            // 朝向（rad）
    float vx, vy;         // 速度
    float score;          // 置信度
    int   class_id;       // 类别
};

// 输出：已确认目标轨迹列表
std::vector<Track> confirmed_tracks = tracker.update(detections, timestamp);

// Track 核心属性
track.track_id;       // 唯一 ID
track.x, track.y;    // 平滑后的 BEV 位置
track.vx, track.vy;  // 估计速度
track.history;        // 轨迹点历史（最近 30 帧）
```

### 跟踪配置

```cpp
TrackerConfig cfg;
cfg.max_dist        = 3.0f;   // 匹配距离阈值（米）
cfg.max_lost_frames = 3;      // 最多连续丢失帧数
cfg.min_hits_confirm= 2;      // 确认目标所需最少匹配次数
cfg.dt              = 0.05f;  // 时间步长（秒）
```

---

## 阶段 D：感知管线（端到端流程）

### 描述

`PerceptionPipeline` 将 BEV 检测与多目标跟踪串联，接收多相机帧，输出检测框和目标轨迹。

### 关键文件

| 文件 | 功能 |
|------|------|
| `src/pipeline/perception_pipeline.hpp` | 管线接口定义 |
| `src/pipeline/perception_pipeline.cpp` | 管线实现（初始化、推理、跟踪、回调） |

### 管线数据流

```
CameraFrame
    ↓
[1] update_geometry_tensors()   ← valid_c_idx / x / y 上传 GPU
    ↓
[2] core->forward(images)       ← TRT 推理（前处理 + backbone + BEV head）
    ↓
[3] filter_boxes(score_thr)     ← 阈值过滤 + 类别过滤
    ↓
[4] tracker.update(detections)  ← 多目标跟踪（可选）
    ↓
[5] result_callback(result)     ← 用户自定义回调（保存、发布、可视化）
```

### 使用示例

```cpp
PipelineConfig cfg;
cfg.model_name      = "resnet18int8";
cfg.score_thr       = 0.3f;
cfg.enable_tracking = true;
cfg.output_dir      = "outputs/frames";
cfg.output_format   = "json";

PerceptionPipeline pipeline;
pipeline.init(cfg);
pipeline.set_callback([](const PerceptionResult& result) {
    printf("帧 %d: %zu 检测, %zu 轨迹\n",
           result.frame_id, result.detections.size(), result.tracks.size());
});

CameraFrame frame = FrameLoader::load_from_dir("outputs/frames/frame_00000", 0);
pipeline.process(frame, stream);
pipeline.reset_tracker();  // 切换场景时重置
```

---

## 阶段 E：ROS 集成与系统部署

### 描述

ROS 节点封装，订阅相机 topic，发布检测框和目标轨迹。

### 关键文件

| 文件 | 功能 |
|------|------|
| `ros/fastbev_ros_node.cpp` | ROS 节点实现 |
| `ros/README.md` | 编译和运行说明 |

### 订阅/发布接口

```
订阅:  /camera/front/image_raw, /camera/front_right/image_raw, ...（6路）
发布:  /fastbev/detections  (vision_msgs/Detection3DArray)
       /fastbev/tracks       (std_msgs/String, JSON)
       /fastbev/bev_image    (sensor_msgs/Image)
```

### 当前状态

- ✅ 节点结构、CMakeLists、launch 文件框架完成
- ✅ TRT 推理核心与 ROS 桥接代码完成
- 🚧 需要与实际相机驱动联调
- 🚧 消息类型待根据下游需求确认

---

## 完整端到端运行示例

### 1. 基础链路（example-data）

```bash
source tool/environment.sh
./build/fastbev example-data/example-data resnet18int8 int8 --score-thr 0.3
python tool/draw.py --pred-path model/resnet18int8/result.txt --vis-path outputs/vis.png
```

### 2. NuScenes 全链路

```bash
conda activate bev

# Step 1: NuScenes → 帧目录
python tools/nuscenes_adapter.py \
    --nuscenes-dir data/nuscenes --version v1.0-mini \
    --out-dir outputs/frames --num-frames 40

# Step 2: 批量推理（结果存入各帧目录）
source tool/environment.sh
./build/fastbev outputs/frames resnet18int8 int8 \
    --score-thr 0.3 --output-format json --batch --no-warmup

# Step 3: 视频生成
conda activate bev
python tools/video_demo.py \
    --frames-dir outputs/frames --out-dir outputs/video \
    --score-thr 0.3 --fps 6
# → outputs/video/fastbev_demo.mp4
```

### 3. auto-infer 模式（边推理边渲染）

```bash
conda activate bev
python tools/video_demo.py \
    --frames-dir outputs/frames --out-dir outputs/video \
    --model resnet18int8 --score-thr 0.3 --auto-infer --fps 6
```

---

## 已知问题与注意事项

| 问题 | 原因 | 状态 |
|------|------|------|
| NuScenes 帧 0 检测（已修复） | `cv2.imwrite` 保存 BGR，`stbi_load` 读为 RGB，通道颠倒 | ✅ 修复：保存前转 `img[:,:,::-1]` |
| NuScenes 帧 0 检测（已修复） | Python 适配器缩放图像至 704×256，C++ 按 1600×900 读取缓冲区越界 | ✅ 修复：不缩放，保存原始分辨率 |
| K 矩阵 y 方向偏移（已修复） | 用非等比缩放 `256/900`，应为等比 `0.44` + `crop_y=70` | ✅ 修复 |
| auto-infer 结果路径错误（已修复） | 单帧模式写 `model/<model>/result.json`，非帧目录 | ✅ 修复：`video_demo.py` 自动复制 |

---

## 版本管理建议

纳入 git：`src/  tool/  tools/  ros/  configs/  demo/  README.md  README_PROJECT.md`

`.gitignore`：`build/  outputs/  data/  model/*.plan  model/*/build/  example-data/  __pycache__/`
