# CUDA-FastBEV — 高性能纯视觉 BEV 3D 目标检测

基于 NVIDIA TensorRT 的 [FastBEV](https://github.com/Sense-GVT/Fast-BEV) 纯相机 BEV 感知推理框架。支持 NuScenes 原始数据、多帧视频可视化、多目标跟踪接口及 ROS Noetic 实时接口。

> **原版说明**: This repository contains sources and model for Fast-BEV inference using CUDA & TensorRT, with PTQ/QAT int8 quantization support.

**完整链路**：`多相机图像 → 标定/同步 → BEV 检测 → 多目标跟踪 → 轨迹输出 → ROS/系统集成`
1、python tool/draw.py --pred-path model/resnet18/result.txt 
                    --vis-path  model/resnet18/vis.png将这个单帧绘制的结果也能通过nusences的数据集生成视频。
2、另外当前运行python tools/video_demo.py 
    --frames-dir outputs/frames 
    --out-dir    outputs/video 
    --model      resnet18int8 
    --score-thr  0.3 
    --auto-infer 
    --fps 6是能够生成结果，也有检测框，但绘制的时候没有开了其他车辆和自车的相对关系，绘制出来很乱，航向角明显不对，自车也没有速度。因此这里你需要重点区改进。另外箭头也不要绘制出来重点是看车的状态。你需要考虑绘制出来的障碍物的具体类别，比如人就绘制小一点等。具体可以参考python tool/draw.py的绘制方式。然后3Dbox在图像上最好也绘制出来，如python tool/draw.py一样。自车速度，自车航向和自车信息均需要考虑。
3、最后当前如果先运行Step 1 · NuScenes 数据适配中生成frames，在使用绘图的demo去绘图和视频，还是会出现0帧的报错。请你将这个bug也修复。


---

## 目录结构

```
CUDA-FastBEV/
├── src/                        # C++ 推理核心
│   ├── main.cpp                # 推理入口（支持过滤/JSON/批量模式）
│   ├── fastbev/                # FastBEV 预/后处理模块
│   ├── common/                 # 张量、可视化、Timer 等公共组件
│   ├── tracking/               # 多目标跟踪模块（track.hpp / tracker.hpp）
│   ├── camera/                 # 相机帧管理、多路缓冲、标定加载
│   └── pipeline/               # 完整感知管线（检测+跟踪+回调）
├── tools/                      # Python 工具链
│   ├── nuscenes_adapter.py     # NuScenes 原始数据 → C++ 帧目录（✅ 支持 v1.0-mini）
│   └── video_demo.py           # 多帧 BEV+相机视频生成
├── ros/                        # ROS Noetic 接口
│   ├── fastbev_ros_node.cpp
│   ├── CMakeLists.txt
│   ├── package.xml
│   └── launch_fastbev_node.sh
├── tool/                       # 原有脚本
│   ├── draw.py                 # 单帧可视化
│   ├── build_trt_engine.sh
│   ├── run.sh
│   └── environment.sh
├── data/nuscenes/              # NuScenes 数据集（v1.0-mini 已验证）
├── model/                      # TRT 模型文件
├── configs/                    # 模型配置
├── ptq/                        # 量化工具
└── example-data/               # 示例数据
```

---

## 性能指标（NuScenes val）

| 模型 | 框架 | 精度 | mAP | FPS |
|:----:|:----:|:----:|:---:|:---:|
| ResNet18 | TensorRT | FP16 | 24.3 | 113.6 (RTX 2080Ti) |
| ResNet18-PTQ | TensorRT | FP16+INT8 | 23.89 | 143.8 |
| ResNet18-head-PTQ | TensorRT | FP16+INT8 | 23.83 | 144.9 |

---

## 环境依赖

- CUDA ≥ 11.0，CUDNN ≥ 8.2，TensorRT ≥ 8.5.0
- libprotobuf-dev == 3.6.1，Compute Capability ≥ sm_80
- Python ≥ 3.6（`conda activate bev`）

---

## 快速开始

### 1. 下载模型和数据

```bash
# 下载并解压（Google Drive）
unzip model.zip
unzip nuScenes-example-data.zip
```

### 2. 配置环境

```bash
# 修改 TensorRT/CUDA 路径后激活
source tool/environment.sh
```

### 3. 编译 TRT 引擎 + C++ 程序

```bash
bash tool/build_trt_engine.sh
mkdir -p build && cd build && cmake .. && make -j$(nproc) && cd ..
```

### 4. 单帧推理

```bash
# 基本推理（输出 TXT）
./build/fastbev example-data resnet18

# 带过滤参数（JSON 输出）
./build/fastbev example-data resnet18 fp16 \
    --score-thr 0.4 --classes 0,8 --output-format json

# 单帧可视化
python tool/draw.py --pred-path model/resnet18/result.txt 
                    --vis-path  model/resnet18/vis.png
```

---

## Step 1 · NuScenes 数据适配

```bash
conda activate bev

# ✅ 推荐：直接读取原始 NuScenes 数据集（无需 mmdet3d）
python tools/nuscenes_adapter.py 
    --nuscenes-dir data/nuscenes 
    --version      v1.0-mini 
    --out-dir      outputs/frames 
    --num-frames   50

# 指定场景（scene-0061 等）
python tools/nuscenes_adapter.py 
    --nuscenes-dir data/nuscenes 
    --version      v1.0-mini 
    --out-dir      outputs/frames 
    --scene-names  scene-0061,scene-0103 
    --num-frames   100

# 兼容模式（从 example-data.pth）
python tools/nuscenes_adapter.py 
    --pth-path example-data/example-data/example-data.pth 
    --out-dir  outputs/frames

# NuScenes pkl 模式（需要 mmdet3d 预处理的 pkl）
python tools/nuscenes_adapter.py 
    --pkl-path      /data/nuscenes/nuscenes_infos_val.pkl 
    --nuscenes-root /data/nuscenes 
    --out-dir       outputs/frames 
    --num-frames    50
```

每帧输出：`valid_c_idx.tensor`（[6,160000] float32）、`x.tensor`、`y.tensor`（[6,160000] int64）+ 6 路相机图像。

---

## Step 2 · 模块化过滤与 JSON 输出

```bash
# 过滤参数说明
./build/fastbev <data_dir> <model> [precision] [options]
  --score-thr <float>        置信度阈值（默认 0.5）
  --classes   <0,1,...>      类别过滤（0=car...9=traffic_cone）
  --output-format <txt|json> 输出格式（默认 txt）
  --batch                    批量模式（遍历 frame_* 子目录）
  --no-warmup                跳过 warmup

# 批量推理
./build/fastbev outputs/frames resnet18 fp16 \
    --score-thr 0.35 --output-format json --batch --no-warmup
```

---

## Step 3 · 多帧视频可视化

```bash
conda activate bev

# 使用已有推理结果（批量推理后）直接渲染
python tools/video_demo.py 
    --frames-dir outputs/frames 
    --out-dir    outputs/video 
    --score-thr  0.25 
    --fps 6 --bev-size 800 --cam-width 480

# 自动触发 C++ 推理 + 视频生成（需要 --auto-infer）
python tools/video_demo.py 
    --frames-dir outputs/frames 
    --out-dir    outputs/video 
    --model      resnet18int8 
    --score-thr  0.3 
    --auto-infer 
    --fps 6


python tools/video_demo.py 
  --frames-dir outputs/frames 
  --out-dir outputs/video 
  --score-thr 0.25 
  --fps 6 
  --bev-size 800 
  --cam-width 480 
  --bev-mode world 
  --ego-heading-offset-deg 90


# 输出: outputs/video/fastbev_demo.mp4
```

帧布局：左侧 3×2 相机网格 | 右侧 BEV 俯视图 + 底部信息栏。

---

## 多目标跟踪（Tracking 模块）

跟踪模块位于 `src/tracking/`，提供与 BEV 检测解耦的目标跟踪接口。详见 [src/tracking/README.md](src/tracking/README.md)。

```cpp
// C++ 接口示例
#include "tracking/tracker.hpp"

fastbev::tracking::TrackerConfig tc;
tc.max_dist = 3.0f;   // 匹配距离阈值（米）
tc.max_lost_frames = 3;

fastbev::tracking::Tracker tracker(tc);

// 每帧调用
std::vector<fastbev::tracking::Detection> dets = convert(bboxes);
auto confirmed_tracks = tracker.update(dets, timestamp);
```

---

## 感知管线（Pipeline 模块）

`src/pipeline/` 将检测、跟踪、过滤打包为统一接口。详见 [src/pipeline/README.md](src/pipeline/README.md)。

```cpp
fastbev::pipeline::PipelineConfig cfg;
cfg.model_name       = "resnet18int8";
cfg.score_thr        = 0.4f;
cfg.enable_tracking  = true;

fastbev::pipeline::PerceptionPipeline pipeline(cfg);
pipeline.init();

auto result = pipeline.process(frame);
// result.detections  — 过滤后检测
// result.tracks      — 带 track_id 轨迹
// result.latency_ms  — 推理耗时
```

---

## Step 4 · ROS Noetic 实时接口

```bash
# 编译
ln -s $(pwd)/ros ~/catkin_ws/src/fastbev_ros
cd ~/catkin_ws && source /opt/ros/noetic/setup.bash
catkin_make && source devel/setup.bash

# 运行
rosrun fastbev_ros fastbev_ros_node \
    _model:=resnet18int8 \
    _score_thr:=0.4 \
    _geometry_dir:=$(pwd)/outputs/frames/frame_00000
```

**订阅**: `/cam_front/image_raw` × 6（近似时间同步）  
**发布**: `/fastbev/detections`（MarkerArray）、`/fastbev/detections_info`（JSON）

---

## 完整推理链路（快速验证）

```bash
conda activate bev
cd /home/dfg-autoware/BEV_projects/CUDA-FastBEV

# 1. 准备 NuScenes 帧数据
python tools/nuscenes_adapter.py \
    --nuscenes-dir data/nuscenes --version v1.0-mini \
    --out-dir outputs/frames --num-frames 20

# 2. 批量 BEV 推理（JSON 输出）
./build/fastbev outputs/frames resnet18int8 int8 \
    --score-thr 0.3 --output-format json --batch --no-warmup

# 3. 生成可视化视频
python tools/video_demo.py \
    --frames-dir outputs/frames --out-dir outputs/video \
    --score-thr 0.25 --fps 4
# 结果: outputs/video/fastbev_demo.mp4
```

---

## PTQ 量化

```bash
python ptq/ptq_bev.py    # PTQ 量化
python ptq/export_onnx.py  # 导出 ONNX
```

## 示例结果

![BEV 检测结果](demo/sample0_vis_int8_head.png)
![BEV 检测结果](demo/sample1_vis_int8_head.png)

---

## 模块索引

| 模块 | 路径 | 说明 |
|------|------|------|
| 数据适配 | [tools/nuscenes_adapter.py](tools/nuscenes_adapter.py) | NuScenes 原始数据 → 帧目录 |
| 视频可视化 | [tools/video_demo.py](tools/video_demo.py) | 多帧 BEV+相机视频 |
| 多目标跟踪 | [src/tracking/](src/tracking/) | [README](src/tracking/README.md) |
| 相机管理 | [src/camera/](src/camera/) | [README](src/camera/README.md) |
| 感知管线 | [src/pipeline/](src/pipeline/) | [README](src/pipeline/README.md) |
| ROS 接口 | [ros/](ros/) | Noetic 订阅发布节点 |

---

## 参考

- [Fast-BEV (Sense-GVT)](https://github.com/Sense-GVT/Fast-BEV)
- [NVIDIA Lidar AI Solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution)

