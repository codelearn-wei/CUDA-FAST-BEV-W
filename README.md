# CUDA-FastBEV — 高性能纯视觉 BEV 3D 目标检测

基于 NVIDIA TensorRT 的 [FastBEV](https://github.com/Sense-GVT/Fast-BEV) 纯相机 BEV 感知推理框架，支持 NuScenes 数据、多帧视频可视化及 ROS Noetic 实时接口。

> **原版说明**: This repository contains sources and model for Fast-BEV inference using CUDA & TensorRT, with PTQ/QAT int8 quantization support.


## 目录结构

```
CUDA-FastBEV/
├── src/                        # C++ 推理核心
│   ├── main.cpp                # 推理入口（支持过滤/JSON/批量模式）
│   ├── fastbev/                # FastBEV 预/后处理模块
│   └── common/                 # 张量、可视化、Timer 等公共组件
├── tools/                      # Python 工具链
│   ├── nuscenes_adapter.py     # Step 1：NuScenes → C++ 数据格式
│   └── video_demo.py           # Step 3：多帧 BEV+相机视频生成
├── ros/                        # Step 4：ROS Noetic 接口
│   ├── fastbev_ros_node.cpp    # ROS 订阅/发布节点
│   ├── CMakeLists.txt
│   ├── package.xml
│   └── launch_fastbev_node.sh
├── tool/                       # 原有工具脚本
│   ├── draw.py                 # 单帧可视化
│   ├── build_trt_engine.sh
│   ├── run.sh
│   └── environment.sh
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
python tool/draw.py --pred-path model/resnet18/result.txt \
                    --vis-path  model/resnet18/vis.png
```

---

## Step 1 · NuScenes 数据适配

```bash
conda activate bev

# 兼容模式（从 example-data.pth）
python tools/nuscenes_adapter.py \
    --pth-path example-data/example-data/example-data.pth \
    --out-dir  outputs/frames

# NuScenes 完整模式
python tools/nuscenes_adapter.py \
    --pkl-path      /data/nuscenes/nuscenes_infos_val.pkl \
    --nuscenes-root /data/nuscenes \
    --out-dir       outputs/frames \
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

# 自动推理 + 视频生成
python tools/video_demo.py \
    --frames-dir outputs/frames \
    --out-dir    outputs/video \
    --model      resnet18 \
    --score-thr  0.35 \
    --auto-infer \
    --fps 6 --bev-size 800 --cam-width 480

# 输出: outputs/video/fastbev_demo.mp4
```

帧布局：左侧 3×2 相机网格 | 右侧 BEV 俯视图 + 底部信息栏。

---

## Step 4 · ROS Noetic 实时接口

```bash
# 编译
ln -s $(pwd)/ros ~/catkin_ws/src/fastbev_ros
cd ~/catkin_ws && source /opt/ros/noetic/setup.bash
catkin_make && source devel/setup.bash

# 运行
rosrun fastbev_ros fastbev_ros_node \
    _model:=resnet18 \
    _score_thr:=0.5 \
    _geometry_dir:=$(pwd)/example-data/example-data
```

**订阅**: `/cam_front/image_raw` × 6（近似时间同步）  
**发布**: `/fastbev/detections`（MarkerArray）、`/fastbev/detections_info`（JSON）

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

## 参考

- [Fast-BEV (Sense-GVT)](https://github.com/Sense-GVT/Fast-BEV)
- [NVIDIA Lidar AI Solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution)
