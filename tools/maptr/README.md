# MapTRv2 HD 地图模块

本模块实现自动驾驶 HD 地图元素的提取与可视化，并将地图信息叠加到 CUDA-FastBEV 感知流水线的 BEV 视图中。

---

## 功能概述

| 功能 | 说明 |
|------|------|
| **双模式地图提取** | 优先使用 NuScenes Map Expansion GT；若不可用，自动降级为自车轨迹推断 |
| **BEV 地图叠加** | 将道路边界、车道线绘制到俯视图，与检测框共同显示 |
| **与 FastBEV 并行** | 地图提取独立运行，不影响原有推理流程 |
| **MapTRv2 神经网络推理** | 支持接入 MapTRv2 模型权重做在线 HD 地图预测（需额外下载） |

---

## 目录结构

```
tools/maptr/
├── __init__.py               # 包入口，导出主要接口
├── nusc_map_extractor.py     # 地图元素提取（双模式：GT + 轨迹降级）
├── maptr_model_infer.py      # MapTRv2 模型推理接口（subprocess 启动器）
├── maptr_infer_worker.py     # 推理 Worker（在 maptr conda 环境中运行）
├── setup_maptr_env.sh        # maptr conda 环境安装脚本
├── run_maptr.py              # CLI 入口脚本
└── README.md                 # 本文档
```

---

## 快速开始

### 1. 准备帧数据

首先运行 NuScenes 适配器生成帧目录（含 `meta.json`）：

```bash
conda activate bev
python tools/nuscenes_adapter.py 
    --nuscenes-dir data/nuscenes --version v1.0-mini 
    --out-dir outputs/frames
```

### 2. 提取地图元素

```bash
python tools/maptr/run_maptr.py 
    --frames-dir outputs/frames 
    --nuscenes-dir data/nuscenes 
    --version v1.0-mini 
    --mode gt 
    --overwrite
```

每个帧目录下生成 `map_result.json`。

### 3. 生成带地图的视频

```bash
python tools/video_demo.py 
    --frames-dir outputs/frames 
    --out-dir outputs/video 
    --fps 6 --bev-size 800 
    --score-thr 0.25 --bev-mode world
```

BEV 面板将自动叠加地图元素（绿色道路边界 + 青色车道线）。

若不需要地图叠加，加 `--no-map` 参数。

---

## 地图提取模式

### 模式 1：NuScenes Map Expansion GT（精确）

**条件**：`data/nuscenes/maps/expansion/singapore-onenorth.json` 等矢量地图文件存在。

**获取方式**：从 [NuScenes 官网](https://www.nuscenes.org/nuscenes#download) 下载 "Map expansion" 包，解压到 `data/nuscenes/maps/expansion/`：

```
data/nuscenes/maps/expansion/
├── singapore-onenorth.json      # ~100MB
├── singapore-hollandvillage.json
├── singapore-queenstown.json
└── boston-seaport.json
```

**地图元素类型**：
- `lane_divider`、`road_divider` → `divider`（青色）
- `ped_crossing` → `ped_crossing`（蓝色，填充多边形）
- `road_segment`、`walkway` → `boundary`（绿色）

### 模式 2：自车轨迹推断（降级，始终可用）

**条件**：无需额外数据，自动启用。

**原理**：
1. 收集所有帧的 `ego_translation_global` 全局位置
2. 将轨迹投影到当前帧的自车坐标系
3. 向两侧偏移生成道路边界（±7.5m）和车道线（±3.75m）
4. 中心轨迹作为车道中心线

**生成的元素**：
```
road_boundary_left / right   ±7.5m 偏移  →  boundary（绿色）
lane_divider_left  / right   ±3.75m 偏移 →  divider（青色）
lane_center                  0m 偏移     →  divider（青色）
```

---

## `map_result.json` 格式

每帧生成一个 JSON 文件：

```json
{
  "source": "gt" | "trajectory",
  "location": "singapore-onenorth",
  "frame_dir": "frame_00000",
  "ego_translation": [411.3, 1180.9, 0.0],
  "ego_yaw": 1.234,
  "patch_size": 120.0,
  "elements": [
    {
      "type": "boundary" | "divider" | "ped_crossing",
      "subtype": "road_boundary_left",
      "score": 0.8,
      "points": [[x1, y1], [x2, y2], ...],
      "is_polygon": false
    }
  ]
}
```

**坐标约定**：`points` 使用自车坐标系（`x=right, y=forward`，单位：米），与 FastBEV 检测结果坐标系相同。

---

## `run_maptr.py` 命令行参数

```
python tools/maptr/run_maptr.py [选项]

必选：
  --frames-dir     帧目录的父目录（含 frame_* 子目录）

可选：
  --nuscenes-dir   NuScenes 数据集根目录（默认 data/nuscenes）
  --version        数据集版本（默认 v1.0-mini）
  --mode           gt / model / auto（默认 gt）
  --ckpt           MapTRv2 模型权重路径（model 模式）
  --score-thr      置信度阈值（默认 0.3）
  --patch-size     地图查询范围（米，默认 120.0）
  --overwrite      覆盖已有 map_result.json
  --verify         运行后统计验证结果
  --frame          只处理单帧（调试用）
```

---

## `video_demo.py` 地图相关参数

```
python tools/video_demo.py [原有参数...] [地图参数]

  --no-map   禁用地图叠加（即使 map_result.json 存在）
```

---

## MapTRv2 神经网络推理（可选）

MapTRv2 需要 mmcv 1.x，与 bev 环境（mmcv 2.x）不兼容，因此采用独立 conda 环境 + subprocess 方式运行。

### 前提条件

- MapTRv2 项目已位于 `tools/MapTRv2/`（含 `projects/` 和 `mmdetection3d/`）
- 模型权重已下载：`model/maptr/maptr_nano_r18_110e.pth`
- 对应配置文件：`model/maptr/config/maptr_nano_r18_110e.py`

### 1. 安装 maptr 环境

```bash
# 在项目根目录执行（约需 10~30 分钟）
bash tools/maptr/setup_maptr_env.sh
```

该脚本将：
- 创建 `maptr` conda 环境（Python 3.8）
- 安装 PyTorch 1.10.0 + CUDA 11.1
- 安装 mmcv-full 1.4.0 / mmdet 2.14.0 / mmsegmentation 0.14.1
- 安装 MapTRv2 自带的 mmdetection3d（开发模式）
- 编译 `geometric_kernel_attn` 自定义 CUDA 算子

### 2. 检查模型状态

```bash
conda run -n bev python tools/maptr/run_maptr.py --check-model
```

### 3. 运行模型推理

```bash
# 批量推理所有帧（通过 subprocess 调用 maptr 环境中的 worker）
conda run -n bev python tools/maptr/run_maptr.py \
    --frames-dir outputs/frames \
    --mode model \
    --ckpt model/maptr/maptr_nano_r18_110e.pth \
    --score-thr 0.3 --overwrite --verify
```

### 4. 与 FastBEV 并行运行

```bash
# 在两个终端并行执行：
# 终端 1 —— FastBEV 感知
./build/fastbev outputs/frames resnet18int8 int8 --score-thr 0.25 --batch

# 终端 2 —— MapTRv2 地图推理（同时进行）
conda run -n bev python tools/maptr/run_maptr.py \
    --frames-dir outputs/frames --mode model --overwrite
```

两者独立写入各帧的 `result.json`（FastBEV）和 `map_result.json`（MapTRv2），互不干扰。

### 模型推理技术说明

- **图像预处理**：704×256 → resize 到 320×180 → pad 到 320×192 → 归一化（ImageNet）
- **内参处理**：使用 `meta.json` 中 `cam_intrinsics_raw`（1600×900 尺度），乘以 scale=0.2
- **输出格式**：`pts_3d` [N, 20, 2]，坐标在自车 BEV 坐标系（x=right, y=forward，单位 m）
- **map_result.json source 字段**：`"model"`（区分于 GT 的 `"gt"`）

> **注意**：若 `maptr` 环境未安装，`--mode model` 将自动回退到 GT 模式。

---

## 坐标系说明

```
  NuScenes 全局坐标系          自车坐标系（ego frame）
  ─────────────────            ──────────────────────
     y(north)                       y(forward)
       │                                │
       │                                │
       └──── x(east)         x(right) ──┘

  转换：
    right   = (px - tx) * sin(yaw) - (py - ty) * cos(yaw)
    forward = (px - tx) * cos(yaw) + (py - ty) * sin(yaw)
```

BEV 显示坐标系（canvas）与自车/检测结果坐标系一致：纵轴=前方(↑)，横轴=左侧(←)。

---

## 完整流水线示例

```bash
conda activate bev

# 1. 生成帧数据
python tools/nuscenes_adapter.py \
    --nuscenes-dir data/nuscenes --version v1.0-mini \
    --out-dir outputs/frames --num-frames 60

# 2. FastBEV 感知推理
./build/fastbev outputs/frames resnet18int8 int8 \
    --score-thr 0.25 --output-format json --batch

# 3. 地图提取（与感知结果无依赖，可并行执行）
python tools/maptr/run_maptr.py \
    --frames-dir outputs/frames --mode gt

# 4. 生成可视化视频（含检测框 + 地图叠加）
python tools/video_demo.py \
    --frames-dir outputs/frames \
    --out-dir outputs/video \
    --fps 6 --bev-size 800 --score-thr 0.25
```
