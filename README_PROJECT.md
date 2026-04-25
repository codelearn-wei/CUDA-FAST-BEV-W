# CUDA-FAST-BEV-W 项目架构与开发路线

这个文档不替代现有 [README.md](/home/dfg-autoware/BEV_projects/CUDA-FastBEV/README.md)，而是作为项目级说明，面向后续工程化开发、版本管理和功能扩展。

项目目标不是只停在单帧 3D 检测，而是逐步形成一条完整链路：

`多相机图像 -> 标定/同步 -> BEV 检测 -> 多目标跟踪 -> 轨迹输出 -> ROS/系统集成`

---

## 一、项目定位

`CUDA-FAST-BEV-W` 面向车端多相机感知研发，核心是把纯视觉 BEV 检测能力做成一个可持续扩展的工程底座。当前仓库已经具备：

- 基于 TensorRT 的 FastBEV 推理能力
- 单帧示例数据推理
- TRT engine 构建脚本
- 检测结果可视化
- NuScenes 数据适配脚本
- ROS 接口雏形

下一阶段的目标，是在这个仓库内继续沉淀：

- 多相机输入管理
- 在线/离线时序缓存
- 目标级 tracking
- 轨迹级输出接口
- 更完整的系统组织和部署方式

---

## 二、推荐目录组织

当前仓库已经有可用基础。后续建议以“按能力模块”而不是“按脚本散放”的方式演进。

```text
CUDA-FastBEV/
├── src/                       # C++ 核心推理与运行时
│   ├── common/                # Tensor、日志、计时器、可视化、配置解析
│   ├── fastbev/               # 检测前处理 / TRT 推理 / 后处理
│   ├── tracking/              # 多目标跟踪（未来新增）
│   ├── camera/                # 相机管理、时间同步、标定读取（未来新增）
│   ├── pipeline/              # 从图像到目标轨迹的总流程编排（未来新增）
│   └── main.cpp               # CLI 入口
├── tool/                      # 运行、画图、环境脚本
├── tools/                     # Python 数据准备、视频生成、辅助分析
├── ros/                       # ROS 节点与消息桥接
├── configs/                   # 模型、阈值、相机参数、跟踪参数
├── model/                     # ONNX / plan / 权重（通常不直接纳入 git）
├── demo/                      # 展示图和示意资源
├── example-data/              # 轻量示例输入（通常不直接纳入 git）
├── outputs/                   # 推理结果、视频、日志
├── nuscenes/                  # 数据索引或实验缓存
└── docs/                      # 设计文档（后续可新增）
```

---

## 三、当前代码职责

### 1. 检测主链

- [src/main.cpp](/home/dfg-autoware/BEV_projects/CUDA-FastBEV/src/main.cpp)
  - 主程序入口
  - 加载图像、几何张量、engine
  - 执行单帧检测并写出结果

- `src/fastbev/`
  - `fastbev_pre.cpp` / `fastbev_post.cpp`
  - `postprecess.cpp`
  - 负责前处理、TensorRT 推理封装、框解码、NMS

- `src/common/`
  - Tensor 管理
  - TensorRT 封装
  - 可视化几何定义

### 2. 运行与构建

- [tool/environment.sh](/home/dfg-autoware/BEV_projects/CUDA-FastBEV/tool/environment.sh)
  - TensorRT / CUDA / 第三方依赖路径

- [tool/build_trt_engine.sh](/home/dfg-autoware/BEV_projects/CUDA-FastBEV/tool/build_trt_engine.sh)
  - ONNX -> TensorRT engine
  - 现已支持单模型或 `all` 批量构建

- [tool/run.sh](/home/dfg-autoware/BEV_projects/CUDA-FastBEV/tool/run.sh)
  - CMake 编译并运行推理

### 3. 数据与展示

- [tool/draw.py](/home/dfg-autoware/BEV_projects/CUDA-FastBEV/tool/draw.py)
  - 读取检测结果并绘制相机 + BEV 可视化

- `tools/nuscenes_adapter.py`
  - NuScenes 数据向项目输入格式转换

- `tools/video_demo.py`
  - 多帧结果转视频

### 4. 系统接口

- `ros/fastbev_ros_node.cpp`
  - ROS 节点封装基础
  - 是后续接入在线相机流、发布检测和轨迹消息的入口

---

## 四、当前项目数据流

当前检测链路可以概括为：

1. 读取 6 路相机图像
2. 读取几何辅助张量：`valid_c_idx.tensor`、`x.tensor`、`y.tensor`
3. 执行 `fastbev_pre_trt.plan`
4. 执行 `fastbev_post_trt_decode.plan`
5. 输出 3D 检测框：`result.txt`
6. 通过 `tool/draw.py` 绘制结果图

这条链已经具备“单帧感知”能力，但离“车辆轨迹感知系统”还差两个关键层：

- 时序管理
- 跟踪管理

---

## 五、未来开发路线

后续建议按下面顺序推进，这样风险更可控，系统也更容易保持稳定。

### 阶段 A：把检测链打磨成稳定模块

目标：先把当前单帧检测做成稳定、可复现实验结果的基础能力。

建议事项：

- 统一配置文件，不把关键阈值散落在源码和脚本中
- 明确 `model/`、`outputs/`、`example-data/` 的输入输出边界
- 为推理结果补充 `json` 输出格式
- 保留单帧和批量离线两种运行方式

### 阶段 B：接入多相机工程输入

目标：把离线图片输入扩展到真实多相机流。

建议新增模块：

- `src/camera/`
  - 相机配置读取
  - 时间戳同步
  - 多路图像缓存
  - 标定参数加载

建议能力：

- 支持离线图像目录输入
- 支持 ROS topic 输入
- 支持按相机名管理图像

### 阶段 C：加入 Tracking

目标：把检测结果升级成连续目标轨迹。

建议新增模块：

- `src/tracking/`
  - Track 数据结构
  - 状态机（new / tracked / lost / removed）
  - 关联器（IoU / center distance / velocity gating）
  - 卡尔曼滤波器或简化运动模型

建议输出：

- 每个目标的 `track_id`
- 轨迹历史点
- 速度和朝向平滑结果

这一步完成后，系统就从“检测器”变成了“感知跟踪器”。

### 阶段 D：形成统一感知管线

目标：打通从图像到轨迹输出的主流程。

建议新增模块：

- `src/pipeline/`
  - `PerceptionPipeline`
  - `FrameContext`
  - `DetectionStage`
  - `TrackingStage`
  - `OutputStage`

目标输出：

- 检测框
- 目标轨迹
- 可视化结果
- ROS 发布消息
- 日志和性能统计

### 阶段 E：面向实车/系统部署

目标：真正接近可部署的车端感知工程。

建议事项：

- 参数热加载
- 多线程或流水线并行
- GPU/CPU 资源监控
- 输入异常保护
- 轨迹超时清理
- 录包与回放工具
- 在线调试可视化

---

## 六、建议优先新增的代码模块

如果马上开始下一阶段开发，我建议优先落这几个文件：

```text
src/
├── tracking/
│   ├── track.hpp
│   ├── track.cpp
│   ├── tracker.hpp
│   ├── tracker.cpp
│   └── kalman_filter.hpp
├── camera/
│   ├── camera_frame.hpp
│   ├── multi_camera_buffer.hpp
│   ├── calibration.hpp
│   └── calibration.cpp
├── pipeline/
│   ├── perception_pipeline.hpp
│   └── perception_pipeline.cpp
```

优先级建议：

1. `tracker`
2. `multi_camera_buffer`
3. `perception_pipeline`
4. ROS 输出轨迹消息

---

## 七、版本管理建议

这个仓库后续建议只管理“代码、配置、文档、轻量 demo 资源”，不要把大模型和大数据直接入库。

建议 git 管理范围：

- `src/`
- `tool/`
- `tools/`
- `ros/`
- `configs/`
- `demo/`
- `README.md`
- `README_PROJECT.md`

建议忽略：

- `build/`
- `outputs/`
- `nuscenes/`
- `example-data/`
- `model/`
- TensorRT `.plan`
- 中间日志和缓存

---

## 八、项目里程碑定义

建议把后续开发目标拆成四个里程碑：

### M1：稳定检测版

- TRT 检测稳定运行
- 单帧 / 批量 / 可视化完整

### M2：多相机输入版

- 支持真实多相机输入
- 支持标定和时间同步

### M3：轨迹输出版

- 引入 tracking
- 输出 `track_id` 和轨迹历史

### M4：系统集成版

- ROS 在线发布
- 视频展示
- 录包、回放、调试工具齐备

---

## 九、最终目标

最终希望这个仓库演进成一个完整的视觉感知工程，而不是一个只跑单帧 demo 的模型仓库。

目标形态是：

- 输入：多路相机图像
- 中间：BEV 检测 + 时序跟踪
- 输出：障碍物检测框、目标 ID、轨迹、速度、可视化结果、ROS 消息

也就是一条真正可持续扩展的链路：

`Image -> BEV Detection -> Multi-Object Tracking -> Trajectory -> System Integration`

---

## 十、建议下一步

如果按这个路线继续做，最值得马上开始的是两件事：

1. 把 `tracking/` 模块加进来
2. 把 `camera/` 和 `pipeline/` 组织起来

这样后面无论你接真实车载相机，还是接 ROS 在线输入，整个项目都不会越改越散。
