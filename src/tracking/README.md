# Tracking 模块

> 路径：`src/tracking/`

---

## 概述

本模块实现**多目标跟踪（MOT）**，输入为每帧 BEV 检测结果，输出为带有稳定 `track_id` 的目标轨迹。

算法：简化 SORT（Simple Online and Realtime Tracking），去除卡尔曼滤波依赖，采用低通滤波 + 贪心匈牙利匹配，适合嵌入式环境。

---

## 文件说明

| 文件 | 职责 |
|------|------|
| `track.hpp` | 单目标轨迹数据结构、生命周期状态机、历史轨迹记录 |
| `tracker.hpp` | 多目标跟踪器接口声明和配置参数 |
| `tracker.cpp` | 跟踪器实现：预测、代价矩阵、匈牙利匹配、状态更新 |

---

## 数据流

```
BEV 检测结果 (BoundingBox[])
        ↓ 转换为 Detection[]
  Tracker::update(detections, timestamp)
        ↓
  1. 预测所有现有轨迹 (predict)
  2. 构建代价矩阵 (BEV 中心距离，同类别)
  3. 匈牙利匹配
  4. 更新匹配轨迹 / 创建新轨迹 / 标记丢失
        ↓
  confirmed Track[] (is_confirmed() == true)
```

---

## 接口

### Detection（输入）

```cpp
struct Detection {
    float x, y, z;       // BEV 中心（lidar 坐标系，米）
    float w, l, h;       // 尺寸
    float yaw;           // 偏航角（弧度）
    float vx, vy;        // 速度（可选，来自检测器）
    float score;         // 置信度
    int   class_id;      // NuScenes 类别 id（0~9）
};
```

### Track（输出）

```cpp
struct Track {
    uint64_t   track_id;          // 全局唯一跟踪 ID
    int        class_id;
    TrackState state;             // New / Active / Lost / Removed

    float x, y, z;               // 当前估计位置
    float vx, vy;                // 速度估计
    float yaw, w, l, h;
    float score;

    std::vector<TrackPoint> history;  // 历史轨迹（最近 30 帧）
};
```

### Tracker 主接口

```cpp
TrackerConfig cfg;
cfg.max_dist        = 3.0f;  // 匹配距离阈值（米）
cfg.max_lost_frames = 3;     // 丢失超时帧数
cfg.min_hits_confirm= 2;     // 确认所需最少匹配次数
cfg.dt              = 0.05;  // 帧间隔（秒）

Tracker tracker(cfg);

// 每帧调用
std::vector<Detection> dets = convert_bboxes(bboxes);
auto confirmed_tracks = tracker.update(dets, timestamp);
```

---

## 与 BEV 检测结果的对接

```cpp
// fastbev::post::transbbox::BoundingBox → Detection
fastbev::tracking::Detection to_detection(
    const fastbev::post::transbbox::BoundingBox& box)
{
    return {
        box.position.x, box.position.y, box.position.z,
        box.size.w, box.size.l, box.size.h,
        box.z_rotation,
        box.velocity.vx, box.velocity.vy,
        box.score,
        box.id
    };
}
```

---

## 状态机

```
         ┌────────────────────────────┐
  new    │ hits < min_hits_confirm    │  time_since_update > max_lost
Track ───┤ state = New                ├──────────────────────────────► Removed
         │ (not in output)            │
         └──────────────┬─────────────┘
                        │ hits >= min_hits
                        ▼
                    Active ──── 连续更新 ────► Active
                        │
                        │ 一帧未匹配
                        ▼
                      Lost ──── 再次匹配 ────► Active
                        │
                        │ > max_lost_frames
                        ▼
                     Removed（不再输出）
```

---

## 扩展方向

1. **卡尔曼滤波**：替换 `Track::update()` 和 `Track::predict()` 中的低通滤波，用标准 8 维 KF（x,y,z,vx,vy,yaw,w,l）
2. **IoU 匹配**：对于 BEV 框，可计算旋转 IoU 代替中心距离，提升遮挡场景的匹配精度
3. **ReID 嵌入**：引入视觉特征，提升长期跟踪和遮挡恢复能力
4. **多类别分离跟踪**：为每个类别维护独立跟踪器实例
