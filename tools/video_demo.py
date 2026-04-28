#!/usr/bin/env python3
"""
CUDA-FastBEV 多帧可视化视频生成工具

功能：
  - 逐帧调用 C++ 推理二进制（./build/fastbev），并读取 JSON 结果
  - 渲染 BEV 俯视图（带格网、ego 车辆、检测框、速度箭头、图例）
  - 组合 6 路相机画面 + BEV 视图为完整帧
  - 输出 MP4 视频文件，可选逐帧保存图像

用法:
  # 基于批量推理结果生成视频（先用 ./build/fastbev --batch 生成 JSON）
  python tools/video_demo.py \\
      --frames-dir outputs/frames \\
      --out-dir    outputs/video \\
      --model      resnet18 \\
      --score-thr  0.35

  # 自动调用 C++ 推理（需要已编译 build/fastbev）
  python tools/video_demo.py \\
      --frames-dir outputs/frames \\
      --out-dir    outputs/video \\
      --model      resnet18 \\
      --auto-infer

环境: conda activate bev
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# 加载检测后处理滤波器（可选，失败时降级为简单 score 过滤）
try:
    from tools.detection_filter import DetectionFilter, FilterConfig
    _FILTER_AVAILABLE = True
except ImportError:
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from detection_filter import DetectionFilter, FilterConfig
        _FILTER_AVAILABLE = True
    except ImportError:
        _FILTER_AVAILABLE = False

# ─── NuScenes 类别配置 ────────────────────────────────────────────────────────
CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

CLASS_COLORS = {
    "car":                   (255, 140,  0),
    "truck":                 ( 70, 170, 255),
    "construction_vehicle":  (  0, 200, 255),
    "bus":                   ( 40, 120, 255),
    "trailer":               (180, 110, 255),
    "barrier":               (180, 180, 180),
    "motorcycle":            (  0, 220, 120),
    "bicycle":               ( 40, 255, 200),
    "pedestrian":            (255,  90, 180),
    "traffic_cone":          (160, 200,  80),
}

BG_PRIMARY = (22, 28, 26)
BG_PANEL = (16, 18, 20)
FG_TEXT = (235, 238, 240)
FG_SUBTEXT = (168, 176, 180)
FG_GRID = (62, 72, 68)
FG_AXIS = (115, 128, 124)
EGO_COLOR = (245, 245, 245)

# ─── BEV 视野参数 ────────────────────────────────────────────────────────────
EGO_FORWARD   =  60.0   # 前向 60m
EGO_BACKWARD  = -15.0   # 后向 15m
EGO_SIDE      =  25.0   # 侧向 25m
# bounds = (forward_min, forward_max, left_min, left_max)
BEV_BOUNDS    = (EGO_BACKWARD, EGO_FORWARD, -EGO_SIDE, EGO_SIDE)


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def alpha_blend(base, overlay, alpha):
    return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)


def metric_to_canvas(points_xy: np.ndarray, bounds, canvas_size: int):
    """将 [N, 2] 的 lidar xy 坐标转为画布像素坐标 [N, 2]。
    坐标系: x=前方(上), y=左侧(左) → 正俯视图方向。
    """
    fwd_min, fwd_max, left_min, left_max = bounds
    width  = float(max(left_max  - left_min,  1e-6))
    height = float(max(fwd_max   - fwd_min,   1e-6))
    # y>0 在 NuScenes 中为左侧，映射到画布左侧（col 小）
    px = (left_max - points_xy[:, 1]) / width  * (canvas_size - 1)
    py = (fwd_max  - points_xy[:, 0]) / height * (canvas_size - 1)
    return np.stack([px, py], axis=1).astype(np.int32)


def draw_text_tag(image, text, anchor, color, font_scale=0.45):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x, y = int(anchor[0]), int(anchor[1])
    tl = (x, max(0, y - th - base - 6))
    br = (min(image.shape[1] - 1, x + tw + 8), y)
    cv2.rectangle(image, tl, br, (12, 12, 12), -1)
    cv2.rectangle(image, tl, br, color, 1)
    cv2.putText(image, text, (tl[0] + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245), 1, cv2.LINE_AA)


def draw_title_bar(image, title):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (20, 20, 20), -1)
    cv2.putText(panel, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (240, 240, 240), 2, cv2.LINE_AA)
    return panel


def rotate_points_xy(points_xy: np.ndarray, yaw_rad: float) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points_xy @ rot.T


def model_local_to_display(points_xy: np.ndarray) -> np.ndarray:
    """
    将模型/检测结果使用的局部坐标轴转换为 BEV 画布约定坐标。

    结合连续帧结果可见：
      - 第 2 维会随着自车前进而明显缩短，说明它更像 forward
      - 第 1 维更像横向 right/left 偏移

    因此这里采用：
      model local: x=right, y=forward
      display:     x=forward, y=left
    即：
      display_forward = local_forward = y
      display_left    = -local_right = -x
    """
    if points_xy.size == 0:
        return points_xy
    out = np.empty_like(points_xy, dtype=np.float32)
    out[:, 0] = points_xy[:, 1]
    out[:, 1] = -points_xy[:, 0]
    return out


def transform_model_local_to_world_relative(points_xy: np.ndarray, heading_global: float) -> np.ndarray:
    """
    将模型局部坐标（x=right, y=forward）旋转到以 ego 为中心的全局 north-up 视图。

    NuScenes global:
      x ≈ east
      y ≈ north

    display / metric_to_canvas:
      axis-0 = forward on screen (up)
      axis-1 = left on screen (left)

    做法：
      1. 用 heading_global 定义车辆 forward 方向在全局中的朝向
      2. local forward 沿车辆 heading
      3. local right 沿车辆右侧方向
      4. 再映射到 north-up 的 display 坐标
    """
    if points_xy.size == 0:
        return points_xy

    c = np.cos(heading_global)
    s = np.sin(heading_global)

    local_right = points_xy[:, 0]
    local_forward = points_xy[:, 1]

    global_x = local_forward * c + local_right * s
    global_y = local_forward * s - local_right * c

    out = np.empty_like(points_xy, dtype=np.float32)
    out[:, 0] = global_y
    out[:, 1] = -global_x
    return out


def wrap_angle_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


# ─── BEV 背景渲染 ─────────────────────────────────────────────────────────────

def render_bev_background(bounds, bev_size: int,
                          ego_yaw_global: float = None,
                          bev_mode: str = "world") -> np.ndarray:
    """绘制带同心圆格网的深色 BEV 背景。"""
    canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    # 渐变背景
    grad = np.linspace(42, 16, bev_size, dtype=np.uint8)
    canvas[:, :, 0] = (grad * 0.55).astype(np.uint8)[:, None]
    canvas[:, :, 1] = grad[:, None]
    canvas[:, :, 2] = (grad * 0.70).astype(np.uint8)[:, None]

    fwd_min, fwd_max, left_min, left_max = bounds
    max_r = int(max(abs(fwd_min), abs(fwd_max), abs(left_min), abs(left_max)))

    # 同心圆刻度
    for meter in range(10, max_r + 1, 10):
        angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        circle_pts = np.column_stack([meter * np.cos(angles), meter * np.sin(angles)])
        circle_canvas = metric_to_canvas(circle_pts, bounds, bev_size)
        cv2.polylines(canvas, [circle_canvas], True, FG_GRID, 1, cv2.LINE_AA)
        lp = metric_to_canvas(np.array([[meter, 0.0]]), bounds, bev_size)[0]
        cv2.putText(canvas, f"{meter}m", tuple(lp + [5, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, FG_SUBTEXT, 1, cv2.LINE_AA)

    # 格线
    for v in np.arange(math.ceil(fwd_min / 10) * 10, fwd_max + 0.1, 10):
        pts = metric_to_canvas(np.array([[v, left_min], [v, left_max]]), bounds, bev_size)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), FG_GRID, 1, cv2.LINE_AA)
    for v in np.arange(math.ceil(left_min / 10) * 10, left_max + 0.1, 10):
        pts = metric_to_canvas(np.array([[fwd_min, v], [fwd_max, v]]), bounds, bev_size)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), FG_GRID, 1, cv2.LINE_AA)

    # 坐标轴
    ax = metric_to_canvas(np.array([[fwd_min, 0], [fwd_max, 0]]), bounds, bev_size)
    ay = metric_to_canvas(np.array([[0, left_min], [0, left_max]]), bounds, bev_size)
    cv2.line(canvas, tuple(ax[0]), tuple(ax[1]), FG_AXIS, 1, cv2.LINE_AA)
    cv2.line(canvas, tuple(ay[0]), tuple(ay[1]), FG_AXIS, 1, cv2.LINE_AA)

    # Ego 车辆示意图
    ego_pts = np.array([
        [-1.0, -2.2],
        [-1.0,  2.2],
        [ 0.0,  4.2],
        [ 1.0,  2.2],
        [ 1.0, -2.2],
    ], dtype=np.float32)
    if bev_mode == "world" and ego_yaw_global is not None:
        ego_pts = transform_model_local_to_world_relative(ego_pts, float(ego_yaw_global))
    else:
        ego_pts = model_local_to_display(ego_pts)
    ego_canvas = metric_to_canvas(ego_pts, bounds, bev_size)
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, ego_canvas, EGO_COLOR)
    canvas[:] = alpha_blend(canvas, overlay, 0.18)
    cv2.polylines(canvas, [ego_canvas], True, EGO_COLOR, 2, cv2.LINE_AA)
    center_pts = np.array([[0.0, 0.0], [0.0, 8.0]], dtype=np.float32)
    if bev_mode == "world" and ego_yaw_global is not None:
        center_pts = transform_model_local_to_world_relative(center_pts, float(ego_yaw_global))
    else:
        center_pts = model_local_to_display(center_pts)
    center = metric_to_canvas(center_pts[:1], bounds, bev_size)[0]
    forward = metric_to_canvas(center_pts[1:], bounds, bev_size)[0]
    cv2.arrowedLine(canvas, tuple(center), tuple(forward),
                    EGO_COLOR, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(canvas, "ego", (center[0] - 24, center[1] + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, EGO_COLOR, 1, cv2.LINE_AA)
    return canvas


# ─── BEV 检测框渲染 ───────────────────────────────────────────────────────────

# draw_boxes_indexes for BEV (bottom 4 corners) and full 3D box
BEV_EDGES  = [(0, 1), (1, 2), (2, 3), (3, 0)]
BOX3D_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def boxes_to_corners_3d(detections: list) -> np.ndarray:
    """
    将检测结果列表转换为 [N, 8, 3] 的 3D 角点坐标（LiDAR 坐标系）。
    与 tool/draw.py 的 boxes_to_corners_lidar 保持一致。

    角点顺序（0-7）：底面 0-3（z 负方向），顶面 4-7（z 正方向）。
    底面角点 0,1 为"头部"侧（用于朝向指示）。
    """
    offsets = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float32)

    corners_list = []
    for det in detections:
        x, y, z = det.get("center_xyz", [0, 0, 0])
        w, l, h  = det.get("size_xyz",   [1, 1, 1])
        yaw      = float(det.get("yaw", 0))
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        std = offsets.copy()
        std[:, 0] *= w * 0.5
        std[:, 1] *= l * 0.5
        std[:, 2] *= h * 0.5

        corners = np.empty_like(std)
        corners[:, 0] = x + std[:, 0] * cos_y + std[:, 1] * sin_y
        corners[:, 1] = y + std[:, 0] * (-sin_y) + std[:, 1] * cos_y
        corners[:, 2] = z + std[:, 2]
        corners_list.append(corners)

    if not corners_list:
        return np.zeros((0, 8, 3), dtype=np.float32)
    return np.stack(corners_list, axis=0)


def boxes_to_bev_corners(boxes_np: np.ndarray) -> np.ndarray:
    """
    输入: [N, 7]  (x, y, z, w, l, h, yaw)
    输出: [N, 4, 2]  底面四角的 xy 坐标（lidar 坐标系）
    """
    if len(boxes_np) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    n = len(boxes_np)
    x, y, z, w, l, h, yaw = (boxes_np[:, i] for i in range(7))
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    # 局部坐标 (未旋转)
    offsets = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float32)  # [4, 2]
    corners = np.zeros((n, 4, 2), dtype=np.float32)
    for ci, (ox, oy) in enumerate(offsets):
        lx = ox * w / 2.0
        ly = oy * l / 2.0
        corners[:, ci, 0] = x + lx * cos_y + ly * sin_y
        corners[:, ci, 1] = y + lx * -sin_y + ly * cos_y
    return corners


def render_bev_detections(
    canvas: np.ndarray,
    detections: list,
    bounds,
    bev_size: int,
    ego_yaw_global: float = None,
    bev_mode: str = "world",
    show_labels: bool = False,
) -> dict:
    """将检测结果绘制到 BEV 画布上，返回图例计数字典。
    朝向：BEV 坐标系中 x=前方（画布上方），y=左侧（画布左侧）。
    不绘制速度箭头，仅显示朝向。
    """
    legend_counts = {}
    order = sorted(range(len(detections)),
                   key=lambda i: detections[i].get("score", 0))

    corners_3d = boxes_to_corners_3d(detections)  # [N, 8, 3]

    for idx in order:
        det    = detections[idx]
        score  = float(det.get("score", 0))
        label  = det.get("label_name", CLASS_NAMES[min(det.get("label", 0), 9)])
        color  = CLASS_COLORS.get(label, (180, 180, 180))

        cx = float(det.get("center_xyz", [0, 0, 0])[0])
        cy = float(det.get("center_xyz", [0, 0, 0])[1])
        w  = float(det.get("size_xyz",   [1, 1, 1])[0])

        # BEV 底面四角（corners 0-3）
        bottom_world = corners_3d[idx, :4, :2]  # [4, 2]
        if bev_mode == "world" and ego_yaw_global is not None:
            bottom_world = transform_model_local_to_world_relative(bottom_world, float(ego_yaw_global))
        else:
            bottom_world = model_local_to_display(bottom_world)
        bottom_c = metric_to_canvas(bottom_world, bounds, bev_size)
        center_world = np.array([[cx, cy]], dtype=np.float32)
        if bev_mode == "world" and ego_yaw_global is not None:
            center_world = transform_model_local_to_world_relative(center_world, float(ego_yaw_global))
        else:
            center_world = model_local_to_display(center_world)
        center_c = metric_to_canvas(center_world, bounds, bev_size)[0]

        # 朝向指示改为和原始 draw.py 一致：取前侧两角中点
        head_world = corners_3d[idx, [2, 3], :2].mean(axis=0, keepdims=True).astype(np.float32)
        if bev_mode == "world" and ego_yaw_global is not None:
            head_world = transform_model_local_to_world_relative(head_world, float(ego_yaw_global))
        else:
            head_world = model_local_to_display(head_world)
        head_c = metric_to_canvas(head_world,
                                  bounds, bev_size)[0]

        # 填充半透明底面 + 边框
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, bottom_c, color)
        canvas[:] = alpha_blend(canvas, overlay, 0.22)
        for s, e in BEV_EDGES:
            cv2.line(canvas, tuple(bottom_c[s]), tuple(bottom_c[e]),
                     color, 2, cv2.LINE_AA)
        # 朝向线（从中心到头部）
        cv2.line(canvas, tuple(center_c), tuple(head_c),
                 color, 3, cv2.LINE_AA)

        if show_labels:
            text   = f"{label[:3]} {score:.2f}"
            anchor = (bottom_c[0][0] + 5, bottom_c[0][1] - 5)
            draw_text_tag(canvas, text, anchor, color, font_scale=0.38)
        legend_counts[label] = legend_counts.get(label, 0) + 1

    return legend_counts


# ── 地图元素颜色 ──────────────────────────────────────────────────────────────
_MAP_COLORS = {
    "divider":     (0, 220, 220),    # 青色  — 车道/道路分隔线
    "ped_crossing":(80, 120, 255),   # 蓝色  — 人行横道
    "boundary":    (140, 200, 90),   # 绿色  — 道路边界
}
_MAP_ALPHA_FILL = 0.25   # polygon 填充透明度
_MAP_LINE_THICKNESS = 2


def render_map_overlay(
    canvas: np.ndarray,
    map_elements: list,
    bounds,
    bev_size: int,
    ego_yaw_global: float = None,
    bev_mode: str = "world",
) -> None:
    """
    将地图元素（来自 map_result.json）叠加到 BEV 画布上（原地修改）。

    地图元素坐标系：x=right, y=forward（与 FastBEV 检测结果相同）
    绘制顺序：boundary → ped_crossing → divider（由底到顶）。
    """
    if not map_elements:
        return

    # 按类型分组，决定绘制顺序
    order = ["boundary", "ped_crossing", "divider"]
    by_type: dict = {t: [] for t in order}
    for elem in map_elements:
        t = elem.get("type", "boundary")
        if t in by_type:
            by_type[t].append(elem)
        else:
            by_type.setdefault(t, []).append(elem)

    for t in order + [k for k in by_type if k not in order]:
        for elem in by_type.get(t, []):
            pts_local = np.array(elem.get("points", []), dtype=np.float32)
            if len(pts_local) < 2:
                continue

            # 坐标变换（与检测结果相同的流程）
            if bev_mode == "world" and ego_yaw_global is not None:
                pts_disp = transform_model_local_to_world_relative(
                    pts_local, float(ego_yaw_global))
            else:
                pts_disp = model_local_to_display(pts_local)

            pts_canvas = metric_to_canvas(pts_disp, bounds, bev_size)  # [N, 2]
            color = _MAP_COLORS.get(t, (160, 160, 160))

            if elem.get("is_polygon", False) and len(pts_canvas) >= 3:
                # 多边形半透明填充
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts_canvas], color)
                canvas[:] = alpha_blend(canvas, overlay, _MAP_ALPHA_FILL)
                cv2.polylines(canvas, [pts_canvas], isClosed=True,
                              color=color, thickness=_MAP_LINE_THICKNESS,
                              lineType=cv2.LINE_AA)
            else:
                # 折线
                cv2.polylines(canvas, [pts_canvas], isClosed=False,
                              color=color, thickness=_MAP_LINE_THICKNESS,
                              lineType=cv2.LINE_AA)


def render_bev_panel(detections: list, bounds, bev_size: int,
                     ego_speed_mps: float = 0.0,
                     ego_yaw_global: float = None,
                     display_yaw_global: float = None,
                     ego_motion_yaw_global: float = None,
                     ego_speed_forward_mps: float = None,
                     bev_mode: str = "world",
                     show_labels: bool = False,
                     map_elements: list = None) -> np.ndarray:
    """生成完整 BEV 检测结果面板（含背景、地图叠加、检测框、图例、自车信息）。"""
    canvas = render_bev_background(
        bounds, bev_size,
        ego_yaw_global=display_yaw_global,
        bev_mode=bev_mode)

    # 地图叠加（绘在检测框之前，避免遮挡）
    if map_elements:
        render_map_overlay(
            canvas, map_elements, bounds, bev_size,
            ego_yaw_global=display_yaw_global,
            bev_mode=bev_mode)

    legend = render_bev_detections(
        canvas, detections, bounds, bev_size,
        ego_yaw_global=display_yaw_global,
        bev_mode=bev_mode,
        show_labels=show_labels)

    # 自车信息框（左下角）
    speed_kmh = ego_speed_mps * 3.6
    ego_lines = [f"speed {speed_kmh:4.1f} km/h"]
    if ego_speed_forward_mps is not None:
        ego_lines.append(f"forward {ego_speed_forward_mps * 3.6:4.1f} km/h")
    if display_yaw_global is not None:
        hdg_deg = math.degrees(ego_yaw_global) % 360
        ego_lines.append(f"heading {hdg_deg:5.1f} deg")
    if ego_motion_yaw_global is not None:
        ego_lines.append(f"motion  {math.degrees(ego_motion_yaw_global) % 360:5.1f} deg")
    bx, by = 12, bev_size - len(ego_lines) * 23 - 18
    bw, bh = 196, len(ego_lines) * 23 + 14
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), BG_PANEL, -1)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (96, 156, 220), 1)
    cv2.putText(canvas, "ego", (bx + 8, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (96, 156, 220), 1, cv2.LINE_AA)
    for i, line in enumerate(ego_lines):
        cv2.putText(canvas, line, (bx + 8, by + 20 + i * 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, FG_TEXT, 1, cv2.LINE_AA)

    if display_yaw_global is not None:
        compass_center = (bx + bw - 34, by + 30)
        cv2.circle(canvas, compass_center, 18, (96, 156, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "N", (compass_center[0] - 6, compass_center[1] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, FG_SUBTEXT, 1, cv2.LINE_AA)
        arrow_tip = (
            int(compass_center[0] + 14 * math.cos(float(display_yaw_global))),
            int(compass_center[1] - 14 * math.sin(float(display_yaw_global))),
        )
        cv2.arrowedLine(canvas, compass_center, arrow_tip,
                        EGO_COLOR, 2, cv2.LINE_AA, tipLength=0.35)

    # 图例（右侧）
    legend_y = 54
    for label_name, count in sorted(legend.items(),
                                    key=lambda kv: (-kv[1], kv[0]))[:8]:
        color = CLASS_COLORS.get(label_name, (180, 180, 180))
        cv2.rectangle(canvas,
                      (bev_size - 180, legend_y - 12),
                      (bev_size - 166, legend_y + 2), color, -1)
        cv2.putText(canvas, f"{label_name}: {count}",
                    (bev_size - 158, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, FG_TEXT, 1, cv2.LINE_AA)
        legend_y += 20

    return draw_title_bar(canvas, "Top-down BEV Obstacles")


# ─── 相机网格渲染 ─────────────────────────────────────────────────────────────

CAMERA_SHORT_NAMES = [
    "FRONT", "FRONT_RIGHT", "FRONT_LEFT",
    "BACK",  "BACK_LEFT",   "BACK_RIGHT",
]


def project_corners_to_camera(
    corners_lidar: np.ndarray,  # [N, 8, 3]
    E: np.ndarray,              # [4, 4] lidar→camera extrinsic
    K: np.ndarray,              # [3, 3] camera intrinsic (for display image size)
    img_w: int,
    img_h: int,
) -> tuple:
    """
    将 N 个检测框的 8 角点从 LiDAR 坐标投影到相机图像坐标。

    返回:
        corners_img: [N, 8, 2]  像素坐标（display 图像尺寸）
        valid:       [N, 8]     每个角点是否在相机前方且在图像范围内的 bool
    """
    n = len(corners_lidar)
    if n == 0:
        return np.zeros((0, 8, 2)), np.zeros((0, 8), dtype=bool)

    R = E[:3, :3]
    t = E[:3, 3]

    flat = corners_lidar.reshape(-1, 3)  # [N*8, 3]
    cam  = flat @ R.T + t                # [N*8, 3]  lidar→camera

    depth = cam[:, 2]
    valid_depth = depth > 0.1

    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(valid_depth, (K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]), -1)
        v = np.where(valid_depth, (K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]), -1)

    valid_uv = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    valid_all = valid_depth & valid_uv

    corners_img  = np.stack([u, v], axis=1).reshape(n, 8, 2)
    valid_flat   = valid_all.reshape(n, 8)
    return corners_img, valid_flat


def draw_3d_boxes_on_image(
    img: np.ndarray,
    corners_img: np.ndarray,   # [N, 8, 2]
    valid: np.ndarray,          # [N, 8]
    colors: list,               # list of N (r,g,b) tuples
    scores: list,               # list of N float scores
    thickness: int = 3,
):
    """在相机图像上绘制 3D 检测框（12 条棱）。"""
    sort_ids = np.argsort(scores)
    for aid in sort_ids:
        color = colors[aid]
        score = scores[aid]
        shade = max(0.5, min(score * 1.2, 1.0))
        draw_color = tuple(int(c * shade) for c in color)
        for i, j in BOX3D_EDGES:
            if valid[aid, i] and valid[aid, j]:
                pt1 = tuple(corners_img[aid, i].astype(int).tolist())
                pt2 = tuple(corners_img[aid, j].astype(int).tolist())
                cv2.line(img, pt1, pt2, draw_color, thickness, cv2.LINE_AA)


def compose_camera_grid(
    frame_dir: "Path",
    detections: list,
    cam_width: int,
) -> np.ndarray:
    """将 6 路相机图像拼为 3×2 网格，并在每路图像上叠加 3D 检测框投影。"""
    cam_height = int(cam_width * 9 / 16)

    # 读取 meta.json 获取标定数据（用于 3D 框投影）
    meta_path = frame_dir / "meta.json"
    extrinsics_raw   = None  # [6][4][4]
    intrinsics_raw   = None  # [6][3][3]
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if "lidar2cam_extrinsics" in meta and "cam_intrinsics_raw" in meta:
            extrinsics_raw = meta["lidar2cam_extrinsics"]
            intrinsics_raw = meta["cam_intrinsics_raw"]

    # 预计算所有 3D 角点（LiDAR 坐标系，仅在有标定时使用）
    corners_3d = None
    det_colors = []
    det_scores = []
    if extrinsics_raw and intrinsics_raw and detections:
        corners_3d = boxes_to_corners_3d(detections)  # [N, 8, 3]
        for det in detections:
            label = det.get("label_name", CLASS_NAMES[min(det.get("label", 0), 9)])
            det_colors.append(CLASS_COLORS.get(label, (180, 180, 180)))
            det_scores.append(float(det.get("score", 0)))

    images = []
    for cam_idx, cam_name in enumerate(CAMERA_SHORT_NAMES):
        img_path = frame_dir / f"{cam_idx}-{cam_name}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((cam_height, cam_width, 3), dtype=np.uint8)
            cv2.putText(img, "missing", (cam_width // 4, cam_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)
        else:
            img = cv2.resize(img, (cam_width, cam_height))

        # 投影 3D 检测框到该相机图像
        if (corners_3d is not None and len(corners_3d) > 0
                and extrinsics_raw[cam_idx] is not None
                and intrinsics_raw[cam_idx] is not None):
            E = np.array(extrinsics_raw[cam_idx], dtype=np.float64)
            K_raw = np.array(intrinsics_raw[cam_idx], dtype=np.float64)
            # 将 K 缩放到 display 尺寸（cam_width × cam_height）
            scale_x = cam_width  / 1600.0
            scale_y = cam_height / 900.0
            K_display = K_raw.copy()
            K_display[0] *= scale_x
            K_display[1] *= scale_y
            corners_img, valid = project_corners_to_camera(
                corners_3d, E, K_display, cam_width, cam_height)
            draw_3d_boxes_on_image(img, corners_img, valid,
                                   det_colors, det_scores,
                                   thickness=max(1, cam_width // 240))

        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (12, 12, 12), -1)
        cv2.putText(img, cam_name, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        images.append(img)

    while len(images) < 6:
        images.append(np.zeros((cam_height, cam_width, 3), dtype=np.uint8))
    top    = np.concatenate(images[:3], axis=1)
    bottom = np.concatenate(images[3:6], axis=1)
    grid   = np.concatenate([top, bottom], axis=0)
    return draw_title_bar(grid, "Multi-Camera Input  (3D boxes projected)")


# ─── 信息栏 ───────────────────────────────────────────────────────────────────

def compose_info_bar(frame_idx: int, total_frames: int,
                     num_dets: int, width: int,
                     timestamp: int = 0,
                     ego_speed_mps: float = 0.0,
                     ego_yaw_deg: float = None,
                     ego_forward_speed_mps: float = None) -> np.ndarray:
    bar = np.zeros((44, width, 3), dtype=np.uint8)
    bar[:] = BG_PANEL
    ts_str = ""
    if timestamp:
        ts_str = f"  ts {timestamp // 1_000_000}.{(timestamp % 1_000_000) // 1000:03d}s"
    ego_str = f"  ego {ego_speed_mps*3.6:.1f} km/h"
    if ego_forward_speed_mps is not None:
        ego_str += f"  forward {ego_forward_speed_mps*3.6:.1f}"
    if ego_yaw_deg is not None:
        ego_str += f"  hdg {ego_yaw_deg % 360:.1f} deg"
    text = (f"frame={frame_idx}/{total_frames}  "
            f"dets={num_dets}  "
            f"thr={args_global.score_thr:.2f}"
            f"{ts_str}{ego_str}")
    cv2.putText(bar, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.54, FG_TEXT, 1, cv2.LINE_AA)
    return bar


# ─── joint_result.json 读取辅助 ─────────────────────────────────────────────

_MAP_TYPE_NAMES = {0: "divider", 1: "boundary", 2: "ped_crossing"}


def _convert_joint_dets(joint_dets: list) -> list:
    """将 joint_result.json detections 格式（x/y/z/l/w/h/class_id）
    转换为 video_demo.py 内部格式（center_xyz/size_xyz/label/label_name）。
    """
    out = []
    for d in joint_dets:
        cid = int(d.get("class_id", 0))
        out.append({
            "center_xyz": [d.get("x", 0), d.get("y", 0), d.get("z", 0)],
            "size_xyz":   [d.get("w", 1), d.get("l", 1), d.get("h", 1)],
            "yaw":        d.get("yaw", 0),
            "score":      d.get("score", 0),
            "label":      cid,
            "label_name": CLASS_NAMES[min(cid, len(CLASS_NAMES) - 1)],
        })
    return out


def _load_joint_result(frame_dir: "Path"):
    """读取 joint_result.json，返回 (detections, map_elements)。
    detections 已转换为 video_demo.py 内部格式；
    map_elements 格式与 map_result.json 的 elements 相同（type=str, points=[...]）。
    若文件不存在或解析失败，返回 (None, None)。
    """
    jr_path = frame_dir / "joint_result.json"
    if not jr_path.exists():
        return None, None
    try:
        with open(jr_path) as f:
            jr = json.load(f)
    except Exception:
        return None, None

    raw_dets = jr.get("detections", [])
    detections = _convert_joint_dets(raw_dets)

    map_elems = []
    map_sec = jr.get("map", {})
    for elem in map_sec.get("elements", []):
        raw_type = elem.get("type", 1)
        if isinstance(raw_type, str):
            type_name = raw_type
        else:
            try:
                type_name = _MAP_TYPE_NAMES.get(int(raw_type), "boundary")
            except (TypeError, ValueError):
                type_name = "boundary"

        pts = elem.get("points")
        if pts is None:
            pts = elem.get("pts", [])

        map_elems.append({
            "type": type_name,
            "points": pts,
            "is_polygon": bool(elem.get("is_polygon", False)),
            "score": elem.get("score", 0.0),
            "subtype": elem.get("subtype", type_name),
        })

    return detections, map_elems


# ─── 推理调用 ─────────────────────────────────────────────────────────────────

def run_inference_batch(binary: str, frames_dir: Path, model: str,
                        score_thr: float, classes: str) -> bool:
    """
    调用 C++ 批量推理（--batch 模式）：一次性处理所有帧。
    批量模式直接将每帧的 result.json 写入对应帧目录，无需拷贝。
    返回 True 表示成功。
    """
    cmd = [
        binary,
        str(frames_dir),
        model,
        "int8",
        "--batch",
        "--score-thr", str(score_thr),
        "--output-format", "json",
        "--no-warmup",
    ]
    if classes:
        cmd += ["--classes", classes]
    print(f"[批量推理] {' '.join(cmd)}")
    try:
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if ret.returncode != 0:
            print(f"  [推理失败] returncode={ret.returncode}")
            print(ret.stderr[:500])
            return False
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [推理错误] {e}")
        return False


def run_inference(binary: str, frame_dir: Path, model: str,
                  score_thr: float, classes: str) -> list:
    """
    单帧推理：调用 C++ 推理二进制，读取 JSON 输出。
    若 result.json 已存在则直接读取（避免重复推理）。
    """
    result_path = frame_dir / "result.json"
    if not result_path.exists():
        cmd = [
            binary,
            str(frame_dir),
            model,
            "int8",
            "--score-thr", str(score_thr),
            "--output-format", "json",
            "--no-warmup",
        ]
        if classes:
            cmd += ["--classes", classes]
        try:
            ret = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if ret.returncode != 0:
                print(f"  [推理失败] {frame_dir.name}: {ret.stderr[:200]}")
                return []
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  [推理错误] {e}")
            return []
        # 单帧模式的输出写到 model/<model>/result.json，拷贝到帧目录
        src = Path(f"model/{model}/result.json")
        if src.exists() and not result_path.exists():
            shutil.copy2(src, result_path)

    if not result_path.exists():
        return []
    with open(result_path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


# ─── 主流程 ───────────────────────────────────────────────────────────────────

# 全局 args（用于 compose_info_bar）
args_global = None


def parse_args():
    p = argparse.ArgumentParser(description="CUDA-FastBEV 多帧视频可视化")
    p.add_argument("--frames-dir", type=str, required=True,
                   help="包含 frame_* 子目录的根目录（由 nuscenes_adapter.py 生成）")
    p.add_argument("--out-dir",    type=str, default="outputs/video",
                   help="视频和帧图像输出目录")
    p.add_argument("--model",      type=str, default="resnet18",
                   help="模型名称（对应 model/<name>/build/ 下的引擎）")
    p.add_argument("--score-thr",  type=float, default=0.35,
                   help="可视化置信度阈值")
    p.add_argument("--classes",    type=str, default="",
                   help="类别过滤（逗号分隔的 id，空=全部）")
    p.add_argument("--bev-size",   type=int, default=800,
                   help="BEV 面板尺寸（像素）")
    p.add_argument("--cam-width",  type=int, default=480,
                   help="单路相机宽度（像素）")
    p.add_argument("--fps",        type=int, default=6,
                   help="输出视频帧率")
    p.add_argument("--video-name", type=str, default="fastbev_demo.mp4",
                   help="输出视频文件名")
    p.add_argument("--bev-mode", type=str, default="world", choices=["world", "ego"],
                   help="BEV 视角模式：world=以 ego 为中心但按全局朝向显示，ego=固定车头朝上")
    p.add_argument("--show-labels", action="store_true",
                   help="在 BEV 框旁显示类别与分数标签")
    p.add_argument("--ego-heading-offset-deg", type=float, default=0.0,
                   help="额外施加到自车朝向显示的偏角（度），用于调试 lidar 相对 ego 的固定偏航")
    p.add_argument("--save-frames",action="store_true",
                   help="同时保存每帧合成图像")
    p.add_argument("--auto-infer", action="store_true",
                   help="自动调用 C++ 推理（使用批量模式，一次处理所有帧）")
    p.add_argument("--per-frame-infer", action="store_true",
                   help="逐帧调用推理（慢，用于调试；默认 --auto-infer 使用批量模式）")
    p.add_argument("--binary",     type=str, default="./build/fastbev",
                   help="C++ 推理二进制路径")
    p.add_argument("--num-frames", type=int, default=None,
                   help="最多处理帧数（默认全部）")
    # ── 过滤器选项 ──
    p.add_argument("--no-size-filter", action="store_true",
                   help="禁用尺寸范围过滤")
    p.add_argument("--no-vel-filter",  action="store_true",
                   help="禁用速度合理性过滤")
    p.add_argument("--bev-nms-dist",   type=float, default=0.8,
                   help="BEV 中心距离 NMS 阈值（米），0=禁用，默认 0.8")
    p.add_argument("--no-map",          action="store_true",
                   help="禁用地图叠加（即使 map_result.json 存在也不显示）")
    p.add_argument("--joint",           action="store_true",
                   help="从 joint_result.json 读取检测和地图（joint_inference --save-json 的输出）")
    return p.parse_args()


def main():
    global args_global
    args = parse_args()
    args_global = args

    frames_dir = Path(args.frames_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_frames:
        (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    # 枚举帧目录
    if not frames_dir.exists():
        print(f"[错误] 帧目录不存在: {frames_dir.resolve()}")
        print(f"  请先运行: python tools/nuscenes_adapter.py --out-dir {args.frames_dir} ...")
        sys.exit(1)

    frame_dirs = sorted([d for d in frames_dir.iterdir()
                         if d.is_dir() and d.name.startswith("frame_")])
    if args.num_frames:
        frame_dirs = frame_dirs[:args.num_frames]

    if not frame_dirs:
        print(f"[错误] 未找到 frame_* 子目录: {frames_dir.resolve()}")
        print(f"  目录内容: {[d.name for d in frames_dir.iterdir()][:10]}")
        print(f"  请先运行: python tools/nuscenes_adapter.py --out-dir {args.frames_dir} ...")
        sys.exit(1)

    print(f"共 {len(frame_dirs)} 帧，开始渲染...")

    # ── 构建检测滤波器 ────────────────────────────────────────────────
    det_filter = None
    if _FILTER_AVAILABLE:
        class_ids = None
        if args.classes:
            class_ids = {int(c.strip()) for c in args.classes.split(',') if c.strip()}
        det_filter = DetectionFilter(FilterConfig(
            score_thr=args.score_thr,
            class_ids=class_ids,
            enable_size_filter=not args.no_size_filter,
            enable_velocity_filter=not args.no_vel_filter,
            bev_nms_dist=args.bev_nms_dist,
        ))
        print(f"[过滤器] score>={args.score_thr}  "
              f"size={'on' if not args.no_size_filter else 'off'}  "
              f"vel={'on' if not args.no_vel_filter else 'off'}  "
              f"bev-nms={args.bev_nms_dist}m")
    if args.joint:
        # joint 模式：读取 joint_result.json，不触发 C++ 推理
        missing_jr = [d for d in frame_dirs if not (d / "joint_result.json").exists()]
        if missing_jr:
            print(f"[joint 模式] {len(missing_jr)}/{len(frame_dirs)} 帧缺少 joint_result.json")
            print(f"  请先运行: ./build/joint_inference {args.frames_dir} <model> <precision> --save-json")
        else:
            print(f"[joint 模式] 所有帧已有 joint_result.json，直接读取")
    else:
        missing = [d for d in frame_dirs if not (d / "result.json").exists()]
        if missing:
            print(f"[推理] {len(missing)}/{len(frame_dirs)} 帧缺少 result.json，触发批量推理...")
            if not Path(args.binary).exists():
                print(f"[错误] 推理二进制不存在: {args.binary}")
                print(f"  请先编译: cd build && cmake .. && make -j$(nproc)")
                sys.exit(1)
            ok = run_inference_batch(args.binary, frames_dir, args.model,
                                     args.score_thr, args.classes)
            if not ok:
                print("  [降级] 批量推理失败，尝试逐帧推理...")
                args.per_frame_infer = True
            else:
                still_missing = [d for d in frame_dirs if not (d / "result.json").exists()]
                if still_missing:
                    print(f"  [警告] 批量推理后仍有 {len(still_missing)} 帧缺少结果")
        else:
            print(f"[推理] 所有帧已有 result.json，直接读取")

    # 视频写入器（延迟初始化）
    video_writer = None
    video_path   = out_dir / args.video_name

    # ego 状态跟踪（用于计算帧间速度）
    prev_ego_translation = None
    prev_timestamp       = None
    prev_ego_yaw_global  = None

    for render_idx, frame_dir in enumerate(frame_dirs):
        # ── 1. 读取或执行推理 ──
        if getattr(args, 'joint', False):
            # joint 模式：从 joint_result.json 读取检测 + 地图
            _jdets, _jmap = _load_joint_result(frame_dir)
            if _jdets is None:
                print(f"  [警告] {frame_dir.name}：joint_result.json 缺失，该帧无结果")
                all_dets = []
                _jmap = []
            else:
                all_dets = _jdets
        elif getattr(args, 'per_frame_infer', False):
            all_dets = run_inference(
                args.binary, frame_dir, args.model,
                args.score_thr, args.classes)
            _jmap = None
        else:
            result_path = frame_dir / "result.json"
            if result_path.exists():
                with open(result_path) as f:
                    try:
                        all_dets = json.load(f)
                    except json.JSONDecodeError:
                        all_dets = []
            else:
                print(f"  [警告] {frame_dir.name}：result.json 缺失，该帧无检测结果")
                all_dets = []
            _jmap = None

        # 读取 meta.json（时间戳 + ego 位姿）
        timestamp            = 0
        ego_translation      = None
        ego_yaw_global       = None
        lidar_yaw_in_ego     = 0.0
        meta_path = frame_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            timestamp       = meta.get("timestamp", 0)
            ego_translation = meta.get("ego_translation_global", None)
            ego_yaw_global  = meta.get("ego_yaw_global",         None)
            lidar_yaw_in_ego = meta.get("lidar_yaw_in_ego",      0.0)

        # 计算 ego 速度（相邻帧全局位移 / 时间差）
        ego_speed_mps = 0.0
        ego_forward_speed_mps = None
        ego_motion_yaw_global = None
        if (prev_ego_translation is not None and ego_translation is not None
                and prev_timestamp and timestamp and timestamp > prev_timestamp):
            dt = (timestamp - prev_timestamp) / 1e6   # 微秒 → 秒
            dx = ego_translation[0] - prev_ego_translation[0]
            dy = ego_translation[1] - prev_ego_translation[1]
            ego_speed_mps = math.sqrt(dx**2 + dy**2) / max(dt, 1e-6)
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                ego_motion_yaw_global = math.atan2(dy, dx)
            yaw_for_velocity = ego_yaw_global if ego_yaw_global is not None else prev_ego_yaw_global
            if yaw_for_velocity is not None:
                ego_forward_speed_mps = (dx * math.cos(yaw_for_velocity) +
                                         dy * math.sin(yaw_for_velocity)) / max(dt, 1e-6)
        prev_ego_translation = ego_translation
        prev_timestamp       = timestamp
        prev_ego_yaw_global  = ego_yaw_global

        # 按 score_thr 过滤（或使用完整规则过滤器）
        if det_filter is not None:
            detections = det_filter.filter(all_dets)
        else:
            detections = [d for d in all_dets if d.get("score", 0) >= args.score_thr]
        display_yaw_global = None
        if ego_motion_yaw_global is not None:
            display_yaw_global = ego_motion_yaw_global + math.radians(args.ego_heading_offset_deg)
        elif ego_yaw_global is not None:
            display_yaw_global = (
                float(ego_yaw_global)
                + float(lidar_yaw_in_ego)
                + math.radians(args.ego_heading_offset_deg)
            )

        # ── 2. 读取地图元素（若存在）──
        map_elements = None
        if not getattr(args, 'no_map', False):
            if getattr(args, 'joint', False):
                # joint 模式：地图已在步骤 1 读取
                map_elements = _jmap if _jmap else None
            else:
                map_json = frame_dir / "map_result.json"
                if map_json.exists():
                    try:
                        with open(map_json) as _mf:
                            _mr = json.load(_mf)
                        map_elements = _mr.get("elements", [])
                    except Exception:
                        map_elements = None

        # ── 3. 渲染 BEV 面板（含自车速度/朝向/地图）──
        bev_panel = render_bev_panel(
            detections, BEV_BOUNDS, args.bev_size,
            ego_speed_mps=ego_speed_mps,
            ego_yaw_global=ego_yaw_global,
            display_yaw_global=display_yaw_global,
            ego_motion_yaw_global=ego_motion_yaw_global,
            ego_speed_forward_mps=ego_forward_speed_mps,
            bev_mode=args.bev_mode,
            show_labels=args.show_labels,
            map_elements=map_elements)

        # ── 4. 组合相机网格（含 3D 框投影）──
        cam_grid = compose_camera_grid(frame_dir, detections, args.cam_width)
        cam_h = cam_grid.shape[0]
        bev_resized = cv2.resize(bev_panel, (int(bev_panel.shape[1] * cam_h / bev_panel.shape[0]), cam_h))
        combined = np.concatenate([cam_grid, bev_resized], axis=1)

        # ── 5. 添加信息栏（含 ego 速度/朝向）──
        info_bar = compose_info_bar(
            render_idx + 1, len(frame_dirs),
            len(detections), combined.shape[1],
            timestamp=timestamp,
            ego_speed_mps=ego_speed_mps,
            ego_yaw_deg=math.degrees(ego_yaw_global) if ego_yaw_global is not None else None,
            ego_forward_speed_mps=ego_forward_speed_mps)
        frame_out = np.concatenate([combined, info_bar], axis=0)

        # ── 6. 初始化视频写入器 ──
        if video_writer is None:
            h, w = frame_out.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
            print(f"视频分辨率: {w}×{h} @{args.fps}fps → {video_path}")

        video_writer.write(frame_out)

        # ── 7. 可选保存帧图像 ──
        if args.save_frames:
            frame_img_path = out_dir / "frames" / f"{render_idx:05d}.jpg"
            cv2.imwrite(str(frame_img_path), frame_out)

        if render_idx % 10 == 0 or render_idx == len(frame_dirs) - 1:
            print(f"  渲染进度: {render_idx + 1}/{len(frame_dirs)}  "
                  f"检测框: {len(detections)}")

    if video_writer:
        video_writer.release()
        print(f"\n视频已生成: {video_path}")
    else:
        print("[警告] 没有帧被写入视频")


if __name__ == "__main__":
    main()
