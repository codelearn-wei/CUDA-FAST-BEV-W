#!/usr/bin/env python3
"""
video_tracking_map.py — BEV 跟踪轨迹 + 地图联合视频生成器

输入：各帧 joint_result.json（由 run_pipeline --save-json 生成）
      含 tracks / detections / map 三个字段

输出：tracking_map.mp4
      画面内容：
        - BEV 俯视图（深色背景 + 格网）
        - 地图叠加（boundary / divider / ped_crossing）
        - 原始检测框（灰色半透明）
        - 跟踪框（类别颜色 + track_id 标签）
        - 历史轨迹 trail（渐变透明度）
        - 速度箭头
        - 自车标记 + 图例

用法:
  python tools/video_tracking_map.py \\
      --frames-dir outputs/frames1 \\
      --out-dir    outputs/video_fixed \\
      --fps 6 --bev-size 800
"""

import sys
import os
import json
import math
import argparse
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2

# ─── 导入 video_demo 渲染工具 ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from video_demo import (
        CLASS_NAMES, CLASS_COLORS,
        metric_to_canvas, model_local_to_display,
        render_bev_background, render_map_overlay,
        boxes_to_corners_3d, BEV_BOUNDS, BEV_EDGES,
        alpha_blend, draw_text_tag, EGO_COLOR,
        BG_PRIMARY, FG_GRID, FG_TEXT, FG_SUBTEXT,
        transform_model_local_to_world_relative,
    )
    _HAS_VIDEO_DEMO = True
except ImportError as e:
    print(f"[警告] 无法导入 video_demo: {e}，使用内置定义")
    _HAS_VIDEO_DEMO = False

if not _HAS_VIDEO_DEMO:
    # ── 内置 fallback 定义 ────────────────────────────────────────────────
    CLASS_NAMES = [
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
    ]
    CLASS_COLORS = {
        "car":                  (255, 140,   0),
        "truck":                ( 70, 170, 255),
        "construction_vehicle": (  0, 200, 255),
        "bus":                  ( 40, 120, 255),
        "trailer":              (180, 110, 255),
        "barrier":              (180, 180, 180),
        "motorcycle":           (  0, 220, 120),
        "bicycle":              ( 40, 255, 200),
        "pedestrian":           (255,  90, 180),
        "traffic_cone":         (160, 200,  80),
    }
    BG_PRIMARY = (22, 28, 26)
    FG_GRID    = (62, 72, 68)
    FG_TEXT    = (235, 238, 240)
    FG_SUBTEXT = (168, 176, 180)
    EGO_COLOR  = (245, 245, 245)

    BEV_BOUNDS  = (-15.0, 60.0, -25.0, 25.0)   # (fwd_min, fwd_max, left_min, left_max)
    BEV_EDGES   = [(0, 1), (1, 2), (2, 3), (3, 0)]

    def alpha_blend(base, overlay, alpha):
        return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)

    def draw_text_tag(image, text, anchor, color, font_scale=0.45):
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        x, y = int(anchor[0]), int(anchor[1])
        tl = (x, max(0, y - th - base - 6))
        br = (min(image.shape[1]-1, x+tw+8), y)
        cv2.rectangle(image, tl, br, (12, 12, 12), -1)
        cv2.rectangle(image, tl, br, color, 1)
        cv2.putText(image, text, (tl[0]+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245,245,245), 1, cv2.LINE_AA)

    def metric_to_canvas(points_xy: np.ndarray, bounds, canvas_size: int):
        fwd_min, fwd_max, left_min, left_max = bounds
        width  = float(max(left_max  - left_min,  1e-6))
        height = float(max(fwd_max   - fwd_min,   1e-6))
        px = (left_max - points_xy[:, 1]) / width  * (canvas_size - 1)
        py = (fwd_max  - points_xy[:, 0]) / height * (canvas_size - 1)
        return np.stack([px, py], axis=1).astype(np.int32)

    def model_local_to_display(points_xy: np.ndarray) -> np.ndarray:
        if points_xy.size == 0:
            return points_xy
        out = np.empty_like(points_xy, dtype=np.float32)
        out[:, 0] = points_xy[:, 1]
        out[:, 1] = -points_xy[:, 0]
        return out

    def transform_model_local_to_world_relative(points_xy, heading_global):
        if points_xy.size == 0:
            return points_xy
        c = np.cos(heading_global)
        s = np.sin(heading_global)
        local_right   = points_xy[:, 0]
        local_forward = points_xy[:, 1]
        global_x = local_forward * c + local_right * s
        global_y = local_forward * s - local_right * c
        out = np.empty_like(points_xy, dtype=np.float32)
        out[:, 0] = global_y
        out[:, 1] = -global_x
        return out

    def boxes_to_corners_3d(detections: list) -> np.ndarray:
        offsets = np.array([
            [-1,-1,-1],[+1,-1,-1],[+1,+1,-1],[-1,+1,-1],
            [-1,-1,+1],[+1,-1,+1],[+1,+1,+1],[-1,+1,+1],
        ], dtype=np.float32)
        corners_list = []
        for det in detections:
            x, y, z = det.get("center_xyz", [0,0,0])
            w, l, h  = det.get("size_xyz",   [1,1,1])
            yaw      = float(det.get("yaw", 0))
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            std = offsets.copy()
            std[:, 0] *= w * 0.5
            std[:, 1] *= l * 0.5
            std[:, 2] *= h * 0.5
            corners = np.empty_like(std)
            corners[:, 0] = x + std[:, 0]*cos_y + std[:, 1]*sin_y
            corners[:, 1] = y + std[:, 0]*(-sin_y) + std[:, 1]*cos_y
            corners[:, 2] = z + std[:, 2]
            corners_list.append(corners)
        if not corners_list:
            return np.zeros((0, 8, 3), dtype=np.float32)
        return np.stack(corners_list, axis=0)

    def render_bev_background(canvas_size: int, bounds) -> np.ndarray:
        canvas = np.full((canvas_size, canvas_size, 3), BG_PRIMARY, dtype=np.uint8)
        fwd_min, fwd_max, left_min, left_max = bounds
        for r_m in [10, 20, 30, 40, 50]:
            pts = []
            for ang in range(361):
                px = r_m * math.cos(math.radians(ang))
                py = r_m * math.sin(math.radians(ang))
                c = metric_to_canvas(
                    np.array([[py, px]], dtype=np.float32), bounds, canvas_size)
                pts.append(tuple(c[0]))
            pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts_arr], True, FG_GRID, 1, cv2.LINE_AA)
        # ego axes
        origin = metric_to_canvas(np.array([[0,0]], dtype=np.float32), bounds, canvas_size)[0]
        fwd    = metric_to_canvas(np.array([[5,0]], dtype=np.float32), bounds, canvas_size)[0]
        right  = metric_to_canvas(np.array([[0,5]], dtype=np.float32), bounds, canvas_size)[0]
        cv2.arrowedLine(canvas, tuple(origin), tuple(fwd),   (100,200,100), 1, tipLength=0.2)
        cv2.arrowedLine(canvas, tuple(origin), tuple(right), (100,100,200), 1, tipLength=0.2)
        return canvas

    def render_map_overlay(canvas, map_elements, bounds, bev_size, **kwargs):
        if not map_elements:
            return
        _MAP_COLORS = {
            "divider":      (0, 220, 220),
            "ped_crossing": (80, 120, 255),
            "boundary":     (140, 200, 90),
        }
        overlay = canvas.copy()
        for elem in map_elements:
            t = elem.get("type", "boundary")
            pts_raw = elem.get("points", elem.get("pts", []))
            if not pts_raw:
                continue
            pts_np = np.array(pts_raw, dtype=np.float32)
            if pts_np.ndim == 1:
                pts_np = pts_np.reshape(-1, 2)
            pts_np = pts_np[:, :2]
            # convert (x=right, y=forward) → display (forward, left)
            disp = np.empty_like(pts_np)
            disp[:, 0] = pts_np[:, 1]
            disp[:, 1] = -pts_np[:, 0]
            pts_c = metric_to_canvas(disp, bounds, bev_size)
            color = _MAP_COLORS.get(t, (160, 160, 160))
            if elem.get("is_polygon", False):
                cv2.fillPoly(overlay, [pts_c.reshape(-1,1,2)], color)
            else:
                cv2.polylines(overlay, [pts_c.reshape(-1,1,2)], False, color, 2, cv2.LINE_AA)
        canvas[:] = alpha_blend(canvas, overlay, 0.30)


# ─── 常数 ────────────────────────────────────────────────────────────────────

TRAIL_LEN     = 15      # 历史轨迹保留帧数
TRAIL_MIN_ALPHA = 0.10  # 最老历史点的透明度
TRAIL_MAX_ALPHA = 0.70  # 最新历史点的透明度

_MAP_TYPE_NAMES = {0: "divider", 1: "boundary", 2: "ped_crossing"}


# ─── 数据加载 ────────────────────────────────────────────────────────────────

def _convert_dets(raw_dets: list) -> list:
    """将 x/y/z/l/w/h/class_id 格式转为渲染格式（center_xyz/size_xyz/label_name）。"""
    out = []
    for d in raw_dets:
        cid = int(d.get("class_id", d.get("label", 0)))
        out.append({
            "center_xyz": [d.get("x", 0), d.get("y", 0), d.get("z", 0)],
            "size_xyz":   [d.get("w", 1), d.get("l", 1), d.get("h", 1)],
            "yaw":        d.get("yaw", 0),
            "score":      d.get("score", 0),
            "label":      cid,
            "label_name": CLASS_NAMES[min(cid, len(CLASS_NAMES)-1)],
        })
    return out


def _convert_tracks(raw_tracks: list) -> list:
    """将 tracks.json 格式转为渲染格式（center_xyz/size_xyz/label_name/track_id/velocity）。"""
    out = []
    for t in raw_tracks:
        cid = int(t.get("class_id", 0))
        pos = t.get("position", [0, 0, 0])
        sz  = t.get("size", [1, 1, 1])
        vel = t.get("velocity", [0, 0])
        out.append({
            "track_id":   t.get("track_id", -1),
            "center_xyz": pos,
            "size_xyz":   sz,
            "yaw":        t.get("yaw", 0),
            "score":      t.get("score", 0),
            "label":      cid,
            "label_name": CLASS_NAMES[min(cid, len(CLASS_NAMES)-1)],
            "velocity":   vel,
        })
    return out


def _parse_map_elements(map_sec: dict) -> list:
    """解析 joint_result.json 中的 map 字段为 render_map_overlay 所需格式。"""
    elems = []
    for elem in map_sec.get("elements", []):
        raw_type = elem.get("type", 1)
        if isinstance(raw_type, str):
            type_name = raw_type
        else:
            try:
                type_name = _MAP_TYPE_NAMES.get(int(raw_type), "boundary")
            except (TypeError, ValueError):
                type_name = "boundary"
        pts = elem.get("points", elem.get("pts", []))
        elems.append({
            "type":       type_name,
            "points":     pts,
            "is_polygon": bool(elem.get("is_polygon", False)),
            "score":      elem.get("score", 0.0),
        })
    return elems


def load_frames(frames_dir: Path):
    """加载所有帧数据，返回帧列表（dicts）。"""
    frame_paths = sorted(
        d for d in frames_dir.iterdir()
        if d.is_dir() and d.name.startswith("frame_")
    )
    frames = []
    missing = 0
    for fd in frame_paths:
        jr_path = fd / "joint_result.json"
        if not jr_path.exists():
            missing += 1
            frames.append({
                "dir": fd, "detections": [], "tracks": [], "map": [],
                "timestamp": 0, "ego_yaw_global": None,
                "ego_translation": None,
            })
            continue
        try:
            with open(jr_path) as f:
                jr = json.load(f)
        except Exception:
            missing += 1
            frames.append({"dir": fd, "detections": [], "tracks": [], "map": [],
                            "timestamp": 0, "ego_yaw_global": None,
                            "ego_translation": None})
            continue

        dets   = _convert_dets(jr.get("detections", []))
        tracks = _convert_tracks(jr.get("tracks", []))
        map_sec      = jr.get("map", {})
        map_elements = _parse_map_elements(map_sec)

        # 优先从 map 段读取 ego 位姿（更精确，是 MapTR 推理时刻的位姿）
        ego_yaw   = None
        ego_trans = None
        timestamp = jr.get("timestamp", 0)
        if isinstance(map_sec, dict):
            et = map_sec.get("ego_translation")
            ey = map_sec.get("ego_yaw")
            if et is not None and len(et) >= 2:
                ego_trans = et
                ego_yaw   = ey

        # 退回到 meta.json
        if ego_yaw is None:
            meta_path = fd / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    ego_yaw   = meta.get("ego_yaw_global")
                    ego_trans = meta.get("ego_translation_global")
                    if not timestamp:
                        timestamp = meta.get("timestamp", 0)
                except Exception:
                    pass

        frames.append({
            "dir":             fd,
            "detections":      dets,
            "tracks":          tracks,
            "map":             map_elements,
            "map_has_data":    bool(map_elements),
            "timestamp":       timestamp,
            "ego_yaw_global":  ego_yaw,
            "ego_translation": ego_trans,
        })

    if missing:
        print(f"  [警告] {missing}/{len(frame_paths)} 帧缺少 joint_result.json")
    return frames


# ─── ego 运动补偿 ────────────────────────────────────────────────────────────

def compensate_map_ego(map_elements: list,
                       src_trans: list, src_yaw: float,
                       dst_trans: list, dst_yaw: float) -> list:
    """
    将 map_elements 中的点集从 src_ego 坐标系变换到 dst_ego 坐标系。

    每个 element 的 points 是 [[x, y], ...] 格式，坐标在 ego frame 里。
    变换路径：src_ego → global → dst_ego
      p_global = R(src_yaw) * p + src_trans
      p_dst    = R(-dst_yaw) * (p_global - dst_trans)
    """
    import copy
    if not map_elements:
        return map_elements
    cs, ss = math.cos(src_yaw), math.sin(src_yaw)
    cd, sd = math.cos(dst_yaw), math.sin(dst_yaw)
    sx, sy = float(src_trans[0]), float(src_trans[1])
    dx, dy = float(dst_trans[0]), float(dst_trans[1])

    compensated = []
    for elem in map_elements:
        new_elem = copy.copy(elem)
        old_pts = elem.get("points", [])
        new_pts = []
        for pt in old_pts:
            px, py = float(pt[0]), float(pt[1])
            # src_ego → global
            gx = sx + cs * px - ss * py
            gy = sy + ss * px + cs * py
            # global → dst_ego  (R^-1(dst_yaw) = [[cd, sd], [-sd, cd]])
            ddx, ddy = gx - dx, gy - dy
            new_pts.append([ cd * ddx + sd * ddy,
                            -sd * ddx + cd * ddy])
        new_elem = dict(elem)
        new_elem["points"] = new_pts
        compensated.append(new_elem)
    return compensated


# ─── BEV 渲染 ────────────────────────────────────────────────────────────────

def _pts_to_canvas(pts_xy: np.ndarray, ego_yaw, bounds, bev_size):
    """将检测坐标系点转为画布坐标（自动选择 world / local 模式）。"""
    if ego_yaw is not None:
        disp = transform_model_local_to_world_relative(
            pts_xy.reshape(-1, 2), float(ego_yaw))
    else:
        disp = model_local_to_display(pts_xy.reshape(-1, 2))
    return metric_to_canvas(disp, bounds, bev_size)


def render_ego(canvas: np.ndarray, bounds, bev_size: int):
    """绘制自车标记（白色矩形 + 朝向箭头）。"""
    origin = metric_to_canvas(np.array([[0, 0]], dtype=np.float32), bounds, bev_size)[0]
    front  = metric_to_canvas(np.array([[4, 0]], dtype=np.float32), bounds, bev_size)[0]
    # 车体框
    hw, hl = 10, 16
    pts = np.array([
        [origin[0]-hw, origin[1]+hl],
        [origin[0]+hw, origin[1]+hl],
        [origin[0]+hw, origin[1]-hl],
        [origin[0]-hw, origin[1]-hl],
    ], dtype=np.int32)
    cv2.polylines(canvas, [pts.reshape(-1,1,2)], True, EGO_COLOR, 2)
    cv2.arrowedLine(canvas, tuple(origin), tuple(front),
                    EGO_COLOR, 2, cv2.LINE_AA, tipLength=0.4)


def render_detections_dim(
    canvas: np.ndarray,
    detections: list,
    ego_yaw,
    bounds,
    bev_size: int,
    alpha: float = 0.25,
):
    """绘制原始检测框（灰色半透明，表示"未确认"目标）。"""
    if not detections:
        return
    corners_3d = boxes_to_corners_3d(detections)
    overlay = canvas.copy()
    for idx, det in enumerate(detections):
        bottom_world = corners_3d[idx, :4, :2].astype(np.float32)
        bottom_c = _pts_to_canvas(bottom_world, ego_yaw, bounds, bev_size)
        color = (100, 100, 100)
        cv2.polylines(overlay, [bottom_c.reshape(-1,1,2)], True, color, 1, cv2.LINE_AA)
    canvas[:] = alpha_blend(canvas, overlay, alpha)


def render_tracks(
    canvas: np.ndarray,
    tracks: list,
    track_history: dict,   # track_id → [(x,y), ...]
    ego_yaw,
    bounds,
    bev_size: int,
    show_velocity: bool = True,
    show_label: bool = True,
):
    """绘制跟踪框 + 轨迹 trail + 速度箭头。"""
    if not tracks:
        return {}

    corners_3d = boxes_to_corners_3d(tracks)
    legend = {}

    for idx, trk in enumerate(tracks):
        tid    = trk.get("track_id", -1)
        label  = trk.get("label_name", "car")
        color  = CLASS_COLORS.get(label, (180, 180, 180))
        cx, cy, _ = trk.get("center_xyz", [0, 0, 0])
        vx, vy    = trk.get("velocity", [0, 0])

        # ── 轨迹 trail ────────────────────────────────────────────────────
        history = track_history.get(tid, [])
        if len(history) >= 2:
            n = len(history)
            for hi in range(n - 1):
                p0 = np.array([history[hi]],     dtype=np.float32)
                p1 = np.array([history[hi + 1]], dtype=np.float32)
                c0 = _pts_to_canvas(p0, ego_yaw, bounds, bev_size)[0]
                c1 = _pts_to_canvas(p1, ego_yaw, bounds, bev_size)[0]
                # 透明度：越新越不透明
                ratio = (hi + 1) / n
                a = TRAIL_MIN_ALPHA + (TRAIL_MAX_ALPHA - TRAIL_MIN_ALPHA) * ratio
                trail_color = tuple(int(c * a) for c in color)
                thickness = max(1, int(2 * ratio))
                cv2.line(canvas, tuple(c0), tuple(c1),
                         trail_color, thickness, cv2.LINE_AA)

        # ── 跟踪框 ────────────────────────────────────────────────────────
        bottom_world = corners_3d[idx, :4, :2].astype(np.float32)
        bottom_c     = _pts_to_canvas(bottom_world, ego_yaw, bounds, bev_size)
        center_world = np.array([[cx, cy]], dtype=np.float32)
        center_c     = _pts_to_canvas(center_world, ego_yaw, bounds, bev_size)[0]

        # 朝向线（head 为前两角中点）
        head_world = corners_3d[idx, [2, 3], :2].mean(axis=0, keepdims=True).astype(np.float32)
        head_c     = _pts_to_canvas(head_world, ego_yaw, bounds, bev_size)[0]

        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, bottom_c, color)
        canvas[:] = alpha_blend(canvas, overlay, 0.30)

        cv2.polylines(canvas, [bottom_c.reshape(-1,1,2)], True, color, 2, cv2.LINE_AA)
        cv2.line(canvas, tuple(center_c), tuple(head_c), color, 3, cv2.LINE_AA)

        # ── 速度箭头 ──────────────────────────────────────────────────────
        if show_velocity:
            speed = math.sqrt(vx**2 + vy**2)
            if speed > 0.3:   # 静止目标不画箭头
                scale = 1.5
                vel_world = np.array([[cx + vx*scale, cy + vy*scale]], dtype=np.float32)
                vel_c = _pts_to_canvas(vel_world, ego_yaw, bounds, bev_size)[0]
                cv2.arrowedLine(canvas, tuple(center_c), tuple(vel_c),
                                color, 2, cv2.LINE_AA, tipLength=0.35)

        # ── track_id 标签 ─────────────────────────────────────────────────
        if show_label:
            label_str = f"{label[:3]} #{tid}"
            anchor = (bottom_c[0][0] + 4, bottom_c[0][1] - 4)
            draw_text_tag(canvas, label_str, anchor, color, font_scale=0.38)

        legend[label] = legend.get(label, 0) + 1

    return legend


def render_legend(canvas: np.ndarray, legend: dict, track_count: int,
                  det_count: int, map_count: int, frame_idx: int, fps: float):
    """右上角绘制图例 + 统计信息。"""
    h, w = canvas.shape[:2]
    y = 20
    # 帧信息
    cv2.putText(canvas, f"Frame {frame_idx:04d}",
                (w - 130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                FG_TEXT, 1, cv2.LINE_AA)
    y += 20
    cv2.putText(canvas, f"Tracks: {track_count}",
                (w - 130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (140, 220, 140), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(canvas, f"Dets: {det_count}",
                (w - 130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                FG_SUBTEXT, 1, cv2.LINE_AA)
    y += 18
    cv2.putText(canvas, f"Map: {map_count}",
                (w - 130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 200, 200), 1, cv2.LINE_AA)
    y += 28
    # 类别图例
    for lbl, cnt in sorted(legend.items(), key=lambda x: -x[1]):
        color = CLASS_COLORS.get(lbl, (180, 180, 180))
        cv2.rectangle(canvas, (w-130, y-10), (w-122, y+2), color, -1)
        cv2.putText(canvas, f"{lbl[:10]}: {cnt}",
                    (w-118, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    FG_TEXT, 1, cv2.LINE_AA)
        y += 16


def render_scale_bar(canvas: np.ndarray, bounds, bev_size: int):
    """左下角绘制比例尺（10m）。"""
    bar_m = 10.0
    fwd_min, fwd_max, left_min, left_max = bounds
    width_m = left_max - left_min
    bar_px = int(bar_m / width_m * bev_size)
    x0, y0 = 20, bev_size - 25
    x1 = x0 + bar_px
    cv2.line(canvas, (x0, y0), (x1, y0), FG_TEXT, 2)
    cv2.line(canvas, (x0, y0-4), (x0, y0+4), FG_TEXT, 2)
    cv2.line(canvas, (x1, y0-4), (x1, y0+4), FG_TEXT, 2)
    cv2.putText(canvas, "10m", (x0 + bar_px//2 - 15, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, FG_TEXT, 1, cv2.LINE_AA)


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="BEV 跟踪轨迹 + 地图联合视频生成器",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--frames-dir", default="outputs/frames",
                   help="帧目录（含 frame_XXXXX 子目录）")
    p.add_argument("--out-dir",    default="outputs/video",
                   help="输出视频目录")
    p.add_argument("--video-name", default="tracking_map.mp4",
                   help="输出视频文件名（默认 tracking_map.mp4）")
    p.add_argument("--fps",        type=float, default=6.0,
                   help="视频帧率（默认 6）")
    p.add_argument("--bev-size",   type=int, default=800,
                   help="BEV 画布尺寸（像素，默认 800）")
    p.add_argument("--trail-len",  type=int, default=TRAIL_LEN,
                   help=f"轨迹历史帧数（默认 {TRAIL_LEN}）")
    p.add_argument("--no-map",     action="store_true",
                   help="不绘制地图")
    p.add_argument("--no-dets",    action="store_true",
                   help="不绘制原始检测框（仅跟踪框）")
    p.add_argument("--no-velocity",action="store_true",
                   help="不绘制速度箭头")
    p.add_argument("--score-thr",  type=float, default=0.0,
                   help="跟踪结果 score 过滤阈值（默认 0，显示所有）")
    return p.parse_args()


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载帧数据 ────────────────────────────────────────────────────────
    print(f"[video_tracking_map] 读取帧目录: {frames_dir}")
    frames = load_frames(frames_dir)
    if not frames:
        print("  错误：未找到任何帧数据")
        sys.exit(1)
    print(f"  加载 {len(frames)} 帧，开始渲染...")

    bounds   = BEV_BOUNDS
    bev_size = args.bev_size

    # ── 视频写入器 ────────────────────────────────────────────────────────
    video_path = out_dir / args.video_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc,
                              args.fps, (bev_size, bev_size))
    if not writer.isOpened():
        print(f"  错误：无法创建视频写入器: {video_path}")
        sys.exit(1)
    print(f"  视频分辨率: {bev_size}×{bev_size} @{args.fps}fps → {video_path}")

    # ── 轨迹历史缓冲 ─────────────────────────────────────────────────────
    track_history: dict = defaultdict(list)   # track_id → [(cx, cy), ...]
    trail_len = args.trail_len

    # ego 运动补偿所需的"上一次有效地图"缓存
    last_valid_map_elems = []
    last_valid_ego_trans = None
    last_valid_ego_yaw   = None

    # ── 逐帧渲染 ─────────────────────────────────────────────────────────
    for fi, frame in enumerate(frames):
        ego_yaw   = frame.get("ego_yaw_global")
        ego_trans = frame.get("ego_translation")
        dets      = frame["detections"]
        tracks    = [t for t in frame["tracks"]
                     if t.get("score", 1) >= args.score_thr]

        # ── 地图：有则直接用，无则 ego 补偿复用上帧 ──────────────────────
        raw_map_elems = frame["map"] if not args.no_map else []
        if raw_map_elems:
            map_elems = raw_map_elems
            # 缓存本帧地图 + ego pose（供后续帧补偿用）
            if ego_trans is not None and ego_yaw is not None:
                last_valid_map_elems = raw_map_elems
                last_valid_ego_trans = ego_trans
                last_valid_ego_yaw   = ego_yaw
        elif (not args.no_map
              and last_valid_map_elems
              and last_valid_ego_trans is not None
              and ego_trans is not None
              and ego_yaw is not None):
            # ego 运动补偿：将上帧地图从 last_valid_ego 变换到当前 ego
            map_elems = compensate_map_ego(
                last_valid_map_elems,
                last_valid_ego_trans, last_valid_ego_yaw,
                ego_trans, ego_yaw,
            )
        else:
            map_elems = raw_map_elems  # 无补偿信息，空地图或原样使用

        # 更新轨迹历史（每帧追加当前位置）
        active_ids = set()
        for trk in tracks:
            tid = trk.get("track_id", -1)
            cx, cy, _ = trk.get("center_xyz", [0, 0, 0])
            track_history[tid].append((cx, cy))
            if len(track_history[tid]) > trail_len:
                track_history[tid] = track_history[tid][-trail_len:]
            active_ids.add(tid)

        # ── 渲染 BEV 画布 ─────────────────────────────────────────────────
        canvas = render_bev_background(bounds, bev_size)

        # 地图叠加
        if map_elems:
            render_map_overlay(canvas, map_elems, bounds, bev_size)

        # 原始检测框（灰色半透明）
        if not args.no_dets and dets:
            render_detections_dim(canvas, dets, ego_yaw, bounds, bev_size,
                                  alpha=0.30)

        # 跟踪框 + 轨迹
        legend = render_tracks(
            canvas, tracks, track_history,
            ego_yaw, bounds, bev_size,
            show_velocity=not args.no_velocity,
            show_label=True,
        )

        # 自车标记
        render_ego(canvas, bounds, bev_size)

        # 图例 + 比例尺
        render_legend(canvas, legend,
                      track_count=len(tracks),
                      det_count=len(dets),
                      map_count=len(map_elems),
                      frame_idx=fi,
                      fps=args.fps)
        render_scale_bar(canvas, bounds, bev_size)

        # 标题（标记地图来源）
        if not map_elems:
            map_label = "Map: N/A"
        elif frame.get("map_has_data"):
            map_label = f"Map: {len(map_elems)} (fresh)"
        else:
            map_label = f"Map: {len(map_elems)} (ego-comp)"
        title = f"BEV Tracking + Map  |  {map_label}"
        cv2.putText(canvas, title, (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, FG_TEXT, 1, cv2.LINE_AA)

        writer.write(canvas)

        if (fi + 1) % 10 == 0 or fi + 1 == len(frames):
            print(f"  渲染进度: {fi+1}/{len(frames)}"
                  f"  tracks={len(tracks)}  map={len(map_elems)}")

    writer.release()
    print(f"\n视频已生成: {video_path}")
    print(f"  帧数: {len(frames)}  时长: {len(frames)/args.fps:.1f}s  @{args.fps}fps")


if __name__ == "__main__":
    main()
