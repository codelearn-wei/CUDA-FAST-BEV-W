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
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

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
    """将 [N, 2] 的 lidar xy 坐标转为画布像素坐标 [N, 2]。"""
    fwd_min, fwd_max, left_min, left_max = bounds
    width  = float(max(left_max  - left_min,  1e-6))
    height = float(max(fwd_max   - fwd_min,   1e-6))
    px = (points_xy[:, 1] - left_min) / width  * (canvas_size - 1)
    py = (fwd_max - points_xy[:, 0])  / height * (canvas_size - 1)
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


# ─── BEV 背景渲染 ─────────────────────────────────────────────────────────────

def render_bev_background(bounds, bev_size: int) -> np.ndarray:
    """绘制带同心圆格网的深色 BEV 背景。"""
    canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    # 渐变背景
    grad = np.linspace(42, 16, bev_size, dtype=np.uint8)
    canvas[:, :, 0] = (grad * 0.65).astype(np.uint8)[:, None]
    canvas[:, :, 1] = grad[:, None]
    canvas[:, :, 2] = (grad * 0.95).astype(np.uint8)[:, None]

    fwd_min, fwd_max, left_min, left_max = bounds
    max_r = int(max(abs(fwd_min), abs(fwd_max), abs(left_min), abs(left_max)))

    # 同心圆刻度
    for meter in range(10, max_r + 1, 10):
        angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        circle_pts = np.column_stack([meter * np.cos(angles), meter * np.sin(angles)])
        circle_canvas = metric_to_canvas(circle_pts, bounds, bev_size)
        cv2.polylines(canvas, [circle_canvas], True, (60, 80, 88), 1, cv2.LINE_AA)
        lp = metric_to_canvas(np.array([[meter, 0.0]]), bounds, bev_size)[0]
        cv2.putText(canvas, f"{meter}m", tuple(lp + [5, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 190, 200), 1, cv2.LINE_AA)

    # 格线
    for v in np.arange(math.ceil(fwd_min / 10) * 10, fwd_max + 0.1, 10):
        pts = metric_to_canvas(np.array([[v, left_min], [v, left_max]]), bounds, bev_size)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), (44, 58, 64), 1, cv2.LINE_AA)
    for v in np.arange(math.ceil(left_min / 10) * 10, left_max + 0.1, 10):
        pts = metric_to_canvas(np.array([[fwd_min, v], [fwd_max, v]]), bounds, bev_size)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), (44, 58, 64), 1, cv2.LINE_AA)

    # 坐标轴
    ax = metric_to_canvas(np.array([[fwd_min, 0], [fwd_max, 0]]), bounds, bev_size)
    ay = metric_to_canvas(np.array([[0, left_min], [0, left_max]]), bounds, bev_size)
    cv2.line(canvas, tuple(ax[0]), tuple(ax[1]), (110, 126, 138), 2, cv2.LINE_AA)
    cv2.line(canvas, tuple(ay[0]), tuple(ay[1]), (110, 126, 138), 2, cv2.LINE_AA)

    # Ego 车辆示意图
    ego_pts = np.array([[-2.2, -1.0], [2.2, -1.0], [4.2, 0.0],
                         [2.2,  1.0], [-2.2,  1.0]], dtype=np.float32)
    ego_canvas = metric_to_canvas(ego_pts, bounds, bev_size)
    overlay = canvas.copy()
    cv2.fillConvexPoly(overlay, ego_canvas, (250, 250, 250))
    canvas[:] = alpha_blend(canvas, overlay, 0.22)
    cv2.polylines(canvas, [ego_canvas], True, (255, 255, 255), 3, cv2.LINE_AA)
    center  = metric_to_canvas(np.array([[0.0, 0.0]]),  bounds, bev_size)[0]
    forward = metric_to_canvas(np.array([[8.0, 0.0]]),  bounds, bev_size)[0]
    cv2.arrowedLine(canvas, tuple(center), tuple(forward),
                    (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.25)
    return canvas


# ─── BEV 检测框渲染 ───────────────────────────────────────────────────────────

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
) -> dict:
    """将检测结果绘制到 BEV 画布上，返回图例计数字典。"""
    legend_counts = {}
    order = sorted(range(len(detections)),
                   key=lambda i: detections[i].get("score", 0))

    for idx in order:
        det    = detections[idx]
        cx     = float(det.get("center_xyz", [0, 0, 0])[0])
        cy     = float(det.get("center_xyz", [0, 0, 0])[1])
        dx     = float(det.get("size_xyz",   [1, 1, 1])[0])
        dy     = float(det.get("size_xyz",   [1, 1, 1])[1])
        yaw    = float(det.get("yaw", 0))
        score  = float(det.get("score", 0))
        label  = det.get("label_name", CLASS_NAMES[min(det.get("label", 0), 9)])
        color  = CLASS_COLORS.get(label, (180, 180, 180))
        vel    = det.get("velocity_xy", [0, 0])

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        half_offsets = [
            [ dx/2,  dy/2], [ dx/2, -dy/2],
            [-dx/2, -dy/2], [-dx/2,  dy/2],
        ]
        bottom = np.array([[cx + ox * cos_y + oy * sin_y,
                             cy + ox * -sin_y + oy * cos_y]
                            for ox, oy in half_offsets], dtype=np.float32)
        bottom_c = metric_to_canvas(bottom, bounds, bev_size)
        center_c = metric_to_canvas(np.array([[cx, cy]]), bounds, bev_size)[0]
        # 朝向箭头：车头方向
        head_x = cx + dx / 2 * cos_y
        head_y = cy + dx / 2 * -sin_y
        head_c = metric_to_canvas(np.array([[head_x, head_y]]), bounds, bev_size)[0]

        # 填充 + 边框
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, bottom_c, color)
        canvas[:] = alpha_blend(canvas, overlay, 0.22)
        for s, e in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            cv2.line(canvas, tuple(bottom_c[s]), tuple(bottom_c[e]),
                     color, 3, cv2.LINE_AA)
        cv2.arrowedLine(canvas, tuple(center_c), tuple(head_c),
                        color, 3, cv2.LINE_AA, tipLength=0.3)

        # 速度箭头
        vx, vy = float(vel[0]), float(vel[1])
        if vx**2 + vy**2 > 0.04:
            vtip_x = cx + vx * 1.5
            vtip_y = cy + vy * 1.5
            vtip_c = metric_to_canvas(np.array([[vtip_x, vtip_y]]), bounds, bev_size)[0]
            cv2.arrowedLine(canvas, tuple(center_c), tuple(vtip_c),
                            (245, 245, 245), 2, cv2.LINE_AA, tipLength=0.22)

        # 文字标签
        text   = f"{label} {score:.2f}"
        anchor = (bottom_c[0][0] + 5, bottom_c[0][1] - 5)
        draw_text_tag(canvas, text, anchor, color)
        legend_counts[label] = legend_counts.get(label, 0) + 1

    return legend_counts


def render_bev_panel(detections: list, bounds, bev_size: int) -> np.ndarray:
    """生成完整 BEV 检测结果面板（含背景、检测框、图例）。"""
    canvas = render_bev_background(bounds, bev_size)
    legend = render_bev_detections(canvas, detections, bounds, bev_size)

    # 图例
    legend_y = 58
    for label_name, count in sorted(legend.items(),
                                    key=lambda kv: (-kv[1], kv[0]))[:8]:
        color = CLASS_COLORS.get(label_name, (180, 180, 180))
        cv2.rectangle(canvas,
                      (bev_size - 210, legend_y - 14),
                      (bev_size - 195, legend_y + 2), color, -1)
        cv2.putText(canvas, f"{label_name}: {count}",
                    (bev_size - 186, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
        legend_y += 22

    return draw_title_bar(canvas, "BEV Top-down Detection")


# ─── 相机网格渲染 ─────────────────────────────────────────────────────────────

CAMERA_SHORT_NAMES = [
    "FRONT", "FRONT_RIGHT", "FRONT_LEFT",
    "BACK",  "BACK_LEFT",   "BACK_RIGHT",
]


def compose_camera_grid(frame_dir: Path, cam_width: int) -> np.ndarray:
    """将 6 路相机图像拼为 3×2 网格。"""
    cam_height = int(cam_width * 9 / 16)
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
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (12, 12, 12), -1)
        cv2.putText(img, cam_name, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        images.append(img)

    while len(images) < 6:
        images.append(np.zeros((cam_height, cam_width, 3), dtype=np.uint8))
    top    = np.concatenate(images[:3], axis=1)
    bottom = np.concatenate(images[3:6], axis=1)
    grid   = np.concatenate([top, bottom], axis=0)
    return draw_title_bar(grid, "Multi-Camera Input")


# ─── 信息栏 ───────────────────────────────────────────────────────────────────

def compose_info_bar(frame_idx: int, total_frames: int,
                     num_dets: int, width: int) -> np.ndarray:
    bar = np.zeros((52, width, 3), dtype=np.uint8)
    bar[:] = (18, 18, 18)
    text = (f"frame={frame_idx}/{total_frames}  "
            f"detections={num_dets}  "
            f"score_thr={args_global.score_thr:.2f}  "
            f"model={args_global.model}")
    cv2.putText(bar, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, (240, 240, 240), 2, cv2.LINE_AA)
    return bar


# ─── 推理调用 ─────────────────────────────────────────────────────────────────

def run_inference(binary: str, frame_dir: Path, model: str,
                  score_thr: float, classes: str) -> list:
    """
    调用 C++ 推理二进制，读取 JSON 输出。
    若 result.json 已存在则直接读取（避免重复推理）。
    """
    result_path = frame_dir / "result.json"
    if not result_path.exists():
        cmd = [
            binary,
            str(frame_dir),
            model,
            "fp16",
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
    p.add_argument("--save-frames",action="store_true",
                   help="同时保存每帧合成图像")
    p.add_argument("--auto-infer", action="store_true",
                   help="自动调用 C++ 推理二进制（./build/fastbev）")
    p.add_argument("--binary",     type=str, default="./build/fastbev",
                   help="C++ 推理二进制路径")
    p.add_argument("--num-frames", type=int, default=None,
                   help="最多处理帧数（默认全部）")
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
    frame_dirs = sorted([d for d in frames_dir.iterdir()
                         if d.is_dir() and d.name.startswith("frame_")])
    if args.num_frames:
        frame_dirs = frame_dirs[:args.num_frames]

    if not frame_dirs:
        print(f"[错误] 未找到 frame_* 子目录: {frames_dir}")
        sys.exit(1)
    print(f"共 {len(frame_dirs)} 帧，开始渲染...")

    # 视频写入器（延迟初始化）
    video_writer = None
    video_path   = out_dir / args.video_name

    for render_idx, frame_dir in enumerate(frame_dirs):
        # ── 1. 读取或执行推理 ──
        if args.auto_infer:
            detections = run_inference(
                args.binary, frame_dir, args.model,
                args.score_thr, args.classes)
        else:
            result_path = frame_dir / "result.json"
            if result_path.exists():
                with open(result_path) as f:
                    detections = json.load(f)
            else:
                print(f"  [跳过] {frame_dir.name}：缺少 result.json（使用 --auto-infer）")
                detections = []

        # 按 score_thr 过滤
        detections = [d for d in detections if d.get("score", 0) >= args.score_thr]

        # ── 2. 渲染 BEV 面板 ──
        bev_panel = render_bev_panel(detections, BEV_BOUNDS, args.bev_size)

        # ── 3. 组合相机网格 ──
        cam_grid = compose_camera_grid(frame_dir, args.cam_width)
        # 将 BEV 面板缩放到与相机网格同高
        cam_h = cam_grid.shape[0]
        bev_resized = cv2.resize(bev_panel, (int(bev_panel.shape[1] * cam_h / bev_panel.shape[0]), cam_h))
        # 横向拼接：左侧相机 | 右侧 BEV
        combined = np.concatenate([cam_grid, bev_resized], axis=1)

        # ── 4. 添加信息栏 ──
        info_bar = compose_info_bar(
            render_idx + 1, len(frame_dirs),
            len(detections), combined.shape[1])
        frame_out = np.concatenate([combined, info_bar], axis=0)

        # ── 5. 初始化视频写入器 ──
        if video_writer is None:
            h, w = frame_out.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
            print(f"视频分辨率: {w}×{h} @{args.fps}fps → {video_path}")

        video_writer.write(frame_out)

        # ── 6. 可选保存帧图像 ──
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
