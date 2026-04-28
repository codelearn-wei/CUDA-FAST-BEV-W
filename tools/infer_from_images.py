#!/usr/bin/env python3
"""
infer_from_images.py — 单样本端到端推理脚本

只需提供 6 张 NuScenes 相机图片，无需手动运行数据预处理。
BEV 检测+跟踪（C++）和 MapTR 地图推理（Python）并行运行。
最终输出：result.json / tracks.json / map_result.json 以及 tracking 可视化视频。

用法（单样本快速测试）:
  # 方式1：从已有 frame 目录直接推理（跳过数据预处理）
  python tools/infer_from_images.py --frame-dir outputs/frames/frame_00000

  # 方式2：从 NuScenes 数据集按 sample token 推理
  python tools/infer_from_images.py \\
      --nuscenes-dir data/nuscenes --version v1.0-mini \\
      --sample-token ca9a282c9e77460f8360f564131a8af5

  # 方式3：指定6张图片目录（需要 meta.json 内含标定，或通过 --nuscenes-dir 自动查找）
  python tools/infer_from_images.py \\
      --images-dir /path/to/six_images \\
      --nuscenes-dir data/nuscenes --version v1.0-mini

  # 连续帧批量并行推理（BEV串行+MapTR并行）
  python tools/infer_from_images.py \\
      --nuscenes-dir data/nuscenes --version v1.0-mini \\
      --num-frames 50 --out-dir outputs/frames \\
      --parallel-map

环境: conda activate bev
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np


# ───────────────────────────────────────────────────────────────────────────
# 几何张量计算（同 nuscenes_adapter.py，内联以减少依赖）
# ───────────────────────────────────────────────────────────────────────────

N_VOXELS   = np.array([200, 200, 4], dtype=np.float64)
VOXEL_SIZE = np.array([0.5,  0.5,  1.5], dtype=np.float64)
FEAT_HEIGHT = 64   # 256 / 4
FEAT_WIDTH  = 176  # 704 / 4
MAGIC = 0x33FF1101
DTYPE_MAP = {"float32": 3, "float16": 2, "int32": 1, "int64": 4,
             "uint64": 5, "uint32": 6, "int8": 7, "uint8": 8}


def save_tensor(arr: np.ndarray, path: str):
    dtype_str = str(arr.dtype)
    with open(path, "wb") as f:
        f.write(struct.pack("3i", MAGIC, arr.ndim, DTYPE_MAP[dtype_str]))
        f.write(struct.pack(f"{arr.ndim}i", *arr.shape))
        f.write(arr.tobytes())


def get_voxel_points(origin):
    nx, ny, nz = 200, 200, 4
    new_origin = origin - N_VOXELS / 2.0 * VOXEL_SIZE
    ix = np.arange(nx, dtype=np.float64)
    iy = np.arange(ny, dtype=np.float64)
    iz = np.arange(nz, dtype=np.float64)
    shape = (nx, ny, nz)
    pts_x = np.broadcast_to((ix[None, :, None, None] * VOXEL_SIZE[0] + new_origin[0]).reshape(1, nx, 1, 1), (1,) + shape)
    pts_y = np.broadcast_to((iy[None, None, :, None] * VOXEL_SIZE[1] + new_origin[1]).reshape(1, 1, ny, 1), (1,) + shape)
    pts_z = np.broadcast_to((iz[None, None, None, :] * VOXEL_SIZE[2] + new_origin[2]).reshape(1, 1, 1, nz), (1,) + shape)
    return np.concatenate([pts_x, pts_y, pts_z], axis=0).copy()


def compute_geometry_tensors_per_cam(extrinsics, intrinsics, origin, feat_stride=4):
    num_cameras = len(extrinsics)
    num_voxels  = 200 * 200 * 4
    pts_3d = get_voxel_points(origin).reshape(3, -1)
    pts_h  = np.vstack([pts_3d, np.ones((1, num_voxels), dtype=np.float64)])

    valid_c_idx = np.zeros((num_cameras, num_voxels), dtype=np.float32)
    valid_x     = np.zeros((num_cameras, num_voxels), dtype=np.int64)
    valid_y     = np.zeros((num_cameras, num_voxels), dtype=np.int64)

    for cam_idx, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
        if intr is None:
            continue
        E = np.array(ext, dtype=np.float64)
        K_feat = intr[:3, :3].copy().astype(np.float64)
        K_feat[0] /= feat_stride
        K_feat[1] /= feat_stride
        P       = K_feat @ E[:3, :]
        pts_2d  = P @ pts_h
        z_val   = pts_2d[2]
        safe_z  = np.where(np.abs(z_val) > 1e-6, z_val, 1.0)
        x_img   = np.round(pts_2d[0] / safe_z).astype(np.int64)
        y_img   = np.round(pts_2d[1] / safe_z).astype(np.int64)
        valid_mask = ((z_val > 0) &
                      (x_img >= 0) & (x_img < FEAT_WIDTH) &
                      (y_img >= 0) & (y_img < FEAT_HEIGHT))
        valid_c_idx[cam_idx]         = valid_mask.astype(np.float32)
        valid_x[cam_idx, valid_mask] = x_img[valid_mask]
        valid_y[cam_idx, valid_mask] = y_img[valid_mask]
    return valid_c_idx, valid_x, valid_y


def _quat_to_rotation_matrix(q_wxyz):
    try:
        from scipy.spatial.transform import Rotation
        w, x, y, z = q_wxyz
        return Rotation.from_quat([x, y, z, w]).as_matrix()
    except ImportError:
        w, x, y, z = q_wxyz
        return np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
            [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
            [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
        ], dtype=np.float64)


# ───────────────────────────────────────────────────────────────────────────
# NuScenes 单帧数据准备
# ───────────────────────────────────────────────────────────────────────────

CAMERA_ORDER = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                "CAM_BACK",  "CAM_BACK_LEFT",   "CAM_BACK_RIGHT"]
SHORT_NAMES  = {"CAM_FRONT": "FRONT", "CAM_FRONT_RIGHT": "FRONT_RIGHT",
                "CAM_FRONT_LEFT": "FRONT_LEFT", "CAM_BACK": "BACK",
                "CAM_BACK_LEFT": "BACK_LEFT", "CAM_BACK_RIGHT": "BACK_RIGHT"}
OUTPUT_W, OUTPUT_H = 704, 256
ORIG_W,   ORIG_H   = 1600, 900


def prepare_nuscenes_frames(nuscenes_dir, version, out_dir,
                             sample_token=None, num_frames=1, start_idx=0, stride=1):
    """
    从原始 NuScenes 数据准备一个或多个帧目录。
    若 sample_token 不为 None，则只准备该单帧（frame_00000）。
    """
    nuscenes_dir = Path(nuscenes_dir)
    json_dir = nuscenes_dir / version
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(name):
        return json.load(open(str(json_dir / name)))

    sensor_map = {s["token"]: s for s in _load("sensor.json")}
    cal_map    = {c["token"]: c for c in _load("calibrated_sensor.json")}
    ep_map     = {e["token"]: e for e in _load("ego_pose.json")}
    sample_map = {s["token"]: s for s in _load("sample.json")}
    scene_list = _load("scene.json")
    sd_list    = _load("sample_data.json")

    # sample_data 分组
    sd_by_sample_channel = {}
    for sd in sd_list:
        if not sd["is_key_frame"]:
            continue
        cal_entry    = cal_map[sd["calibrated_sensor_token"]]
        sensor_entry = sensor_map[cal_entry["sensor_token"]]
        stok = sd["sample_token"]
        chan = sensor_entry["channel"]
        sd_by_sample_channel.setdefault(stok, {})[chan] = (sd, cal_entry, sensor_entry)

    # lidar 标定
    lidar_cal_by_sample = {}
    for sd in sd_list:
        ce = cal_map[sd["calibrated_sensor_token"]]
        se = sensor_map[ce["sensor_token"]]
        if se["channel"] == "LIDAR_TOP" and sd["is_key_frame"]:
            lidar_cal_by_sample[sd["sample_token"]] = ce

    # 收集有序样本
    if sample_token:
        ordered_tokens = [sample_token]
    else:
        ordered_tokens = []
        for scene in scene_list:
            tok = scene["first_sample_token"]
            while tok:
                s = sample_map.get(tok)
                if s is None:
                    break
                ordered_tokens.append(s["token"])
                tok = s["next"]
        indices = list(range(start_idx,
                             min(len(ordered_tokens), start_idx + num_frames * stride),
                             stride))[:num_frames]
        ordered_tokens = [ordered_tokens[i] for i in indices]

    print(f"  准备 {len(ordered_tokens)} 帧...")

    prepared = []
    for render_idx, stok in enumerate(ordered_tokens):
        sample = sample_map.get(stok, {"token": stok, "timestamp": 0, "next": ""})
        frame_dir = out_dir / f"frame_{render_idx:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        lidar_cal = lidar_cal_by_sample.get(stok)
        R_l = _quat_to_rotation_matrix(lidar_cal["rotation"]) if lidar_cal else np.eye(3)
        t_l = np.array(lidar_cal["translation"], dtype=np.float64) if lidar_cal else np.zeros(3)
        lidar_yaw_in_ego = math.atan2(R_l[1, 0], R_l[0, 0])

        extrinsics     = []
        intrinsics_all = []
        intrinsics_raw = []
        cam_channels   = sd_by_sample_channel.get(stok, {})

        for cam_idx, cam_name in enumerate(CAMERA_ORDER):
            short = SHORT_NAMES[cam_name]
            if cam_name not in cam_channels:
                extrinsics.append(np.eye(4)); intrinsics_all.append(None); intrinsics_raw.append(None)
                continue
            sd_entry, cam_cal, _ = cam_channels[cam_name]
            img_path = nuscenes_dir / sd_entry["filename"]
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((ORIG_H, ORIG_W, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_dir / f"{cam_idx}-{short}.jpg"), img[:, :, ::-1])  # BGR→RGB

            K_raw = np.array(cam_cal["camera_intrinsic"], dtype=np.float64)
            resize_lim = float(OUTPUT_W) / ORIG_W
            resized_h  = int(ORIG_H * resize_lim)
            crop_y_off = (resized_h - OUTPUT_H) // 2
            K = K_raw.copy()
            K[0] *= resize_lim
            K[1] *= resize_lim
            K[1, 2] -= crop_y_off
            intrinsics_all.append(K)
            intrinsics_raw.append(K_raw)

            R_c = _quat_to_rotation_matrix(cam_cal["rotation"])
            t_c = np.array(cam_cal["translation"], dtype=np.float64)
            E   = np.eye(4, dtype=np.float64)
            E[:3, :3] = R_c.T @ R_l
            E[:3,  3] = R_c.T @ (t_l - t_c)
            extrinsics.append(E)

        # 几何张量
        origin = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        valid_c_idx, valid_x, valid_y = compute_geometry_tensors_per_cam(
            extrinsics, intrinsics_all, origin)
        save_tensor(valid_c_idx, str(frame_dir / "valid_c_idx.tensor"))
        save_tensor(valid_x,     str(frame_dir / "x.tensor"))
        save_tensor(valid_y,     str(frame_dir / "y.tensor"))

        # ego pose
        ego_translation_global = [0.0, 0.0, 0.0]
        ego_yaw_global = 0.0
        if "LIDAR_TOP" in cam_channels:
            lidar_sd_entry, _, _ = cam_channels["LIDAR_TOP"]
            ep = ep_map.get(lidar_sd_entry.get("ego_pose_token", ""), {})
            if ep:
                ego_translation_global = list(ep["translation"])
                qw, qx, qy, qz = ep["rotation"]
                ego_yaw_global = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz))

        meta = {
            "frame_idx":              render_idx,
            "sample_token":           stok,
            "timestamp":              sample.get("timestamp", 0),
            "origin":                 origin.tolist(),
            "camera_names":           [SHORT_NAMES[c] for c in CAMERA_ORDER],
            "ego_translation_global": ego_translation_global,
            "ego_yaw_global":         ego_yaw_global,
            "lidar_yaw_in_ego":       lidar_yaw_in_ego,
            "lidar2cam_extrinsics":   [e.tolist() for e in extrinsics],
            "cam_intrinsics_raw":     [k.tolist() if k is not None else None for k in intrinsics_raw],
        }
        with open(frame_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        prepared.append({"frame_idx": render_idx, "frame_dir": str(frame_dir),
                          "sample_token": stok})
        if render_idx % 10 == 0 or render_idx == 0:
            print(f"    [{render_idx+1}/{len(ordered_tokens)}] {frame_dir.name}")

    return prepared


def prepare_from_images_dir(images_dir, out_dir, meta_json_path=None, frame_idx=0):
    """
    从包含6张图片的目录准备帧数据。
    若提供 meta_json_path，使用其中的标定参数；否则使用 images_dir/meta.json。
    图片命名格式: 0-FRONT.jpg, 1-FRONT_RIGHT.jpg, ..., 5-BACK_RIGHT.jpg
    """
    images_dir = Path(images_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir  = out_dir / f"frame_{frame_idx:05d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # 复制图片
    cam_files = ["0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg",
                 "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"]
    for f in cam_files:
        src = images_dir / f
        dst = frame_dir / f
        if src.exists():
            shutil.copy(src, dst)
        else:
            print(f"  [警告] 缺少图片: {src}")

    # 读取或复制 meta.json
    meta_src = Path(meta_json_path) if meta_json_path else images_dir / "meta.json"
    if meta_src.exists():
        with open(meta_src) as f:
            meta = json.load(f)
        # 如果 meta.json 中有 extrinsics，用于计算张量
        extrinsics_raw = meta.get("lidar2cam_extrinsics")
        intrinsics_raw = meta.get("cam_intrinsics_raw")
        if extrinsics_raw and intrinsics_raw:
            resize_lim = float(OUTPUT_W) / ORIG_W
            resized_h  = int(ORIG_H * resize_lim)
            crop_y_off = (resized_h - OUTPUT_H) // 2
            extrinsics = [np.array(e, dtype=np.float64) for e in extrinsics_raw]
            intrinsics = []
            for K_raw in intrinsics_raw:
                if K_raw is None:
                    intrinsics.append(None)
                    continue
                K = np.array(K_raw, dtype=np.float64)
                K[0] *= resize_lim
                K[1] *= resize_lim
                K[1, 2] -= crop_y_off
                intrinsics.append(K)
            origin = np.array(meta.get("origin", [0.0, 0.0, -1.0]), dtype=np.float64)
            valid_c_idx, valid_x, valid_y = compute_geometry_tensors_per_cam(
                extrinsics, intrinsics, origin)
            save_tensor(valid_c_idx, str(frame_dir / "valid_c_idx.tensor"))
            save_tensor(valid_x,     str(frame_dir / "x.tensor"))
            save_tensor(valid_y,     str(frame_dir / "y.tensor"))
            meta["frame_idx"] = frame_idx
            with open(frame_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            return str(frame_dir)

    # fallback: 检查目标目录中是否已有张量
    if all((frame_dir / t).exists() for t in
           ["valid_c_idx.tensor", "x.tensor", "y.tensor"]):
        print("  找到已有几何张量，跳过重新计算")
    else:
        raise RuntimeError(
            f"无法找到标定信息。请提供包含 lidar2cam_extrinsics 和 cam_intrinsics_raw 的 meta.json，"
            f"或使用 --nuscenes-dir 模式。")
    return str(frame_dir)


# ───────────────────────────────────────────────────────────────────────────
# 推理执行
# ───────────────────────────────────────────────────────────────────────────

def run_bev_tracking(project_root, frames_dir, model, score_thr, output_format="json"):
    """运行 C++ tracking_demo（BEV 检测 + 多目标跟踪）。"""
    binary = os.path.join(project_root, "build", "tracking_demo")
    if not os.path.exists(binary):
        raise FileNotFoundError(
            f"未找到 build/tracking_demo，请先编译:\n"
            f"  cd {project_root}/build && cmake .. && make -j$(nproc)")
    cmd = [binary, frames_dir, model, "int8",
           "--score-thr", str(score_thr),
           "--output-format", output_format,
           "--batch"]
    print(f"\n▶ [BEV+Tracking] {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=project_root)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  ✘ BEV 推理失败 (exit={result.returncode})")
        return False, dt

    # 统计帧率
    n_frames = sum(1 for d in Path(frames_dir).iterdir()
                   if d.is_dir() and d.name.startswith("frame_"))
    fps = n_frames / dt if dt > 0 else 0
    print(f"  ✔ BEV+Tracking 完成 ({dt:.1f}s, {fps:.1f} FPS, {n_frames} 帧)")
    return True, dt


def run_maptr(project_root, frames_dir, score_thr, overwrite=False):
    """运行 MapTR 地图推理（Python，conda bev 环境）。"""
    script = os.path.join(project_root, "tools", "maptr", "run_maptr.py")
    if not os.path.exists(script):
        print(f"  [跳过] 未找到 {script}")
        return False, 0.0
    cmd = [sys.executable, script,
           "--frames-dir", frames_dir,
           "--mode", "model",
           "--score-thr", str(score_thr)]
    if overwrite:
        cmd.append("--overwrite")
    print(f"\n▶ [MapTR] {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=project_root)
    dt = time.time() - t0
    ok = result.returncode == 0
    print(f"  {'✔' if ok else '✘'} MapTR 完成 ({dt:.1f}s)")
    return ok, dt


def run_parallel_infer(project_root, frames_dir, model, score_thr, overwrite=False):
    """BEV+Tracking 和 MapTR 并行执行，返回 (bev_ok, map_ok, bev_dt, map_dt)。"""
    results = {}

    def _bev():
        ok, dt = run_bev_tracking(project_root, frames_dir, model, score_thr)
        results["bev"] = (ok, dt)

    def _maptr():
        ok, dt = run_maptr(project_root, frames_dir, score_thr, overwrite)
        results["maptr"] = (ok, dt)

    t_bev   = threading.Thread(target=_bev,   name="bev")
    t_maptr = threading.Thread(target=_maptr, name="maptr")
    t_bev.start()
    t_maptr.start()
    t_bev.join()
    t_maptr.join()

    bev_ok,   bev_dt   = results.get("bev",   (False, 0.0))
    maptr_ok, maptr_dt = results.get("maptr", (False, 0.0))
    return bev_ok, maptr_ok, bev_dt, maptr_dt


# ───────────────────────────────────────────────────────────────────────────
# Tracking 可视化（调用 video_visualize_tracking.py 核心函数）
# ───────────────────────────────────────────────────────────────────────────

def visualize_tracking(project_root, frames_dir, out_dir, fps=6):
    """
    调用 tools/video_visualize_tracking.py 生成 tracking 可视化视频。
    输出: out_dir/tracking_result.mp4
    """
    script = os.path.join(project_root, "tools", "video_visualize_tracking.py")
    out_video = os.path.join(out_dir, "tracking_result.mp4")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, script,
           "--frames-dir", frames_dir,
           "--out-video",  out_video,
           "--fps",        str(fps)]
    print(f"\n▶ [Tracking 可视化] {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=project_root)
    dt = time.time() - t0
    ok = result.returncode == 0
    if ok:
        print(f"  ✔ 可视化完成 ({dt:.1f}s): {out_video}")
    else:
        # fallback: 直接调用内部函数
        print(f"  [fallback] 直接调用 visualize_tracking_from_json...")
        try:
            sys.path.insert(0, os.path.join(project_root, "tools"))
            from video_visualize_tracking import visualize_tracking_from_json
            visualize_tracking_from_json(
                frames_dir, out_video, fps=fps,
                dpi=150, figsize=(12, 12),
                swap_y_sign=False, swap_axes=True, bitrate="10M")
            ok = True
        except Exception as e:
            print(f"  ✘ 可视化失败: {e}")
    return ok, out_video


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="从6张图片（或NuScenes数据）端到端推理：BEV检测+跟踪 || MapTR地图（并行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 数据源（三选一）─────────────────────────────────────────────────────
    src = p.add_argument_group("数据源（三选一）")
    src.add_argument("--frame-dir",    type=str, default=None,
                     help="直接使用已有帧目录（跳过数据预处理）")
    src.add_argument("--images-dir",   type=str, default=None,
                     help="包含6张图片（0-FRONT.jpg…5-BACK_RIGHT.jpg）的目录")
    src.add_argument("--nuscenes-dir", type=str, default=None,
                     help="NuScenes 根目录（含 samples/ 和版本 JSON）")

    # ── NuScenes 参数 ────────────────────────────────────────────────────────
    nu = p.add_argument_group("NuScenes 参数（--nuscenes-dir 时生效）")
    nu.add_argument("--version",      type=str, default="v1.0-mini")
    nu.add_argument("--sample-token", type=str, default=None,
                    help="指定单个 sample token（不指定则顺序处理 --num-frames 帧）")
    nu.add_argument("--num-frames",   type=int, default=50,
                    help="批量处理帧数（默认 50）")
    nu.add_argument("--start-idx",    type=int, default=0)
    nu.add_argument("--stride",       type=int, default=1)

    # ── 推理参数 ─────────────────────────────────────────────────────────────
    inf = p.add_argument_group("推理参数")
    inf.add_argument("--model",      type=str,   default="resnet18int8",
                     help="BEV 模型名称（默认 resnet18int8）")
    inf.add_argument("--score-thr",  type=float, default=0.3)
    inf.add_argument("--fps",        type=int,   default=6)
    inf.add_argument("--out-dir",    type=str,   default="outputs/frames",
                     help="帧数据输出目录")
    inf.add_argument("--video-dir",  type=str,   default="outputs/video",
                     help="视频输出目录")
    inf.add_argument("--overwrite",  action="store_true",
                     help="覆盖已有结果")

    # ── 执行控制 ─────────────────────────────────────────────────────────────
    ctrl = p.add_argument_group("执行控制")
    ctrl.add_argument("--parallel-map",    action="store_true",
                      help="BEV 和 MapTR 并行运行（需要足够 GPU 显存）")
    ctrl.add_argument("--skip-data-prep",  action="store_true",
                      help="跳过数据预处理（--out-dir 已有帧数据）")
    ctrl.add_argument("--skip-bev",        action="store_true",
                      help="跳过 BEV 推理（已有 result.json / tracks.json）")
    ctrl.add_argument("--skip-map",        action="store_true",
                      help="跳过 MapTR 推理")
    ctrl.add_argument("--skip-video",      action="store_true",
                      help="跳过视频生成")
    ctrl.add_argument("--only-tracking-vis", action="store_true",
                      help="只生成 tracking 可视化视频（跳过所有推理）")

    return p.parse_args()


def main():
    args   = parse_args()
    script = Path(__file__).resolve()
    root   = script.parent.parent  # 项目根目录
    os.chdir(root)

    frames_dir = args.out_dir

    print(f"\n{'='*60}")
    print(f"  CUDA-FastBEV 端到端推理 (BEV || MapTR)")
    print(f"  模型: {args.model}  阈值: {args.score_thr}  fps: {args.fps}")
    print(f"{'='*60}\n")

    total_t0 = time.time()

    # ── 只生成可视化 ─────────────────────────────────────────────────────────
    if args.only_tracking_vis:
        fd = args.frame_dir or frames_dir
        visualize_tracking(str(root), fd, args.video_dir, args.fps)
        return

    # ── Step 1: 数据预处理 ────────────────────────────────────────────────────
    if not args.skip_data_prep:
        if args.frame_dir:
            # 直接用已有帧目录，无需预处理
            frames_dir = args.frame_dir
            print(f"[Step1] 使用已有帧目录: {frames_dir}")
        elif args.images_dir:
            print(f"[Step1] 从图片目录准备帧数据: {args.images_dir}")
            prepare_from_images_dir(args.images_dir, frames_dir, frame_idx=0)
            print(f"  ✔ 帧数据 → {frames_dir}/frame_00000")
        elif args.nuscenes_dir:
            print(f"[Step1] 从 NuScenes 准备帧数据 ({args.version})")
            prepare_nuscenes_frames(
                nuscenes_dir  = args.nuscenes_dir,
                version       = args.version,
                out_dir       = frames_dir,
                sample_token  = args.sample_token,
                num_frames    = args.num_frames,
                start_idx     = args.start_idx,
                stride        = args.stride,
            )
        else:
            print("[Step1] 跳过（未指定数据源，假设帧数据已在 --out-dir）")
    else:
        if args.frame_dir:
            frames_dir = args.frame_dir
        print(f"[Step1] 已跳过数据预处理，使用: {frames_dir}")

    # ── Step 2 & 3: BEV+Tracking 和 MapTR ────────────────────────────────────
    timings = {}

    if args.skip_bev and args.skip_map:
        print("[Step2/3] 跳过所有推理")
    elif args.parallel_map and not args.skip_bev and not args.skip_map:
        print("\n[Step2+3] BEV+Tracking || MapTR 并行运行...")
        bev_ok, map_ok, bev_dt, map_dt = run_parallel_infer(
            str(root), frames_dir, args.model, args.score_thr, args.overwrite)
        timings["bev+tracking (并行)"] = bev_dt
        timings["maptr (并行)"]         = map_dt
        if not bev_ok:
            print("  BEV 推理失败")
            sys.exit(1)
    else:
        if not args.skip_bev:
            print("\n[Step2] BEV 检测 + 多目标跟踪...")
            ok, dt = run_bev_tracking(str(root), frames_dir, args.model, args.score_thr)
            timings["bev+tracking"] = dt
            if not ok:
                print("  BEV 推理失败，中止")
                sys.exit(1)
        if not args.skip_map:
            print("\n[Step3] MapTR 地图推理...")
            ok, dt = run_maptr(str(root), frames_dir, args.score_thr, args.overwrite)
            timings["maptr"] = dt

    # ── Step 4: Tracking 可视化 ───────────────────────────────────────────────
    if not args.skip_video:
        print("\n[Step4] Tracking 可视化...")
        ok, out_video = visualize_tracking(str(root), frames_dir, args.video_dir, args.fps)
        timings["tracking_vis"] = 0.0  # 内部已计时

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    total_dt = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  完成  (总计 {total_dt:.1f}s)")
    for k, v in timings.items():
        print(f"    {k:<30} {v:.1f}s")
    print(f"{'='*60}")
    if not args.skip_video:
        print(f"  Tracking 视频: {args.video_dir}/tracking_result.mp4")
    print()


if __name__ == "__main__":
    main()
