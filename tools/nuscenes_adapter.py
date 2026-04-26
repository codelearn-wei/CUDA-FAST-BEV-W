#!/usr/bin/env python3
"""
NuScenes 数据适配器
将 NuScenes 样本数据转换为 CUDA-FastBEV C++ 推理所需的格式。

用法:
  # 直接读取原始 NuScenes 数据集（推荐，无需 mmdet3d）
  python tools/nuscenes_adapter.py \
      --nuscenes-dir /home/dfg-autoware/BEV_projects/CUDA-FastBEV/data/nuscenes \
      --version v1.0-mini \
      --out-dir outputs/frames \
      --num-frames 50

  # 从 NuScenes mmdet3d pkl 文件准备帧数据
  python tools/nuscenes_adapter.py \
      --pkl-path /data/nuscenes/nuscenes_infos_val.pkl \
      --nuscenes-root /data/nuscenes \
      --out-dir outputs/frames \
      --num-frames 50

  # 使用已有的 example-data.pth（兼容现有格式）
  python tools/nuscenes_adapter.py \
      --pth-path example-data/example-data/example-data.pth \
      --out-dir outputs/frames
"""

import argparse
import json
import math
import os
import struct
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np


# ─── Tensor 序列化（与 C++ tensor.cu 格式兼容）─────────────────────────────

DTYPE_MAP = {
    "float32": 3,
    "float16": 2,
    "int32":   1,
    "int64":   4,
    "uint64":  5,
    "uint32":  6,
    "int8":    7,
    "uint8":   8,
}
MAGIC = 0x33FF1101


def save_tensor(arr: np.ndarray, path: str):
    """保存 numpy 数组为 .tensor 格式（与 nv::Tensor::load 兼容）。"""
    dtype_str = str(arr.dtype)
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"不支持的 dtype: {dtype_str}")
    with open(path, "wb") as f:
        header = struct.pack("3i", MAGIC, arr.ndim, DTYPE_MAP[dtype_str])
        f.write(header)
        dims = struct.pack(f"{arr.ndim}i", *arr.shape)
        f.write(dims)
        f.write(arr.tobytes())


# ─── 几何张量计算 ──────────────────────────────────────────────────────────

# 体素网格参数（与模型配置一致）
N_VOXELS   = np.array([200, 200, 4], dtype=np.float64)
VOXEL_SIZE = np.array([0.5,  0.5,  1.5], dtype=np.float64)

# 特征图尺寸（图像分辨率 256×704 下采样 4×）
FEAT_HEIGHT = 64   # 256 / 4
FEAT_WIDTH  = 176  # 704 / 4


def get_voxel_points(origin: np.ndarray) -> np.ndarray:
    """
    生成体素网格的 3D 坐标。
    返回: shape [3, Nx, Ny, Nz]（x/y/z 方向各 200/200/4 个）
    """
    nx, ny, nz = int(N_VOXELS[0]), int(N_VOXELS[1]), int(N_VOXELS[2])
    new_origin = origin - N_VOXELS / 2.0 * VOXEL_SIZE  # 体素左下角

    ix = np.arange(nx, dtype=np.float64)
    iy = np.arange(ny, dtype=np.float64)
    iz = np.arange(nz, dtype=np.float64)
    grid_z, grid_y, grid_x = np.meshgrid(iz, iy, ix, indexing='ij')  # broadcast

    # 实际 3D 坐标 [3, Nz, Ny, Nx] → 转置为 [3, Nx, Ny, Nz]
    pts_x = ix[None, :, None, None] * VOXEL_SIZE[0] + new_origin[0]  # [1, Nx, 1, 1]
    pts_y = iy[None, None, :, None] * VOXEL_SIZE[1] + new_origin[1]
    pts_z = iz[None, None, None, :] * VOXEL_SIZE[2] + new_origin[2]

    # Broadcast 并堆叠为 [3, Nx, Ny, Nz]
    shape = (nx, ny, nz)
    pts_x = np.broadcast_to(pts_x.reshape(1, nx, 1, 1), (1,) + shape)
    pts_y = np.broadcast_to(pts_y.reshape(1, 1, ny, 1), (1,) + shape)
    pts_z = np.broadcast_to(pts_z.reshape(1, 1, 1, nz), (1,) + shape)
    return np.concatenate([pts_x, pts_y, pts_z], axis=0).copy()  # [3, 200, 200, 4]


def compute_geometry_tensors(
    extrinsics: list,        # list of 6 extrinsic matrices [4×4]
    intrinsic: np.ndarray,   # [3×3] intrinsic（图像尺度）
    origin: np.ndarray,      # 体素网格原点（lidar 坐标系）
    feat_stride: int = 4,
) -> tuple:
    """
    计算 valid_c_idx / valid_x / valid_y 张量，用于 BEV Pooling。

    返回:
        valid_c_idx: [6, 160000] float32，有效体素在此相机上投影标志
        valid_x:     [6, 160000] int64，特征图 x 坐标
        valid_y:     [6, 160000] int64，特征图 y 坐标
    """
    num_cameras = len(extrinsics)
    nx, ny, nz = int(N_VOXELS[0]), int(N_VOXELS[1]), int(N_VOXELS[2])
    num_voxels = nx * ny * nz  # 160000

    # 特征图尺度的内参
    K_feat = intrinsic[:3, :3].copy().astype(np.float64)
    K_feat[0] /= feat_stride  # fx, cx
    K_feat[1] /= feat_stride  # fy, cy

    # 体素点坐标 [3, 200×200×4] → [3, 160000]
    pts_3d = get_voxel_points(origin).reshape(3, -1)  # [3, 160000]
    pts_h  = np.vstack([pts_3d, np.ones((1, num_voxels), dtype=np.float64)])  # [4, 160000]

    valid_c_idx = np.zeros((num_cameras, num_voxels), dtype=np.float32)
    valid_x     = np.zeros((num_cameras, num_voxels), dtype=np.int64)
    valid_y     = np.zeros((num_cameras, num_voxels), dtype=np.int64)

    for cam_idx, ext in enumerate(extrinsics):
        E = np.array(ext, dtype=np.float64)  # [4×4]
        # 构造投影矩阵 P = K_feat @ E[:3, :]   [3×4]
        P = K_feat @ E[:3, :]  # [3, 4]

        pts_2d = P @ pts_h   # [3, 160000]
        z_val  = pts_2d[2]   # [160000]

        # 安全归一化（避免除零）
        safe_z   = np.where(np.abs(z_val) > 1e-6, z_val, 1.0)
        x_img    = np.round(pts_2d[0] / safe_z).astype(np.int64)
        y_img    = np.round(pts_2d[1] / safe_z).astype(np.int64)

        valid_mask = (
            (z_val > 0) &
            (x_img >= 0) & (x_img < FEAT_WIDTH) &
            (y_img >= 0) & (y_img < FEAT_HEIGHT)
        )

        valid_c_idx[cam_idx]        = valid_mask.astype(np.float32)
        valid_x[cam_idx, valid_mask] = x_img[valid_mask]
        valid_y[cam_idx, valid_mask] = y_img[valid_mask]

    return valid_c_idx, valid_x, valid_y


# ─── 从 example-data.pth 准备帧数据 ─────────────────────────────────────────

def prepare_frame_from_pth(pth_path: str, out_dir: Path, frame_idx: int = 0,
                           image_src_dir: str = None):
    """
    从现有的 example-data.pth 准备帧目录（兼容旧格式）。
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("请安装 torch: conda activate bev")

    data = torch.load(pth_path, map_location="cpu")
    img_meta = data["img_metas"].data[0][0]
    lidar2img = img_meta["lidar2img"]

    origin     = np.array(lidar2img["origin"], dtype=np.float64)
    intrinsic  = np.array(lidar2img["intrinsic"], dtype=np.float64)
    extrinsics = [np.array(e, dtype=np.float64) for e in lidar2img["extrinsic"]]
    lidar_yaw_in_ego = 0.0

    frame_dir = out_dir / f"frame_{frame_idx:05d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # 计算几何张量
    valid_c_idx, valid_x, valid_y = compute_geometry_tensors(
        extrinsics, intrinsic, origin)
    save_tensor(valid_c_idx,  str(frame_dir / "valid_c_idx.tensor"))
    save_tensor(valid_x,      str(frame_dir / "x.tensor"))
    save_tensor(valid_y,      str(frame_dir / "y.tensor"))

    # 复制图像
    camera_names = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
    src_root = Path(pth_path).parent if image_src_dir is None else Path(image_src_dir)
    for cam_idx, cam_name in enumerate(camera_names):
        src = src_root / f"{cam_idx}-{cam_name}.jpg"
        dst = frame_dir / f"{cam_idx}-{cam_name}.jpg"
        if src.exists():
            shutil.copy(src, dst)
        else:
            # 尝试从 img_info 获取路径
            try:
                img_info = img_meta["img_info"][cam_idx]
                orig_path = img_info.get("filename", "")
                if orig_path and Path(orig_path).exists():
                    img = cv2.imread(orig_path)
                    cv2.imwrite(str(dst), img)
                else:
                    print(f"  [警告] 找不到图像: {src}")
            except Exception:
                print(f"  [警告] 找不到图像: {src}")

    # 保存帧元信息
    meta = {
        "frame_idx": frame_idx,
        "origin": origin.tolist(),
        "camera_names": camera_names,
        "lidar_yaw_in_ego": lidar_yaw_in_ego,
    }
    with open(frame_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  帧 {frame_idx:05d} → {frame_dir}")
    return frame_dir


# ─── 从 NuScenes pkl 文件准备帧序列 ─────────────────────────────────────────

def prepare_frames_from_nuscenes_pkl(
    pkl_path: str,
    nuscenes_root: str,
    out_dir: Path,
    start_idx: int = 0,
    num_frames: int = 50,
    stride: int = 1,
    image_width: int = 1600,
    image_height: int = 900,
    output_width: int = 704,
    output_height: int = 256,
):
    """
    从 NuScenes 格式的 pkl 文件批量准备帧目录。

    pkl 文件通常由 BEV_Fast 的 create_data_bevdet.py 或 mmdet3d 生成，
    每个样本包含 img_metas 信息（lidar2img extrinsics/intrinsics/origin）。
    """
    try:
        import pickle
    except ImportError:
        raise RuntimeError("pickle 不可用")

    with open(pkl_path, "rb") as f:
        data_infos = pickle.load(f)

    if isinstance(data_infos, dict):
        # mmdet3d 格式: {'infos': [...], 'metadata': {...}}
        infos = data_infos.get("infos", data_infos.get("data_list", []))
    else:
        infos = data_infos

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = list(range(start_idx,
                               min(len(infos), start_idx + num_frames * stride),
                               stride))
    print(f"共准备 {len(frame_indices)} 帧 (总样本数: {len(infos)})")

    prepared = []
    for render_idx, data_idx in enumerate(frame_indices):
        info = infos[data_idx]
        frame_dir = _prepare_single_nuscenes_info(
            info, nuscenes_root, out_dir, render_idx,
            image_width, image_height, output_width, output_height
        )
        prepared.append({
            "frame_idx": render_idx,
            "data_idx":  data_idx,
            "frame_dir": str(frame_dir),
            "sample_token": info.get("token", ""),
            "timestamp": info.get("timestamp", 0),
        })

    # 保存序列清单
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(prepared, f, indent=2)
    print(f"序列清单 → {manifest_path}")
    return prepared


def _prepare_single_nuscenes_info(
    info: dict,
    nuscenes_root: str,
    out_dir: Path,
    frame_idx: int,
    image_width: int = 1600,
    image_height: int = 900,
    output_width: int = 704,
    output_height: int = 256,
) -> Path:
    """
    处理单个 NuScenes info 字典，生成帧目录。
    info 格式兼容 mmdet3d NuScenes pkl（包含 cams 字段）。
    """
    frame_dir = out_dir / f"frame_{frame_idx:05d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    camera_order = [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK",  "CAM_BACK_LEFT",   "CAM_BACK_RIGHT",
    ]
    name_to_short = {
        "CAM_FRONT":       "FRONT",
        "CAM_FRONT_RIGHT": "FRONT_RIGHT",
        "CAM_FRONT_LEFT":  "FRONT_LEFT",
        "CAM_BACK":        "BACK",
        "CAM_BACK_LEFT":   "BACK_LEFT",
        "CAM_BACK_RIGHT":  "BACK_RIGHT",
    }

    cams = info.get("cams", {})
    if not cams:
        raise ValueError(f"info 中缺少 'cams' 字段，请确认 pkl 文件格式")

    # ── 加载图像 ──────────────────────────────────────────────────────────
    for cam_idx, cam_name in enumerate(camera_order):
        if cam_name not in cams:
            print(f"  [警告] 缺少相机: {cam_name}")
            continue
        cam_info = cams[cam_name]
        img_path = cam_info.get("data_path", "")
        if not os.path.isabs(img_path):
            img_path = os.path.join(nuscenes_root, img_path)

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [警告] 无法读取图像: {img_path}")
            img = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (output_width, output_height))

        short_name = name_to_short[cam_name]
        out_img_path = frame_dir / f"{cam_idx}-{short_name}.jpg"
        cv2.imwrite(str(out_img_path), img)

    # ── 计算几何张量 ──────────────────────────────────────────────────────
    extrinsics = []
    intrinsic  = None
    origin_z   = None  # z 原点由 point_cloud_range 决定

    for cam_name in camera_order:
        if cam_name not in cams:
            extrinsics.append(np.eye(4))
            continue
        cam_info = cams[cam_name]

        # 内参（原始图像尺度）
        if intrinsic is None:
            cam_intrinsic = np.array(cam_info["cam_intrinsic"], dtype=np.float64)
            # 扩展为 3×3 或 4×4
            if cam_intrinsic.shape == (3, 3):
                K = cam_intrinsic
            else:
                K = cam_intrinsic[:3, :3]
            # 如果图像经过缩放，需要调整内参
            scale_x = output_width  / image_width
            scale_y = output_height / image_height
            K[0] *= scale_x
            K[1] *= scale_y
            intrinsic = K

        # 外参：lidar → camera（标准 NuScenes 格式）
        # sensor2lidar_rotation [3×3] + sensor2lidar_translation [3]
        # 需要求逆得到 lidar2sensor
        R_s2l = np.array(cam_info["sensor2lidar_rotation"], dtype=np.float64)
        t_s2l = np.array(cam_info["sensor2lidar_translation"], dtype=np.float64)

        # lidar2sensor: E = [R^T | -R^T * t]
        R_l2s = R_s2l.T
        t_l2s = -R_l2s @ t_s2l

        E = np.eye(4, dtype=np.float64)
        E[:3, :3] = R_l2s
        E[:3,  3] = t_l2s
        extrinsics.append(E)

    if intrinsic is None:
        raise ValueError("无法获取内参矩阵")

    # 体素网格原点（lidar 坐标系，z 方向取范围中心）
    # point_cloud_range = [-50, -50, -5, 50, 50, 3] → z_center = (-5+3)/2 = -1
    # 但实际 origin 会由 KittiSetOrigin 设置，这里用默认值
    origin_default = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    origin = origin_default

    valid_c_idx, valid_x, valid_y = compute_geometry_tensors(
        extrinsics, intrinsic, origin)
    save_tensor(valid_c_idx, str(frame_dir / "valid_c_idx.tensor"))
    save_tensor(valid_x,     str(frame_dir / "x.tensor"))
    save_tensor(valid_y,     str(frame_dir / "y.tensor"))

    # ── 保存帧元信息 ──────────────────────────────────────────────────────
    meta = {
        "frame_idx":    frame_idx,
        "sample_token": info.get("token", ""),
        "timestamp":    info.get("timestamp", 0),
        "origin":       origin.tolist(),
        "camera_names": [name_to_short[c] for c in camera_order],
        "camera_paths": {
            name_to_short[cam_name]: str(frame_dir / f"{ci}-{name_to_short[cam_name]}.jpg")
            for ci, cam_name in enumerate(camera_order)
        },
    }
    with open(frame_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if frame_idx % 10 == 0:
        print(f"  帧 {frame_idx:05d} [{info.get('token', '')[:12]}...] → {frame_dir}")
    return frame_dir


# ─── 从原始 NuScenes 数据集目录准备帧序列 ───────────────────────────────────

def _quat_to_rotation_matrix(q_wxyz) -> np.ndarray:
    """将 NuScenes 四元数 [w, x, y, z] 转换为 3×3 旋转矩阵。"""
    try:
        from scipy.spatial.transform import Rotation
        w, x, y, z = q_wxyz
        return Rotation.from_quat([x, y, z, w]).as_matrix()
    except ImportError:
        # 手动实现四元数→旋转矩阵
        w, x, y, z = q_wxyz
        return np.array([
            [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
            [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
            [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
        ], dtype=np.float64)


def prepare_frames_from_nuscenes_raw(
    nuscenes_dir: str,
    out_dir: Path,
    version: str = "v1.0-mini",
    start_idx: int = 0,
    num_frames: int = 50,
    stride: int = 1,
    output_width: int = 704,
    output_height: int = 256,
    scene_names: list = None,
) -> list:
    """
    从原始 NuScenes 数据集目录（包含 v1.0-mini/ 等 JSON 文件和 samples/ 目录）
    批量准备推理帧，无需 mmdet3d 或 pkl 文件。

    Args:
        nuscenes_dir:  NuScenes 根目录（包含 samples/, v1.0-mini/ 等）
        out_dir:       输出帧目录
        version:       数据集版本（v1.0-mini / v1.0-trainval）
        start_idx:     起始帧索引（按 keyframe 顺序）
        num_frames:    要准备的帧数量
        stride:        帧间隔
        output_width:  输出图像宽（模型输入尺寸，默认 704）
        output_height: 输出图像高（默认 256）
        scene_names:   指定 scene 名列表（None 表示全部）
    """
    nuscenes_dir = Path(nuscenes_dir)
    json_dir = nuscenes_dir / version

    # 加载所有 JSON 元数据
    def load_json(name):
        return json.load(open(str(json_dir / name)))

    sensor_list   = load_json("sensor.json")
    cal_list      = load_json("calibrated_sensor.json")
    sd_list       = load_json("sample_data.json")
    sample_list   = load_json("sample.json")
    ep_list       = load_json("ego_pose.json")
    scene_list    = load_json("scene.json")

    # 构建查找表
    sensor_map = {s["token"]: s for s in sensor_list}
    cal_map    = {c["token"]: c for c in cal_list}
    ep_map     = {e["token"]: e for e in ep_list}
    sample_map = {s["token"]: s for s in sample_list}

    # 为每个 sample_token 建立 channel → sample_data 映射
    sd_by_sample_channel = {}
    for sd in sd_list:
        if not sd["is_key_frame"]:
            continue
        cal_entry    = cal_map[sd["calibrated_sensor_token"]]
        sensor_entry = sensor_map[cal_entry["sensor_token"]]
        stok         = sd["sample_token"]
        chan         = sensor_entry["channel"]
        if stok not in sd_by_sample_channel:
            sd_by_sample_channel[stok] = {}
        sd_by_sample_channel[stok][chan] = (sd, cal_entry, sensor_entry)

    # 按时间戳排序 keyframe 样本
    all_scene_names = {sc["name"] for sc in scene_list}
    if scene_names:
        filtered_scenes = [sc for sc in scene_list if sc["name"] in scene_names]
    else:
        filtered_scenes = scene_list

    # 遍历场景，收集有序 keyframe
    ordered_samples = []
    for scene in filtered_scenes:
        tok = scene["first_sample_token"]
        while tok:
            s = sample_map.get(tok)
            if s is None:
                break
            ordered_samples.append(s)
            tok = s["next"]

    print(f"共找到 {len(ordered_samples)} 个 keyframe，从第 {start_idx} 帧开始，"
          f"步长 {stride}，准备 {num_frames} 帧")

    # 找 LIDAR_TOP 标定（多 log 对应多个标定，按场景获取也可，这里缓存所有）
    lidar_cals_by_sensor = {}
    for c in cal_list:
        if sensor_map[c["sensor_token"]]["channel"] == "LIDAR_TOP":
            lidar_cals_by_sensor[c["token"]] = c

    # 为每个 sample_token 找 LIDAR_TOP 对应的标定 token
    lidar_cal_by_sample = {}
    for sd in sd_list:
        cal_entry    = cal_map[sd["calibrated_sensor_token"]]
        sensor_entry = sensor_map[cal_entry["sensor_token"]]
        if sensor_entry["channel"] == "LIDAR_TOP" and sd["is_key_frame"]:
            lidar_cal_by_sample[sd["sample_token"]] = cal_entry

    camera_order = [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK",  "CAM_BACK_LEFT",   "CAM_BACK_RIGHT",
    ]
    name_to_short = {
        "CAM_FRONT":       "FRONT",
        "CAM_FRONT_RIGHT": "FRONT_RIGHT",
        "CAM_FRONT_LEFT":  "FRONT_LEFT",
        "CAM_BACK":        "BACK",
        "CAM_BACK_LEFT":   "BACK_LEFT",
        "CAM_BACK_RIGHT":  "BACK_RIGHT",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    indices = list(range(start_idx,
                         min(len(ordered_samples), start_idx + num_frames * stride),
                         stride))[:num_frames]

    prepared = []
    for render_idx, data_idx in enumerate(indices):
        sample = ordered_samples[data_idx]
        stok   = sample["token"]

        frame_dir = out_dir / f"frame_{render_idx:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        # ── 获取 lidar 标定 ──────────────────────────────────────────────
        lidar_cal = lidar_cal_by_sample.get(stok)
        if lidar_cal is None:
            # fallback: 直接从 sample_data 找 LIDAR_TOP
            for sd in sd_list:
                if sd["sample_token"] == stok and sd["is_key_frame"]:
                    ce = cal_map[sd["calibrated_sensor_token"]]
                    se = sensor_map[ce["sensor_token"]]
                    if se["channel"] == "LIDAR_TOP":
                        lidar_cal = ce
                        break
        if lidar_cal is None:
            print(f"  [警告] 样本 {stok[:12]} 找不到 LIDAR_TOP 标定，跳过")
            continue

        R_l = _quat_to_rotation_matrix(lidar_cal["rotation"])
        t_l = np.array(lidar_cal["translation"], dtype=np.float64)
        lidar_yaw_in_ego = math.atan2(R_l[1, 0], R_l[0, 0])

        # ── 处理 6 路相机 ─────────────────────────────────────────────────
        extrinsics = []
        intrinsics_all = []
        intrinsics_raw = []   # 原始内参（1600×900 尺度，用于 3D 框投影到原图）
        cam_channels = sd_by_sample_channel.get(stok, {})

        for cam_idx, cam_name in enumerate(camera_order):
            if cam_name not in cam_channels:
                print(f"  [警告] 样本 {stok[:12]} 缺少相机: {cam_name}")
                extrinsics.append(np.eye(4))
                intrinsics_all.append(None)
                intrinsics_raw.append(None)
                continue

            sd_entry, cam_cal, _ = cam_channels[cam_name]

            # ── 图像加载（不缩放，C++ 内部做 resize+crop）────────────────
            orig_w = sd_entry.get("width",  1600)
            orig_h = sd_entry.get("height",  900)
            img_path = nuscenes_dir / sd_entry["filename"]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [警告] 无法读取: {img_path}")
                img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            # C++ 期望原始分辨率（1600×900），内部 CUDA kernel 做 resize+crop
            short_name = name_to_short[cam_name]
            cv2.imwrite(str(frame_dir / f"{cam_idx}-{short_name}.jpg"), img)

            # ── 内参：镜像 C++ 的 resize_lim + center_crop 变换 ─────────
            # C++ 做: resize(1600×900, resize_lim=0.44) → 704×396
            #         center_crop(704×396, 704×256): crop_y=70, crop_x=0
            K_raw = np.array(cam_cal["camera_intrinsic"], dtype=np.float64)  # 3×3
            resize_lim = float(output_width) / orig_w   # 704/1600 = 0.44
            resized_h  = int(orig_h * resize_lim)       # int(900*0.44) = 396
            crop_y_off = (resized_h - output_height) // 2  # (396-256)//2 = 70
            K = K_raw.copy()
            K[0] *= resize_lim    # 等比缩放 fx, cx
            K[1] *= resize_lim    # 等比缩放 fy, cy
            K[1, 2] -= crop_y_off  # cy 减去垂直裁剪偏移
            intrinsics_all.append(K)
            intrinsics_raw.append(K_raw)  # 保存原始内参用于 3D 框投影

            # ── 外参：lidar → camera ──────────────────────────────────────
            # ego 中相机坐标系: P_ego = R_c @ P_cam + t_c
            # lidar → ego:      P_ego = R_l @ P_lidar + t_l
            # lidar → camera:   P_cam = R_c^T @ (R_l @ P_lidar + t_l - t_c)
            R_c = _quat_to_rotation_matrix(cam_cal["rotation"])
            t_c = np.array(cam_cal["translation"], dtype=np.float64)

            R_l2c = R_c.T @ R_l
            t_l2c = R_c.T @ (t_l - t_c)

            E = np.eye(4, dtype=np.float64)
            E[:3, :3] = R_l2c
            E[:3,  3] = t_l2c
            extrinsics.append(E)

        # 选用第一个有效相机的内参（所有相机用同一标量缩放，但各相机内参不同；
        # FastBEV 原始实现用单个 intrinsic 矩阵，这里使用 FRONT 相机）
        front_idx = camera_order.index("CAM_FRONT")
        intrinsic = intrinsics_all[front_idx]
        if intrinsic is None:
            # fallback
            intrinsic = np.array([[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]])

        # ── 计算几何张量 ──────────────────────────────────────────────────
        # 每个相机独立使用自己的内参计算投影
        origin = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        valid_c_idx, valid_x, valid_y = compute_geometry_tensors_per_cam(
            extrinsics, intrinsics_all, origin)

        save_tensor(valid_c_idx, str(frame_dir / "valid_c_idx.tensor"))
        save_tensor(valid_x,     str(frame_dir / "x.tensor"))
        save_tensor(valid_y,     str(frame_dir / "y.tensor"))

        # ── 获取 ego pose（自车全局位姿） ──────────────────────────────────
        ego_translation_global = [0.0, 0.0, 0.0]
        ego_yaw_global = 0.0
        if "LIDAR_TOP" in cam_channels:
            lidar_sd_entry, _, _ = cam_channels["LIDAR_TOP"]
            ep = ep_map.get(lidar_sd_entry.get("ego_pose_token", ""), {})
            if ep:
                ego_translation_global = list(ep["translation"])  # [x, y, z] 全局坐标
                qw, qx, qy, qz = ep["rotation"]  # NuScenes 四元数 [w, x, y, z]
                ego_yaw_global = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz)
                )

        # ── 保存帧元信息 ──────────────────────────────────────────────────
        meta = {
            "frame_idx":    render_idx,
            "sample_token": stok,
            "timestamp":    sample.get("timestamp", 0),
            "origin":       origin.tolist(),
            "camera_names": [name_to_short[c] for c in camera_order],
            # 自车全局位姿（用于计算速度、显示朝向）
            "ego_translation_global": ego_translation_global,
            "ego_yaw_global":         ego_yaw_global,
            "lidar_yaw_in_ego":       lidar_yaw_in_ego,
            # 相机标定（用于将 3D LiDAR 框投影到相机图像）
            "lidar2cam_extrinsics": [e.tolist() for e in extrinsics],
            "cam_intrinsics_raw": [
                k.tolist() if k is not None else None
                for k in intrinsics_raw
            ],
        }
        with open(frame_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if render_idx % 10 == 0 or render_idx == 0:
            print(f"  帧 {render_idx:05d} [{stok[:12]}...] → {frame_dir}")

        prepared.append({
            "frame_idx":    render_idx,
            "data_idx":     data_idx,
            "frame_dir":    str(frame_dir),
            "sample_token": stok,
            "timestamp":    sample.get("timestamp", 0),
        })

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(prepared, f, indent=2)
    print(f"序列清单 → {manifest_path}  (共 {len(prepared)} 帧)")
    return prepared


def compute_geometry_tensors_per_cam(
    extrinsics: list,       # list of 6 E [4×4] lidar→camera
    intrinsics: list,       # list of 6 K [3×3]（各相机独立内参，已缩放到输出尺寸）
    origin: np.ndarray,
    feat_stride: int = 4,
) -> tuple:
    """
    与 compute_geometry_tensors 功能相同，但每个相机使用各自的内参矩阵。
    """
    num_cameras = len(extrinsics)
    nx, ny, nz  = int(N_VOXELS[0]), int(N_VOXELS[1]), int(N_VOXELS[2])
    num_voxels  = nx * ny * nz

    pts_3d = get_voxel_points(origin).reshape(3, -1)
    pts_h  = np.vstack([pts_3d, np.ones((1, num_voxels), dtype=np.float64)])

    valid_c_idx = np.zeros((num_cameras, num_voxels), dtype=np.float32)
    valid_x     = np.zeros((num_cameras, num_voxels), dtype=np.int64)
    valid_y     = np.zeros((num_cameras, num_voxels), dtype=np.int64)

    for cam_idx, (ext, intr) in enumerate(zip(extrinsics, intrinsics)):
        if intr is None:
            continue
        E = np.array(ext, dtype=np.float64)
        K = intr[:3, :3].copy().astype(np.float64)
        # 特征图尺度
        K_feat = K.copy()
        K_feat[0] /= feat_stride
        K_feat[1] /= feat_stride

        P     = K_feat @ E[:3, :]
        pts_2d = P @ pts_h
        z_val  = pts_2d[2]

        safe_z = np.where(np.abs(z_val) > 1e-6, z_val, 1.0)
        x_img  = np.round(pts_2d[0] / safe_z).astype(np.int64)
        y_img  = np.round(pts_2d[1] / safe_z).astype(np.int64)

        valid_mask = (
            (z_val > 0) &
            (x_img >= 0) & (x_img < FEAT_WIDTH) &
            (y_img >= 0) & (y_img < FEAT_HEIGHT)
        )

        valid_c_idx[cam_idx]         = valid_mask.astype(np.float32)
        valid_x[cam_idx, valid_mask] = x_img[valid_mask]
        valid_y[cam_idx, valid_mask] = y_img[valid_mask]

    return valid_c_idx, valid_x, valid_y

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NuScenes → CUDA-FastBEV 数据适配器")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--nuscenes-dir", type=str,
                      help="原始 NuScenes 数据集根目录（含 samples/ 和 v1.0-*/）")
    mode.add_argument("--pkl-path",  type=str, help="NuScenes mmdet3d pkl 文件路径")
    mode.add_argument("--pth-path",  type=str, help="example-data.pth 文件路径（兼容模式）")

    p.add_argument("--version",       type=str, default="v1.0-mini",
                   help="NuScenes 数据集版本（仅 --nuscenes-dir 模式）")
    p.add_argument("--scene-names",   type=str, default=None,
                   help="指定场景名，逗号分隔（仅 --nuscenes-dir 模式）")
    p.add_argument("--nuscenes-root", type=str, default="/data/nuscenes",
                   help="NuScenes 数据集根目录（--pkl-path 模式下的图像根目录）")
    p.add_argument("--out-dir",     type=str, default="outputs/frames",
                   help="输出帧目录")
    p.add_argument("--start-idx",   type=int, default=0,
                   help="起始帧索引")
    p.add_argument("--num-frames",  type=int, default=50,
                   help="要准备的帧数量")
    p.add_argument("--stride",      type=int, default=1,
                   help="帧间隔")
    p.add_argument("--image-src-dir", type=str, default=None,
                   help="（pth 模式）图像文件目录")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pth_path:
        print(f"[兼容模式] 从 {args.pth_path} 准备帧数据")
        prepare_frame_from_pth(
            pth_path     = args.pth_path,
            out_dir      = out_dir,
            frame_idx    = 0,
            image_src_dir= args.image_src_dir or str(Path(args.pth_path).parent),
        )
    elif args.pkl_path:
        print(f"[NuScenes pkl 模式] 从 {args.pkl_path} 准备 {args.num_frames} 帧")
        prepare_frames_from_nuscenes_pkl(
            pkl_path      = args.pkl_path,
            nuscenes_root = args.nuscenes_root,
            out_dir       = out_dir,
            start_idx     = args.start_idx,
            num_frames    = args.num_frames,
            stride        = args.stride,
        )
    else:
        scene_names = None
        if args.scene_names:
            scene_names = [s.strip() for s in args.scene_names.split(",")]
        print(f"[NuScenes 原始模式] 从 {args.nuscenes_dir} ({args.version}) "
              f"准备 {args.num_frames} 帧")
        prepare_frames_from_nuscenes_raw(
            nuscenes_dir  = args.nuscenes_dir,
            out_dir       = out_dir,
            version       = args.version,
            start_idx     = args.start_idx,
            num_frames    = args.num_frames,
            stride        = args.stride,
            scene_names   = scene_names,
        )
    print("完成！")


if __name__ == "__main__":
    main()
