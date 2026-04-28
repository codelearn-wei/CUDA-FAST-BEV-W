#!/usr/bin/env python3
"""
prepare_frame_dir.py — 将相机图像 + 标定参数打包为 CUDA-FastBEV 帧目录

用法：

  # 方式 1：图像目录 + meta.json（推荐给摄像头开发者）
  python tools/prepare_frame_dir.py \\
      --images-dir /path/to/images \\
      --meta /path/to/calib.json \\
      --out-dir outputs/frames \\
      --compute-tensors

  # 方式 2：逐一指定 6 张图像 + meta.json
  python tools/prepare_frame_dir.py \\
      --images front.jpg front_right.jpg front_left.jpg \\
               back.jpg  back_left.jpg   back_right.jpg \\
      --meta calib.json \\
      --out-dir outputs/frames

  # 方式 3：从 NuScenes 数据集准备多帧（调用 nuscenes_adapter.py）
  python tools/nuscenes_adapter.py \\
      --nuscenes-dir data/nuscenes --version v1.0-mini \\
      --out-dir outputs/frames --num-frames 50

图像顺序（6 路）:
  0 = FRONT, 1 = FRONT_RIGHT, 2 = FRONT_LEFT,
  3 = BACK,  4 = BACK_LEFT,   5 = BACK_RIGHT

meta.json 格式（标定文件，必须字段）:
  见下方 META_TEMPLATE 或 README "帧目录数据格式" 章节。

环境: conda activate bev（或任何安装了 numpy/opencv 的 Python 环境）
"""

import argparse
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_W, OUTPUT_H = 704, 256   # 模型输入图像尺寸
ORIG_W,   ORIG_H   = 1600, 900  # 标定时的原始分辨率

CAM_NAMES_SHORT = ["FRONT", "FRONT_RIGHT", "FRONT_LEFT",
                   "BACK",  "BACK_LEFT",   "BACK_RIGHT"]

# 体素网格参数（与训练配置一致，勿修改）
N_VOXELS   = np.array([200, 200, 4], dtype=np.float64)
VOXEL_SIZE = np.array([0.5,  0.5,  1.5], dtype=np.float64)
FEAT_H, FEAT_W = 64, 176   # 256/4, 704/4

# .tensor 文件魔数和类型映射
TENSOR_MAGIC = 0x33FF1101
DTYPE_MAP = {"float32": 3, "int64": 4}

# meta.json 模板（供用户参考）
META_TEMPLATE = {
    "frame_idx": 0,
    "timestamp": 0,
    "origin": [0.0, 0.0, -1.0],
    "camera_names": ["FRONT", "FRONT_RIGHT", "FRONT_LEFT",
                     "BACK", "BACK_LEFT", "BACK_RIGHT"],
    "ego_translation_global": [0.0, 0.0, 0.0],
    "ego_yaw_global": 0.0,
    "lidar2cam_extrinsics": [
        "# 每路相机：4x4 矩阵，lidar→camera 齐次变换（row-major）",
        "# [[r00,r01,r02,tx],[r10,...],[r20,...],[0,0,0,1]]",
        "# 6 路相机各自独立，顺序同 camera_names"
    ],
    "cam_intrinsics_raw": [
        "# 每路相机：3x3 内参矩阵，对应 ORIG_W×ORIG_H（1600×900）分辨率",
        "# [[fx,0,cx],[0,fy,cy],[0,0,1]]",
        "# 若相机实际分辨率不同，需缩放内参到等效 1600×900 基准"
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# 几何张量计算
# ─────────────────────────────────────────────────────────────────────────────

def _save_tensor(arr: np.ndarray, path: str):
    dtype_str = str(arr.dtype)
    with open(path, "wb") as f:
        f.write(struct.pack("3i", TENSOR_MAGIC, arr.ndim, DTYPE_MAP[dtype_str]))
        f.write(struct.pack(f"{arr.ndim}i", *arr.shape))
        f.write(arr.tobytes())


def _get_voxel_points(origin: np.ndarray) -> np.ndarray:
    """返回体素中心点坐标 (3, 200, 200, 4)，reshape 为 (3, N)。"""
    new_origin = origin - N_VOXELS / 2.0 * VOXEL_SIZE
    nx, ny, nz = 200, 200, 4
    ix = np.arange(nx, dtype=np.float64)
    iy = np.arange(ny, dtype=np.float64)
    iz = np.arange(nz, dtype=np.float64)
    shape = (nx, ny, nz)
    px = np.broadcast_to(
        (ix[:, None, None] * VOXEL_SIZE[0] + new_origin[0]).reshape(nx, 1, 1), shape)
    py = np.broadcast_to(
        (iy[None, :, None] * VOXEL_SIZE[1] + new_origin[1]).reshape(1, ny, 1), shape)
    pz = np.broadcast_to(
        (iz[None, None, :] * VOXEL_SIZE[2] + new_origin[2]).reshape(1, 1, nz), shape)
    pts = np.stack([px, py, pz], axis=0).reshape(3, -1).copy()
    return pts  # (3, 160000)


def compute_geometry_tensors(extrinsics, intrinsics_scaled, origin):
    """
    计算三个几何张量，供 C++ BEV 推理使用。

    参数:
        extrinsics:        list[np.ndarray 4x4]  lidar→camera，6 路
        intrinsics_scaled: list[np.ndarray 3x3]  缩放到 OUTPUT_W×OUTPUT_H 的内参
        origin:            np.ndarray(3,)         体素网格中心，通常 [0,0,-1]

    返回:
        valid_c_idx: (6, 160000) float32  — 哪个相机覆盖该体素
        valid_x:     (6, 160000) int64    — 投影到特征图的 x 坐标
        valid_y:     (6, 160000) int64    — 投影到特征图的 y 坐标
    """
    n_cams  = len(extrinsics)
    n_voxels = 200 * 200 * 4
    pts_3d = _get_voxel_points(origin)           # (3, N)
    pts_h  = np.vstack([pts_3d, np.ones((1, n_voxels), dtype=np.float64)])  # (4, N)

    valid_c_idx = np.zeros((n_cams, n_voxels), dtype=np.float32)
    valid_x     = np.zeros((n_cams, n_voxels), dtype=np.int64)
    valid_y     = np.zeros((n_cams, n_voxels), dtype=np.int64)

    feat_stride = OUTPUT_W // FEAT_W  # = 4
    for ci, (E, K) in enumerate(zip(extrinsics, intrinsics_scaled)):
        if K is None:
            continue
        K_feat = K[:3, :3].copy()
        K_feat[0] /= feat_stride
        K_feat[1] /= feat_stride
        P      = K_feat @ E[:3, :]          # (3, 4)
        pts_2d = P @ pts_h                  # (3, N)

        z_val  = pts_2d[2]
        safe_z = np.where(np.abs(z_val) > 1e-6, z_val, 1.0)
        xi = np.round(pts_2d[0] / safe_z).astype(np.int64)
        yi = np.round(pts_2d[1] / safe_z).astype(np.int64)

        mask = ((z_val > 0) &
                (xi >= 0) & (xi < FEAT_W) &
                (yi >= 0) & (yi < FEAT_H))
        valid_c_idx[ci]        = mask.astype(np.float32)
        valid_x[ci, mask]      = xi[mask]
        valid_y[ci, mask]      = yi[mask]

    return valid_c_idx, valid_x, valid_y


# ─────────────────────────────────────────────────────────────────────────────
# 图像处理
# ─────────────────────────────────────────────────────────────────────────────

def _resize_crop(img: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """缩放 + 中央裁剪到 OUTPUT_W×OUTPUT_H，与训练预处理一致。"""
    # 按宽度缩放
    scale = OUTPUT_W / orig_w
    new_w = OUTPUT_W
    new_h = int(orig_h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # 中央裁剪高度
    crop_top = (new_h - OUTPUT_H) // 2
    img_cropped = img_resized[crop_top: crop_top + OUTPUT_H, :, :]
    return img_cropped


def _scale_intrinsics(K_raw: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    """将原始分辨率内参缩放到 OUTPUT_W×OUTPUT_H。"""
    scale = OUTPUT_W / orig_w
    new_h_scaled = int(orig_h * scale)
    crop_top = (new_h_scaled - OUTPUT_H) // 2
    K = K_raw.copy()
    K[0] *= scale
    K[1] *= scale
    K[1, 2] -= crop_top
    return K


# ─────────────────────────────────────────────────────────────────────────────
# 主逻辑
# ─────────────────────────────────────────────────────────────────────────────

def prepare_frame(
    image_paths: list,
    meta_json_path: str,
    out_dir: str,
    frame_idx: int = 0,
    compute_tensors: bool = True,
    overwrite: bool = False,
) -> str:
    """
    创建一个 frame_NNNNN/ 目录，包含：
      - 0-FRONT.jpg … 5-BACK_RIGHT.jpg（统一缩放到 704×256）
      - meta.json（含标定信息）
      - valid_c_idx.tensor, x.tensor, y.tensor（若 compute_tensors=True）

    参数:
        image_paths:      6 张图像的路径列表（顺序同 CAM_NAMES_SHORT）
        meta_json_path:   标定 JSON 文件路径（字段见 README）
        out_dir:          输出根目录（frame_NNNNN 在此目录下创建）
        frame_idx:        帧编号（0-based）
        compute_tensors:  是否预计算几何张量（推荐，可加速推理）
        overwrite:        若 frame 目录已存在，是否覆盖

    返回:
        frame 目录的绝对路径
    """
    frame_dir = Path(out_dir) / f"frame_{frame_idx:05d}"
    if frame_dir.exists() and not overwrite:
        print(f"  [跳过] {frame_dir} 已存在（传入 --overwrite 强制重建）")
        return str(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    # ── 读取 meta.json ──────────────────────────────────────────────────────
    with open(meta_json_path) as f:
        meta = json.load(f)

    extrinsics_raw = meta.get("lidar2cam_extrinsics")
    intrinsics_raw = meta.get("cam_intrinsics_raw")
    if not extrinsics_raw or not intrinsics_raw:
        raise ValueError(
            "meta.json 缺少必要字段 lidar2cam_extrinsics 或 cam_intrinsics_raw\n"
            "  运行 python tools/prepare_frame_dir.py --print-template 查看完整格式")

    if len(image_paths) != 6:
        raise ValueError(f"需要恰好 6 张图像，当前提供了 {len(image_paths)} 张")

    orig_w = meta.get("orig_width",  ORIG_W)
    orig_h = meta.get("orig_height", ORIG_H)

    # ── 处理图像 ────────────────────────────────────────────────────────────
    extrinsics_list     = []
    intrinsics_scaled   = []

    for cam_idx, (img_path, cam_name) in enumerate(zip(image_paths, CAM_NAMES_SHORT)):
        # 读图并缩放裁剪
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [警告] 无法读取图像: {img_path}，用黑图替代")
            img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        else:
            ih, iw = img.shape[:2]
            if iw != orig_w or ih != orig_h:
                # 自适应：以实际尺寸为基准缩放
                orig_w_this = iw
                orig_h_this = ih
            else:
                orig_w_this = orig_w
                orig_h_this = orig_h
            img = _resize_crop(img, orig_w_this, orig_h_this)

        out_name = f"{cam_idx}-{cam_name}.jpg"
        cv2.imwrite(str(frame_dir / out_name), img)

        # 标定
        E_raw = extrinsics_raw[cam_idx]
        K_raw = intrinsics_raw[cam_idx]

        E = np.array(E_raw, dtype=np.float64) if E_raw is not None else np.eye(4)
        extrinsics_list.append(E)

        if K_raw is not None:
            K_raw_np = np.array(K_raw, dtype=np.float64)
            K_scaled  = _scale_intrinsics(K_raw_np, orig_w, orig_h)
            intrinsics_scaled.append(K_scaled)
        else:
            intrinsics_scaled.append(None)

    # ── 写入 meta.json ──────────────────────────────────────────────────────
    meta_out = {
        "frame_idx":              frame_idx,
        "timestamp":              meta.get("timestamp", 0),
        "origin":                 meta.get("origin", [0.0, 0.0, -1.0]),
        "camera_names":           CAM_NAMES_SHORT,
        "ego_translation_global": meta.get("ego_translation_global", [0.0, 0.0, 0.0]),
        "ego_yaw_global":         meta.get("ego_yaw_global", 0.0),
        "lidar2cam_extrinsics":   [e.tolist() for e in extrinsics_list],
        "cam_intrinsics_raw":     [
            k.tolist() if k is not None else None for k in
            [np.array(r, dtype=np.float64) if r is not None else None
             for r in intrinsics_raw]
        ],
    }
    with open(frame_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    # ── 计算几何张量（可选）────────────────────────────────────────────────
    if compute_tensors:
        origin = np.array(meta_out["origin"], dtype=np.float64)
        print(f"  计算几何张量（origin={origin.tolist()}）...")
        vc, vx, vy = compute_geometry_tensors(extrinsics_list, intrinsics_scaled, origin)
        _save_tensor(vc, str(frame_dir / "valid_c_idx.tensor"))
        _save_tensor(vx, str(frame_dir / "x.tensor"))
        _save_tensor(vy, str(frame_dir / "y.tensor"))
        print(f"  ✓ 几何张量已保存（valid_c_idx/x/y .tensor）")

    print(f"  ✓ 帧目录已创建: {frame_dir}")
    return str(frame_dir)


def _find_images_in_dir(images_dir: Path) -> list:
    """
    在目录中按名称模式查找 6 路相机图像。
    支持两种命名格式：
      - {index}-{CAMNAME}.jpg  （例如 0-FRONT.jpg）
      - 任意含有 front/back 等关键词的 jpg 文件（按顺序匹配）
    """
    candidates = []
    for cam_idx, cam_name in enumerate(CAM_NAMES_SHORT):
        # 优先精确匹配 0-FRONT.jpg 格式
        exact = images_dir / f"{cam_idx}-{cam_name}.jpg"
        if exact.exists():
            candidates.append(str(exact))
            continue
        # 模糊匹配：文件名含 cam_name（不区分大小写）
        found = sorted(images_dir.glob(f"*{cam_name.lower()}*.jpg"))
        if not found:
            found = sorted(images_dir.glob(f"*{cam_name}*.jpg"))
        if found:
            candidates.append(str(found[0]))
        else:
            candidates.append(None)

    missing = [f"{i}-{CAM_NAMES_SHORT[i]}" for i, p in enumerate(candidates) if p is None]
    if missing:
        raise FileNotFoundError(
            f"在 {images_dir} 中找不到以下相机的图像: {missing}\n"
            f"  期望文件名格式: 0-FRONT.jpg, 1-FRONT_RIGHT.jpg, ...")

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="从相机图像 + 标定参数创建 CUDA-FastBEV 帧目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法：图像目录 + 标定文件
  python tools/prepare_frame_dir.py \\
      --images-dir /path/to/six_images \\
      --meta calib.json \\
      --out-dir outputs/frames \\
      --compute-tensors

  # 逐一指定 6 张图像（FRONT → BACK_RIGHT 顺序）
  python tools/prepare_frame_dir.py \\
      --images front.jpg fr.jpg fl.jpg back.jpg bl.jpg br.jpg \\
      --meta calib.json --out-dir outputs/frames

  # 打印 meta.json 模板（标定文件格式参考）
  python tools/prepare_frame_dir.py --print-template

  # 批量准备多帧（loop --frame-idx）
  for i in $(seq 0 49); do
    python tools/prepare_frame_dir.py \\
        --images-dir /data/cam_frames/frame_$(printf '%05d' $i) \\
        --meta /data/calib.json \\
        --frame-idx $i --out-dir outputs/frames --compute-tensors
  done
""")

    p.add_argument("--images-dir", type=str, default=None,
                   help="包含 6 路相机图像的目录（支持 0-FRONT.jpg 或 *front*.jpg 命名）")
    p.add_argument("--images", nargs=6, metavar=("FRONT", "FR", "FL", "BACK", "BL", "BR"),
                   help="直接指定 6 张图像路径（顺序: FRONT FR FL BACK BL BR）")
    p.add_argument("--meta", type=str, required=False,
                   help="标定 JSON 文件路径（必须包含 lidar2cam_extrinsics + cam_intrinsics_raw）")
    p.add_argument("--out-dir", type=str, default="outputs/frames",
                   help="输出根目录（默认: outputs/frames）")
    p.add_argument("--frame-idx", type=int, default=0,
                   help="帧编号（默认: 0）")
    p.add_argument("--compute-tensors", action="store_true",
                   help="预计算几何张量（valid_c_idx.tensor 等），推荐，可显著加速推理")
    p.add_argument("--overwrite", action="store_true",
                   help="若帧目录已存在则覆盖")
    p.add_argument("--print-template", action="store_true",
                   help="打印 meta.json 格式模板并退出")

    args = p.parse_args()

    if args.print_template:
        print("# meta.json 标定文件模板")
        print("# 将此文件作为 --meta 参数传入")
        print("#")
        print("# 字段说明:")
        print("#   lidar2cam_extrinsics: [6][4][4]  lidar→camera 齐次变换矩阵")
        print("#     每路相机独立，旋转部分 E[:3,:3] 为相机系到 lidar 系的旋转的转置")
        print("#     平移部分 E[:3,3] = R_cam.T @ (t_lidar - t_cam)")
        print("#   cam_intrinsics_raw: [6][3][3]  原始分辨率（1600×900）内参矩阵")
        print("#     若相机实际分辨率不同，在 meta.json 中同时指定:")
        print("#     orig_width / orig_height（脚本自动按比例缩放内参）")
        print("#   ego_translation_global: [x,y,z]  自车位置（全局坐标系，单位米）")
        print("#   ego_yaw_global: float             自车航向角（全局，弧度）")
        print("#   origin: [x,y,z]                   体素网格中心（通常 [0,0,-1]，勿改）")
        print()
        import json as _json
        template = {
            "frame_idx": 0,
            "timestamp": 0,
            "orig_width": 1600,
            "orig_height": 900,
            "origin": [0.0, 0.0, -1.0],
            "camera_names": CAM_NAMES_SHORT,
            "ego_translation_global": [0.0, 0.0, 0.0],
            "ego_yaw_global": 0.0,
            "lidar2cam_extrinsics": [
                [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            ],
            "cam_intrinsics_raw": [
                [[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]],
                [[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]],
                [[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]],
                [[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]],
                [[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]],
                [[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]],
            ],
        }
        print(_json.dumps(template, indent=2))
        return

    # ── 解析图像路径 ──────────────────────────────────────────────────────
    if args.images:
        image_paths = [os.path.abspath(p) for p in args.images]
    elif args.images_dir:
        image_paths = _find_images_in_dir(Path(args.images_dir))
    else:
        print("错误：请指定 --images-dir 或 --images", file=sys.stderr)
        p.print_help()
        sys.exit(1)

    if not args.meta:
        # 尝试从 images-dir 自动查找 meta.json
        if args.images_dir and (Path(args.images_dir) / "meta.json").exists():
            meta_path = str(Path(args.images_dir) / "meta.json")
            print(f"  自动使用标定文件: {meta_path}")
        else:
            print("错误：请指定 --meta 标定文件（运行 --print-template 查看格式）",
                  file=sys.stderr)
            sys.exit(1)
    else:
        meta_path = os.path.abspath(args.meta)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    frame_dir = prepare_frame(
        image_paths     = image_paths,
        meta_json_path  = meta_path,
        out_dir         = out_dir,
        frame_idx       = args.frame_idx,
        compute_tensors = args.compute_tensors,
        overwrite       = args.overwrite,
    )
    print(f"\n完成！帧目录: {frame_dir}")
    print(f"接下来运行推理:")
    print(f"  ./build/joint_inference {out_dir} resnet18int8 int8 \\")
    print(f"      --map-mode model --ckpt model/maptr/maptr_nano_r18_110e.pth --save-json")


if __name__ == "__main__":
    main()
