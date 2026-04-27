#!/usr/bin/env python3
"""
maptr_infer_worker.py — MapTRv2 模型推理 Worker（在 maptr conda 环境中运行）
=============================================================================

在 maptr conda 环境（mmcv 1.x + MapTRv2 mmdetection3d）中运行，
处理 FastBEV 帧目录，输出 map_result.json。

用法（由 maptr_model_infer.py 调用，不建议直接运行）：
  conda run -n maptr python tools/maptr/maptr_infer_worker.py \
      --frames-dir outputs/frames \
      --config model/maptr/config/maptr_nano_r18_110e.py \
      --checkpoint model/maptr/maptr_nano_r18_110e.pth \
      --score-thr 0.3 [--overwrite]

输出：每个 frame_dir 下的 map_result.json（source="model"）
"""

import os
import sys
import json
import math
import argparse
import traceback
from pathlib import Path

import numpy as np

# ─── 路径设置（在任何 import 之前）──────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MAPTR_REPO   = os.path.join(PROJECT_ROOT, 'tools', 'MapTRv2')

# GeometricKernelAttention .so 所在目录（需要在 sys.path 最前面）
GKA_DIR = os.path.join(
    MAPTR_REPO,
    'projects', 'mmdet3d_plugin', 'maptr', 'modules', 'ops',
    'geometric_kernel_attn'
)

# 把 MapTRv2 项目、mmdetection3d 和 GKA 算子目录插入路径最前面
for p in [GKA_DIR, MAPTR_REPO, os.path.join(MAPTR_REPO, 'mmdetection3d')]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 切换工作目录到 MapTRv2（projects/mmdet3d_plugin 要求相对导入）
_old_cwd = os.getcwd()
os.chdir(MAPTR_REPO)

# ─── 注册 MapTRv2 所有模型（必须在 build_model 之前）─────────────────────────
import projects.mmdet3d_plugin  # noqa: E402,F401 — 触发 registry 注册

# ─── 模型相关常量 ────────────────────────────────────────────────────────────
MAP_CLASSES   = ['divider', 'ped_crossing', 'boundary']
IMG_MEAN      = np.array([123.675, 116.28,  103.53 ], dtype=np.float32)
IMG_STD       = np.array([58.395,  57.12,   57.375 ], dtype=np.float32)
MAPTR_SCALE   = 0.2              # 原始图像（1600×900）的缩放比例
ORIG_W, ORIG_H = 1600, 900       # NuScenes 原始图像分辨率
TARGET_W = int(ORIG_W * MAPTR_SCALE)  # 320
TARGET_H = int(ORIG_H * MAPTR_SCALE)  # 180
PAD_DIVISOR   = 32
PAD_H = math.ceil(TARGET_H / PAD_DIVISOR) * PAD_DIVISOR  # 192
PAD_W = TARGET_W                 # 320（已整除 32）
N_CAMS = 6


def _pad_to_divisor(img: np.ndarray, divisor: int) -> np.ndarray:
    """右/下填充图像至 divisor 的倍数。"""
    h, w = img.shape[:2]
    new_h = math.ceil(h / divisor) * divisor
    new_w = math.ceil(w / divisor) * divisor
    if new_h == h and new_w == w:
        return img
    padded = np.zeros((new_h, new_w, 3), dtype=img.dtype)
    padded[:h, :w] = img
    return padded


def preprocess_image(img_path: str) -> np.ndarray:
    """
    加载并预处理单张相机图像。
    步骤：读取 BGR → 转 RGB → resize 到 320×180 → pad 到 320×192 → normalize → [C,H,W]
    """
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        # 如果读取失败，返回全零张量
        return np.zeros((3, PAD_H, PAD_W), dtype=np.float32)
    # BGR → RGB
    img = img[:, :, ::-1].astype(np.float32)
    # resize 到 320×180
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    # pad 到 320×192
    img = _pad_to_divisor(img, PAD_DIVISOR)
    # 归一化（ImageNet mean/std，RGB）
    img = (img - IMG_MEAN) / IMG_STD
    # [H, W, C] → [C, H, W]
    return img.transpose(2, 0, 1).astype(np.float32)


def build_intrinsic_scaled(K_raw: list) -> np.ndarray:
    """
    将原始内参（1600×900 尺度）缩放到 MapTRv2 输入分辨率（320×180）。
    scale = 0.2，仅缩放 fx, fy, cx, cy，z 行不变。
    """
    K = np.array(K_raw, dtype=np.float64)  # [3, 3]
    K[0, :] *= MAPTR_SCALE   # 缩放 fx, 0, cx
    K[1, :] *= MAPTR_SCALE   # 缩放 0, fy, cy
    return K


def build_lidar2img(K_scaled: np.ndarray, lidar2cam: list) -> np.ndarray:
    """
    计算 lidar2img = K_4x4 @ lidar2cam_extrinsic。
    K_scaled: [3,3] 已缩放内参
    lidar2cam: [4,4] lidar 到相机的外参矩阵
    """
    K_4x4 = np.eye(4, dtype=np.float64)
    K_4x4[:3, :3] = K_scaled
    E = np.array(lidar2cam, dtype=np.float64)  # [4, 4]
    return K_4x4 @ E


def build_can_bus(ego_trans: list, ego_yaw: float) -> np.ndarray:
    """
    构造 MapTRv2 can_bus 18 维向量。
    [0:3]  = ego_translation_global
    [3:7]  = quaternion [qx, qy, qz, qw] from yaw = [0, 0, sin(yaw/2), cos(yaw/2)]
    [7:17] = 0（无速度/加速度）
    [17]   = ego_yaw_global（弧度）
    """
    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[0:3] = ego_trans[:3]
    can_bus[3]   = 0.0
    can_bus[4]   = 0.0
    can_bus[5]   = math.sin(ego_yaw / 2.0)
    can_bus[6]   = math.cos(ego_yaw / 2.0)
    can_bus[17]  = ego_yaw
    return can_bus


def build_input_for_frame(frame_dir: str):
    """
    从 frame_dir 构建 MapTRv2 推理所需的输入。

    Returns:
        img_tensor: numpy [N_cams, 3, PAD_H, PAD_W]
        img_metas_dict: 单帧 img_metas 字典
        camera_names: list of camera name strings
    """
    meta_path = os.path.join(frame_dir, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    # ── 相机内参（原始 1600×900 尺度，缩放到 MapTRv2 输入） ──────────────────
    cam_intrinsics_raw = meta.get('cam_intrinsics_raw', [])
    lidar2cam_list     = meta.get('lidar2cam_extrinsics', [])
    ego_trans          = meta.get('ego_translation_global', [0.0, 0.0, 0.0])
    ego_yaw            = float(meta.get('ego_yaw_global', 0.0))
    camera_names       = meta.get('camera_names', [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ])
    sample_token       = meta.get('sample_token', 'dummy_token')

    # ── 预处理图像 ────────────────────────────────────────────────────────────
    img_arrays = []
    lidar2img_list = []
    for i in range(N_CAMS):
        # 图像文件名格式: {idx}-{CAM_NAME}.jpg
        img_file = os.path.join(frame_dir, f'{i}-{camera_names[i]}.jpg')
        img_arr = preprocess_image(img_file)    # [3, 192, 320]
        img_arrays.append(img_arr)

        # 计算 lidar2img
        if i < len(cam_intrinsics_raw) and cam_intrinsics_raw[i] is not None:
            K_scaled = build_intrinsic_scaled(cam_intrinsics_raw[i])
        else:
            K_scaled = np.eye(3, dtype=np.float64)
        if i < len(lidar2cam_list) and lidar2cam_list[i] is not None:
            l2i = build_lidar2img(K_scaled, lidar2cam_list[i])
        else:
            l2i = np.eye(4, dtype=np.float64)
        lidar2img_list.append(l2i)

    img_tensor = np.stack(img_arrays, axis=0)  # [N_cams, 3, 192, 320]

    # ── img_metas ────────────────────────────────────────────────────────────
    img_metas_dict = {
        'scene_token':       sample_token[:8] + '_scene',
        'sample_token':      sample_token,
        'can_bus':           build_can_bus(ego_trans, ego_yaw),
        'lidar2img':         [l2i.astype(np.float32) for l2i in lidar2img_list],
        'img_shape':         [(PAD_H, PAD_W, 3)] * N_CAMS,
        'pad_shape':         [(PAD_H, PAD_W, 3)] * N_CAMS,
        'scale_factor':      1.0,
        'flip':              False,
        'pcd_horizontal_flip': False,
        'pcd_vertical_flip':   False,
        'box_mode_3d':       None,
        'box_type_3d':       None,
        'img_norm_cfg':      dict(
            mean=IMG_MEAN.tolist(),
            std=IMG_STD.tolist(),
            to_rgb=True),
        'prev_bev_exists':   False,
        'ori_shape':         [(ORIG_H, ORIG_W, 3)] * N_CAMS,
    }

    return img_tensor, img_metas_dict, camera_names


def parse_model_output(bbox_results, score_thr: float) -> list:
    """
    将 MapTR 模型输出转换为 map_result.json 格式的 elements 列表。

    bbox_results: list[dict]，每个 dict 有 'pts_bbox' 键，
                  其中包含 pts_3d [N,20,2], scores_3d [N], labels_3d [N]
    Returns: list of element dicts
    """
    if not bbox_results:
        return []

    item = bbox_results[0]  # batch_size=1
    # 结果嵌套在 'pts_bbox' 下
    result = item.get('pts_bbox', item)

    pts_3d    = result.get('pts_3d', None)     # [N, pts, 2]
    scores_3d = result.get('scores_3d', None)  # [N]
    labels_3d = result.get('labels_3d', None)  # [N]

    if pts_3d is None or len(pts_3d) == 0:
        return []

    # 转换为 numpy
    if hasattr(pts_3d, 'numpy'):
        pts_3d    = pts_3d.numpy()
        scores_3d = scores_3d.numpy()
        labels_3d = labels_3d.numpy()

    elements = []
    for i in range(len(pts_3d)):
        score = float(scores_3d[i])
        if score < score_thr:
            continue
        label = int(labels_3d[i])
        if label < 0 or label >= len(MAP_CLASSES):
            continue
        cls_name = MAP_CLASSES[label]

        # pts shape [pts, 2], coordinates: x=lateral(right), y=longitudinal(forward)
        pts = pts_3d[i]  # [20, 2] or [20, 3]
        if pts.shape[-1] >= 2:
            pts = pts[:, :2]  # take only x, y

        elements.append({
            'type':       cls_name,
            'subtype':    cls_name,
            'score':      round(score, 4),
            'points':     pts.tolist(),
            'is_polygon': False,
        })
    return elements


def infer_single_frame(model, frame_dir: str, score_thr: float, device) -> list:
    """对单帧运行 MapTRv2 推理，返回 elements 列表。"""
    import torch

    img_tensor, img_metas_dict, _ = build_input_for_frame(frame_dir)

    # [N_cams, 3, H, W] → [1, N_cams, 3, H, W] → torch tensor
    img = torch.from_numpy(img_tensor).unsqueeze(0).to(device)  # [1, 6, 3, 192, 320]

    # img_metas: [[dict]] (batch=1, queue=1)
    # img 必须包装成列表：forward_test 期望 img[0] = [1, N, C, H, W]
    img_metas = [[img_metas_dict]]

    with torch.no_grad():
        results = model.forward(return_loss=False, img=[img], img_metas=img_metas)

    return parse_model_output(results, score_thr)


def load_model(config_path: str, checkpoint_path: str, device: str):
    """
    加载 MapTRv2 模型（需要 mmcv 1.x 环境）。
    """
    import mmcv
    from mmdet3d.models import build_model

    # 加载配置
    cfg = mmcv.Config.fromfile(config_path)

    # 关闭训练时的特殊模块
    cfg.model.pretrained = None
    cfg.model.train_cfg  = None

    # 构建模型
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # 加载权重（mmcv 1.x 方式）
    from mmcv.runner import load_checkpoint
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    model.to(device)
    model.eval()
    return model, cfg


def process_frames_dir(args):
    """批量处理 frames_dir 中的所有 frame_* 子目录。"""
    frames_dir = args.frames_dir
    frame_dirs = sorted([
        os.path.join(frames_dir, d)
        for d in os.listdir(frames_dir)
        if d.startswith('frame_') and
           os.path.isdir(os.path.join(frames_dir, d))
    ])

    if not frame_dirs:
        print(f'[Worker] 未找到 frame_* 子目录: {frames_dir}')
        return 0

    # 确定 checkpoint 和 config 的绝对路径
    config_path = os.path.join(PROJECT_ROOT, args.config) \
        if not os.path.isabs(args.config) else args.config
    ckpt_path = os.path.join(PROJECT_ROOT, args.checkpoint) \
        if not os.path.isabs(args.checkpoint) else args.checkpoint

    if not os.path.exists(ckpt_path):
        print(f'[Worker] 错误：checkpoint 不存在: {ckpt_path}')
        return 0
    if not os.path.exists(config_path):
        print(f'[Worker] 错误：config 不存在: {config_path}')
        return 0

    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'[Worker] 设备: {device}')
    print(f'[Worker] 加载模型: {config_path}')
    print(f'[Worker] Checkpoint: {ckpt_path}')

    try:
        model, cfg = load_model(config_path, ckpt_path, device)
    except Exception as e:
        print(f'[Worker] 模型加载失败: {e}')
        traceback.print_exc()
        return 0

    print(f'[Worker] 开始推理，共 {len(frame_dirs)} 帧, score_thr={args.score_thr}')

    success = 0
    for i, frame_dir in enumerate(frame_dirs):
        out_path = os.path.join(frame_dir, 'map_result.json')
        if not args.overwrite and os.path.exists(out_path):
            success += 1
            continue

        try:
            # 读取 ego 信息（保存到结果中）
            with open(os.path.join(frame_dir, 'meta.json')) as f:
                meta = json.load(f)

            elements = infer_single_frame(model, frame_dir, args.score_thr, device)

            result = {
                'source':          'model',
                'model':           'MapTRv2',
                'location':        meta.get('location', 'unknown'),
                'frame_dir':       os.path.basename(frame_dir),
                'ego_translation': meta.get('ego_translation_global', [0, 0, 0]),
                'ego_yaw':         meta.get('ego_yaw_global', 0.0),
                'patch_size':      60.0,   # pc_range: x±15, y±30
                'elements':        elements,
            }
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)
            success += 1

            if (i + 1) % 10 == 0 or i == 0:
                print(f'[Worker] 进度: {i+1}/{len(frame_dirs)}  '
                      f'元素数: {len(elements)}')
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f'[Worker] 帧 {os.path.basename(frame_dir)} 失败: {e}')
            traceback.print_exc()

    print(f'[Worker] 完成: {success}/{len(frame_dirs)} 帧成功')
    return success


def main():
    parser = argparse.ArgumentParser(
        description='MapTRv2 推理 Worker（在 maptr 环境中运行）')
    parser.add_argument(
        '--frames-dir', required=True,
        help='包含 frame_* 子目录的顶层目录')
    parser.add_argument(
        '--config',
        default='model/maptr/config/maptr_nano_r18_110e.py',
        help='MapTRv2 配置文件路径（相对项目根目录）')
    parser.add_argument(
        '--checkpoint',
        default='model/maptr/maptr_nano_r18_110e.pth',
        help='MapTRv2 权重文件路径（相对项目根目录）')
    parser.add_argument(
        '--score-thr', type=float, default=0.3,
        help='置信度阈值（低于此值的预测被过滤）')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='覆盖已有的 map_result.json 文件')
    args = parser.parse_args()

    # 恢复工作目录（以便相对路径正确解析）
    os.chdir(_old_cwd)

    success_count = process_frames_dir(args)
    sys.exit(0 if success_count > 0 else 1)


if __name__ == '__main__':
    main()
