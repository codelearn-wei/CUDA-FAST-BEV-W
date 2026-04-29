#!/usr/bin/env python3
"""
maptr_server.py — MapTRv2 持久化 GPU 推理服务器
=================================================

在 maptr conda 环境中长驻运行，通过 Unix domain socket 接收 C++ 的推理请求，
使用 GPU 执行 MapTRv2 推理，将结果写入 <frame_dir>/map_result.json。

特性：
  - 模型只加载一次（启动时），避免每帧重复初始化的开销
  - GPU 推理（PyTorch CUDA）
  - 最新帧语义：若 GPU 忙时收到新请求，丢弃旧请求，优先处理最新帧
  - C++ 侧 fire-and-forget（无需等待推理完成）

协议（Unix socket，行格式，UTF-8）：
  C++ → Server:   INFER <frame_dir>\\n       推理请求
  C++ → Server:   QUIT\\n                    关闭服务器
  Server → C++:   READY\\n                   模型已加载，可以开始推理
  Server → C++:   BYE\\n                     确认关闭

用法（由 MapRunnerPersistent 自动调用）：
  conda run -n maptr python tools/maptr/maptr_server.py \\
      --socket /tmp/fastbev_maptr_12345.sock \\
      --config model/maptr/config/maptr_nano_r18_110e.py \\
      --checkpoint model/maptr/maptr_nano_r18_110e.pth \\
      --score-thr 0.3

  如需手动测试：
  echo "INFER outputs/frames/frame_00000" | nc -U /tmp/fastbev_maptr_test.sock
"""

import os
import sys
import json
import math
import socket
import threading
import argparse
import traceback
import time
from pathlib import Path

# ─── 路径设置 ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MAPTR_REPO   = os.path.join(PROJECT_ROOT, 'tools', 'MapTRv2')

GKA_DIR = os.path.join(
    MAPTR_REPO,
    'projects', 'mmdet3d_plugin', 'maptr', 'modules', 'ops',
    'geometric_kernel_attn'
)
for p in [GKA_DIR, MAPTR_REPO, os.path.join(MAPTR_REPO, 'mmdetection3d')]:
    if p not in sys.path:
        sys.path.insert(0, p)

_old_cwd = os.getcwd()
os.chdir(MAPTR_REPO)

import projects.mmdet3d_plugin  # noqa: E402,F401 — 触发 registry 注册

os.chdir(_old_cwd)  # 恢复工作目录，frame_dir 为绝对路径时无影响

# ─── 推理常量 ────────────────────────────────────────────────────────────────
MAP_CLASSES   = ['divider', 'ped_crossing', 'boundary']

import numpy as np
IMG_MEAN      = np.array([123.675, 116.28,  103.53 ], dtype=np.float32)
IMG_STD       = np.array([58.395,  57.12,   57.375 ], dtype=np.float32)
MAPTR_SCALE   = 0.2
ORIG_W, ORIG_H = 1600, 900
TARGET_W = int(ORIG_W * MAPTR_SCALE)  # 320
TARGET_H = int(ORIG_H * MAPTR_SCALE)  # 180
PAD_DIVISOR   = 32
PAD_H = math.ceil(TARGET_H / PAD_DIVISOR) * PAD_DIVISOR  # 192
PAD_W = TARGET_W  # 320
N_CAMS = 6


# ─── 图像预处理（与 maptr_infer_worker.py 保持一致） ──────────────────────────

def _pad_to_divisor(img: np.ndarray, divisor: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = math.ceil(h / divisor) * divisor
    new_w = math.ceil(w / divisor) * divisor
    if new_h == h and new_w == w:
        return img
    padded = np.zeros((new_h, new_w, 3), dtype=img.dtype)
    padded[:h, :w] = img
    return padded


def preprocess_image_gpu(img_path: str) -> np.ndarray:
    """使用 cv2 解码（比 stb_image 快），resize 并归一化。"""
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((3, PAD_H, PAD_W), dtype=np.float32)
    img = img[:, :, ::-1].astype(np.float32)       # BGR→RGB
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    img = _pad_to_divisor(img, PAD_DIVISOR)
    img = (img - IMG_MEAN) / IMG_STD
    return img.transpose(2, 0, 1).astype(np.float32)  # [C,H,W]


def build_lidar2img(K_raw, lidar2cam, scale=MAPTR_SCALE):
    K = np.array(K_raw, dtype=np.float64)
    K[0, :] *= scale
    K[1, :] *= scale
    K_4x4 = np.eye(4, dtype=np.float64)
    K_4x4[:3, :3] = K
    E = np.array(lidar2cam, dtype=np.float64)
    return (K_4x4 @ E).astype(np.float32)


def build_can_bus(ego_trans, ego_yaw):
    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[0:3] = ego_trans[:3]
    can_bus[5]   = math.sin(ego_yaw / 2.0)
    can_bus[6]   = math.cos(ego_yaw / 2.0)
    can_bus[17]  = ego_yaw
    return can_bus


def build_input(frame_dir: str):
    """构建 MapTRv2 推理输入（6 路图像并行预处理）。"""
    import concurrent.futures

    meta_path = os.path.join(frame_dir, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)

    cam_K_raw      = meta.get('cam_intrinsics_raw', [])
    l2c_list       = meta.get('lidar2cam_extrinsics', [])
    ego_trans      = meta.get('ego_translation_global', [0., 0., 0.])
    ego_yaw        = float(meta.get('ego_yaw_global', 0.))
    cam_names      = meta.get('camera_names', [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])
    sample_token   = meta.get('sample_token', 'dummy_token')

    # 并行加载 6 路图像（线程池，CPU 解码）
    img_paths = [os.path.join(frame_dir, f'{i}-{cam_names[i]}.jpg')
                 for i in range(N_CAMS)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CAMS) as ex:
        img_arrays = list(ex.map(preprocess_image_gpu, img_paths))

    img_tensor = np.stack(img_arrays, axis=0)  # [6, 3, 192, 320]

    l2i_list = []
    for i in range(N_CAMS):
        K  = cam_K_raw[i]  if i < len(cam_K_raw)  and cam_K_raw[i]  else None
        E  = l2c_list[i]   if i < len(l2c_list)   and l2c_list[i]   else None
        l2i = build_lidar2img(K, E) if K and E else np.eye(4, dtype=np.float32)
        l2i_list.append(l2i)

    img_metas = {
        'scene_token':         sample_token[:8] + '_scene',
        'sample_token':        sample_token,
        'can_bus':             build_can_bus(ego_trans, ego_yaw),
        'lidar2img':           l2i_list,
        'img_shape':           [(PAD_H, PAD_W, 3)] * N_CAMS,
        'pad_shape':           [(PAD_H, PAD_W, 3)] * N_CAMS,
        'scale_factor':        1.0,
        'flip':                False,
        'pcd_horizontal_flip': False,
        'pcd_vertical_flip':   False,
        'box_mode_3d':         None,
        'box_type_3d':         None,
        'img_norm_cfg':        dict(mean=IMG_MEAN.tolist(), std=IMG_STD.tolist(), to_rgb=True),
        'prev_bev_exists':     False,
        'ori_shape':           [(ORIG_H, ORIG_W, 3)] * N_CAMS,
    }
    return img_tensor, img_metas, meta


def parse_output(bbox_results, score_thr: float) -> list:
    """与 maptr_infer_worker.py 的 parse_model_output 完全一致。"""
    if not bbox_results:
        return []
    item = bbox_results[0]
    result = item.get('pts_bbox', item)
    pts_3d    = result.get('pts_3d',    None)
    scores_3d = result.get('scores_3d', None)
    labels_3d = result.get('labels_3d', None)
    if pts_3d is None or len(pts_3d) == 0:
        return []
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
        pts = pts_3d[i][:, :2].tolist()
        elements.append({
            'type':       MAP_CLASSES[label],
            'subtype':    MAP_CLASSES[label],
            'score':      round(score, 4),
            'points':     pts,
            'is_polygon': False,
        })
    return elements


# ─── 模型加载 ────────────────────────────────────────────────────────────────

def load_model(config_path: str, ckpt_path: str, device: str):
    import mmcv
    from mmdet3d.models import build_model
    from mmcv.runner import load_checkpoint

    os.chdir(MAPTR_REPO)  # mmdet3d 要求在项目根目录
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg  = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt  = load_checkpoint(model, ckpt_path, map_location='cpu')
    if 'CLASSES' in ckpt.get('meta', {}):
        model.CLASSES = ckpt['meta']['CLASSES']
    model.to(device)
    model.eval()
    os.chdir(_old_cwd)
    return model


# ─── 推理函数 ────────────────────────────────────────────────────────────────

def infer_frame(model, frame_dir: str, score_thr: float, device: str) -> bool:
    """对单帧执行 GPU 推理，写入 map_result.json。返回是否成功。"""
    import torch

    try:
        t0 = time.time()
        img_tensor, img_metas, meta = build_input(frame_dir)

        img  = torch.from_numpy(img_tensor).unsqueeze(0).to(device)
        metas = [[img_metas]]

        with torch.no_grad():
            results = model.forward(return_loss=False, img=[img], img_metas=metas)

        elements = parse_output(results, score_thr)
        t1 = time.time()

        result = {
            'source':          'model',
            'model':           'MapTRv2',
            'location':        meta.get('location', 'unknown'),
            'frame_dir':       os.path.basename(frame_dir),
            'ego_translation': meta.get('ego_translation_global', [0, 0, 0]),
            'ego_yaw':         meta.get('ego_yaw_global', 0.0),
            'patch_size':      60.0,
            'elements':        elements,
            'infer_ms':        round((t1 - t0) * 1000, 1),
        }
        out_path = os.path.join(frame_dir, 'map_result.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f'[MapTRServer] {os.path.basename(frame_dir)}: '
              f'{len(elements)} elements, {result["infer_ms"]:.1f}ms', flush=True)
        return True

    except Exception as e:
        print(f'[MapTRServer] 推理失败 {frame_dir}: {e}', flush=True)
        traceback.print_exc()
        return False


# ─── 服务器主类 ──────────────────────────────────────────────────────────────

class MapTRServer:
    """
    持久化 MapTRv2 GPU 推理服务器。

    架构（两线程）：
      net_thread:    接收 C++ 请求，更新 pending_dir（最新帧语义）
      worker_thread: 当 pending_dir 变化时执行 GPU 推理

    最新帧语义：若 GPU 正在处理帧 N，收到帧 N+2 请求时，
    pending_dir 直接更新为 N+2（丢弃 N+1），保证处理最新帧。
    """

    def __init__(self, model, score_thr: float, device: str, conn, verbose: bool):
        self.model      = model
        self.score_thr  = score_thr
        self.device     = device
        self.conn       = conn
        self.verbose    = verbose

        # 最新帧缓冲（None = 无待处理请求）
        self.pending_dir  = None
        self.pending_lock = threading.Lock()
        self.pending_ev   = threading.Event()

        # 关闭标志
        self.stop_flag = threading.Event()

    def _worker(self):
        """GPU 推理工作线程：始终处理最新请求。"""
        while not self.stop_flag.is_set():
            # 等待新请求（最多 0.5s 超时，以便检查 stop_flag）
            if not self.pending_ev.wait(timeout=0.5):
                continue
            self.pending_ev.clear()

            with self.pending_lock:
                frame_dir = self.pending_dir
                self.pending_dir = None

            if frame_dir:
                infer_frame(self.model, frame_dir, self.score_thr, self.device)

    def _recv_line(self) -> str:
        """从 socket 读取一行文本（阻塞）。"""
        buf = b''
        while True:
            ch = self.conn.recv(1)
            if not ch:
                return ''
            if ch == b'\n':
                return buf.decode('utf-8', errors='ignore').strip()
            buf += ch

    def run(self):
        """主服务循环（在 net_thread 中运行）。"""
        # 启动 GPU 工作线程
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()

        # 通知 C++ 服务已就绪
        try:
            self.conn.sendall(b'READY\n')
        except Exception:
            pass

        print('[MapTRServer] 就绪，等待推理请求...', flush=True)

        while True:
            try:
                line = self._recv_line()
            except Exception:
                break

            if not line:
                break

            if line.startswith('INFER '):
                frame_dir = line[6:].strip()
                if not os.path.isdir(frame_dir):
                    print(f'[MapTRServer] 警告：帧目录不存在: {frame_dir}', flush=True)
                    continue
                # 更新最新帧（最新帧语义：丢弃旧请求）
                with self.pending_lock:
                    old = self.pending_dir
                    self.pending_dir = frame_dir
                if old and old != frame_dir and self.verbose:
                    print(f'[MapTRServer] 丢弃旧请求: {os.path.basename(old)}',
                          flush=True)
                self.pending_ev.set()

            elif line == 'QUIT':
                print('[MapTRServer] 收到 QUIT，关闭...', flush=True)
                try:
                    self.conn.sendall(b'BYE\n')
                except Exception:
                    pass
                break

        self.stop_flag.set()
        worker.join(timeout=10)
        print('[MapTRServer] 工作线程已退出', flush=True)


# ─── 入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MapTRv2 持久化 GPU 推理服务器')
    parser.add_argument('--socket',     required=True,
                        help='Unix domain socket 路径（如 /tmp/fastbev_maptr_12345.sock）')
    parser.add_argument('--config',     required=True,
                        help='MapTRv2 config 文件路径')
    parser.add_argument('--checkpoint', required=True,
                        help='MapTRv2 checkpoint 文件路径（.pth）')
    parser.add_argument('--score-thr',  type=float, default=0.3,
                        help='置信度阈值（默认 0.3）')
    parser.add_argument('--device',     default='cuda:0',
                        help='推理设备（默认 cuda:0）')
    parser.add_argument('--verbose',    action='store_true',
                        help='打印详细日志（帧丢弃提示等）')
    args = parser.parse_args()

    # 绝对路径处理
    config_path = os.path.join(PROJECT_ROOT, args.config) \
        if not os.path.isabs(args.config) else args.config
    ckpt_path   = os.path.join(PROJECT_ROOT, args.checkpoint) \
        if not os.path.isabs(args.checkpoint) else args.checkpoint

    if not os.path.exists(config_path):
        print(f'[MapTRServer] 错误：config 不存在: {config_path}', flush=True)
        sys.exit(1)
    if not os.path.exists(ckpt_path):
        print(f'[MapTRServer] 错误：checkpoint 不存在: {ckpt_path}', flush=True)
        sys.exit(1)

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    import torch
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'[MapTRServer] 加载模型（device={device}）...', flush=True)
    t0 = time.time()
    model = load_model(config_path, ckpt_path, device)
    print(f'[MapTRServer] 模型加载完成，耗时 {(time.time()-t0):.1f}s', flush=True)

    # ── 启动 Unix socket 服务器 ───────────────────────────────────────────────
    sock_path = args.socket
    if os.path.exists(sock_path):
        os.remove(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(sock_path)
    server.listen(1)
    print(f'[MapTRServer] 监听 {sock_path}，等待 C++ 连接...', flush=True)

    try:
        conn, _ = server.accept()
        print('[MapTRServer] C++ 客户端已连接', flush=True)

        srv = MapTRServer(model, args.score_thr, device, conn, args.verbose)
        srv.run()

    finally:
        try:
            server.close()
        except Exception:
            pass
        if os.path.exists(sock_path):
            os.remove(sock_path)
        print('[MapTRServer] 已退出', flush=True)


if __name__ == '__main__':
    main()
