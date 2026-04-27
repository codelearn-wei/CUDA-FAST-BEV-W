"""
maptr_model_infer.py — MapTRv2 模型推理（subprocess 方式）
============================================================

在 bev conda 环境中运行，通过 subprocess 调用 maptr_infer_worker.py
（该 worker 运行在 maptr conda 环境中，带有 mmcv 1.x / mmdet 2.x 依赖）。

支持三种模式：
  model      — 调用 MapTRv2 神经网络（需要 maptr 环境 + 权重文件）
  gt         — NuScenes Map Expansion GT（需要 expansion JSON）
  trajectory — 自车轨迹推断（始终可用）

用法：
  from tools.maptr.maptr_model_infer import MapInference
  infer = MapInference(mode='model')
  result = infer.extract_frame('outputs/frames/frame_00000', save=True)
"""

import os
import sys
import json
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── 路径常量 ────────────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))

MAPTR_REPO_DIR    = os.path.join(PROJECT_ROOT, 'tools', 'MapTRv2')
MAPTR_WORKER      = os.path.join(_THIS_DIR, 'maptr_infer_worker.py')
MAPTR_SETUP_SH    = os.path.join(_THIS_DIR, 'setup_maptr_env.sh')

DEFAULT_CKPT_REL  = 'model/maptr/maptr_nano_r18_110e.pth'
DEFAULT_CFG_REL   = 'model/maptr/config/maptr_nano_r18_110e.py'
MAPTR_ENV_NAME    = 'maptr'

# 编译后的 .so 文件目录
OPS_DIR = os.path.join(
    MAPTR_REPO_DIR,
    'projects', 'mmdet3d_plugin', 'maptr', 'modules', 'ops',
    'geometric_kernel_attn'
)

# ─── 模型可用性检测 ───────────────────────────────────────────────────────────

def check_model_available(ckpt_path: str = DEFAULT_CKPT_REL) -> tuple:
    """检查 MapTRv2 模型推理是否可用。"""
    # 1. 检查 maptr conda 环境
    conda_exec = shutil.which('conda')
    if conda_exec is None:
        return False, 'conda 未找到'
    try:
        out = subprocess.run(
            ['conda', 'env', 'list'],
            capture_output=True, text=True, timeout=10
        )
        if MAPTR_ENV_NAME not in out.stdout:
            return False, (
                f"conda 环境 '{MAPTR_ENV_NAME}' 未安装\n"
                f"  → 运行: bash {MAPTR_SETUP_SH}"
            )
    except Exception as e:
        return False, f'conda env list 失败: {e}'

    # 2. 检查 checkpoint 文件
    ckpt_abs = os.path.join(PROJECT_ROOT, ckpt_path) \
        if not os.path.isabs(ckpt_path) else ckpt_path
    if not os.path.exists(ckpt_abs):
        return False, f'Checkpoint 不存在: {ckpt_abs}'

    # 3. 检查 MapTRv2 repo
    if not os.path.isdir(MAPTR_REPO_DIR):
        return False, f'MapTRv2 项目目录不存在: {MAPTR_REPO_DIR}'

    # 4. 检查 ops 是否已编译
    if os.path.isdir(OPS_DIR):
        so_files = [f for f in os.listdir(OPS_DIR) if f.endswith('.so')]
    else:
        so_files = []
    if not so_files:
        return False, (
            f'geometric_kernel_attn CUDA 算子未编译（无 .so 文件）\n'
            f'  → 运行: bash {MAPTR_SETUP_SH}'
        )

    return True, 'OK'


# ─── MapTRv2 推理类（subprocess 方式）────────────────────────────────────────

class MapTRv2Infer:
    """通过 subprocess 调用 maptr 环境中的 worker 运行 MapTRv2 推理。"""

    def __init__(self,
                 ckpt_path: str = DEFAULT_CKPT_REL,
                 cfg_path:  str = DEFAULT_CFG_REL,
                 score_thr: float = 0.3):
        self.ckpt_path = ckpt_path
        self.cfg_path  = cfg_path
        self.score_thr = score_thr

        self.available, self._reason = check_model_available(ckpt_path)
        if not self.available:
            logger.info(f'[MapTRv2] 模型不可用: {self._reason}')

    def infer_frames(self, frames_dir: str,
                     overwrite: bool = False,
                     verbose: bool = True) -> int:
        """批量推理，通过 subprocess 调用 worker。返回 0=成功，非0=失败。"""
        if not self.available:
            print(f'[MapTRv2] 不可用: {self._reason}')
            print(f'[MapTRv2] 请运行: bash {MAPTR_SETUP_SH}')
            return 1

        cmd = [
            'conda', 'run', '-n', MAPTR_ENV_NAME,
            '--no-capture-output',
            'python', MAPTR_WORKER,
            '--frames-dir', frames_dir,
            '--config',     self.cfg_path,
            '--checkpoint', self.ckpt_path,
            '--score-thr',  str(self.score_thr),
        ]
        if overwrite:
            cmd.append('--overwrite')

        if verbose:
            print(f'[MapTRv2] 启动推理 worker（maptr 环境）...')

        try:
            ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
            return ret.returncode
        except Exception as e:
            print(f'[MapTRv2] Worker 启动失败: {e}')
            return 1


# ─── 统一接口：MapInference ───────────────────────────────────────────────────

class MapInference:
    """
    统一地图推理接口。

    mode:
      'model'      → MapTRv2 神经网络（需 maptr env + checkpoint）
      'gt'         → NuScenes GT（需 expansion JSON）
      'trajectory' → 自车轨迹推断（始终可用）
      'auto'       → model > gt > trajectory 自动选择
    """

    def __init__(self,
                 mode: str = 'auto',
                 nuscenes_dir: str = 'data/nuscenes',
                 version: str = 'v1.0-mini',
                 ckpt_path: str = DEFAULT_CKPT_REL,
                 cfg_path:  str = DEFAULT_CFG_REL,
                 score_thr: float = 0.3,
                 patch_size: float = 120.0,
                 verbose: bool = True):
        self.mode    = mode
        self.verbose = verbose

        from tools.maptr.nusc_map_extractor import NuScenesMapExtractor
        self._extractor = NuScenesMapExtractor(
            nuscenes_dir=nuscenes_dir,
            version=version,
            patch_size=patch_size,
            verbose=verbose
        )
        self._model_infer = MapTRv2Infer(
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            score_thr=score_thr
        )
        self._effective_mode = self._resolve_mode(mode)
        if verbose:
            _names = {
                'model': 'MapTRv2 神经网络',
                'gt': 'NuScenes GT',
                'trajectory': '轨迹降级',
            }
            print(f'[MapInference] 实际模式: {_names.get(self._effective_mode, self._effective_mode)}')

    def _resolve_mode(self, requested: str) -> str:
        if requested == 'model':
            if self._model_infer.available:
                return 'model'
            print('[MapInference] model 不可用，回退到 gt/trajectory')
            return 'gt'
        if requested == 'auto':
            if self._model_infer.available:
                return 'model'
            return 'gt'
        return requested

    def extract_frame(self, frame_dir: str, save: bool = False) -> Optional[dict]:
        """提取单帧地图（gt/trajectory 模式，或读取 model 已生成的结果）。"""
        if self._effective_mode == 'model':
            result_path = os.path.join(frame_dir, 'map_result.json')
            if os.path.exists(result_path):
                with open(result_path) as f:
                    return json.load(f)
        return self._extractor.extract_frame(frame_dir, save=save)

    def extract_frames(self, frames_dir: str, overwrite: bool = False) -> int:
        """
        批量提取所有帧的地图元素。

        model 模式：通过 subprocess 启动 worker
        gt/trajectory 模式：本地批量提取
        """
        if self._effective_mode == 'model':
            ret = self._model_infer.infer_frames(
                frames_dir, overwrite=overwrite, verbose=self.verbose)
            if ret != 0:
                print('[MapInference] 模型推理失败，回退到 GT 模式')
                return self._extractor.extract_frames(
                    frames_dir, overwrite=overwrite)
            return sum(
                1 for d in Path(frames_dir).iterdir()
                if d.is_dir() and d.name.startswith('frame_')
                and (d / 'map_result.json').exists()
            )
        return self._extractor.extract_frames(frames_dir, overwrite=overwrite)

    @property
    def effective_mode(self) -> str:
        return self._effective_mode

    def print_status(self):
        print(f'[MapInference] 请求模式: {self.mode}')
        print(f'[MapInference] 实际模式: {self._effective_mode}')
        avail, reason = check_model_available(self._model_infer.ckpt_path)
        print(f'[MapInference] MapTRv2 可用: {avail}')
        if not avail:
            print(f'  原因: {reason}')
