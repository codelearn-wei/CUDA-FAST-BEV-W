"""
nusc_map_extractor.py — NuScenes 地图提取（双模式：GT + 轨迹降级）
==================================================================

优先使用 NuScenes Map Expansion API（需要 maps/expansion/*.json 文件）；
若文件缺失，自动降级为"自车轨迹推断"模式，依然输出合理的道路信息。

降级模式说明：
  - 收集 frames_dir 中所有帧的 ego_translation_global
  - 在当前帧的自车坐标系中，将轨迹重投影为道路中心线
  - 向两侧偏移生成道路边界线
  - 结果标记 source="trajectory" 而非 "gt"

坐标约定（与 video_demo.py 一致）：
  - 自车坐标系：x = 右侧（right），y = 前方（forward）
  - 单位：米（m）

输出文件：<frame_dir>/map_result.json
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── 地图 expansion JSON 搜索路径 ─────────────────────────────────────────
def _find_expansion_dir(dataroot: str) -> Optional[str]:
    """搜索 NuScenes map expansion JSON 目录，返回路径或 None。"""
    # 优先在数据集目录中查找
    candidate = os.path.join(dataroot, 'maps', 'expansion')
    if os.path.isdir(candidate):
        jsons = [f for f in os.listdir(candidate) if f.endswith('.json')]
        if jsons:
            return candidate
    # 在 nuscenes pip 包目录中查找（部分安装可能包含 sample 数据）
    try:
        import nuscenes as _nuscenes_pkg
        pkg_dir = os.path.dirname(_nuscenes_pkg.__file__)
        for sub in ['map_expansion', 'tests/data/maps/expansion',
                    'tests/mini_data/maps/expansion']:
            p = os.path.join(pkg_dir, sub)
            if os.path.isdir(p):
                jsons = [f for f in os.listdir(p) if f.endswith('.json')]
                if jsons:
                    return p
    except Exception:
        pass
    return None

# ─── 地图元素类型映射 ────────────────────────────────────────────────────────
# NuScenes 图层  →  MapTRv2 语义类别
LAYER_TYPE_MAP = {
    "lane_divider":  "divider",
    "road_divider":  "divider",
    "ped_crossing":  "ped_crossing",
    "stop_line":     "divider",
    "road_segment":  "boundary",
    "walkway":       "boundary",
}

# 线元素（LineString）图层
LINE_LAYERS = {"lane_divider", "road_divider", "stop_line"}

# 面元素（Polygon）图层 —— 取外轮廓线
POLYGON_LAYERS = {"ped_crossing", "road_segment", "walkway"}

# 所有需要查询的图层
ALL_LAYERS = list(LAYER_TYPE_MAP.keys())

# BEV 感知范围（米），与 video_demo.py 中 BEV_BOUNDS 对齐
DEFAULT_PATCH_SIZE = 120.0   # 查询范围（正方形边长，全局坐标）

# 每种元素最大点数（降采样，避免折线过密）
MAX_POINTS_PER_ELEMENT = 20


# ─── 坐标变换工具 ────────────────────────────────────────────────────────────

def global_to_ego(pts_global: np.ndarray,
                  ego_tx: float, ego_ty: float,
                  cos_yaw: float, sin_yaw: float) -> np.ndarray:
    """将 NuScenes 全局坐标点变换到自车局部坐标系（x=right, y=forward）。

    NuScenes 全局坐标系（新加坡/波士顿城市地图）：
      x ≈ east，y ≈ north

    自车坐标系（与 model local 一致）：
      x = right（车辆右侧）
      y = forward（车辆前方）

    变换公式：
      ego_right   = (px - tx) * sin(yaw) - (py - ty) * cos(yaw)
      ego_forward = (px - tx) * cos(yaw) + (py - ty) * sin(yaw)

    Args:
        pts_global: [N, 2] ndarray，全局坐标 (x, y)
        ego_tx, ego_ty: 自车全局位置
        cos_yaw, sin_yaw: ego_yaw_global 的余弦 / 正弦

    Returns:
        [N, 2] ndarray，自车坐标 (right, forward)
    """
    dx = pts_global[:, 0] - ego_tx
    dy = pts_global[:, 1] - ego_ty
    right   = dx * sin_yaw - dy * cos_yaw
    forward = dx * cos_yaw + dy * sin_yaw
    return np.stack([right, forward], axis=1).astype(np.float32)


def resample_line(pts: np.ndarray, max_pts: int) -> np.ndarray:
    """等距重采样折线，使点数不超过 max_pts。"""
    if len(pts) <= max_pts:
        return pts
    idx = np.round(np.linspace(0, len(pts) - 1, max_pts)).astype(int)
    return pts[idx]


# ─── NuScenesMapExtractor ────────────────────────────────────────────────────

class NuScenesMapExtractor:
    """
    从 NuScenes v1.0 数据集提取 HD 地图元素。

    自动降级策略（按优先级）：
      1. NuScenesMap expansion JSON（maps/expansion/*.json）  → source="gt"
      2. 自车轨迹推断（frame_*/meta.json ego_translation_global）→ source="trajectory"

    地图元素格式（map_result.json）：
        {
          "source": "gt" | "trajectory",
          "location": str,
          "ego_translation": [x, y, z],
          "ego_yaw": float,
          "patch_size": float,
          "elements": [
            {
              "type": "divider" | "ped_crossing" | "boundary",
              "subtype": str,
              "score": 1.0,
              "points": [[x1, y1], ...],   // 自车坐标系（x=right, y=forward）
              "is_polygon": bool
            }, ...
          ]
        }
    """

    def __init__(self,
                 nuscenes_dir: str,
                 version: str = "v1.0-mini",
                 patch_size: float = DEFAULT_PATCH_SIZE,
                 verbose: bool = False):
        self.nuscenes_dir = nuscenes_dir
        self.version = version
        self.patch_size = patch_size
        self.verbose = verbose

        self._nusc = None           # NuScenes API（懒加载）
        self._nusc_maps: Dict = {}  # map_name → NuScenesMap（懒加载）
        self._sample_to_location: Dict[str, str] = {}

        # 检测 expansion JSON 可用性（只检测一次）
        self._expansion_dir: Optional[str] = _find_expansion_dir(nuscenes_dir)
        self._expansion_available: Optional[bool] = None  # None = 未测试

        # 轨迹数据缓存（lazy）
        self._traj_builder: Optional["TrajectoryMapBuilder"] = None

    # ── 懒加载 NuScenes API ──────────────────────────────────────────────────

    def _get_nusc(self):
        if self._nusc is None:
            from nuscenes.nuscenes import NuScenes
            self._nusc = NuScenes(
                version=self.version,
                dataroot=self.nuscenes_dir,
                verbose=self.verbose
            )
            for scene in self._nusc.scene:
                log = self._nusc.get('log', scene['log_token'])
                loc = log['location']
                tok = scene['first_sample_token']
                while tok:
                    self._sample_to_location[tok] = loc
                    sample = self._nusc.get('sample', tok)
                    tok = sample['next']
        return self._nusc

    def _get_map(self, location: str):
        """加载 NuScenesMap，若 expansion JSON 不存在则抛出 FileNotFoundError。"""
        if location not in self._nusc_maps:
            from nuscenes.map_expansion.map_api import NuScenesMap
            # 若找到了 expansion 目录但不在默认位置，可能需要 symlink
            # （NuScenesMap 内部固定从 <dataroot>/maps/expansion/ 读取）
            self._nusc_maps[location] = NuScenesMap(
                dataroot=self.nuscenes_dir,
                map_name=location
            )
        return self._nusc_maps[location]

    def _check_expansion(self, location: str = "singapore-onenorth") -> bool:
        """快速测试 expansion JSON 是否真实可用（只测试一次）。"""
        if self._expansion_available is not None:
            return self._expansion_available
        try:
            self._get_nusc()   # 初始化 NuScenes
            self._get_map(location)
            self._expansion_available = True
            print("[MapExtractor] 模式: NuScenes Map Expansion GT ✓")
        except FileNotFoundError as e:
            self._expansion_available = False
            print(f"[MapExtractor] map expansion JSON 未找到 → 启用轨迹降级模式")
            if self.verbose:
                print(f"  原因: {e}")
        except Exception as e:
            self._expansion_available = False
            print(f"[MapExtractor] NuScenesMap 初始化失败: {e} → 启用轨迹降级模式")
        return self._expansion_available

    # ── 核心提取逻辑 ─────────────────────────────────────────────────────────

    def extract_frame(self, frame_dir: str,
                      save: bool = False) -> Optional[dict]:
        """
        提取单帧的地图元素。自动选择 GT 或轨迹降级模式。

        Args:
            frame_dir: 帧目录路径（包含 meta.json）
            save: 是否自动保存为 map_result.json

        Returns:
            dict: 地图结果（若失败返回 None）
        """
        meta_path = os.path.join(frame_dir, 'meta.json')
        if not os.path.exists(meta_path):
            logger.warning(f"meta.json 不存在: {meta_path}")
            return None

        with open(meta_path) as f:
            meta = json.load(f)

        sample_token = meta.get('sample_token', '')
        ego_trans    = meta.get('ego_translation_global', [0.0, 0.0, 0.0])
        ego_yaw      = float(meta.get('ego_yaw_global', 0.0))

        # 初次调用时确定 location（用于测试 expansion 可用性）
        location = None
        try:
            nusc = self._get_nusc()
            location = self._sample_to_location.get(sample_token, 'singapore-onenorth')
        except Exception:
            location = 'singapore-onenorth'

        # ─── 模式 1：NuScenesMap expansion ──────────────────────────────────
        if self._check_expansion(location):
            return self._extract_gt(frame_dir, meta, sample_token,
                                    ego_trans, ego_yaw, location, save)

        # ─── 模式 2：轨迹降级 ────────────────────────────────────────────────
        return self._extract_trajectory(frame_dir, meta, ego_trans, ego_yaw,
                                        location, save)

    # ── GT 提取（expansion JSON）────────────────────────────────────────────

    def _extract_gt(self, frame_dir, meta, sample_token,
                    ego_trans, ego_yaw, location, save) -> Optional[dict]:
        """使用 NuScenesMap API 提取 GT 地图元素。"""
        try:
            nusc_map = self._get_map(location)
        except Exception as e:
            logger.error(f"加载地图 {location} 失败: {e}")
            return None

        ego_x, ego_y = float(ego_trans[0]), float(ego_trans[1])
        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)

        elements = self._extract_elements_gt(nusc_map, ego_x, ego_y, cos_y, sin_y)

        result = {
            "source":          "gt",
            "location":        location,
            "frame_dir":       os.path.basename(frame_dir),
            "ego_translation": ego_trans,
            "ego_yaw":         ego_yaw,
            "patch_size":      self.patch_size,
            "elements":        elements,
        }
        if save:
            self._save_result(result, frame_dir)
        return result

    def _extract_elements_gt(self, nusc_map, ego_x, ego_y,
                              cos_y, sin_y) -> List[dict]:
        """查询 nusc_map patch 内所有地图元素，变换到自车坐标系。"""
        try:
            from shapely.geometry import (LineString, MultiLineString,
                                           Polygon, MultiPolygon)
        except ImportError:
            logger.error("需要安装 shapely: pip install shapely")
            return []

        half = self.patch_size / 2.0
        patch_box = (ego_x, ego_y, self.patch_size, self.patch_size)

        try:
            geom_dict = nusc_map.get_map_geom(patch_box, 0.0, ALL_LAYERS)
        except Exception as e:
            logger.error(f"get_map_geom 失败: {e}")
            return []

        elements = []
        for layer_name, geoms in geom_dict:
            elem_type  = LAYER_TYPE_MAP.get(layer_name, layer_name)
            is_polygon = layer_name in POLYGON_LAYERS

            for geom in geoms:
                if geom is None or geom.is_empty:
                    continue
                coord_segs = _extract_coords(geom)
                for pts_patch in coord_segs:
                    if len(pts_patch) < 2:
                        continue
                    # get_map_geom 返回的坐标已经是 patch-local（已减去 ego 位置）
                    # 只需旋转到自车坐标系（x=right, y=forward），不需再减 ego 位置
                    dx = pts_patch[:, 0]
                    dy = pts_patch[:, 1]
                    right   = dx * sin_y - dy * cos_y
                    forward = dx * cos_y + dy * sin_y
                    pts_ego = np.stack([right, forward], axis=1).astype(np.float32)
                    in_range = (
                        (np.abs(pts_ego[:, 0]) < half) &
                        (np.abs(pts_ego[:, 1]) < half)
                    )
                    if not in_range.any():
                        continue
                    pts_ego = resample_line(pts_ego, MAX_POINTS_PER_ELEMENT)
                    elements.append({
                        "type":       elem_type,
                        "subtype":    layer_name,
                        "score":      1.0,
                        "points":     pts_ego.tolist(),
                        "is_polygon": is_polygon,
                    })
        return elements

    # ── 轨迹降级提取 ─────────────────────────────────────────────────────────

    def _extract_trajectory(self, frame_dir, meta, ego_trans,
                             ego_yaw, location, save) -> Optional[dict]:
        """使用自车轨迹推断道路元素（不需要 expansion JSON）。"""
        # 初始化轨迹构建器（需要知道 frames 目录）
        frames_dir = str(Path(frame_dir).parent)
        if self._traj_builder is None or \
                self._traj_builder.frames_dir != frames_dir:
            self._traj_builder = TrajectoryMapBuilder(frames_dir, self.verbose)

        elements = self._traj_builder.build(ego_trans, ego_yaw, self.patch_size)

        result = {
            "source":          "trajectory",
            "location":        location or "unknown",
            "frame_dir":       os.path.basename(frame_dir),
            "ego_translation": ego_trans,
            "ego_yaw":         ego_yaw,
            "patch_size":      self.patch_size,
            "elements":        elements,
        }
        if save:
            self._save_result(result, frame_dir)
        return result

    # ── 批量提取 ─────────────────────────────────────────────────────────────

    def extract_frames(self, frames_dir: str,
                       overwrite: bool = False) -> int:
        """
        批量提取目录下所有 frame_* 子目录的地图元素。

        Returns:
            成功处理的帧数
        """
        frames_dir_path = Path(frames_dir)
        frame_dirs = sorted(frames_dir_path.glob("frame_*"))

        if not frame_dirs:
            print(f"[MapExtractor] 未找到帧目录: {frames_dir}")
            return 0

        # 预加载 NuScenes（确定模式）
        try:
            self._get_nusc()
        except Exception:
            pass

        # 确定模式（用第一个帧的位置测试）
        first_meta = frame_dirs[0] / 'meta.json'
        test_location = 'singapore-onenorth'
        if first_meta.exists():
            with open(first_meta) as f:
                m = json.load(f)
            tok = m.get('sample_token', '')
            test_location = self._sample_to_location.get(tok, test_location)
        self._check_expansion(test_location)

        # 若使用轨迹模式，预加载全部轨迹
        if not self._expansion_available:
            self._traj_builder = TrajectoryMapBuilder(str(frames_dir_path),
                                                      self.verbose)
            print(f"[MapExtractor] 轨迹降级模式："
                  f"已加载 {len(self._traj_builder.all_ego_global)} 个位置点")

        success = 0
        for frame_dir in frame_dirs:
            out_path = frame_dir / 'map_result.json'
            if out_path.exists() and not overwrite:
                success += 1
                continue
            result = self.extract_frame(str(frame_dir), save=True)
            if result is not None:
                success += 1
                if self.verbose:
                    src = result.get("source", "?")
                    n = len(result.get("elements", []))
                    print(f"  [{frame_dir.name}] source={src} "
                          f"elements={n}")

        total = len(frame_dirs)
        mode = "GT" if self._expansion_available else "轨迹降级"
        print(f"[MapExtractor] 完成({mode}): {success}/{total} 帧")
        return success

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    def _save_result(self, result: dict, frame_dir: str):
        out_path = os.path.join(frame_dir, 'map_result.json')
        with open(out_path, 'w') as f:
            json.dump(result, f)

    @staticmethod
    def save(result: dict, path: str):
        with open(path, 'w') as f:
            json.dump(result, f)

    @staticmethod
    def load(path: str) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None


# ─── TrajectoryMapBuilder ─────────────────────────────────────────────────────

# 道路参数（单位：米）
ROAD_HALF_WIDTH  = 3.75   # 单车道半宽（1 个标准车道宽度）
ROAD_DIVIDER_OFFSET = 0.0 # 中心线
LEFT_BOUNDARY_OFFSET  = -ROAD_HALF_WIDTH * 2   # 左侧道路边界
RIGHT_BOUNDARY_OFFSET =  ROAD_HALF_WIDTH * 2   # 右侧道路边界
LEFT_DIVIDER_OFFSET   = -ROAD_HALF_WIDTH       # 左侧车道线
RIGHT_DIVIDER_OFFSET  =  ROAD_HALF_WIDTH       # 右侧车道线

TRAJ_SMOOTH_N = 5          # 轨迹滑动平均窗口
TRAJ_RESAMPLE_N = 30       # 输出折线点数


class TrajectoryMapBuilder:
    """
    从 ego 轨迹推断道路元素（无需 HD 地图文件）。

    在自车坐标系中生成：
      - boundary: 左右道路边界（±ROAD_HALF_WIDTH*2 偏移）
      - divider:  车道分隔线（±ROAD_HALF_WIDTH 偏移）
    """

    def __init__(self, frames_dir: str, verbose: bool = False):
        self.frames_dir = frames_dir
        self.verbose = verbose
        self.all_ego_global: List[Tuple[float, float]] = []  # [(x,y), ...]
        self._load_trajectory()

    def _load_trajectory(self):
        """收集 frames_dir 中所有帧的自车全局位置，按 frame_idx 排序。"""
        frames_dir = Path(self.frames_dir)
        entries = []
        for fd in sorted(frames_dir.glob("frame_*")):
            meta_path = fd / 'meta.json'
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    m = json.load(f)
                trans = m.get('ego_translation_global', [0, 0, 0])
                idx   = m.get('frame_idx', 0)
                entries.append((idx, float(trans[0]), float(trans[1])))
            except Exception:
                pass
        entries.sort(key=lambda e: e[0])
        self.all_ego_global = [(e[1], e[2]) for e in entries]
        if self.verbose:
            print(f"[TrajectoryMap] 已加载 {len(self.all_ego_global)} 个轨迹点")

    def build(self, ego_trans: list, ego_yaw: float,
              patch_size: float) -> List[dict]:
        """
        为当前帧构建轨迹地图元素列表。

        Args:
            ego_trans:  当前帧自车全局位置 [x, y, z]
            ego_yaw:    当前帧自车航向角（弧度）
            patch_size: 感知范围（正方形边长，米）

        Returns:
            地图元素列表（自车坐标系，x=right, y=forward）
        """
        if len(self.all_ego_global) < 2:
            return []

        ego_x = float(ego_trans[0])
        ego_y = float(ego_trans[1])
        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)
        half  = patch_size / 2.0

        # 1. 将全部轨迹点变换到当前帧自车坐标系
        pts_global = np.array(self.all_ego_global, dtype=np.float32)
        pts_ego = global_to_ego(pts_global, ego_x, ego_y, cos_y, sin_y)

        # 2. 按 forward(y) 排序，过滤到感知范围内
        sort_idx = np.argsort(pts_ego[:, 1])
        pts_ego  = pts_ego[sort_idx]
        mask = (
            (np.abs(pts_ego[:, 0]) < half) &
            (np.abs(pts_ego[:, 1]) < half)
        )
        pts_filt = pts_ego[mask]

        if len(pts_filt) < 2:
            # 即使过滤后点太少，也生成一条沿 forward 方向的直线作为中心线
            return self._fallback_straight_lines(patch_size)

        # 3. 滑动平均平滑
        pts_filt = _smooth_line(pts_filt, window=TRAJ_SMOOTH_N)

        # 4. 去除重复点（距离过近的）
        pts_filt = _deduplicate(pts_filt, min_dist=0.5)

        if len(pts_filt) < 2:
            return self._fallback_straight_lines(patch_size)

        # 5. 生成各条平行线（偏移）
        elements = []

        # 道路边界（左 / 右）
        for offset, subtype in [
            (LEFT_BOUNDARY_OFFSET,  "road_boundary_left"),
            (RIGHT_BOUNDARY_OFFSET, "road_boundary_right"),
        ]:
            pts_offset = _offset_polyline(pts_filt, offset)
            if pts_offset is not None and len(pts_offset) >= 2:
                pts_r = resample_line(pts_offset, TRAJ_RESAMPLE_N)
                elements.append({
                    "type":       "boundary",
                    "subtype":    subtype,
                    "score":      0.8,
                    "points":     pts_r.tolist(),
                    "is_polygon": False,
                })

        # 车道分隔线（左 / 右）
        for offset, subtype in [
            (LEFT_DIVIDER_OFFSET,  "lane_divider_left"),
            (RIGHT_DIVIDER_OFFSET, "lane_divider_right"),
        ]:
            pts_offset = _offset_polyline(pts_filt, offset)
            if pts_offset is not None and len(pts_offset) >= 2:
                pts_r = resample_line(pts_offset, TRAJ_RESAMPLE_N)
                elements.append({
                    "type":       "divider",
                    "subtype":    subtype,
                    "score":      0.8,
                    "points":     pts_r.tolist(),
                    "is_polygon": False,
                })

        # 中心线（轨迹本身）
        pts_center = resample_line(pts_filt, TRAJ_RESAMPLE_N)
        elements.append({
            "type":       "divider",
            "subtype":    "lane_center",
            "score":      0.9,
            "points":     pts_center.tolist(),
            "is_polygon": False,
        })

        return elements

    def _fallback_straight_lines(self, patch_size: float) -> List[dict]:
        """当轨迹点过少时，生成沿 forward 方向的直线作为道路元素。"""
        y_near = -patch_size / 2
        y_far  =  patch_size / 2
        elements = []
        for offset, etype, subtype in [
            (LEFT_BOUNDARY_OFFSET,  "boundary", "road_boundary_left"),
            (LEFT_DIVIDER_OFFSET,   "divider",  "lane_divider_left"),
            (0.0,                   "divider",  "lane_center"),
            (RIGHT_DIVIDER_OFFSET,  "divider",  "lane_divider_right"),
            (RIGHT_BOUNDARY_OFFSET, "boundary", "road_boundary_right"),
        ]:
            elements.append({
                "type":       etype,
                "subtype":    subtype,
                "score":      0.5,
                "points":     [[offset, y_near], [offset, y_far]],
                "is_polygon": False,
            })
        return elements


# ─── 折线处理工具函数 ──────────────────────────────────────────────────────────

def _smooth_line(pts: np.ndarray, window: int = 5) -> np.ndarray:
    """滑动平均平滑折线（边缘用 clamp）。"""
    if len(pts) <= window:
        return pts
    result = pts.copy().astype(np.float32)
    hw = window // 2
    for i in range(len(pts)):
        lo = max(0, i - hw)
        hi = min(len(pts), i + hw + 1)
        result[i] = pts[lo:hi].mean(axis=0)
    return result


def _deduplicate(pts: np.ndarray, min_dist: float = 0.5) -> np.ndarray:
    """去除距离过近的重复点。"""
    if len(pts) == 0:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        dist = np.linalg.norm(pts[i] - pts[keep[-1]])
        if dist >= min_dist:
            keep.append(i)
    return pts[keep]


def _offset_polyline(pts: np.ndarray, offset: float) -> Optional[np.ndarray]:
    """
    将折线向法线方向偏移 offset 米（正 = 右侧，负 = 左侧）。

    使用每段的平均法向量来平滑偏移方向。
    """
    if len(pts) < 2:
        return None
    n = len(pts)
    normals = np.zeros((n, 2), dtype=np.float32)

    # 计算每点处的法向量（平均相邻两段）
    for i in range(n):
        tangents = []
        if i > 0:
            t = pts[i] - pts[i - 1]
            l = np.linalg.norm(t)
            if l > 1e-6:
                tangents.append(t / l)
        if i < n - 1:
            t = pts[i + 1] - pts[i]
            l = np.linalg.norm(t)
            if l > 1e-6:
                tangents.append(t / l)
        if not tangents:
            normals[i] = np.array([1.0, 0.0])
            continue
        avg_t = np.mean(tangents, axis=0)
        l = np.linalg.norm(avg_t)
        if l < 1e-6:
            normals[i] = np.array([1.0, 0.0])
        else:
            avg_t /= l
            # 右法向量（向右偏移为正）
            normals[i] = np.array([avg_t[1], -avg_t[0]])

    result = pts + normals * offset
    return result


def _extract_coords(geom) -> List[np.ndarray]:
    """从 shapely Geometry 提取坐标点列表（用于 GT 模式）。"""
    segs = []
    gtype = geom.geom_type
    if gtype == 'LineString':
        segs.append(np.array(geom.coords)[:, :2])
    elif gtype == 'MultiLineString':
        for part in geom.geoms:
            segs.append(np.array(part.coords)[:, :2])
    elif gtype == 'Polygon':
        segs.append(np.array(geom.exterior.coords)[:, :2])
        for interior in geom.interiors:
            segs.append(np.array(interior.coords)[:, :2])
    elif gtype == 'MultiPolygon':
        for poly in geom.geoms:
            segs.append(np.array(poly.exterior.coords)[:, :2])
    elif gtype == 'GeometryCollection':
        for part in geom.geoms:
            segs.extend(_extract_coords(part))
    return segs


# ─── 便捷函数 ────────────────────────────────────────────────────────────────

def extract_map_for_frames(frames_dir: str,
                            nuscenes_dir: str = "data/nuscenes",
                            version: str = "v1.0-mini",
                            overwrite: bool = False,
                            patch_size: float = DEFAULT_PATCH_SIZE,
                            verbose: bool = True) -> int:
    """
    批量提取所有帧的地图元素（自动选择 GT / 轨迹降级模式）。

    Args:
        frames_dir:    帧目录的父目录（含 frame_* 子目录）
        nuscenes_dir:  NuScenes 数据集根目录
        version:       数据集版本（如 'v1.0-mini'）
        overwrite:     是否覆盖已有结果
        patch_size:    查询范围（米）
        verbose:       是否打印进度

    Returns:
        成功帧数
    """
    extractor = NuScenesMapExtractor(
        nuscenes_dir=nuscenes_dir,
        version=version,
        patch_size=patch_size,
        verbose=verbose,
    )
    return extractor.extract_frames(frames_dir, overwrite=overwrite)
