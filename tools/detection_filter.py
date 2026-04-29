#!/usr/bin/env python3
"""
目标检测后处理滤波器 (tools/detection_filter.py)

提供 DetectionFilter 类，对 BEV 检测结果进行多维度规则过滤，
适配整个「推理-检测-跟踪-输出」pipeline。

用法示例：
    from tools.detection_filter import DetectionFilter, FilterConfig, build_filter

    # 方式一：完整配置
    cfg = FilterConfig(
        score_thr=0.35,
        class_ids={0, 1, 3, 8},   # car, truck, bus, pedestrian
        bev_nms_dist=0.8,
    )
    filt = DetectionFilter(cfg)
    filtered = filt.filter(raw_detections)

    # 方式二：快捷工厂
    filt = build_filter(score_thr=0.3, classes="0,1,8")
    filtered = filt.filter(raw_detections)

detection dict 格式（来自 result.json）:
    {
      "label": int,                # 类别 ID  0-9
      "label_name": str,           # 类别名称
      "score": float,              # 置信度
      "center_xyz": [x, y, z],    # BEV 中心（lidar 坐标，米）
      "size_xyz": [w, l, h],      # 尺寸（米）
      "yaw": float,               # 偏航角（弧度）
      "velocity_xy": [vx, vy]     # 速度（m/s）
    }
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ─── NuScenes 10 类 ────────────────────────────────────────────────────────────

CLASS_NAMES: List[str] = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# ─── 每类合理尺寸约束 [w_min, w_max, l_min, l_max, h_min, h_max]（米） ────────
# 宽(w) / 长(l) / 高(h)  基于 NuScenes 官方标注统计

DEFAULT_SIZE_LIMITS: Dict[int, Tuple[float, float, float, float, float, float]] = {
    0:  (1.2, 3.0,   3.0,  6.5,  0.8, 2.5),   # car
    1:  (1.5, 4.0,   3.5, 18.0,  1.5, 5.0),   # truck
    2:  (1.5, 6.0,   3.0, 20.0,  1.5, 8.0),   # construction_vehicle
    3:  (2.0, 4.5,   5.0, 25.0,  2.0, 5.5),   # bus
    4:  (2.0, 4.5,   5.0, 30.0,  1.5, 5.0),   # trailer
    5:  (0.2, 2.0,   0.2,  3.0,  0.3, 2.5),   # barrier
    6:  (0.4, 1.5,   1.0,  3.5,  0.8, 2.5),   # motorcycle
    7:  (0.3, 1.2,   1.0,  3.0,  0.5, 2.5),   # bicycle
    8:  (0.0, 1.8,   0.2,  1.2,  0.5, 2.2),   # pedestrian (模型输出 h≈0.7-1.0m, w/l≈0.3-0.55m)
    9:  (0.1, 0.7,   0.1,  0.7,  0.3, 1.2),   # traffic_cone
}

# ─── 每类最大速度（m/s） ──────────────────────────────────────────────────────

DEFAULT_MAX_VELOCITY: Dict[int, float] = {
    0: 55.6,   # car:   ~200 km/h
    1: 38.9,   # truck: ~140 km/h
    2: 16.7,   # construction_vehicle: ~60 km/h
    3: 33.3,   # bus:   ~120 km/h
    4: 27.8,   # trailer: ~100 km/h
    5:  5.0,   # barrier: 近静止
    6: 44.4,   # motorcycle: ~160 km/h
    7: 13.9,   # bicycle: ~50 km/h
    8:  8.3,   # pedestrian: ~30 km/h (sprint)
    9:  2.0,   # traffic_cone: 静止
}


# ─── FilterConfig ──────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    """
    过滤器配置。所有规则均可独立开关，方便 pipeline 集成。

    Attributes:
        score_thr:              置信度下限（低于此值的框被丢弃）
        class_ids:              允许的类别 ID 集合。None = 保留所有类别
        enable_size_filter:     是否启用尺寸范围过滤
        size_limits:            自定义尺寸约束，None = 使用内置默认值
        enable_velocity_filter: 是否过滤速度异常目标
        max_velocity:           自定义速度上限，None = 使用内置默认值
        enable_bev_nms:         是否对 BEV 中心距离过近的同类框做 NMS
        bev_nms_dist:           NMS 距离阈值（米），仅对同类别框生效
        min_z:                  检测框中心 z 坐标下限（滤除地下噪声）
        max_z:                  检测框中心 z 坐标上限（滤除天空噪声）
    """
    # 置信度
    score_thr: float = 0.35

    # 类别白名单
    class_ids: Optional[Set[int]] = None

    # 尺寸过滤
    enable_size_filter: bool = True
    size_limits: Optional[Dict[int, Tuple[float, float, float, float, float, float]]] = None

    # 速度合理性过滤
    enable_velocity_filter: bool = True
    max_velocity: Optional[Dict[int, float]] = None

    # BEV NMS
    enable_bev_nms: bool = True
    bev_nms_dist: float = 0.8

    # z 高度范围
    min_z: float = -5.0
    max_z: float =  5.0


# ─── DetectionFilter ──────────────────────────────────────────────────────────

class DetectionFilter:
    """
    多规则检测结果过滤器。

    设计为无状态（纯函数式），可安全地在多线程环境中复用同一实例。
    每次调用 filter() 独立处理，不保留帧间状态。

    Args:
        cfg: FilterConfig 实例
    """

    def __init__(self, cfg: FilterConfig = None):
        if cfg is None:
            cfg = FilterConfig()
        self.cfg = cfg
        self._size_lim = cfg.size_limits or DEFAULT_SIZE_LIMITS
        self._max_vel  = cfg.max_velocity or DEFAULT_MAX_VELOCITY

    # ── 公开接口 ───────────────────────────────────────────────────────────────

    def filter(self, detections: List[dict]) -> List[dict]:
        """
        对检测结果应用所有启用的过滤规则。

        过滤顺序：
          1. 置信度 (score_thr)
          2. 类别白名单 (class_ids)
          3. z 高度范围 (min_z / max_z)
          4. 尺寸范围 (size_limits)
          5. 速度合理性 (max_velocity)
          6. BEV 中心距离 NMS (bev_nms_dist)

        Args:
            detections: 检测框列表，每项为来自 result.json 的 dict

        Returns:
            过滤后的检测框列表（保持原始 dict 格式，不修改输入）
        """
        passed = [d for d in detections if self._pass_rules(d)]
        if self.cfg.enable_bev_nms and len(passed) > 1:
            passed = self._bev_nms(passed)
        return passed

    def stats(self, before: List[dict], after: List[dict]) -> str:
        """返回过滤统计字符串（调试/日志用）。"""
        n_rem = len(before) - len(after)
        return f"filter: {len(before)} → {len(after)} (-{n_rem})"

    def _center_xyz(self, det: dict) -> Tuple[float, float, float]:
        xyz = det.get("center_xyz")
        if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
            return float(xyz[0]), float(xyz[1]), float(xyz[2])
        return (
            float(det.get("x", 0.0)),
            float(det.get("y", 0.0)),
            float(det.get("z", 0.0)),
        )

    def _size_xyz(self, det: dict) -> Tuple[float, float, float]:
        size = det.get("size_xyz")
        if isinstance(size, (list, tuple)) and len(size) >= 3:
            return float(size[0]), float(size[1]), float(size[2])
        return (
            float(det.get("w", 1.0)),
            float(det.get("l", 1.0)),
            float(det.get("h", 1.0)),
        )

    def _velocity_xy(self, det: dict) -> Tuple[float, float]:
        vel = det.get("velocity_xy")
        if isinstance(vel, (list, tuple)) and len(vel) >= 2:
            return float(vel[0]), float(vel[1])
        return (
            float(det.get("vx", 0.0)),
            float(det.get("vy", 0.0)),
        )

    def _class_id(self, det: dict) -> int:
        if "label" in det:
            return int(det.get("label", -1))
        return int(det.get("class_id", -1))

    # ── 内部规则（每条独立） ───────────────────────────────────────────────────

    def _pass_rules(self, det: dict) -> bool:
        cid   = self._class_id(det)
        score = det.get("score", 0.0)
        xyz   = self._center_xyz(det)
        size  = self._size_xyz(det)
        vel   = self._velocity_xy(det)

        # 1. 置信度
        if score < self.cfg.score_thr:
            return False

        # 2. 类别白名单
        if self.cfg.class_ids is not None and cid not in self.cfg.class_ids:
            return False

        # 3. z 高度范围（滤除地下 / 高空伪目标）
        if not (self.cfg.min_z <= xyz[2] <= self.cfg.max_z):
            return False

        # 4. 尺寸范围
        # if self.cfg.enable_size_filter and cid in self._size_lim:
        #     lim = self._size_lim[cid]
        #     w, l, h = float(size[0]), float(size[1]), float(size[2])
        #     if not (lim[0] <= w <= lim[1]):
        #         return False
        #     if not (lim[2] <= l <= lim[3]):
        #         return False
        #     if not (lim[4] <= h <= lim[5]):
        #         return False

        # 5. 速度合理性
        if self.cfg.enable_velocity_filter:
            vx, vy = float(vel[0]), float(vel[1])
            speed  = math.sqrt(vx * vx + vy * vy)
            max_v  = self._max_vel.get(cid, 60.0)
            if speed > max_v:
                return False

        return True

    def _bev_nms(self, detections: List[dict]) -> List[dict]:
        """
        BEV 中心距离 NMS：
          - 按 score 降序排列
          - 对同类别框：若 BEV 中心距离 < bev_nms_dist，抑制低分框
        """
        dets      = sorted(detections, key=lambda d: d.get("score", 0.0), reverse=True)
        suppressed = [False] * len(dets)
        dist_sq   = self.cfg.bev_nms_dist ** 2

        for i in range(len(dets)):
            if suppressed[i]:
                continue
            xi, yi = self._center_xyz(dets[i])[:2]
            ci = self._class_id(dets[i])
            for j in range(i + 1, len(dets)):
                if suppressed[j]:
                    continue
                if self._class_id(dets[j]) != ci:
                    continue
                xj, yj = self._center_xyz(dets[j])[:2]
                if (xi - xj) ** 2 + (yi - yj) ** 2 < dist_sq:
                    suppressed[j] = True

        return [d for i, d in enumerate(dets) if not suppressed[i]]


# ─── 便捷工厂 ──────────────────────────────────────────────────────────────────

def build_filter(
    score_thr: float = 0.35,
    classes: str = None,
    bev_nms_dist: float = 0.8,
    enable_size_filter: bool = True,
    enable_velocity_filter: bool = True,
    min_z: float = -5.0,
    max_z: float = 5.0,
) -> DetectionFilter:
    """
    快速构建 DetectionFilter。

    Args:
        score_thr:             置信度阈值
        classes:               逗号分隔类别 ID 字符串，如 "0,1,3,8"。None = 全部
        bev_nms_dist:          BEV NMS 距离阈值（米）
        enable_size_filter:    是否启用尺寸过滤
        enable_velocity_filter:是否启用速度过滤
        min_z:                 z 下限（米）
        max_z:                 z 上限（米）

    Returns:
        DetectionFilter 实例
    """
    class_ids: Optional[Set[int]] = None
    if classes:
        class_ids = {int(c.strip()) for c in classes.split(",") if c.strip()}

    return DetectionFilter(FilterConfig(
        score_thr=score_thr,
        class_ids=class_ids,
        bev_nms_dist=bev_nms_dist,
        enable_size_filter=enable_size_filter,
        enable_velocity_filter=enable_velocity_filter,
        min_z=min_z,
        max_z=max_z,
    ))


# ─── 命令行自测 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("用法: python detection_filter.py <result.json> [score_thr] [classes]")
        sys.exit(0)

    with open(sys.argv[1]) as f:
        dets = json.load(f)

    score = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    cls   = sys.argv[3] if len(sys.argv) > 3 else None
    filt  = build_filter(score_thr=score, classes=cls)

    result = filt.filter(dets)
    print(filt.stats(dets, result))
    print(json.dumps(result, indent=2, ensure_ascii=False))
