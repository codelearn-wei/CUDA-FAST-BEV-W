"""
MapTRv2 地图感知模块
====================

功能：
  - NuScenes GT 地图提取（始终可用）
  - MapTRv2 模型推理（需要下载 checkpoint + 安装 mmdet3d）
  - 与 FastBEV 检测可并行运行，互不干扰

输出格式统一为 map_result.json，集成到 video_demo.py 的 BEV 视图中。
"""

from .nusc_map_extractor import NuScenesMapExtractor, extract_map_for_frames

__all__ = ["NuScenesMapExtractor", "extract_map_for_frames"]
