#!/usr/bin/env python3
"""
run_maptr.py — MapTRv2 地图感知命令行入口
==========================================

用法：
  # NuScenes GT 模式（总是可用，无需 checkpoint）
  python tools/maptr/run_maptr.py \
      --frames-dir  outputs/frames \
      --nuscenes-dir data/nuscenes \
      --mode gt

  # 自动选择（模型可用则用模型，否则用 GT）
  python tools/maptr/run_maptr.py \
      --frames-dir  outputs/frames \
      --mode auto

  # MapTRv2 模型推理（需要 checkpoint）
  python tools/maptr/run_maptr.py \
      --frames-dir  outputs/frames \
      --mode model \
      --ckpt        model/maptr/maptrv2_tiny_r50_24e.pth

  # 验证单帧（打印结果统计）
  python tools/maptr/run_maptr.py \
      --frames-dir  outputs/frames \
      --mode gt \
      --verify

  # 覆盖已有结果
  python tools/maptr/run_maptr.py \
      --frames-dir  outputs/frames \
      --mode gt \
      --overwrite
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

# 使项目根目录下的 tools.maptr 可直接 import
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.maptr.maptr_model_infer import MapInference, check_model_available


# ─── 结果验证与统计 ──────────────────────────────────────────────────────────

def verify_results(frames_dir: str, n_check: int = 5) -> bool:
    """
    检查 map_result.json 是否存在且有效。

    返回 True 表示验证通过。
    """
    frames_dir = Path(frames_dir)
    frame_dirs = sorted(frames_dir.glob("frame_*"))[:n_check]

    if not frame_dirs:
        print("[验证] 未找到帧目录")
        return False

    all_ok = True
    total_elements = 0
    type_counter = Counter()

    print(f"\n[验证] 检查前 {len(frame_dirs)} 帧的地图结果：")
    for fd in frame_dirs:
        map_path = fd / "map_result.json"
        if not map_path.exists():
            print(f"  ✗ {fd.name}: map_result.json 缺失")
            all_ok = False
            continue

        with open(map_path) as f:
            result = json.load(f)

        elements = result.get("elements", [])
        n = len(elements)
        total_elements += n
        for e in elements:
            type_counter[e.get("type", "?")] += 1

        source = result.get("source", "?")
        loc    = result.get("location", "?")
        print(f"  ✓ {fd.name}: {n:3d} 个元素  [source={source}  loc={loc}]")

    if all_ok:
        print(f"\n[验证] 通过 ✓")
        print(f"  平均元素数: {total_elements / max(len(frame_dirs), 1):.1f}")
        print(f"  类型分布: {dict(type_counter)}")
    else:
        print("\n[验证] 部分帧失败 ✗")

    return all_ok


def check_model_status():
    """打印 MapTRv2 模型可用性状态。"""
    avail, reason = check_model_available()
    print("=" * 60)
    print("MapTRv2 模型状态")
    print("=" * 60)
    if avail:
        print("  ✓ 模型可用，运行 --mode model 使用神经网络推理")
    else:
        print(f"  ✗ 模型不可用：\n    {reason.replace(chr(10), chr(10) + '    ')}")
        print("\n  → 当前将使用 NuScenes GT 模式（--mode gt）")
        print("  → GT 模式提供与 MapTRv2 完全相同格式的高精度 GT 地图")
    print("=" * 60)


# ─── 主函数 ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="MapTRv2 地图感知：提取 HD 地图元素",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--frames-dir", default="outputs/frames",
        help="帧目录的父目录（默认: outputs/frames）"
    )
    parser.add_argument(
        "--nuscenes-dir", default="data/nuscenes",
        help="NuScenes 数据集根目录（默认: data/nuscenes）"
    )
    parser.add_argument(
        "--version", default="v1.0-mini",
        help="NuScenes 版本（默认: v1.0-mini）"
    )
    parser.add_argument(
        "--mode", choices=["gt", "model", "auto"], default="auto",
        help="推理模式：gt=GT提取 model=神经网络 auto=自动（默认: auto）"
    )
    parser.add_argument(
        "--ckpt", default="model/maptr/maptr_nano_r18_110e.pth",
        help="MapTRv2 checkpoint 路径（默认: model/maptr/maptr_nano_r18_110e.pth）"
    )
    parser.add_argument(
        "--score-thr", type=float, default=0.5,
        help="模型置信度阈值（仅 model 模式，默认: 0.5）"
    )
    parser.add_argument(
        "--patch-size", type=float, default=120.0,
        help="地图查询范围（米，正方形边长，默认: 120）"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="覆盖已有的 map_result.json"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="运行后验证结果（打印统计信息）"
    )
    parser.add_argument(
        "--check-model", action="store_true",
        help="仅检查 MapTRv2 模型状态，不运行推理"
    )
    parser.add_argument(
        "--frame", default=None,
        help="仅处理单帧（如 frame_00000）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 仅检查模型状态
    if args.check_model:
        check_model_status()
        return

    # ── 初始化推理接口
    mi = MapInference(
        nuscenes_dir=args.nuscenes_dir,
        version=args.version,
        mode=args.mode,
        ckpt_path=args.ckpt,
        score_thr=args.score_thr,
        patch_size=args.patch_size,
        verbose=True,
    )

    print(f"\n[MapTRv2] 推理模式: {mi.effective_mode}")
    print(f"[MapTRv2] 帧目录:   {args.frames_dir}")

    # ── 单帧模式
    if args.frame is not None:
        frame_dir = str(Path(args.frames_dir) / args.frame)
        if not os.path.isdir(frame_dir):
            print(f"[错误] 帧目录不存在: {frame_dir}")
            sys.exit(1)
        result = mi.extract_frame(frame_dir, save=True)
        if result is None:
            print("[错误] 推理失败")
            sys.exit(1)
        n = len(result.get("elements", []))
        print(f"\n[完成] {n} 个地图元素已保存到 {frame_dir}/map_result.json")
        if args.verify:
            verify_results(args.frames_dir, n_check=1)
        return

    # ── 批量模式
    print(f"[MapTRv2] 开始批量处理...")
    n_success = mi.extract_frames(args.frames_dir, overwrite=args.overwrite)

    # ── 验证
    if args.verify:
        verify_results(args.frames_dir, n_check=10)


if __name__ == "__main__":
    main()
