#!/usr/bin/env python3
"""
CUDA-FastBEV 端到端推理流水线

将 NuScenes 数据预处理 → BEV 检测+跟踪 → MapTR 地图推理 → 视频可视化
整合为单条命令，支持并行执行和耗时统计。

用法:
  # 完整流水线（推荐）
  conda run -n bev python tools/run_pipeline.py \\
      --nuscenes-dir data/nuscenes --version v1.0-mini \\
      --num-frames 50 --fps 6

  # 跳过数据预处理（已有 outputs/frames）
  conda run -n bev python tools/run_pipeline.py --skip-data-prep

  # 仅 BEV 推理，不做 MapTR
  conda run -n bev python tools/run_pipeline.py --skip-map

  # 并行执行 BEV 和 MapTR（需要足够 GPU 显存）
  conda run -n bev python tools/run_pipeline.py --parallel-infer

环境: conda activate bev
"""

import argparse
import os
import subprocess
import sys
import time
import threading
from pathlib import Path


# ─── ANSI 颜色 ─────────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _fmt_dt(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{int(secs//60)}m{secs%60:.0f}s"


def run_step(cmd: str, name: str, cwd: str = None, env: dict = None,
             capture: bool = False) -> tuple:
    """执行一个流水线步骤，打印命令和耗时，返回 (success, elapsed_seconds)。"""
    print(f"\n{_BOLD}{_CYAN}▶ [{name}]{_RESET}")
    print(f"  {cmd}")
    t0 = time.time()
    result = subprocess.run(
        cmd, shell=True, cwd=cwd or os.getcwd(), env=env or os.environ.copy(),
        capture_output=capture,
        text=capture,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  {_GREEN}✔ 完成 ({_fmt_dt(elapsed)}){_RESET}")
    else:
        print(f"  {_RED}✘ 失败 (exit={result.returncode}, {_fmt_dt(elapsed)}){_RESET}")
        if capture and result.stderr:
            print(result.stderr[-2000:])
    return result.returncode == 0, elapsed


def run_parallel(tasks: list) -> dict:
    """
    并行执行多个任务，每个任务为 (cmd, name, cwd) 元组。
    返回 {name: (success, elapsed)} 字典。
    """
    results = {}
    threads = []

    def _worker(cmd, name, cwd):
        ok, dt = run_step(cmd, name, cwd=cwd)
        results[name] = (ok, dt)

    for cmd, name, cwd in tasks:
        t = threading.Thread(target=_worker, args=(cmd, name, cwd))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results


def check_build(project_root: str) -> bool:
    """检查 C++ 二进制是否存在。"""
    binary = os.path.join(project_root, "build", "tracking_demo")
    if not os.path.exists(binary):
        print(f"{_RED}[错误] 未找到 build/tracking_demo。")
        print(f"  请先编译: cd build && source ../tool/environment.sh && cmake .. && make -j$(nproc){_RESET}")
        return False
    return True


def check_model(project_root: str, model: str) -> bool:
    """检查 TRT 模型文件是否存在。"""
    plan = os.path.join(project_root, "model", model, "build",
                        "fastbev_pre_trt.plan")
    if not os.path.exists(plan):
        print(f"{_YELLOW}[警告] 未找到 TRT plan: {plan}")
        print(f"  请先用 tool/build_trt_engine.sh 构建引擎{_RESET}")
        return False
    return True


def parse_args():
    p = argparse.ArgumentParser(
        description="CUDA-FastBEV 端到端流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 数据源
    p.add_argument("--nuscenes-dir", default="data/nuscenes",
                   help="NuScenes 根目录")
    p.add_argument("--version", default="v1.0-mini",
                   help="NuScenes 版本")
    p.add_argument("--num-frames", type=int, default=50,
                   help="要处理的帧数（默认 50）")
    p.add_argument("--stride", type=int, default=1,
                   help="帧间隔（默认 1）")
    p.add_argument("--scene-names", default=None,
                   help="指定场景名（逗号分隔），None 表示全部")

    # 路径
    p.add_argument("--out-dir",   default="outputs/frames",
                   help="帧数据输出目录")
    p.add_argument("--video-dir", default="outputs/video",
                   help="视频输出目录")
    p.add_argument("--model",     default="resnet18",
                   help="BEV 模型名称（默认 resnet18）")

    # 推理参数
    p.add_argument("--score-thr", type=float, default=0.3,
                   help="置信度阈值（默认 0.3）")
    p.add_argument("--fps",       type=int,   default=6,
                   help="输出视频帧率")
    p.add_argument("--bev-size",  type=int,   default=800,
                   help="BEV 画布尺寸（像素）")
    p.add_argument("--cam-width", type=int,   default=480,
                   help="相机图像宽度（像素）")

    # 流程控制
    p.add_argument("--skip-data-prep", action="store_true",
                   help="跳过数据预处理（已有 out-dir）")
    p.add_argument("--skip-bev",       action="store_true",
                   help="跳过 BEV 检测+跟踪")
    p.add_argument("--skip-map",       action="store_true",
                   help="跳过 MapTR 地图推理")
    p.add_argument("--skip-video",     action="store_true",
                   help="跳过视频生成")
    p.add_argument("--parallel-infer", action="store_true",
                   help="BEV 和 MapTR 并行推理（需足够 GPU 显存）")
    p.add_argument("--overwrite",      action="store_true",
                   help="覆盖已有推理结果")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    print(f"\n{_BOLD}{'='*60}")
    print(f"  CUDA-FastBEV 端到端流水线")
    print(f"  帧数: {args.num_frames}  模型: {args.model}  阈值: {args.score_thr}")
    print(f"{'='*60}{_RESET}\n")

    timings = {}
    total_t0 = time.time()

    # ── Step 0: 预检 ─────────────────────────────────────────────────────────
    if not args.skip_bev and not check_build(str(project_root)):
        sys.exit(1)
    if not args.skip_bev:
        check_model(str(project_root), args.model)  # 警告但不中止

    # ── Step 1: 数据预处理 ──────────────────────────────────────────────────
    if not args.skip_data_prep:
        scene_arg = (f"--scene-names {args.scene_names}"
                     if args.scene_names else "")
        cmd = (
            f"python tools/nuscenes_adapter.py "
            f"--nuscenes-dir {args.nuscenes_dir} "
            f"--version {args.version} "
            f"--out-dir {args.out_dir} "
            f"--num-frames {args.num_frames} "
            f"--stride {args.stride} "
            f"{scene_arg}"
        )
        ok, dt = run_step(cmd, "Step1 数据预处理")
        timings["data_prep"] = dt
        if not ok:
            print(f"{_RED}流水线中止{_RESET}")
            sys.exit(1)

    # ── Step 2 & 3: BEV 检测+跟踪 / MapTR 并行或串行 ────────────────────────
    bev_cmd = (
        f"./build/tracking_demo {args.out_dir} {args.model} fp16 "
        f"--score-thr {args.score_thr} "
        f"--output-format json --batch"
    )
    map_cmd = (
        f"conda run -n bev python tools/maptr/run_maptr.py "
        f"--frames-dir {args.out_dir} "
        f"--mode model "
        f"--score-thr {args.score_thr} "
        + ("--overwrite " if args.overwrite else "")
    )

    if not args.skip_bev and not args.skip_map and args.parallel_infer:
        # 并行：BEV(C++) 和 MapTR(Python) 同时跑
        print(f"\n{_BOLD}{_CYAN}▶ [Step2+3 并行推理]{_RESET}")
        print(f"  BEV  : {bev_cmd}")
        print(f"  MapTR: {map_cmd}")
        res = run_parallel([
            (bev_cmd,  "Step2 BEV检测+跟踪", str(project_root)),
            (map_cmd,  "Step3 MapTR地图",     str(project_root)),
        ])
        for name, (ok, dt) in res.items():
            timings[name] = dt
            if not ok:
                print(f"{_RED}[{name}] 失败，流水线中止{_RESET}")
                sys.exit(1)
    else:
        if not args.skip_bev:
            ok, dt = run_step(bev_cmd, "Step2 BEV检测+跟踪")
            timings["bev_tracking"] = dt
            if not ok:
                print(f"{_RED}BEV 推理失败，流水线中止{_RESET}")
                sys.exit(1)

        if not args.skip_map:
            ok, dt = run_step(map_cmd, "Step3 MapTR地图推理")
            timings["map_infer"] = dt
            if not ok:
                print(f"{_YELLOW}MapTR 推理失败（将跳过地图显示）{_RESET}")
                # 非致命：继续生成视频，不含地图

    # ── Step 4: 视频生成 ─────────────────────────────────────────────────────
    if not args.skip_video:
        video_cmd = (
            f"python tools/video_demo.py "
            f"--frames-dir {args.out_dir} "
            f"--out-dir {args.video_dir} "
            f"--score-thr {args.score_thr} "
            f"--fps {args.fps} "
            f"--bev-size {args.bev_size} "
            f"--cam-width {args.cam_width}"
        )
        ok, dt = run_step(video_cmd, "Step4 视频生成")
        timings["video"] = dt

    # ── 汇总耗时 ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print(f"\n{_BOLD}{'='*60}")
    print(f"  流水线完成  (总计 {_fmt_dt(total_elapsed)})")
    print(f"{'='*60}{_RESET}")
    print(f"{'步骤':<22} {'耗时':>10}")
    print("-" * 34)
    for name, dt in timings.items():
        label = {"data_prep": "数据预处理",
                 "bev_tracking": "BEV检测+跟踪",
                 "Step2 BEV检测+跟踪": "BEV检测+跟踪",
                 "map_infer": "MapTR地图推理",
                 "Step3 MapTR地图": "MapTR地图推理",
                 "video": "视频生成"}.get(name, name)
        print(f"  {label:<20} {_fmt_dt(dt):>10}")
    print("-" * 34)
    print(f"  {'总计':<20} {_fmt_dt(total_elapsed):>10}")
    print()

    if not args.skip_video and os.path.isdir(args.video_dir):
        videos = list(Path(args.video_dir).glob("*.mp4"))
        if videos:
            print(f"  视频输出: {videos[0]}")
    if not args.skip_bev:
        n_frames = sum(1 for d in Path(args.out_dir).iterdir()
                       if d.is_dir() and d.name.startswith("frame_"))
        if n_frames > 0:
            fps_bev = n_frames / timings.get("bev_tracking",
                                              timings.get("Step2 BEV检测+跟踪", 1))
            print(f"  BEV 推理吞吐: {fps_bev:.1f} FPS ({n_frames} 帧)")
    print()


if __name__ == "__main__":
    main()
