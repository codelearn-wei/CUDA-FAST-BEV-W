import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json
import re
import cv2
from collections import defaultdict

# ─── 地图元素颜色（对应 MapElementType: 0=Divider, 1=Boundary, 2=PedCrossing）
MAP_ELEM_COLORS = {0: '#00CFCF', 1: '#00C040', 2: '#4080FF'}
MAP_ELEM_LABELS = {0: 'divider', 1: 'boundary', 2: 'ped_crossing'}


def parse_joint_result_frames(frames_dir, yaw_offset=0.0):
    """解析 joint_result.json，优先使用全局坐标（global_position/global_yaw）"""
    tracking_data = defaultdict(list)
    detection_data = defaultdict(list)
    map_data = defaultdict(list)

    subdirs = sorted(
        [d for d in os.listdir(frames_dir) if re.match(r'frame_\d+', d)],
        key=lambda s: int(re.search(r'\d+', s).group()))

    for subdir in subdirs:
        frame_num = int(re.search(r'\d+', subdir).group())
        jpath = os.path.join(frames_dir, subdir, 'joint_result.json')
        if not os.path.exists(jpath):
            continue
        with open(jpath, 'r') as f:
            jr = json.load(f)

        for t in jr.get('tracks', []):
            if 'global_position' in t and 'global_yaw' in t:
                pos = t['global_position']
                yaw = t['global_yaw']
            else:
                pos = t['position']
                yaw = t.get('yaw', 0.0)
            sz = t['size']          # [w, l, h]
            x, y = pos[0], pos[1]
            w, l = sz[0], sz[1]
            yaw = yaw + yaw_offset
            tracking_data[frame_num].append((t['track_id'], x, y, l, w, yaw))

        for d in jr.get('detections', []):
            x, y = d.get('x', 0.0), d.get('y', 0.0)
            l, w = d.get('l', 1.0), d.get('w', 1.0)
            yaw = d.get('yaw', 0.0) + yaw_offset
            detection_data[frame_num].append((None, x, y, l, w, yaw))

        for elem in jr.get('map', {}).get('elements', []):
            type_id = int(elem.get('type', 0))
            pts_raw = elem.get('pts', [])
            if len(pts_raw) >= 2:
                pts = [[float(p[0]), float(p[1])] for p in pts_raw if len(p) >= 2]
                if pts:
                    map_data[frame_num].append({'type_id': type_id, 'pts': pts})

    return tracking_data, detection_data, map_data


def parse_tracking_json_frames(frames_dir, yaw_offset=0.0):
    """解析 tracks.json，优先使用全局坐标（global_position/global_yaw）"""
    frame_data = defaultdict(list)
    subdirs = [d for d in os.listdir(frames_dir) if re.match(r'frame_\d+', d)]
    subdirs.sort(key=lambda s: int(re.search(r'\d+', s).group()))

    for subdir in subdirs:
        frame_num = int(re.search(r'\d+', subdir).group())
        json_path = os.path.join(frames_dir, subdir, 'tracks.json')
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            tracks = json.load(f)

        for track in tracks:
            track_id = track['track_id']
            if 'global_position' in track and 'global_yaw' in track:
                pos = track['global_position']
                yaw = track['global_yaw']
            else:
                pos = track['position']
                yaw = track['yaw']
            size = track['size']
            x, y = pos[0], pos[1]
            w, l = size[0], size[1]   # w: 宽度（横向），l: 长度（纵向）
            yaw = yaw + yaw_offset
            frame_data[frame_num].append((track_id, x, y, l, w, yaw))

    return frame_data


def parse_detection_json_frames(frames_dir, yaw_offset=0.0):
    """解析 result.json（检测框），仍为局部坐标"""
    frame_data = defaultdict(list)
    subdirs = [d for d in os.listdir(frames_dir) if re.match(r'frame_\d+', d)]
    subdirs.sort(key=lambda s: int(re.search(r'\d+', s).group()))

    for subdir in subdirs:
        frame_num = int(re.search(r'\d+', subdir).group())
        json_path = os.path.join(frames_dir, subdir, 'result.json')
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            detections = json.load(f)

        for det in detections:
            center = det.get('center_xyz')
            size = det.get('size_xyz')
            yaw = det.get('yaw', 0.0) + yaw_offset
            if center is None or size is None:
                continue
            x, y, z = center
            l, w = size[1], size[0]   # l: 长度（纵向），w: 宽度（横向）
            frame_data[frame_num].append((None, x, y, l, w, yaw))

    return frame_data


def rotated_rectangle_vertices(cx, cy, length, width, heading_rad):
    """
    返回旋转矩形的四个顶点（世界坐标）。
    参数:
        cx, cy: 矩形中心坐标
        length: 纵向长度（沿航向方向）
        width:  横向宽度（垂直于航向方向）
        heading_rad: 航向角，以 X 轴正方向为 0 弧度，逆时针为正。
    返回:
        (4,2) numpy数组，四个顶点坐标 (x, y)
    """
    dir_x = np.cos(heading_rad)
    dir_y = np.sin(heading_rad)
    perp_x = -dir_y
    perp_y = dir_x

    half_l = length / 2.0
    half_w = width / 2.0

    corners_local = np.array([
        [-half_w, -half_l],
        [ half_w, -half_l],
        [ half_w,  half_l],
        [-half_w,  half_l]
    ])

    world_corners = np.zeros_like(corners_local)
    world_corners[:, 0] = cx + corners_local[:, 0] * perp_x + corners_local[:, 1] * dir_x
    world_corners[:, 1] = cy + corners_local[:, 0] * perp_y + corners_local[:, 1] * dir_y
    return world_corners


def get_color_for_id(track_id, cmap_name='tab20'):
    cmap = plt.get_cmap(cmap_name)
    return cmap((track_id * 13) % 20)


def round_to_even(number):
    return int(np.ceil(number / 2) * 2)


def load_ego_data(frames_dir, all_frames, frame_to_subdir):
    """
    从各帧的 meta.json 中读取自车全局位置和朝向。
    frame_to_subdir: 字典，frame_num -> subdir_name (如 0 -> 'frame_0' 或 'frame_000')
    返回字典: {frame_id: (x, y, heading_rad)}
    """
    ego_dict = {}
    for frame_id in all_frames:
        if frame_id not in frame_to_subdir:
            continue
        subdir = frame_to_subdir[frame_id]
        meta_path = os.path.join(frames_dir, subdir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                trans = meta.get("ego_translation_global")
                yaw = meta.get("ego_yaw_global")
                if trans is not None and yaw is not None:
                    ego_dict[frame_id] = (trans[0], trans[1], yaw)
            except Exception as e:
                print(f"Warning: cannot read ego from {meta_path}: {e}")
    return ego_dict


def visualize_tracking_from_json(frames_dir, output_video_path, fps=6, dpi=400, figsize=(16, 16),
                                  bitrate="50M", xlim=None, ylim=None, use_joint_result=False,
                                  yaw_offset=0.0, dynamic_view=True, view_size=(80.0, 80.0),
                                  draw_ego=True, ego_size=(2.0, 4.0), ego_centered=True,
                                  smooth_factor=0.7):
    """
    渲染 BEV 视频，支持全局/局部坐标，航向角以 X 轴正方向为 0 度（逆时针为正）。
    
    新增强化特性:
        ego_centered: 当为 True 且全局模式时，视野强制以自车为中心（不跟随其他目标）。
        smooth_factor: 对自车中心位置进行指数平滑的系数（0~1），值越小抖动抑制越强，默认0.7。
    """
    # 首先构建 frame_num -> subdir_name 的映射（所有 frame_* 目录）
    all_subdirs = [d for d in os.listdir(frames_dir) if re.match(r'frame_\d+', d)]
    frame_to_subdir = {}
    for subdir in all_subdirs:
        frame_num = int(re.search(r'\d+', subdir).group())
        frame_to_subdir[frame_num] = subdir

    # 解析数据
    if use_joint_result:
        tracking_data, detection_data, map_data = parse_joint_result_frames(frames_dir, yaw_offset=yaw_offset)
        print("[joint 模式] 读取 joint_result.json（含地图元素）")
    else:
        tracking_data = parse_tracking_json_frames(frames_dir, yaw_offset=yaw_offset)
        detection_data = parse_detection_json_frames(frames_dir, yaw_offset=yaw_offset)
        map_data = {}

    if not tracking_data and not detection_data:
        print("No data found.")
        return

    all_frames = sorted(set(tracking_data.keys()) | set(detection_data.keys()))
    print(f"Total frames with data: {len(all_frames)}")

    # 加载自车数据
    ego_data = {}
    if draw_ego:
        ego_data = load_ego_data(frames_dir, all_frames, frame_to_subdir)
        print(f"Loaded ego data for {len(ego_data)} frames")
        if len(ego_data) > 0:
            sample_frame = next(iter(ego_data.items()))
            print(f"Ego sample (frame {sample_frame[0]}): pos=({sample_frame[1][0]:.2f}, {sample_frame[1][1]:.2f}), yaw={sample_frame[1][2]:.3f} rad")
        else:
            print("Warning: No ego data loaded. Check if meta.json exists and contains ego_translation_global/ego_yaw_global.")

    # 自动检测是否使用全局坐标
    use_global = False
    for frame_id in all_frames:
        if tracking_data.get(frame_id):
            first_track = tracking_data[frame_id][0]
            if abs(first_track[1]) > 100:
                use_global = True
                break
    if not use_global:
        for frame_id in all_frames:
            if detection_data.get(frame_id):
                first_det = detection_data[frame_id][0]
                if abs(first_det[1]) > 100:
                    use_global = True
                    break
    if not use_global and ego_data:
        for frame_id, (ex, ey, _) in ego_data.items():
            if abs(ex) > 100 or abs(ey) > 100:
                use_global = True
                break

    if use_global:
        print("检测到全局坐标，将使用全局 BEV 视图，并隐藏检测框（坐标系不匹配）")
        print("航向角解释：X 轴正方向为 0°，逆时针为正。")
        if ego_centered and draw_ego and ego_data:
            print("已启用【以自车为中心】模式，视野将跟随自车并平滑过渡。")
    else:
        print("使用局部坐标（自车坐标系）")
        print("注意：局部航向角需通过 --yaw-offset 转换为 X 轴为 0 的定义（默认 0 表示指向 X 正方向）")

    # 确定视野范围策略
    fix_range = (xlim is not None and ylim is not None) or not use_global or not dynamic_view
    if fix_range:
        if xlim is None or ylim is None:
            all_x, all_y = [], []
            for frame_id in all_frames:
                for (_, x, y, l, w, yaw) in tracking_data.get(frame_id, []):
                    corners = rotated_rectangle_vertices(x, y, l, w, yaw)
                    all_x.extend(corners[:, 0])
                    all_y.extend(corners[:, 1])
                if not use_global:
                    for (_, x, y, l, w, yaw) in detection_data.get(frame_id, []):
                        corners = rotated_rectangle_vertices(x, y, l, w, yaw)
                        all_x.extend(corners[:, 0])
                        all_y.extend(corners[:, 1])
                if use_global and draw_ego and frame_id in ego_data:
                    ex, ey, _ = ego_data[frame_id]
                    all_x.append(ex)
                    all_y.append(ey)
            if all_x:
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)
                margin_x = max(1.0, (x_max - x_min) * 0.1)
                margin_y = max(1.0, (y_max - y_min) * 0.1)
                xlim = (x_min - margin_x, x_max + margin_x)
                ylim = (y_min - margin_y, y_max + margin_y)
            else:
                xlim = (-50, 50)
                ylim = (-50, 50)
        else:
            xlim = (xlim[0], xlim[1])
            ylim = (ylim[0], ylim[1])
    else:
        view_w, view_h = view_size
        print(f"动态视野模式：视野大小 {view_w} x {view_h} 米")
        if ego_centered and use_global and draw_ego and ego_data:
            print("  视野中心：自车（平滑跟随）")
        else:
            print("  视野中心：跟踪框中心（若无则跟随自车）")
        prev_center = None
        smooth_center = None  # 用于平滑的中心位置

    # Matplotlib 准备
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10

    fig_width, fig_height = figsize
    pixel_width = round_to_even(int(fig_width * dpi))
    pixel_height = round_to_even(int(fig_height * dpi))
    adjusted_figsize = (pixel_width / dpi, pixel_height / dpi)

    print(f"Resolution: {pixel_width}x{pixel_height}")

    fig, ax = plt.subplots(figsize=adjusted_figsize, dpi=dpi)
    ax.set_aspect('equal')
    ax.set_xlabel('X (right) [m]', fontsize=12)
    ax.set_ylabel('Y (forward) [m]', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (pixel_width, pixel_height))
    if not out_video.isOpened():
        print("Error: Could not create video writer.")
        plt.close(fig)
        return

    for idx, frame_id in enumerate(all_frames):
        print(f"Rendering frame {idx+1}/{len(all_frames)} (frame_id={frame_id})", end='\r')
        ax.clear()

        if fix_range:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
        else:
            # 决定视野中心
            if ego_centered and use_global and draw_ego and frame_id in ego_data:
                # 使用自车位置（平滑处理）
                ex, ey, _ = ego_data[frame_id]
                raw_center = (ex, ey)
                if smooth_center is None:
                    smooth_center = raw_center
                else:
                    # 指数平滑: new = alpha * raw + (1-alpha) * old
                    alpha = smooth_factor
                    smooth_center = (alpha * raw_center[0] + (1 - alpha) * smooth_center[0],
                                     alpha * raw_center[1] + (1 - alpha) * smooth_center[1])
                center_x, center_y = smooth_center
                prev_center = (center_x, center_y)  # 用于后续帧丢失时保持
            else:
                # 原有逻辑：跟随跟踪框中心，若无则跟随自车
                tracks_this = tracking_data.get(frame_id, [])
                centers = [(x, y) for (_, x, y, l, w, yaw) in tracks_this]
                if not centers and use_global and draw_ego and frame_id in ego_data:
                    ex, ey, _ = ego_data[frame_id]
                    centers = [(ex, ey)]
                if centers:
                    center_x = np.mean([c[0] for c in centers])
                    center_y = np.mean([c[1] for c in centers])
                    prev_center = (center_x, center_y)
                elif prev_center is not None:
                    center_x, center_y = prev_center
                else:
                    center_x, center_y = 0.0, 0.0
                    prev_center = (center_x, center_y)
                # 此处不进行平滑，保持原样
            ax.set_xlim(center_x - view_w/2, center_x + view_w/2)
            ax.set_ylim(center_y - view_h/2, center_y + view_h/2)

        ax.set_aspect('equal')
        ax.set_xlabel('X (right) [m]', fontsize=12)
        ax.set_ylabel('Y (forward) [m]', fontsize=12)
        ax.set_title(f'Frame {frame_id}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)

        # 检测框（仅局部模式）
        if not use_global:
            for (_, x, y, l, w, yaw) in detection_data.get(frame_id, []):
                corners = rotated_rectangle_vertices(x, y, l, w, yaw)
                poly = patches.Polygon(corners, closed=True,
                                       facecolor='gray', edgecolor='gray',
                                       linewidth=1.0, alpha=0.3, linestyle='--')
                ax.add_patch(poly)

        # 地图元素
        for elem in map_data.get(frame_id, []):
            type_id = elem['type_id']
            pts = np.array(elem['pts'], dtype=np.float64)
            color = MAP_ELEM_COLORS.get(type_id, '#FF8800')
            ax.plot(pts[:, 0], pts[:, 1],
                    color=color, linewidth=2.0, alpha=0.85, solid_capstyle='round')

        # 跟踪框
        for (track_id, x, y, l, w, yaw) in tracking_data.get(frame_id, []):
            corners = rotated_rectangle_vertices(x, y, l, w, yaw)
            poly = patches.Polygon(corners, closed=True,
                                   facecolor=get_color_for_id(track_id, 'tab20'),
                                   edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.add_patch(poly)
            ax.text(x, y, str(track_id), ha='center', va='center',
                    fontsize=5, color='white', weight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

        # 自车绘制（仅在全局模式且可用时）
        if use_global and draw_ego and frame_id in ego_data:
            ex, ey, e_yaw = ego_data[frame_id]
            ego_w, ego_l = ego_size
            corners = rotated_rectangle_vertices(ex, ey, ego_l, ego_w, e_yaw)
            poly = patches.Polygon(corners, closed=True,
                                   facecolor='deepskyblue', edgecolor='blue',
                                   linewidth=2.5, alpha=0.7)
            ax.add_patch(poly)
            # 箭头指示前方
            arrow_len = 2.0
            dir_x = np.cos(e_yaw)
            dir_y = np.sin(e_yaw)
            tip_x = ex + dir_x * (ego_l/2 + arrow_len)
            tip_y = ey + dir_y * (ego_l/2 + arrow_len)
            ax.arrow(ex, ey, tip_x - ex, tip_y - ey,
                     head_width=0.5, head_length=0.5, fc='red', ec='red', alpha=0.9)
            ax.text(ex, ey - ego_l/2 - 0.8, 'Ego', ha='center', va='top',
                    fontsize=6, color='darkblue', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.1'))

        fig.canvas.draw()
        img_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_rgb = img_rgb.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out_video.write(img_bgr)

    out_video.release()
    plt.close(fig)
    print(f"\n✅ Video saved to: {output_video_path}")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="BEV 可视化 — 航向角以 X 轴正方向为 0 度（逆时针），支持自车为中心动态视野")
    _p.add_argument("--frames-dir", type=str, default=None,
                    help="包含 frame_* 子目录的根目录（默认: outputs/frames）")
    _p.add_argument("--out-video", type=str, default=None,
                    help="输出视频路径（默认: outputs/tracking_result.mp4）")
    _p.add_argument("--joint", action="store_true",
                    help="读取 joint_result.json（含地图元素）")
    _p.add_argument("--fps", type=int, default=6)
    _p.add_argument("--dpi", type=int, default=500)
    _p.add_argument("--figsize", type=int, default=6,
                    help="图像尺寸（英寸，正方形）")
    _p.add_argument("--bitrate", type=str, default="10M",
                    help="（保留参数，OpenCV 忽略）")
    _p.add_argument("--xlim", type=float, nargs=2, metavar=('MIN','MAX'), default=None)
    _p.add_argument("--ylim", type=float, nargs=2, metavar=('MIN','MAX'), default=None)
    _p.add_argument("--yaw-offset", type=float, default=0.0,
                    help="额外航向角偏移（弧度），用于校准。默认 0 表示 JSON 中的 global_yaw 已是 X 轴为 0 的定义。")
    _p.add_argument("--no-dynamic-view", action="store_true",
                    help="关闭动态视野（全局模式下固定全序列范围）")
    _p.add_argument("--view-size", type=float, nargs=2, metavar=('WIDTH','HEIGHT'), default=(80.0, 80.0),
                    help="动态视野大小（米），默认 80x80")
    _p.add_argument("--no-draw-ego", action="store_true",
                    help="不绘制自车（默认绘制）")
    _p.add_argument("--ego-size", type=float, nargs=2, metavar=('WIDTH','LENGTH'), default=(2.0, 4.0),
                    help="自车矩形尺寸（宽度，长度），单位米，默认 2.0x4.0")
    _p.add_argument("--no-ego-centered", action="store_true",
                    help="关闭「以自车为中心」模式（默认开启，即视野跟随自车）")
    _p.add_argument("--smooth-factor", type=float, default=0.7,
                    help="自车中心平滑系数（0~1），值越小越平滑，默认0.7")
    _a = _p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if _a.frames_dir:
        frames_directory = os.path.abspath(_a.frames_dir)
    else:
        frames_directory = os.path.abspath(os.path.join(script_dir, "../outputs/frames"))
    if _a.out_video:
        output_video = os.path.abspath(_a.out_video)
    else:
        output_directory = os.path.abspath(os.path.join(script_dir, "../outputs"))
        os.makedirs(output_directory, exist_ok=True)
        output_video = os.path.join(output_directory, "tracking_result.mp4")

    if not os.path.exists(frames_directory):
        print(f"Directory not found: {frames_directory}")
    else:
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        visualize_tracking_from_json(
            frames_directory,
            output_video,
            fps=_a.fps,
            dpi=_a.dpi,
            figsize=(_a.figsize, _a.figsize),
            bitrate=_a.bitrate,
            xlim=_a.xlim,
            ylim=_a.ylim,
            use_joint_result=_a.joint,
            yaw_offset=_a.yaw_offset,
            dynamic_view=(not _a.no_dynamic_view),
            view_size=_a.view_size,
            draw_ego=(not _a.no_draw_ego),
            ego_size=_a.ego_size,
            ego_centered=(not _a.no_ego_centered),
            smooth_factor=_a.smooth_factor,
        )