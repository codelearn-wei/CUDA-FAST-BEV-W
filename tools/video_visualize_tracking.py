import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json
import re
import subprocess
import shutil
import glob
from collections import defaultdict

# ─── 地图元素颜色（对应 MapElementType: 0=Divider, 1=Boundary, 2=PedCrossing）
MAP_ELEM_COLORS = {0: '#00CFCF', 1: '#00C040', 2: '#4080FF'}
MAP_ELEM_LABELS = {0: 'divider', 1: 'boundary', 2: 'ped_crossing'}


def parse_joint_result_frames(frames_dir, yaw_offset=0.0):
    """
    解析 joint_result.json，并可选择对航向角增加偏移（弧度）。
    """
    tracking_data  = defaultdict(list)
    detection_data = defaultdict(list)
    map_data       = defaultdict(list)

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
            pos = t['position']
            sz  = t['size']          # [w, l, h]
            x, y = pos[0], pos[1]
            w, l = sz[0], sz[1]
            yaw  = t.get('yaw', 0.0) + yaw_offset
            tracking_data[frame_num].append((t['track_id'], x, y, l, w, yaw))

        for d in jr.get('detections', []):
            x, y = d.get('x', 0.0), d.get('y', 0.0)
            l, w = d.get('l', 1.0), d.get('w', 1.0)
            yaw  = d.get('yaw', 0.0) + yaw_offset
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
    """解析 tracks.json，并应用航向角偏移"""
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
            pos = track['position']
            size = track['size']
            yaw = track['yaw'] + yaw_offset
            
            x = pos[0]
            y = pos[1]
            w = size[0]   # 宽度（横向）
            l = size[1]   # 长度（纵向）
            
            frame_data[frame_num].append((track_id, x, y, l, w, yaw))
    
    return frame_data

def parse_detection_json_frames(frames_dir, yaw_offset=0.0):
    """解析 result.json，并应用航向角偏移"""
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
            l = size[1]   # 长度（纵向）
            w = size[0]   # 宽度（横向）
            
            frame_data[frame_num].append((None, x, y, l, w, yaw))
    
    return frame_data

def rotated_rectangle_vertices(cx, cy, len_x, len_y, angle_rad):
    """
    返回旋转矩形的四个顶点（世界坐标）。
    len_x: 横向宽度（沿 x 方向）
    len_y: 纵向长度（沿 y 方向）
    angle_rad: 航向角，0 表示矩形长边沿 y 轴正方向（前向）
    """
    half_x = len_x / 2.0
    half_y = len_y / 2.0
    corners_local = np.array([
        [-half_x, -half_y],
        [ half_x, -half_y],
        [ half_x,  half_y],
        [-half_x,  half_y]
    ])
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = corners_local @ rot_mat.T
    corners[:, 0] += cx
    corners[:, 1] += cy
    return corners

def get_color_for_id(track_id, cmap_name='tab20'):
    cmap = plt.get_cmap(cmap_name)
    return cmap((track_id * 13) % 20)

def round_to_even(number):
    return int(np.ceil(number / 2) * 2)

def create_video_with_ffmpeg_simple(frame_pattern, output_video, fps=6, bitrate="50M"):
    try:
        frames = sorted(glob.glob(frame_pattern.replace('%06d', '*')))
        if not frames:
            print("No frames found!")
            return False
        first_frame = frames[0]
        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x',
            first_frame
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode == 0 and 'x' in result.stdout:
            width, height = map(int, result.stdout.strip().split('x'))
            width = round_to_even(width)
            height = round_to_even(height)
            scale_filter = f"scale={width}:{height},setsar=1"
        else:
            scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-vf', scale_filter,
            '-c:v', 'libx264',
            '-preset', 'veryslow',
            '-b:v', bitrate,
            '-minrate', bitrate,
            '-maxrate', bitrate,
            '-bufsize', '10M',
            '-x264-params', 'nal-hrd=cbr:force-cfr=1',
            '-profile:v', 'high',
            '-level', '5.1',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_video
        ]
        print(f"Running FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        verify_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=bit_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            output_video
        ]
        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
        if verify_result.stdout.strip():
            actual_bitrate = int(verify_result.stdout.strip()) / 1000000
            print(f"✅ Video bitrate: {actual_bitrate:.2f} Mbps")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def visualize_tracking_from_json(frames_dir, output_video_path, fps=6, dpi=400, figsize=(16, 16),
                                  bitrate="50M", xlim=None, ylim=None, use_joint_result=False,
                                  yaw_offset=-np.pi):
    """
    yaw_offset: 对原始航向角增加的偏移（弧度）。默认减去 π，使 0 指向 y 正前方。
    """
    # 解析数据并应用偏移
    if use_joint_result:
        tracking_data, detection_data, map_data = parse_joint_result_frames(frames_dir, yaw_offset=yaw_offset)
        print("[joint 模式] 读取 joint_result.json（含地图元素）")
    else:
        tracking_data  = parse_tracking_json_frames(frames_dir, yaw_offset=yaw_offset)
        detection_data = parse_detection_json_frames(frames_dir, yaw_offset=yaw_offset)
        map_data = {}
    
    if not tracking_data and not detection_data:
        print("No data found.")
        return

    all_frames = sorted(set(tracking_data.keys()) | set(detection_data.keys()))
    print(f"Total frames: {len(all_frames)}")

    # 计算坐标范围（基于原始 x, y）
    if xlim is not None and ylim is not None:
        x_min, x_max = xlim[0], xlim[1]
        y_min, y_max = ylim[0], ylim[1]
    else:
        all_x = []
        all_y = []
        for frame_id in all_frames:
            for (_, x, y, l, w, yaw) in tracking_data.get(frame_id, []):
                # len_x = w, len_y = l
                corners = rotated_rectangle_vertices(x, y, w, l, yaw)
                all_x.extend(corners[:, 0])
                all_y.extend(corners[:, 1])
            for (_, x, y, l, w, yaw) in detection_data.get(frame_id, []):
                corners = rotated_rectangle_vertices(x, y, w, l, yaw)
                all_x.extend(corners[:, 0])
                all_y.extend(corners[:, 1])
        if not all_x:
            return
        if xlim is None:
            x_min, x_max = min(all_x), max(all_x)
            margin_x = max(1.0, (x_max - x_min) * 0.1)
            x_min -= margin_x
            x_max += margin_x
        else:
            x_min, x_max = xlim[0], xlim[1]
        if ylim is None:
            y_min, y_max = min(all_y), max(all_y)
            margin_y = max(1.0, (y_max - y_min) * 0.1)
            y_min -= margin_y
            y_max += margin_y
        else:
            y_min, y_max = ylim[0], ylim[1]

    temp_dir = os.path.join(os.path.dirname(output_video_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Saving temporary PNG frames to: {temp_dir}")

    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10

    fig_width, fig_height = figsize
    pixel_width = int(fig_width * dpi)
    pixel_height = int(fig_height * dpi)
    pixel_width = round_to_even(pixel_width)
    pixel_height = round_to_even(pixel_height)
    adjusted_figsize = (pixel_width / dpi, pixel_height / dpi)
    
    print(f"Resolution: {pixel_width}x{pixel_height}")
    print(f"Target bitrate: {bitrate}")

    for idx, frame_id in enumerate(all_frames):
        print(f"Rendering frame {idx+1}/{len(all_frames)} (frame_id={frame_id})", end='\r')
        fig, ax = plt.subplots(figsize=adjusted_figsize, dpi=dpi)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X (right) [m]', fontsize=12)
        ax.set_ylabel('Y (forward) [m]', fontsize=12)
        ax.set_title(f'Frame {frame_id}', fontsize=14)

        # 绘制检测框（灰色虚线）
        for (_, x, y, l, w, yaw) in detection_data.get(frame_id, []):
            corners = rotated_rectangle_vertices(x, y, w, l, yaw)
            poly = patches.Polygon(corners, closed=True,
                                   facecolor='gray', edgecolor='gray',
                                   linewidth=1.0, alpha=0.3, linestyle='--')
            ax.add_patch(poly)

        # 绘制地图元素
        for elem in map_data.get(frame_id, []):
            type_id = elem['type_id']
            pts = np.array(elem['pts'], dtype=np.float64)
            color = MAP_ELEM_COLORS.get(type_id, '#FF8800')
            ax.plot(pts[:, 0], pts[:, 1],
                    color=color, linewidth=2.0, alpha=0.85, solid_capstyle='round')

        # 绘制跟踪结果
        for (track_id, x, y, l, w, yaw) in tracking_data.get(frame_id, []):
            corners = rotated_rectangle_vertices(x, y, w, l, yaw)
            poly = patches.Polygon(corners, closed=True,
                                   facecolor=get_color_for_id(track_id, 'tab20'),
                                   edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.add_patch(poly)
            ax.text(x, y, str(track_id), ha='center', va='center',
                    fontsize=5, color='white', weight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

        ax.grid(True, linestyle='--', alpha=0.5)
        frame_filename = os.path.join(temp_dir, f"frame_{idx:06d}.png")
        fig.savefig(frame_filename, dpi=dpi, facecolor='white', edgecolor='none')
        plt.close(fig)

    print(f"\nAll PNG frames saved. Total: {len(all_frames)}")

    frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
    success = create_video_with_ffmpeg_simple(frame_pattern, output_video_path, fps=fps, bitrate=bitrate)
    if success:
        print(f"✅ High-bitrate video saved to: {output_video_path}")
    else:
        print("❌ Failed to create video")

    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("Done!")

if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Tracking 可视化 — 将 tracks.json / joint_result.json 渲染为 BEV 视频")
    _p.add_argument("--frames-dir", type=str, default=None,
                    help="包含 frame_* 子目录的帧数据根目录（默认: outputs/frames）")
    _p.add_argument("--out-video",  type=str, default=None,
                    help="输出视频路径（默认: outputs/tracking_result.mp4）")
    _p.add_argument("--joint",      action="store_true",
                    help="读取 joint_result.json（含 BEV tracks + 地图元素），替代默认的 tracks.json")
    _p.add_argument("--fps",        type=int, default=6)
    _p.add_argument("--dpi",        type=int, default=500,
                    help="渲染 DPI（越大越清晰但越慢，默认 500）")
    _p.add_argument("--figsize",    type=int, default=6,
                    help="图像尺寸（正方形，单位英寸，默认 6）")
    _p.add_argument("--bitrate",    type=str, default="10M",
                    help="输出视频码率")
    _p.add_argument("--xlim",       type=float, nargs=2, metavar=('MIN','MAX'), default=[-40.0, 40.0],
                    help="手动设置 X 轴范围")
    _p.add_argument("--ylim",       type=float, nargs=2, metavar=('MIN','MAX'), default=[-40.0, 40.0],
                    help="手动设置 Y 轴范围")
    _p.add_argument("--yaw-offset", type=float, default=-np.pi,
                    help="航向角偏移（弧度），默认 -π 使 0 指向 y 正前方")
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
        os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
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
        )