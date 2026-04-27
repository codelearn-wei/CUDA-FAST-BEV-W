import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import json
import re
from collections import defaultdict

def parse_json_frames(frames_dir, swap_y_sign=True):
    """
    解析 frames 目录下的所有 tracks.json 文件
    返回 frame_data: dict {frame_index: [(track_id, x, y, l, w, yaw), ...]}
    其中 x 为前向坐标，y 为侧向坐标，l 为长度，w 为宽度，yaw 为偏航角（弧度）
    """
    frame_data = defaultdict(list)
    
    # 获取所有子文件夹，匹配 frame_xxxxx 模式
    subdirs = [d for d in os.listdir(frames_dir) if re.match(r'frame_\d+', d)]
    # 按数字排序
    subdirs.sort(key=lambda s: int(re.search(r'\d+', s).group()))
    
    for subdir in subdirs:
        # 提取帧序号
        frame_num = int(re.search(r'\d+', subdir).group())
        json_path = os.path.join(frames_dir, subdir, 'tracks.json')
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skip")
            continue
        
        with open(json_path, 'r') as f:
            tracks = json.load(f)
        
        for track in tracks:
            track_id = track['track_id']
            pos = track['position']        # [x, y, z]
            size = track['size']           # [l, w, h]
            yaw = track['yaw']
            
            x = pos[0]
            y = pos[1]
            l = size[0]    # 长度（前向）
            w = size[1]    # 宽度（侧向）
            
            if swap_y_sign:
                y = -y
                yaw = -yaw
            
            frame_data[frame_num].append((track_id, x, y, l, w, yaw))
    
    return frame_data

def rotated_rectangle_vertices(cx, cy, len_x, len_y, angle_rad):
    """返回矩形四个顶点坐标（原始坐标系）"""
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

def visualize_tracking_from_json(frames_dir, output_video_path, fps=6, dpi=100, figsize=(10, 10),
                                  swap_y_sign=True, swap_axes=True):
    """
    可视化跟踪结果（BEV图），从 frames_dir 读取每帧的 tracks.json
    swap_axes: 若为 True，则水平轴显示侧向 Y，垂直轴显示前向 X（即 X 上下显示）
    """
    # 解析所有帧数据
    frame_data = parse_json_frames(frames_dir, swap_y_sign=swap_y_sign)
    if not frame_data:
        print("No valid data found.")
        return

    frames = sorted(frame_data.keys())
    print(f"Total frames: {len(frames)}")

    # 收集坐标范围（基于旋转矩形的真实顶点，考虑交换轴）
    all_x = []   # 绘图水平轴坐标
    all_y = []   # 绘图垂直轴坐标
    for frame_id in frames:
        objects = frame_data[frame_id]
        for (track_id, x, y, l, w, yaw) in objects:
            # 计算原始坐标系下的四个顶点
            corners = rotated_rectangle_vertices(x, y, l, w, yaw)
            if swap_axes:
                # 绘图坐标：水平 = 原始 y，垂直 = 原始 x
                plot_corners = corners[:, [1, 0]]  # (y, x)
            else:
                plot_corners = corners
            all_x.extend(plot_corners[:, 0])
            all_y.extend(plot_corners[:, 1])

    if not all_x:
        return
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin_x = max(1.0, (x_max - x_min) * 0.1)
    margin_y = max(1.0, (y_max - y_min) * 0.1)
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y

    # 创建临时图获取尺寸
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    if swap_axes:
        ax.set_xlabel('Y (right) [m]')
        ax.set_ylabel('X (forward) [m]')
    else:
        ax.set_xlabel('X (forward) [m]')
        ax.set_ylabel('Y (right) [m]')
    ax.set_title('Tracking Results (BEV)')
    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    h_img, w_img = img.shape[:2]
    plt.close(fig)

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w_img, h_img))

    # 逐帧绘制
    for idx, frame_id in enumerate(frames):
        print(f"Rendering frame {idx+1}/{len(frames)} (frame_id={frame_id})", end='\r')
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        if swap_axes:
            ax.set_xlabel('Y (right) [m]')
            ax.set_ylabel('X (forward) [m]')
        else:
            ax.set_xlabel('X (forward) [m]')
            ax.set_ylabel('Y (right) [m]')
        ax.set_title(f'Frame {frame_id}')

        objects = frame_data[frame_id]
        for (track_id, x, y, l, w, yaw) in objects:
            # 计算原始矩形顶点
            corners = rotated_rectangle_vertices(x, y, l, w, yaw)
            if swap_axes:
                corners_swapped = corners[:, [1, 0]]  # (y, x)
            else:
                corners_swapped = corners

            poly = patches.Polygon(corners_swapped, closed=True,
                                   facecolor=get_color_for_id(track_id, 'tab20'),
                                   edgecolor='black', linewidth=1, alpha=0.6)
            ax.add_patch(poly)

            # 标注 ID，位置也要交换
            if swap_axes:
                text_x, text_y = y, x
            else:
                text_x, text_y = x, y
            ax.text(text_x, text_y, str(track_id), ha='center', va='center',
                    fontsize=6, color='white', weight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

        ax.grid(True, linestyle='--', alpha=0.5)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
        plt.close(fig)

    video_writer.release()
    print(f"\nVideo saved to {output_video_path}")

if __name__ == "__main__":
    # 获取本脚本所在的目录（例如 /home/.../CUDA-FastBEV/tools）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建 frames 目录和输出目录的绝对路径（相对于脚本所在目录的上一级）
    frames_directory = os.path.abspath(os.path.join(script_dir, "../outputs/frames"))
    output_directory = os.path.abspath(os.path.join(script_dir, "../outputs"))
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    output_video = os.path.join(output_directory, "tracking_result.mp4")
    
    if not os.path.exists(frames_directory):
        print(f"Directory not found: {frames_directory}")
        print(f"Please check that the path exists. Script is at: {script_dir}")
    else:
        visualize_tracking_from_json(frames_directory, output_video, fps=6, dpi=300, figsize=(12, 12),
                                      swap_y_sign=False, swap_axes=True)