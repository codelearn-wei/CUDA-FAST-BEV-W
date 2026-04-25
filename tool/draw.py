import argparse
import os
import sys
import types
import cv2
import numpy as np
import torch

CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

CLASS_COLORS = {
    0: (255, 158, 0),
    1: (255, 99, 71),
    2: (233, 150, 70),
    3: (255, 69, 0),
    4: (255, 140, 0),
    5: (112, 128, 144),
    6: (255, 61, 99),
    7: (220, 20, 60),
    8: (0, 0, 230),
    9: (47, 79, 79),
}


class _DataContainer:
    def __init__(self, data=None, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
        self._data = data
        self._stack = stack
        self._padding_value = padding_value
        self._cpu_only = cpu_only
        self._pad_dims = pad_dims

    @property
    def data(self):
        return self._data


class _LiDARInstance3DBoxes:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _Box3DMode:
    LIDAR = 0
    CAM = 1
    DEPTH = 2

    def __init__(self, value=None):
        self.value = value


def ensure_mmcv_parallel_stub():
    if "mmcv.parallel" in sys.modules:
        return

    mmcv_module = types.ModuleType("mmcv")
    parallel_module = types.ModuleType("mmcv.parallel")
    data_container_module = types.ModuleType("mmcv.parallel.data_container")
    parallel_module.DataContainer = _DataContainer
    data_container_module.DataContainer = _DataContainer
    mmcv_module.parallel = parallel_module
    sys.modules["mmcv"] = mmcv_module
    sys.modules["mmcv.parallel"] = parallel_module
    sys.modules["mmcv.parallel.data_container"] = data_container_module


def ensure_mmdet3d_stub():
    if "mmdet3d.core.bbox.structures.lidar_box3d" in sys.modules:
        return

    mmdet3d_module = types.ModuleType("mmdet3d")
    core_module = types.ModuleType("mmdet3d.core")
    bbox_module = types.ModuleType("mmdet3d.core.bbox")
    structures_module = types.ModuleType("mmdet3d.core.bbox.structures")
    lidar_box3d_module = types.ModuleType("mmdet3d.core.bbox.structures.lidar_box3d")
    box_3d_mode_module = types.ModuleType("mmdet3d.core.bbox.structures.box_3d_mode")

    lidar_box3d_module.LiDARInstance3DBoxes = _LiDARInstance3DBoxes
    box_3d_mode_module.Box3DMode = _Box3DMode

    mmdet3d_module.core = core_module
    core_module.bbox = bbox_module
    bbox_module.structures = structures_module
    structures_module.lidar_box3d = lidar_box3d_module
    structures_module.box_3d_mode = box_3d_mode_module

    sys.modules["mmdet3d"] = mmdet3d_module
    sys.modules["mmdet3d.core"] = core_module
    sys.modules["mmdet3d.core.bbox"] = bbox_module
    sys.modules["mmdet3d.core.bbox.structures"] = structures_module
    sys.modules["mmdet3d.core.bbox.structures.lidar_box3d"] = lidar_box3d_module
    sys.modules["mmdet3d.core.bbox.structures.box_3d_mode"] = box_3d_mode_module


def resolve_data_root(root):
    if os.path.exists(os.path.join(root, "example-data.pth")):
        return root

    nested = os.path.join(root, "example-data")
    if os.path.exists(os.path.join(nested, "example-data.pth")):
        return nested

    return root


def rotation_matrix_z(yaw):
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def boxes_to_corners_lidar(boxes):
    if len(boxes) == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)

    offsets = np.array(
        [
            [-1.0, -1.0, -1.0],
            [+1.0, -1.0, -1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, +1.0, +1.0],
            [-1.0, +1.0, +1.0],
        ],
        dtype=np.float32,
    )

    all_corners = []
    for box in boxes.astype(np.float32):
        x, y, z, w, l, h, yaw = box[:7]
        std_corners = np.empty_like(offsets)
        std_corners[:, 0] = w * offsets[:, 0] * 0.5
        std_corners[:, 1] = l * offsets[:, 1] * 0.5
        std_corners[:, 2] = h * offsets[:, 2] * 0.5

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        corners = np.empty_like(std_corners)
        corners[:, 0] = x + std_corners[:, 0] * cos_yaw + std_corners[:, 1] * sin_yaw
        corners[:, 1] = y + std_corners[:, 0] * -sin_yaw + std_corners[:, 1] * cos_yaw
        corners[:, 2] = z + std_corners[:, 2]
        all_corners.append(corners)

    return np.stack(all_corners, axis=0)


def read_txt_to_array(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    data_list = []
    for line in lines:
        elements = line.strip().split()
        data_list.extend(map(float, elements))

    data_arr = np.array(data_list)
    data_arr = data_arr.reshape(-1, 9)
    bboxes   = data_arr[:, :7]
    scores   = data_arr[:, 8]
    cls_arr  = data_arr[:, 7]

    return bboxes, scores, cls_arr


def class_name(class_id):
    class_id = int(class_id)
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"


def class_color(class_id):
    return CLASS_COLORS.get(int(class_id), (255, 255, 0))


def shade_color(color, factor):
    factor = float(np.clip(factor, 0.0, 1.0))
    return tuple(int(c * factor) for c in color)


def read_data_file(file_path):
    ensure_mmcv_parallel_stub()
    ensure_mmdet3d_stub()
    info_dict = {}
    data = torch.load(os.path.join(file_path, "example-data.pth"))
    info_dict["cams"] = {}
    j = 0

    for file in data['img_metas'].data[0][0]['img_info']:
        file_arr = file['filename'].split('/')[-2]
        info_dict["cams"][file_arr]={}
        info_dict["cams"][file_arr]['img_path'] = f"{file_path}/{j}-{file_arr[4:]}.jpg"
        
        lidar2img_extra = data['img_metas'].data[0][0]['lidar2img']["lidar2img_extra"][j]
        lidar2img_aug = data['img_metas'].data[0][0]['lidar2img']["lidar2img_aug"][j]
        
        R = lidar2img_extra["sensor2lidar_rotation"]
        T = lidar2img_extra["sensor2lidar_translation"]
        I = lidar2img_extra["cam_intrinsic"]
        info_dict["cams"][file_arr]['sensor2lidar_rotation'] = R
        info_dict["cams"][file_arr]['sensor2lidar_translation'] = T
        info_dict["cams"][file_arr]['cam_intrinsic'] = I
        info_dict["cams"][file_arr]['lidar2img'] = {}
        info_dict["cams"][file_arr]['lidar2img']['extrinsic'] = data['img_metas'].data[0][0]['lidar2img']['extrinsic'][j]
        info_dict["cams"][file_arr]['lidar2img']['intrinsic'] = data['img_metas'].data[0][0]['lidar2img']['intrinsic'][:3,:3]
        info_dict["cams"][file_arr]['post_tran'] = lidar2img_aug['post_tran']
        info_dict["cams"][file_arr]['post_rot'] = lidar2img_aug['post_rot']
        
        j = j + 1
        
        
    return info_dict


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array([[max_lumi, 0, max_lumi], 
                       [max_lumi, 0, 0],
                       [max_lumi, max_lumi, 0],
                       [0, max_lumi, 0], 
                       [0, max_lumi, max_lumi], 
                       [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank     = np.floor(gray * num_rank).astype(np.int32)
    diff     = (gray - rank / num_rank) * num_rank
    return tuple((colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous  = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1), dtype=points_lidar.dtype)], axis=1)
    
    lidar2camera              = camrera_info['lidar2img']['extrinsic']
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera             = points_camera_homogeneous[:, :3]
    valid                     = np.ones((points_camera.shape[0]), dtype=bool)
    valid                     = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera             = points_camera / points_camera[:, 2:3]
    camera2img                = camrera_info['lidar2img']['intrinsic']
    points_img                = points_camera @ camera2img.T
    post_aug                  = np.eye(3, dtype=np.float32)
    post_aug[:2, :2]          = camrera_info['post_rot'][:2,:2]
    post_aug[:2, 2]           = camrera_info['post_tran'][:2]
    points_img                = np.linalg.inv(post_aug) @ points_img.transpose(1,0)
    points_img                = points_img.transpose(1,0)[:, :2]
    return points_img, valid

def main(data_root, pred_path, vis_path, score_thr=0.2, draw_labels=False):
    data_root               = resolve_data_root(data_root)
    info_dict               = read_data_file(data_root)
    cam_info_dict           = info_dict['cams']

    bboxes, scores, cls_arr = read_txt_to_array(pred_path)

    # 定义绘制BEV视角下框的索引
    draw_boxes_indexes_bev      = [(0, 1), (1, 2), (2, 3), (3, 0)]
    draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), 
                                   (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    canva_size   = 1000
    show_range   = 50
    scale_factor = 4
    # 定义相机视角列表
    print('start visualizing results')
    pred_boxes        = np.array(bboxes, dtype=np.float32)
    corners_lidar     = boxes_to_corners_lidar(pred_boxes).reshape(-1, 3)

    pred_flag = np.ones((corners_lidar.shape[0] // 8, ), dtype=bool)
    scores    = np.array(scores, dtype=np.float32)
    cls_arr    = np.array(cls_arr, dtype=np.int32)

    # 构建预测框的标志数组
    sort_ids  = np.argsort(-scores)

    # 对相机视角进行可视化
    imgs = []
    views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for view in views:
            
        img = cv2.imread(cam_info_dict[view]['img_path'])
        

        # 将雷达坐标转换为图像坐标并绘制目标框
        corners_img, valid = lidar2img(corners_lidar, cam_info_dict[view])
        valid = valid.reshape(-1, 8)
        corners_img = corners_img.reshape(-1, 8, 2).astype(np.int32)
        for aid in sort_ids:
            if scores[aid] < score_thr and pred_flag[aid]:
                continue
            color = class_color(cls_arr[aid])
            line_color = shade_color(color, min(scores[aid] * 1.2, 1.0))
            for index in draw_boxes_indexes_img_view:
                if valid[aid, index[0]] and valid[aid, index[1]]:
                    cv2.line(img, corners_img[aid, index[0]], corners_img[aid, index[1]],
                            color=line_color, thickness=scale_factor)
            if draw_labels and valid[aid].any():
                anchor = corners_img[aid].min(axis=0)
                text = f"{class_name(cls_arr[aid])} {scores[aid]:.2f}"
                cv2.putText(img, text, (int(anchor[0]), int(anchor[1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA)
        imgs.append(img)

    # 构建BEV视图的画布
    canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)


    ## 绘制中心点和距离
    center_ego = (0, 0)
    center_canvas = int((center_ego[0] + show_range) / show_range / 2.0 * canva_size)
    cv2.circle(canvas, center=(center_canvas, center_canvas), radius=1, color=(255, 255, 255), thickness=0)
    dis = 10
    for r in range(dis, 100, dis):
        r_canvas = int(r / show_range / 2.0 * canva_size)
        cv2.circle(canvas, center=(center_canvas, center_canvas), radius=r_canvas, color=depth2color(r), thickness=0)
        
    
    # 绘制BEV视角下的预测框
    corners_lidar          = corners_lidar.reshape(-1, 8, 3)
    corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
    bottom_corners_bev     = corners_lidar[:, [0, 1, 2, 3], :2]
    bottom_corners_bev     = (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
    bottom_corners_bev     = np.round(bottom_corners_bev).astype(np.int32)
    center_bev             = corners_lidar[:, [0, 1, 2, 3], :2].mean(axis=1)
    head_bev               = corners_lidar[:, [0, 1], :2].mean(axis=1)
    canter_canvas          = (center_bev + show_range) / show_range / 2.0 * canva_size
    center_canvas          = canter_canvas.astype(np.int32)
    head_canvas            = (head_bev + show_range) / show_range / 2.0 * canva_size
    head_canvas            = head_canvas.astype(np.int32)

    # 在BEV视角下绘制预测框
    for rid in sort_ids:
        score = scores[rid]
        if score < score_thr and pred_flag[rid]:
            continue
        color = shade_color(class_color(cls_arr[rid]), min(score * 1.2, 1.0) if pred_flag[rid] else 1.0)
        for index in draw_boxes_indexes_bev:
            cv2.line(canvas, bottom_corners_bev[rid, index[0]], bottom_corners_bev[rid, index[1]],
                    color, thickness=2)
        cv2.line(canvas, center_canvas[rid], head_canvas[rid], color, 2, lineType=8)
        if draw_labels:
            text = f"{class_name(cls_arr[rid])} {score:.2f}"
            cv2.putText(canvas, text, (int(center_canvas[rid][0]) + 3, int(center_canvas[rid][1]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # 融合图像视角和BEV视角的结果
    img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3), dtype=np.uint8)
    img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
    img_back = np.concatenate([imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]], axis=1)
    img[900 + canva_size * scale_factor:, :, :] = img_back
    img = cv2.resize(img, (int(1600 / scale_factor * 3), int(900 / scale_factor * 2 + canva_size)))
    w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
    img[int(900 / scale_factor):int(900 / scale_factor) + canva_size, w_begin:w_begin + canva_size, :] = canvas

    # 保存可视化结果
    cv2.imwrite(vis_path, img)
    print(f'Saved visual result to {vis_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="example-data")
    parser.add_argument("--pred-path", default="model/resnet18int8/result.txt")
    parser.add_argument("--vis-path", default=None)
    parser.add_argument("--score-thr", type=float, default=0.2)
    parser.add_argument("--draw-labels", action="store_true")
    args = parser.parse_args()

    resolved_root = resolve_data_root(args.data_root)
    vis_path = args.vis_path or os.path.join(resolved_root, "sample0_vis.png")

    main(args.data_root, args.pred_path, vis_path, args.score_thr, args.draw_labels)
