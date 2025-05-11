import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import glob
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import PIL.Image

import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg11, ResNet18_Weights
from torchvision import transforms

from classes import class_averages_dict
from regressor import RegressionNN
from dataset.waymo import waymo_modular
from utils.visualize import plot
from depth_estimation import DepthAnything, Unik3D
from utils.convertions import ultralytics_to_BBox
from utils.transforms import backproject_pixels_using_depth

cfg = SimpleNamespace(
    track=True,
    show_result=True,
    save_result=True,
    yolo_model_name="yolo11x", # one in [yolo11n, yolo11s, yolo11m, yolo11l, yolo11x]
    output_path=None,
    cache_dir="cache",
    depth_estimation_name="DepthAnything", # one in [DepthAnything, Unik3D]
    depth_estimation_size="Large", # one in [Small, Base, Large]

)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(                 # Normalize based on ImageNet stats
        mean=[0.485, 0.456, 0.406],       # ImageNet mean for R, G, B channels
        std=[0.229, 0.224, 0.225]         # ImageNet std for R, G, B channels
    ),
])

cam2world = np.array([[0., 0., 1., 0.],
                      [-1., 0., 0., 0.],
                      [0., -1., 0., 0.],
                      [0., 0., 0., 1.]])


depth_estimator = eval(cfg.depth_estimation_name)(size=cfg.depth_estimation_size)
yolo_model = YOLO(cfg.yolo_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["car", "truck", "bus", "person", "bicycle"]
reg_models = RegressionNN(class_names=class_names).to(device)
reg_models.load_state_dict(torch.load("regression_model.pt", map_location=device))
reg_models.eval()


waymo_iterator = waymo_modular.waymo_camera_image_iterator(root="/home/manuel/ma/DATASETS/waymo_3d_cam/val")

# cache_dir = Path(cfg.cache_dir)
# cache_dir.mkdir(parents=True, exist_ok=True)

# loop images
for iteration_idx, data in tqdm(enumerate(iter(waymo_iterator))):
    images_pil, intrinsics, extrinsics, name, timestamp, lidar_boxes = data

    depth = depth_estimator(images_pil[0])

    # CACHING
    # cache_path = cache_dir / f"{name}_{timestamp}.npy"
    #
    # if cache_path.exists():
    #     depth = np.load(cache_path)
    # else:
    #     depth = depth_estimator(images_pil[0])
    #     np.save(cache_path, depth)

    # TODO: add class 7 (truck)
    if cfg.track:
        dets2_ul = yolo_model.track(source=images_pil[0], classes=[0, 2, 3, 5, 7], imgsz=640, device=0, conf=0.3, persist=True, verbose=False)[0]
    else: # mode=detect
        dets2_ul = yolo_model(images_pil[0], classes=[0, 2, 3, 5, 7], imgsz=640, device=0, conf=0.3, verbose=False)[0]

    dets2, center_uv1 = ultralytics_to_BBox(dets2_ul)
    if len(dets2) < 1:
        continue

    waymo_boxes = []
    waymo_classes = []
    waymo_scores = []

    z_offsets = np.zeros(len(dets2))
    rotation_offsets = np.zeros(len(dets2))
    extent_offsets = np.zeros((len(dets2), 3))


    # run inference on all image crops (TODO MAKE BATCH)
    with torch.no_grad():
        for det_index, det in enumerate(dets2):
            # det contains
            # det.box_2d  # list of int with len 4. Contains x1, y1, x2, y2
            # det.detected_class # str
            # det.track_id  # int or None
            # det.conf  # float
            det.detected_class = "car" if det.detected_class in ["truck", "bus"] else det.detected_class
            det.detected_class = "bicycle" if det.detected_class == "motorcycle" else det.detected_class

            # NN inference
            img_part = images_pil[0].crop(det.box_2d)
            img_part_torch = transform(img_part).to(device).unsqueeze(0)

            preds = reg_models(img_part_torch, [det.detected_class])[0]
            z_offset, extent_offset, sin_rotation, cos_rotation = preds[0], preds[1:4], preds[4], preds[5]

            z_offsets[det_index] = z_offset.item()
            rotation_offsets[det_index] = torch.atan2(sin_rotation, cos_rotation)
            extent_offsets[det_index] = extent_offset.numpy(force=True)

    # project the bounding box centers to world coordinates

    world_points = backproject_pixels_using_depth(center_uv1.T, K_inv=np.linalg.inv(intrinsics[0][:3, :3]), ext_inv=cam2world, depth=depth, z_offsets=z_offsets).T # shape N,3
    center_angles = np.arctan2(world_points[:, 1], world_points[:, 0])  # yaw angle from 0.0 to obj
    center_angles += rotation_offsets


    for det_index, det in enumerate(dets2):
        # combine regressed values with averages
        final_location = world_points[det_index]#  + location_offset.squeeze(0).numpy(force=True)
        final_extents = class_averages_dict[det.detected_class] + extent_offsets[det_index]
        final_rotation = center_angles[det_index]#  + rotation_offset.item()

        # get waymo prediction for submission
        waymo_boxes.append([*final_location, *final_extents, final_rotation]) # x,y,z,l,w,h,r
        waymo_class = 1 if det.detected_class in ['car', 'bus', 'truck'] else 2 if det.detected_class == "person" else 4 if det.detected_class == "bicycle" else 1
        waymo_classes.append(waymo_class)
        waymo_scores.append(det.conf.item())  #todo: change yolo conf to orientation/location conf?

    show_result = True
    save_result = False
    if show_result or save_result:
        gt_lidar_boxes_numpy = np.array(lidar_boxes[
                                         [
                                             "[LiDARBoxComponent].box.center.x",
                                             "[LiDARBoxComponent].box.center.y",
                                             "[LiDARBoxComponent].box.center.z",
                                             "[LiDARBoxComponent].box.size.x",
                                             "[LiDARBoxComponent].box.size.y",
                                             "[LiDARBoxComponent].box.size.z",
                                             "[LiDARBoxComponent].box.heading"
                                         ]
                                     ])

        plot(img=images_pil[0], gt_bboxes=gt_lidar_boxes_numpy , pred_boxes=np.array(waymo_boxes), show=show_result)


        # TODO
        # track_id_str = str(int(det.track_id.item())) if det.track_id is not None else "-1"
        # obj_info = {
        #     "frame_id": frame_id,
        #     "track_id": track_id,
        #     "object_class": "vehicle",
        #     "alpha": -10,
        #     "box_height": dim.tolist()[1], # z
        #     "box_width": dim.tolist()[0], # y
        #     "box_length": dim.tolist()[2], #x
        #     "box_center_x": location[2],
        #     "box_center_y": -location[0],
        #     "box_center_z": location[1],
        #     "box_heading": alpha,
        #     "box_speed" :0.0,
        # }
        # objects.append(obj_info)


