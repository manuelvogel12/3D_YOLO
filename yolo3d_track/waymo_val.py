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

from regressor import RegressionNN
from dataset.waymo import waymo_modular
from dataset.waymo import waymo_submission
from utils.visualize import plot
from depth_estimation import DepthAnything, Unik3D # noqa
from utils.convertions import ultralytics_to_BBox, class_averages_dict
from utils.transforms import backproject_pixels_using_depth

cfg = SimpleNamespace(
    track=False,
    show_result=True,
    save_result=False,
    make_submission = True,
    yolo_model_name="yolo11x", # one in [yolo11n, yolo11s, yolo11m, yolo11l, yolo11x]
    output_path="output",
    # cache_dir="cache",
    depth_estimation_name="Unik3D", # one in [DepthAnything, Unik3D]
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

class_names = class_averages_dict.keys()
reg_models = RegressionNN(class_names=class_names).to(device)
reg_models.load_state_dict(torch.load("latest_regression_model.pt", map_location=device))
reg_models.eval()


waymo_iterator = waymo_modular.waymo_camera_image_iterator(root="/home/manuel/ma/DATASETS/waymo_3d_cam/val")
prediction_objects_waymo = defaultdict(dict)

output_dir = Path(cfg.output_path)
output_dir.mkdir(parents=True, exist_ok=True)
# cache_dir = Path(cfg.cache_dir)
# cache_dir.mkdir(parents=True, exist_ok=True)

# loop images
for iteration_idx, data in tqdm(enumerate(iter(waymo_iterator)), total=40000):
    # if iteration_idx > 2000:
    #     break

    waymo_boxes = []
    waymo_classes = []
    waymo_scores = []

    all_images_pil, all_intrinsics, all_extrinsics, name, timestamp, lidar_boxes = data
    # img_pil, intr, extr = all_images_pil[0], all_intrinsics[0], all_extrinsics[0]
    for img_pil, intr, extr in zip(all_images_pil, all_intrinsics, all_extrinsics):
        # using only image 0 for now (FRONT FACING)

        f_x, f_y = intr[0, 0], intr[1, 1]
        c_x, c_y = intr[0, 2], intr[1, 2]

        depth = depth_estimator(img_pil, f_x=f_x, f_y=f_y, c_x=c_x, c_y=c_y)

        if cfg.track:
            dets2_ul = yolo_model.track(source=img_pil, classes=[0, 1, 2, 3, 5, 7], imgsz=640, device=0, conf=0.2, persist=True, verbose=False)[0]  #, retina_masks=True
        else: # mode=detect
            dets2_ul = yolo_model(img_pil, classes=[0, 1, 2, 3, 5, 7], imgsz=640, device=0, conf=0.2, verbose=False)[0]

        dets2, center_uv1 = ultralytics_to_BBox(dets2_ul)
        if len(dets2) < 1:
            continue

        crops_torch = torch.stack([transform(img_pil.crop(det.box_2d)) for det in dets2]).to(device)
        det_classes = [det.detected_class for det in dets2]

        preds =  reg_models(crops_torch, det_classes)
        dist_offsets, extent_offsets, sin_rotations, cos_rotations = preds[:, 0], preds[:, 1:4], preds[:, 4], preds[:, 5]
        # shapes: (N,) (N,3) (N,) (N,)

        # move to numpy if necessary
        dist_offsets = dist_offsets.numpy(force=True)
        dist_offsets = np.array([1.5 if cl =="vehicle" else 0.1 for cl in det_classes]) # TODO: REMOVE
        # print("WARNING; USING DEFAULT VALUES")
        rotation_offsets = torch.atan2(sin_rotations, cos_rotations).numpy(force=True)
        extent_offsets = extent_offsets.numpy(force=True)
        # print(dist_offsets)

        # project the bounding box centers to world coordinates
        world_points = backproject_pixels_using_depth(center_uv1.T, K_inv=np.linalg.inv(intr[:3, :3]), ext=extr@cam2world, depth=depth, dist_offsets=dist_offsets).T # shape N,3
        center_angles = np.arctan2(world_points[:, 1], world_points[:, 0])  # yaw angle from 0.0 to obj
        center_angles += rotation_offsets


        for det_index, det in enumerate(dets2):
            # det contains
            # det.box_2d  # list of int with len 4. Contains x1, y1, x2, y2
            # det.detected_class # str
            # det.track_id  # int or None
            # det.conf  # float

            # combine regressed values with averages
            final_location = world_points[det_index]#  + location_offset.squeeze(0).numpy(force=True)
            final_extents = class_averages_dict[det.detected_class] + extent_offsets[det_index]
            final_rotation = center_angles[det_index]#  + rotation_offset.item()

            # get waymo prediction for submission
            waymo_boxes.append([*final_location, *final_extents, final_rotation]) # x,y,z,l,w,h,r
            waymo_class = 1 if det.detected_class == 'vehicle' else 2 if det.detected_class == "pedestrian" else 4 if det.detected_class == "cyclist" else -1
            if waymo_class == -1:
                raise RuntimeError("class not found")
            waymo_classes.append(waymo_class)
            waymo_scores.append(det.conf.item())  #todo: change yolo conf to orientation/location conf?

    waymo_boxes, waymo_classes, waymo_scores = np.array(waymo_boxes), np.array(waymo_classes), np.array(waymo_scores)

    if cfg.make_submission:
        prediction_objects_waymo[name][timestamp] = waymo_submission.make_inference_objects(name, timestamp, waymo_boxes, waymo_classes, waymo_scores)


    if cfg.show_result or cfg.save_result:
        # gt_lidar_boxes_numpy = np.array(lidar_boxes[
        #                                  [
        #                                      "[LiDARBoxComponent].box.center.x",
        #                                      "[LiDARBoxComponent].box.center.y",
        #                                      "[LiDARBoxComponent].box.center.z",
        #                                      "[LiDARBoxComponent].box.size.x",
        #                                      "[LiDARBoxComponent].box.size.y",
        #                                      "[LiDARBoxComponent].box.size.z",
        #                                      "[LiDARBoxComponent].box.heading"
        #                                  ]
        #                              ])

        gt_lidar_boxes_numpy = np.array(lidar_boxes[
                                         [
                                             "[LiDARCameraSyncedBoxComponent].camera_synced_box.center.x",
                                             "[LiDARCameraSyncedBoxComponent].camera_synced_box.center.y",
                                             "[LiDARCameraSyncedBoxComponent].camera_synced_box.center.z",
                                             "[LiDARCameraSyncedBoxComponent].camera_synced_box.size.x",
                                             "[LiDARCameraSyncedBoxComponent].camera_synced_box.size.y",
                                             "[LiDARCameraSyncedBoxComponent].camera_synced_box.size.z",
                                             "[LiDARCameraSyncedBoxComponent].camera_synced_box.heading"
                                         ]
                                     ])

        save_path = output_dir / f"{name}_{timestamp}.png" if cfg.save_result else None
        plot(img=img_pil, gt_bboxes=gt_lidar_boxes_numpy , pred_boxes=waymo_boxes, K=intr[:3,:3], extr=extr@cam2world, show=cfg.show_result, save_path=save_path)

if cfg.make_submission:
    waymo_submission.package_submission(prediction_objects_waymo, "Sub4_mv_3DYOLO", desc="3D YOLO with DepthAnything Large and yolo11x")
    waymo_submission.submission_to_bin(prediction_objects_waymo, "3DYOLO_sub4.bin")

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

# TO SUBMIT:
# tar cvf mv_3DYOLO.tar mv_3DYOLO
# gzip mv_3DYOLO.tar



# Testing  1: Depth Anything Large, YOLO11x
# 39987it [6:23:11,  1.74it/s]

# BINARY RES: BAD




