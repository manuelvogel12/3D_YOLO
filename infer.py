import os
import sys
import time
import json
from collections import defaultdict
from pathlib import Path
import glob
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import PIL.Image
import pandas as pd

import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg11, ResNet18_Weights
from torchvision import transforms

from regressor import RegressionNN
from utils.visualize import plot
from depth_estimation import DepthAnything, Unik3D # noqa
from utils.convertions import ultralytics_to_BBox, class_averages_dict
from utils.transforms import backproject_pixels_using_depth

cfg = SimpleNamespace(
    img_folder=Path("/home/manuel/Repositories/3D_YOLO11/data_0a_sample"),
    extr_in_world_coordinates=False,
    track=True,
    show_result=True,
    save_result=False,
    save_to_track=True,
    yolo_model_name="yolo11x", # one in [yolo11n, yolo11s, yolo11m, yolo11l, yolo11x]
    output_path="infer",
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


prediction_objects_waymo = defaultdict(dict)

output_dir = Path(cfg.output_path)
output_dir.mkdir(parents=True, exist_ok=True)

objects = []
for idx, file_name in tqdm(enumerate(sorted(cfg.img_folder.iterdir()))):
    waymo_boxes = []
    waymo_classes = []
    waymo_scores = []

    with PIL.Image.open(file_name) as img_pil:
        img_pil.load()

        # TODO: load from data
        f_x, f_y, c_x, c_y = 1.112574951171875000e+03, 1.112437988281250000e+03, 4.881284790039062500e+02, 7.191560058593750000e+02

        intr = np.array([[f_x, 0, c_x],
                         [0, f_y, c_y],
                         [0, 0,   1]])

        extr = np.array([[ 1.01217465e-03, -5.66775445e-03,  9.99983430e-01, 1.51909995e+00],
                        [-9.99977112e-01, -6.69516949e-03,  9.74221097e-04, 2.58000009e-02],
                        [ 6.68953685e-03, -9.99961495e-01, -5.67440130e-03, 1.80649996e+00],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

        if cfg.extr_in_world_coordinates:
            extr = extr@cam2world


        frame_id = int(file_name.stem[:6])
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

        # project the bounding box centers to world coordinates
        world_points = backproject_pixels_using_depth(center_uv1.T, K_inv=np.linalg.inv(intr[:3, :3]), ext=extr, depth=depth, dist_offsets=dist_offsets).T # shape N,3
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

            obj_info = {
                "frame_id": frame_id,
                "track_id": int(det.track_id.item()),
                "object_class": det.detected_class,
                "alpha": -10,
                "box_height": float(final_extents[2]), # z
                "box_width": float(final_extents[1]), # y
                "box_length": float(final_extents[0]), #x
                "box_center_x": float(final_location[0]),
                "box_center_y": float(final_location[1]),
                "box_center_z": float(final_location[2]),
                "box_heading": float(final_rotation),
                "box_speed" :0.0,
            }
            print(obj_info)
            objects.append(obj_info)

    waymo_boxes, waymo_classes, waymo_scores = np.array(waymo_boxes), np.array(waymo_classes), np.array(waymo_scores)


    if cfg.show_result or cfg.save_result:

        save_path = output_dir / f"{idx}.png" if cfg.save_result else None
        plot(img=img_pil, gt_bboxes=np.array([]) , pred_boxes=waymo_boxes, K=intr[:3,:3], extr=extr, show=cfg.show_result, save_path=save_path)


    # SAVE RESULTS TO track_info.txt:
    if cfg.save_to_track:
        df = pd.DataFrame(objects)

        # filter out short-lived vehicles
        track_counts = df['track_id'].value_counts()
        valid_track_ids = track_counts[track_counts >= 100].index
        df = df[df['track_id'].isin(valid_track_ids)]

        df.to_csv('track_info.txt', index=False, sep=" ")


        # save track to track_camera_vis.json
        track_camera_vis = {track_id: {frame_id:[] for frame_id in range(0, 299 + 1)}
                            for track_id in valid_track_ids}

        for i, row in df.iterrows():
            track_id = row['track_id']
            frame_id = row['frame_id']
            track_camera_vis[track_id][frame_id] = [1]

        with open('track_camera_vis.json', 'w') as f:
            json.dump(track_camera_vis, f, indent=4)