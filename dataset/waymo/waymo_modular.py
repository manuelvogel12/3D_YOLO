import os
import random
import io
from pathlib import Path
import logging

import pandas as pd
from PIL import Image
import numpy as np
from numpy.linalg import inv, norm

import depth_estimation
from utils.transforms import backproject_pixels_using_depth, compute_distances_from_z_values, project_box_to_camera
from utils.convertions import class_averages_dict


def waymo_camera_image_iterator(root="."):
    root = Path(root)
    cam_dir = root / "camera_image"
    calib_dir = root / "camera_calibration"
    lidar_dir = root / "lidar_box"
    camera_box_dir = root / "lidar_camera_synced_box"

    for cam_file in sorted(cam_dir.glob("*.parquet")):

        cam_df = pd.read_parquet(cam_file)
        calib_df = pd.read_parquet(calib_dir / cam_file.name)
        lidar_df = pd.read_parquet(lidar_dir / cam_file.name)
        camera_box_df = pd.read_parquet(camera_box_dir / cam_file.name)

        name = cam_df.iloc[0]['key.segment_context_name']

        intrinsics_dict = {}
        extrinsics_dict = {}
        for _, row in calib_df.iterrows():
                f_x = row['[CameraCalibrationComponent].intrinsic.f_u']
                f_y = row['[CameraCalibrationComponent].intrinsic.f_v']
                c_x = row['[CameraCalibrationComponent].intrinsic.c_u']
                c_y = row['[CameraCalibrationComponent].intrinsic.c_v']

                intrinsics_dict[row['key.camera_name']] = np.array([
                    [f_x, 0, c_x, 0],
                    [0, f_y, c_y, 0],
                    [0,   0,   1, 0]
                ])

                ext_np = row['[CameraCalibrationComponent].extrinsic.transform'].reshape(4,4)
                extrinsics_dict[row['key.camera_name']] = ext_np

        # lidar DF needed for type, camera df for x,y,z
        joined_boxes = pd.merge(
            lidar_df[
                (lidar_df['key.segment_context_name'] == name) &
                (lidar_df['[LiDARBoxComponent].type'].isin([1, 2, 4]))
                ],
            camera_box_df[camera_box_df['key.segment_context_name'] == name],
            on=["key.laser_object_id", "key.segment_context_name", "key.frame_timestamp_micros"],
            how='inner'
        )

        for frame_num, timestamp in enumerate(cam_df['key.frame_timestamp_micros'].unique()):

            # if frame_num > 10:
            #     print("skipping to next scene")
            #     break

            gt_boxes = joined_boxes[joined_boxes['key.frame_timestamp_micros'] == timestamp]

            frame_rows = cam_df[cam_df['key.frame_timestamp_micros'] == timestamp]
            images = []
            K_list = []
            ext_list = []
            for _, row in frame_rows.iterrows():
                img = Image.open(io.BytesIO(row['[CameraImageComponent].image']))
                K = intrinsics_dict[row['key.camera_name']]
                ext = extrinsics_dict[row['key.camera_name']]
                images.append(img)
                K_list.append(K)
                ext_list.append(ext)

            yield images, K_list, ext_list, name, timestamp, gt_boxes


def waymo_train_camera_image_iterator(root="."):
    """
    Iterator over waymo training data.
    In each iteration, it returns:
    crop: PIL image of the cropped region
    object_label_str: str. One of "car", "pedestrian", "cyclist"
    box_offset: np.array of shape (3,) representing the offset of the box center from the center obtained by the predicted depth.
    ext_diff: np.array of shape (3,) the difference between the class average size and the size of the box.
    heading_diff: np.array of shape (3,). difference between the heading of the box and the angle to the object. Given in sin(theta) and cos(theta) format.
    """
    root = Path(root)
    cam_dir = root / "camera_image"
    calib_dir = root / "camera_calibration"
    # lidar_dir = root / "lidar_box"
    camera_box_dir = root / "lidar_camera_synced_box"
    # camera_box_dir = root / "camera_box"
    projected_lidar_box_dir = root / "projected_lidar_box"

    cam_to_world = np.array([[0., 0., 1., 0.],
                             [-1., 0., 0., 0.],
                             [0., -1., 0., 0.],
                             [0., 0., 0., 1.]])

    depth_estimator = depth_estimation.DepthAnything(size="Large")
    depth_estimator_name = depth_estimator.name

    parquet_files = list(cam_dir.glob("*.parquet"))

    if len(parquet_files) == 0:
        raise RuntimeError("DATASET DOES NOT CONTAIN PARQUET FILES. PLEASE CHECK PATH.")
    while True:

        cam_file = random.choice(parquet_files)

        cam_df = pd.read_parquet(cam_file)
        calib_df = pd.read_parquet(calib_dir / cam_file.name)
        # lidar_df = pd.read_parquet(lidar_dir / cam_file.name)
        camera_box_df = pd.read_parquet(camera_box_dir / cam_file.name)
        projected_lidar_box_df = pd.read_parquet(projected_lidar_box_dir / cam_file.name)


        camera_box_df = camera_box_df.merge(projected_lidar_box_df,
                                  on=["key.segment_context_name", "key.frame_timestamp_micros", "key.laser_object_id"],
                                  how='inner')
        cam_index_to_use = 1 # todo: make random
        name = cam_df.iloc[0]['key.segment_context_name']
        random_timestamps = random.sample(list(cam_df['key.frame_timestamp_micros'].unique()), 10)

        # Iterate over the 10 random timestamps
        for timestamp in random_timestamps:
            frame_rows = cam_df[cam_df['key.frame_timestamp_micros'] == timestamp]

            # Pick one random camera view
            row = frame_rows[frame_rows["key.camera_name"] == cam_index_to_use].iloc[0]
            img = Image.open(io.BytesIO(row['[CameraImageComponent].image']))

            # Load calibration for this camera
            calib_row = calib_df[calib_df['key.camera_name'] == cam_index_to_use].iloc[0]
            f_u, f_v, c_u, c_v = calib_row['[CameraCalibrationComponent].intrinsic.f_u'], calib_row['[CameraCalibrationComponent].intrinsic.f_v'], calib_row['[CameraCalibrationComponent].intrinsic.c_u'], calib_row['[CameraCalibrationComponent].intrinsic.c_v']
            K = np.array([
                [f_u, 0, c_u, 0],
                [0, f_v, c_v, 0],
                [0, 0, 1, 0]
            ])
            extr = calib_row['[CameraCalibrationComponent].extrinsic.transform'].reshape(4, 4)

            cache_path = Path(f"/media/manuel/T7/cache/{depth_estimator_name}") / f"{name}_{timestamp}.npy"
            #cache_path.parent.mkdir(exist_ok=True, parents=True)

            if cache_path.exists():
                try:
                    depth = np.load(cache_path)
                    use_cached = True
                except (EOFError, OSError):
                    depth = depth_estimator(img, f_u=f_u, f_v=f_v, c_u=c_u, c_v=c_v).numpy(force=True)
                    use_cached = False
            else:
                depth = depth_estimator(img, f_u=f_u, f_v=f_v, c_u=c_u, c_v=c_v).numpy(force=True)
                np.save(cache_path, depth[::8, ::8])
                use_cached = False

            downsampled_factor = 8 if use_cached else 1

            df_name_box = "[LiDARCameraSyncedBoxComponent].camera_synced_box" #old: "[LiDARBoxComponent].box"
            filtered_camera_box_df = camera_box_df[
                (camera_box_df['key.segment_context_name'] == name) &
                (camera_box_df['key.frame_timestamp_micros'] == timestamp) &
                (camera_box_df['key.camera_name'] == cam_index_to_use) &
                (camera_box_df['[ProjectedLiDARBoxComponent].type'].isin([1, 2, 4]))
            ]

            # Get the location estimate from depth
            u_array = np.array(filtered_camera_box_df['[ProjectedLiDARBoxComponent].box.center.x']).astype(np.int32)
            v_array = np.array(filtered_camera_box_df['[ProjectedLiDARBoxComponent].box.center.y']).astype(np.int32)


            depth_estimated_z_values = depth[v_array //downsampled_factor, u_array // downsampled_factor]
            depth_estimated_distances = compute_distances_from_z_values(depth_estimated_z_values, u_array, v_array, f_u, f_v, c_u, c_v)

            # ones_array = np.ones_like(u_array)
            # uv1 = np.stack([u_array//downsampled_factor, v_array//downsampled_factor, ones_array], axis=0) # shape 3, N
            # depth_estimated_world_points = backproject_pixels_using_depth(uv1, inv(K[:3,:3]), ext@cam_to_world, depth).T # shape N,3
            # assert depth_estimated_world_points.shape[1] == 3

            width, height = img.size
            for row_idx, (_, box) in enumerate(filtered_camera_box_df.iterrows()):
                box_center = np.array([
                    box[f'{df_name_box}.center.x'],
                    box[f'{df_name_box}.center.y'],
                    box[f'{df_name_box}.center.z']
                ])
                box_extend = np.array([
                    box[f'{df_name_box}.size.x'],
                    box[f'{df_name_box}.size.y'],
                    box[f'{df_name_box}.size.z']
                ])
                box_heading = box[f'{df_name_box}.heading']
                label = box['[ProjectedLiDARBoxComponent].type']  # TYPE_VEHICLE = 1; TYPE_PEDESTRIAN = 2; TYPE_SIGN = 3; TYPE_CYCLIST = 4;
                object_label_str = "vehicle" if label == 1 else "pedestrian" if label == 2 else "cyclist" if label == 4 else "NONE"


                # projected box to get the crop
                # projected_points = project_box_to_camera(np.array([*box_center, *box_extend, box_heading ]), K[:3,:3], extr@cam_to_world)
                # if projected_points is None:
                #     continue
                # x1 = np.min(projected_points[0, :])
                # x2 = np.max(projected_points[0, :])
                # y1 = np.min(projected_points[1, :])
                # y2 = np.max(projected_points[1, :])
                #

                # preprojected:
                cx = box['[ProjectedLiDARBoxComponent].box.center.x']
                cy = box['[ProjectedLiDARBoxComponent].box.center.y']
                w = box['[ProjectedLiDARBoxComponent].box.size.x']
                h = box['[ProjectedLiDARBoxComponent].box.size.y']

                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)

                crop = img.crop((x1, y1, x2, y2))


                ### Calculate Regression values (location, extends and angle diff)

                # 1: Location offset (not in use anymore)
                # box_loc_from_depth = depth_estimated_world_points[row_idx]
                # box_offset = box_loc_from_depth - box_center

                # 2: depth offset (difference between real center and estimated distance)
                box_center_camera_frame = extr[:3, :3].T @ (box_center - extr[:3, 3])
                distance_diff = norm(box_center_camera_frame) - depth_estimated_distances[row_idx].item()
                # print("real", norm(box_center_camera_frame), "estimated", depth_estimated_distances[row_idx].item())

                # 3: ext offset (difference between real extents and default ones)
                ext_diff = box_extend - class_averages_dict[object_label_str]

                # get angle diff
                angle_to_object = np.arctan2(box_center[1], box_center[0])
                heading_rad = box_heading - angle_to_object
                sin_theta = np.sin(heading_rad)
                cos_theta = np.cos(heading_rad)
                heading_diff = np.array([sin_theta, cos_theta])

                yield crop, object_label_str, distance_diff, ext_diff, heading_diff



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mode = "train_iterator"

    if mode == "train_iterator":
        iterator = waymo_train_camera_image_iterator(root="/home/manuel/ma/DATASETS/waymo_3d_cam/val")

        from tqdm import tqdm
        for data in tqdm(iterator):
            crop = data[0]
            crop.show()
            pass

    elif mode == "inf_iterator":
        iterator = waymo_camera_image_iterator(root="/home/manuel/ma/DATASETS/waymo_3d_cam/val")

        for images, calibs, extrinsics, name, timestamp, lidar_boxes in iterator:
            plt.clf()
            plt.imshow(images[0])
            plt.axis('off')
            plt.title("Waymo Camera Frame")
            plt.pause(0.2)

            print("Calibration:")
            print(calibs[0])


