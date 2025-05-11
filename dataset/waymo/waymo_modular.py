import os
import random
import io
from pathlib import Path
import logging

import pandas as pd
from PIL import Image
import numpy as np
from numpy.linalg import inv, norm

import tensorflow as tf
logging.getLogger("tensorflow").setLevel(logging.ERROR)


from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2, submission_pb2
import depth_estimation
from utils.transforms import backproject_pixels_using_depth
from classes import class_averages_dict


def waymo_camera_image_iterator(root="."):
    root = Path(root)
    cam_dir = root / "camera_image"
    calib_dir = root / "camera_calibration"
    lidar_dir = root / "lidar_box"

    for cam_file in sorted(cam_dir.glob("*.parquet")):
        calib_file = calib_dir / cam_file.name
        lidar_file = lidar_dir / cam_file.name

        cam_df = pd.read_parquet(cam_file)
        calib_df = pd.read_parquet(calib_file)
        lidar_df = pd.read_parquet(lidar_file)

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

        for timestamp in cam_df['key.frame_timestamp_micros'].unique():

            lidar_boxes = lidar_df[
                (lidar_df['key.segment_context_name'] == name) &
                (lidar_df['key.frame_timestamp_micros'] == timestamp)
                ]

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

            yield images, K_list, ext_list, name, timestamp, lidar_boxes


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
    lidar_dir = root / "lidar_box"
    # camera_box_dir = root / "camera_box"
    projected_lidar_box_dir = root / "projected_lidar_box"

    cam_to_world = np.array([[0., 0., 1., 0.],
                             [-1., 0., 0., 0.],
                             [0., -1., 0., 0.],
                             [0., 0., 0., 1.]])


    depth_estimator = depth_estimation.DepthAnything(size="Large")

    parquet_files = list(cam_dir.glob("*.parquet"))

    if len(parquet_files) == 0:
        raise RuntimeError("DATASET DOES NOT CONTAIN PARQUET FILES. PLEASE CHECK PATH.")
    while True:

        cam_file = random.choice(parquet_files)

        cam_df = pd.read_parquet(cam_file)
        calib_df = pd.read_parquet(calib_dir / cam_file.name)
        lidar_df = pd.read_parquet(lidar_dir / cam_file.name)
        # camera_box_df = pd.read_parquet(camera_box_dir / cam_file.name)
        projected_lidar_box_df = pd.read_parquet(projected_lidar_box_dir / cam_file.name)


        lidar_df = lidar_df.merge(projected_lidar_box_df,
                                  on=["key.segment_context_name", "key.frame_timestamp_micros", "key.laser_object_id"],
                                  how='inner')

        cam_index_to_use = 1
        name = cam_df.iloc[0]['key.segment_context_name']
        random_timestamps = random.sample(list(cam_df['key.frame_timestamp_micros'].unique()), 10)

        # Iterate over the 10 random timestamps
        for timestamp in random_timestamps:
            frame_rows = cam_df[cam_df['key.frame_timestamp_micros'] == timestamp]

            # Pick one random camera view
            row = frame_rows[frame_rows["key.camera_name"] == cam_index_to_use].iloc[0]
            img = Image.open(io.BytesIO(row['[CameraImageComponent].image']))

            cache_path = Path("/media/manuel/T7/cache/DepthAnythingLarge") / f"{name}_{timestamp}.npy"

            if cache_path.exists():
                depth = np.load(cache_path)
                use_cached = True
            else:
                depth = depth_estimator(img)
                np.save(cache_path, depth[::8, ::8].numpy(force=True))
                use_cached = False

            # Load calibration for this camera
            # calib_row = calib_df[calib_df['key.camera_name'] == cam_index_to_use].iloc[0]
            # K = np.array([
            #     [calib_row['[CameraCalibrationComponent].intrinsic.f_u'], 0, calib_row['[CameraCalibrationComponent].intrinsic.c_u'], 0],
            #     [0, calib_row['[CameraCalibrationComponent].intrinsic.f_v'], calib_row['[CameraCalibrationComponent].intrinsic.c_v'], 0],
            #     [0, 0, 1, 0]
            # ])
            # ext = calib_row['[CameraCalibrationComponent].extrinsic.transform'].reshape(4, 4)

            lidar_boxes = lidar_df[
                (lidar_df['key.segment_context_name'] == name) &
                (lidar_df['key.frame_timestamp_micros'] == timestamp) &
                (lidar_df['key.camera_name'] == cam_index_to_use) &
                (lidar_df['[LiDARBoxComponent].type'].isin([1, 2, 4]))
            ]

            # Get the location estimate from depth
            u_array = np.array(lidar_boxes['[ProjectedLiDARBoxComponent].box.center.x']).astype(np.int32)
            v_array = np.array(lidar_boxes['[ProjectedLiDARBoxComponent].box.center.y']).astype(np.int32)

            if use_cached:
                u_array = u_array//8
                v_array = v_array//8

            estimated_z_at_centers = depth[v_array, u_array]

            # ones_array = np.ones_like(u_array)
            # uv1 = np.stack([u_array, v_array, ones_array], axis=0) # shape 3, N
            # depth_estimated_world_points = backproject_pixels_using_depth(uv1, inv(K[:3,:3]), ext@cam_to_world, depth).T # shape N,3
            # assert depth_estimated_world_points.shape[1] == 3

            width, height = img.size
            for row_idx, (_, box) in enumerate(lidar_boxes.iterrows()):
                box_center = np.array([
                    box['[LiDARBoxComponent].box.center.x'],
                    box['[LiDARBoxComponent].box.center.y'],
                    box['[LiDARBoxComponent].box.center.z']
                ])
                box_extend = np.array([
                    box['[LiDARBoxComponent].box.size.x'],
                    box['[LiDARBoxComponent].box.size.y'],
                    box['[LiDARBoxComponent].box.size.z']
                ])
                label = box['[LiDARBoxComponent].type']  # TYPE_VEHICLE = 1; TYPE_PEDESTRIAN = 2; TYPE_SIGN = 3; TYPE_CYCLIST = 4;
                object_label_str = "car" if label == 1 else "person" if label == 2 else "bicycle" if label == 4 else "NONE"


                # projected box to get the crop
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

                # 2: depth offset
                z_axis_distance_diff = box_center[0] - estimated_z_at_centers[row_idx].item()

                # 3: ext offset
                ext_diff = box_extend - class_averages_dict[object_label_str]

                # get angle diff
                angle_to_object = np.arctan2(box_center[1], box_center[0])
                heading_rad = box['[LiDARBoxComponent].box.heading'] - angle_to_object
                sin_theta = np.sin(heading_rad)
                cos_theta = np.cos(heading_rad)
                heading_diff = np.array([sin_theta, cos_theta])


                yield crop, object_label_str, z_axis_distance_diff, ext_diff, heading_diff




def make_inference_objects(context_name, timestamp, boxes, classes, scores):
  """Create objects based on inference results of a frame.

  Args:
    context_name: The context name of the segment.
    timestamp: The timestamp of the frame.
    boxes: A [N, 7] float numpy array that describe the inferences boxes of the
      frame, assuming each row is of the form [center_x, center_y, center_z,
      length, width, height, heading].
    classes: A [N] numpy array that describe the inferences classes. See
      label_pb2.Label.Type for the class values. TYPE_VEHICLE = 1;
      TYPE_PEDESTRIAN = 2; TYPE_SIGN = 3; TYPE_CYCLIST = 4;
    scores: A [N] float numpy array that describe the detection scores.

  Returns:
    A list of metrics_pb2.Object.
  """
  objects = []
  for i in range(boxes.shape[0]):
    x, y, z, l, w, h, heading = boxes[i]
    cls = classes[i]
    score = scores[i]
    objects.append(
        metrics_pb2.Object(
            object=label_pb2.Label(
                box=label_pb2.Label.Box(
                    center_x=x,
                    center_y=y,
                    center_z=z,
                    length=l,
                    width=w,
                    height=h,
                    heading=heading),
                type=label_pb2.Label.Type.Name(cls),
                id=f'{cls}_{i}'),
            score=score,
            context_name=context_name,
            frame_timestamp_micros=timestamp))
  return objects




def package_submission(prediction_objects, submission_file_base, desc):
    # Pack to submission.
    num_submission_shards = 4  # Please modify accordingly.

    if not os.path.exists(submission_file_base):
        os.makedirs(submission_file_base)
    sub_file_names = [
        os.path.join(submission_file_base, part)
        for part in [f'part{i}' for i in range(num_submission_shards)]
    ]

    submissions = [
        submission_pb2.Submission(inference_results=metrics_pb2.Objects())
        for i in range(num_submission_shards)
    ]

    obj_counter = 0
    for c_name, frames in prediction_objects.items():
        for timestamp, objects in frames.items():
            for obj in objects:
                submissions[obj_counter %
                            num_submission_shards].inference_results.objects.append(obj)
                obj_counter += 1

    for i, shard in enumerate(submissions):
        shard.task = submission_pb2.Submission.CAMERA_ONLY_DETECTION_3D
        shard.authors[:] = ['MV']  # Please modify accordingly.
        shard.affiliation = 'TODO'  # Please modify accordingly.
        shard.account_name = 'manu12121999@gmail.com'  # Please modify accordingly.
        shard.unique_method_name = 'mv_YOLO3D_def'  # Please modify accordingly.
        shard.method_link = 'todo'  # Please modify accordingly.
        shard.description = desc  # Please modify accordingly.
        shard.sensor_type = submission_pb2.Submission.CAMERA_ALL
        shard.number_past_frames_exclude_current = 0  # Please modify accordingly.
        shard.object_types[:] = [
            label_pb2.Label.TYPE_VEHICLE, label_pb2.Label.TYPE_PEDESTRIAN,
            label_pb2.Label.TYPE_CYCLIST
        ]
        with tf.io.gfile.GFile(sub_file_names[i], 'wb') as fp:
            fp.write(shard.SerializeToString())


def submission_to_bin(prediction_objects, name):
    # Pack to binary
    all_predictions = metrics_pb2.Objects()

    # Flatten the nested dictionary into a single list of metrics_pb2.Object
    for context_name in prediction_objects:
        for timestamp in prediction_objects[context_name]:
            all_predictions.objects.extend(prediction_objects[context_name][timestamp])

    # Write to binary file
    with tf.io.gfile.GFile(name, "wb") as f:
        f.write(all_predictions.SerializeToString())

    print("Saved predictions to predictions.bin")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mode = "train_iterator"

    if mode == "train_iterator":
        iterator = waymo_train_camera_image_iterator(root="/home/manuel/ma/DATASETS/waymo_3d_cam/val")

        from tqdm import tqdm
        for data in tqdm(iterator):
            #print(data)
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


