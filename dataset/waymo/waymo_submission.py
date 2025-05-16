import os
import tensorflow as tf

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2, submission_pb2

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
        shard.unique_method_name = '3D_YOLO_DAL_Y11m'  # Please modify accordingly.
        shard.method_link = 'https://github.com/manuelvogel12/3D_YOLO'  # Please modify accordingly.
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

    print(f"Saved predictions to {name}")

