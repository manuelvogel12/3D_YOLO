import torch
import numpy as np


ultralytics_class_names_to_waymo = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "cyclist",
    "person": "pedestrian",
}


class_averages_dict = {
    "vehicle":     np.array([4.0, 1.8, 1.6]),   # length, width, height
    "cyclist": np.array([1.8, 0.6, 1.5]),
    "pedestrian":  np.array([0.8, 0.6, 1.75]),
}



class BBox:
    def __init__(self, box_2d:list[int], detected_class:str, track_id=-1, conf=0.5):
        self.box_2d = box_2d
        self.detected_class = detected_class
        self.track_id = track_id
        self.conf = conf


def ultralytics_to_BBox(dets2_ul):
    dets2 = []
    center_uv1 = []
    for box in dets2_ul.boxes:
        cls_name = dets2_ul.names[box.cls.item()]
        bbox_2d_list = box.xyxy[0].to(torch.int32).flatten().tolist() # in order: x1, y1, x2, y2
        center_uv1.append([
            (bbox_2d_list[0] + bbox_2d_list[2]) // 2,  # between 0 and 1920
            (bbox_2d_list[1] + bbox_2d_list[3]) // 2,  # between 0 and 1280
            1])
        waymo_class_name = ultralytics_class_names_to_waymo[cls_name]
        dets2.append(BBox(bbox_2d_list, waymo_class_name, track_id=box.id, conf=box.conf))
    center_uv1 = np.array(center_uv1) # shape N,3
    return dets2, center_uv1