import torch
import numpy as np
from classes import BBox


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
        dets2.append(BBox(bbox_2d_list, cls_name, track_id=box.id, conf=box.conf))
    center_uv1 = np.array(center_uv1) # shape N,3
    return dets2, center_uv1