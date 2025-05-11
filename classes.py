import numpy as np

class BBox:
    def __init__(self, box_2d:list[int], detected_class:str, track_id=-1, conf=0.5):
        self.box_2d = box_2d
        self.detected_class = detected_class
        self.track_id = track_id
        self.conf = conf

class_averages_dict = {   #waymo labels don't differentiate between car, truck and bus
    "car":     np.array([4.0, 1.8, 1.6]),   # length, width, height
    "truck":   np.array([4.0, 1.8, 1.6]),
    "bus":     np.array([4.0, 1.8, 1.6]),
    "person":  np.array([0.8, 0.6, 1.75]),
    "bicycle": np.array([1.8, 0.6, 1.5]),
}

# class_averages_dict = {
#     "car":     np.array([4.0, 1.8, 1.6]),   # length, width, height
#     "truck":   np.array([10.0, 2.5, 3.5]),
#     "bus":     np.array([12.0, 2.6, 3.2]),
#     "person":  np.array([0.8, 0.6, 1.75]),
#     "bicycle": np.array([1.8, 0.6, 1.5]),
# }