from abc import ABC, abstractmethod
from typing import Union

from PIL import Image
import numpy as np
import torch
import logging

try:
    from unik3d.models import UniK3D
    from unik3d.utils.camera import OPENCV, Pinhole
except ImportError:
    print("Unik3D not installed. Please use DepthAnyhing as depth estimator.")


class DepthEstimator(ABC):
    def __init__(self, size="Small"):
        pass

    @abstractmethod
    def __call__(self, image: Union[Image, np.ndarray, torch.Tensor], **kwargs) -> torch.Tensor:
        pass


class DepthAnything(DepthEstimator):
    def __init__(self, size="Small"):
        super().__init__()

        from transformers import pipeline
        logging.getLogger("transformers").setLevel(logging.ERROR)
        # size options: "Small", "Base", "Large"
        self.pipe = pipeline(task="depth-estimation", model=f"depth-anything/Depth-Anything-V2-Metric-Outdoor-{size}-hf", use_fast=True)
        self.name = f"DepthAnything{size}"


    def __call__(self, image, **kwargs):
        match image:
            case Image.Image():
                depth = self.pipe(image)["predicted_depth"]
            case np.ndarray(): # in RGB format (
                depth = self.pipe(Image.fromarray(image))["predicted_depth"]
            case _:
                raise TypeError(f"Unsupported image type: {type(image)}")
        assert depth.ndim == 2
        return depth


class Unik3D(DepthEstimator):
    def __init__(self, size="Large"):
        super().__init__()

        # Move to CUDA, if any
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vit_version = "vitl" if size == "Large" else "vitb"
        self.model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-{vit_version}").to(self.device)
        self.model.resolution_level=9 # between 1 and 9, lower is faster, higher is (allegedly) more accurate
        self.name = f"Unik3D"

        # Load the RGB image and the normalization will be taken care of by the model
        # image_path = "input.jpg"
        # rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1)  # C, H, W


    def native_infer(self, torch_image, f_x, f_y, c_x, c_y):
        # torch_image: torch.Tensor shape (3, H, W)
        rgb = torch_image.to(self.device)

        cam_params = torch.tensor([f_x,
                                   f_y,
                                   c_x,
                                   c_y,
                                   0.000000000000000000e+00,
                                   0.000000000000000000e+00,
                                   0.000000000000000000e+00,
                                   0.000000000000000000e+00,
                                   0.000000000000000000e+00,
                                   0, 0, 0, 0, 0, 0, 0]).to(self.device)

        waymo_camera = OPENCV(cam_params)

        predictions = torch.clamp(self.model.infer(rgb, waymo_camera)["depth"],0,100)[0,0].cpu()
        return predictions


    def __call__(self, image, f_x, f_y, c_x, c_y, **kwargs):

        match image:
            case Image.Image():
                torch_image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
            case np.ndarray():
                torch_image = torch.from_numpy(image).permute(2, 0, 1)
            case torch.Tensor():
                torch_image = image
            case _:
                raise TypeError(f"Unsupported image type: {type(image)}")

        depth = self.native_infer(torch_image, f_x, f_y, c_x, c_y)

        assert depth.ndim == 2
        return depth