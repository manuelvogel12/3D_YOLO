from abc import ABC, abstractmethod

from PIL import Image
import numpy as np
import torch
import logging


class DepthEstimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, image, **kwargs):
        pass


class DepthAnything(DepthEstimator):
    def __init__(self, size="Small"):
        super().__init__()

        from transformers import pipeline
        logging.getLogger("transformers").setLevel(logging.ERROR)
        # size options: "Small", "Base", "Large"
        self.pipe = pipeline(task="depth-estimation", model=f"depth-anything/Depth-Anything-V2-Metric-Outdoor-{size}-hf", use_fast=True)


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
    def __init__(self):
        super().__init__()
        from unik3d.models import UniK3D
        from unik3d.utils.camera import OPENCV, Pinhole

        # Move to CUDA, if any
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(self.device)

        # Load the RGB image and the normalization will be taken care of by the model
        # image_path = "input.jpg"
        # rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1)  # C, H, W


    def native_infer(self, torch_image, fx, fy, cx, cy):
        # torch_image: torch.Tensor shape (3, H, W)
        rgb = torch_image.to(self.device)

        cam_params = torch.tensor([fx,
                                   fy,
                                   cx,
                                   cy,
                                   0.000000000000000000e+00,
                                   0.000000000000000000e+00,
                                   0.000000000000000000e+00,
                                   0.000000000000000000e+00,
                                   0.000000000000000000e+00,
                                   0, 0, 0, 0, 0, 0, 0]).to(self.device)

        waymo_camera = OPENCV(cam_params)

        predictions = torch.clamp(self.model.infer(rgb, waymo_camera)["depth"],0,100)[0,0].cpu()
        return predictions.numpy()


    def __call__(self, image, fx, fy, cx, cy):

        match image:
            case Image.Image():
                torch_image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
            case np.ndarray():
                torch_image = torch.from_numpy(image).permute(2, 0, 1)
            case torch.Tensor():
                torch_image = image
            case _:
                raise TypeError(f"Unsupported image type: {type(image)}")

        depth = self.native_infer(torch_image, fx, fy, cx, cy)

        assert depth.ndim == 2
        return depth