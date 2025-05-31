import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class RegressorHead(nn.Module):
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1 + 3 + 2)  # offset (1) + extent (3) + heading (2)
        )

    def forward(self, x):
        out = self.head(x)

        # offset_z = out[:, 0]
        # extent = out[:, 1:4]
        # sin_theta = out[:, 4]
        # cos_theta = out[:, 5]
        # heading = torch.atan2(sin_theta, cos_theta)  # angle in radians
        return out # offset_z, extent, heading


class RegressionNN(nn.Module):
    def __init__(self, class_names):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # remove fc layer

        self.feature_dim = 512
        self.class_heads = nn.ModuleDict({
            cls_name: RegressorHead(self.feature_dim)
            for cls_name in class_names
        })

    def forward(self, x:torch.Tensor, class_names: list[str]):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        output = []
        # output_offset_z =[]
        # output_extent = []
        # output_heading = []

        # Iterate over class names for each sample in the batch
        for i, class_name in enumerate(class_names):
            class_head = self.class_heads[class_name]  # Get the head for the current class
            # o,e,h = class_head(features[i:i+1])
            # output_offset_z.append(o)
            # output_extent.append(e)
            # output_heading.append(h)
            output.append(class_head(features[i:i+1]))  # Apply class head to features[i]

        # Stack the results into a tensor
        # return torch.cat(output_offset_z), torch.cat(output_extent), torch.cat(output_heading)
        return torch.cat(output)


