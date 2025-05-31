import os
import random

from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from regressor import RegressionNN
from dataset.waymo.waymo_modular import waymo_train_camera_image_iterator
from utils.convertions import class_averages_dict

class WaymoRegressionDataset(Dataset):
    def __init__(self, root=".", transform=None, length=10000):
        """
        Wraps the generator as a PyTorch Dataset.
        `length` is an artificial length since the generator is infinite.
        """
        self.iterator = waymo_train_camera_image_iterator(root)
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        crop, class_name, dist_offset, extent_diff, heading_diff = next(self.iterator)

        if self.transform:
            crop = self.transform(crop)

        return {
            'image': crop,
            'class_name': class_name,
            'dist_offset': torch.tensor(dist_offset, dtype=torch.float32),
            'extent_diff': torch.tensor(extent_diff, dtype=torch.float32),
            'heading_diff': torch.tensor(heading_diff, dtype=torch.float32)
        }



class WaymoRegressionIterableDataset(IterableDataset):
    def __init__(self, root=".", transform=None):
        """
        An iterable dataset for the Waymo dataset. This allows infinite iteration.
        """
        self.root = root
        self.transform = transform

    def __iter__(self):
        # Create a generator here to yield samples one by one
        iterator = waymo_train_camera_image_iterator(self.root)
        for crop, class_name, dist_offset, extent_diff, heading_diff in iterator:
            if self.transform:
                crop = self.transform(crop)

            yield {
                'image': crop,
                'class_name': class_name,
                'dist_offset': torch.tensor(dist_offset, dtype=torch.float32),
                'extent_diff': torch.tensor(extent_diff, dtype=torch.float32),
                'heading_diff': torch.tensor(heading_diff, dtype=torch.float32)
            }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Config -----
    class_names = class_averages_dict.keys()
    batch_size = 16
    num_epochs = 15
    batches_per_epoch = 1000
    val_batches_per_epoch = 200
    lr = 1e-4
    save_path = "e4_regression_model.pt"

    # ----- Transforms -----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ----- Dataset & Dataloader -----
    train_dataset = WaymoRegressionIterableDataset(root="/media/manuel/T7/waymo_modular/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # , persistent_workers=True)

    val_dataset = WaymoRegressionIterableDataset(root="/home/manuel/ma/DATASETS/waymo_3d_cam/val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # , persistent_workers=True)

    # ----- Model, Loss, Optimizer -----
    model = RegressionNN(class_names).to(device)
    # resume_training = False
    # if resume_training:
    #     model.load_state_dict(torch.load("latest_regression_model.pt", map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # ----- Training Loop -----
    model.train()
    best_loss = float("inf")


    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_pos_loss = 0.0
        epoch_ext_loss = 0.0
        epoch_heading_loss = 0.0
        sum_dist_offset_preds =  0.0

        for i, batch in tqdm(enumerate(train_loader), total=batches_per_epoch):
            images = batch['image'].to(device)  # (B, C, H, W)
            class_names_batch = batch['class_name']  # list of strings
            dist_offsets_gt = batch['dist_offset'].to(device)  # (B, )
            extents_gt = batch['extent_diff'].to(device)  # (B, 3)
            headings_gt = batch['heading_diff'].to(device)  # (B, 2) sin, cos

            loss_batch = 0.0
            optimizer.zero_grad()


            # Forward pass for the entire batch
            preds = model(images, class_names_batch)
            dist_offset_preds, extent_preds, heading_preds = preds[:, 0], preds[:, 1:4], preds[:, 4:6]

            # Compute the loss for the entire batch
            loss_offset = mse_loss(dist_offset_preds, dist_offsets_gt) / 80
            loss_extent = mse_loss(extent_preds, extents_gt)
            loss_heading = mse_loss(heading_preds, headings_gt)

            loss = loss_offset + loss_extent + loss_heading
            loss.backward()
            optimizer.step()

            loss_batch += loss.item()
            epoch_pos_loss += loss_offset.item()
            epoch_ext_loss += loss_extent.item()
            epoch_heading_loss += loss_heading.item()

            sum_dist_offset_preds += dist_offset_preds.mean().item()

            epoch_loss += loss_batch
            if i >= batches_per_epoch:
                break

        avg_train_loss = epoch_loss / batches_per_epoch
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")
        print(f"Train pos loss: {epoch_pos_loss / batches_per_epoch:.4f}")
        print(f"Train ext loss: {epoch_ext_loss / batches_per_epoch:.4f}")
        print(f"Train heading loss: {epoch_heading_loss / batches_per_epoch:.4f}")

        print(f"Train avg DISTANCE offset preds: {sum_dist_offset_preds / batches_per_epoch:.4f}")

        torch.save(model.state_dict(), f"latest_{save_path}")
        print(f"Saved latest model to latest_{save_path}")

        # ----- Validation Loop -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j, val_batch in enumerate(val_loader):
                val_images = val_batch['image'].to(device)
                val_class_names = val_batch['class_name']
                val_dist_offsets_gt = val_batch['dist_offset'].to(device)
                val_extents_gt = val_batch['extent_diff'].to(device)
                val_headings_gt = val_batch['heading_diff'].to(device)

                val_preds = model(val_images, val_class_names)
                val_dist_offsets_preds, val_extent_preds, val_heading_preds = val_preds[:, 0], val_preds[:, 1:4], val_preds[:, 4:6]

                val_loss_offset = mse_loss(val_dist_offsets_preds, val_dist_offsets_gt) / 80
                val_loss_extent = mse_loss(val_extent_preds, val_extents_gt)
                val_loss_heading = mse_loss(val_heading_preds, val_headings_gt)

                val_loss += (val_loss_offset + val_loss_extent + val_loss_heading).item()

                if j >= val_batches_per_epoch:
                    break

        avg_val_loss = val_loss / val_batches_per_epoch
        print(f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {avg_val_loss:.4f}")
        model.train()


        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path}")

    print("Training finished.")
