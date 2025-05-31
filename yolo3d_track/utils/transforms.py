import numpy as np
import torch

def compute_distances_from_z_values(z_values, u_array, v_array, f_u, f_v, c_u, c_v):
    x_values = (u_array - c_u) * z_values / f_u
    y_values = (v_array - c_v) * z_values / f_v
    distances = np.sqrt(x_values**2 + y_values**2 + z_values**2)
    return distances


def project_box_to_camera(box, K, extr):
    """

    :param box: np.array of shape (7,) where each row is [x, y, z, l, w, h, yaw]
    :param K: camera intrinsics matrix, np.array of shape (3, 3)
    :param extr: camera extrinsics matrix, np.array of shape (4, 4)
    :return: np.array of shape (2, 8) where each row is [u,v]
    """
    x, y, z, l, w, h, yaw = box

    # 3D bounding box corners in local object frame (centered at 0)
    corners = np.array([
        [l / 2, w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [-l / 2, -w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [l / 2, w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [-l / 2, -w / 2, h / 2],
        [-l / 2, w / 2, h / 2],
    ]).T  # shape: (3, 8)

    # Yaw rotation around Z axis

    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])  # shape: (3, 3)

    # Rotate and translate box to world coordinates
    corners_world = R_yaw @ corners + np.array([[x], [y], [z]])  # shape: (3, 8)

    # Convert to homogeneous coordinates for projection
    corners_world_hom = np.vstack((corners_world, np.ones((1, corners_world.shape[1]))))  # (4, 8)

    # Transform to camera coordinates
    corners_cam = np.linalg.inv(extr) @ corners_world_hom  # shape: (4, 8)
    corners_cam = corners_cam[:3, :]  # remove homogeneous row



    # Project into image plane using camera intrinsics
    projected = K @ corners_cam  # shape: (3, 8)
    projected[:2, :] /= projected[2, :]  # normalize by z (perspective division)
    if np.any(projected[2, :] <= 0):
        return None

    return projected[:2, :]  # (2, 8)


def backproject_pixels_using_depth(uv1, K_inv, ext, depth, dist_offsets=None):
    """
    Backprojects pixels to 3D world points.

    :param uv1: np.array shape (3, N)
    :param K_inv: np.array shape (3, 3)
    :param ext: np.array shape (4, 4)
    :param depth: torch.tensor OR np.array of shape (H, W)
    :return: world_points: np.array shape (3, N)
    """

    # Compute ray directions in camera space
    assert uv1.shape[0] == 3
    u,v = uv1[0], uv1[1]

    h, w = depth.shape
    assert np.all((0 <= v) & (v < h))
    assert np.all((0 <= u) & (u < w))
    if dist_offsets is None:
        dist_offsets = np.zeros(len(u))

    depth_values = [depth[vi, ui].item() + offset for ui, vi, offset in zip(u,v, dist_offsets)]  # add 1 to approximate middle
    uvz = uv1 * np.array(depth_values)[np.newaxis, :]
    unprojected = K_inv @ uvz  # shape: (3, N)

    # Add distance offsets:
    dir = unprojected / np.linalg.norm(unprojected, axis=0)[np.newaxis, :]
    unprojected += dir * dist_offsets[np.newaxis, :]

    # Transform rays to world space
    R = ext[:3, :3]                       # shape: (3, 3)
    t = ext[:3, 3:4]                      # shape: (3, 1)

    # Compute 3D world points
    points_world = R @ unprojected + t

    return points_world  # shape: (3, N)


def depth_to_pointcloud(depth: torch.Tensor, mask: (torch.Tensor|None),
                        fx: float, fy: float, cx: float, cy: float):
    """
    Convert depth image and mask to a point cloud using explicit intrinsics.

    Args:
        depth: (H, W) tensor of depth values
        mask: (H, W) tensor with True where depth is valid, False elsewhere (Optional)
        fx, fy: focal lengths
        cx, cy: principal point

    Returns:
        points: (N, 3) tensor of 3D points
    """
    if mask is None:
        mask = torch.ones_like(depth, dtype=torch.bool)

    device = depth.device
    H, W = depth.shape

    # Create pixel coordinate grid
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.float()
    y = y.float()

    # Flatten
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = depth.reshape(-1)
    mask_reshaped = mask.reshape(-1)

    # Apply mask
    x = x[mask_reshaped]
    y = y[mask_reshaped]
    z = z[mask_reshaped]

    # Back-project
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z

    points = torch.stack((X, Y, Z), dim=1)  # (N, 3)
    return points