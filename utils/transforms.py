import numpy as np


def backproject_pixels_using_depth(uv1, K_inv, ext_inv, depth, z_offsets=None):
    """
    Backprojects pixels to 3D world points.

    :param uv1: np.array shape (3, N)
    :param K_inv: np.array shape (3, 3)
    :param ext_inv: np.array shape (4, 4)
    :param depth: torch.tensor OR np.array of shape (H, W)
    :return: world_points: np.array shape (3, N)
    """

    # Compute ray directions in camera space
    assert uv1.shape[0] == 3
    u,v = uv1[0], uv1[1]

    h, w = depth.shape
    assert np.all((0 <= v) & (v < h))
    assert np.all((0 <= u) & (u < w))
    if z_offsets is None:
        z_offsets = np.zeros(len(u))

    depth_values = [depth[vi, ui].item() + offset for ui, vi, offset in zip(u,v, z_offsets)]  # add 1 to approximate middle
    uvz = uv1 * np.array(depth_values)[np.newaxis, :]
    unprojected = K_inv @ uvz  # shape: (3, N)

    # Transform rays to world space
    R = ext_inv[:3, :3]                       # shape: (3, 3)
    t = ext_inv[:3, 3:4]                      # shape: (3, 1)

    # Compute 3D world points
    points_world = R @ unprojected + t

    return points_world  # shape: (3, N)
