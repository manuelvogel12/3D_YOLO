import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import ImageDraw, Image

from utils.transforms import project_box_to_camera


def plot(img, gt_bboxes, pred_boxes, K, extr, show=True, save_path=None):
    """
    Plot the image and a bev of the predicted and ground truth boxes.
    :param img: PIL Image
    :param gt_bboxes: np.array of shape (N, 7) where each row is [x, y, z, w, l, h, yaw]
    :param pred_boxes: np.array of shape (N, 7) where each row is [x, y, z, w, l, h, yaw]
    :param K: 3x3 intrinsic matrix
    param extr: 4x4 extrinsic matrix. If None, the image is assumed to be in the camera frame.
    :param show: whether to display the plot
    :param save_path: where to save the plot. If None, the plot is not saved.
    :return: Nothing
    """
    if plt.get_fignums():
        fig = plt.gcf()
        fig.clf()
    else:
        fig = plt.figure()

    axs = fig.subplots(1, 2)

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # --- Right: Bird’s Eye View (X-Y)
    ax = axs[1]
    ax.set_aspect('equal')
    ax.set_title("2D Bounding Boxes (BEV)")
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (sideways)")

    def draw_box_in_image(box, img, K, extr, color):
        draw = ImageDraw.Draw(img)
        projected_box_points = project_box_to_camera(box, K, extr)  # shape: (2, 8)

        if projected_box_points is not None:

            # Define box edges (pairs of indices into the 8 corners)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
            ]

            for start, end in edges:
                m_color = "blue" if (start, end) in [(0,1) or (4,5)] else color
                u1, v1 = projected_box_points[:, start]
                u2, v2 = projected_box_points[:, end]
                draw.line([(u1, v1), (u2, v2)], fill=m_color, width=5)

        return img



    def draw_box(box, color, label=None):
        x, y, z = box[:3]
        w, l, h = box[3:6]
        yaw = box[6]
        if not 0 < x < 60 or not -30 < y < 30:
            return

        # Box corners
        corners = np.array([
            [-w/2, -l/2],
            [-w/2,  l/2],
            [ w/2,  l/2],
            [ w/2, -l/2]
        ])

        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        rotated = (R @ corners.T).T + np.array([x, y])

        rect = patches.Polygon(rotated, closed=True, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Orientation arrow
        direction = R @ np.array([w/2, 0])
        ax.arrow(x, y, direction[0], direction[1], head_width=0.2, head_length=0.4, fc=color, ec=color)

        if label:
            # ax.text(x, y, label, color=color)
            ax.text(x, y, f"{z:.2}", color=color)

    # Draw GT boxes (green)
    for box in gt_bboxes:
        assert box.shape == (7,)
        draw_box(box, 'green', label='GT')
        img = draw_box_in_image(box, img, K, extr, color='green')

    # Draw predicted boxes (red)
    for box in pred_boxes:
        assert box.shape == (7,)
        draw_box(box, 'red', label='Pred')
        img = draw_box_in_image(box, img, K, extr, color='red')

    ax.grid(True)
    ax.set_xlim(0, 80)
    ax.set_ylim(-30, 30)

    # --- Left: Image
    axs[0].imshow(img)
    axs[0].set_title("Image")
    axs[0].axis("off")

    plt.tight_layout()

    if show:
        plt.pause(0.5)
    if save_path:
        plt.savefig(save_path)