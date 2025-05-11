import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


def plot(img, gt_bboxes, pred_boxes, show=True, save_path=None):
    """
    Plot the image and a bev of the predicted and ground truth boxes.
    :param img: PIL Image
    :param gt_bboxes: np.array of shape (N, 7) where each row is [x, y, z, w, l, h, yaw]
    :param pred_boxes: np.array of shape (N, 7) where each row is [x, y, z, w, l, h, yaw]
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

    # --- Left: Image (convert BGR to RGB)
    axs[0].imshow(img)
    axs[0].set_title("Image")
    axs[0].axis("off")

    # --- Right: Bird’s Eye View (X-Y)
    ax = axs[1]
    ax.set_aspect('equal')
    ax.set_title("2D Bounding Boxes (BEV)")
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (sideways)")

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

        rect = patches.Polygon(rotated, closed=True, linewidth=2,
                               edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Orientation arrow
        direction = R @ np.array([w/2, 0])
        ax.arrow(x, y, direction[0], direction[1],
                 head_width=0.2, head_length=0.4, fc=color, ec=color)

        if label:
            ax.text(x, y, label, color=color)

    # Draw GT boxes (green)
    for box in gt_bboxes:
        assert box.shape == (7,)
        draw_box(box, 'green', label='GT')

    # Draw predicted boxes (red)
    for box in pred_boxes:
        assert box.shape == (7,)
        draw_box(box, 'red', label='Pred')

    ax.grid(True)
    ax.set_xlim(0, 80)
    ax.set_ylim(-30, 30)

    plt.tight_layout()

    if show:
        plt.pause(0.5)
    if save_path:
        plt.savefig(save_path)