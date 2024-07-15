# import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_grasp(
        ax,
        grasp,
        color='r',
        arrow_len=4,
        arrow_head_len=2,
        arrow_head_width=3,
        arrow_width=1,
        jaw_len=3,
        jaw_width=1.0,
        grasp_center_size=0.01,
        grasp_center_style='+',
        grasp_axis_width=1,
        grasp_axis_style='--',
        show_center=False,
        show_axis=True,
        scale=1.5):
    """Plots a 2D grasp with arrow and jaw.

    Args:
        grasp: 2D grasp to plot.
        color: Color of plotted grasp.
        arrow_len: Length of arrow body.
        arrow_head_len: Length of arrow head.
        arrow_head_width: Width of arrow head.
        arrow_width: Width of arrow body.
        jaw_len: Length of jaw line.
        jaw_width: Line width of jaw line.
        grasp_center_size: Thickness of grasp center.
        grasp_center_style: Style of center of grasp.
        grasp_axis_width: Line width of grasp axis.
        grasp_axis_style: Style of grasp axis line.
        show_center: Whether or not to plot the grasp center.
        show_axis: Whether or not to plot the grasp axis.
    """
    # Plot grasp center
    if show_center:
        ax.scatter(grasp.center[0],
                   grasp.center[1],
                   c=color,
                   linewidths=scale * grasp_center_size)

    # Compute axis and jaw locations.
    axis = grasp.axis
    g1 = grasp.center - 0.5 * float(grasp.width_pixel) * axis
    g2 = grasp.center + 0.5 * float(grasp.width_pixel) * axis

    # Start location of grasp jaw 1.
    g1p = g1 - scale * arrow_len * axis
    # Start location of grasp jaw 2.
    g2p = g2 + scale * arrow_len * axis

    # Plot grasp axis.
    if show_axis:
        ax.plot([g1[0], g2[0]], [g1[1], g2[1]],
                color=color,
                linewidth=scale * grasp_axis_width,
                linestyle=grasp_axis_style)

    # Direction of jaw line.
    jaw_dir = scale * jaw_len * np.array([axis[1], -axis[0]])

    # Length of arrow.
    alpha = scale * (arrow_len - arrow_head_len)

    # Plot jaw 1.
    jaw_line1 = np.c_[g1 + jaw_dir, g1 - jaw_dir].T
    ax.plot(jaw_line1[:, 0],
            jaw_line1[:, 1],
            linewidth=scale * jaw_width,
            c=color)
    ax.arrow(g1p[0],
             g1p[1],
             alpha * axis[0],
             alpha * axis[1],
             width=scale * arrow_width,
             head_width=scale * arrow_head_width,
             head_length=scale * arrow_head_len,
             fc=color,
             ec=color)

    # Plot jaw 2.
    jaw_line2 = np.c_[g2 + jaw_dir, g2 - jaw_dir].T
    ax.plot(jaw_line2[:, 0],
            jaw_line2[:, 1],
            linewidth=scale * jaw_width,
            c=color)
    ax.arrow(g2p[0],
             g2p[1],
             -alpha * axis[0],
             -alpha * axis[1],
             width=scale * arrow_width,
             head_width=scale * arrow_head_width,
             head_length=scale * arrow_head_len,
             fc=color,
             ec=color)


def plot_grasp_on_image(image, grasps, grasp_quality=None):
    """Plot the final grasp overlaid on the input image.

    Args:
        image: The whole image captured from the camera.
        grasp: The chosen Grasp2D instance.
        grasp_quality: The corresponding grasp quality.
    """
    fig, ax = plt.subplots()
    plt.imshow(image, cmap=plt.cm.gray_r)
    for g in grasps:
        plot_grasp(ax, g, scale=1.5, show_center=False, show_axis=True)

    if grasp_quality:
        plt.title('Planned grasp on depth (Q=%f)' % (grasp_quality))

    plt.show()


def plot_sampled_grasps(grasp_images,
                        grasp_poses,
                        grasp_qualities,
                        num_sampled_grasps):
    """Plot sampled grasps.

    The cropped images corresponding to each sampled grasp will be plot in a
    square array.

    Args:
        grasp_images: A list or array of grasp images.
        grasp_poses: A list of array of grasp poses.
        grasp_qualities: The predicted grasp qualities.
        num_sampled_grasps:
    """
    plt.figure(figsize=(10, 10))

    sorted_indices = np.argsort(grasp_qualities.flatten())[::-1]
    d = int(np.ceil(np.sqrt(num_sampled_grasps)))
    for i in range(min(num_sampled_grasps, len(sorted_indices))):
        ind = sorted_indices[i]
        plt.subplot(d, d, i + 1)
        plt.imshow(grasp_images[ind].squeeze(axis=-1),
                   cmap=plt.cm.gray_r)

        # Visualize purposeless grasps.
        plt.title('id: %d\ndepth: %.2f\nquality: %.2f'
                  % (i + 1, grasp_poses[ind], grasp_qualities[ind]))

    plt.tight_layout()
    plt.show()
