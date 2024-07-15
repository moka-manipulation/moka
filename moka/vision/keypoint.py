import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# farthest point sampling
def fps(points, n_samples):
    """
    points: [N, 2] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype="int")  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float("inf")  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
            (points[last_added] - points[points_left]) ** 2
        ).sum(
            -1
        )  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(
            dist_to_last_added_point, dists[points_left]
        )  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]


def get_keypoints_from_segmentation(segmasks, num_samples=5, include_center=True):
    """
    Args:
        mask: a dict of segmentation masks, each mask is a dict with keys: 'mask', 'bbox', 'score'
        objects: a list of objects, each string is a query
    """
    object_vertices = {}
    for object_name in segmasks.keys():
        vertices = segmasks[object_name]["vertices"]
        mask = segmasks[object_name]["mask"]

        center_point = vertices.mean(0)
        # if mask[int(center_point[1])][
        #     int(center_point[0])
        # ] and include_center:  # ignore if geometric mean is not in mask
        if include_center:
            vertices = np.concatenate([center_point[None, ...], vertices], axis=0)

        if vertices.shape[0] > num_samples:
            kps = fps(vertices, num_samples)
        else:
            kps = vertices

        kps = np.concatenate([kps[[0]], kps[1:][kps[1:, 1].argsort()]], axis=0)
        object_vertices[object_name] = kps

    return object_vertices


def plot_keypoints(
    image,
    target_image_shape,
    object_vertices,
    objects,
    return_PIL=False,
    fname=None,
):
    """
    Args:
        image: PIL image
        mask: a dict of segmentation masks, each mask is a dict with keys: 'mask', 'bbox', 'score'
        objects: a list of objects, each string is a query
    """
    image = image.resize(target_image_shape, Image.LANCZOS)
    plt.imshow(image)
    w, h = target_image_shape

    total_kps = 0
    color_list = ["green", "red", "cyan", "magenta", "yellow", "red", "white", "blue"]
    for id, obj in enumerate(objects):
        kps = object_vertices[obj]
        for i in range(len(kps)):
            kp = kps[i]
            xytext = (
                min(max(20, kp[0]), w - 20),
                min(max(20, kp[1]), h - 20),
            )  # crop for display
            color = color_list[id % len(color_list)]
            plt.plot(kp[0], kp[1], color=color, marker="o", markersize=6)
            plt.annotate(str(total_kps), kp, xytext, size=12)
            total_kps += 1

    plt.axis('off')
    plt.plot()

    if fname is not None:
        plt.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0)
    else:
        # save to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, transparent=True, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image = Image.open(buf)
        return image
