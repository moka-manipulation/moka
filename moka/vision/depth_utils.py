import numpy as np
import scipy

from moka.vision import image_utils
from moka.utils.transformations import change_position_to_frame
from moka.utils.transformations import change_position_from_frame


transform = image_utils.transform
crop = image_utils.crop


def preprocess_depth_list(depth_list, return_m=True):
    all_depths = np.stack(depth_list, axis=0)
    all_depths = np.nan_to_num(
        all_depths, nan=1e7, posinf=1e7, neginf=-1e7
    )
    valid_mask = (all_depths < 5e3) & (all_depths > -1e4)
    #
    # valid_mask_size = valid_mask.shape[1] * valid_mask.shape[2]
    # for i in range(valid_mask.shape[0]):
    #     # if this mask is almost all zeros, then we should just skip it
    #     if valid_mask[i].sum() < 0.05 * valid_mask_size:
    #         print('skipping depth image {}'.format(i))
    #         valid_mask[i] = False

    all_depths = np.where(
        valid_mask, all_depths, np.zeros_like(all_depths)
    )
    avg_depth = all_depths.sum(0) / ((all_depths > 0).sum(0) + 1e-10)
    if return_m:
        return avg_depth / 1000.0
    else:
        return avg_depth


def downsample(data, ratio=0.5):
    factor = int(1 / ratio)
    return data[::factor, ::factor]


def upsample(data, ratio=0.5):
    factor = int(1 / ratio)

    new_data = np.zeros(
        (data.shape[0] * factor, data.shape[1] * factor, *data.shape[2:])
    )

    # Fill the new array with the original values
    new_data[::factor, ::factor] = data
    kernel_1d = scipy.signal.windows.boxcar(factor)
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    # Apply the kernel by convolution, seperately in each axis
    new_data = scipy.signal.convolve(new_data, kernel_2d, mode="same")
    return new_data


def inpaint(data):
    """Fills in the zero pixels in the depth image.

    Parameters:
        data: The raw depth image.

    Returns:
        new_data: The inpainted depth image.
    """
    # Form inpaint kernel.
    inpaint_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    cur_data = data.copy()
    EPS = 1e-5
    zeros = cur_data < EPS
    count = 0
    while np.any(zeros) and count < 100:
        neighbors = scipy.signal.convolve2d(
            (cur_data > EPS), inpaint_kernel, mode='same', boundary='symm'
        )
        avg_depth = scipy.signal.convolve2d(
            cur_data, inpaint_kernel, mode='same', boundary='symm'
        )
        avg_depth[neighbors > EPS] = (
            avg_depth[neighbors > EPS] / neighbors[neighbors > EPS]
        )
        avg_depth[neighbors < EPS] = 0.0
        avg_depth[data > EPS] = data[data > EPS]
        cur_data = avg_depth

        zeros = cur_data < EPS
        count += 1

    inpainted_data = cur_data

    new_data = np.copy(data)
    new_data[data < EPS] = inpainted_data[data < EPS]

    return new_data


def deproject_pixel_to_3d(
    depth, point, camera_intrinsics, camera_extrinsics=None,
    is_worldframe=False
):
    # depth = np.array(depth)
    point = np.array(point)
    camera_extrinsics = np.array(camera_extrinsics)

    camera_matrix = camera_intrinsics['cameraMatrix']
    x, y = point
    camera_xyz = np.linalg.inv(camera_matrix) @ np.array([x, y, 1])
    z = depth
    camera_xyz *= z
    # get coordinate in hand frame
    if is_worldframe:
        hand_xyz = change_position_to_frame(camera_xyz, camera_extrinsics)
        return hand_xyz
    else:
        return camera_xyz


def project_point_to_camera(point, camera_intrinsics, camera_extrinsics=None,
                            is_worldframe=False):
    """Projects a point cloud onto the camera image plane.

    Args:
        point: 3D point to project onto the camera image plane.

    Returns:
        pixel: 2D pixel location in the camera image.
    """
    point = np.array(point)
    camera_extrinsics = np.array(camera_extrinsics)
    intrinsics = camera_intrinsics['cameraMatrix']

    if is_worldframe:
        point = change_position_from_frame(point, camera_extrinsics)

    projected = np.dot(point, intrinsics.T)
    projected = np.divide(projected, np.tile(projected[..., 2:3], [3]))
    projected = np.round(projected)
    pixel = np.array(projected[..., :2]).astype(np.int16)
    return pixel


def threshold_gradients(data, threshold):
    """Get the threshold gradients.

    Creates a new DepthImage by zeroing out all depths
    where the magnitude of the gradient at that point is
    greater than threshold.

    Args:
        data: The raw depth image.
        threhold: A threshold for the gradient magnitude.

    Returns:
        A new DepthImage created from the thresholding operation.
    """
    data = np.copy(data)
    gx, gy = np.gradient(data.astype(np.float32))
    gradients = np.zeros([gx.shape[0], gx.shape[1], 2])
    gradients[:, :, 0] = gx
    gradients[:, :, 1] = gy
    gradient_magnitudes = np.linalg.norm(gradients, axis=2)
    ind = np.where(gradient_magnitudes > threshold)
    data[ind[0], ind[1]] = 0.0
    return data


def gamma_noise(data, gamma_shape=1000):
    """Apply multiplicative denoising to the images.

    Args:
        data: A numpy array of 3 or 4 dimensions.

    Returns:
        The corrupted data with the applied noise.
    """
    if data.ndim == 3:
        images = data[np.newaxis, :, :, :]
    else:
        images = data

    num_images = images.shape[0]
    gamma_scale = 1.0 / gamma_shape

    mult_samples = scipy.stats.gamma.rvs(
        gamma_shape, scale=gamma_scale, size=num_images
    )
    mult_samples = mult_samples[:, np.newaxis, np.newaxis, np.newaxis]
    new_images = data * mult_samples

    if data.ndim == 3:
        return new_images[0]
    else:
        return new_images


def gaussian_noise(data, prob=0.5, rescale_factor=4.0, sigma=0.005):
    """Add correlated Gaussian noise.

    Args:
        data: A numpy array of 3 or 4 dimensions.

    Returns:
        The corrupted data with the applied noise.
    """
    if data.ndim == 3:
        images = data[np.newaxis, :, :, :]
    else:
        images = data

    num_images = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]
    sample_height = int(image_height / rescale_factor)
    sample_width = int(image_width / rescale_factor)
    num_pixels = sample_height * sample_width

    new_images = []

    for i in range(num_images):
        image = images[i, :, :, 0]

        if np.random.rand() < prob:
            gp_noise = scipy.stats.norm.rvs(scale=sigma, size=num_pixels)
            gp_noise = gp_noise.reshape(sample_height, sample_width)
            gp_noise = scipy.misc.imresize(
                gp_noise, rescale_factor, interp='bicubic', mode='F'
            )
            image[image > 0] += gp_noise[image > 0]

        new_images.append(image[:, :, np.newaxis])

    new_images = np.stack(new_images)

    if data.ndim == 3:
        return new_images[0]
    else:
        return new_images
