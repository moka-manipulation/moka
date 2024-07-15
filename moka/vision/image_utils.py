"""Utilities for RGB images.
"""
import cv2
import numpy as np
import scipy
import PIL.Image


def transform(data, translation, theta):
    """Create a new image by translating and rotating the current image.

    Args:
        translation: The XY translation vector.
        theta: Rotation angle in radians, with positive meaning
            counter-clockwise.

    Returns:
        An image of the same type that has been rotated and translated.
    """
    translation_map = np.float32(
        [[1, 0, translation[1]], [0, 1, translation[0]]]
    )
    translation_map_affine = np.r_[translation_map, [[0, 0, 1]]]

    theta = np.rad2deg(theta)
    rotation_map = cv2.getRotationMatrix2D(
        (data.shape[1] / 2, data.shape[0] / 2), theta, 1
    )
    rotation_map_affine = np.r_[rotation_map, [[0, 0, 1]]]

    full_map = rotation_map_affine.dot(translation_map_affine)
    full_map = full_map[:2, :]

    transformed_data = cv2.warpAffine(
        data,
        full_map,
        (data.shape[1], data.shape[0]),
        flags=cv2.INTER_NEAREST,
    )

    return transformed_data.astype(data.dtype)


def crop(data, height, width, c0=None, c1=None):
    """Crop the image centered around c0, c1.

    Args:
        height: The height of the desired image.
        width: The width of the desired image.
        c0: The center height point at which to crop. If not specified, the
            center of the image is used.
        c1: The center width point at which to crop. If not specified, the
            center of the image is used.

    Returns:
        A cropped Image of the same type.
    """
    # compute crop center px
    height = int(np.round(height))
    width = int(np.round(width))

    if c0 is None:
        c0 = float(data.shape[0]) / 2

    if c1 is None:
        c1 = float(data.shape[1]) / 2

    # crop using PIL
    desired_start_row = int(np.floor(c0 - float(height) / 2))
    desired_end_row = int(np.floor(c0 + float(height) / 2))
    desired_start_col = int(np.floor(c1 - float(width) / 2))
    desired_end_col = int(np.floor(c1 + float(width) / 2))

    pil_image = PIL.Image.fromarray(data)
    cropped_pil_image = pil_image.crop(
        (
            desired_start_col,
            desired_start_row,
            desired_end_col,
            desired_end_row,
        )
    )
    crop_data = np.array(cropped_pil_image)

    if crop_data.shape[0] != height or crop_data.shape[1] != width:
        raise ValueError('Crop dims are incorrect.')

    return crop_data.astype(data.dtype)


def inpaint(data, rescale_factor=1.0, window_size=3):
    """Fills in the zero pixels in the RGB image.

    Parameters:
        data: The raw image.
        rescale_factor: Amount to rescale the image for inpainting, smaller
            numbers increase speed.
        window_size: Size of window to use for inpainting.

    Returns:
        new_data: The inpainted imaga.
    """
    # Resize the image
    resized_data = scipy.misc.imresize(
        data, rescale_factor, interp='nearest'
    )

    # Inpaint smaller image.
    mask = 1 * (np.sum(resized_data, axis=2) == 0)

    inpainted_data = cv2.inpaint(
        resized_data, mask.astype(np.uint8), window_size, cv2.INPAINT_TELEA
    )

    # Fill in zero pixels with inpainted and resized image.
    filled_data = scipy.misc.imresize(
        inpainted_data, 1.0 / rescale_factor, interp='bilinear'
    )

    new_data = np.copy(data)
    new_data[data == 0] = filled_data[data == 0]

    return new_data
