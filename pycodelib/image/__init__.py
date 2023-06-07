import numpy as np
import skimage


def heatmap_overlay_rgb(img: np.ndarray, heatmap: np.ndarray):
    """

    Args:
        img:
        heatmap:

    Returns:

    """
    # if img.dtype == np.uint8 or img.max() > 1:
    #     img = img.astype(np.float32)
    #     img /= 255.
    img = skimage.img_as_float32(img)
    heatmap = skimage.img_as_float32(heatmap)
    overlaid = heatmap + img
    max_val = np.max(overlaid)
    overlaid /= max_val
    # overlaid *= 255
    # overlaid = np.uint8(overlaid)
    overlaid = skimage.img_as_uint(overlaid)
    return overlaid
