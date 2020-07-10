import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def heatmap_overlay_rgb(img: np.ndarray, heatmap: np.ndarray):
    """

    Args:
        img:
        heatmap:

    Returns:

    """
    # noinspection PyArgumentList
    if img.dtype == np.uint8 or img.max() > 1:
        img = img.astype(np.float32)
        img /= 255.
    overlaid = heatmap + img
    max_val = np.max(overlaid)
    overlaid /= max_val
    overlaid *= 255
    overlaid = np.uint8(overlaid)
    return overlaid
