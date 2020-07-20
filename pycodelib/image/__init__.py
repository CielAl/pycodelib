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


def _pad_width_add_channel(img_dim, pad_width):
    assert img_dim >= len(pad_width)
    channel_dims = img_dim - len(pad_width)
    pad_width += ((0, 0), ) * channel_dims
    return pad_width


def _edge_pad_helper(img_dim, pad_size, direction: str = 'both'):
    pad_option_dict = {
        'both': (pad_size, ) * 2,
        'before': (pad_size, 0),
        'after': (0, pad_size),
    }
    pad_width = (pad_option_dict[direction],)
    return _pad_width_add_channel(img_dim, pad_width)


def edge_pad(array: np.ndarray, pad_size: int, direction: str = 'both', mode='constant', **kwargs):
    img_dim = array.ndim
    pad_width = _edge_pad_helper(img_dim, pad_size, direction=direction)
    return np.pad(array, pad_width=pad_width, mode=mode, **kwargs)


def non_overlap_pad(image: np.ndarray, patch_size, mode='constant', **kwargs):
    pad_size = patch_size - np.asarray(image.shape[: 2]) % patch_size
    pad_width_hw = ((0, pad_size[0]), (0, pad_size[1]))
    pad_width = _pad_width_add_channel(image.ndim, pad_width_hw)
    return np.pad(image, pad_width, mode, **kwargs)
