import numpy as np
from typing import Tuple
import logging
from sklearn.feature_extraction.image import extract_patches
from functools import reduce
from operator import mul


def extract_patch(image: np.ndarray,
                  patch_shape: Tuple[int, ...],
                  stride: int,
                  flatten: bool = True) -> np.ndarray:
    """
    Extract patches from image.
    Shape: If flatten:  N * H * W* C. \
        If not flatten: x * y * 1 * H * W, while N = x * y
    Args:
        image (): Input image to patchify
        patch_shape (): Shape of the patch.
        stride ():  Extraction step.
        flatten ():

    Returns:

    """
    assert not np.isscalar(image), f"does not support scalar input:{image}"
    if len(patch_shape) == 0:
        patch_shape = 1
    logging.debug(f'image_shape, {(image.shape, patch_shape)}')
    insufficient_size = (x < y for (x, y) in zip(image.shape, patch_shape))
    if any(insufficient_size):
        pad_size = tuple(
            max((y - x) / 2, 0)
            for (x, y) in zip(image.shape, patch_shape))

        pad_size = tuple((int(np.ceil(x)), int(np.floor(x))) for x in pad_size)
        image = np.pad(image, pad_size, 'wrap')
    patches = extract_patches(image, patch_shape, stride)
    if flatten:
        patches = patches.reshape((-1,) + patch_shape)
    return patches


def re_stitch(patches: np.ndarray, patch_size: Tuple[int, ...], dest_shape: Tuple[int, ...]):
    """
    Only works for non-overlapping patches.
    Returns:

    """
    dest_size = reduce(mul, dest_shape)
    assert dest_size == patches.size, f"Image pixel number disagrees:" \
        f" target = {dest_size}. original = {patches.size}"
    if (patches.ndim - len(patch_size)) == 1:
        extraction_step = np.asarray(patch_size)
        patch_indices_shape = tuple(((np.array(dest_shape) - np.array(patch_size)) //
                               np.array(extraction_step)) + 1)

        patches = patches.reshape(patch_indices_shape + patch_size)
    assert patches.ndim == len(patch_size) + 3, f"Unexpected patche group shape: {patches.ndim}"
    output = patches.swapaxes(1, 3).reshape(dest_shape)
    return output
