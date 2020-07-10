from typing import Sequence
import logging
from functools import reduce
from operator import mul
from .helpers import *

from pycodelib.image.extraction.helpers import extract_patches_helper

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


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
    assert image.ndim - len(patch_shape) <= 1
    if image.ndim > len(patch_shape):
        # add the missing dimension into shape
        patch_shape = patch_shape + (image.ndim, )

    logger.debug(f'image_shape, {(image.shape, patch_shape)}')
    insufficient_size = (x < y for (x, y) in zip(image.shape, patch_shape))
    if any(insufficient_size):
        pad_size = tuple(
            max((y - x) / 2, 0)
            for (x, y) in zip(image.shape, patch_shape))
        pad_size = tuple((int(np.ceil(x)), int(np.floor(x))) for x in pad_size)
        # if len(pad_size) < image.ndim:
        #    pad_size = pad_size + ((0, 0), )
        image = np.pad(image, pad_size, 'wrap')
    patches = extract_patches_helper(image, patch_shape, stride)
    if flatten:
        patches = patches.reshape((-1,) + patch_shape)
    return patches


def mask_thresholded(patches_im: np.ndarray,
                     patches_mask: np.ndarray,
                     hw_shape: Sequence[int],
                     thresh: float,
                     dispose: bool,
                     mask_out: bool):
    """

    Args:
        patches_im:
        patches_mask:
        hw_shape:
        thresh:
        dispose:
        mask_out:
    Returns:

    """

    # type_order[-1] as the type of mask.
    # mask_axis as the dimension of single masks in the high dimensional mask patch-array.
    # excluded in the mean calculation
    mask_axis = tuple(
                        range(
                            -1*len(hw_shape),
                            0)
                    )

    assert len(mask_axis) > 0
    # Tissue screening by mask region.
    # patches_mask may be un-normalized
    valid_patch = (patches_mask > 0).mean(axis=mask_axis)

    valid_tag = valid_patch >= thresh
    # whether flatten and dispose the output.
    # Assume
    row_ind, col_ind = valid_coordinate_rc(valid_tag, dispose)

    logger.debug(f"after flatten: {patches_im.shape}")
    if dispose:
        # todo flatten
        # shape of item in valid_tag is scalar --> (), otherwise singleton axis emerges, and makes it
        # complicated when size of valid_tag is 1.

        patches_im = patches_im[valid_tag, :]
        patches_mask = patches_mask[valid_tag, :]

    # validate the length of image patch array and mask patch array
    assert patches_im.shape[0] == patches_mask.shape[0], f"Length mismatch" \
        f"{patches_im.shape[0]}, {patches_mask.shape[0]}"

    if mask_out:
        bool_mask = patches_mask.astype(bool)
        patches_im = mask_mul_helper(patches_im, bool_mask)
    return (patches_im, patches_mask, row_ind, col_ind), valid_tag


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
