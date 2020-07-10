import numbers

import numpy as np
from typing import Tuple

from numpy.lib import stride_tricks


def valid_coordinate_rc(valid_tag: np.ndarray, dispose_flag: bool) -> Tuple[np.ndarray, np.ndarray]:
    valid_tag = np.atleast_2d(valid_tag)
    if dispose_flag:
        row_ind, col_ind = np.where(valid_tag)
    else:
        row_ind, col_ind = np.where(np.ones_like(valid_tag))
    return row_ind, col_ind


def flatten_patch_helper(patches: np.ndarray, patch_shape: Tuple[int, ...], squeeze_dim: Tuple[int, ...] = None):
    flatten: np.ndarray = np.atleast_1d(patches.reshape((-1,) + patch_shape))
    if squeeze_dim is not None:
        flatten = flatten.squeeze(axis=squeeze_dim)
    return flatten


def flatten_patch(patches, patch_shape, flatten_flag: bool, squeeze_dim: Tuple[int, ...] = None):
    if flatten_flag:
        patches = flatten_patch_helper(patches, patch_shape, squeeze_dim=squeeze_dim)
    return np.atleast_1d(patches)


def extract_patches_helper(arr, patch_shape=8, extraction_step=1):
    """Copied from SKLEARN
    Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : int or tuple of length arr.ndim.default=8
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : int or tuple of length arr.ndim, default=1
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return patches


def mask_mul_helper(img: np.ndarray, mask: np.ndarray):
    assert 0 <= img.ndim - mask.ndim <= 1
    if img.ndim > mask.ndim:
        mask = np.expand_dims(mask, axis=-1)
    return np.atleast_2d(img * mask)
