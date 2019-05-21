import numpy as np


def coord_recover_xy(x, y, window_location: tuple, do_translation: bool = True
                     , do_scaling: bool = False, inverse: bool = False):
    points = np.asarray([[x, y]])
    result = coord_recover(points, window_location, do_translation=do_translation
                           , do_scaling=do_scaling, inverse=inverse)
    return result[0], result[1]


def coord_recover(points: np.ndarray, window_location: tuple, do_translation: bool = True
                  , do_scaling: bool = False, inverse: bool = False):
    """
        Scale first, then translate.
    :param points:
    :param window_location:
    :param do_translation:
    :param do_scaling:
    :param inverse:
    :return:
    """
    scale_factor, left_top_x, left_top_y, width, height = window_location
    if do_scaling:
        if inverse:
            points = points * scale_factor
        else:
            points = points / scale_factor

    if do_translation:
        # The find_contours returns points (y,x)
        translation = np.asarray([[left_top_y, left_top_x]])
        points = points + translation

    return points
