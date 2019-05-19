import numpy as np


def coord_recover_xy(x, y, window_location):
    points = np.asarray([[x,y]])
    result = coord_recover(points, window_location)
    return result[0], result[1]


def coord_recover(points: np.ndarray, window_location: tuple):
    scale_to_original, left_top_x, left_top_y, width, height = window_location
    translation = np.asarray([[left_top_x, left_top_y]])
    return (points + translation) / scale_to_original
