import glob,os,sys,re
import numpy as np
import cv2
from skimage import measure
from typing import List, Tuple


class RoiMask(object):
    SEPARATOR: str = '_'
    DEFAULT_CLASS_LIST: List[str] = ['No path', 'BCC', 'SCC']

    def __init__(self, im_dir: str, class_list: List[str] = None):
        self.__im_dir = im_dir
        self.__mask = cv2.imread(self.__im_dir)
        if class_list is None:
            class_list = type(self).DEFAULT_CLASS_LIST
        self.__class_list = class_list

    @property
    def class_list(self):
        return self.__class_list

    def parse_name_helper(self, name_components):
        index_match_array = np.asarray(
            [
                np.asarray(
                    [
                        re.search(class_name, component, re.IGNORECASE) is not None
                        for class_name in self.class_list
                    ]).any()
                for component in name_components
            ]
        ).nonzero()[0]
        assert index_match_array.size == 1 and index_match_array[0] < len(name_components) - 1\
            , "Invalid Class Information"
        index_match = index_match_array[0]
        return index_match

    def parse_name(self) -> Tuple[int, List[str]]:
        basename = os.path.basename(self.__im_dir)
        name_components = basename.split(type(self).SEPARATOR)
        # nonzero returns a tuple of array
        index_match = self.parse_name_helper(name_components)
        return index_match, name_components

    def get_contour(self, level=1, num_vertices=None) -> List[np.ndarray]:
        assert self.mask.ndim >= 2
        if self.mask.shape[-1] > 1:
            self = self.mask[:, :, 0]
        contours_list = measure.find_contours(self.mask[:, :, 0], level=level)
        if num_vertices is not None:
            for idx, contour in enumerate(contours_list):
                contour_size = contour.shape[0]
                contours_list[idx] = contour[0::contour_size // num_vertices]
        return contours_list


class Extractor(object):

    def __init__(self):
        pass