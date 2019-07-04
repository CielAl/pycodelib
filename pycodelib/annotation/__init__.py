import glob
import os
import re
import numpy as np
import cv2
from skimage import measure
from typing import List, Tuple, Dict
from .coordinates import coord_recover
from . import xml as axml
import xml.etree.ElementTree as ElementTree
from tqdm import tqdm
from xml.dom import minidom


class RoiMask(object):
    SEPARATOR: str = '_'
    DEFAULT_CLASS_LIST: List[str] = ['No path', 'BCC', 'SCC']

    def __init__(self, im_dir: str, class_list: List[str] = None,
                 do_translation: bool = True, do_scaling: bool = True, inverse_scaling: bool = True,
                 coordinate_fn=coord_recover):
        self.__im_dir = im_dir
        self.__mask = cv2.imread(self.__im_dir)
        if class_list is None:
            class_list = type(self).DEFAULT_CLASS_LIST
        self.__class_list = class_list
        self.do_translation = do_translation
        self.do_scaling = do_scaling
        self.inverse_scaling = inverse_scaling
        self.coordinate_fn = coordinate_fn

    @property
    def class_list(self):
        return self.__class_list

    @property
    def mask(self):
        return self.__mask

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
        assert index_match_array.size == 1 and index_match_array[0] < len(name_components) - 1,\
            "Invalid Class Information"
        index_match = index_match_array[0]
        return index_match

    def parse_name(self) -> Tuple[int, List[str]]:
        basename = os.path.basename(self.__im_dir)
        name_components = basename.split(type(self).SEPARATOR)
        # nonzero returns a tuple of array
        index_match = self.parse_name_helper(name_components)
        return index_match, name_components

    def extract_image_info(self):
        index_match, name_components = self.parse_name()
        window_location = tuple(float(x) for x in name_components[index_match + 1][1:-1].split(','))
        slide_name_basename = type(self).SEPARATOR.join(name_components[0:index_match])
        class_name = name_components[index_match]
        return slide_name_basename, class_name, window_location

    def get_contours(self, level=1, num_vertices=None) -> List[np.ndarray]:
        assert self.mask.ndim >= 2
        if self.mask.shape[-1] > 1:
            mask = self.mask[:, :, 0]
        else:
            mask = self.mask
        contours_list = measure.find_contours(mask, level=level)
        if num_vertices is not None:
            for idx, contour in enumerate(contours_list):
                contour_size = contour.shape[0]
                slice_step = max(1, contour_size // num_vertices)

                assert slice_step > 0, f"slice_step0:{str(contour_size)}|#:{len(contours_list)}"
                contours_list[idx] = contour[0::slice_step]
        return contours_list

    def get_annotations(self, num_vertices: int = 50) -> Tuple[List[axml.Annotation], str, str]:
        # one ROI may be broken down to multiple contours
        contours = self.get_contours(num_vertices=num_vertices)
        slide_name_basename, class_name, window_location = self.extract_image_info()
        annotation_list = []
        for old_contour in contours:
            annotation = axml.Annotation.build()
            new_contour = self.coordinate_fn(old_contour, window_location,
                                             self.do_translation, self.do_scaling, self.inverse_scaling)
            annotation.add_contour(new_contour, region_class=class_name)
            annotation_list.append(annotation)
        return annotation_list, slide_name_basename, class_name


class Extractor(object):
    NUM_VERTICES: int = 50

    @staticmethod
    def file_wildcard(path: str, extension: str) -> str:
        return os.path.join(path, f"*.{extension}")

    @staticmethod
    def file_list(path: str, extension: str) -> List[str]:
        wildcard = Extractor.file_wildcard(path, extension)
        return glob.glob(wildcard)

    def __init__(self, roi_dir, wsi_dir, export_dir, roi_ext='png', wsi_ext='ndpi',
                 do_translation=True, do_scaling=True, inverse_scaling=True,
                 coordinate_fn=coord_recover):
        self.roi_dir = roi_dir
        self.wsi_dir = wsi_dir
        self.export_dir = export_dir

        self.roi_extension = roi_ext
        self.wsi_extension = wsi_ext

        self.do_scaling = do_scaling
        self.do_translation = do_translation
        self.inverse_scaling = inverse_scaling
        # self.roi_wildcard = type(self).file_wildcard(self.roi_dir, self.roi_extension)
        # self.wsi_wildcard = type(self).file_wildcard(self.wsi_dir, self.wsi_extension)
        self.roi_list = type(self).file_list(self.roi_dir, self.roi_extension)
        self.wsi_list = type(self).file_list(self.wsi_dir, self.wsi_extension)
        self.coordinate_fn = coordinate_fn
        print("Mask location:", self.roi_dir, ' Extension:', self.roi_extension)
        print('Export to:', self.export_dir)
        print('Translate the Coordinates?', self.do_translation)
        print('Scaling the coordinates?', self.do_scaling)
        print('Is the scaling factor target/origin?', self.inverse_scaling)

    def extract(self) -> Dict[str, axml.AnnotationGroup]:
        annotation_dict: Dict[str, axml.AnnotationGroup] = {}
        for idx, roi_file in enumerate(tqdm(self.roi_list)):
            binary_mask: RoiMask = RoiMask(roi_file, do_translation=self.do_translation,
                                           do_scaling=self.do_scaling, inverse_scaling=self.inverse_scaling,
                                           coordinate_fn=self.coordinate_fn)
            result: Tuple[List[axml.Annotation], str, str] = \
                binary_mask.get_annotations(num_vertices=Extractor.NUM_VERTICES)
            annotation_list, slide_name_basename, class_name = result
            annotation_dict.setdefault(slide_name_basename, axml.AnnotationGroup.build())
            ann_group = annotation_dict.get(slide_name_basename)
            for annotation in annotation_list:
                ann_group.add_annotation(annotation)
        return annotation_dict

    def write(self, annotation_dict: Dict[str, axml.AnnotationGroup], pretty: bool = False):
        for slide_name_basename, annotation_group in annotation_dict.items():
            xml_name = f"{slide_name_basename}.xml"
            if not os.path.exists(self.export_dir):
                os.makedirs(self.export_dir)
            xml_fullname = os.path.join(self.export_dir, xml_name)
            with open(xml_fullname, 'w') as output_stream:
                if not pretty:
                    tree_root: ElementTree.ElementTree = ElementTree.ElementTree(annotation_group)
                    tree_root.write(output_stream, encoding='unicode')
                else:
                    xml_string = ElementTree.tostring(annotation_group).decode('utf-8')
                    xml_string_pretty = minidom.parseString(xml_string).toprettyxml(indent='\t')
                    output_stream.write(xml_string_pretty)
