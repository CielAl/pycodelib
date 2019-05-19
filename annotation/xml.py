# from common import require_not_none
import numpy as np
import xml.etree.ElementTree as ElementTree


class AbstractElement(ElementTree.Element):
    NODE_NAME: str = None
    ATTRIB: dict = {}

    @property
    def current_id(self):
        return self.__id

    def increment_id(self):
        self.__id += 1

    def __init__(self, **extra):
        super().__init__(type(self).NODE_NAME, attrib=type(self).ATTRIB, **extra)
        self.__id = 1

    @classmethod
    def build(cls, **extra):
        return cls(**extra)


class Vertex(AbstractElement):
    NODE_NAME: str = "Vertex"

    @classmethod
    def build(cls, x: float, y: float):
        node = cls()
        node.set('X', str(x))
        node.set('Y', str(y))
        return node


class Vertices(AbstractElement):
    NODE_NAME: str = "Vertices"

    def add_vertex(self, v: Vertex):
        self.append(v)

    def add_coord(self, x: float, y: float):
        v = Vertex.build(x, y)
        self.add_vertex(v)

    def add_contour(self, contour: np.ndarray):
        assert contour.ndim == 2, "Contour must be 2d nd-array: # points x 2"
        assert contour.shape[1] == 2, "Contour must have only two columns: x and y"
        for point in contour:
            x = point[0]
            y = point[1]
            self.add_coord(x, y)


class Region(AbstractElement):
    NODE_NAME: str = "Region"

    def add_vertices(self, vertices: Vertices):
        self.append(vertices)

    def add_contour(self, contour: np.ndarray):
        vertices: Vertices = Vertices.build()
        vertices.add_contour(contour)
        self.add_vertices(vertices)


class RegionGroup(AbstractElement):
    NODE_NAME: str = "Regions"

    @classmethod
    def build(cls, **extra):
        node = cls(**extra)
        ElementTree.SubElement(node, 'RegionAttributeHeaders')
        return node

    def add_region(self, region: Region):
        self.append(region)
        region.set("ID", str(self.current_id))
        self.increment_id()


class Annotation(AbstractElement):
    NODE_NAME: str = "Annotation"
    ATTRIB: dict = {
        "LineColor": "62453",
    }
    TAG_CLASS: str = "PathClass"

    @property
    def regions(self) -> RegionGroup:
        return self.__regions

    def __init__(self, **extra):
        super().__init__(**extra)
        self.__regions = RegionGroup.build(**extra)
        self.append(self.__regions)

    @classmethod
    def build(cls, **extra):
        node = cls(**extra)
        # node.set
        return node

    def add_region(self, region: Region):
        self.regions.add_region(region)

    def add_contour(self, contour: np.ndarray, region_class: str = None):
        region: Region = Region.build()
        region.add_contour(contour)
        self.add_region(region)
        if region_class is not None:
            self.set(type(self).TAG_CLASS, region_class)


class AnnotationGroup(AbstractElement):
    NODE_NAME: str = "Annotations"
    ATTRIB: dict = {
        "MicronsPerPixel": "0.25"
    }

    def add_annotation(self, annotation: Annotation):
        self.append(annotation)
        annotation.set("ID", str(self.current_id))
        self.increment_id()

    def add_contour(self, contour: np.ndarray, region_class: str = None):
        annotation = Annotation.build()
        annotation.add_contour(contour, region_class=region_class)
        self.add_annotation(annotation)

