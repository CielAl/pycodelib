from typing import List, Dict
from common import require_not_none
import xml.etree.ElementTree as ElementTree


"""
    This is but a scratch for static typing.
    All but existing wheels.
"""


class Field(object):

    @property
    def tag(self) -> str:
        return self._tag

    def __init__(self, tag: str, value: object):
        self._tag = tag
        self.value = value

    @classmethod
    def build(cls, tag: str, value: object):
        require_not_none(tag, "Tag")
        require_not_none(value, "Value")
        return cls(tag=tag, value=value)


class Node(ElementTree.Element):

    @property
    def fields(self) -> List[Field]:
        return self._fields

    @property
    def node(self) -> str:
        return self._node

    @property
    def self_close(self) -> bool:
        return self._self_close

    def __init__(self, node: str, self_close: bool = False):
        super().__init__(str)
        self._node = node
        self._self_close = self_close
        self._fields: List[Field] = []
        self._tag_table: Dict[str, bool] = {}

    def __record_tag(self, tag: str):
        require_not_none(tag, "Tag")
        self._tag_table.set(tag, True)

    def __has_tag(self, tag: str) -> bool:
        """
        Note: tag is ensured to be not None by both Field.build and Node.__record_tag
        :param tag:
        :return:
        """
        return self._tag_table.get(tag) is not None

    def __verify_tag(self, tag: str, raise_on_none: bool = True) -> bool:
        is_tag_valid: bool = require_not_none(tag, "Tag", raise_error=not raise_on_none)
        success: bool = is_tag_valid and self.__has_tag(tag)
        return success

    def add_field_by_tag(self, tag: str, value: object, raise_on_none: bool = True) -> bool:
        return self.add_field(Field.build(tag, value), raise_on_none)

    def add_field(self, element: Field, raise_on_none: bool = True):
        success = self.__verify_tag(element.tag, raise_on_none=raise_on_none)
        if success:
            self.fields.append(element)
            self.__record_tag(element.tag)
        return success

    @classmethod
    def build(cls, node: str, self_close: bool = False, elements: List[Field] = None):
        node_obj = cls(node=node, self_close=self_close)
        for single_element in elements:
            node_obj.add_field(single_element, raise_on_none=True)
