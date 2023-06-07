"""
Just playing around with codes.
"""
import logging


class Debugger(object):
    def __init__(self, name=None):
        self._logger = logging.getLogger(name)
        self.level = logging.DEBUG

    @staticmethod
    def multi_str(*msg):
        msg_str = ", ".join(tuple(str(x) for x in msg))
        return msg_str

    def log(self, *msg):
        self.log_lvl(self.level, *msg)

    def log_lvl(self, level, *msg):
        if level is None:
            return
        msg_str = type(self).multi_str(*msg)
        if level < 0:
            print(msg_str)
        else:
            self._logger.log(level=level, msg=msg_str)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = level
