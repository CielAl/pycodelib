from datetime import datetime
import numpy as np
import os
from typing import Dict, Union, Callable, Type
import torch
from torchnet.logger import VisdomLogger, VisdomPlotLogger
from torchnet.meter import AverageValueMeter
from torchvision.utils import make_grid
from torchvision.utils import save_image
import logging
dbg_logger = logging.getLogger(__name__)
dbg_logger.setLevel(logging.DEBUG)


def log_image(im_logger: VisdomLogger, img: torch.Tensor, batch_size: int = None):
    if batch_size is None:
        batch_size = img.shape[0]
    grid_img = make_grid(img.detach().cpu(), nrow=int(batch_size ** 0.5), normalize=True, range=(0, 1)).numpy()
    im_logger.log(grid_img)
    return grid_img


class SimpleLogger:
    DEFAULT_PORT: int = 8097
    DEFAULT_ENV_TITLE: str = 'CycleShuffle'
    LOSS_LOGGER_NAME: str = 'Loss'
    LOG_KEY_TITLE: str = 'title'
    LOG_KEY_LEGEND: str = 'legend'

    LOG_TYPE_LINE: str = 'line'
    LOG_TYPE_IMAGE: str = 'image'

    @staticmethod
    def time_stamp():
        """
        Add a time stamp as the suffix of the environment title.
        Returns:

        """
        return f"{datetime.now().strftime('%y/%m/%d|%H:%M:%S')}"

    @staticmethod
    def _default_env(env: str):
        """
        Default environment using DEFAULT_ENV_TITLE and time stamp, if given env is empty or none.
        Args:
            env: input environment title

        Returns:

        """
        if env is None or env == '':
            env = f"{SimpleLogger.DEFAULT_ENV_TITLE}_{SimpleLogger.time_stamp()}"
        return env

    @staticmethod
    def _update_legend(loss_logger: VisdomPlotLogger, new_legend: str):
        """
        Add legend to the plot logger if not exist.
        Args:
            loss_logger: a PlotLogger
            new_legend: legend string fot the PlotLogger (e.g. name of the lines)

        Returns:

        """
        assert loss_logger is not None
        existing_legends = loss_logger.opts.get(SimpleLogger.LOG_KEY_LEGEND, [])
        if new_legend in existing_legends:
            dbg_logger.warning('Legend already exists')
            return
        existing_legends.append(new_legend)
        loss_logger.opts[SimpleLogger.LOG_KEY_LEGEND] = existing_legends

    def _new_loss_logger(self) -> VisdomPlotLogger:
        """
        Create the VisdomPlotLogger in the given environment
        Returns:

        """
        return VisdomPlotLogger(SimpleLogger.LOG_TYPE_LINE, opts={
             SimpleLogger.LOG_KEY_TITLE: SimpleLogger.LOSS_LOGGER_NAME
                                                            }, env=self._env)

    def __init__(self, image_out: str, env: str = None, port=DEFAULT_PORT):
        self._env = SimpleLogger._default_env(env)
        self._port: int = port
        # always have one single loss logger, logged by multiple loss meter
        self._loss_meters_dict: Dict[str, AverageValueMeter] = dict()
        self.loss_logger: VisdomPlotLogger = self._new_loss_logger()
        # multiple image logger
        self._image_logger_dict: Dict[str, VisdomLogger] = dict()

        self._image_out = image_out
        os.makedirs(self._image_out, exist_ok=True)

    @staticmethod
    def _add_dict_helper(key: str,
                         which_dict: Dict,
                         obj_type: Type,
                         msg_if_exist: str,
                         constructor: Callable, *args, **kwargs):
        """
        Add meter to the dict. Create if not exist.
        Args:
            key:
            which_dict:
            obj_type:
            msg_if_exist:
            constructor: Constructor for the Meter if not already exist
            *args: positional args for constructor
            **kwargs: keyword args for the constructor

        Returns:
        True if successfully created. False if already exist or fail the type check.
        """
        existing = which_dict.get(key, None)
        if existing is not None and isinstance(existing, obj_type):
            if msg_if_exist is not None:
                dbg_logger.warning(msg_if_exist)
            return False
        which_dict[key] = constructor(*args, **kwargs)
        return True

    @staticmethod
    def _add_loss_meter_helper(which_dict: Dict,
                               which_logger: VisdomPlotLogger,
                               name: str, warning_msg: str = None,
                               constructor: Callable = AverageValueMeter,
                               *args,
                               **kwargs
                               ):
        """
        Wrapper to add the loss meter and add the legend
        Args:
            which_dict:
            which_logger:
            name:
            warning_msg:
            constructor: Callable to create the meter
            *args,
            **kwargs
        Returns:
        boolean flag of creating.
        """
        # todo type check and constructor
        success = SimpleLogger._add_dict_helper(name,
                                                which_dict,
                                                object,
                                                warning_msg,
                                                constructor,
                                                *args, **kwargs)
        if success:
            SimpleLogger._update_legend(which_logger,  name)
        return success

    def add_loss_meter(self, name: str, warning_msg: str = None):
        """
        Add an AverageValueMeter to the dict
        Args:
            name:
            warning_msg:

        Returns:

        """
        return SimpleLogger._add_loss_meter_helper(self._loss_meters_dict,
                                                   self.loss_logger,
                                                   name=name,
                                                   warning_msg=warning_msg,
                                                   constructor=AverageValueMeter)

    @staticmethod
    def get_dict_value(which_dict: Dict,
                       which_logger: VisdomPlotLogger,
                       key: str,
                       create_if_not_exist: bool,
                       constructor: Callable,
                       *args,
                       **kwargs):
        existing = which_dict.get(key, None)
        if existing is None and create_if_not_exist:
            # must update the legend
            SimpleLogger._add_loss_meter_helper(which_dict, which_logger, name=key,
                                                warning_msg=None, constructor=constructor, *args, **kwargs)

            existing = which_dict.get(key)
        return existing

    def log_loss(self, loss_name, loss_var: Union[np.ndarray, torch.Tensor]):
        which_meter = SimpleLogger.get_dict_value(self._loss_meters_dict,
                                                  self.loss_logger,
                                                  loss_name, True,
                                                  AverageValueMeter)
        if isinstance(loss_var, torch.Tensor):
            num_element = loss_var.numel()
            loss_var = loss_var.detach().cpu().data
        elif isinstance(loss_var, np.ndarray):
            loss_var = torch.from_numpy(loss_var)
            num_element = loss_var.size
        else:
            raise TypeError(f'Not Tensor or Numpy{type(loss_var)}')
        assert isinstance(loss_var, torch.Tensor)
        if num_element > 1:
            loss_var = loss_var.mean()
        which_meter.add(loss_var)

    def log_loss_group(self, loss_group: Dict[str, Union[np.ndarray, torch.Tensor]]):
        for loss_name, loss_var in loss_group.items():
            self.log_loss(loss_name, loss_var)

    def plot_loss(self, loss_name: str, x_val: float, reset: bool = True):
        loss_meter = self._loss_meters_dict[loss_name]
        self.loss_logger.log(x_val, loss_meter.value()[0], name=loss_name)
        if reset:
            loss_meter.reset()

    def plot_all_loss(self, epoch):
        for loss_name, loss_meter in self._loss_meters_dict.items():
            self.plot_loss(loss_name, x_val=epoch, reset=True)
            # self.loss_logger.log(epoch, loss_meter.value()[0], name=loss_name)
            # loss_meter.reset()

    def add_img_logger(self, title: str, warning_msg: str = f'Duplicate title detected'):
        opts = {
            SimpleLogger.LOG_KEY_TITLE: title,
                }
        success = SimpleLogger._add_dict_helper(title,
                                                self._image_logger_dict,
                                                VisdomLogger,
                                                warning_msg,
                                                VisdomLogger,
                                                plot_type=SimpleLogger.LOG_TYPE_IMAGE,
                                                opts=opts,
                                                env=self._env
                                                )

        return success

    def plot_image(self, title: str, images: torch.Tensor):
        opts: Dict[str, str] = {
            SimpleLogger.LOG_KEY_TITLE: title,
        }
        which_logger = SimpleLogger.get_dict_value(self._image_logger_dict,
                                                   self.loss_logger,
                                                   title,
                                                   True,
                                                   VisdomLogger,
                                                   plot_type=SimpleLogger.LOG_TYPE_IMAGE,
                                                   opts=opts,
                                                   env=self._env
                                                   )
        grids = log_image(which_logger, images)
        return grids

    def plot_image_dict(self, image_dict: Dict[str, torch.Tensor]):
        for image_key, image_tensor in image_dict.items():
            self.plot_image(title=image_key, images=image_tensor)

    def save_image_helper(self, images: torch.Tensor, epoch: int, data_name: str):
        img_file_name = f"{epoch}_{data_name}.png"
        fullname = os.path.join(self._image_out, img_file_name)
        save_image(images, fullname)

    def save_image_dict(self, epoch: int, images_dict: Dict[str, torch.Tensor]):
        for data_name, img in images_dict.items():
            self.save_image_helper(img, epoch, data_name)

    @property
    def loss_meter_dict(self):
        return self._loss_meters_dict

    def get_loss(self, loss_name):
        return self._loss_meters_dict[loss_name].value()[0]
