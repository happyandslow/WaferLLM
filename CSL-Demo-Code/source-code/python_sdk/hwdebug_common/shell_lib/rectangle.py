import logging
import re

from cerebras.das_common.das_commonpybind import Dimensions
from cerebras.hal.halpybind import Rectangle, is_contained_in


class RectangleCommand:
    """
    The rectangle command is a setting command which lets the user sets the rectangle
    once for all their subsequent queries.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

        self._select_list = []

    def rectangle_list(self):
        return self._select_list
    
    def set_rect_to_device(self, dim: Dimensions):
        if dim.empty():
            self._logger.warning("Bad device rect %sx%s", dim.width(), dim.height())
            return

        device_rect = Rectangle(0, 0, dim)
        self._select_list = [("", device_rect)]

    def clear(self):
        """
        Clears the selected rectangle.
        """
        self._select_list = []

    def _parse_csv_pair(self, pair: str, err_msg: str = ""):
        ret = (None, None)

        res = re.match("(\d+),(\d+)", pair)
        if res is None:
            if err_msg != "":
                self._logger.warning(err_msg)
        else:
            ret = (int(res.group(1)), int(res.group(2)))

        return ret

    def select(self, rect: str, dim: str, device_dim: Dimensions, name: str = ""):
        """
        Selects a rectangle or set the current rectangle to new rectangle.
        Input is 4 integers separated by commas.
        For example: 1,2,3,4 stands for x=1, y=2, w=3, h=4
        """
        res = self._parse_csv_pair(rect, " Incorrect input. The expected format of RECT is <x>,<y>")
        if res[0] is None:
            return
        
        dim_res = self._parse_csv_pair(dim, " Incorrect input. The expected format of '-d is <width>,<height>")
        if dim_res[0] is None:
            return

        input_dim = Dimensions(dim_res[0], dim_res[1])
        input_rect = Rectangle(res[0], res[1], input_dim)

        if device_dim is None:
            self._logger.warning("no Fabric dimensions")
        else:
            device_rect = Rectangle(0, 0, device_dim)
            if not is_contained_in(input_rect, device_rect):
                self._logger.warning(
                    " Input rectangle is out of bounds. Fabric dimension: %d x %d",
                    device_dim.width(),
                    device_dim.height(),
                )
                return

        self._select_list.append((name, input_rect))

    def _remove_from_list(self, rect_list, rect):
        if not rect_list:
            return []
        
        new_list = []
        for r in rect_list:
            rc = r  # rect used for comparing
            if isinstance(r, tuple): # _select_list is a tuple (name, rect)
                rc = r[1]

            if (
                (
                    rect.width() != 0
                    and rc.x() == rect.x()
                    and rc.y() == rect.y()
                    and rc.width() == rect.width()
                    and rc.height() == rect.height()
                )
                or
                (
                    rc.x() == rect.x()
                    and rc.y() == rect.y()
                )
            ):
                self._logger.info(" Removing %s", r)
                continue
            
            new_list.append(r)
        
        return new_list

    def deselect(self, rect: str, dimension: str, all: bool):
        if not self._select_list:
            return

        if all:
            self._select_list = []
            return

        rect_res = self._parse_csv_pair(rect, " Incorrect input. The expected format of RECT is <x>,<y>")
        if rect_res[0] is None:
            return
        
        dim_res = self._parse_csv_pair(dimension)
        input_dim = Dimensions(0, 0)
        if dim_res[0] is not None:
            input_dim = Dimensions(dim_res[0], dim_res[1])

        input_rect = Rectangle(rect_res[0], rect_res[1], input_dim)

        self._select_list = self._remove_from_list(self._select_list, input_rect)

    def show(self):
        if not self._select_list:
            self._logger.info(" Rectangle: nothing selected")
            return
        self._logger.info(" Rectangle(s):")
        for rect in self._select_list:
            self._logger.info("  %s %s selected", rect[0], rect[1])
