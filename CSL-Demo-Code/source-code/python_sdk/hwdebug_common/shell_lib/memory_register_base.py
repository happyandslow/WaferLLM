import logging
from abc import ABC
from pathlib import Path
from typing import Optional

from cerebras.das_common.das_commonpybind import Dimensions
from cerebras.hal.halpybind import Rectangle, is_contained_in


class MemoryRegisterBase(ABC):
    '''
    Register base class
    Contains 2 helper functions to determine user rectangle and output file file name.
    '''

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._output_filename = None

    def create_rectangle(
        self, dim: Dimensions, input_rect: Optional[Rectangle],
    ) -> Rectangle:
        """
        \param dim: Largest possible dimension, it's should be the fabric
        dimension.
        \param input_rect: If not supplied, it will read the entire dimension.
        """
        assert dim.area()
        device_rect = Rectangle(0, 0, dim)

        # Modify the file name when user request for a smaller rectangle
        if input_rect is not None:
            # Expects dimension fits
            assert is_contained_in(input_rect, device_rect)
            device_rect = input_rect

        return device_rect

    def create_filename(
        self,
        prefix: str,
        workdir: Path,
        input_rect: Optional[Rectangle],
        input_name: Optional[str],
        symbol_name: Optional[str],
        address: Optional[int],
        length: Optional[int],
    ) -> str:
        """
        Sets the output _output_filename to "register-top.log" by default. The top gets replaced
        if either input_rect or input_name is supplied.

        \param input_rect is the input recthange. When supplied, use it to for suffix
        as output file to -x<X>y<Y>w<Width>h<Height>
        \param input_name is the input name. When supplied, use the input_name as a suffix
        as output file to -<input_name>. The input_name parameter supersedes input_rect.
        """
        suffix = "top"

        if input_name:
            suffix = input_name
        elif input_rect is not None:
            suffix = f"x{input_rect.x()}y{input_rect.y()}w{input_rect.width()}h{input_rect.height()}"

        if symbol_name:
            suffix = f"{suffix}_{symbol_name}"
        if address and address != -1:
            suffix = f"{suffix}_{address:04x}"
            if length and length != -1:
                end_address = address + length
                suffix = f"{suffix}_{end_address:04x}"

        return str(workdir / f"{prefix}-{suffix}.log")
