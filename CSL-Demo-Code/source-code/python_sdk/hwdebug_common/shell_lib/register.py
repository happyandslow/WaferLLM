import logging
from typing import Optional

from cerebras.hal.halpybind import Rectangle
from cerebras.hwdebug_common.debug_lib.debugcommonpybind import read_registers
from cerebras.hwdebug_common.shell_lib.memory_register_base import (
    MemoryRegisterBase,
)


class Register(MemoryRegisterBase):
    """
    Register class
    read and write to registers
    """

    def __init__(self, logger: logging.Logger):
        super().__init__(logger)

    def read(
        self, cm_dev, core_file: Optional[str], input_rect: Rectangle,
    ):
        """
        Reads the config registers and output goes to stdout.
        """
        # `read_registers` accepts a string as a core_file name
        core_filename = ""
        if core_file is not None:
            core_filename = core_file

        read_registers(input_rect, core_filename, cm_dev, "")

    def write(self):
        pass
