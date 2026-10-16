import logging
from typing import List, Optional
import cerebras.message.messagepybind as Message

from cerebras.hal.halpybind import (
    Rectangle,
    ReadItem
)
from cerebras.das_common.das_commonpybind import (
    Coordinate,
    Dimensions
)
from cerebras.hwdebug_common.debug_lib.debugcommonpybind import (
    read_memory,
    write_memory,
    write_memory_rect_list
)
from cerebras.hwdebug_common.shell_lib.memory_register_base import (
    MemoryRegisterBase,
)


class Memory(MemoryRegisterBase):
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def read(
        self,
        cm_dev,
        core_file: Optional[str],
        rect,
        address: int,
        length: int,
        symbol_name: str,
        symbol_all: bool,
        elf_symbol
    ):
        """
        Reads the PE memory into file name memory-<input_name>.log
        If input_rect is not supplied, it will read the entire dimension
        supplied. The file name will be memory-top.log
        """

        # `read_memory` accepts a string as a core_file name
        read_list = None
        core_filename = ""
        if core_file is not None:
            core_filename = core_file

        if symbol_name:
            read_list = self.read_list_from_name(elf_symbol, symbol_name)
        elif symbol_all:
            read_list = self.read_list_all(elf_symbol)
        else:
            assert address >= 0, "Expects address greater than or equal to 0."
            assert length > 0, "Expects length greater than 0."
            read_list = self.read_list_from_addr(address, length)
        
        if not read_list:
            return

        read_memory(rect, core_filename, cm_dev, read_list, "")

    def read_list_from_name(self, elf_symbol, symbol_name: str):
        """
        generate a list of items to read from a given symbol name
        """
        
        read_list = []
        
        """
        TODO this can be inefficient for a large number of symbols/large rectangle
        this will need to be redone later
        """
        coord_list = elf_symbol.get_coordinates_for_symbol(symbol_name)
        if not coord_list:
            self._logger.warn("Invalid symbol name \"%s\"", symbol_name)
            return None
        
        for coord, so in coord_list:
            read_item = ReadItem()
            read_item.name = symbol_name
            read_item.addr = int(so.value/2)
            read_item.len = int(so.size/2)
            read_item.x = coord.x()
            read_item.y = coord.y()
            read_item.use_coord = True
            read_list.append(read_item)

        return read_list

    def read_list_all(self, elf_symbol):
        """
        generate a list of items from  the name of a symbol
        """
        
        read_list = []

        """
        TODO this can be inefficient for a large number of symbols/large rectangle
        this will need to be redone later
        """
        for sym_index in range(elf_symbol.get_symbol_count()):
            symbol_name = elf_symbol.get_symbol_name(sym_index)
            coord_list = elf_symbol.get_coordinates_for_symbol(symbol_name)
            for coord, so in coord_list:
                read_item = ReadItem()
                read_item.name = symbol_name
                read_item.addr = int(so.value/2)
                read_item.len = int(so.size/2)
                read_item.x = coord.x()
                read_item.y = coord.y()
                read_item.use_coord = True
                read_list.append(read_item)

        return read_list

    def read_list_from_addr(self, address: int, length: int):
        """
        generate a list of items from a memory address and length
        """

        read_item = ReadItem()
        read_item.addr = address
        read_item.len = length
        read_item.use_coord = False
        read_list = [read_item]

        return read_list

    def write(
        self,
        cm_dev,
        rect: Rectangle,
        address: int,
        length: int,
        values: List,
    ):
        write_memory_rect_list(cm_dev, rect, address, length, values)
