import logging
from typing import Optional

from cerebras.das_common.das_commonpybind import Coordinate
from cerebras.hal.halpybind import Rectangle, is_contained_in
from cerebras.hwdebug_common.debug_lib.debugcommonpybind import (
    get_symbol_type_string,
)


class Image:
    """
    The image command is for list the ELF for a rectangle or look up a symbol of a given name.
    """

    def __init__(self, logger: Optional[logging.Logger]):
        self._logger = logger

    def _print_name(self, pe_coord: Coordinate, name: str):
        self._logger.info(" %s %s", pe_coord, name)

    def list(
        self, elf_symbol, all_pe: bool, rect: Optional[Rectangle],
    ):
        """
        Shows which rectangle is mapped to which ELF.
        If all_pe is true, then it will override the pe_list and iterate through the
        entire fabric.
        """
        assert elf_symbol is not None

        # All means print the name in each coordinates in the entire fabric
        if all_pe:
            d = elf_symbol.dimensions()
            for y in range(d.height()):
                for x in range(d.width()):
                    pe_coord = Coordinate(x, y)
                    name = elf_symbol.get_name_from_coordinate(pe_coord)
                    self._print_name(pe_coord, name)
        # No rectangle is created
        elif rect is None:
            self._logger.info(" No rectangle created")
        # Otherwise, print the name in each coordinates within the rectangle
        else:
            self._logger.info(" Rectangle: %s", rect)
            for y in range(rect.y(), rect.y() + rect.height()):
                for x in range(rect.x(), rect.x() + rect.width()):
                    pe_coord = Coordinate(x, y)
                    name = elf_symbol.get_name_from_coordinate(pe_coord)
                    self._print_name(pe_coord, name)

    def _print_symbol_table(
        self, elf_symbol, pe_coord: Coordinate, symbol_map,
    ):
        """
        Prints PE coordinate
        Prints header
        Prints symbol value, type, size, and name
        """
        self._logger.info(" %s", pe_coord)
        if len(symbol_map) == 0:
            self._logger.info(" Empty symbol table")
            return

        self._logger.info(" Num:\tValue\tSize\tType\tName")
        for idx, symbol_index in enumerate(symbol_map):
            symbol_name = elf_symbol.get_symbol_name(symbol_index)
            symbol = symbol_map[symbol_index]
            type_name = get_symbol_type_string(symbol.type)
            self._logger.info(
                " %d\t%s\t%d\t%s\t%s",
                idx,
                hex(symbol.value),
                symbol.size,
                type_name,
                symbol_name,
            )

    def lookup(
        self,
        elf_symbol,
        all_pe: bool,
        rect: Optional[Rectangle],
        symbol_name: Optional[str],
    ):
        """
        Shows all PE in the rectangles that contains the symbol's name.
        If all_pe is true, then it will override the pe_list and iterate through the
        entire fabric.
        If symbol_name is empty, it will print all the symbol names in the rectangle.
        """
        if not all_pe and rect is None:
            self._logger.info(" No rectangle created")

        # Prints the entire symbol table
        if symbol_name is None:
            if all_pe:
                d = elf_symbol.dimensions()
                for y in range(d.height()):
                    for x in range(d.width()):
                        pe_coord = Coordinate(x, y)
                        symbol_map = elf_symbol.get_symbols_from_coordinate(
                            pe_coord
                        )
                        self._print_symbol_table(
                            elf_symbol, pe_coord, symbol_map
                        )
            else:
                self._logger.info(" Rectangle: %s", rect)
                for y in range(rect.y(), rect.y() + rect.height()):
                    for x in range(rect.x(), rect.x() + rect.width()):
                        pe_coord = Coordinate(x, y)
                        symbol_map = elf_symbol.get_symbols_from_coordinate(
                            pe_coord
                        )
                        self._print_symbol_table(
                            elf_symbol, pe_coord, symbol_map
                        )
        else:
            pe_symbol = elf_symbol.get_coordinates_for_symbol(symbol_name)
            self._logger.info(" PE:\t\t\t\tValue\tSize\tType\tName")
            for pe_symbol in pe_symbol:
                pe_coord = pe_symbol[0]

                # check if pe_coord inside the rectangle
                if not all_pe:
                    assert rect is not None
                    pe_rect = Rectangle(pe_coord.x(), pe_coord.y(), 1, 1)
                    if not is_contained_in(pe_rect, rect):
                        continue

                symbol = pe_symbol[1]
                type_name = get_symbol_type_string(symbol.type)
                self._logger.info(
                    " %s\t%s\t%d\t%s\t%s",
                    pe_coord,
                    hex(symbol.value),
                    symbol.size,
                    type_name,
                    symbol_name,
                )

    def info(self, elf_symbol):
        """
        Shows the fabric dimension.
        """
        assert elf_symbol is not None
        d = elf_symbol.dimensions()
        self._logger.info(" Fabric dimension: %d x %d", d.width(), d.height())
