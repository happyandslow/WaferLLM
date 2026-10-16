#!/usr/bin/env python3

'''
sdk debug PE symbol dump
'''
from pathlib import Path
from typing import Dict, Iterator, Union

from cerebras.elf.cself import ELFMemory

SYMBOL_SKIP_PREFIXES = ('@__', '__', 'LLVM')
SYMBOL_STRIP_SUFFIXES = ('$local',)


def coordinate_matches(x, y, elf_x, elf_y):
    """
    Check if there is a coordinate match
    """
    if x == -1 or y == -1:
        return True
    return x == elf_x and y == elf_y


def strip_suffixes(name: str) -> str:
    """
    remove internal suffixes from a given symbol
    """
    for suffix in SYMBOL_STRIP_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class PESymbolDump:
    """
    PESymbolDump object
    """

    def __init__(self, elf: Union[str, Path]):
        self.elf = ELFMemory(elf)

        rects = list(self.elf.iter_rectangles())
        if len(rects) > 1:
            raise RuntimeError("Has more than one rectangle. Not a CSL ELF?")

        self.coord = (rects[0].logical_x, rects[0].logical_y)

    def get_coordinates(self):
        """
        Get the coordinates
        """
        return self.coord

    def iter_symbols(self) -> Iterator[Dict[str, Union[str, int]]]:
        """
        Get symbols
        @yield dictionaries of the name, address, size, and type.
        """
        # Skip certain symbols with no symbol names or symbols begins with certain prefix
        def skip_symbol(symbol_name):
            if len(symbol_name) == 0 or symbol_name.startswith(
                SYMBOL_SKIP_PREFIXES
            ):
                return True
            return False

        # Get all the symbol then filter out the unnecessary ones
        for symbol in self.elf.iter_symbols():
            if skip_symbol(symbol.name):
                continue
            yield {
                "name": strip_suffixes(symbol.name),
                "size": symbol.size,
                "address": symbol.address,
                "type": symbol.type,
            }
