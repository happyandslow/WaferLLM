import os
from typing import List, Optional

from cerebras.das_common.das_commonpybind import (
    load_routes_from_elf,
    Coordinate,
    Dimensions
)
from cerebras.elf.cself import ELFMemory
from cerebras.hwdebug_common.viz_service.base import ERROR_CODES, HTTPJsonError


class ElfInfo:
    """ holds information about each elf file """

    def __init__(self, elf_file):
        self.elfobj_ = ELFMemory(elf_file)

    def read_data(self, x, y, address, size):
        data_bytes: List[Optional[int]] = self.elfobj_.read_data(
            x, y, address, size
        )
        assert len(data_bytes) == size, "unexpected ELFMemory.read_data result"

        # convert to a list of words
        assert size % 2 == 0, "unaligned reads are not supported"
        data_words: List[int] = []
        for i in range(0, size, 2):
            low = data_bytes[i] or 0
            high = data_bytes[i + 1] or 0
            data_words.append((high << 8) | low)

        return data_words


class CoreCache:
    """
    This class survives multiple REST api calls by being kept in
    the plugin manager directory under indexed by path name

    This class should keep information that is repetitive, like
    examining the directories for artifacts, creating HAL/DAS contexts
    for debugging apis

    """

    def __init__(self, corefilepath):
        """
        Provide a cache for core file
        """
        if not os.path.exists(corefilepath):
            raise HTTPJsonError(
                ERROR_CODES.NO_CORE,
                "Core file {path} does not exist or is inaccessable)",
                {"path": corefilepath},
            )

        self.core = None
        try:
            self.core = ElfInfo(corefilepath)
        except:
            raise HTTPJsonError(
                ERROR_CODES.CORE_INVALID_FORMAT,
                "Error loading core file {filename}",
                {"filename": corefilepath},
            )

    def getRoutes(self, reqpath, x, y, w, h):
        c = Coordinate(x, y)
        d = Dimensions(w, h)
        return load_routes_from_elf(reqpath, c, d)
