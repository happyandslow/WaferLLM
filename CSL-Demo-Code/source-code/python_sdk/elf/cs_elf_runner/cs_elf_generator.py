"""Methods and classes needed for ``cs_elf_generator``.
"""

import functools
import hashlib
from collections import defaultdict
from functools import partial
from typing import Dict, List, Tuple

import numpy as np

from cerebras.elf.cs_elf_runner.lib import util
from cerebras.elf.cself import (
    ELFLoader,
    ELFMemory,
    PathLike,
    Rectangle,
    SymbolRecord,
)
from cerebras.elf.cselfpybind import make_coord_elf

# pylint: disable = no-name-in-module
from cerebras.elf.poke_data import poke_data

# pylint: disable = no-name-in-module
from cerebras.sdk.debug.lib.rectangleopspybind import get_debugger

# pylint: disable = no-name-in-module
from cerebras.stack.tools.contextpy import CsContext


# pylint: disable=too-many-instance-attributes
class CSELFGenerator:
    """Loads and runs ELF files on the Cerebras system and generate data ELF
       or coordinate ELF

    Args:
        elf_list: List of ELF filenames.

    **Example**:

        .. code-block:: python

            runner = CSELFGenerator(["a.elf", "b.elf"])
            runner.set_symbol(px, py, <symbol>, my_np_tensor)

    """

    def __init__(self, elf_list: List[PathLike]):
        # These are the ELF files that eventually get loaded on the fabric
        # This list will be added to for routing etc
        assert elf_list, "Must have a non-empty list of ELF files to run."
        self.elf_list = [str(elf) for elf in elf_list]  # copy input ELF list
        self.orig_elf_list = self.elf_list.copy()

        # declare fabric dimension variables
        self.fab_h = 0
        self.fab_w = 0
        self.elf_top = (0, 0)
        self.elf_bot = (0, 0)

        # See note in _process_elfs_during_init() about why this is not using
        # defaultdict(dict)
        self.coords_to_symbol_table: Dict[
            Tuple[int, int], Dict[str, SymbolRecord]
        ] = {}

        # process input ELF files and set fabric dimensions, full-fabric mode,
        # and target architecture in CsContext()
        self._process_elfs_during_init()

        print(f"Configured with fabric of size {self.fab_w}x{self.fab_h}")
        print(
            "Using bounding-box covering active PEs from "
            f"{self.elf_top} to {self.elf_bot} (inclusive)."
        )

        assert self.fab_h > 0 and self.fab_w > 0, "invalid fabric dimensions!"

        #  read the architecture from one of our ELF files.
        first_elf = ELFMemory(elf_list[0])
        self.arch = first_elf.get_target().value

    @functools.lru_cache()
    def _get_debugger(self):
        """Get debugger for the kernel."""
        cmaddr = None
        return get_debugger(cmaddr, 0, 0, self.fab_w, self.fab_h)

    @functools.lru_cache()
    def _get_coords_to_elf(self) -> Dict[Tuple[int, int], str]:
        """Get a mapping of X,Y coordinates to ELF file names."""
        coords_to_elf = defaultdict(set)

        for elf in self.orig_elf_list:
            elf_memory = ELFMemory(elf)
            for rect in elf_memory.iter_rectangles():
                for col in range(rect.logical_x, rect.logical_x + rect.width):
                    for row in range(
                        rect.logical_y, rect.logical_y + rect.height
                    ):
                        coords_to_elf[col, row].add(elf)

        return coords_to_elf

    def _get_sym_addr_map(
        self,
        rectangle: Rectangle,
        symbol: str,
    ) -> (np.ndarray, int):
        """
        Builds a two-dimensional array where values are the memory addresses of
        the given symbol on the PE that corresponds to the array indices.
        Values are only valid within the given rectangle, and zero otherwise.
        Also validates that the size of the symbol in memory matches the
        expected size, in bytes.

        Args:
            rectangle:
            Rectangle of PEs which contain the given symbol, defined as:
                    ((column, row), (width, height))
            symbol: Name of the symbol.

        Returns:
            A tuple. First component is a two-dimensional numpy array matching
            the fabric size containing memory addresses. Second component is
            symbol size in bytes.
        """
        expected_size = None
        addr_arr = np.zeros((self.fab_w, self.fab_h), np.dtype(int))
        (col, row), (width, height) = rectangle
        for c in range(col, col + width):
            for r in range(row, row + height):
                sym_table = self.coords_to_symbol_table[c, r]
                sym_addr, sym_size = sym_table[symbol]

                if expected_size is None:
                    expected_size = sym_size
                elif expected_size != sym_size:
                    raise RuntimeError(
                        f"Symbol {symbol} has differing sizes within the "
                        f"rectangle. PE {c},{r} has {sym_size} bytes vs "
                        f"{expected_size} bytes seen previously."
                    )

                addr_arr[c, r] = sym_addr

        if expected_size is None:
            raise RuntimeError(
                f"symbol {symbol} not found in rectangle {rectangle}"
            )

        return addr_arr, expected_size

    def set_symbol(self, col: int, row: int, symbol: str, data: np.ndarray):
        """
        Initialize the value of 'symbol' at given PE coordinates with given data.
        Note: each call to this function creates a temporary elf file, so prefer
              'set_symbol_rect' over calling this in a loop.

        Args:
             col: Column of the PE
             row: Row of the PE
             symbol: Symbol to initialize
             data: Numpy array of data with appropriate type
        """
        padded = np.full(
            (
                1,
                1,
            )
            + data.shape,
            data,
        )
        return self.set_symbol_rect(symbol, padded, offset=(col, row))

    def set_symbol_rect(
        self, symbol: str, data: np.ndarray, offset: Tuple[int, int] = (0, 0)
    ):
        """
        Initialize the value of 'symbol' on multiple PEs.

        Args:
           symbol: Symbol to initialize
           data:
                  Numpy array of two or more dimensions which specifies the
                  initialization value on each PE. The first two dimensions
                  of 'data' are the column and row coordinates of the target
                  PE relative to 'offset'.

                  For example, on a 3x3 fabric, applying this data with offset
                  (1,1):

                    [[0, 1], [2, 3]]

                  would set value '0' on PE 1,1, value '1' on PE 1,2, value '2' on
                  PE 2,1, and value '3' on PE 2,2. Other PEs would not be
                  initialized.
              offset:
                  Offset the application of data by this many columns and rows.
        """
        if data.ndim < 2:
            raise RuntimeError(
                "set_symbol_rect: data must be 2+ dimensional, where the first "
                "two dimensions represent the rectangle of PEs (col, row)"
            )

        dims = (data.shape[0], data.shape[1])
        offset_x, offset_y = offset
        rectangle = (offset, dims)
        sym_addrs, per_pe_size = self._get_sym_addr_map(rectangle, symbol)

        expected_per_pe_size = data[0, 0].nbytes
        if expected_per_pe_size != per_pe_size:
            raise RuntimeError(
                f"Size of given data ({expected_per_pe_size} bytes) for symbol "
                f"{symbol} does not match symbol size ({per_pe_size} bytes)."
            )

        sym_addr_rects = util.get_contiguous_rectangles(sym_addrs, rectangle)

        bits = []
        per_pe_sizes = []
        rectangles = []
        sym_addrs = []

        for sym_addr, rectangle in sym_addr_rects:
            (col, row), (width, height) = rectangle
            subdata = data[
                col - offset_x : col - offset_x + width,
                row - offset_y : row - offset_y + height,
            ]

            # We ask for a numpy array that we can index into in the usual
            # Cerebras (col, row) convention. So data[2,3] contains the data for
            # PE (2, 3). However, numpy treats the first index as rows, and the
            # second as columns. Accordingly, we need to swap the first two axes
            # so ravel() flattens the sub-arrays in the order expected by
            # poke_data.
            bits.append(subdata.swapaxes(0, 1).ravel().tobytes())
            per_pe_sizes.append(per_pe_size)
            rectangles.append(rectangle)
            sym_addrs.append(sym_addr)

        # Since an ELF file will be written to disk with the symbol contents,
        # mangle its name to avoid collisions if this is used by concurrently
        # executing programs that share a working directory

        elf_coords = f"{offset_x}_{offset_y}_{dims[0]}_{dims[1]}"
        fab_dims = f"{self.fab_w}_{self.fab_h}"
        hasher = hashlib.md5()
        hasher.update(data.tobytes())
        res = hasher.hexdigest()
        result_elf = f"set_symbol_{elf_coords}_{symbol}_{fab_dims}_{res}.elf"

        poke_data(
            result_elf,
            bits,
            self.fab_w,
            self.fab_h,
            rectangles,
            sym_addrs,
            per_pe_sizes,
        )

        return result_elf

    def _process_elfs_during_init(self):
        """Process list of ELF files during CSELFGenerator construction.
        Note: this initializes member variables and should not be called
        outside of __init__
        """
        fabric_dimensions = None
        # top_x/top_y are initialized properly while processing the first elf
        top_x = top_y = bot_x = bot_y = 0
        for elf in self.elf_list:
            elf_memory = ELFMemory(elf)
            elf_loader = ELFLoader(elf_file=elf)
            # By calling this for every elf, we assert that they all have the
            # same architecture, and that the architecture matches any override
            # specified by the CS_TARGET environment variable OR prior
            # invocations of elf-generating libraries such as InitTensor. This
            # is to prevent an insidious footgun of accidentally running with
            # ELF files compiled for different architectures.
            CsContext().set_target_from_elf_flags(elf_memory.get_flags())

            elf_fabric_dims = elf_memory.get_fabric_dimensions()
            if fabric_dimensions is None:
                fabric_dimensions = elf_fabric_dims
                # Initialize top_x/top_y to fab_w/fab_h, respectively, as they
                # are updated using min() in iter_coordinates loop below.
                top_x = self.fab_w = elf_fabric_dims[0]
                top_y = self.fab_h = elf_fabric_dims[1]
            else:
                assert elf_fabric_dims == fabric_dimensions, (
                    f"Fabric dimensions {elf_fabric_dims} in {elf} do not match "
                    f"fabric dimensions {fabric_dimensions} of other ELF files. All "
                    f"ELF files must have the same fabric dimensions."
                )

            for col, row in elf_memory.iter_coordinates():
                top_x = min(top_x, col)
                top_y = min(top_y, row)
                bot_x = max(bot_x, col)
                bot_y = max(bot_y, row)
                # This is not a 'defaultdict(dict)' on purpose!
                # If it was, the update operation would have to be:
                #
                #     self.coords_to_symbol_table[col, row].update(d)
                #
                # But the 'update' method on dictionaries is extremely slow,
                # even when one of the dictionaries is empty. By making this a
                # regular dictionary, we can avoid 'update' unless the key is
                # actually present.
                if (col, row) in self.coords_to_symbol_table:
                    self.coords_to_symbol_table[col, row].update(
                        elf_loader.symbols
                    )
                else:
                    self.coords_to_symbol_table[col, row] = elf_loader.symbols

        # if the bounding box does not have space for a 1x halo of PEs, we have
        # to run in "full fabric mode" (without a halo)
        self.full_fabric_mode = (
            top_x < 1
            or top_y < 1
            or bot_x >= self.fab_w - 1
            or bot_y >= self.fab_h - 1
        )

        self.elf_top = (top_x, top_y)
        self.elf_bot = (bot_x, bot_y)

    def gen_coord_elf(self):
        """
        Generate the ELF containing the runtime coordinates
        """
        coord_elf = f"{self.fab_w}x{self.fab_h}_coord.elf"
        make_coord_elf(self.arch, self.fab_w, self.fab_h, coord_elf)
        return coord_elf

    def get_symbol(
        self,
        col: int,
        row: int,
        symbol: str,
        dtype: np.dtype,
    ) -> np.ndarray:
        """
        Read the value of 'symbol' of given type at given PE coordinates.
        Note: each call to this function scans the whole fabric, so prefer
            'get_symbol_rect' over calling this in a loop.

        Args:
            col: Column of the PE
            row: Row of the PE
            symbol: Symbol to read
            dtype: Numpy dtype of values contained by symbol
            size: Number of values contained by symbol

        Returns:
            Numpy array of symbol values.
        """
        res = self.get_symbol_rect(((col, row), (1, 1)), symbol, dtype)
        return res[0][0]

    def get_symbol_rect(
        self,
        rectangle: Rectangle,
        symbol: str,
        dtype: np.dtype,
    ) -> np.ndarray:
        """
        Read the value of 'symbol' of given type for multiple PEs.

        Args:
            rectangle: Rectangle specified as ((col, row), (width, height))
            symbol: Symbol to read
            dtype: Numpy dtype of values contained by symbol
            size: Number of values contained by symbol

        Returns:
            Numpy array of symbol values. The first two dimensions of the
            returned array are PE coordinates (column, row) relative to the
            rectangle.
        """
        (col, row), (width, height) = rectangle
        sym_addrs, per_pe_size = self._get_sym_addr_map(rectangle, symbol)
        sym_addr_rects = util.get_contiguous_rectangles(sym_addrs, rectangle)

        elem_size_bytes = np.empty(1, dtype).nbytes
        fabric_debugger = self._get_debugger()
        res = np.zeros((width, height, per_pe_size // elem_size_bytes), dtype)
        for sym_addr, rect in sym_addr_rects:

            def read_cb(r: Rectangle, cb_col: int, cb_row: int, addr, bits):
                if util.in_rectangle(r, cb_col, cb_row):
                    nonlocal res
                    res[cb_col - col, cb_row - row] = np.frombuffer(bits, dtype)

            # fabric_debugger.read takes address in words, and we have it in
            # bytes, so divide by 2
            fabric_debugger.read(
                sym_addr // 2, per_pe_size, partial(read_cb, rect)
            )

        return res
