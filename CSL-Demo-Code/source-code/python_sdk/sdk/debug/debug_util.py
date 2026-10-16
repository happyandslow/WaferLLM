"""This module provides the debug_util class
"""
import functools
import os
import struct
from glob import glob

import numpy as np

from cerebras.elf.cs_elf_runner.lib import util
from cerebras.elf.cself import ELFLoader, ELFMemory
from cerebras.sdk.debug.lib.rectangleopspybind import get_debugger

Rectangle = util.Rectangle


def get_csl_debug_module_symbol_name(
    debug_module_key: str, symbol: str, legacy_naming_convention: bool = False
) -> str:
    """
    Retrieves the name of a symbol exported by a CSL debug module.

    CSL debug modules have two internal state variables we need to access:
    `buffer` and `buffer_index`. The names of these variables in the ELF file
    depend on a user-supplied key. For example, if the debug module is
    imported as:

        const debug_module = @import_module(
            "<debug>",
            .{ .key = "my_key",
               .buffer_size = 1000,
             }
        );

    then the imported debug module will expose symbols named
    `$$csl_debug.my_key.buffer` and `$$csl_debug.my_key.buffer_index`. This
    naming convention is an internal implementation detail of CSL's debug
    module, and is subject to change.

    Args:
        debug_module_key: The user-supplied key for the CSL debug module.
        symbol: The symbol to be retrieved.
        legacy_naming_convention: If True, use the older "legacy" naming
            convention. This option will be removed once migration to the new
            naming convention is complete.

    Returns:
        The ELF name of the requested debug module symbol.
    """
    return (
        f'{debug_module_key}.tr.{symbol}'
        if legacy_naming_convention
        else f'$$csl_debug.{debug_module_key}.{symbol}'
    )


# pylint: disable=too-many-instance-attributes
class debug_util:
    """Loads ELF files on the Cerebras system in order to dump the symbols.

  The user does not need to export the symbols in the kernel. The debug_util
  dumps the core and looks for the symbols in the ELFs. If the symbol at Px.y
  is not found in the corresponding ELF, debug_util emits an error.

  The common error is either a wrong coordinate passing in get_symbol() or the
  coordinate is correct but the compiler removes the tensor due to some
  optimization technique. One can use readelf to check if the symbol exists or
  not. If not, the easy way is to export such symbol in the kernel to keep the
  symbol in the ELF.

  The current version only supports simfabric locally.

  Args:
        dirname: a folder containing ELF filenames.
        cmaddr: ``'IP_ADDRESS:PORT'`` string of CM This argument is not needed
            if running on simfabric.

  **Example**:

        .. code-block:: python

            from cerebras.sdk.debug import debug_util

            # run the app
            # dirname is the path to ELFs
            simulator = SdkRuntime(dirname, cmaddr=args.cmaddr)
            simulator.load()
            simulator.run()
            ...
            simulator.stop()

            # retrieve symbols after the run
            debug_mod = debug_util.debug_util(dirname, cmaddr=args.cmaddr)
            # assume the core rectangle starts at P4.1, the dimension is
            # width-by-height and we want to retrieve the symbol y for every PE
            core_offset_x = 4
            core_offset_y = 1
            for py in range(height):
              for px in range(width):
                t = debug_mod.get_symbol(core_offset_x+px, core_offset_y+py, 'y', np.float32)
                print(f"At (py, px) = {py, px}, symbol y = {t}")

  """

    def __init__(self, dirname: str, cmaddr=None):

        self.cmaddr = cmaddr

        assert cmaddr is None, "Only support simfab"

        # WARNING: to make _get_debugger/set_symbol/set_rect work,
        # we need orig_elf_list, fab_w and fab_h as well
        self.orig_elf_list = glob(f"{dirname}/bin/out_[0-9]*.elf")
        east_dir = os.path.join(dirname, "east")
        elfs_east = glob(f"{east_dir}/bin/out_[0-9]*.elf")
        west_dir = os.path.join(dirname, "west")
        elfs_west = glob(f"{west_dir}/bin/out_[0-9]*.elf")

        self.orig_elf_list = self.orig_elf_list + elfs_east + elfs_west

        # read ELFs to extract
        # - fab_w and fab_h
        # - target: EIN or FYN
        # - core_elf_top, core_elf_bot
        self._process_elfs_during_init()

        # width and height of core rectangle
        self.core_w = self.core_elf_bot[0] - self.core_elf_top[0] + 1
        self.core_h = self.core_elf_bot[1] - self.core_elf_top[1] + 1

    def _process_elfs_during_init(self):
        """Process list of ELFs during CSRunner construction.
       Note: this initializes member variables and should not be called
       outside of __init__
    """
        fabric_dimensions = None
        rect_list = []

        # is there a target architecture implied by `$CS_TARGET`?
        target = os.environ.get("CS_TARGET")

        for elf in self.orig_elf_list:
            elf_memory = ELFMemory(elf)

            # Check all ELF files agree with both each other and any implied
            # target architecture. This is to prevent an insidious footgun of
            # accidentally running with ELF files compiled for different
            # architectures.
            this_target = elf_memory.get_target().abbreviation
            if target is None:
                target = this_target
            elif target != this_target:
                raise RuntimeError(
                    f"mismatched target architectures '{target}' and '{this_target}'"
                )

            elf_fabric_dims = elf_memory.get_fabric_dimensions()
            if fabric_dimensions is None:
                fabric_dimensions = elf_fabric_dims
                self.fab_w = elf_fabric_dims[0]
                self.fab_h = elf_fabric_dims[1]
            else:
                assert elf_fabric_dims == fabric_dimensions, (
                    f"Fabric dimensions {elf_fabric_dims} in {elf} do not match "
                    f"fabric dimensions {fabric_dimensions} of other ELFs. All "
                    f"ELFs must have the same fabric dimensions."
                )

            for rect in elf_memory.iter_rectangles():
                rect_list.append(rect)

        # WARNING: need to record target because connect_and_run() will generate ELFs
        # for demux/cmd fan/mux with the same --arch flag.
        self.target = target

        # Find coordinates of the top-left-most and the bottom-right-most PEs,
        # which define the bounding box of the kernel.
        top_x = min(rect.logical_x for rect in rect_list)
        top_y = min(rect.logical_y for rect in rect_list)
        bot_x = max(rect.logical_x + rect.width for rect in rect_list) - 1
        bot_y = max(rect.logical_y + rect.height for rect in rect_list) - 1

        self.core_elf_top = (top_x, top_y)
        self.core_elf_bot = (bot_x, bot_y)

    @functools.lru_cache()
    def _get_debugger(self):
        """ Get debugger for the kernel.
    """
        return get_debugger(self.cmaddr, 0, 0, self.fab_w, self.fab_h)

    def _get_sym_addr_map(
        self, rectangle: Rectangle, symbol: str,
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
        for elf in self.orig_elf_list:
            loader = ELFLoader(elf_file=elf)
            for c, r in ELFMemory(elf).iter_coordinates():
                if util.in_rectangle(rectangle, c, r):
                    sym_addr, sym_size = loader.get_symbol_addr(symbol)

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

    def get_symbol(
        self, col: int, row: int, symbol: str, dtype: np.dtype,
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

    Returns:
           Numpy array of symbol values.
    """
        res = self.get_symbol_rect(((col, row), (1, 1)), symbol, dtype)
        return res[0][0]

    def get_symbol_rect(
        self, rectangle: Rectangle, symbol: str, dtype: np.dtype,
    ) -> np.ndarray:
        """
    Read the value of 'symbol' of given type for multiple PEs.

    Args:
            rectangle: Rectangle specified as ((col, row), (width, height))
            symbol: Symbol to read
            dtype: Numpy dtype of values contained by symbol

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
                sym_addr // 2, per_pe_size, functools.partial(read_cb, rect)
            )

        return res

    def read_trace(self, col: int, row: int, key: str) -> list:
        """Parse a CSL trace buffer.

        Args:
            col: Column of PE that contains trace buffer
            row: Row of PE that contains trace buffer
            key:
                Key of imported trace module. For instance, if the CSL code
                contains:

                    const t = @import_module(
                      "<debug>",
                      .{ .key = "my_trace",
                         .buffer_size = ...
                       }
                    );

                Then 'key' is 'my_trace'.

        Returns:
            Heterogeneous list of trace values.
        """
        # Retrieve debug data.
        def get_buffer_index(legacy: bool):
            sym = get_csl_debug_module_symbol_name(
                key, 'buffer_index', legacy_naming_convention=legacy
            )
            return self.get_symbol(col, row, sym, np.uint16)[0]

        try:
            buffer_index = get_buffer_index(legacy=False)
        except RuntimeError:
            buffer_index = get_buffer_index(legacy=True)

        def get_buff(legacy: bool):
            sym = get_csl_debug_module_symbol_name(
                key, 'buffer', legacy_naming_convention=legacy
            )
            return self.get_symbol(col, row, sym, np.uint8)

        try:
            buff = get_buff(legacy=False)
        except RuntimeError:
            buff = get_buff(legacy=True)

        # buffer_index is the index into the buffer array, which is an array of u16
        # values. We fetch the buffer as np.uint8 (bytes), so we need to multiply
        # the index by 2 to translate from 16-bit words to bytes.
        buffer_index *= 2
        buff = list(buff[:buffer_index])
        result = []
        while buff:
            # The tag on device is 16 bits, but the type tag is always the lower 8
            # bits. Recall that device is little-endian.
            tag = chr(buff.pop(0))
            tag_high_bits = buff.pop(0)

            def _unpack(tag: str, packed: bytes):
                # 't' is the tag used by the CSL trace library for 'timestamp',
                # and does not exist in the 'struct' library
                if tag == 't':
                    # A timestamp is 48 bits, which is not supported by 'struct'
                    # so pad with zeros and parse as a 64-bit int
                    packed = b''.join([packed, b'\x00\x00'])
                    tag = 'Q'
                result = struct.unpack(f'<{tag}', packed)[0]
                # if the tag is terminated with 's', we need to decode the bytes
                # into a python string
                if tag[-1] == 's':
                    result = result.split(b'\x00')[0].decode('utf-8')
                return result

            if tag in '?bB':  # bools, i8, and u8, respectively
                # values are single-byte and the trace library packs them into the
                # high bits of the tag_word
                result.append(_unpack(tag, tag_high_bits))
                continue

            # 't' is the tag used by the CSL trace library for 'timestamp', and does
            # not exist in the 'struct' library
            if tag == 't':
                size = 6  # a timestamp is 48 bits, so 6 bytes
            # 's' is the tag use by a utf-8 string, and we pack the number of bytes
            # in the buffer in the second byte. The buffer size is always even.
            # The maximum string size is the maximum even buffer length (254).
            elif tag == 's':
                size = tag_high_bits
                tag = str(size) + tag
            else:
                size = struct.Struct(f'<{tag}').size

            assert size % 2 == 0

            packed = b''.join(buff[:size])
            buff = buff[size:]
            result.append(_unpack(tag, packed))

        return result
