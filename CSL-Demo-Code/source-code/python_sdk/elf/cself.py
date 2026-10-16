"""API for ``cself``-related functions"""

import functools
import subprocess
from collections import namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ByteString, Dict
from typing import Iterable as IterableType
from typing import Iterator, List, Optional, Set, Tuple, Union

import numpy as np
from elftools.elf.enums import ENUM_ST_INFO_TYPE

from cerebras.elf.cselfpybind import ELFObject, ELFSegments

STT_FUNC = ENUM_ST_INFO_TYPE["STT_FUNC"]

RectangleStart = Tuple[int, int]
RectangleExtent = Tuple[int, int]
Rectangle = Tuple[RectangleStart, RectangleExtent]

Symbol = namedtuple('Symbol', ['name', 'type', 'address', 'size'])
SymbolRecord = namedtuple('SymbolRecord', ['addr', 'size'])


@dataclass
class Segment:
    """
    Segment data from an ELF file.
    """

    addr: int
    extent: Rectangle
    mem_size: int
    data_size: int
    delta: int
    data: ByteString
    p_flags: Tuple[bool, bool, bool]


RectangleDescriptor = namedtuple(
    'RectangleDescriptor', ['logical_x', 'logical_y', 'width', 'height']
)
IndexedSegment = Tuple[int, Segment]


PathLike = Union[Path, str]
OptionalPath = Optional[PathLike]


class ELFLoader:
    """A convenience wrapper for ELF files.

    Wraps ``ELFObjects`` for an executable and a core ELF files.
    """

    def __init__(
        self,
        elf_file: OptionalPath = None,
        core_file: OptionalPath = None,
    ):
        self.symbols: Dict[str, SymbolRecord] = {}
        self.segments: List[Segment] = []

        if elf_file is None:
            self.elf = None
        else:
            self.set_elf(elf_file)

        if core_file is None:
            self.core = None
        else:
            self.set_core(core_file)

    def get_fabric_dimensions(self) -> Tuple[int, int]:
        """Return the fabric dimensions stored in the ELF."""
        if self.elf is None:
            raise RuntimeError(
                "not ELF set (missing `ELFLoader.set_elf` call?)"
            )
        return self.elf.get_fabric_dimensions()

    def set_elf(self, elf_file: PathLike):
        """Set the elf file used to determine symbol addresses."""
        self.elf = ELFObject(str(elf_file))
        self.symbols = {}

        def _iterate_symbols(sym_name, sym_type, addr, size):
            self.symbols[sym_name] = SymbolRecord(addr, size)

        self.elf.iterate_symbols(_iterate_symbols, type=ELFObject.OBJECT)

    def set_core(self, core_file: PathLike):
        """Set the core file."""
        self.core = ELFObject(str(core_file))
        self.segments = []

        def _iterate_segments(
            origin, extent, addr, msz, dsz, delta, data, p_flags
        ):
            self.segments += [
                Segment(
                    addr,
                    (origin, extent),
                    msz,
                    dsz,
                    delta,
                    memoryview(data),
                    p_flags,
                )
            ]

        self.core.iterate_segments(_iterate_segments)

    def get_symbol_addr(self, sym_name: str) -> SymbolRecord:
        """Get the byte address and size for a symbol."""
        if sym_name not in self.symbols:
            raise RuntimeError(f'{sym_name} not found')
        return self.symbols[sym_name]

    def print_segments(self):
        """Print the segments contained in the core file."""
        for v in self.segments:
            print(v)

    def _get_symbol_segment(
        self, sym: SymbolRecord, sym_name: Optional[str] = None
    ) -> Segment:
        """Get the segment in the core file containing the symbol record."""
        addr = sym.addr
        size = sym.size
        for v in self.segments:
            if v.addr <= addr and v.addr + v.mem_size >= size:
                return v
        assert (
            False
        ), f'Segment for {sym_name or "<unknown>"} at addr {addr} and size {size} not found'
        return None

    def get_symbol_segment(self, sym_name: str) -> Segment:
        """Get the segment in the core file which contains the symbol."""
        sym = self.get_symbol_addr(sym_name)
        return self._get_symbol_segment(sym)

    def _get_as_array(
        self,
        sym: SymbolRecord,
        local_shape: Union[int, IterableType[int]],
        sym_type: np.dtype,
        sym_name: Optional[str] = None,
    ) -> np.ndarray:
        """Get the symbol record's value as a numpy array.

        The symbol is assumed to have the numpy type ``sym_type``, and
        the shape ``local_shape`` on each of the PEs. The shape of the
        array returned is ``[sz_x, sz_y, *local_shape]``, where
        ``[sz_x, sz_y]`` is the shape of the (sub-)kernel where the
        symbol is distributed.
        """
        segment = self._get_symbol_segment(sym)
        addr = sym.addr
        size = sym.size
        offset = addr - segment.addr
        assert (
            offset >= 0
        ), f'Invalid segment for {sym_name or "<unknown>"} at addr {addr} and size {size}'
        pe_size = segment.extent[1]
        idx = 0
        if isinstance(local_shape, Iterable):
            local_shape = list(local_shape)
        else:
            local_shape = [local_shape]
        ret = np.zeros([pe_size[0], pe_size[1]] + local_shape, sym_type)

        # if this segment’s backing data is larger than its in-memory
        # footprint, it is a so-called “scatter section”
        is_scatter = segment.mem_size < segment.data_size

        for y in range(pe_size[1]):
            for x in range(pe_size[0]):
                # scatter sections have an inferable stride
                if is_scatter:
                    pe_offset = segment.mem_size * idx
                else:
                    pe_offset = 0

                bits = segment.data[pe_offset:]
                arr = np.frombuffer(
                    bits[offset : offset + size], dtype=sym_type
                )
                arr = np.reshape(arr, local_shape)
                ret[x, y, :] = arr
                idx += 1
        return ret

    def get_as_array(
        self,
        sym_name: str,
        local_shape: Union[int, IterableType[int]],
        sym_type: np.dtype,
    ) -> np.ndarray:
        """Get the symbol value as a numpy array.

        The symbol is assumed to have the numpy type ``sym_type``, and
        the shape ``local_shape`` on each of the PEs. The shape of the
        array returned is ``[sz_x, sz_y, *local_shape]``, where
        ``[sz_x, sz_y]`` is the shape of the (sub-)kernel where the
        symbol is distributed.
        """
        sym = self.get_symbol_addr(sym_name)
        return self._get_as_array(sym, local_shape, sym_type)

    def get_as_array_from_addr(
        self,
        addr: int,
        local_shape: Union[int, IterableType[int]],
        sym_type: np.dtype,
    ) -> np.ndarray:
        """Get the symbol value at the given address as a numpy array.

        The symbol is assumed to have the numpy type ``sym_type``, and
        the shape ``local_shape`` on each of the PEs. The shape of the
        array returned is ``[sz_x, sz_y, *local_shape]``, where
        ``[sz_x, sz_y]`` is the shape of the (sub-)kernel where the
        symbol is distributed.
        """
        size = np.prod(local_shape) * sym_type(1).itemsize
        sym = SymbolRecord(addr, size)
        return self._get_as_array(sym, local_shape, sym_type)

    def get_as_array_from(
        self, column: int, row: int, sym_name: str, sym_type: np.dtype
    ) -> np.ndarray:
        """Get the symbol value as a numpy array.

        Args:
            column (int): column index of PE from which to read memory
            row (int): row index of PE from which to read memory
            sym_name (str): name of symbol on PE
            sym_type (dtype): base type of data at symbol. e.g. if the symbol
                              points to an array of int16, this should be `np.int16`

        Returns:
            np.darray: the data for sym_name on PE (column, row)
        """
        segment = self.get_symbol_segment(sym_name)
        sym = self.get_symbol_addr(sym_name)
        addr = sym.addr
        size = sym.size
        offset = addr - segment.addr
        assert offset >= 0, f'Invalid segment for {sym_name}'
        pe_size = segment.extent[1]
        # Index of PE within local kernel is row size times row plus column.
        idx = pe_size[0] * row + column
        pe_offset = segment.mem_size * idx
        bits = segment.data[pe_offset:]
        return np.frombuffer(bits[offset : offset + size], dtype=sym_type)


class TargetDescription:
    """
    Naming of a Cerebras target device.

    See ../arch/target_arch.hh for the canonical definition of this information.

    Args:
        value: The numerical constant used to indicate this target architecture
            in ELF file fields.
        codename: The internal name for this target.
        product: The external name for this target if it has been announced.
    """

    def __init__(
        self,
        value: int,
        abbreviation: str,
        codename: str,
        llvm_target: Optional[str],
        llvm_cpu: Optional[str],
        product: Optional[str],
    ):
        self.value = value
        self.abbreviation = abbreviation
        self.codename = codename
        self.llvm_target = llvm_target
        self.llvm_cpu = llvm_cpu
        self.product = product


TARGETS = {
    ELFObject.CS_EINSTEIN: TargetDescription(
        ELFObject.CS_EINSTEIN, "EIN", "Einstein", "cs", "einstein", "WSE-1"
    ),
    ELFObject.CS_FEYNMAN: TargetDescription(
        ELFObject.CS_FEYNMAN, "FYN", "Feynman", "cs", "feynman", "WSE-2"
    ),
    ELFObject.CS_SCHRODINGER: TargetDescription(
        ELFObject.CS_SCHRODINGER,
        "SDR",
        "Schrodinger",
        "cs_sdr",
        "schrodinger",
        "WSE-3",
    ),
}
assert (
    ELFObject.CS_UNDEF_TARGET == ELFObject.CS_SCHRODINGER + 3
), "missing architecture support in ELFMemory.flags_to_target"


class ELFMemory:
    """
    A wrapper for inspecting the memory of one or more ELF files.

    When multiple ELF files are given, they are treated as a stack with later
    ELF files to be “layered” over earlier ELF files, as would be done by a
    Cerebras ELF loader.

    Args:
        elf_files: Paths to the ELF files to inspect.
    """

    def __init__(self, *elf_files: PathLike):
        self.elfs: List[ELFObject] = []
        for elf in elf_files:
            self.push_elf(elf)

    def push_elf(self, elf_file: PathLike, index: Optional[int] = None):
        """
        Add an ELF file to the contained stack.

        Args:
            elf_file: Path to an ELF file to add.
            index: Position in the stack at which to add this ELF file. Omit
                this to add it at the top/end.
        """

        # check the new arrival agrees with our existing population
        elf = ELFObject(str(elf_file))
        if len(self.elfs) > 0:
            old_dimensions = self.get_fabric_dimensions()
            new_dimensions = elf.get_fabric_dimensions()
            if old_dimensions != new_dimensions:
                raise RuntimeError(
                    f"ELF file {elf_file} has dimensions {new_dimensions} that "
                    "differ from the existing stacked ELF files that have "
                    f"dimensions {old_dimensions}"
                )

            old_target = self.get_flags()
            new_target = elf.get_flags()
            if old_target != new_target:
                raise RuntimeError(
                    f"ELF file {elf_file} targets architecture {new_target} "
                    "that differs from the existing stacked ELF files that "
                    f"target architecture {old_target}"
                )

        # invalidate the segment cache that will need to be recomputed
        self.get_segments.cache_clear()

        if index is None:
            index = len(self.elfs)
        self.elfs.insert(index, ELFObject(str(elf_file)))

    def pop_elf(self, index: Optional[int] = None):
        """
        Remove an ELF file from the contained stack.

        Args:
            index: Index to remove. Omit this to remove the last ELF file.
        """

        # invalidate the segment cache that will need to be recomputed
        self.get_segments.cache_clear()

        if index is None:
            index = len(self.elfs) - 1
        self.elfs.pop(index)

    def iter_segments(self) -> Iterator[Segment]:
        """
        Get the segments in this ELF file(s).

        This returns all segments as they are provided by the underlying C++
        API. This is strictly more general and more flexible than
        ``get_segments``, which returns a proxy object. If you need raw access
        to ELF segments, ``get_segments`` limits which segments you receive and
        which fields of those segments are accessible, so callers should in
        general prefer ``iter_segments``.

        Yields:
            All segments from this ELF stack.
        """
        for elf in self.elfs:
            segments = []

            def _iterate_segments(
                rect_root: RectangleStart,
                rect_dim: Tuple[int, int],
                address: int,
                mem_size: int,
                data_size: int,
                delta: int,
                data: bytes,
                p_flags: Tuple[bool, bool, bool],
            ):
                nonlocal segments
                x, y = rect_root
                width, height = rect_dim
                extent = (rect_root, (x + width, y + height))
                segments += [
                    Segment(
                        address,
                        extent,
                        mem_size,
                        data_size,
                        delta,
                        memoryview(data),
                        p_flags,
                    )
                ]

            elf.iterate_segments(_iterate_segments)
            yield from segments

    @functools.lru_cache()
    def get_segments(self) -> ELFSegments:
        """Retrieve segments."""

        segments = ELFSegments()

        for segment in self.iter_segments():
            # “inflate” the data information from a weak reference
            # (`memoryview`) to a value (`bytes`) in case callers depend on
            # this
            segment.data = segment.data.tobytes()

            origin, end = segment.extent
            segments.put(segment.addr, segment.mem_size, origin, end, segment)

        return segments

    def get_fabric_dimensions(self) -> Tuple[int, int]:
        """Return the fabric dimensions stored in the ``ELFObject``."""
        assert len(self.elfs) > 0, "accessing an empty `ELFMemory` stack"
        return self.elfs[0].get_fabric_dimensions()

    def get(self, addr: int) -> List[Tuple[Rectangle, IndexedSegment]]:
        """Return the segments containing ``addr``, or ``[]`` if ``addr`` not
        present.
        """
        x, y = self.get_fabric_dimensions()
        return self.get_segments().get(addr, (0, 0), (x - 1, y - 1))

    def get_flags(self) -> int:
        """Return the ELF flags (which contain target device information)."""
        assert len(self.elfs) > 0, "accessing an empty `ELFMemory` stack"
        return self.elfs[0].get_flags()

    def get_target(self) -> TargetDescription:
        """Return a description of the target architecture of this ELF file."""
        return ELFMemory.flags_to_target(self.get_flags())

    def get_clang_flags(self) -> List[str]:
        """
        Return the target-specific STC compilation flags for the architecture
        this ELF file(s) targets.

        Returns:
            A list of command line options to Clang.
        """
        target = self.get_target()
        if target.llvm_target is None or target.llvm_cpu is None:
            raise NotImplementedError(
                f"target '{target.abbreviation} has no front end STC compiler "
                "support"
            )

        return ["-target", target.llvm_target, f"-mcpu={target.llvm_cpu}"]

    @staticmethod
    def flags_to_target(flags: int) -> TargetDescription:
        """
        Translate an ELF target architecture value to a description of that
        target.
        """
        return TARGETS[flags]

    @staticmethod
    def abbreviation_to_target(abbreviation: str) -> TargetDescription:
        """
        Translate a target architecture abbreviation, e.g. ``"FYN"``, to its
        description.
        """
        for desc in TARGETS.values():
            if desc.abbreviation == abbreviation:
                return desc

        raise KeyError(f"unknown target architecture '{abbreviation}'")

    def get_section(self, name: str) -> Optional[bytes]:
        """
        Return a section’s data.

        Args:
            name: The name of the section to lookup.

        Returns:
            The data contained in this section or ``None`` if this section does
            not exist.
        """

        # iterate in reverse because later ELF files are assumed to take
        # precedence over earlier ELF files
        for elf in reversed(self.elfs):
            section = elf.get_section(name)
            if section is not None:
                return section

        return None

    def get_task_name(self, x: int, y: int, task_id: int) -> Optional[str]:
        """
        Return the source task name associated with a given task ID.

        This is an alternative to ``.get_task_names`` for callers who believe
        the task they are looking up has only a single name. If multiple names
        are found, an exception will be raised.

        Args:
            x: X-coordinate of the PE to examine.
            y: Y-coordinate of the PE to examine.
            task_id: ID entry in the task table to examine.

        Returns:
            The name associated with the given task on the given PE or ``None``
            if this task appears to have no associated names.
        """

        names = list(self.get_task_names(x, y, task_id))

        if len(names) > 1:
            raise RuntimeError(f"task has multiple names: {names}")

        if len(names) == 0:
            return None

        return names[0]

    def get_task_names(self, x: int, y: int, task_id: int) -> Iterator[str]:
        """
        Return all source task names associated with a given task ID.

            .. note::

                This method only recognizes idioms used by the CSL compiler. In
                future it may be extended to recognize idioms used by other
                compilers.

        Args:
            x: X-coordinate of the PE to examine.
            y: Y-coordinate of the PE to examine.
            task_id: ID entry in the task table to examine.

        Yields:
            All names associated with the given task on the given PE.
        """

        if len(self.elfs) == 0:
            return

        task_info = _get_target_task_info(self.get_flags())

        # read the address of the task table, a 16-bit pointer, for the given PE
        ENTPT_BASE_ADDR = 0x7E1E  # see RTL tile map
        POINTER_SIZE = 2  # bytes
        task_table_addr_bytes = self.read_data(
            x, y, ENTPT_BASE_ADDR * 2, POINTER_SIZE
        )
        assert len(task_table_addr_bytes) == POINTER_SIZE

        # if there is no valid pointer here, we have no task table
        if any(byte is None for byte in task_table_addr_bytes):
            return
        if all(byte == 0 for byte in task_table_addr_bytes):
            return

        # reconstruct the address and convert it from word units to byte units
        task_table_addr = (
            sum(b << (i * 8) for i, b in enumerate(task_table_addr_bytes)) * 2
        )

        # If we have any function that
        # claims to reside _within_ the given ID’s entry, consider it a
        # name for this task. This occurs when a task is inlined into the task
        # table.
        ENTRY_SIZE = task_info.entry_size
        task_base = task_table_addr + task_id * ENTRY_SIZE
        for symbol in self.iter_symbols(stt_type=STT_FUNC):
            # skip symbols whose name indicates the CSL compiler emitted them in
            # place of an unused task or that they're a trampoline (which will
            # be decoded later)
            if symbol.name.startswith(
                "task_table.nops."
            ) or symbol.name.startswith("trampoline."):
                continue
            if symbol.address in range(task_base, task_base + ENTRY_SIZE):
                yield symbol.name

        # retrieve the data for the task table entry itself
        entry_bytes = self.read_data(x, y, task_base, ENTRY_SIZE)
        assert len(entry_bytes) == ENTRY_SIZE

        # if the entry is not populated, we know nothing further about it
        if any(byte is None for byte in entry_bytes):
            return
        entry = sum(b << (i * 8) for i, b in enumerate(entry_bytes))

        # does the entry match the pattern the CSL compiler emits for an unused
        # task?
        if entry == task_info.empty_entry_encoding:
            return

        # otherwise, only continue if we recognize the entry as the pattern the
        # CSL compiler emits to initialize the stack pointer and jump to an
        # out-of-line function.
        target = _decode_sp_init_trampoline(entry, self.get_flags())
        if target is None:
            target = _decode_jump_only_trampoline(entry, self.get_flags())
        if target is None:
            return

        # consider any function that claims to live at this address a name for
        # this task
        for symbol in self.iter_symbols(stt_type=STT_FUNC):
            if symbol.address == target:
                yield symbol.name

    def iter_rectangles(self) -> Iterator[RectangleDescriptor]:
        """
        Iterate over all PE rectangles loaded to by these ELFs.

        Iteration order is unspecified and duplicates may occur.
        """

        for elf in self.elfs:
            # Accumulated rectangles. We end up materializing this list because
            # that is all that is available through the C API, but this
            # method’s type signature is intended to enable a future
            # implementation to lazily iterate through the rectangles.
            rs: List[RectangleDescriptor] = []

            def _collect(origin, extent, *_):
                nonlocal rs
                r = RectangleDescriptor(
                    origin[0], origin[1], extent[0], extent[1]
                )
                rs += [r]

            elf.iterate_segments(_collect)

            yield from rs

    def iter_coordinates(self) -> Iterator[Tuple[int, int]]:
        """
        Iterate over all PE coordinates loaded to by these ELFs.

        Unlike ``iter_rectangles``, iteration order is in lexical order and
        contains no duplicates.
        """

        # accumulate unique coordinates
        coords: Set[Tuple[int, int]] = set()
        for rect in self.iter_rectangles():
            for x in range(rect.logical_x, rect.logical_x + rect.width):
                for y in range(rect.logical_y, rect.logical_y + rect.height):
                    coords.add((x, y))

        # yield these in order
        yield from sorted(coords)

    def get_extent(self) -> Rectangle:
        """
        Get a rectangle covering the loaded-to PE region of these ELFs.

        Returns:
            A rectangle that encompasses all PEs that are loaded to.
        """
        upper_left = None
        lower_right = None
        for elf in self.elfs:
            x, y, width, height = elf.get_extent()
            if upper_left is None:
                upper_left = (x, y)
                lower_right = (x + width, y + height)
            else:
                upper_left = (min(upper_left[0], x), min(upper_left[1], y))
                lower_right = (
                    max(lower_right[0], x + width),
                    max(lower_right[1], y + height),
                )

        return (
            upper_left,
            (lower_right[0] - upper_left[0], lower_right[1] - upper_left[1]),
        )

    def get_symbol(self, name: str) -> Symbol:
        """
        Lookup a symbol in the symbol table(s) of these ELFs.

        The symbol table of a set of ELFs is considered to be the union of all
        their symbol tables. This may result in duplicate symbols and/or
        conflicting symbol definitions.

        This method assumes the caller knows their is a unique symbol
        corresponding to the name they are looking up. If nothing corresponding
        to the name is found or if multiple matching symbols are found, an
        exception will be thrown.

        Args:
            name: Which named symbol to find.

        Returns:
            The unique matching symbol.
        """
        symbol = None
        for elf in self.elfs:
            candidate = elf.get_symbol(name)
            if candidate is None:
                continue
            match = Symbol(
                name, candidate.type, candidate.value, candidate.size
            )
            if symbol is None:
                symbol = match
            else:
                if symbol != match:
                    raise RuntimeError(
                        f"symbol is not unique ({symbol} != {match})"
                    )

        if symbol is None:
            raise RuntimeError(f"symbol {name} not found")

        return symbol

    def iter_symbols(
        self, name: Optional[str] = None, stt_type: Optional[int] = None
    ) -> Iterator[Symbol]:
        """
        Iterate over symbols in the symbol table of these ELFs.

        The symbol table of a set of ELFs is considered to be the union of all
        their symbol tables. This may result in duplicate symbols and/or
        conflicting symbol definitions.

        Args:
            name: An optional filter to apply of which named symbols to yield.
            stt_type: An optional filter to apply of which type of symbols to
                yield.

        Yields:
            Any matching symbols found in these ELFs.
        """

        # Accumulated symbols. We end up materializing this list because that
        # is all that is available through the C API, but this method’s type
        # signature is intended to enable a future implementation to lazily
        # iterate through the symbols.
        symbols: List[Symbol] = []

        def symbol_append(
            sym_name: str, sym_type: int, address: int, size: int
        ):
            symbols.append(Symbol(sym_name, sym_type, address, size))

        for elf in self.elfs:
            elf.iterate_symbols(symbol_append, name, stt_type)
            yield from symbols
            symbols.clear()

    def read_data(
        self, x: int, y: int, address: int, size: int
    ) -> List[Optional[int]]:
        """
        Read data that would be loaded to the given address on the given PE.

        This method simulates the behavior of an ELF loader, deriving the bytes
        that would be loaded to the given range. That is, it takes into account
        segment ordering and both blocks and scatter segments.

            .. warning::

                This interface should only be used for one-shot reads of small
                amounts of data. Attempting to implement something like a full
                ELF loader using this method will be extremely slow.

        Though data is returned as a sequence of optional ints with ``None``
        representing addresses that were not loaded to, such bytes are actually
        zeroed currently. The type signature indicates entries are optional to
        remain compatible with a possible future improvement that would
        distinguish such unbacked addresses.

        Args:
            x: X-coordinate of the PE to examine.
            y: Y-coordinate of the PE to examine.
            address: Byte address of the base of the region.
            size: Byte extent of the region.

        Returns:
            A byte-by-byte list of values covering the requested range. That is,
            ``len`` of the returned value will be `size`.
        """

        # if we have no ELF files, this range is not loaded to
        if len(self.elfs) == 0:
            return [None] * size

        width, height = self.get_fabric_dimensions()
        assert x in range(
            width
        ), f"PE X-coordinate {x} is beyond fabric width {width}"
        assert y in range(
            height
        ), f"PE Y-coordinate {y} is beyond fabric height {height}"

        assert address >= 0, f"read from invalid address {hex(address)}"
        assert size >= 0, f"read of invalid size {size} bytes"

        # move ELF-by-ELF, letting each one override data from its predecessors
        data = [None] * size
        for elf in self.elfs:
            layer = elf.read_data((x, y), address, size)
            for offset, byte in enumerate(layer):
                if byte is not None:
                    data[offset] = byte

        return data

    def __getitem__(self, addr: int):
        """Return the data stored at ``addr`` across the PEs as a list.

        Each list element is a tuple of the top left corner
        of the PE array, and a ``numpy`` array. The ``numpy`` array will
        have the shape of the rectangle where data for ``addr`` was
        scattered. Returns ``[]`` if ``addr`` not present in the ELF file
        segments.
        """
        segments = self.get(addr)
        ret = []
        for _, segment in segments:
            assert addr in range(
                segment.addr,
                segment.addr + segment.mem_size,
            ), "ELFSegments.get returned a segment that did not cover requested address"
            data = np.frombuffer(segment.data, np.uint16)
            b = segment.extent[0]
            e = segment.extent[1]
            width = e[0] - b[0]
            height = e[1] - b[1]
            data = np.reshape(data, [width, height, segment.mem_size // 2])
            offset = (addr - segment.addr) // 2
            ret += [(b, data[:, :, offset : offset + 1].squeeze(-1))]
        return ret

    def __setitem__(self, k, _):
        """Setting item via indexing raises an exception."""
        raise NotImplementedError('Writing to an address not implemented')


def make_default_elf(
    width: int,
    height: int,
    arch: str,
    path: OptionalPath = None,
) -> str:
    """
    Create a default ELF of the given size. Return its name.

    Args:
        width: Width of the ELF to construct in PEs
        height: Height of the ELF to construct in PEs
        arch: Target architecture.
        path: Output path to write default ELF to. If not provided, a filename
            in the current directory will be generated.
    """
    if path is None:
        default_elf = f"{width}x{height}_default.elf"
    else:
        default_elf = str(path)

    args = [
        "cself_default",
        f"--arch={arch}",
        f"--fabric_width={width}",
        f"--fabric_height={height}",
        f"--width={width}",
        f"--height={height}",
        f"--arch={arch}",
        f"--out={default_elf}",
    ]

    subprocess.check_call(args)

    return default_elf


def _decode_sp_init_trampoline(entry: int, flags: int) -> Optional[int]:
    """
    Try to decode a task table trampoline containing stack pointer initialization instructions.

    Args:
        entry: the task table entry to be decoded.

    Returns:
        The jump target if the trampoline can be decoded. Otherwise, ``None``.
    """
    if flags not in (ELFObject.CS_EINSTEIN, ELFObject.CS_FEYNMAN):
        # Trampolines which initialize the stack pointer aren't expected on
        # these targets
        return None

    # Match the following pattern:
    # MOVRI R7 = <anything>
    # SLL16 R7 = R7, <any immediate>
    # JMP <anything>
    R7 = 0x7
    MOVRI_SUBOPCODE = 0x1C
    MOVRI_OPCODE = MOVRI_SUBOPCODE << 1
    SLL16_SUBOPCODE = 0x16
    SLL16_OPCODE = (SLL16_SUBOPCODE << 2) | 0x1
    SLL16_OP0T = 0x2
    SLL16_OP1T = 0x3
    JMP_SUBOPCODE = 0x1A
    JMP_OPCODE = (JMP_SUBOPCODE) << 1
    instruction0 = entry & ((1 << 20) - 1)
    if (instruction0 & ((1 << 6) - 1)) != MOVRI_OPCODE:
        return None
    if ((instruction0 >> 15) & ((1 << 4) - 1)) != R7:
        return None
    instruction1 = (
        ((entry >> 20) & ((1 << 8) - 1))
        | (((entry >> 28) & 0x1) << 19)
        | (((entry >> 52) & ((1 << 11) - 1)) << 8)
    )
    if (instruction1 & ((1 << 7) - 1)) != SLL16_OPCODE:
        return None
    if ((instruction1 >> 7) & ((1 << 2) - 1)) != SLL16_OP0T:
        return None
    if ((instruction1 >> 9) & ((1 << 2) - 1)) != SLL16_OP1T:
        return None
    if ((instruction1 >> 15) & ((1 << 4) - 1)) != R7:
        return None
    instruction2 = (entry >> 32) & ((1 << 20) - 1)
    if (instruction2 & ((1 << 6) - 1)) != JMP_OPCODE:
        return None

    # extract the (byte) address of the immediate (imm14) of the `jmp`
    target = (instruction2 >> 6) * 2
    return target


def _decode_jump_only_trampoline(entry: int, flags: int) -> Optional[int]:
    """
    Try to decode a task table trampoline containing a NOP and a JMP.

    Args:
        entry: the task table entry to be decoded.

    Returns:
        The jump target if the trampoline can be decoded. Otherwise, ``None``.
    """
    if flags in (ELFObject.CS_EINSTEIN, ELFObject.CS_FEYNMAN):
        return _decode_jump_only_trampoline_fyn(entry)
    if flags in (ELFObject.CS_SCHRODINGER, ELFObject.CS_HEISENBERG):
        return _decode_jump_only_trampoline_sdr(entry)
    return None


def _decode_jump_only_trampoline_sdr(entry: int) -> Optional[int]:
    PT = 0b110
    NOP_SUBOPCODE = 0x45
    NOP_REST = 0x832
    NOP = (PT << 28) | (0b1001 << 24) | (NOP_SUBOPCODE << 17) | NOP_REST
    JMP_SUBOPCODE = 0x18
    JMP_MSBS = (PT << 8) | (0b11 << 6) | JMP_SUBOPCODE

    instruction0 = entry & ((1 << 32) - 1)
    if instruction0 != NOP:
        return None

    instruction1 = (entry >> 32) & ((1 << 32) - 1)
    if ((instruction1 >> 20) & ((1 << 12) - 1)) != JMP_MSBS:
        return None

    # extract the (byte) address of the immediate (imm14) of the `jmp`
    jmp_encoded_imm = ((instruction1 >> 4) & ((1 << 6) - 1)) | (
        (instruction1 >> 12) & ((1 << 8) - 1)
    )
    return jmp_encoded_imm * 4


def _decode_jump_only_trampoline_fyn(entry: int) -> Optional[int]:
    NOP_SUBOPCODE = 0x8
    NOP_OPCODE = (NOP_SUBOPCODE << 4) | 0xF
    NOP_REST = 0x6
    NOP = (NOP_REST << 8) | NOP_OPCODE
    JMP_SUBOPCODE = 0x1A
    JMP_OPCODE = (JMP_SUBOPCODE) << 1

    instruction0 = entry & ((1 << 20) - 1)
    if instruction0 != NOP:
        return None

    instruction1 = (
        ((entry >> 20) & ((1 << 8) - 1))
        | (((entry >> 28) & 0x1) << 19)
        | (((entry >> 52) & ((1 << 11) - 1)) << 8)
    )
    if (instruction1 & ((1 << 6) - 1)) != JMP_OPCODE:
        return None

    # extract the (byte) address of the immediate (imm14) of the `jmp`
    target = (instruction1 >> 6) * 2
    return target


@dataclass
class _TargetTaskInfo:
    entry_size: int  # bytes
    empty_entry_encoding: int


def _get_target_task_info(flags: int) -> _TargetTaskInfo:
    if flags in (ELFObject.CS_EINSTEIN, ELFObject.CS_FEYNMAN):
        return _TargetTaskInfo(
            entry_size=8,
            empty_entry_encoding=0x8068068F88F0068F,
        )
    if flags in (ELFObject.CS_SCHRODINGER, ELFObject.CS_HEISENBERG):
        return _TargetTaskInfo(
            entry_size=16,
            empty_entry_encoding=0x798A0832798A0832798A0832698A0832,
        )
    assert False, f"Unsupported target {flags}"
