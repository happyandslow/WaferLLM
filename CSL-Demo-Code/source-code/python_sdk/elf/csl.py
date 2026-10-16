#!/usr/bin/env python3

"""
API for accessing the ``.pe_debug_info`` section in a CSL ELF
"""

import json
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Union

from cerebras.elf.cself import ELFMemory


def has_color_info(elfdata: BinaryIO) -> bool:
    """
    Does this ELF file stream have CSL-derived debug information?

    Args:
        elfdata: Stream to parse.

    Returns:
        True iff the JSON debug section is found.
    """

    # Only import this here, so this script remains importable for users who do
    # not have elftools available and do not want to call this function.
    from elftools.elf.elffile import ELFFile

    elf = ELFFile(elfdata)

    # Lookup the section CSL uses to store debug info.
    info = elf.get_section_by_name(".pe_debug_info")

    return info is not None


PEDebugInfo = Dict[str, Any]
"""
Information extracted from JSON from a CSL-compiler-produced ``.pe_debug_info``
section.
"""


def _decode_pe_debug_info(data: bytes) -> List[PEDebugInfo]:
    """
    Parse data originating from a CSL-compiler-generated ``.pe_debug_info``
    section into a structured form.

    Args:
        data: Raw data from a ``.pe_debug_info`` section.

    Returns:
        Decoded PE debug info.
    """

    # Decode it as JSON.
    try:
        pe_debug_info = json.loads(data)
    except Exception as e:
        raise RuntimeError(
            ".pe_debug_info section contains invalid JSON"
        ) from e

    # ``.pe_debug_info`` can contain a single JSON dictionary or a list of
    # dictionaries, so deal with this wrinkle to simplify our caller’s job
    if isinstance(pe_debug_info, dict):
        return [pe_debug_info]
    return pe_debug_info


def get_color_info(elfdata: BinaryIO) -> List[PEDebugInfo]:
    """
    Read CSL-derived debug information out of an ELF file stream.

    Args:
        elfdata: Stream to parse.

    Returns:
        Decoded JSON from the debug section of the ELF stream.
    """

    # Only import this here, so this script remains importable for users who do
    # not have elftools available and do not want to call this function.
    from elftools.elf.elffile import ELFFile

    elf = ELFFile(elfdata)

    # Retrieve the section CSL uses to store debug info.
    info = elf.get_section_by_name(".pe_debug_info")
    if info is None:
        raise RuntimeError(
            ".pe_debug_info section not found (is this not a "
            "CSL-originating ELF file?)"
        )

    return _decode_pe_debug_info(info.data())


def has_color_info_in_file(elfpath: Union[str, Path]) -> bool:
    """
    Does this ELF file have CSL-derived debug information?

    Args:
        elfpath: Path to ELF file to parse.

    Returns:
        True iff the JSON debug section is found.
    """

    # try to retrieve the `.pe_debug_info` section
    pedi = ELFMemory(elfpath).get_section(".pe_debug_info")

    # did it exist?
    return pedi is not None


def get_color_info_from_file(elfpath: Union[str, Path]) -> List[PEDebugInfo]:
    """
    Read CSL-derived debug information out of an ELF file.

    Args:
        elfpath: Path to ELF file to parse.

    Returns:
        Decoded JSON from the debug section of the ELF file.
    """

    # retrieve the `.pe_debug_info` section
    pedi = ELFMemory(elfpath).get_section(".pe_debug_info")
    assert pedi is not None, ".pe_debug_info section not found"

    # parse it
    return _decode_pe_debug_info(pedi)
