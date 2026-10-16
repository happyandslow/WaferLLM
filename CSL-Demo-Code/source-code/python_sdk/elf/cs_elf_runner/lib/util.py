"""Utility functions used by ``cs_elf_runner``.
"""

from typing import List, Optional, Tuple

import numpy as np

from cerebras.elf.cself import Rectangle


def in_rectangle(rectangle: Rectangle, x: int, y: int) -> bool:
    """Determine whether given point is within given rectangle.

    Args:
        rectangle: Rectangle specified as ((col, row), (width, height))
        x: point x-coordinate (column)
        y: point y-coordinate (row)

    Returns:
        True if point is within the rectangle, False otherwise
    """

    (col, row), (width, height) = rectangle
    return x in range(col, col + width) and y in range(row, row + height)


def search_different(
    array: np.ndarray, rectangle: Rectangle
) -> Optional[Tuple[int, int]]:
    """
    Traverse the subset of the array within the given rectangle in row-major
    order to find the first value that differs from the value at the top-left
    corner of the rectangle.

    Args:
        array: Two-dimensional numpy array of values.
        rectangle:
            Rectangle describing a slice into the array, specified as
            ((col, row), (width, height))

    Returns:
        None if all values in the rectangle are the same, a (col, row)
        coordinate otherwise.
    """

    # Search for first value that is different from given value.
    (col, row), (width, height) = rectangle
    value = array[col, row]
    for y in range(row, row + height):
        for x in range(col, col + width):
            if array[x, y] != value:
                return x, y

    return None


def get_contiguous_rectangles(
    array: np.ndarray, rectangle0: Rectangle
) -> List[Tuple[int, Rectangle]]:
    """
    Return a list of rectangles which completely cover the given rectangle such
    that, for each returned rectangle, all the values in that rectangle in the
    given array are the same.

    Args:
        array: Two-dimensional numpy array of values.
        rectangle0:
            Rectangle describing a slice into the array, specified as
            ((col, row), (width, height))

    Returns:
        List of rectangles and the value they contain.
    """

    stack = [rectangle0]
    rectangles = []
    while stack:
        rectangle = stack.pop()

        (col, row), (width, height) = rectangle
        if width == 0 or height == 0:
            continue

        value = array[col, row]

        if width == 1 and height == 1:
            rectangles.append((value, rectangle))
            continue

        search_result = search_different(array, rectangle)

        if search_result is None:
            rectangles.append((value, rectangle))
            continue

        x, y = search_result

        if y != row:
            above_rect = ((col, row), (width, y - row))
            rectangles.append((value, above_rect))

        left_width = x - col
        new_height = height - y + row

        if left_width > 0:
            stack.append(((col, y), (left_width, new_height)))

        right_width = width - x + col
        stack.append(((x, y), (right_width, new_height)))

    return rectangles


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
