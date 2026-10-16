"""
General-purpose parsing routines
"""


def bool_from_str(boolstr):
    """
    Interpets a string as a boolean in a 'human-friendly' manner.

    Acceptable False values are:
      'false', 'no', 'f', 'n', or '0'
    Acceptable True values are:
      'true', 'yes', 't', 'y', or a non-zero integer
    Case is ignored.

    :param str boolstr: The string to interpret as a boolean

    Example:
      >>> from cerebras.common.parsers import bool_from_str
      >>> bool_from_str('no')
      False
      >>> bool_from_str('yes')
      True
      >>> bool_from_str('False')
      False
      >>> bool_from_str('True')
      True
      >>> bool_from_str('0')
      False
      >>> bool_from_str('1')
      True
    """
    if isinstance(boolstr, bool):
        # it's already a bool; let's allow that
        return boolstr
    if boolstr is None:
        return False
    if not boolstr:
        raise ValueError("Cannot interpret empty string: " + repr(boolstr))
    # Numeric string:
    if boolstr.isdecimal():
        return bool(int(boolstr))  # e.g. '0' -> False, '1' -> True
    # String like 'True' or 'False':
    boolstr_lc = boolstr.lower()
    if boolstr_lc in ["true", "t", "yes", "y"]:
        return True
    elif boolstr_lc in ["false", "f", "no", "n"]:
        return False
    raise ValueError("Cannot interpret string as bool: " + repr(boolstr))


def uint_list_from_str(argstr):
    """
    Parses a string which represents a comma-separated list of non-negative
    ints.  Also expands any ranges specified with a dash (including both end
    values; this is different from the Python range() built-in).

    Returns a list of non-negative ints.

    Examples:
      >>> uint_list_from_str('3')
      [3]
      >>> uint_list_from_str(None)
      []
      >>> uint_list_from_str('3,4,5')
      [3, 4, 5]
      >>> uint_list_from_str('3-5')
      [3, 4, 5]
    """
    if not argstr:
        return []
    subtest_nums = []
    # split on commas
    for arg in argstr.split(','):
        arg = arg.strip()
        if not arg:
            continue
        # expand ranges:
        if '-' in arg:
            lhs, rhs = arg.split('-')
            lhs = int(lhs.strip())
            rhs = int(rhs.strip())
            startval = min(lhs, rhs)
            endval = max(lhs, rhs)
            subtest_nums.extend(list(range(startval, endval + 1)))
        else:
            subtest_nums.append(int(arg))
    return subtest_nums
