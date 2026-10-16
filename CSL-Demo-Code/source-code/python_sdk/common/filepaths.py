"""
The library describes where Cerebras components are installed.

The intention is to encapsulate in one place the differences between a
customer-deliverable installation, and a development workspace.
"""

import os

import cerebras.common.git


def get_shlib_dir():
    """
    Returns the full path to a folder where Cerebras-specific .so files
    are located.
    In development workspaces, this will be $GITTOP/obj/lib
    """
    gittop = cerebras.common.git.get_gittop()
    return os.path.join(gittop, "obj", "lib")


def get_bin_dir():
    """
    Returns the full path to a folder where Cerebras executables are
    installed.
    In development workspaces, this will be $GITTOP/obj/bin
    """
    gittop = cerebras.common.git.get_gittop()
    return os.path.join(gittop, "obj", "bin")
