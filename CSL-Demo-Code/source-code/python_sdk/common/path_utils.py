"""
helper routines for generating files
"""

import os
import shutil


def make_and_check_writable(directory):
    """
    Create a directory and ensure that it is writable
    """
    os.makedirs(directory, exist_ok=True)
    assert os.access(directory, os.W_OK), f"{directory} not writable"


def make_and_ensure_empty(directory):
    """
    Deletes and recreates a directory to ensure it is fresh
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def copy_into(directory, filename):
    """
    Makes a local copy of filename in directory and returns absolute path to it
    """
    shutil.copy(filename, directory)
    return os.path.join(directory, os.path.basename(filename))

