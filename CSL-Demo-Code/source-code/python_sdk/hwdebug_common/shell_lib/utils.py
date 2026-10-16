import re
from pathlib import Path
from typing import List


def rglob_file(dir_path: Path, pattern: str) -> List[Path]:
    """
    Search recursively in specified directory and finds all the files the specified pattern.

    @param dir_path  directory to start the search.
    @param pattern   the file to look for, i.e. *.json.

    @return a list of relative path to dir_path where the files are found sorted by reverse chronological order.
    """
    found_path = list(dir_path.rglob(pattern))
    found_path.sort(key = lambda path: path.stat().st_mtime, reverse = True)
    found_path = list(map(lambda path: path.relative_to(dir_path), found_path))
    return found_path


def _glob_num_file_recur(
    dir_path: Path, pattern: str, stop_at: int, parent_dir: Path
) -> List[Path]:
    found_path = []

    # Base case in recursion. Found everything previously. Return an empty list.
    if stop_at <= 0:
        return found_path

    # First get all the files that match this pattern under dir_path
    for path in dir_path.glob(pattern):
        found_path.append(path.relative_to(parent_dir))

    # Stops if found everything user ask for, i.e. remaining is 0 or less.
    remain = stop_at - len(found_path)
    # Found up to the number per user asked
    if remain <= 0:
        return found_path

    # Depth first search, resursively go into directory structure to find the file.
    for child in dir_path.iterdir():
        if child.is_dir():
            child_found = _glob_num_file_recur(
                child, pattern, remain, parent_dir
            )
            # Update remain and concatenate the list
            remain = remain - len(child_found)
            found_path = found_path + child_found
            # Found up to the number per user asked and exit
            if remain <= 0:
                return found_path

    return found_path


def glob_num_file(dir_path: Path, pattern: str, num_file: int) -> List[Path]:
    """
    Depth first search into dir_path to find pattern. It stops as soon as it founds equal
    or more than number of files asked.
    The purpose of implementing a custom DFS search for a file is to save time during
    when use with auto-detecting for initialization of compile context.
    It is possible the number return files is greater than number than num_file specified when
    the matching file pattern are directly under the same parent directory.

    @param dir_path  directory to start the search.
    @param pattern   the file to look for, i.e. *.json.
    @param num_file  number of files to find before stopping.

    @return a list of relative path to dir_path where the files are found.
    """
    assert num_file > 0
    return _glob_num_file_recur(dir_path, pattern, num_file, dir_path)


def parse_index_input(file_index: str) -> int:
    """
    Parse input to make sure it's integer. Returns an unsigned integer
    if it's valid.
    """
    m = re.search("\[(\d+)\]", file_index)
    if m:
        return int(m.group(1))

    return -1
