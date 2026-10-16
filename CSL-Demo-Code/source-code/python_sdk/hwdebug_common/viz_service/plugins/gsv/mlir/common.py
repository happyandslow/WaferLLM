import os
from pathlib import Path

STALL_VIZ = 'stall_viz'
DBG_RTIR = 'dbg_rt_mlir.json'
KERNEL_TREE = 'kernel_tree.json'


def get_mlir_path(chief_dir):
    path = Path(chief_dir) / "ws_rt.mlir"
    if path.exists():
        return str(path)

    """For kernel unit tests, the mlir file is not ws_rt.mlir.
       Hence walking the directory to get the MLIR path"""
    for file in os.listdir(chief_dir):
        if file.endswith('mlir'):
            return str(Path(chief_dir) / file)


def validate_layout_request(chief_dir):
    """Check existence of passed in chief_dir and essential files for layout.
     Return proper error message for missing file. Return None if no error found.

    :param chief_dir: chief workdir
    :type chief_dir: str
    """
    if not Path(chief_dir).exists():
        return f"Chief workdir not found: {chief_dir}"
    if not Path(chief_dir).is_dir():
        return f"Chief workdir is not a directory: {chief_dir}"

    dbg_rtir_path = Path(chief_dir) / DBG_RTIR
    if not dbg_rtir_path.exists():
        return f"Cannot find {DBG_RTIR} in {chief_dir}"


def validate_mlir_graph_request(chief_dir):
    """Check existence of passed in chief_dir and essential files for mlir graph.
     Return proper error message for missing file. Return None if no error found.

    :param chief_dir: chief workdir
    :type chief_dir: str
    """
    if not Path(chief_dir).exists():
        return f"Chief workdir not found: {chief_dir}"
    if not Path(chief_dir).is_dir():
        return f"Chief workdir is not a directory: {chief_dir}"

    dbg_rtir_path = Path(chief_dir) / DBG_RTIR
    if not dbg_rtir_path.exists():
        return f"Cannot find {DBG_RTIR} in {chief_dir}"


def validate_mlir_source_request(chief_dir):
    """Check existence of passed in chief_dir and essential files for mlir source.
     Return proper error message for missing file. Return None if no error found.

    :param chief_dir: chief workdir
    :type chief_dir: str
    """
    if not Path(chief_dir).exists():
        return f"Chief workdir not found: {chief_dir}"
    if not Path(chief_dir).is_dir():
        return f"Chief workdir is not a directory: {chief_dir}"

    mlir_path = get_mlir_path(chief_dir)
    if mlir_path is None:
        return f'Cannot find .mlir file in {chief_dir}'
