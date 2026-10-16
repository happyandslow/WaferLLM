from pathlib import Path

DEBUG_DUMP_PREFIX = "debug_dump_uid_"
KERNEL_TREE = "kernel_tree.json"
KERNEL_ANNOTATION = "kernel_annotation.json"


def validate_chief_dir(chief_dir):
    """Check for existence of chief dir. Return None if no error found.

    :param chief_dir: chief workdir
    :type chief_dir: str
    """
    chief_path = Path(chief_dir)
    if not chief_path.exists():
        return f"Chief workdir not found: {chief_dir}"
    if not chief_path.is_dir():
        return f"Chief workdir is not a directory: {chief_dir}"

    if chief_path.name.startswith(DEBUG_DUMP_PREFIX):
        # When FE sends debug dump directory, kernel tree is in the parent dir
        chief_path = chief_path.parent

    kernel_tree_path = chief_path / KERNEL_TREE
    if not kernel_tree_path.exists():
        return f"Cannot find kernel tree file: {kernel_tree_path}"

    kernel_annotation_path = chief_path / KERNEL_ANNOTATION
    if not kernel_annotation_path.exists():
        return f"Cannot find kernel annotation file: {kernel_annotation_path}"
