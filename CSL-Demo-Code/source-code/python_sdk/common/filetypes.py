"""
Code to identify Cerebras-specific file formats
"""
import subprocess


def is_cs1_elf(filepath):
    """
    Returns True if the file is a CS-1 ELF (plan or core), else False.

    Relies on the `cs-readelf` command.
    :param str filepath: e.g. 'core.elf'
    """
    # Sanity check: first, confirm the file is readable.  The purpose of this
    # check is to make sure this function throws an appropriate Python
    # exception (usually an OSError subclass) if the file does not exist, is
    # not readable, is a directory, or any problem of that sort:
    with open(filepath, 'rb'):
        pass
    # Run cs-readelf, and use its return value to determing if the file
    # is a CS-1 ELF or not:
    try:
        subprocess.check_output(["cs-readelf", filepath],
                                stderr=subprocess.STDOUT)
        return True  # cs-readelf exited 0, so the file is valid
    except subprocess.CalledProcessError:
        return False  # cs-readelf exited non-zero, so the file is not valid
    # Note: ELF format details can be found here:
    # https://en.wikipedia.org/wiki/Executable_and_Linkable_Format
    # When cs-readelf verifies the ELF header, among other things it checks
    # that the e_machine field (a little-endian unsigned short int at offset
    # 0x12) has value 0xf8.
