"""
helper function to get version of release
"""
import os
import logging

def release_version():
    """
    returns release version information
    """
    rel_version = "0.0.0" # release version
    version_path = "/cbcore/etc/version.json"
    if os.path.exists(version_path):
        with open(version_path, 'r') as f:
            rel_version = f.read()
    else:
        logging.warning("version file not found")
    return rel_version
