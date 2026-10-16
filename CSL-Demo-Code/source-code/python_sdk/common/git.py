"""
Routines for querying git

These routines do *not* throw exceptions if called from outside of a
git repo; they just return empty strings.
"""
import os
import logging
import subprocess


def _run_git_cmd_get_out(cmd, cwd="."):
    """ If the command fails, an error is logged and None is returned """
    try:
        result = subprocess.run(cmd, text=True, cwd=cwd, capture_output=True, check=True)
        return result.stdout.rstrip()
    except subprocess.CalledProcessError as ex:
        logging.exception(' '.join(cmd) + ": " + ex.stderr)
        return None


def get_gittop(ignore_envvar=False):
    """
    If $GITTOP is set, returns that.
    Otherwise, determines it by running 'git rev-parse --show-toplevel'.
    Returns None if not running in a git repo.

    WARNING: this may depend on your current working directory

    :param bool ignore_envvar: if True, ignore value of $GITTOP and always
                               determine based on current working directory
    """
    if not ignore_envvar:
        # try to determine it from the environment:
        for envvar in ("GITTOP", "GIT_ROOT"):
            gittop = os.environ.get(envvar, None)
            if gittop and os.path.isdir(gittop):
                return gittop
    # try to determine it by running git commands:
    cmd = ["git", "rev-parse", "--show-toplevel"]
    gittop = _run_git_cmd_get_out(cmd, cwd=".")
    if gittop and os.path.isdir(gittop):
        return gittop
    # looks like we're not in a git repo:
    return None


def is_git_tree_clean(ignore_envvar=False):
    """
    Returns True if the current git workspace is clean, else False

    :param bool ignore_envvar: if True, ignore value of $GITTOP and always
                               determine based on current working directory
    """
    cwd = get_gittop(ignore_envvar=ignore_envvar)
    cmd = ["git", "status", "--untracked-files=no", "--porcelain"]
    out = _run_git_cmd_get_out(cmd, cwd=cwd)
    return out == ""


def get_git_branch(ignore_envvar=False):
    """
    Returns the current git branch, or None on error
    Equivalent to: `git rev-parse --abbrev-ref HEAD`

    :param bool ignore_envvar: if True, ignore value of $GITTOP and always
                               determine based on current working directory
    """
    cwd = get_gittop(ignore_envvar=ignore_envvar)
    cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    return _run_git_cmd_get_out(cmd, cwd=cwd)


def get_latest_git_commit_short(ignore_envvar=False, length=10):
    """
    Returns the short form of the latest git commit
    Equivalent to: `git rev-parse --short HEAD`

    :param bool ignore_envvar: if True, ignore value of $GITTOP and always
                               determine based on current working directory
    :param length: Length of short commit SHA-1
    """
    cwd = get_gittop(ignore_envvar=ignore_envvar)
    cmd = ["git", "rev-parse", f"--short={length}", "HEAD"]
    git_hash = _run_git_cmd_get_out(cmd, cwd=cwd)
    if len(git_hash) > length:
        raise ValueError(f"Obtained unique short hash: {git_hash} of length: {len(git_hash)}.\n"
                         f"Git computes a short hash of at least {length} characters - may be "
                         f"longer when needed to avoid collisions. \nConsider increasing the "
                         f"default value passed to length.")
    return git_hash


def get_latest_git_commit(ignore_envvar=False):
    """
    Returns the latest commit ID, or None on error

    Equivalent to: `git log -1 --format=%H`

    :param bool ignore_envvar: if True, ignore value of $GITTOP and always
                               determine based on current working directory
    """
    cwd = get_gittop(ignore_envvar=ignore_envvar)
    cmd = ["git", "log", "-1", "--format=%H"]
    return _run_git_cmd_get_out(cmd, cwd=cwd)

