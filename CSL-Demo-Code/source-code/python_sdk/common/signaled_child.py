"""
Utilities for launching child processes whose lifetimes should be scoped to
that of their parent.
"""
import ctypes
import functools
import os
import signal

try:
    from setproctitle import setproctitle
except ImportError:

    def setproctitle(proctitle):
        pass


libc = ctypes.CDLL("libc.so.6", use_errno=True)


def signal_on_parent_thread_death(sig: signal.Signals = signal.SIGTERM):
    """
    SET_PDEATHSIG is a linux-only API that sends this process a signal (default
    SIGTERM) when the *thread* that spawns it exits.

    Args:
        signal: Signal number to send (defaults to SIGTERM)
    """
    # see include/uapi/linux/prctl.h in kernel
    PR_SET_PDEATHSIG = 1
    result = libc.prctl(PR_SET_PDEATHSIG, sig)
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(
            errno, f"prctl failed to set pdeathsig: {os.strerror(errno)}"
        )


def signaled_child(target, proctitle=None):
    """
    A function decorator that calls signal_on_parent_thread_death and
    optionally sets the proctitle to be a user friendly string. This ensures
    the child process does not outlive the parent in case things are not
    terminated cleanly.

    Useful for wrapping the target function of a multiprocessing.Process

    Args:
        target (Callable): Function to wrap as the target child.
        proctitle: Optional string when not used a function decorator. Sets
                   this as the process title using setproctitle in the target.

    Examples:

       @signaled_child
       def my_thread():
           return

       signaled_target = signaled_child(my_thread)
       signaled_target = signaled_child(my_thread, proctitle="my thread")
    """

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        signal_on_parent_thread_death()
        if proctitle:
            setproctitle(proctitle)
        return target(*args, **kwargs)

    return wrapper


def with_proctitle(proctitle):
    """
    A function decorator that sets the proctitle to be a user friendly string.

    Examples:

       @signaled_child
       @with_proctitle("my-cool-process")
       def my_thread():
           return
    """

    def _decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            setproctitle(proctitle)
            return func(*args, **kwargs)

        return wrapper

    return _decorator
