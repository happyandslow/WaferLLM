"""
Module for common Cerebras decorators.
"""

import os
import logging
import resource
import warnings
import functools
import decorator


def deprecated(func):
    """
    This is a decorator which can be used to mark functions
    deprecated. It will result in a warning being emitted
    when the function is used.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        """
        Inner wrapper function.
        :param args:
        :param kwargs:
        :return:
        """
        log = logging.getLogger(__name__)
        log.warning("Call to deprecated function %s", func.__name__)
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    return new_func


def deprecated_msg(message):
    """
    This is a decorator which can be used to mark functions
    deprecated. It will result in a warning being emitted
    when the function is used and printing the message passed by decorator.
    """

    def outer_wrapper(func):
        """ First wrapper function. """
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Second wrapper function"""
            log = logging.getLogger(__name__)
            log.warning("Call to deprecated function %s, %s", func.__name__,
                        message)
            warnings.warn(
                "Call to deprecated function {}.".format(func.__name__),
                category=DeprecationWarning)

            return func(*args, **kwargs)
        return new_func
    return outer_wrapper

#pylint: disable=keyword-arg-before-vararg
@decorator.decorator
def extend_environment(orig_func, env_vars=None, *args, **kwargs):
    """
    Decorates a function so that the running environment of this
    function is extended by dictionary `env_vars` before the function
    is called and it is returned to its original state after the
    call has been completed.
    """
    # First collect the old environment as we extend the current one
    old_env = {}
    env_vars = env_vars or {}
    for arg_name, arg_value in env_vars.items():
        if arg_name in os.environ:
            # Then preserve this for later on
            old_env[arg_name] = os.environ[arg_name]
        # And extend the current environment
        os.environ[arg_name] = arg_value

    try:
        return orig_func(*args, **kwargs)
    finally:
        # And always recover our previous
        for arg_name in env_vars:
            if arg_name in old_env:
                os.environ[arg_name] = old_env[arg_name]
            else:
                # Else, delete it from the environment
                del os.environ[arg_name]


def increased_rlimit_nofile(orig_func):
    """
    Decorates a function that needs a higher RLIMIT_NOFILE limit.
    On function return, the original limit is restored.
    Note that this is not a good idea for multi-threaded applications, since
    these limits are process-global.
    On error, a message is logged, but no exception is raised.

    Example:
      from cerebras.common.decorators import increased_rlimit_nofile

      @increased_rlimit_nofile
      def some_func_which_opens_many_files():
          ...
    """
    @functools.wraps(orig_func)
    def wrapper(*args, **kwargs):
        #
        # Increase the open files limit
        #
        orig_nofile_soft, orig_nofile_hard = resource.getrlimit(
            resource.RLIMIT_NOFILE)
        if orig_nofile_hard == resource.RLIM_INFINITY:
            # Note: On MacOS `sysctl kern.maxfilesperproc` is the actual max.
            new_nofile_soft = max(orig_nofile_soft, 10240)
        else:
            new_nofile_soft = orig_nofile_hard
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (new_nofile_soft, orig_nofile_hard))
        except (ValueError, resource.error) as exc:
            logging.getLogger(__name__).error(str(exc))
        try:
            #
            # Call the wrapped function
            #
            return orig_func(*args, **kwargs)
        finally:
            #
            # Restore the original open files limit
            #
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE,
                                   (orig_nofile_soft, orig_nofile_hard))
            except (ValueError, resource.error) as exc:
                logging.getLogger(__name__).error(str(exc))
    return wrapper

