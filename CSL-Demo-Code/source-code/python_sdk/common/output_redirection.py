#!/usr/bin/env python3
"""
Provides context managers for redirecting output
"""

import os
import sys


class _OutputRedirected(object):
    """
    Base class which redirects output
    """
    _streamname = None
    _output_fd = None
    _file_open_mode = 'a'
    _fdopen_mode = 'w'

    def __init__(self, fh):
        """
        :param file fh: file to redirect stdout to; may also be
                                a str (filename) or int (file descriptor)
        """
        assert self._streamname in ['stdin', 'stdout', 'stderr']
        assert isinstance(self._output_fd, int)
        self._close_on_exit = False
        # We expect a fh, but we can also accept a filename or a
        # file descriptor.
        if isinstance(fh, str):
            fh = open(fh, self._file_open_mode)
            self._close_on_exit = True  # we opened it, so we should close it
        elif isinstance(fh, int):
            fh = os.fdopen(fh, self._fdopen_mode)
        #
        # original output stream
        #
        self._save_fh = getattr(sys, self._streamname)
        try:
            self._save_fd = os.dup(self._output_fd)
        except OSError:
            self._save_fd = None  # e.g. stdout is closed
        #
        # new output stream to redirect to
        #
        self._redir_fh = fh
        try:
            self._redir_fd = fh.fileno()
        except (OSError, NameError, NotImplementedError, ValueError):
            self._redir_fd = None  # e.g. it's a StringIO

    def __enter__(self):
        self._save_fh.flush()
        # redirect output (>> self._redir_fh)
        setattr(sys, self._streamname, self._redir_fh)
        if self._redir_fd is not None:
            os.dup2(self._redir_fd, self._output_fd, inheritable=True)
        return self

    def __exit__(self, *args):
        self._redir_fh.flush()
        # restore output (>> self._save_fh)
        setattr(sys, self._streamname, self._save_fh)
        if self._redir_fd is not None and self._save_fd is not None:
            os.dup2(self._save_fd, self._output_fd, inheritable=True)
        # clean up
        if self._close_on_exit:
            self._redir_fh.close()


class stdout_redirected(_OutputRedirected):  # pylint: disable=invalid-name
    """
    This is the same as contextlib.redirect_stdout, except that it redirects
    at the file descriptor level, too.
    """
    _streamname = 'stdout'
    _output_fd = 1


class stderr_redirected(_OutputRedirected):  # pylint: disable=invalid-name
    """
    This is similar to contextlib.redirect_stdout, except for stderr, and
    it redirects at the file descriptor level, too.
    """
    _streamname = 'stderr'
    _output_fd = 2
