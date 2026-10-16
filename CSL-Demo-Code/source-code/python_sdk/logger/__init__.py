from . import loggerpybind as _logger
import traceback
import inspect
import sys
import os

LOGGER_PATH = "testing"

def _format_msg(msg : str):
    """
    Inspect python call stack
    """
    #stack = inspect.stack()
    #frame = stack[2]
    #return (frame.filename.split("/")[-1], frame.lineno, frame.function, msg)
    return ("", 0, "", msg)

# public
def set_logger_path(path : str):
    """
    Set the logging path
    """
    _logger.set_logger_path(path)

def info(msg : str, *args):
    """
    Info level logging
    """
    msg = msg % args
    info = _format_msg(msg)
    _logger._info(info[0], info[1], info[2], info[3])

def warning(msg : str, *args):
    """
    Warning level logging
    """
    msg = msg % args
    info = _format_msg(msg)
    _logger._warning(info[0], info[1], info[2], info[3])

def debug(msg : str, *args):
    """
    Warning level logging
    """
    msg = msg % args
    info = _format_msg(msg)
    _logger._debug(info[0], info[1], info[2], info[3])

def error(msg : str, *args):
    """
    Warning level logging
    """
    msg = msg % args
    info = _format_msg(msg)
    _logger._error(info[0], info[1], info[2], info[3])

def fatal(msg : str, *args):
    """
    Warning level logging
    """
    msg = msg % args
    info = _format_msg(msg)
    _logger._fatal(info[0], info[1], info[2], info[3])

def start(logfile_prefix: str):
    """
    Start the logger service
    """
    _logger.logger_start(logfile_prefix)

def reinit(logfile_prefix: str):
    """
    Stop the logger service
    """
    _logger.logger_reinitialize(logfile_prefix)

