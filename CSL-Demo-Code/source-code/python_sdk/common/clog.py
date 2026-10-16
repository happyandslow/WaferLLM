"""
A function to configure the a given logger.

Example:

import cerebras.common.clog as clog
logger = logging.getLogger(__name__)
def main():
   clog.setup_logger(logger=logger)
   parser = argparse.ArgumentParser(...)
   clog.add_parser_argument(parser, level='INFO');
"""
import sys
import logging
import argparse
from contextlib import contextmanager

LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

LOG_FORMAT_FILE = "%(asctime)s %(relativeCreated)6d "
LOG_FORMAT_FILE += (
    "[%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s"
)
LOG_FORMAT_CONSOLE = "[%(levelname)s] %(filename)s(%(lineno)s) >> %(message)s"

# pylint: disable=line-too-long
def log_level_string_to_int(log_level_string):
    """Convert log level string to int."""
    if not log_level_string in LOG_LEVEL_STRINGS:
        message = (
            f'invalid choice: {log_level_string} (choose '
            f'from {LOG_LEVEL_STRINGS})'
        )
        raise argparse.ArgumentTypeError(message)
    log_level_int = getattr(logging, log_level_string, logging.INFO)
    assert isinstance(log_level_int, int)
    return log_level_int


def add_parser_argument(parser, level='INFO'):
    """Add --log-level arg to a parser."""
    parser.add_argument(
        '--log-level',
        default=level,
        dest='log_level',
        type=str,
        choices=LOG_LEVEL_STRINGS,
        nargs='?',
        help='Set the logging output level.',
    )


@contextmanager
def logging_to_file_context(logfilename, name=None, level=logging.DEBUG):
    """
    Context manager which temporarily captures log messages in a file, e.g.:

      with logging_to_file_context("/tmp/log.txt"):
          ...

    :param str logfilename:
    :param str name: logger to redirect, same meaning as logging.getLogger(),
                     (defaults to the root logger)
    :param int level: log level of the file handler, subject to the global
                      logging config
    """
    logger = logging.getLogger(name=name)
    if isinstance(level, str):
        level = log_level_string_to_int(level)
    handler = logging.FileHandler(logfilename, mode='a', encoding='utf-8')
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT_FILE))
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


def setup_logger(logfilename=None, stdout_loglevel=logging.INFO, logger=None):
    """Configure the given logger (if None, then we use the root logger)

    Log at debug level to a file (filename is configurable)
    Also log info and above to stdout (level is configurable)

    :param str logfilename: filename to write logs into.  If None, will be
                            auto-generated
    :param int stdout_loglevel: at what level to dump messages to stdout
    """
    logger = logger or logging.getLogger()

    # Log to a file
    if logfilename:
        formatter = logging.Formatter(fmt=LOG_FORMAT_FILE)
        handler = logging.FileHandler(logfilename, mode='a', encoding='utf-8')
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    # Log to stdout
    formatter = logging.Formatter(fmt=LOG_FORMAT_CONSOLE)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(stdout_loglevel)
    logger.addHandler(handler)
