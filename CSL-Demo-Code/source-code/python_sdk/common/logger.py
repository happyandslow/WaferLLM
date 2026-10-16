'''
Helper functions for handling equivalence between log levels.
'''

import logging
from cerebras.pb.common.logging_pb2 import (
    QUIET, NORMAL, VERBOSE, DEBUG, LL_DEFAULT, LogLevel
)

_OTHERS_TO_CB_LOG = {
    # Names from logging libraries
    "info": NORMAL,
    "warn": NORMAL,
    "warning": NORMAL,
    "error": QUIET,
    "fatal": QUIET,

    # Names from cerebras loggers (TODO: STANDARIZE ON NAMES ABOVE ONLY).
    "quiet": QUIET,
    "normal": NORMAL,
    "verbose": VERBOSE,
    "debug": DEBUG,
    "default": NORMAL,

    # cerebras ones
    QUIET: QUIET,
    NORMAL: NORMAL,
    VERBOSE: VERBOSE,
    DEBUG: DEBUG,
    LL_DEFAULT: NORMAL,

    # logging ones
    logging.DEBUG: DEBUG,
    logging.INFO: NORMAL,
    logging.WARN: NORMAL,
    logging.ERROR: QUIET,
    logging.FATAL: QUIET,
    logging.NOTSET: NORMAL,
}


_CB_TO_LOGGING = {
    # Names from cerebras loggers (TODO: STANDARIZE ON NAMES ABOVE ONLY).
    "quiet": logging.ERROR,
    "normal": logging.INFO,
    "verbose": logging.INFO,
    "debug": logging.DEBUG,
    QUIET: logging.ERROR,
    NORMAL: logging.INFO,
    VERBOSE: logging.DEBUG,
    DEBUG: logging.DEBUG,
}


def log_verbosity_to_cerebras(log_level):
    """
    @brief      Given a string, TF logging mode, or log mode from `logging,`
                this method returns the equivalent cerebras logging enum
                LogLevel.
                Throws error if given instance is not supported.

    @param      log_level  The log level

    @return     LogLevel corresponding to argument
    """
    if isinstance(log_level, str):
        log_level = log_level.lower().strip()

    if log_level in ["default", LL_DEFAULT]:
        # Then return this file's default logger verbosity
        return log_verbosity_to_cerebras(
            logging.getLogger(__name__).getEffectiveLevel()
        )

    if log_level in _OTHERS_TO_CB_LOG:
        return _OTHERS_TO_CB_LOG[log_level]

    # Else we simply throw an error
    raise ValueError('Unsupported log level "{}"'.format(log_level))


def cerebras_to_log_verbosity(cerebras_log_level):
    """
    @brief      Returns the logging loglevel corresponding to the given
                cerebras LogLevel object (or string). Raises ValueError if not
                a valid cerebras::LogLevel or string representation.

    @param      cerebras_log_level  The cerebras log level

    @return     The corresponding LogLevel in the `logging` library.
    """
    if isinstance(cerebras_log_level, str):
        # Then let's clean it up
        cerebras_log_level = cerebras_log_level.lower().strip()

    if cerebras_log_level in ["default", LL_DEFAULT]:
        # Then return this file's default logger verbosity
        return logging.getLogger(__name__).getEffectiveLevel()

    if cerebras_log_level in _CB_TO_LOGGING:
        return _CB_TO_LOGGING[cerebras_log_level]
    # Else we simply throw an error
    raise ValueError('Unsupported cerebras log level "{}"'.format(
        cerebras_log_level
    ))


def cerebras_log_level_to_str(log_level):
    """
    @brief      Returns the string equivalent to the given cerebras LogLevel.
                Throws ValueError if not valid cerebras LogLevel.

    @param      log_level  The cerebras log level

    @return     The string corresponding to the given LogLevel.
    """
    if log_level not in _OTHERS_TO_CB_LOG:
        raise ValueError('Unsupported cerebras log level "{}"'.format(
            log_level
        ))
    return LogLevel.Name(_OTHERS_TO_CB_LOG[log_level]).lower()

