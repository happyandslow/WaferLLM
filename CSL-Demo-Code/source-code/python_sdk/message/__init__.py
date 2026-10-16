from . import messagepybind as _message
import traceback
import inspect
import sys
import os

"""
Confidential and Proprietary Information of
Cerebras Systems  http://www.cerebras.net
Copyright (c) 2020

This defines the python commands to control the
messages. This is similar to the c-level API
for message. These commands can be put in
the initialization file.

Importing the package initializes message manager.
We default to hiding the welcome messages.
"""
_message.initialize(False)


def print(tag: str, format: str = "", *args):
    """
    Prints a message with a given tag.
    :param tag The message tag that identifies the message object
    :param format The output string, possibly with formatb directives
    :param *args zero or more arguments of the format string
    :example
    >>> message.print("ABCD123","File %s does not exists", filename)
    ABCD123 ERROR: File myfile does not exist
    """
    stack = inspect.stack()
    frame = stack[1]
    func = frame.function
    filen = frame.filename.split("/")[-1]
    if format == "" and _message.exists(tag):
        format = _message.format_string(tag)

    # note: only formatting if arguments are given to format with..
    if format != "" and len(args) > 0 :
        format = format % args

    _message.print(tag, format, func, filen, frame.lineno)


def register(level: str, tag: str, format: str):
    """
    Registers a new message.
    Due to thread-safety concerns we are disabling this feature
    and requiring that messages are pre-registered in C++ instead.
    This is the python API to make a new message with a given tag
    :param level  The message level (fatal, error, warning, info, verbose or debug)
    :param tag    The tag identifier of the new message.
    :param format The default format string of the new message.
    :example
    >>> message.register("INFO","ABCD444","Flux capacitor is %s charged")
    >>> message.print("ABCD144","","fully")
    ABCD144 12:13:45  Flux capacitor is fully charged
    """
    _message.register_message(level, tag, format)


def info(format: str, *args):
    """
    Prints a default message with level 'info'
    :param format Text or format string
    :param optional format arguments
    :example
    >>> msg.info("Hello %s", "world")
    PYTH034 06:18:05  Hello world
    """
    stack = inspect.stack()
    frame = stack[1]
    func = frame.function
    filen = frame.filename.split("/")[-1]
    msg = format % args
    _message.info(msg, func, filen, frame.lineno)

def fatal(msg: str, *args):
    stack = inspect.stack()
    frame = stack[1]
    func = frame.function
    filen = frame.filename.split("/")[-1]
    msg = msg % args
    _message.print("PYTH030", msg, func, filen, frame.lineno)

def error(msg: str, *args):
    stack = inspect.stack()
    frame = stack[1]
    func = frame.function
    filen = frame.filename.split("/")[-1]
    msg = msg % args
    _message.error(msg, func, filen, frame.lineno)


def warning(msg: str, *args):
    stack = inspect.stack()
    frame = stack[1]
    func = frame.function
    filen = frame.filename.split("/")[-1]
    msg = msg % args
    _message.warning(msg, func, filen, frame.lineno)


def verbose(msg: str, *args):
    stack = inspect.stack()
    frame = stack[1]
    func = frame.function
    filen = frame.filename.split("/")[-1]
    msg = msg % args
    _message.verbose(msg, func, filen, frame.lineno)


def debug(msg: str, *args):
    stack = inspect.stack()
    frame = stack[1]
    func = frame.function
    filen = frame.filename.split("/")[-1]
    msg = msg % args
    _message.debug(msg, func, filen, frame.lineno)


def report(tagFilter: str = '.*'):
    """
    Prints a report of the messages that were printed.
    :param filter optional tag filter to which the message applies
    """
    _message.report(tagFilter)


def initialize(welcome_message: bool = False):
    """
    Initializes the system manually, which may be done to re-read config files
    after messages are registered.
    """
    _message.initialize(welcome_message)

def report_all(tagFilter: str = '.*'):
    """
    Reports all messages, regardless whether printed or not.
    :param tagFilter optional tag filter to which the message applies
    """
    _message.report_all(tagFilter)


def set_level(tag_filter: str, to_level: str, only_level: str = ""):
    """
    Sets the level of one or more messages
    :param tagFilter Tag or regexp of tags to which the setting applies
    :param to_level level to change to
    :param only_level only change messages of this level
    :example
    # make message AB123 INFO:
    >>> set_level("AB123", "INFO")
    # make all DEBUG messages in group "AB" INFO:
    >>> set_level("AB*", "INFO", "DEBUG")
    """
    _message.set_level_only(tag_filter, to_level, only_level)


def reset_level(tag_filter: str = ".*", only_level: str = None):
    """
    Resets the level of one or more messages back to
    factory default.
    :param tag_filter Tag or regexp of tags to reset
    :param only_level only change messages of this level
    :example:
    reset_level()            # resets all messages back to default
    reset_level("AB*")       # resets all messages macthing "AB*"
    reset_level("AB*","INFO")# resets all INFO messages matching AB*
    """
    _message.set_level_only(tag_filter, "RESET", only_level)


def set_print_level(tag_filter: str, to_level: str):
    """
    Overrides the global default print level of one or more messages.
    `set_print_level("*", level)` has the same effect at
    `set_default_level(level)` except that it only affects already-registered
    messages. Any new messages registered after this call won't get the new
    print level unless this function is called again with matching tags.
    :param tagFilter Tag or regexp of tags to which the setting applies
    :param to_level level to set the print level to.
    :example
    # make message AB123 print if it has a higher or equal severity that INFO
    # regardless of the global default level.
    >>> set_print_level("AB123", "INFO")
    """
    _message.set_print_level(tag_filter, to_level)


def reset_print_level(tag_filter: str = ".*"):
    """
    Resets the print level of one or more messages back to
    factory default.
    :param tag_filter Tag or regexp of tags to reset
    :example:
    reset_print_level()            # resets print level of all messages back to default
    reset_print_level("AB*")       # resets print level of all messages macthing "AB*"
    """
    _message.set_print_level(tag_filter, "RESET")


def level(tag: str = ""):
    """
    Returns the level of a given message.
    """
    return _message.level(tag)


def set_default_level(level: str = "INFO"):
    """
    Sets the global default logging level.
    :example
    >>> message.set_default_level("VERBOSE")
    """
    _message.set_default_level(level)


def set_module_level(moduleName: str, level: str):
    """
    Sets the module-specific default-logging level
    :example
    >>> message.set_default_level("python", "OFF")
    """
    _message.set_module_level(moduleName, level)


def format_string(tag: str):
    """
    Returns the built-in format string of a message.
    """
    return _message.format_string(tag)


def last_printed(tag: str):
    """
    Returns the message string that was last printed.
    :example

    """
    return _message.last_printed(tag)


def print_count(tag: str):
    """
    Returns the number of times a message was printed
    """
    return _message.print_count(tag)


def hit_count(tag: str):
    """
    Returns the number of times a message print attempt was made.
    """
    return _message.hit_count(tag)


def print_limit(tag: str):
    """
    Returns the number of times a message was printed
    """
    return _message.print_limit(tag)


def set_print_limit(tag: str, limit: int):
    """
    Sets the print limit of a message.
    Set to -1 to disable the limit.
    """
    _message.set_print_limit(tag, limit)


def set_line_wrap(num_characters: int = 0, tag_or_regexp: str = None):
    """
    Sets the print limit of a message.
    Set to -1 to disable the limit.
    """
    _message.set_line_wrap(num_characters, tag_or_regexp)


def exists(tag: str):
    """
    Returns True if the message exists, false otherwise
    """
    return _message.exists(tag)

def release_version():
    """
    Returns the build release version
    """
    return _message.release_version()

def will_print(tag: str):
    """
    Returns True if the message exists and will print
    """
    return _message.will_print(tag)


def total_print_count(level: str = "INFO"):
    """
    Returns the overall number messages printed by level
    """
    return _message.total_print_count(level)


def set_logfile(filepath: str = None):
    """
    Sets the log file to which all output will be directed
    :example
    >>> message.set_logfile("file.log")
    """
    _message.set_logfile(filepath)

def set_output_to_stderr(to_stderr: bool = True):
    """
    Same as set_output_redirect_mode("ALL_STDERR")
    Sets the output stream destination. The default stream is stdout
    :param to_stderr True to set to stderr, False to set back to stdout
    :example
    >>> message.set_output_to_stderr()
    """
    _message.set_output_to_stderr(to_stderr)

def set_output_redirect_mode(mode: str = "ALL_STDOUT"):
    """
    Sets where normal output is redirected:
         ALL_STDOUT                        All output into stdout
         CERR_TO_STDERR                    std::cerr into stderr, rest stdout
         ERRORS_AND_FATALS_TO_STDERR       Message level ERROR and FATAL to stderr, rest to stdout
         ALL_STDERR                        All output to stderr
    :example
    >>> message.set_output_redirect_mode("CERR_TO_STDERR")
    """
    return _message.set_output_redirect_mode(mode)

def set_g3log(value: bool):
    """
    Switches on or off g3log functionality.
    """
    _message.enable_g3log(value)


def set_cout_intercept(value: bool):
    """
    Switches on or off intercept of cout and cerr
    When on, any cout output appears as a default message
    """
    _message.enable_cout_intercept(value)


def set_output_capture(enable: bool = True, mute: bool = False):
    """
    Switches on or off output capturing.
    :param enable Enables output capture
    :param mute Disables message output to screen
    """
    _message.set_output_capture(enable, mute)


def captured_output():
    """
    Gets the captured output
    :example
    >>> message.output_capture(True)
    >>> message.info("hello")
    >>> captured = message.captured_output()
    >>> message.output_capture(False)
    """
    return _message.captured_output()


def set_preamble(preamble: str, tagFilter: str = '.*'):
    """
    Sets premable on some (or all) messages
    :param preamble
    :param args an optional tag filter to which the preamble applies
      "tag"       - the message tag.
      "timestamp" - timesstamp with each message.
      "custom"    - The string set using set_custom_preamble
      "millisec"  - A timestamp in milliseconds.
      "hostname"  - the computer host name.
      "username"  - the name of the computer user.
      "thread"    - the thread ID, except for the main thread.
      "allthread" - the thread ID of all threads.
      "file"      - the filename and line number (if available).
      "function"  - the function from which the message was sent.
      "module"    - the module ID that was set at compile.
    """
    _message.set_preamble(preamble, tagFilter)

def set_custom_preamble(custom_preamble: str):
    """
    Sets a global custom preamble string
    :param custom_preamble
    :example
    >>> message.set_custom_preamble("process2")
    After this the tag could be: 'ABCD-123 12:32 process2'
    """
    _message.set_custom_preamble(custom_preamble)


def set_default_level(level: str = "INFO"):
    """
    Sets the global default logging level
    :example
    >>> message.set_default_level("VERBOSE")
    """
    _message.set_default_level(level)


def capture_output(commands: str):
    """
    executes commands, returns output in string
    """
    logfile = temp.log
    msg.set_logfile(logfile)


def read_config_file_python(filepath: str):
    """
    Reads a config file in this python shell
    """
    print("PYTH401", "", filepath)
    exec(open(filepath).read())

def read_config_file(filepath: str):
    """
    Reads a config file using safe built-in parser
    This avoid any python-specific issues
    """
    print("PYTH402", "", filepath)
    _message.read_config_file(filepath)


def read_default_config_files():
    """
    Read all config files set by search path
    """
    # default is current directory, followed by home directory
    search_path = os.getenv("CEREBRAS_CONFIG_PATH",".:~/.cerebras")
    config_file_name = "cerebras_config.py"

    path_list = search_path.split(':')

    print("PYTH402","Looking for %s in %s", config_file_name, path_list)

    # process paths from back to front
    i = len(path_list) - 1
    while i >= 0:
        filepath = path_list[i] + '/' + config_file_name
        if path_list[i][0] == '~':
            homedir = os.path.expanduser("~")
            filepath = filepath.replace("~",homedir)

        if os.path.isfile(filepath):
            read_config_file(filepath)

        i -= 1


#
# Configuration files are read by the C++ core
# during startup
#read_default_config_files()
