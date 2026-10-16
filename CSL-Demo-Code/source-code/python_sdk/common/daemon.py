#!/usr/bin/env python3
"""
Convenience tool to launch a daemon
"""
import logging
import os
import time
import sys
import signal

logger = logging.getLogger(__name__)
VERBOSE = logger.info
INFO = logger.info
WARNING = logger.warning
ERROR = logger.error
DEBUG = logger.debug

_DEFAULT_START_TIMEOUT = 100
_DEFAULT_STOP_TIMEOUT = 100


class Daemon:
    """
    Do the UNIX double-fork magic, see Stevens' "Advanced
    Programming in the UNIX Environment" for details (ISBN 0201563177)

    Usage:

    In your __main__ entry code, you need to have three options (start and stop).
    This allows three modes of your program:
       --start - start in daemon mode
       --stop - stop daemon running in background
       --out_dir - directory where pid file and up file go.  This could be omitted
                   if you use tmp files

    If --start or --stop is not specified, then your program is executed in the foreground

    Next, you need to update your __main__ code just after parsing args, to start/stop the daemon

    Example:
    def Main():
        args = parser.parse_args()

        up_file = os.path.join(args.out_dir, "db_up")
        pid_file = os.path.join(args.out_dir, "db_pid")

        daemon = Daemon("{your program name here, for logging}", up_file, pid_file)
        if args.stop is True:
            daemon.stop()

        os.makedirs(args.out_dir, exist_ok=True)

        if args.start is True:
            daemon.start()
            # note, parent does not return, and waits for up_file to be created
            #  code executed after this line is in the daemon

            # let parent process know we are ready (it is your responsibility to do this)
            with open(up_file, "w") as f:
                f.write("")

        # note, now if start was specified, you are in a deteached state, stdio will be captured
        # and logged to {name}.log, or logfile parameter if specified.

    """

    def __init__(self, name, up_file, pid_file, logfile=None, debug=False,
                 start_timeout=_DEFAULT_START_TIMEOUT,
                 stop_timeout=_DEFAULT_STOP_TIMEOUT):
        self.up_file = up_file
        self.pid_file = pid_file
        self.name = name
        self.debug = debug
        if logfile is None:
            logfile = os.path.join(os.path.dirname(self.up_file), f"{name}.log")

        self.logfile = logfile
        self._start_timeout = start_timeout
        self._stop_timeout = start_timeout


    def stop(self):
        """ stop a background daemon """

        if os.path.exists(self.up_file):
            with open(self.pid_file, 'r') as file:
                pid = int(file.read())
            os.kill(pid, signal.SIGHUP)

            start = time.time()
            while os.path.exists(self.up_file):
                time.sleep(1)
                if os.path.exists(self.up_file):
                    os.kill(pid, signal.SIGINT)
                if time.time() > start + self._stop_timeout:
                    ERROR("timeout waiting for %s to die", self.name)
                    sys.exit(1)
            os.remove(self.pid_file)
            INFO("%s stopped", self.name)

        else:
            WARNING("%s not running", self.name)
        sys.exit(0)

    def start(self, callback=None):
        """ daemonize the current program and wait for the up file
            to be present (indicating daemon started)  """
        if os.path.exists(self.up_file):
            ERROR("%s already running", self.name)
            sys.exit(1)
        if os.path.exists(self.pid_file):
            ERROR("%s already running", self.name)
            sys.exit(1)

        INFO("Starting %s in background", self.name)
        newpid = os.fork()
        if newpid != 0:
            # This is the parent process, so wait for the child to start up
            start = time.time()
            # Wait for the double-forked daemon child process to create its
            # PID file (this should be quick):
            while not os.path.exists(self.pid_file):
                time.sleep(1)
                if time.time() > start + 30:
                    ERROR("Child process %s failed to fork", self.name)
                    sys.exit(1)
            with open(self.pid_file, 'r') as filehandle:
                daemon_child_pid = int(filehandle.read())
            # Wait for the child to start doing its thing:
            while not os.path.exists(self.up_file):
                time.sleep(1)
                if time.time() > start + self._start_timeout:
                    ERROR("timeout waiting for %s to be ready", self.name)
                    sys.exit(1)
                # Also, check if it crashed:
                try:
                    # Note: if we "kill" with signal 0, it doesn't actually
                    # send a signal, it just checks whether we can send a
                    # signal, so this can be used to check if the PID exists.
                    os.kill(daemon_child_pid, 0)
                except ProcessLookupError:
                    ERROR("Daemon child %s crashed, see logs.", self.name)
                    sys.exit(1)
            INFO("%s started", self.name)
            if callback:
                callback()
            sys.exit(0)
        else:
            # Decouple from parent environment
            os.chdir(os.getcwd())
            os.setsid()
            os.umask(0o22)
            devnull = "/dev/null"
            if hasattr(os, "devnull"):
                # Python has set os.devnull on this system, use it instead as it might be different
                # than /dev/null.
                devnull = os.devnull

            INFO("Logging daemon to: %s", self.logfile)

            if not self.debug:
                f = open(self.logfile, 'w')
                ffd = f.fileno()

                devnull = "/dev/null"
                sys.stdout.flush()
                os.close(0)
                os.close(1)
                os.close(2)
                os.dup2(ffd, 1)
                os.dup2(ffd, 2)
                devnull_fd = os.open(devnull, os.O_RDWR)
                os.dup2(devnull_fd, 0)
                os.close(devnull_fd)
                logging.info("After dup")
                print("handlers being re-done")
                os.dup2(ffd, 1)
                os.dup2(ffd, 2)
                f.close()

            # Do second fork
            try:
                pid = os.fork()
                if pid > 0:
                    # Exit from second parent
                    sys.exit(0)
            except OSError as e:
                sys.stderr.write(
                    "fork #2 failed: %d (%s)\n" % (e.errno, e.strerror))
                sys.exit(1)

            # Write out our PID, so the parent can detect we've started and
            # kill us later:
            with open(self.pid_file, 'w') as file:
                file.write(str(os.getpid()))


