""" Profiler, a hook to trace local variables.
"""
import os
import sys
import imp
import json
import threading
import logging
import tempfile
from datetime import datetime
from functools import wraps


LOG = logging.getLogger(__name__)
if not LOG.handlers:
    LOG.addHandler(logging.NullHandler())


# Constants
SYNC_LOCK = threading.Lock()


class Profiler(object):

    """
    Context hook (decorator) to trace local variables from a functional
    method and dump the sourced into a file in json format.

    By default, 'output_file' can be located under '/var/tmp' if it is not
    instructed by calling program. for e.g. "/var/tmp/model_profiling.json"

    The profiling stats file is always appendable if the particular file
    already exists. This is the expectation by design, where the aim is to
    have centralized data for profiling or benchmarking.

    Usage:

    from  cerebras.common.utils import Profiler

    Template:
    @Profiler("train_time", "loss", "max_steps")
    def mnist_train_model(...):
        ....
        ....
        start_time = time.time()
        max_steps = params.get(max_steps')
        loss = tf.reduce_mean(...)
        ....
        ....
        # time taken for each step or batch
        train_time = time.time() - start_time

   file: 252_mnist_single_layer.py
   def test_mnist_training():
        mnist_train_model()

   <snip> of /var/tmp/model_profiling.json,
   {
        "12_15_2017T13:59:29:301556":{
            "loss":24.732345,
            "max_steps":8,
            "name":"mnist_train_model",
            "train_time":20.020105123519897
       }
    }

    TODO:
        1. db handler to update the given db.
        2. An interface to analytics.
    """

    def __init__(self, *tracevars, **configvars):
        # tracevars: Variables to be traced
        # configvars: Config params such as 'output_file'
        self._local_vars = {}
        self._func = None
        self._trace_vars = tracevars
        self._configvars = configvars
        self._output_file = None

    def __enter__(self):
        """ Context entry """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Context exit """
        self._local_vars = None
        sys.setprofile(None)

    @property
    def output_file(self):
        """ Set file location to dump sourced data. """
        self._output_file = self._configvars.get('output_file', None)
        if self._output_file is None:
            if os.environ.get('TMPDIR', None) is None:
                os.environ['TMPDIR'] = "/var/tmp"
                imp.reload(tempfile)
            self._output_file = tempfile.NamedTemporaryFile(prefix="Profile_",
                                                            suffix='.json',
                                                            delete=False).name
            LOG.info("Profile output file can be located at "
                     "[%s] ", self._output_file)
        return self._output_file

    @output_file.setter
    def output_file(self, outfile):
        """ """
        self._output_file = outfile

    @staticmethod
    def _write_to_json_file(resultdict, output_file):
        """ Unbound method to create output json file."""
        if os.path.isfile(output_file):
            read_dict = None
            with SYNC_LOCK:
                with open(output_file, "r") as rfile:
                    read_data = rfile.read()
                    if read_data.strip():
                        read_dict = json.loads(str(read_data))
                        read_dict = read_dict.copy()
                        resultdict.update(read_dict)

                with open(output_file, "w+") as lval_file:
                    json.dump(resultdict, lval_file, indent=4,
                              sort_keys=True, separators=(',', ':'))
        else:
            with open(output_file, "w+") as lval_file:
                json.dump(resultdict, lval_file, indent=4,
                          sort_keys=True, separators=(',', ':'))

    # Context manager decorator
    def __call__(self, func):
        """ """
        self._func = func
        # Trace function must have three arguments: frame, event, and arg.
        # frame: Current stack frame.
        # event: This is a string: 'call', 'line', 'return', 'exception'.
        # arg  : Depends on the event type. For "return", arg is the value
        # to be returned.

        @wraps(func)
        def wrapper(*args, **kwargs):
            """ wrapper """
            #pylint: disable=unused-argument
            def profiler(frame, event, arg):
                """ Define an action. """
                # We are concerned with reading local variable
                # only when it is 'return', else ignore it
                if event == 'call':
                    self._local_vars = {}
                # The trace function is invoked with an event set to 'return';
                # A function (or other code block) is about to return.
                elif event == 'return':
                    # Frame objects represent execution frames.f_locals is the
                    # dictionary used to look up local variables
                    self._local_vars = frame.f_locals.copy()
                # The local trace function is called; arg is the value that
                # will be returned, or None if the event is caused by an
                # exception being raised. The trace function's return value
                # is ignored.
                elif event == 'exception':
                    # exception, value, traceback
                    pass
                # End of profiler

            assert sys.getprofile() is None, (
                "Profiling function is already set.")
            # Tracer is activated on next call, return or exception
            sys.setprofile(profiler)

            result_dict = {}
            with self:
                # trace the function call
                res = self._func(*args, **kwargs)

                td = datetime.today()
                timestamp = str(td.strftime('%m_%d_%YT%H:%M:%S:%f'))
                result_dict[timestamp] = {}
                result_dict[timestamp]["name"] = func.__name__

                for tvar in self._trace_vars:
                    result_dict[str(td.strftime('%m_%d_%YT%H:%M:%S:%f'))][
                        tvar] = self._local_vars[tvar.strip()]

                self._write_to_json_file(result_dict, self.output_file)

            return res

        return wrapper


@Profiler("train_time", "loss", "max_steps")
def test_profiler():
    """ unittest method """
    import time
    # dummy calc
    loss = 24.732345
    start_time = time.time()
    max_steps = 8
    time.sleep(5)
    train_time = time.time() - start_time
    return (start_time, loss, train_time, max_steps)


if __name__ == '__main__':
    test_profiler()
