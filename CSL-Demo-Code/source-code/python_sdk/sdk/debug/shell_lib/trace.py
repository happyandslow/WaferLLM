import logging
import os
from pathlib import Path
from typing import Optional

import cerebras.sdk.debug.lib.wavelet_trace.wavelettracepybind as wavelettracepybind
from cerebras.hwdebug_common.shell_lib.utils import (
    parse_index_input,
    rglob_file,
)
from cerebras.sdk.debug.lib.symbol.csldebugpybind import SimfabWaveletTraceCtf


class Trace:
    # Constants for SDK simulator
    SIMFAB_JSON_LOG = "simfab_traces.json"
    SIMFAB_CTF_DIR = "simfab_traces"
    if "CSL_SIMFAB_JSON_TRACE" in os.environ:
        SIMFAB_LOG = SIMFAB_JSON_LOG
    else:
        SIMFAB_LOG = SIMFAB_CTF_DIR

    def __init__(self, logger: Optional[logging.Logger]):
        self._logger = logger
        self._trace_dir_list = []
        self._absolute_trace_path = None  # Type: Path

        # Library that reads the trace
        self._wavelet_trace = None
        self._wavelet_timeline = None

    def init_wavelet_trace(self, trace_dir_path: Path):
        assert trace_dir_path is not None

        self._simfab_log = str(trace_dir_path / self.SIMFAB_LOG)

        self._wavelet_timeline = wavelettracepybind.SimfabWaveletTimeline(self._simfab_log)

        self._wavelet_trace = SimfabWaveletTraceCtf(self._simfab_log)

        # TODO: get elf_dir and initialize instruction trace

    def simfab_log(self):
        return self._simfab_log

    def wavelet_timeline(self):
        return self._wavelet_timeline

    def wavelet_trace(self):
        return self._wavelet_trace

    def instruction_trace(self):
        return self._instruction_trace

    def absolute_trace_path(self) -> Path:
        return self._absolute_trace_path

    def trace_dir(self):
        """
        Returns the internal cached list of post-run directories found by list_trace.
        """
        return self._trace_dir_list

    def list_trace(self, workdir: Path, display_to_stdout: bool) -> int:
        """
        Cache a list of simulator traces relative path to workdir.
        """
        self._trace_dir_list = list(rglob_file(workdir, self.SIMFAB_LOG))

        # Remove file name and just keep parent directory
        for idx in range(len(self._trace_dir_list)):
            self._trace_dir_list[idx] = self._trace_dir_list[idx].parent

        # Display to screen for user
        if display_to_stdout and self._logger:
            for idx, trace_dir in enumerate(self._trace_dir_list):
                self._logger.info(" [%d] %s", idx, trace_dir)

        return len(self._trace_dir_list)

    def select_trace(self, tracedir_idx: str, workdir: Path) -> int:
        """
        Select the trace specified by user. Use any value is trace list
        """
        tracedir = tracedir_idx
        # User can pass in an index after context list was invoked to type less.
        if tracedir_idx.endswith("]"):
            # Ask the user to cache the list first
            if len(self._trace_dir_list) == 0:
                return -2
            # Parse and error check user inputs
            index = parse_index_input(tracedir_idx)
            if index < 0:
                self._logger.warn(" Invalid input %s.", tracedir_idx)
                return -3
            elif index >= len(self._trace_dir_list):
                self._logger.warn(
                    " Index %s exceeds the trace list.", tracedir_idx
                )
                return -4

            tracedir = self._trace_dir_list[index]

        tracedir_path = Path(tracedir)
        # Prepend workdir if path given is a relative path.
        if not tracedir_path.is_absolute():
            tracedir_path = Path(workdir) / tracedir_path

        if Path(tracedir_path).is_dir():
            self._absolute_trace_path = tracedir_path.resolve()
            self.init_wavelet_trace(self._absolute_trace_path)
            return 0

        self._logger.warn(" %s does not exist.", tracedir)
        return -1

    def show_trace(self):
        if self._absolute_trace_path is None:
            self._logger.info(" Trace: no selected.")
            return

        self._logger.info(" Trace: %s", self._absolute_trace_path)
