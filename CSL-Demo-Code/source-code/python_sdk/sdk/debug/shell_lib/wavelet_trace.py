import logging
from io import TextIOWrapper
from pathlib import Path
from typing import List, Optional

from tabulate import tabulate

from cerebras.das_common.das_commonpybind import Coordinate
from cerebras.hal.halpybind import Rectangle

# TODO: Factor into C++ to avoid duplicated code
# use same mapping as simfabric:
# https://github.com/Cerebras/monolith/blob/master/src/fabricmodel/frame/fnet.h#L33
LINK_INT_TO_CHAR = {
    0: 'E',
    1: 'N',
    2: 'W',
    3: 'S',
    4: 'R',
}


class WaveletTrace:
    WAVELET_FORMATS = ['i16', 'f16', 'f32', 'i16,f16', 'hex']
    FAB_NUM_DATA_COLORS = 24

    def __init__(self, logger: Optional[logging.Logger]):
        self._logger = logger

    def _get_link_char(self, link):
        """
        convert wavelet link int to char
        """
        link_char = LINK_INT_TO_CHAR.get(link, None)
        assert link_char is not None, f'Invalid link {link}'
        return link_char

    def get_wavelet_timeline(
        self, simfab_timeline, x: int, y: int, color: int,
    ):
        """
        Get the wavelet timeline trace for PE (x, y) and color
        """
        return simfab_timeline.get_wvlt_timeline(x, y, color)

    def _write_timeline_to_file(
        self,
        outfile: TextIOWrapper,
        simfab_timeline,
        x: int,
        y: int,
        color: int,
        simfab_trace: str,
    ) -> int:
        """
        Returns the number of wavelet timelines written
        """
        timelines = self.get_wavelet_timeline(
            simfab_timeline, x, y, color
        )
        rows = (
            [
                w.sent_cycle,
                w.ident,
                self._get_link_char(w.dir),
                w.ctrl,
                w.wvlt_idx,
                w.wvlt_data,
                w.received_cycle - w.sent_cycle,
            ]
            for w in timelines
        )
        outfile.write(
            tabulate(
                rows,
                headers=[
                    "Sent cycle",
                    "ID",
                    "Link",
                    "Ctrl",
                    "Index",
                    "Data",
                    "Duration",
                ],
            )
        )
        return len(timelines)

    def read_timeline(
        self,
        workdir: Path,
        simfab_timeline,
        input_rect: Rectangle,
        color: int,
        simfab_trace: str,
    ):
        """
        Reads the content of simfabric_trace ctf and print the wavelet timeline
        """
        assert 0 <= color < 24, f"Expects color:{color} between 0 and 23."
        assert (
            simfab_timeline is not None
        ), f"Expects timeline to be initialized"
        assert simfab_trace is not None

        ret = 0
        for y in range(input_rect.y(), input_rect.y() + input_rect.height()):
            for x in range(input_rect.x(), input_rect.x() + input_rect.width()):
                suffix = f"x{x}y{y}"
                outfile_name = f"wavelet-timeline-{suffix}-c{color}.log"
                outfile_path = workdir / outfile_name
                if self._logger is not None:
                    self._logger.info(
                        " Writing wavelet timelines to %s", outfile_path
                    )
                with open(outfile_path, "w") as outfile:
                    ret += self._write_timeline_to_file(
                        outfile, simfab_timeline, x, y, color, simfab_trace
                    )
        return ret

    def get_wavelet_trace(
        self,
        simfab_wavelet_trace_ctf,
        x: int,
        y: int,
        color: List[int],
        fmt: str,
    ):
        """
        Get the wavelet timeline trace for PE (x, y), filtered by colors (if
        passed in)
        """
        coor = Coordinate(x, y)
        if color:
            return simfab_wavelet_trace_ctf.get_wavelet_trace_per_colors_at(
                coor, color, fmt
            )
        else:
            return simfab_wavelet_trace_ctf.get_wavelet_trace_at(coor, fmt)

    def _write_wavelet_trace_to_file(
        self,
        outfile: TextIOWrapper,
        simfab_wavelet_trace_ctf,
        x: int,
        y: int,
        color: List[int],
        fmt: str,
    ) -> int:
        """
        Returns the number of wavelet written
        """
        wavelet_trace = self.get_wavelet_trace(
            simfab_wavelet_trace_ctf, x, y, color, fmt
        )
        rows = (
            [
                w.cycle,
                w.color,
                w.ctrl,
                w.link,
                w.wavelet_header_bits,
                w.wavelet_data_bits,
            ]
            for w in wavelet_trace
        )
        outfile.write(
            tabulate(
                rows,
                headers=["Cycle", "Color", "Ctrl", "Link", "Index", "Data"],
            )
        )
        return len(wavelet_trace)

    def read_wavelet_trace(
        self,
        workdir: Path,
        simfab_wavelet_trace_ctf,
        input_rect: Rectangle,
        color: List[int],
        fmt: str,
    ):
        """
        Reads the content of simfabric_trace ctf and print the wavelet trace
        """
        assert all(
            0 <= c < 24 for c in color
        ), f"All colors:{color} must be between 0 and 23."
        assert (
            simfab_wavelet_trace_ctf is not None
        ), f"Expects simfab_wavelet_trace_ctf to be initialized"

        for y in range(input_rect.y(), input_rect.y() + input_rect.height()):
            for x in range(input_rect.x(), input_rect.x() + input_rect.width()):
                suffix = f"x{x}y{y}"
                for c in color:
                    suffix += f"c{c}"
                outfile_name = f"wavelet-trace-{suffix}.log"
                outfile_path = workdir / outfile_name
                ret = 0
                if self._logger is not None:
                    self._logger.info(
                        " Writing wavelet traces to %s", outfile_path
                    )
                with open(outfile_path, "w") as outfile:
                    ret += self._write_wavelet_trace_to_file(
                        outfile, simfab_wavelet_trace_ctf, x, y, color, fmt
                    )
        return ret
