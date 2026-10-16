import os
from pathlib import Path

from cerebras.das_common.das_commonpybind import Coordinate
from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.sdk.utils import (
    check_params,
)
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import GetWaveletTraceRequest
from cerebras.pb.viz.wavelet_trace_response_pb2 import (
    WaveletDataTypeResponse,
    WaveletDataTypeResponseWavelets,
)
from cerebras.sdk.sdk_debug_shell import SIMFAB_JSON_LOG
from cerebras.sdk.sdk_debug_wavelet_trace import WaveletTrace
from cerebras.sdk.debug.shell_lib.trace import Trace


class Plugin(PluginBase):
    ALIAS = 'sdk/wavelet_trace'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        # convert incoming json to the request message
        req = GetWaveletTraceRequest()
        query_string_to_request(qs, req)

        if req.compile_directory:
            check_params(mgr, req.compile_directory, req.grid_x, req.grid_y)

        resp = WaveletDataTypeResponse()
        resp.wavelet_data_type = req.wavelet_display_format

        # form the proper response
        if "CSL_SIMFAB_JSON_TRACE" in os.environ:
            simfab_log = os.path.join(req.run_directory, SIMFAB_JSON_LOG)
            wt = WaveletTrace(simfab_log)

            if req.colors:
                it = wt.get_trace_iterator_per_colors(
                    req.grid_x, req.grid_y, req.colors, req.wavelet_display_format
                )
            else:
                it = wt.get_trace_iterator(
                    req.grid_x, req.grid_y, req.wavelet_display_format
                )

            check_cycle = req.cycle_start > 0 or req.cycle_end > 0

            fetched, returned = 0, 0
            # ident of wvlt is available, but not used
            for cycle, color, _, lnk, ctrl, data in it:
                if check_cycle:
                    if cycle < req.cycle_start or cycle > req.cycle_end > 0:
                        continue
                fetched += 1
                if fetched < req.offset:
                    continue
                returned += 1
                if req.limit > 0 and returned > req.limit:
                    break
                wvlt_proto = WaveletDataTypeResponseWavelets()
                wvlt_proto.cycle = cycle
                wvlt_proto.color = color
                wvlt_proto.link = lnk
                wvlt_proto.ctrl = ctrl
                data = data.split(',')
                if req.wavelet_display_format == 'f32':
                    wvlt_proto.wavelet_data_bits = data[0]
                else:
                    wvlt_proto.wavelet_header_bits = data[0]
                    wvlt_proto.wavelet_data_bits = data[1]
                resp.wavelets.append(wvlt_proto)

            self.return_msg(resp)
            return True
        else:
            # CTF trace
            # TODO: trace should be initialized by mgr
            run_dir = Path(req.run_directory)
            trace = mgr.run_directory.get(run_dir, None)
            if trace is None:
                trace = Trace(self.logger)
                if trace.list_trace(run_dir, False) == 1:
                    trace.select_trace('[0]', run_dir)
                else:
                    self.logger.warning("Found more than 1 trace")
                    raise RuntimeError("Found more than 1 trace")
                mgr.run_directory[run_dir] = trace

            wavelet_trace = trace.wavelet_trace()
            coor = Coordinate(req.grid_x, req.grid_y)
            # filter by cycle_start, cycle_end, and max number of trace should
            # be handled by C++, ignore for now
            if req.colors:
                wavelets = wavelet_trace.get_wavelet_trace_per_colors_at(coor, req.colors, req.wavelet_display_format)
            else:
                wavelets = wavelet_trace.get_wavelet_trace_at(coor, req.wavelet_display_format)

            resp.wavelets.extend(wavelets)

            self.return_msg(resp)
            return True


