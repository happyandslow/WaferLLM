import os
from pathlib import Path

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import GetWaveletTimelineRequest
from cerebras.sdk.debug.lib.wavelet_trace import (
    wavelettracepybind as wvlttracepybind,
)
from cerebras.sdk.debug.shell_lib.trace import Trace
from cerebras.sdk.sdk_debug_shell import SIMFAB_JSON_LOG


class Plugin(PluginBase):
    ALIAS = 'sdk/timeline/wavelet'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        # convert incoming json to the request message
        req = GetWaveletTimelineRequest()
        query_string_to_request(qs, req)

        x = req.grid_x
        y = req.grid_y
        color = req.color
        cycle_start = req.cycle_start if req.cycle_start else 0
        cycle_end = req.cycle_end if req.cycle_end else 0

        if "CSL_SIMFAB_JSON_TRACE" in os.environ:
            simfab_log = os.path.join(req.run_directory, SIMFAB_JSON_LOG)
            simfab_timeline = wvlttracepybind.SimfabWaveletTimeline(simfab_log)
            resp = simfab_timeline.get_wvlt_and_bkpr_timeline_proto(
                x, y, color, cycle_start, cycle_end
            )
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

            timeline_trace = trace.wavelet_timeline()
            resp = timeline_trace.get_wvlt_and_bkpr_timeline_proto(
                x, y, color, cycle_start, cycle_end
            )

        self.return_msg(resp)
        return True
