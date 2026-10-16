from pathlib import Path

from cerebras.das_common.das_commonpybind import Coordinate
from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.sdk.utils import (
    check_params,
    get_elf_dir,
)
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import GetInstrTimelineRequest
from cerebras.sdk.debug.lib.symbol.csldebugpybind import (
    get_instruction_timeline,
)
from cerebras.sdk.debug.shell_lib.trace import Trace


class Plugin(PluginBase):
    ALIAS = 'sdk/timeline/instruction'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        # convert incoming json to the request message
        req = GetInstrTimelineRequest()
        query_string_to_request(qs, req)

        check_params(mgr, req.compile_directory, req.grid_x, req.grid_y)

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
        ctf_trace_dir = str(trace.absolute_trace_path() / Trace.SIMFAB_CTF_DIR)

        coor = Coordinate(req.grid_x, req.grid_y)
        # TODO: trace should know elf_dir
        elf_dir = get_elf_dir(req.compile_directory)
        resp = get_instruction_timeline(
            ctf_trace_dir,
            elf_dir,
            coor,
            req.cycle_start,
            req.cycle_end,
        )

        self.return_msg(resp)
        return True
