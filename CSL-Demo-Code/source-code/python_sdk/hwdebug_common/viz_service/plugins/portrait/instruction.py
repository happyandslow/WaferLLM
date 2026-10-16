import os
import sys
from pathlib import Path

from cerebras.das_common.das_commonpybind import Coordinate
from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import (
    GetInstrTraceRequest,
)
from cerebras.pb.viz.instr_trace_response_pb2 import InstrTraceSequence
from cerebras.ws.hwdebug.lib.wsdebugpybind import (
    SimfabInstructionTraceCtf,
)

class Plugin(PluginBase):
    ALIAS = 'portrait/instruction'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug(f"%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        req = GetInstrTraceRequest()
        query_string_to_request(qs, req)

        ctf_dir = str(Path(req.run_directory) / "simfab_traces")
        instr_trace = SimfabInstructionTraceCtf(req.compile_directory, ctf_dir)
        coor = Coordinate(req.grid_x, req.grid_y)
        
        # filter by cycle_start, cycle_end 
        resp = instr_trace.get_instruction_trace_at(
            coor, req.context_id, req.cycle_start, req.cycle_end
        )

        self.return_msg(resp)
        return True
