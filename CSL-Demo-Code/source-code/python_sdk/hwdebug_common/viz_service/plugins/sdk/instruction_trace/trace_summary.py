import os
import sys
from pathlib import Path

from cerebras.das_common.das_commonpybind import Coordinate
from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.sdk.instruction_trace.utils import (
    check_trace_params,
)
from cerebras.hwdebug_common.viz_service.plugins.sdk.utils import (
    get_elf_dir,
)
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import (
    GetInstrTraceSummaryRequest,
    GetInstrTraceSummaryResponse,
)
from cerebras.pb.viz.instr_trace_response_pb2 import InstrTraceSummaryResponse
from cerebras.sdk.debug.lib.symbol.csldebugpybind import (
    SimfabInstructionTraceCtf,
)
from cerebras.sdk.debug.shell_lib.trace import Trace
from cerebras.sdk.sdk_debug_instr_trace import InstrTrace, InstrTraceField
from cerebras.sdk.sdk_debug_shell import SIMFAB_JSON_LOG


class Plugin(PluginBase):
    ALIAS = 'sdk/instruction_trace/trace_summary'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug(f"%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        req = GetInstrTraceSummaryRequest()
        query_string_to_request(qs, req)

        check_trace_params(
            mgr,
            req.compile_directory,
            req.grid_x,
            req.grid_y,
            req.cycle_start,
            req.cycle_end,
        )

        if "CSL_SIMFAB_JSON_TRACE" in os.environ:
            simfab_log = os.path.join(req.run_directory, SIMFAB_JSON_LOG)
            elf_dir = get_elf_dir(req.compile_directory)
            instrTrace = InstrTrace(elf_dir, simfab_log)

            # We don't have a cheap way to obtain the trace summary yet.
            # So we iterate through all traces and compute the summary.
            itr = instrTrace.get_trace_iterator(req.grid_x, req.grid_y)

            check_cycles = False
            if req.cycle_start > 0 or req.cycle_end > 0:
                check_cycles = True

            if req.cycle_start > 0 and req.cycle_end == 0:
                req.cycle_end = sys.maxsize

            total_cycles = 0
            cycles_per_instr = {}
            for values in itr:
                cycle = values[InstrTraceField.cycle.value]
                if check_cycles and (
                    cycle < req.cycle_start or cycle > req.cycle_end
                ):
                    continue

                total_cycles += 1
                opaddr = values[InstrTraceField.opaddr.value]
                if opaddr in cycles_per_instr:
                    (cycles, name) = cycles_per_instr[opaddr]
                    cycles += 1
                    cycles_per_instr[opaddr] = (cycles, name)
                else:
                    cycles_per_instr[opaddr] = (
                        1,
                        values[InstrTraceField.opname.value],
                    )

            sorted_cycles_per_instr = dict(
                sorted(
                    cycles_per_instr.items(),
                    reverse=True,
                    key=lambda item: item[1],
                )
            )

            resp = GetInstrTraceSummaryResponse()
            for opaddr, value in sorted_cycles_per_instr.items():
                (cycles, opname) = value
                summary = InstrTraceSummaryResponse()
                summary.cycles = cycles
                summary.cycles_percentage = (cycles / total_cycles) * 100
                summary.op_addr = opaddr
                summary.instruction = opname

                resp.data.append(summary)
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

            # TODO: make elf_dir should be known to trace, and get instr_trace
            # from trace
            elf_dir = get_elf_dir(req.compile_directory)
            instr_trace = SimfabInstructionTraceCtf(elf_dir, trace.simfab_log())
            coor = Coordinate(req.grid_x, req.grid_y)
            resp = instr_trace.get_instruction_trace_summary(
                coor, req.cycle_start, req.cycle_end
            )

        self.return_msg(resp)
        return True
