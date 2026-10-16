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
    GetInstrTraceRequest,
    GetInstrTraceResponse,
)
from cerebras.pb.viz.instr_trace_response_pb2 import InstrTraceSequence
from cerebras.sdk.debug.lib.symbol.csldebugpybind import (
    SimfabInstructionTraceCtf,
)
from cerebras.sdk.debug.shell_lib.trace import Trace
from cerebras.sdk.sdk_debug_instr_trace import InstrTrace, InstrTraceField
from cerebras.sdk.sdk_debug_shell import SIMFAB_JSON_LOG


class Plugin(PluginBase):
    ALIAS = 'sdk/instruction_trace/trace'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug(f"%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        req = GetInstrTraceRequest()
        query_string_to_request(qs, req)

        check_trace_params(
            mgr,
            req.compile_directory,
            req.grid_x,
            req.grid_y,
            req.cycle_start,
            req.cycle_end,
        )

        resp = GetInstrTraceResponse()

        if "CSL_SIMFAB_JSON_TRACE" in os.environ:
            simfab_log = os.path.join(req.run_directory, SIMFAB_JSON_LOG)
            elf_dir = get_elf_dir(req.compile_directory)
            instrTrace = InstrTrace(elf_dir, simfab_log)

            if req.task_id >= 0:
                itr = instrTrace.get_trace_per_task_iterator(
                    req.grid_x, req.grid_y, req.task_id
                )
            elif req.uthread_id >= 0:
                itr = instrTrace.get_trace_per_uthread_iterator(
                    req.grid_x, req.grid_y, req.uthread_id
                )
            else:
                itr = instrTrace.get_trace_iterator(req.grid_x, req.grid_y)

            check_cycles = False
            if req.cycle_start > 0 or req.cycle_end > 0:
                check_cycles = True

            if req.cycle_start > 0 and req.cycle_end == 0:
                req.cycle_end = sys.maxsize

            for values in itr:
                cycle = values[InstrTraceField.cycle.value]
                if check_cycles and (
                    cycle < req.cycle_start or cycle > req.cycle_end
                ):
                    continue

                sequence = InstrTraceSequence()
                sequence.cycle = cycle
                sequence.op_addr = values[InstrTraceField.opaddr.value]
                sequence.op_name = values[InstrTraceField.opname.value]
                sequence.dest = values[InstrTraceField.dest.value]
                sequence.src0 = values[InstrTraceField.src0.value]
                sequence.src1 = values[InstrTraceField.src1.value]
                sequence.src2 = values[InstrTraceField.src2.value]

                source_line = values[InstrTraceField.source_line.value]
                sequence.source = source_line.source_file
                sequence.source_line = source_line.line
                sequence.task = values[InstrTraceField.task.value]
                resp.data.append(sequence)
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
            # filter by cycle_start, cycle_end, and max number of trace should
            # be handled by C++, ignore for now
            if req.task_id >= 0:
                resp = instr_trace.get_instruction_trace_per_task_at(
                    coor, req.task_id
                )
            elif req.uthread_id >= 0:
                resp = instr_trace.get_instruction_trace_per_uthread_at(
                    coor, req.uthread_id
                )
            else:
                resp = instr_trace.get_instruction_trace_at(
                    coor, req.cycle_start, req.cycle_end
                )

        self.return_msg(resp)
        return True
