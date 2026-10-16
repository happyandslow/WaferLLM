import os
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
    GetInstrTraceTasksRequest,
    GetInstrTraceTasksResponse,
)
from cerebras.pb.viz.instr_trace_response_pb2 import InstrTraceTask
from cerebras.sdk.debug.lib.symbol.csldebugpybind import (
    SimfabInstructionTraceCtf,
)
from cerebras.sdk.debug.shell_lib.trace import Trace
from cerebras.sdk.sdk_debug_instr_trace import InstrTrace
from cerebras.sdk.sdk_debug_shell import SIMFAB_JSON_LOG
from cerebras.hwdebug_common.viz_service.plugins.sdk.utils import get_compile_cache

class Plugin(PluginBase):
    ALIAS = 'sdk/instruction_trace/tasks'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug(f"%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        req = GetInstrTraceTasksRequest()
        query_string_to_request(qs, req)

        check_trace_params(
            mgr,
            req.compile_directory,
            req.grid_x,
            req.grid_y,
            req.cycle_start,
            req.cycle_end,
        )

        # form the proper response
        if "CSL_SIMFAB_JSON_TRACE" in os.environ:
            simfab_log = os.path.join(req.run_directory, SIMFAB_JSON_LOG)
            elf_dir = get_elf_dir(req.compile_directory)
            instrTrace = InstrTrace(elf_dir, simfab_log)

            resp = GetInstrTraceTasksResponse()
            tasks = instrTrace.get_tasks(req.grid_x, req.grid_y)

            for task_id in tasks:
                task = InstrTraceTask()
                task.id = task_id
                resp.tasks.append(task)
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
            resp = instr_trace.get_tasks_at(coor)

            compile_cache = get_compile_cache(mgr, req.compile_directory)
            task_map = compile_cache.get_task_map(req.grid_x, req.grid_y)
            for task in resp.tasks:
                if task.id in task_map:
                    task.name = task_map[task.id]

        self.return_msg(resp)
        return True
