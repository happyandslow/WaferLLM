"""
api/sdk/artifact_locations
return the artifact locations
"""
from pathlib import Path

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.hwdebug_common.shell_lib.target import Target
from cerebras.pb.viz.default_service_pb2 import (
    GetArtifactLocationsRequest,
    GetArtifactLocationsResponse,
)
from cerebras.sdk.debug.shell_lib.csl_debug_context import CslDebugContext
from cerebras.sdk.debug.shell_lib.trace import Trace


class Plugin(PluginBase):
    ALIAS = 'sdk/artifact_locations'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        # convert incoming json to the request message
        req = GetArtifactLocationsRequest()
        query_string_to_request(qs, req)

        # workdir == directory that contains all the artifacts
        initial_work_directory = Path(self.workdir).absolute()
        work_directory = Path(req.work_directory)
        if not work_directory.is_absolute():
            work_directory = initial_work_directory / work_directory

        if not work_directory.is_dir():
            self.return_validation_error('Work directory does not exist')
            return True

        # get compile directories using CslDebugContext
        debug_context = CslDebugContext(self.logger)
        debug_context.list_context(work_directory, False)
        compile_directories = debug_context.context()

        run_context = Trace(self.logger)
        run_context.list_trace(work_directory, False)
        run_directories = run_context.trace_dir()

        target = Target(self.logger, "")
        target.list_target(work_directory, False, Target.COREFILE_ONLY)
        core_file_paths = target.checkpoint_list()

        # form the proper response
        resp = GetArtifactLocationsResponse()
        resp.work_directory = str(work_directory)
        for compile_directory in compile_directories:
            resp.compile_directories.append(str(compile_directory))

        for run_directory in run_directories:
            resp.run_directories.append(str(run_directory))
        for core_file_path in core_file_paths:
            resp.core_file_paths.append(str(core_file_path))
        resp.source_directory = "."

        self.return_msg(resp)
        return True
