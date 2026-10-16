"""
api/sdk/artifact_locations
return the artifact locations
"""

from pathlib import Path

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import (
    GetSettingsRequest,
    GetSettingsResponse,
)


class Plugin(PluginBase):
    ALIAS = 'sdk/save_settings'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        # convert incoming json to the request message
        req = GetSettingsRequest()
        query_string_to_request(qs, req)

        compile_directory = req.compile_directory
        run_directory = req.run_directory
        core_file_path = req.core_file_path
        source_directory = req.source_directory

        # form the proper response
        resp = GetSettingsResponse()
        resp.compile_directory = ""
        resp.run_directory = ""
        resp.core_file_path = ""
        resp.source_directory = ""

        if not Path(compile_directory).is_dir():
            resp.compile_directory = "Compile directory does not exist"

        if run_directory != "" and not Path(run_directory).is_dir():
            resp.run_directory = "Run directory does not exist"

        if core_file_path != "" and not Path(core_file_path).is_file():
            resp.core_file_path = "Core file does not exist"

        if source_directory != "" and not Path(source_directory).is_dir():
            resp.source_directory = "Source directory does not exist"

        self.return_msg(resp)
        return True
