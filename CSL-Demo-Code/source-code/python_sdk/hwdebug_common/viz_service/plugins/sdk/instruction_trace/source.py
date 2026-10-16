from pathlib import Path

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import (
    GetInstrTraceSourceRequest,
    GetInstrTraceSourceResponse,
)


class Plugin(PluginBase):
    ALIAS = 'sdk/instruction_trace/source'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug(f"%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = GetInstrTraceSourceRequest()
        query_string_to_request(qs, req)

        resp = GetInstrTraceSourceResponse()

        # Handle this special case
        if req.source_file == "?":
            resp.lines.append(f"No source line")
            body = VizMessageToJson(resp)
            self.return_json(body)
            return True

        # Get source file
        source_file_path = Path(req.source_file)
        if not source_file_path.is_absolute():
            source_directory = Path(req.source_directory)
            # Validate source directory
            if not source_directory.exists():
                self.return_validation_error(
                    f'Source directory {req.source_directory} does not exist'
                )
                return True
            source_file_path = source_directory / source_file_path
        source_file = str(source_file_path)

        # Validate source file
        if not source_file_path.exists():
            self.return_validation_error(
                f'Source file {req.source_file} does not exist'
            )
            return True

        # Populate the response lines with source file.
        with open(source_file) as source_file_lines:
            for line in source_file_lines:
                resp.lines.append(line.rstrip())

        self.return_msg(resp)
        return True
