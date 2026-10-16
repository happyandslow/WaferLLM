from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.sdk.instruction_trace.utils import (
    check_trace_params,
)
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import (
    GetInstrTraceOperandFormatsRequest,
    GetInstrTraceOperandFormatsResponse,
)


class Plugin(PluginBase):
    ALIAS = 'sdk/instruction_trace/operand_formats'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        req = GetInstrTraceOperandFormatsRequest()
        query_string_to_request(qs, req)

        check_trace_params(
            mgr, req.compile_directory, req.grid_x, req.grid_y, 0, 0
        )

        resp = GetInstrTraceOperandFormatsResponse()

        # available formats are hardcoded for now
        formats = ['i16', 'f16', 'f32', 'hex']

        resp.formats.extend(formats)

        self.return_msg(resp)
        return True
