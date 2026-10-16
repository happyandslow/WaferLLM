"""
api/gsv/dataflow/nav_tree
return the nav tree for dataflow view
"""

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.hwdebug_common.viz_service.plugins.gsv.validation import (
    find_gsv_dir,
    validate_dataflow_view,
    validate_gsv_workdir,
)
from cerebras.pb.viz.viz_global_stall_view_pb2 import WsGSVNavTreeRequest
from cerebras.ws.hwdebug.lib.wsvizpybind import WsGlobalStallViz, WsGSVType


class Plugin(PluginBase):
    ALIAS = 'gsv/dataflow/nav_tree'

    def handle_get(self, handler, url, req, qs):
        """parse the v1 request and dispatch as appropriate"""
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = WsGSVNavTreeRequest()
        query_string_to_request(qs, req)

        gsv_dir = find_gsv_dir(req.work_directory)
        err = validate_gsv_workdir(gsv_dir)
        err = err or validate_dataflow_view(gsv_dir)

        if err:
            self.return_validation_error(err)
            return True

        helper = WsGlobalStallViz(gsv_dir, WsGSVType.DATAFLOW_VIEW, None)
        resp = helper.nav_tree()

        self.return_msg(resp)
        return True
