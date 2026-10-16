"""
api/gsv/summary/tags_delete
delete tag
"""

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.hwdebug_common.viz_service.plugins.gsv.validation import (
    find_gsv_dir,
)
from cerebras.pb.viz.viz_global_stall_view_pb2 import VizWsGSVDeleteTagRequest
from cerebras.ws.hwdebug.lib.wsvizpybind import WsGlobalStallViz, WsGSVType


class Plugin(PluginBase):
    ALIAS = 'gsv/summary/tags_delete'

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = VizWsGSVDeleteTagRequest()
        query_string_to_request(qs, req)

        gsv_dir = find_gsv_dir(req.work_directory)
        helper = WsGlobalStallViz(gsv_dir, WsGSVType.SUMMARY_VIEW, None)
        helper.delete_tag(req.key)
        self.return_json({})
        return True
