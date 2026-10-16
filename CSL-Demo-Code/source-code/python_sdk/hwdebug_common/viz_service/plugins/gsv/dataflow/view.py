"""
api/gsv/dataflow/view
return the netlist for global stall
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
from cerebras.pb.viz.viz_global_stall_view_pb2 import (
    VizWsGlobalStallViewRequest,
)
from cerebras.ws.hwdebug.lib.wsvizpybind import WsGlobalStallViz, WsGSVType


class Plugin(PluginBase):
    ALIAS = 'gsv/dataflow/view'

    def handle_get(self, handler, url, req, qs):
        """parse the v1 request and dispatch as appropriate"""
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = VizWsGlobalStallViewRequest()
        query_string_to_request(qs, req)

        gsv_dir = find_gsv_dir(req.work_directory)
        err = validate_gsv_workdir(gsv_dir)

        if req.view == 'DATAFLOW_VIEW':
            gsvType = WsGSVType.DATAFLOW_VIEW
            err = err or validate_dataflow_view(gsv_dir)
        else:
            raise Exception('view should be DATAFLOW_VIEW')

        if err:
            self.return_validation_error(err)
            return True

        group_ids = None
        if req.HasField('group_ids'):
            group_ids_csv = getattr(req, 'group_ids')
            group_ids = [
                int(item)
                for item in group_ids_csv.split(',')
                if item.strip().isdigit()
            ]
            if len(group_ids) == 0:
                raise Exception('group_ids should be comma separated numbers')

        helper = WsGlobalStallViz(gsv_dir, gsvType, group_ids)
        resp = helper.dataflow_graph()
        self.return_msg(resp)
        return True
