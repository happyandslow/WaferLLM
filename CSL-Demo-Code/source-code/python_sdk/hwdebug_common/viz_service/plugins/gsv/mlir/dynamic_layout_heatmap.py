"""
api/gsv/mlir/dynamic_layout_heatmap
Return heatmap for a tensor layout string
"""

from pathlib import Path

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.hwdebug_common.viz_service.plugins.gsv.mlir.common import (
    DBG_RTIR,
    validate_layout_request,
)
from cerebras.pb.viz.mlir_view_pb2 import DynamicLayoutHeatmapRequest
from cerebras.ws.hwdebug.lib.mlirviewpybind import dynamic_layout_heatmap


class Plugin(PluginBase):
    ALIAS = 'gsv/mlir/dynamic_layout_heatmap'

    def handle_get(self, handler, url, req, qs):
        """parse the v1 request and dispatch as appropriate"""
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = DynamicLayoutHeatmapRequest()
        query_string_to_request(qs, req)

        error = validate_layout_request(req.chief_directory)
        if error:
            self.return_validation_error(error)
            return True

        dbg_rtir_path = Path(req.chief_directory) / DBG_RTIR
        if req.HasField("dimension"):
            json_str = dynamic_layout_heatmap(
                str(dbg_rtir_path), req.layout_str, req.dimension
            )
        else:
            json_str = dynamic_layout_heatmap(
                str(dbg_rtir_path), req.layout_str, None
            )
        self.return_json(json_str, compress=True)

        return True
