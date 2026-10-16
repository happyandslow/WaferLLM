"""
api/gsv/mlir/mlir_graph
"""

from pathlib import Path

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.hwdebug_common.viz_service.plugins.gsv.mlir.common import (
    DBG_RTIR,
    validate_mlir_graph_request,
)
from cerebras.pb.viz.mlir_view_pb2 import MlirGraphRequest
from cerebras.ws.hwdebug.lib.mlirviewpybind import mlir_graph


class Plugin(PluginBase):
    ALIAS = 'gsv/mlir/mlir_graph'

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = MlirGraphRequest()
        query_string_to_request(qs, req)

        error = validate_mlir_graph_request(req.chief_directory)
        if error:
            self.return_validation_error(error)
            return True

        try:
            dbg_rtir_path = Path(req.chief_directory) / DBG_RTIR
            resp = mlir_graph(str(dbg_rtir_path), req.config_id, req.max_level)
        except RuntimeError as e:
            self.return_runtime_error(e)
            return True

        dict = self.msg_to_dict(resp)
        self.return_json(dict)

        return True
