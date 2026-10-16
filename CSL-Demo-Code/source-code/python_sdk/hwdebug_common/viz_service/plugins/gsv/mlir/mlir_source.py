"""
api/gsv/mlir/mlir_source
"""

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.hwdebug_common.viz_service.plugins.gsv.mlir.common import (
    get_mlir_path,
    validate_mlir_source_request,
)
from cerebras.pb.viz.mlir_view_pb2 import MlirSourceRequest
from cerebras.ws.hwdebug.lib.mlirviewpybind import mlir_source


class Plugin(PluginBase):
    ALIAS = 'gsv/mlir/mlir_source'

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = MlirSourceRequest()
        query_string_to_request(qs, req)

        error = validate_mlir_source_request(req.chief_directory)
        if error:
            self.return_validation_error(error)
            return True

        mlir_path = get_mlir_path(req.chief_directory)
        resp = mlir_source(mlir_path)
        dict = self.msg_to_dict(resp)
        self.return_json(dict, compress=True)

        return True
