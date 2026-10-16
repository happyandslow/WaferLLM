"""
api/gsv/mlir/mlir_op_operand
Return a list of operand numbers for a given config ID
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
from cerebras.pb.viz.mlir_view_pb2 import MlirOpOperandRequest
from cerebras.ws.hwdebug.lib.mlirviewpybind import mlir_op_operand


class Plugin(PluginBase):
    ALIAS = 'gsv/mlir/mlir_op_operand'

    def handle_get(self, handler, url, req, qs):
        """parse the v1 request and dispatch as appropriate"""
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = MlirOpOperandRequest()
        query_string_to_request(qs, req)

        error = validate_layout_request(req.chief_directory)
        if error:
            self.return_validation_error(error)
            return True

        dbg_rtir_path = Path(req.chief_directory) / DBG_RTIR
        resp = mlir_op_operand(str(dbg_rtir_path), req.config_id)
        if resp:
            self.return_json(resp, compress=False)
        else:
            self.return_validation_error(
                {
                    "message": "The specified config ID doesn't exist",
                    "repair": False,
                }
            )

        return True
