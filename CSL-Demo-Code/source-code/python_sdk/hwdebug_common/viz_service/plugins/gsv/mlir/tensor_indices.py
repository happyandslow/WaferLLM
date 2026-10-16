"""
api/gsv/mlir/tensor_indices
Return tensor indices for a specified layout
"""

from pathlib import Path

from cerebras.das_common.das_commonpybind import Coordinate
from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.hwdebug_common.viz_service.plugins.gsv.mlir.common import (
    DBG_RTIR,
    validate_layout_request,
)
from cerebras.pb.viz.mlir_view_pb2 import TensorIndicesRequest
from cerebras.ws.hwdebug.lib.mlirviewpybind import tensor_indices


class Plugin(PluginBase):
    ALIAS = 'gsv/mlir/tensor_indices'

    def handle_get(self, handler, url, req, qs):
        """parse the v1 request and dispatch as appropriate"""
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = TensorIndicesRequest()
        query_string_to_request(qs, req)

        error = validate_layout_request(req.chief_directory)
        if error:
            self.return_validation_error(error)
            return True

        dbg_rtir_path = Path(req.chief_directory) / DBG_RTIR
        resp = tensor_indices(
            str(dbg_rtir_path),
            req.config_id,
            req.operand_number,
            Coordinate(req.pe_x, req.pe_y),
        )
        self.return_msg(resp)

        return True
