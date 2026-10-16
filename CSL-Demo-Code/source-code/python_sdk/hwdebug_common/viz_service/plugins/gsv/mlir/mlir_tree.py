"""
api/gsv/mlir/mlir_tree
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
from cerebras.pb.viz.mlir_view_pb2 import MlirTreeRequest
from cerebras.ws.hwdebug.lib.mlirviewpybind import nav_tree


class Plugin(PluginBase):
    ALIAS = 'gsv/mlir/mlir_tree'

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = MlirTreeRequest()
        query_string_to_request(qs, req)

        error = validate_mlir_graph_request(req.chief_directory)
        if error:
            self.return_validation_error(error)
            return True

        try:

            # The navigation tree for MLIR view is provided as a protobuf object
            # Clicking on one of the leaf nodes will show the dependency graph and
            # highlight the source code in the source pane.
            dbg_rtir_path = Path(req.chief_directory) / DBG_RTIR
            nav_tree_str = nav_tree(str(dbg_rtir_path))
        except RuntimeError as e:
            self.return_runtime_error(e)
            return True

        if nav_tree_str:
            self.return_json(nav_tree_str, compress=True)
        else:
            self.return_validation_error(
                {
                    "message": "Layout configs file is outdated.",
                    "repair": True,
                }
            )

        return True
