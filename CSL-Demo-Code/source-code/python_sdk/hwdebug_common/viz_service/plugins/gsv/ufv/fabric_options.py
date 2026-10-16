"""
api/gsv/ufv/fabric_options
Gets debug dump directories, if they exist.
Only called when user specifies only a chief directory while debug
"""
from pathlib import Path

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.hwdebug_common.viz_service.plugins.gsv.ufv.ufv_common import (
    validate_chief_dir,
)
from cerebras.pb.viz.unified_fabric_viewer_pb2 import (
    FabricOptionsRequest
)
from cerebras.ws.hwdebug.lib.unifiedfabricvizpybind import UnifiedFabricViz


class Plugin(PluginBase):
    ALIAS = 'gsv/ufv/fabric_options'

    def handle_get(self, handler, url, req, qs):
        """parse the v1 request and dispatch as appropriate"""
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = FabricOptionsRequest()
        query_string_to_request(qs, req)
        error = validate_chief_dir(req.chief_directory)
        if error:
            self.return_validation_error({"message": error, "repair": False})
            return True

        ufv = UnifiedFabricViz(req.chief_directory)
        resp = ufv.fabric_options()
        
        self.return_msg(resp)
        return True
