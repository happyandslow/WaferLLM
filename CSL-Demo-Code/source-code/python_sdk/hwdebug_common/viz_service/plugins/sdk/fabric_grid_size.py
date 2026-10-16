"""
api/sdk/fabricgridsize
return the fabric size
"""

from google.protobuf.internal import enum_type_wrapper

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.sdk.utils import get_compile_cache
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import GetFabricGridSizeRequest
from cerebras.pb.viz.models_inline_response200_pb2 import (
    _FABRICGRIDSIZERESPONSE_ARTIFACTTYPEENUM,
    _FABRICGRIDSIZERESPONSE_FABRICTYPEENUM,
    FabricGridSizeResponse,
)

# HACK, because in-line enums don't get a wrapper. Could move definition outside message
# to fix this...
FabricTypeEnum = enum_type_wrapper.EnumTypeWrapper(
    _FABRICGRIDSIZERESPONSE_FABRICTYPEENUM
)
ArtifacttypeEnum = enum_type_wrapper.EnumTypeWrapper(
    _FABRICGRIDSIZERESPONSE_ARTIFACTTYPEENUM
)


class Plugin(PluginBase):
    ALIAS = 'sdk/fabric_grid_size'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        # convert incoming json to the request message
        req = GetFabricGridSizeRequest()
        query_string_to_request(qs, req)

        cache = get_compile_cache(mgr, req.compile_directory)

        # form the proper response
        resp = FabricGridSizeResponse()
        resp.x_size, resp.y_size = cache.getFabricSize()
        resp.fabric_type = FabricTypeEnum.FABRICSIMULATOR
        resp.artifact_type = ArtifacttypeEnum.ALL
        # TODO: traces does not exist means this directory is for pre-execution.
        
        self.return_msg(resp)
        return True
