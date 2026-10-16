"""
api/sdk/
return the routes
"""
import copy

from google.protobuf.json_format import ParseDict

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    get_core_cache,
    query_string_to_request,
)
from cerebras.hwdebug_common.viz_service.plugins.sdk.utils import (
    get_compile_cache,
)
from cerebras.hwdebug_common.viz_service.SimpleWebSocketServer import (
    SimpleWebSocketServer as HTTPHandler,
)
from cerebras.pb.viz.default_service_pb2 import GetRoutesRequest


class Plugin(PluginBase):
    """ API used by sdk-gui to return route information used to create
        visual diagram of routing data
    """

    ALIAS = 'sdk/routesraw'

    def handle_get(self, handler: HTTPHandler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("sdk/routes GET: %s", url)
        self.handler = handler
        mgr = self.parent

        # convert incoming json to the request message
        req = GetRoutesRequest()
        query_string_to_request(qs, req)

        # workdir == bin directory that contains the ELFs
        cache = get_compile_cache(mgr, req.compile_directory)

        dx, dy = cache.getFabricSize()

        raw = {}
        if not req.core_file_path:
            # core does not exist means simulation was not run
            # Organize it by color: value is a 2D list, higher rank is y, lower rank is x.
            # Have an empty grid, deepcopy the grid when a new color is found
            # Modify the element y, x with value.
            empty_routing_tbl = [[0] * dx for _ in range(dy)]
            for x in range(dx):
                for y in range(dy):
                    route_tmp = cache.get_pre_run_routes_raw(x, y)
                    # Add each color to raw dictionary
                    for color in route_tmp:
                        # Initialize the table for the first time
                        if color not in raw:
                            raw[color] = copy.deepcopy(empty_routing_tbl)
                        raw[color][y][x] = route_tmp[color]

            # return python dictionary
            self.return_json({ "colors": raw })
        else:
            # get core cache
            core_cache = get_core_cache(mgr, req.core_file_path)
            raw = core_cache.getRoutes(req.core_file_path, req.rect_x, req.rect_y, req.rect_w, req.rect_h)

            # return protobuf object
            self.return_msg(raw)

        return True
