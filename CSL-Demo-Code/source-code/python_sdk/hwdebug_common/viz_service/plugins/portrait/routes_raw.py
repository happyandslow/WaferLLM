"""
api/portrait/routes_raw
return the switch config register raw values
"""

import json
from pathlib import Path

from google.protobuf.json_format import MessageToDict, ParseDict

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    get_core_cache,
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import (
    GetPortraitRoutesRawRequest,
)


class Plugin(PluginBase):
    ALIAS = 'portrait/routes_raw'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        # convert incoming json to the request message
        req = GetPortraitRoutesRawRequest()
        query_string_to_request(qs, req)

        viz_meta = Path(req.compile_directory) / "viz" / "meta.json"
        with open(viz_meta) as f:
            meta_json = json.load(f)
            # Each .map file should come with a .elf file, we read routes_raw
            # from the elf file
            map_files = meta_json["map_files"]
            # directories in meta.json are relative to "viz"
            map_path = (
                Path(req.compile_directory) / "viz" / map_files[req.context_id]
            )
            elf_path = map_path.with_suffix(".elf")

        
        core_cache = get_core_cache(mgr, elf_path)
        raw = core_cache.getRoutes(str(elf_path), req.rect_x, req.rect_y, req.rect_w, req.rect_h)
    
        self.return_msg(raw)
        return True
