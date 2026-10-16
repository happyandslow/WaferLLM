from google.protobuf.json_format import MessageToJson, ParseDict

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.sdk.utils import get_compile_cache
from cerebras.pb.viz.default_service_pb2 import (
    ColorEntry,
    GetColorsRequest,
    GetColorsResponse,
)


# This plugin is deprecated
class Plugin(PluginBase):
    ALIAS = 'sdk/get_colors'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        req = GetColorsRequest()
        qsdict = {}
        for x, y in qs.items():
            qsdict[x] = y[0]
        ParseDict(qsdict, req)

        cache = get_compile_cache(mgr, req.compile_directory)

        resp = GetColorsResponse()

        # If this is a post-run, assume all 24 colors can be used (intended or unintended)
        color_names = range(0, 24) if req.core_file_path else cache.get_colors()
        # Known names from compiler metadata
        known_names_dict = cache.get_color_names()
        for color in color_names:
            entry = ColorEntry()
            entry.color_id = color
            # Populate with known names if exist
            if color in known_names_dict:
                entry.color_name = ",".join(known_names_dict[color])
            else:
                entry.color_name = str(color)
            resp.colors.append(entry)

        self.return_msg(resp)
        return True
