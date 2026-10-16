"""
api/gsv/summary/view
return sections on summary page
"""

import re

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.cbda_query import (
    query_glitches,
)
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.hwdebug_common.viz_service.plugins.gsv.validation import (
    find_gsv_dir,
    validate_gsv_workdir,
    validate_summary_view,
)
from cerebras.pb.viz.viz_global_stall_view_pb2 import VizWsGSVSummaryRequest
from cerebras.ws.hwdebug.lib.wsvizpybind import WsGlobalStallViz, WsGSVType


class Plugin(PluginBase):
    ALIAS = 'gsv/summary/view'

    def handle_get(self, handler, url, req, qs):
        """parse the v1 request and dispatch as appropriate"""
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        req = VizWsGSVSummaryRequest()
        query_string_to_request(qs, req)

        gsv_dir = find_gsv_dir(req.work_directory)
        err = validate_gsv_workdir(gsv_dir) or validate_summary_view(gsv_dir)
        if err:
            self.return_validation_error(err)
            return True

        helper = WsGlobalStallViz(gsv_dir, WsGSVType.SUMMARY_VIEW, None)
        resp = helper.get_summary()

        # Patch in queried glitch data
        # Use the glitch monitor link for each system to get the timestamps to query
        # the database directly.
        # Summarize the results in the Overall section.
        overall_section = None
        glitch_str = "None"
        try:
            systems_to_query = []
            start_time = 0
            end_time = 0
            for section in resp.sections:
                if section.name == "Overall":
                    overall_section = section
                mapping = section.dictionary
                for key in mapping:
                    if "system glitches" in key:
                        systems_to_query.append(section.name)
                        if not start_time:
                            link = mapping[key].link
                            # Parse timestamps from link and convert ms -> s
                            m = re.search(
                                r"from=(?P<from>\d+)&to=(?P<to>\d+)", link
                            )
                            if m:
                                start_time = int(m.group('from')) // 1000
                                end_time = int(m.group('to')) // 1000
                            else:
                                raise ValueError(
                                    "Timestamp could not be parsed from glitch link"
                                )
                        break
            systems_with_glitches = query_glitches(
                systems_to_query, start_time, end_time
            ).keys()
            if systems_with_glitches:
                glitch_str = ", ".join(systems_with_glitches)
        except Exception as e:
            glitch_str = f"Glitch data not available: {e}"
        if overall_section:
            overall_section.dictionary["Systems with glitches"].value = (
                glitch_str
            )

        self.return_msg(resp)
        return True
