"""
v1 viz api

this plugin implements the v1 api /api/v1/*
"""

import os
from urllib.parse import urlencode

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.pb.viz.viz_common_pb2 import VizInputType, VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import VizMenuItem, VizMenuResponse

debug_port_api = [
    "kernels",
    "is_hw_tile",
    "src_files",
    "params",
    "global_color_assignments",
    "offramp_entrypoint_assignments",
    "onramp_entrypoint_assignments",
    "local_index_assignments",
    "onramp_local_color_assignment",
    "offramp_local_color_assignment",
    "onramp_local_to_global_map",
    "offramp_local_to_global_map",
    "onramp_sentinel_to_entrypoint_map",
    "offramp_sentinel_to_entrypoint_map",
    "switches",
    #            "input_queues",
    "output_queues",
    "routes",
    #            "traffic",
    "color_swaps",
]


class Plugin(PluginBase):
    ALIAS = 'menu'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("V2/menu GET: %s", url)
        self.handler = handler

        # convert params from multi to single form
        baseparms = {}
        for x, y in qs.items():
            baseparms[x] = y[0]

        corepath = (
            None if 'core_path' not in baseparms else baseparms['core_path']
        )
        if corepath:
            self.core_file = corepath

        stack_index = None if 'stack' not in qs else int(qs['stack'][0])
        if stack_index is not None:
            mgr = self.parent
            if 'plan_dirs' in mgr.directory:
                stack = mgr.directory['plan_dirs']
                stack_dir = stack[stack_index]
                self.stack_dir = os.path.dirname(stack_dir)

        resp = VizMenuResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_MENU

        id = 0

        if self.core_file:
            m1 = VizMenuItem()
            m1.id = str(id)
            id += 1
            m1.category = "Debugger Tools"
            m1.label = "Address Map"
            params = baseparms
            m1.endpoint = f"/api/v2/vizcode?{urlencode(params)}"
            m1.input_type = VizInputType.PE
            resp.data.append(m1)

            m1 = VizMenuItem()
            m1.id = str(id)
            id += 1
            m1.category = "Debugger tools"
            m1.label = "PE Registers"
            m1.endpoint = f"/api/v2/vizcore?{urlencode(params)}"
            m1.input_type = VizInputType.PE
            resp.data.append(m1)

            m1 = VizMenuItem()
            m1.id = str(id)
            id += 1
            m1.category = "Debugger tools"
            m1.label = "PE Data"
            m1.endpoint = f"/api/v2/vizdata?{urlencode(params)}"
            m1.input_type = VizInputType.PE
            resp.data.append(m1)

        for x in debug_port_api:
            m1 = VizMenuItem()
            m1.id = str(id)
            id += 1
            m1.category = "Plan Debug"
            m1.label = x
            params = baseparms
            m1.endpoint = f"/api/v2/plandebug/{x}?{urlencode(params)}"
            m1.input_type = VizInputType.PE
            resp.data.append(m1)

        params = baseparms

        m1 = VizMenuItem()
        m1.id = str(id)
        id += 1
        m1.category = "Stall Analysis"
        m1.label = "Kernel Ports"
        m1.endpoint = f"/api/v2/kernel?{urlencode(params)}"
        m1.input_type = VizInputType.PE
        resp.data.append(m1)

        m1 = VizMenuItem()
        m1.id = str(id)
        id += 1
        m1.category = "Source Code"
        m1.label = "PE Source Code"
        params = {'type': 'PE'}
        params.update(baseparms)
        m1.endpoint = f"/api/v2/src?{urlencode(params)}"
        m1.input_type = VizInputType.PE
        resp.data.append(m1)

        self.return_msg(resp)
        return True
