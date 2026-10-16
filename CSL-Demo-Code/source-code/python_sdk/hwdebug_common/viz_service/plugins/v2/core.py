"""
top menu plugin
"""
import os
from urllib.parse import quote_plus, urlencode

from cerebras.hwdebug_common.viz_service.base import EnumFiles, PluginBase
from cerebras.pb.viz.viz_common_pb2 import VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import (
    VizMetaItem,
    VizMetaResponse,
    VizMetaType,
)


class Plugin(PluginBase):
    ALIAS = None

    def add_link(self, resp, cat, text, href):
        m1 = VizMetaItem()
        m1.category = cat
        m1.label = text
        m1.endpoint = href
        m1.meta_type = VizMetaType.LINK
        resp.data.append(m1)

    def add_text(self, resp, cat, text):
        m1 = VizMetaItem()
        m1.category = cat
        m1.label = text
        m1.meta_type = VizMetaType.TEXT
        resp.data.append(m1)

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.info("V2/core GET: %s", url)
        self.handler = handler
        mgr = self.parent
        path = req.path
        session_id = qs['session_id'][0]

        stack = mgr.directory['plan_dirs']

        baseparms = {'session_id': session_id}

        resp = VizMetaResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_META

        section = "Available stack compiles"
        stack_index = 0 if 'stack' not in qs else int(qs['stack'][0])
        if len(stack) > 1 and 'stack' not in qs:

            self.add_text(
                resp,
                section,
                "multiple stack compiles found in directory,  select correct stack",
            )
            i = 0
            while len(stack) > i:
                params = {"stack": i}
                params.update(baseparms)
                self.add_link(resp, section, stack[i], f"/?{urlencode(params)}")
                i += 1
        else:
            stack_dir = stack[stack_index]
            self.stack_dir = os.path.dirname(stack_dir)
            self.add_text(resp, section, f"Stack selected:{self.stack_dir}")

        section = "Compile Report"
        reports = EnumFiles("compile_report.txt").search(self.workdir)
        if len(reports):
            for x in reports:
                parms = {'path': x}
                parms.update(baseparms)
                self.add_link(
                    resp, section, x, f"logfile.html?{urlencode(parms)}"
                )

        section = "Available LOG Files"
        numpy = EnumFiles("*.log").search(self.workdir)
        if len(numpy):
            for x in numpy:
                parms = {'path': x}
                parms.update(baseparms)
                self.add_link(
                    resp, section, x, f"logfile.html?{urlencode(parms)}"
                )
        else:
            self.add_text(resp, section, "No log files found")

        section = "Available annotations"
        name = self.find_viz(self.workdir, self.stack_dir)
        if name:
            parms = {
                'graphPath': name,
                'menu_api': quote_plus(f"/api/v2/menu?{urlencode(baseparms)}"),
            }
            stall = EnumFiles("wf_annotation.json").search(self.workdir)
            if len(stall) == 1:
                parms['annotationsPath'] = stall[0]
            parms.update(baseparms)
            self.add_link(
                resp, section, f"Graph: {name}", f"/graph?{urlencode(parms)}"
            )
        else:
            self.add_text(
                resp,
                section,
                "NOTE: select a stack before you can see the graph",
            )

        self.return_msg(resp)
        return True
