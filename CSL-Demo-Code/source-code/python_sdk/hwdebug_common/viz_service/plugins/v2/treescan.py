"""
top menu plugin,  this plugin scans the current directory and presents a list
of features based on what it finds

"""
from urllib.parse import quote_plus, urlencode

from cerebras.hwdebug_common.viz_service.base import EnumFiles, VizMetaBase
from cerebras.pb.viz.viz_common_pb2 import VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import VizMetaResponse


class Plugin(VizMetaBase):
    ALIAS = None

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.info("V2/scan GET: %s", url)
        self.handler = handler

        baseparms = {}
        for x, y in qs.items():
            baseparms[x] = y[0]

        resp = VizMetaResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_META

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

        section = "Available NUMPY Files"
        numpy = EnumFiles("*.np[yz]").search(self.workdir)
        if len(numpy):
            for x in numpy:
                parms = {'path': x}
                parms.update(baseparms)
                self.add_link(
                    resp, section, x, f"numpy.html?{urlencode(parms)}"
                )
        else:
            self.add_text(resp, section, "No NUMPY files found")

        section = "Available annotations"
        name = self.find_viz(self.workdir, self.stack_dir)
        if name:
            parms = {
                'graphPath': name,
                'menuApi': quote_plus(f"/api/v2/menu?{urlencode(baseparms)}"),
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
