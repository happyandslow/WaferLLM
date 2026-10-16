"""
top menu plugin,  this plugin scans the current directory and presents a list
of features based on what it finds

"""
from urllib.parse import urlencode

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

        section = baseparms['section']
        files = EnumFiles(baseparms['name']).search(self.workdir)
        if len(files):
            for x in files:
                parms = {'path': x}
                parms.update(baseparms)
                self.add_link(
                    resp, section, x, f"logfile.html?{urlencode(parms)}"
                )
        else:
            self.add_text(resp, section, f"No {qs['name']} files found")

        self.return_msg(resp)
        return True
