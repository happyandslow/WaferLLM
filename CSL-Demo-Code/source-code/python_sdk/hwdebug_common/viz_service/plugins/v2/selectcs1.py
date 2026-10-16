"""
select CS1 to operate on (debugging)
"""


from cerebras.hwdebug_common.viz_service.base import VizMetaBase
from cerebras.pb.viz.viz_common_pb2 import VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import VizMetaResponse


class Plugin(VizMetaBase):
    ALIAS = None

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("V2/selectcs1 GET: %s", url)
        self.handler = handler
        mgr = self.parent
        reqpath = req.path
        session_id = qs['session_id'][0]
        raise ValueError(
            'TBD - This would allow user to select a CS1 system or simulator to attach to for interactive debugging'
        )

        resp = VizMetaResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_META

        section = "Select CS1 to connect to"

        self.return_msg(resp)
        return True
