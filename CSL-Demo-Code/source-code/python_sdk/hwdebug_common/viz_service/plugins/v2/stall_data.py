"""
v1 viz api

this plugin implements the v1 api /api/v1/*
"""
from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.pb.viz.viz_common_pb2 import VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import (
    VizExplorerResponse,
    VizExplorerResponseType,
)


class Plugin(PluginBase):
    ALIAS = None

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("V2/kernel GET: %s", url)
        self.handler = handler
        path = req.path
        ctx = self.get_das()

        resp = VizAnnotationData()
        resp.viz_links[0] = VisLink()
        # resp.viz_links[0].

        body = {
            "pe": [qs['x'][0], qs['y'][0]],
            "session_id": qs['session_id'][0],
        }
        body = {
            "viz_links": [{"VizAppType": "/log_file",}],
            "custom_boxes": [
                {
                    "title": "Box 1",
                    "description": "The box1 description",
                    "details": {"json": "rendered", "other": "text"},
                },
                {
                    "title": "Box 2",
                    "description": "The box2 description",
                    "details": {"json": "rendered box 2"},
                },
            ],
        }
        resp = VizExplorerResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_RESPONSE
        resp.data.details.response_type = VizExplorerResponseType.ANNOTATIONS
        resp = self.msg_to_dict(resp)
        resp['data']['data'] = body
        self.return_json(resp)
        return True
