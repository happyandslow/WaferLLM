"""
v1 viz api

this plugin implements the v1 api /api/v1/*
"""
from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.pb.viz.viz_explorer_pb2 import VizExplorerResponse


class BaseMeta(PluginBase):
    ALIAS = None

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("V2/kernel GET: %s", url)
        self.handler = handler
        path = req.path
        ctx = self.get_das()

        meta_type, type_, body = self.get_body(self)

        resp = VizExplorerResponse()
        resp.meta.type = type_
        resp.data.details.response_type = type_
        resp = self.msg_to_dict(resp)
        resp['data']['data'] = body
        self.return_json(resp)
        return True
