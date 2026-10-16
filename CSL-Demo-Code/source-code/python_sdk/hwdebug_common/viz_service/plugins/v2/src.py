"""
v1 viz api

this plugin implements the v1 api /api/v1/*
"""
import os

from cerebras.hwdebug_common.viz_service.base import EnumFiles, PluginBase
from cerebras.pb.viz.viz_common_pb2 import VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import (
    VizExplorerResponse,
    VizExplorerResponseType,
)


class Plugin(PluginBase):
    ALIAS = None

    def gen_context(self):
        pe_src_map = {}
        prefix = len(self.stack_dir.split('/'))
        found = EnumFiles("*.lst", only_last_file=False).search(
            self.workdir, self.stack_dir
        )
        for file in found:
            pe = file.split('/')[prefix + 1]
            pes = pe.split('_')
            if len(pes) == 2:
                x = int(pes[0])
                y = int(pes[1])
                pe = str(x) + ',' + str(y)
                if pe not in pe_src_map:
                    pe_src_map[pe] = []
                pe_src_map[pe].append(file)

        self.directory['src'] = pe_src_map

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("V2/kernel GET: %s", url)
        self.handler = handler
        path = req.path
        ctx = self.get_das()

        if 'src' not in self.directory:
            self.gen_context()

        pe_src_map = self.directory['src']

        body = {"pe": [qs['x'][0], qs['y'][0]], "session_id": qs['session_id']}

        resp = VizExplorerResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_RESPONSE
        resp.data.details.response_type = VizExplorerResponseType.CODE
        resp = self.msg_to_dict(resp)
        x = int(qs['x'][0])
        y = int(qs['y'][0])
        pe = str(x) + ',' + str(y)

        if pe in pe_src_map:
            all_ = ""
            for x in pe_src_map[pe]:
                with open(os.path.join(self.workdir, x), 'r') as f:
                    fd = f.read()
                    all_ += os.path.basename(x) + '\r'
                    all_ += fd
        else:
            all_ = f"NO SOURCE FOR PE({pe})"

        resp['data']['data'] = all_
        self.return_json(resp)
        return True
