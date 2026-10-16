"""
top menu plugin
"""
import os

import numpy as np
from google.protobuf.json_format import MessageToDict

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.pb.viz.viz_common_pb2 import VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import VizMetaResponse


class Plugin(PluginBase):
    ALIAS = 'v2.npz'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("V2/numpy GET: %s", url)
        self.handler = handler
        path = qs['path'][0]

        resp = VizMetaResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_META

        body = MessageToDict(resp,)

        fullname = os.path.join(self.workdir, path)
        x = np.load(fullname)

        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)
        np.set_printoptions(
            edgeitems=30,
            linewidth=100000,
            formatter=dict(float=lambda x: "%8.3g" % x),
        )
        if path.endswith('npz'):
            for v in x.files:
                d = x[v]

                print(v, d.shape)
                print(d)

        if 'array' in qs:
            data = {}
            if path.endswith('npz'):
                for v in x.files:
                    data[v] = x[v].tolist()
            else:
                data = x.tolist()
        else:
            data = np.array_str(x)
        body['data'] = {'data': data}
        self.return_json(body)
        return True
