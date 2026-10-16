"""
top menu plugin
"""
import os

import markdown

from cerebras.hwdebug_common.viz_service.base import PluginBase


class Plugin(PluginBase):
    ALIAS = 'v2.npz'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("V2/markdown GET: %s", url)
        self.handler = handler

        path = qs['path'][0]

        path = os.path.join(self.parent.web_dir, path)

        with open(path, 'r') as f:
            data = f.read()
            body = markdown.markdown(data)

        self.return_html(body.encode('utf-8'))
        return True
