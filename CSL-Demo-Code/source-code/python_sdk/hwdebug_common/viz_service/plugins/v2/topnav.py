"""
navagation bar plugin
"""
from urllib.parse import urlencode

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.pb.viz.viz_common_pb2 import VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import (
    VizMetaItem,
    VizMetaResponse,
    VizMetaType,
)


class MenuItem:
    def __init__(self, params, url):
        self.href = url
        self.params = params


class Plugin(PluginBase):
    ALIAS = 'v2.nav'

    def add_menu(self, name, url, params):
        """ add menu item.  Includes params dict() and url
            this item is added to the top menu.  used del_menu()
            to remove item if it is not available anymore """

        mgr = self.parent
        if 'top_nav' not in mgr.directory:
            mgr.directory['top_nav'] = {}

        mgr.directory['top_nav'][name] = MenuItem(params, url)

    def del_menu(self, name):
        mgr = self.parent
        if 'top_nav' in mgr.directory:
            if name in mgr.directory['top_nav']:
                del mgr.directory['top_nav'][name]

    def add_link(self, resp, cat, text, href):
        m1 = VizMetaItem()
        m1.category = cat
        m1.label = text
        m1.endpoint = href
        m1.meta_type = VizMetaType.LINK
        resp.data.append(m1)

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """

        self.logger.debug("V2/topnav GET: %s", url)
        self.handler = handler
        mgr = self.parent

        baseparms = {}
        for x, y in qs.items():
            baseparms[x] = y[0]

        if 'path' in baseparms:
            baseparms.pop('path', None)

        resp = VizMetaResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_META

        # add home
        self.add_link(resp, "topnav", "Home", f"/?{urlencode(baseparms)}")
        if 'top_nav' in mgr.directory:
            for name, item in mgr.directory['top_nav'].items():
                pcopy = baseparms.copy()
                if item.params:
                    pcopy.update(item.params)
                self.add_link(
                    resp, "topnav", name, f"{item.href}?{urlencode(pcopy)}"
                )

        #  Always add usage and about
        params = baseparms.copy()
        self.add_link(
            resp, "topnav", "File Browser", f"/tree.html?{urlencode(params)}"
        )
        params = baseparms.copy()
        self.add_link(
            resp, "topnav", "Xterm", f"/term.html?{urlencode(params)}"
        )
        params = baseparms.copy()
        self.add_link(
            resp, "topnav", "Numpy Chart", f"/chart.html?{urlencode(params)}"
        )
        params = baseparms.copy()
        params['path'] = 'usage.md'
        self.add_link(
            resp, "topnav", "Usage", f"/markdown.html?{urlencode(params)}"
        )
        params = baseparms.copy()
        params['path'] = 'about.md'
        self.add_link(
            resp, "topnav", "About", f"/markdown.html?{urlencode(params)}"
        )

        self.return_msg(resp)

        return True
