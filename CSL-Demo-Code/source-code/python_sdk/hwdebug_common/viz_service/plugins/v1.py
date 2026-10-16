"""
v1 viz api

this plugin implements the v1 api /api/v1/*
"""
import glob
import json
import os
from http import HTTPStatus

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.v2.menu import PluginBase as Menu


class Plugin(PluginBase):
    """ plugin to support v1 interface to provide support files for viz """

    ALIAS = 'v1'

    def find_by_extension(self, urlpath, qs):
        """ service to find files based on an extension """
        path = qs['path'][0]
        spath = path.split('/')

        # dont allow .. in paths for security
        if '..' in spath:
            self.handler.send_response(HTTPStatus.NOT_FOUND)
            return

        # identify cigar files.  it will be helpful later if we could switch the
        # meta type in PB to LairMeta so we could search for meta.type ===
        # 'ST_CIGAR
        try:
            glb = os.path.join(path, '*.' + qs['ext'][0])
            self.logger.debug("Scanning %s", glb)
            fileData = glob.glob(glb)
            fileData = {
                "data": fileData,
                "meta": {"type": 5, "source_file": path},
            }

        finally:
            body = json.dumps(fileData).encode('utf-8')
            self.handler.send_response(HTTPStatus.OK)
            self.handler.send_header("Content-type", "application/json")
            self.handler.send_header("Content-Length", len(body))
            self.handler.end_headers()
            self.handler.wfile.write(body)

    def heatmap(self, urlpath, qs):
        """ convert heatmap from text to json format """
        path = qs['path'][0]
        spath = path.split('/')

        # dont allow .. in paths for security
        if '..' in spath:
            self.handler.send_response(HTTPStatus.NOT_FOUND)
            return

        data = []
        m_min = None
        m_max = None

        top = []
        path = os.path.join(self.workdir, path)
        with open(path, "r") as f:
            for line in f.readlines():
                vals = line.rstrip().split(' ')
                vals = [int(numeric_string) for numeric_string in vals]
                data.append(vals)
                for i in vals:
                    if m_min is None:
                        m_min = i
                    if m_max is None:
                        m_max = i
                    m_min = min(m_min, i)
                    m_max = max(m_max, i)

        height = len(data)
        width = len(data[0])
        ix = 0
        for x in data:
            iy = 0
            for y in x:
                if y == m_max and len(top) < 10:
                    top.append([{'x': iy, 'y': ix}, m_max])
                iy += 1
            ix += 1

        heatmap = {
            "data": {
                "data": data,
                "details": {
                    'max': m_max,
                    'min': m_min,
                    'height': height,
                    'width': width,
                    'topValues': top,
                },
            },
            "meta": {"source_file": path, "type": 4},
        }

        body = json.dumps(heatmap).encode('utf-8')
        self.handler.send_response(HTTPStatus.OK)
        self.handler.send_header("Content-type", "application/json")
        self.handler.send_header("Content-Length", len(body))
        #        self.handler.send_header("Last-Modified",
        #        self.handler.date_time_string(fs.st_mtime))
        self.handler.end_headers()
        self.handler.wfile.write(body)

    def convert_placement(self, fd):
        """ convert cigar file to old viz format """
        res = {}
        res['meta'] = {'_original': fd['meta'], 'source_type': 4}
        arcs = {}
        for arc in fd['cg']['arcs']:
            if 'arc' not in arc:
                continue

            onode = arc['arc']['onode']
            inode = arc['arc']['inode']
            iport = arc['arc']['iport']
            oport = arc['arc']['oport']
            id_ = f'{onode}:{oport}-{inode}:{iport}'
            arcs[id_] = {
                'details': {
                    '_original': arc,
                    'from_port': oport,
                    'prop': arc['prop'],
                    'to_port': iport,
                },
                'from': onode,
                'id': id_,
                'to': inode,
            }

        res['data'] = {'arcs': arcs, 'buses': fd['cg']['buses']}
        del arcs

        ## details
        fabric_height = fd['attributes']['fabric_height']
        fabric_width = fd['attributes']['fabric_width']
        details = {
            '_original': fd['attributes'],
            'attributes': fd['attributes'],
            'fabricSize': {'width': fabric_width, 'height': fabric_height},
            # TODO: scan for placement
            # const hasPlacement =
            #   body?.cg?.nodes &&
            #     Object.values(body.cg.nodes).some((node) => {
            #     return (
            #     Number.isFinite(node?.placement?.x) ||
            #      Number.isFinite(node?.placement?.y)
            'hasPlacement': True,
            'perf': fd['cg']['perf'],
        }

        nodes = {}
        for _, node in fd['cg']['nodes'].items():
            nodes[node['name']] = {
                'details': {
                    '_original': node,
                    'location': node['placement'],
                    'perf': {
                        'compute_dt': node['compute_dt'],
                        "delta_t": node['delta_t'],
                        "ops": node['ops'],
                        "transmission_dt": node['transmission_dt'],
                        "useful_ops": node['useful_ops'],
                    },
                },
                'id': node['name'],
                'location': node['placement'],
                'type': node['flavor'] if 'flavor' in node else "",
            }

        res['data']['details'] = details
        del details
        res['data']['nodes'] = nodes
        del nodes

        ## tree
        tree = []
        for x in fd['graph']['nodes']:
            tree.append({'name': x['name'], 'branch': []})

        res['data']['tree'] = {'branch': tree, 'name': '_TOP'}
        del tree
        return res

    def graph(self, urlpath, qs):
        """ handle cigar graph file conversion """
        path = qs['path'][0]
        spath = path.split('/')

        # dont allow .. in paths for security
        if '..' in spath:
            self.handler.send_response(HTTPStatus.NOT_FOUND)
            return

        path = os.path.join(self.workdir, path)
        with open(path, "r") as f:
            fileData = json.load(f)

        # identify cigar files.  it will be helpful later if we could switch the
        # meta type in PB to LairMeta so we could search for meta.type ===
        # 'ST_CIGAR
        try:
            if fileData["meta"]["type"] == "placement":
                fileData = self.convert_placement(fileData)

        finally:
            body = json.dumps(fileData).encode('utf-8')
            self.handler.send_response(HTTPStatus.OK)
            self.handler.send_header("Content-type", "application/json")
            self.handler.send_header("Content-Length", len(body))
            self.handler.end_headers()
            self.handler.wfile.write(body)

    def relevant_files(self, urlpath, qs):
        path = qs['path'][0]
        spath = path.split('/')

        # dont allow .. in paths for security
        if '..' in spath:
            self.handler.send_response(HTTPStatus.NOT_FOUND)
            return

        dir_ = os.path.dirname(path)
        if dir_.endswith('/viz'):
            path = dir_[:-4]
        else:
            path = dir_

        viz = self.enum_files(
            os.path.join(path, "viz"), "*_viz.json", last_file=False
        )
        annotations = self.enum_files(
            os.path.join(path, "annotations"), "*.json", last_file=False
        )
        heatmaps = self.enum_files(
            os.path.join(path, "heatmaps"), "*.txt", last_file=False
        )

        resp = {
            'data': {
                'viz': sorted(viz),
                'annotations': annotations,
                'heatmaps': sorted(heatmaps),
            },
            'meta': {
                'type': "VIZ_RELEVANT_FILES",
                'stackDirectory': self.stack_dir,
            },
        }
        self.return_json(resp)

    def annotate(self, urlpath, qs):
        """ process annotation file.  wrap in data/meta if needed """
        path = qs['path'][0]
        spath = path.split('/')

        # dont allow .. in paths for security
        if '..' in spath:
            self.handler.send_response(HTTPStatus.NOT_FOUND)
            return

        if not path.startswith("/"):
            path = os.path.join(self.workdir, path)
        with open(path, "r") as f:
            fileData = json.load(f)

        # identify cigar files.  it will be helpful later if we could switch the
        # meta type in PB to LairMeta so we could search for meta.type ===
        # 'ST_CIGAR
        try:
            if not 'meta' in fileData:
                data = {}
                for name, val in fileData.items():
                    data[name] = {'details': val}
                fileData = {
                    "data": data,
                    "meta": {'source_file': path, 'type': 3},
                }

        finally:
            body = json.dumps(fileData).encode('utf-8')
            self.handler.send_response(HTTPStatus.OK)
            self.handler.send_header("Content-type", "application/json")
            self.handler.send_header("Content-Length", len(body))
            self.handler.end_headers()
            self.handler.wfile.write(body)

    def menu(self, qs):
        return Menu.handle_get()

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("V1 API GET: %s", url)
        self.handler = handler
        path = req.path

        if path.startswith("/api/v1/heatmap"):
            self.heatmap(path, qs)
        elif path.startswith("/api/v1/annotations"):
            self.annotate(path, qs)
        elif path.startswith("/api/v1/file/relevant_files"):
            self.relevant_files(path, qs)
        elif path.startswith("/api/v1/file/find_by_extension"):
            self.find_by_extension(path, qs)
        elif (path == "/api/v1/graph/file") and qs['path']:
            self.graph(path, qs)
        elif path == "/api/v1/menu":
            self.parent.load_plugin('v2.menu')
            self.parent.plugins['v2.menu'].handle_get(handler, url, req, qs)
        else:
            return False

        return True
