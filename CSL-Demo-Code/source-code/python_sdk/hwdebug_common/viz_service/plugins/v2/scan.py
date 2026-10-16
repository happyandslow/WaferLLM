"""
top menu plugin,  this plugin scans the current directory and presents a list
of features based on what it finds

"""
import json
import os
from urllib.parse import quote_plus, urlencode

from cerebras.hwdebug_common.viz_service.base import EnumFiles, VizMetaBase
from cerebras.pb.viz.viz_common_pb2 import VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import VizMetaResponse


class Plugin(VizMetaBase):
    ALIAS = None

    def process_system_jobs(self, resp, baseparms):
        section = "Pytest system workdir jobs"
        found = EnumFiles("testinfo.json").search(self.workdir)
        count = 0
        for item in found:
            if 'latest/' in item:
                continue
            try:
                with open(os.path.join(self.workdir, item), "r") as f:
                    info = json.load(f)
                    if 'user_properties' in info:
                        for prop in info['user_properties']:
                            if prop[0] == 'rerun_cmd':
                                rerun_cmd = prop[1]
                                if '--system' in rerun_cmd:
                                    status = ""
                                    if "outcome" in info:
                                        if info['outcome'] == "failed":
                                            status = ' <strong style="color: red;">FAILED</strong>'
                                        else:
                                            status = ' <strong style="color: green;">PASSED</strong>'
                                    link = rerun_cmd + status
                                    parms = {'path': item}
                                    parms.update(baseparms)

                                    dots = os.path.dirname(item).split('.')
                                    date = dots[-4]
                                    date = (
                                        date[4:6]
                                        + '/'
                                        + date[6:8]
                                        + "/"
                                        + date[:4]
                                    )
                                    time = dots[-3]
                                    time = (
                                        date + " " + time[:2] + ':' + time[2:]
                                    )
                                    link = time + " " + link
                                    self.add_link(
                                        resp,
                                        section,
                                        link,
                                        f"/remote.html?{urlencode(parms)}",
                                    )
                                    count += 1
            except Exception:
                self.add_text(
                    resp,
                    section,
                    "coudld not parse {item} for singularity information TODO: use run_summary.json instead",
                )

        if count == 0:
            self.add_text(resp, section, "No remote jobs detected")

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.info("V2/scan GET: %s", url)
        self.handler = handler
        mgr = self.parent
        plug_name = 'v2.topnav'
        TopNav = mgr.get_plugin(plug_name)(self, plug_name, self.viz_link)

        stack = mgr.directory['plan_dirs']
        sdk_artifacts = mgr.directory['sdk_artifacts']

        baseparms = {}
        for x, y in qs.items():
            baseparms[x] = y[0]

        resp = VizMetaResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_META

        done = False
        TopNav.del_menu('graph')
        section = "Available cslang compiles"
        art_dir = (
            None
            if 'artifact_directory' not in qs
            else int(qs['artifact_directory'][0])
        )
        if len(sdk_artifacts) > 1 and 'artifact_directory' not in qs:
            self.add_text(
                resp,
                section,
                '<strong style="color: red;">multiple sdk artifacts found in directory,  select correct artifact</strong>',
            )
        if len(sdk_artifacts):
            TopNav.add_menu('SDK Debugger', "/sdk-gui", baseparms)

        for link in sdk_artifacts:
            params = {"artifact_directory": os.path.dirname(link)}
            params.update(baseparms)
            self.add_link(
                resp,
                section,
                'sdk artifact: ' + link,
                f"/sdk-gui?{urlencode(params)}",
            )

        section = "Available stack compiles"
        stack_index = 0 if 'stack' not in qs else int(qs['stack'][0])
        if len(stack) > 1 and 'stack' not in qs:

            self.add_text(
                resp,
                section,
                '<strong style="color: red;">multiple stack compiles found in directory,  select correct stack</strong>',
            )
            i = 0
            while len(stack) > i:
                has_viz = self.find_viz(self.workdir, os.path.dirname(stack[i]))
                if has_viz:
                    params = {"stack": i}
                    params.update(baseparms)
                    self.add_link(
                        resp, section, stack[i], f"/?{urlencode(params)}"
                    )
                else:
                    self.add_text(
                        resp, section, f"{stack[i]} does not have any viz files"
                    )
                i += 1

            self.process_system_jobs(resp, baseparms)

            done = True

        else:
            if len(stack) == 0:
                if len(sdk_artifacts) == 0:
                    self.add_text(
                        resp,
                        section,
                        f'<strong style="color: red;">ERROR: no stack directories found in directory specified</strong>',
                    )
                    parms = {'path': 'usage.md'}
                    parms.update(baseparms)
                    self.add_link(
                        resp,
                        section,
                        "usage",
                        f"markdown.html?{urlencode(parms)}",
                    )

                    self.process_system_jobs(resp, baseparms)

                done = True
            else:
                self.stack_dir = stack[stack_index]
                self.add_text(
                    resp, section, f"<b>Stack selected:{self.stack_dir}</b>"
                )
                params = {}
                params.update(baseparms)
                params.pop('stack', None)
                self.add_button(
                    resp, section, "deselect stack", f"/?{urlencode(params)}"
                )
                reports = EnumFiles("compile_report.txt").search(
                    os.path.join(self.workdir, self.stack_dir)
                )
                if len(reports):
                    for x in reports:
                        parms = {'path': os.path.join(self.stack_dir, x)}
                        parms.update(baseparms)
                        self.add_button(
                            resp, section, x, f"logfile.html?{urlencode(parms)}"
                        )

        if not done:
            section = "Select "

        if not done:
            section = "Available Core Files"

            corepath = None if 'core_path' not in qs else qs['core_path'][0]
            if corepath:
                self.add_text(
                    resp, section, f"<b>Core selected: {corepath}</b>"
                )
                self.core_file = corepath
                params = {}
                params.update(baseparms)
                params.pop('core_path', None)
                self.add_button(
                    resp, section, "deselect core", f"/?{urlencode(params)}"
                )
            else:
                self.add_text(
                    resp,
                    section,
                    "<b>Select a core file to enable debugger mode to be enabled</b>",
                )
                cores = EnumFiles("*.cs1_0").search(self.workdir)
                if len(cores) > 0:
                    for x in cores:
                        parms = {'core_path': x}
                        parms.update(baseparms)
                        self.add_link(resp, section, x, f"/?{urlencode(parms)}")
                else:
                    self.add_text(resp, section, "<b>No core files found</b>")

        if not done:

            self.add_button(
                resp, section, "file browser", f"/tree.html?{urlencode(params)}"
            )

            section = "Available annotations"
            name = self.find_viz(self.workdir, self.stack_dir)
            parms = {}
            if name:
                stall = EnumFiles("wf_annotation.json").search(self.workdir)
                if len(stall) == 1:
                    parms['annotationsPath'] = stall[0]
            else:
                name = self.stack_dir
                self.add_text(
                    resp,
                    section,
                    f"<b>NOTE: select a stack before you can see the graph.  No viz files in <li>{self.workdir}</li><li>{self.stack_dir}</li></b>",
                )

            parms.update(
                {
                    'graphPath': name,
                    'menuApi': quote_plus(
                        f"/api/v2/menu?{urlencode(baseparms)}"
                    ),
                }
            )
            stall = EnumFiles("wf_annotation.json").search(self.workdir)
            if len(stall) == 1:
                parms['annotationsPath'] = stall[0]
            parms.update(baseparms)
            self.add_link(
                resp, section, f"Graph: {name}", f"/graph?{urlencode(parms)}"
            )
            TopNav.add_menu('Graph', "/graph", parms)

            parms = {}
            parms.update(baseparms)
            section = "CS1/Simulation Setup"
            self.add_link(
                resp,
                section,
                f"Select CS1/Simulator",
                f"/selectcs1.html?{urlencode(parms)}",
            )

        self.add_button(
            resp, "Tools", "xterm", f"/term.html?{urlencode(baseparms)}"
        )
        self.add_button(
            resp, "Tools", "chart", f"/chart.html?{urlencode(baseparms)}"
        )

        self.return_msg(resp)
        return True
