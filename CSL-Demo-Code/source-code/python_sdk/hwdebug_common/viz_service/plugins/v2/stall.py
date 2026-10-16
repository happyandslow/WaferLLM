"""
stall analysis data plugin
"""
import json
import os
import subprocess
from urllib.parse import urlencode

from cerebras.hwdebug_common.viz_service.base import EnumFiles, PluginBase


class Plugin(PluginBase):
    ALIAS = 'stall'

    def gen_stall(self, srv):
        files = EnumFiles("wf_annotation.json").search(self.workdir)
        if len(files) == 1:
            viz = files[0]
        else:
            viz = None
        if viz is None:
            stall_debug = os.path.join(self.workdir, 'stall_debug')
            stack_dir = self.stackdir
            try:
                cmd_list = [
                    str.format(
                        f"cs_stall_debug -i {stack_dir} -o {stall_debug} --ckpt {core_hm.corefile}"
                        f" --wf {steps} --syserr"
                    ),
                    str.format(
                        f"cs_stall_debug -i {stack_dir} -o {stall_debug} --update_viz_link_only "
                        f" --viz_graph viz/*plangen_viz.json --syserr"
                    ),
                ]

                for cmd in cmd_list:
                    self.logger.debug("run %s", cmd)
                    subprocess.check_call(cmd.split(), timeout=300)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f'{cmd} failed') from e

        files = EnumFiles("wf_annotation.json").search(self.workdir)
        if len(files) == 1:
            stall_file = files[0]

        self.logger.debug("Creating json file: %s", srv)
        params = {
            'graphPath': self.viz_link,
            'annotationsPath': stall_file,
            'mainView': "graph",
        }
        viz_url = urlencode(params)

        with open(srv, 'w') as f:
            data = [
                {"ServiceUrl": f"/?{viz_url}", "ServiceName": "Stall Analysis"}
            ]
            json.dump(data, f, indent=4)

    def gen_file(self, qs):
        srv = os.path.join(self.workdir, "stall.json")
        if not os.path.exists(srv):
            self.gen_stall(srv)

        return srv
