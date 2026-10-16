"""
handles access to lab system access
"""
import json
import os
import re
import subprocess

from devinfra.monolith.envapi_bridge import get_envapi

from cerebras.hwdebug_common.viz_service.base import VizMetaBase
from cerebras.pb.viz.viz_common_pb2 import VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import VizMetaResponse


class Plugin(VizMetaBase):
    ALIAS = None

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("V2/remote GET: %s", url)
        self.handler = handler
        mgr = self.parent
        reqpath = req.path
        session_id = qs['session_id'][0]
        path = qs['path'][0]

        baseparms = {}
        for x, y in qs.items():
            baseparms[x] = y[0]

        resp = VizMetaResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_META

        section = "Remote system information/Access"

        self.add_text(resp, section, "TBD - NEEDS WORK")

        rerun_cmd = None
        self.add_text(resp, section, f"host working dir: {path}")
        with open(os.path.join(self.workdir, path), "r") as f:
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
                            parms = {'path': path}
                            parms.update(baseparms)

                            dots = path.split('.')
                            date = dots[-5]
                            date = date[4:6] + '/' + date[6:8] + "/" + date[:4]
                            time = dots[-4]
                            time = date + " " + time[:2] + ':' + time[2:]
                            link = time + " " + link
                            self.add_text(resp, section, link)

        if rerun_cmd:
            reduce = re.sub('[ \t]+', ' ', rerun_cmd)
            system_name = re.search(r'--system ([^ ]+)', reduce)[1]
            envapi = get_envapi()
            if not envapi.system_name_is_valid(system_name):
                self.add_text(
                    resp,
                    section,
                    "Unknown system or system not in db, hostname: "
                    + str(system_name),
                )
            else:
                compile_hosts = envapi.get_compile_pod_hosts(system_name)
                for host in compile_hosts:
                    self.add_text(resp, section, "compile host: " + host)

                log_messages = os.path.join(
                    self.workdir, os.path.dirname(path), "logmessages.txt"
                )
                baseline = None
                try:
                    baseline = subprocess.check_output(
                        ['grep', 'singularity exec.*mkdir', log_messages]
                    )
                except Exception as e:
                    self.add_text(resp, section, str(e))
                if baseline:
                    section = "Commands to start csviz remotely"
                    baseline = baseline.decode('utf-8')
                    remotedir = re.search(r'-B ([^ ,]+):', baseline)[1]

                    command = re.sub(remotedir, '${WORKDIR}', baseline)
                    sif = re.search(r' ([^ ]+\.sif) ', command)[1]
                    command = re.sub(sif, "${SIFIMAGE}", command).split(' ')

                    command = command[: command.index('--')]
                    command = (
                        f'ssh lab@{compile_hosts[0]}'
                        + ' '.join(command)
                        + " -- csviz ${WORKDIR}"
                    )

                    self.add_text(resp, section, "export WORKDIR=" + remotedir)
                    self.add_text(resp, section, "export SIFIMAGE=" + sif)
                    self.add_text(resp, section, command)

        self.return_msg(resp)
        return True
