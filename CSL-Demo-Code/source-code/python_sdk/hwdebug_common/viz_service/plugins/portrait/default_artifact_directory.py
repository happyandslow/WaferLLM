"""
api/portrait/default_artifact_directory
return the artifact directory passed in the command line
"""

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.pb.viz.default_service_pb2 import (
    GetDefaultArtifactDirectoryResponse,
)

class Plugin(PluginBase):
    ALIAS = 'portrait/default_artifact_directory'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        resp = GetDefaultArtifactDirectoryResponse()
        resp.artifact_directory = self.workdir

        self.return_msg(resp)
        return True
