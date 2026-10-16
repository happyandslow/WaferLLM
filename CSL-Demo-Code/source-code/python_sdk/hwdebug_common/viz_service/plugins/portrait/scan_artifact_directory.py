"""
api/portrait/scan_artifact_directory
scan the artifact directory
"""

from pathlib import Path

from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
    SIMFAB_TRACES
)
from cerebras.pb.viz.default_service_pb2 import (
    ScanArtifactDirectoryRequest,
    ScanArtifactDirectoryResponse,
)


class Plugin(PluginBase):
    ALIAS = 'portrait/scan_artifact_directory'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler

        # convert incoming json to the request message
        req = ScanArtifactDirectoryRequest()
        query_string_to_request(qs, req)

        artifact_directory = Path(req.artifact_directory)
        if not artifact_directory.is_dir():
            error = (
                f'Artifact directory {str(artifact_directory)} does not exist'
            )
            self.logger.error(error)
            raise ValueError(error)

        resp = ScanArtifactDirectoryResponse()

        # compile directory is where viz directory is located
        # users may pass directory ending with /viz or directory containing viz
        if artifact_directory.name == 'viz':
            compile_directory = artifact_directory.parent
        elif (artifact_directory / 'viz').is_dir():
            compile_directory = artifact_directory
        else:
            error = f'Compile directory for {str(artifact_directory)} does not exist'
            self.logger.error(error)
            raise ValueError(error)
        resp.compile_directory = str(compile_directory)

        # run directory is where simfab_traces exists
        # At this point, we do not support multiple run directories
        # If none exists, return empty. (for eg, compile only artifacts)
        # run directory can be embedded deep as in sysemu runs
        run_directories = list(compile_directory.rglob(SIMFAB_TRACES))
        if len(run_directories) > 0:
            resp.run_directory = str(run_directories[0].parent)

        # graph annotation path can be empty
        graph_path = compile_directory / "stall_viz" / "viz_graph_anno.json"
        if graph_path.exists():
            resp.graph_annotations_path = str(graph_path)

        # heatmap annotation path can be empty
        heatmap_path = compile_directory / "stall_viz" / "heatmap" / "pe_anno.json"
        if heatmap_path.exists():
            resp.heatmap_annotations_path = str(heatmap_path)

        # for sysemu tests, there is no landing.json
        # detect if viz / landing.json exists
        landing_json = compile_directory / "viz" / "landing.json"
        resp.has_landing_json = landing_json.exists()

        self.return_msg(resp)
        return True
