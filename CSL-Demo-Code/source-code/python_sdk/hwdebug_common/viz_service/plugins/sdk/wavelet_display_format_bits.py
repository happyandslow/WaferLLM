from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.pb.viz.wavelet_display_format_bits_response_pb2 import (
    WaveletDisplayFormatBitsResponse,
)


class Plugin(PluginBase):
    ALIAS = 'sdk/wavelet_display_format_bits'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        # form the proper response
        resp = WaveletDisplayFormatBitsResponse()

        # available formats are hardcoded for now
        # TODO: SW-52813 only return formats relevant to the artifact directory
        formats = ['i16', 'f16', 'f32', 'i16,f16', 'hex']

        resp.wavelet_format.extend(formats)

        self.return_msg(resp)
        return True
