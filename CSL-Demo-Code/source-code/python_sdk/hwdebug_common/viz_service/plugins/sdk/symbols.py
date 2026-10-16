"""
api/sdk/symbols
return a list of symbols read from elf
"""
import glob
import os

import cerebras.sdk.sdk_debug_pe_symbol_dump as sdk_pe_symbol_dump
from cerebras.elf.csl import has_color_info_in_file
from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.hwdebug_common.viz_service.plugins.sdk.utils import (
    check_params,
)
from cerebras.hwdebug_common.viz_service.plugins.common.utils import (
    query_string_to_request,
)
from cerebras.pb.viz.default_service_pb2 import GetSymbolsRequest


class Plugin(PluginBase):
    ALIAS = 'sdk/symbols'

    def handle_get(self, handler, url, req, qs):
        """ parse the v1 request and dispatch as appropriate """
        self.logger.debug("%s GET: %s", Plugin.ALIAS, url)
        self.handler = handler
        mgr = self.parent

        # convert incoming json to the request message
        req = GetSymbolsRequest()
        query_string_to_request(qs, req)

        check_params(mgr, req.compile_directory, req.grid_x, req.grid_y)

        # TODO: form the proper response using protobuf
        # resp = InlineResponse2008()
        resp = {}
        # Load the correct ELF file to get the PESymbolDump object
        symbol_dump = self._get_symbol_dump(req.compile_directory, req.grid_x, req.grid_y)

        # Return an empty string for error handling
        if symbol_dump is None:
            self.return_json({})
            return True

        # Populate the response with a list of symbols dictionary
        resp['symbols'] = list(symbol_dump.iter_symbols())

        self.return_msg(resp)
        return True

    # This section of the code is mostly copied over from:
    # sdk_debug_pe_symbol_dump.py
    def _get_symbol_dump(self, compile_dir, x, y):
        """
        @return sdk_pe_symbol_dump.PESymbolDump when found. Else return None
        """
        # Elf file naming convention
        elf_file_pat = '*.elf'
        elffiles = glob.glob(os.path.join(compile_dir, elf_file_pat))
        # Found no elf files
        if len(elffiles) == 0:
            return None

        # Parse each ELF file to determine to get the ELF file which match the coordiante user request.
        for elf in elffiles:
            symbol_dump = sdk_pe_symbol_dump.PESymbolDump(elf)
            ex, ey = symbol_dump.get_coordinates()
            # Found the right coordinate
            if not sdk_pe_symbol_dump.coordinate_matches(x, y, ex, ey):
                continue

            # Ignore any elf without .pe_debug_info
            if not has_color_info_in_file(elf):
                continue

            return symbol_dump

        # None of the ELF files match the coordinate
        return None
