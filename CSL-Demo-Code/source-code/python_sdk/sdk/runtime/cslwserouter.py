#!/usr/bin/env python3

from typing import Dict, List

import cerebras.sdk.runtime.routepybind as CslRuntimeContext
from cerebras.das_common import Coordinate
from cerebras.sdk.runtime.cslwsenetlist import CslWseNetlist, PEPort


class CslWseRouter:
    def __init__(self, netlist: CslWseNetlist, debug_level: int = 0):
        self._netlist = netlist
        self._debug_level = debug_level

    def route(self):
        """
        Route the ingress to PE and PE to egress.
        """
        # TODO: minimize color overlap/congestions.
        self._route_portlist(
            self._netlist.pe_west_portlist, self._netlist.pe_tile_west_map
        )
        self._route_portlist(
            self._netlist.pe_east_portlist, self._netlist.pe_tile_east_map
        )

    def _route_portlist(self, pe_portlist: List, pe_tile_map: Dict):
        for pe_dir_tuple in pe_portlist:
            pe_tile = pe_dir_tuple[0]
            port_tuple = pe_dir_tuple[1]
            port_dir = port_tuple[0]
            pe_dir = port_tuple[1]
            lvds_index = pe_tile_map[pe_tile]
            if port_dir is PEPort.INPUT:
                self._route_input(lvds_index, pe_tile, pe_dir)
            else:
                self._route_output(pe_tile, lvds_index, pe_dir)

    def _route_input(self, lvds_index, receiver_pe_tile, receiver_in_dir):
        ingress_tile = self._netlist.INGRESS_TILES[lvds_index]
        ingress_pe = Coordinate(ingress_tile.x(), ingress_tile.y())

        if self._debug_level:
            print(f"Ingress tile: {ingress_pe} --> receiver {receiver_pe_tile}")

        input_route = CslRuntimeContext.CslIoRouting(
            ingress_pe,
            Coordinate(receiver_pe_tile.x(), receiver_pe_tile.y()),
            receiver_pe_tile.color(),
            self._netlist.FABRIC_DIMS,
        )
        ret = input_route.route_ingress_to_pe(receiver_in_dir, True)
        assert (
            ret == 0
        ), f"Expects route to input PE to be successful, got {ret}"

        self._netlist.routes[receiver_pe_tile] = input_route

    def _route_output(self, sender_pe_tile, lvds_index, sender_out_dir):
        egress_tile = self._netlist.EGRESS_TILES[lvds_index]
        egress_pe = Coordinate(egress_tile.x(), egress_tile.y())

        if self._debug_level:
            print(f"Sender: {sender_pe_tile} --> egress tile {egress_pe}")

        direct_route = CslRuntimeContext.CslIoRouting(
            Coordinate(sender_pe_tile.x(), sender_pe_tile.y()),
            egress_pe,
            sender_pe_tile.color(),
            self._netlist.FABRIC_DIMS,
        )
        ret = direct_route.route_pe_to_egress(sender_out_dir, True)
        assert ret == 0, f"Expects route to egress to be successful, got {ret}"

        self._netlist.routes[sender_pe_tile] = direct_route
