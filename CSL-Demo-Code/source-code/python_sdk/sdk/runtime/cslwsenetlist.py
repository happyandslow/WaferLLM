#!/usr/bin/env python3

# CSL Wafer Scale Engine Netlist.
# The netlist contains 3 static members, size of the fabric and list of inputs and
# outputs.
# After pin assignment, port list will be populated with user input and output PE
# tile coordinate, color, port direction and cardinal direction. The tile map will be
# popluated with PE coordinate, color as key, value is input or output index.
# After the router has ran, the routes map will be populated.

from enum import Enum
from typing import List

from cerebras.das_common import Dimensions
from cerebras.sdk.streamer.lib.streamerpybind import Tile


class PEPort(Enum):
    OUTPUT = 0
    INPUT = 1


class CslWseNetlist:
    """
    CslWseNetlist is library that stores LVDS tiles, user PE input and output PE tiles,
    and routes. This library is an prototype library for memcpy.

    Args:
        width is the fabric width.
        height is the fabric height.
        ingress is the ingress connection list. The element can be Coordinate or Tile.
        egress is the egress connection list. The element can be Coordinate or Tile.
        debug_level enables printing the info during initialization.
    """

    def __init__(
        self,
        width: int,
        height: int,
        ingress: List[Tile],
        egress: List[Tile],
        debug_level: int = 0,
    ):
        # The library makes the follow assumptions:
        # 1. len(ingress_connections == len(egress_connections)
        #    Since LVDS link are bidirectional, an ingress tile must pair up with egress tile.
        # 2. Both ingress and egress lists are first sorted by x then sorted by y.
        # 3. The same index in ingress_connection and egress_connection comes from the same LVDS link.
        #    We use the same index to identify bidirectional I/O.
        # Longer term: this library will be rewritten in C++ and interface with fabric handler
        # and pin assignment and router.

        # The following members should be treated as constants
        self.FABRIC_DIMS = Dimensions(width, height)
        self.INGRESS_TILES = ingress
        self.EGRESS_TILES = egress
        if debug_level:
            print(
                f"Fabric dimension: w{self.FABRIC_DIMS.width()} x h{self.FABRIC_DIMS.height()}"
            )
            print(f"Number of ingress: {len(self.INGRESS_TILES)}")
            print(f"Number of egress : {len(self.EGRESS_TILES)}")

        # Useful for error checking purposes.
        self.NUM_WEST_INOUT = 0
        self.NUM_EAST_INOUT = 0
        self._update_number_of_lvds_tile()
        if debug_level:
            print(f"Number of West inout : {self.NUM_WEST_INOUT}")
            print(f"Number of East inout : {self.NUM_EAST_INOUT}")

        # Pin assignment data structure
        # Sorted list for user input and output PE coordinates.
        self.pe_west_portlist = []
        self.pe_east_portlist = []

        # Input and output map
        # key: PE (x, y, color)
        # value: index
        # After create_pin_assignments is successful, these two members will be
        # populated for query later.
        self.pe_tile_west_map = {}
        self.pe_tile_east_map = {}

        # This set keeps track of which lvds tile have been assigned globally.
        # Using this prevents any LVDS being used for bidirectional I/O.
        self._used_lvds_indices = set()

        # Route information data structure
        # Dictionary between lvds tile and PE ports
        # key: (x, y, color)
        # value: CslIoRouting
        self.routes = {}

    def _update_number_of_lvds_tile(self):
        """
        Iterate through ingress and egress and count the number of inout (bidirectional)
        tiles. It updates NUM_WEST_INOUT and NUM_EAST_INOUT.
        The function does the follow two checks:
        1. ingress and egress connections are sorted by x,
        2. at least 1 I/O on 
        3. every index in ingress and egress connections are pair as inout,
        """
        assert len(self.INGRESS_TILES) == len(
            self.EGRESS_TILES
        ), "Expects ingress and egress to have the same number of tiles"

        # Save ingress tiles for West and East
        # Errors out if a tile is not on the West or East edge
        prev_tile_x = 0
        for tile in self.INGRESS_TILES:
            assert self._is_edge_tile(
                tile
            ), f"Expects {tile.x()}, {tile.y()} to be at the edge of fabric"

            if tile.x() == 0:
                self.NUM_WEST_INOUT += 1
            else:  # self.FABRIC_DIMS.width() - 1:
                self.NUM_EAST_INOUT += 1

            # check x value must be increasing
            if prev_tile_x <= tile.x():
                prev_tile_x = tile.x()
            else:
                raise RuntimeError("Ingress connection is not sorted by x")

        # Save egress tiles for West and East
        # Errors out if a tile is not on the West or East edge
        prev_tile_x = 0
        west_inout_cnt = 0
        east_inout_cnt = 0
        for tile in self.EGRESS_TILES:
            assert self._is_edge_tile(
                tile
            ), f"Expects {tile.x()}, {tile.y()} to be at the edge of fabric"

            if tile.x() == 0:
                west_inout_cnt += 1
            else:  # self.FABRIC_DIMS.width() - 1:
                east_inout_cnt += 1

            # check x value must be increasing
            if prev_tile_x <= tile.x():
                prev_tile_x = tile.x()
            else:
                raise RuntimeError("Egress connection is not sorted by x")

        if self.NUM_WEST_INOUT == 0:
            raise RuntimeError("No I/O on West edge")

        if self.NUM_EAST_INOUT == 0:
            raise RuntimeError("No I/O on East edge")

        if west_inout_cnt != self.NUM_WEST_INOUT:
            raise RuntimeError("West I/O are not bidirectional")

        if east_inout_cnt != self.NUM_EAST_INOUT:
            raise RuntimeError("East I/O are not bidirectional")

    def _is_edge_tile(self, tile):
        return tile.x() in (0, self.FABRIC_DIMS.width() - 1)

    def streamer_tile_index(self, pe_tuple):
        if pe_tuple in self.pe_tile_west_map:
            return self.pe_tile_west_map[pe_tuple]
        elif pe_tuple in self.pe_tile_east_map:
            return self.pe_tile_east_map[pe_tuple]
        return None

    def is_streamer_tile_used(self, lvds_index):
        """
        Return true if the tile is used by memcpy netlist already.
        False otherwise.
        """
        return lvds_index in self._used_lvds_indices

    def add_streamer_tile_to_used(self, lvds_index):
        """
        Add index to used_lvds_indices
        """
        self._used_lvds_indices.add(lvds_index)

    def is_sufficient_streamer(self):
        """
        Raise a RuntimeError when the number of West (or East) user tile exceeds the
        number the physical LVDS tiles.
        """
        if len(self.pe_west_portlist) > self.NUM_WEST_INOUT:
            raise RuntimeError(
                "Expects number of PE inputs to be less than the number of West ingress tile."
            )

        if len(self.pe_east_portlist) > self.NUM_EAST_INOUT:
            raise RuntimeError(
                "Expects number of PE inputs to be less than the number of West ingress tile."
            )
