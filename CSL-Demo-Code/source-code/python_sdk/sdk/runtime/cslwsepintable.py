#!/usr/bin/env python3

from typing import Dict, List

from cerebras.geometry.geometry import CardinalDirection
from cerebras.sdk.runtime.cslwsenetlist import CslWseNetlist, PEPort


class CslWsePinTable:
    """
    Cerebras Software Language Wafer Scale Engine Pin Table Library captures the
    information about the ingress and egress links on the wafer for WSE and simfab.
    The purpose of the pin table helps assign user input and output tile to ingress
    and egress tile.

    ingress: List of coordinate.
    engress: List of coordinate.
    debug_level enables printing the info during initialization.

    Usage:

.. code-block:: python

    pin_table = CslWsePinTable(fabric_dims, ingress, egress)
    pin_table.create_pin_assignments(input_list, output_list)

    # Get the West and East map for result
    result_west_map = pin_table.pe_tile_west_map
    result_east_map = pin_table.pe_tile_east_map
    """

    # TODO: Rewrite this class in C++ later when production.
    def __init__(self, netlist: CslWseNetlist, debug_level: int = 0):
        self._netlist = netlist
        self._debug_level = debug_level

        # Get the first East index
        self._first_east_index = self._netlist.NUM_WEST_INOUT

    def create_pin_assignments(
        self, input_pe_list: List, output_pe_list: List,
    ):
        """
        Assign each PE to an ingress or egress first
        Algorithm:
        1. Sort input and output into 2 list: West and East list based on Y-coordinate.
        2. Pair up ingress and egress tile like zip and create 2 dictionaries: input and
           output.
        3. Minimize distance for both dictionaries.

        Args:
            input_pe_list is the list of (Tile, receiver_dir). Future will be a list of
            rectangles and color to support distribution and list contains direction.
            output_pe_list is the list of (Tile, sender_dir). Future will be a list of
            rectangles and color to support distribution and list contains direction.
        """
        # Empty list, early exit
        if len(input_pe_list) == 0 and len(output_pe_list) == 0:
            return

        # Iterate over lists to make sure no duplicate Tile.
        # Duplicate x, y, color would cause congestion and eventually incorrect data.
        self._check_duplicate_pe_list(input_pe_list + output_pe_list)

        # Organize input and output into two list into West and East and sort each of
        # them by Y-coordinate.
        self._sort_pe_lists(input_pe_list, output_pe_list)

        if self._debug_level:
            print(f"West={self._netlist.pe_west_portlist}")
            print(f"East={self._netlist.pe_east_portlist}")

        # Assign west and east I/O PEs to a temporary LVDS based on order.
        self._assign_tile_to_lvds()

        if self._debug_level:
            print(f"Before: West={self._netlist.pe_tile_west_map}")
            print(f"Before: East={self._netlist.pe_tile_east_map}")

        # Minimize the distance.
        self._minimize_route_distance()

    def _check_duplicate_pe_list(self, pe_list: List):
        """
        Iterate over list to make sure no duplicate x, y, color.
        """
        pe_tile_set = set()
        for pe_tuple in pe_list:
            pe_tile = pe_tuple[0]
            if pe_tile in pe_tile_set:
                raise RuntimeError(f"Found duplicated tile and color {pe_tile}")

    def _sort_pe_lists(
        self, input_pe_list: List, output_pe_list: List,
    ):
        """
        Sort all input and output on the same edge together for doing an assignment.
        Input will be sorted before output. PEs with the same y-coordinate will be left
        in the same order as the user appends to the original list.

        Args:
            input_pe_list is the list of (Tile, receiver_dir).
            output_pe_list is the list of (Tile, sender_dir).
        """
        # TODO: Balance the West and East list to minimize congestion on either
        # side later. Balancing can only happen for inputs and outputs on the
        # North or South side.

        for pe_tuple in input_pe_list:
            receiver_in_dir = pe_tuple[1]
            if receiver_in_dir is CardinalDirection.WEST:
                port_tuple = (PEPort.INPUT, receiver_in_dir)
                self._netlist.pe_west_portlist.append((pe_tuple[0], port_tuple))
            elif receiver_in_dir is CardinalDirection.EAST:
                port_tuple = (PEPort.INPUT, receiver_in_dir)
                self._netlist.pe_east_portlist.append((pe_tuple[0], port_tuple))
            else:
                raise RuntimeError(
                    "Only support inputs from either West or East"
                )

        for pe_tuple in output_pe_list:
            sender_out_dir = pe_tuple[1]
            if sender_out_dir is CardinalDirection.WEST:
                port_tuple = (PEPort.OUTPUT, sender_out_dir)
                self._netlist.pe_west_portlist.append((pe_tuple[0], port_tuple))
            elif sender_out_dir is CardinalDirection.EAST:
                port_tuple = (PEPort.OUTPUT, sender_out_dir)
                self._netlist.pe_east_portlist.append((pe_tuple[0], port_tuple))
            else:
                raise RuntimeError(
                    "Only support outputs to either West or East"
                )

        # Error check before pin assignment. This is an assertion because we've already
        # checked number of inputs is less than ingress tiles. And so we've done something
        # incorrect during algorithm the allocate which PE port for West or East edge.
        self._netlist.is_sufficient_streamer()

        # Sort by y (row)
        self._netlist.pe_west_portlist.sort(
            key=lambda pe_dir_tuple: pe_dir_tuple[0].y()
        )
        self._netlist.pe_east_portlist.sort(
            key=lambda pe_dir_tuple: pe_dir_tuple[0].y()
        )

    def _assign_tile_to_lvds(self):
        """
        Assign sorted PE lists, West and East, to a LVDS tile on respective side.
        """
        self._netlist.pe_tile_west_map = self._create_tile_to_lvds_map(
            self._netlist.pe_west_portlist, 0
        )

        self._netlist.pe_tile_east_map = self._create_tile_to_lvds_map(
            self._netlist.pe_east_portlist, self._first_east_index
        )

    def _create_tile_to_lvds_map(
        self, sorted_pe_dir_list: List, lvds_begin: int
    ):
        """
        Assign PE tile to either input and output dictionary.
        """
        # Dictionary of key: tuple, value is LVDS index
        tile_lvds_dict = {}

        curr_lvds_index = lvds_begin

        for pe_dir_tuple in sorted_pe_dir_list:
            # Assign to LVDS index and update the used LVDS tile
            # Add to index to coordinate color tuple dictionary and add to used list
            pe_tile = pe_dir_tuple[0]
            tile_lvds_dict[pe_tile] = curr_lvds_index
            self._netlist.add_streamer_tile_to_used(curr_lvds_index)

            # Update both indices because once an input or output tile taken, we don't
            # want that to be reused by the next input or output.
            curr_lvds_index = self._next_lvds_index(
                curr_lvds_index, len(self._netlist.INGRESS_TILES)
            )

        return tile_lvds_dict

    def _next_lvds_index(
        self, curr_lvds_index: int, lvds_index_end: List
    ) -> int:
        """
        Helper function to find next available LVDS index.
        """
        for next_lvds_index in range(curr_lvds_index, lvds_index_end):
            if self._netlist.is_streamer_tile_used(next_lvds_index):
                continue
            return next_lvds_index

        return lvds_index_end

    def _minimize_route_distance(self):
        """
        Minimize the estimated route distance on West and East PE tile and LVDS tile map.
        """
        self._minimize_distance_for_map(
            self._netlist.pe_tile_west_map,
            self._netlist.pe_west_portlist,
            self._netlist.NUM_WEST_INOUT,
        )

        self._minimize_distance_for_map(
            self._netlist.pe_tile_east_map,
            self._netlist.pe_east_portlist,
            self._netlist.NUM_EAST_INOUT,
        )

    def _minimize_distance_for_map(
        self, tile_lvds_dict, sorted_pe_dir_list: List, num_lvds
    ):
        """
        1. Start from the bottom of the list.
        2. Check to see if there any spare LVDS tiles
           if none, finish and done
        3. Shift the LVDS tile mapped to last PE down by 1
        4. Check new length
        5. if new length > current length, the update last_optimal_route_index
           else repeat 3 for next PE.
        6. if last_optimal_route_index >= len(sorted_pe_dir_list), then finish
           else shift LVDS tile on next PE by 1
        """
        # Either list empty or no more spare LVDS available to minimize distance.
        if len(sorted_pe_dir_list) == 0 or len(sorted_pe_dir_list) >= num_lvds:
            return

        last_optimal_route_index = 0
        # sorted_pe_dir_list is list of ((x, y, c), dir)
        last_pe_tuple = sorted_pe_dir_list[-1][0]

        # Make multiple iterations until all distance routes are minimial.
        while last_optimal_route_index != len(sorted_pe_dir_list) - 1:

            # No more room to shift LVDS.
            last_lvds_index = tile_lvds_dict[last_pe_tuple]
            if (
                last_lvds_index == num_lvds - 1
                or last_lvds_index == 2 * num_lvds - 1
            ):
                return

            # Optimize route from last index upto last optimized route.
            curr_route_index = len(sorted_pe_dir_list) - 1
            for pe_dir_tuple in reversed(sorted_pe_dir_list):
                pe_tile = pe_dir_tuple[0]
                # shift by 1
                curr_lvds_index = tile_lvds_dict[pe_tile]
                curr_est_route_dist = self._compute_distance(
                    pe_tile, self._netlist.INGRESS_TILES[curr_lvds_index]
                )
                next_lvds_index = curr_lvds_index + 1
                next_est_route_dist = self._compute_distance(
                    pe_tile, self._netlist.INGRESS_TILES[next_lvds_index]
                )

                # Update the current PE tile to map to new LVDS tile
                if next_est_route_dist < curr_est_route_dist:
                    tile_lvds_dict[pe_tile] = next_lvds_index
                # No more optimization for this iteration.
                else:
                    last_optimal_route_index = curr_route_index

                # Last route index to optimize.
                if curr_route_index == last_optimal_route_index:
                    break

                curr_route_index -= 1

    def _compute_distance(self, pe_tile, tile2) -> int:
        """
        Compute bounding box distance between the two PE as an estimate of routing distance.
        """
        pe_x = pe_tile.x()
        pe_y = pe_tile.y()
        w = 0
        if pe_x > tile2.x():
            w = pe_x - tile2.x()
        elif pe_x < tile2.x():
            w = tile2.x() - pe_x

        h = 0
        if pe_y > tile2.y():
            h = pe_y - tile2.y()
        elif pe_y < tile2.y():
            h = tile2.y() - pe_y

        return w + h

    def ingress_index_list(self) -> List:
        """
        Returns a list of indices used for input.
        """
        return self.get_lvds_index_list(
            self._netlist.pe_tile_west_map,
            self._netlist.pe_west_portlist,
            PEPort.INPUT,
        ) + self.get_lvds_index_list(
            self._netlist.pe_tile_east_map,
            self._netlist.pe_east_portlist,
            PEPort.INPUT,
        )

    def egress_index_list(self) -> List:
        """
        Returns a list of indices used for output.
        """
        return self.get_lvds_index_list(
            self._netlist.pe_tile_west_map,
            self._netlist.pe_west_portlist,
            PEPort.OUTPUT,
        ) + self.get_lvds_index_list(
            self._netlist.pe_tile_east_map,
            self._netlist.pe_east_portlist,
            PEPort.OUTPUT,
        )

    def get_lvds_index_list(
        self, tile_lvds_dict: Dict, sorted_pe_dir_list: List, dir: PEPort
    ):
        lvds_index_list = []
        for pe_dir_tuple in sorted_pe_dir_list:
            port_dir = pe_dir_tuple[1][0]
            if port_dir is dir:
                pe_tile = pe_dir_tuple[0]
                idx = tile_lvds_dict.get(pe_tile, None)
                assert idx is not None
                lvds_index_list.append(idx)
        return lvds_index_list
