#!/usr/bin/env python3


import numpy as np

from cerebras.elf.poke_data import poke_data
from cerebras.sdk.runtime.cslwsenetlist import CslWseNetlist


class CslWseRouterAssembler:
    def __init__(self, netlist: CslWseNetlist, debug_level: int = 0):
        """
        Generates MEMCPY_XY_ROUTES.elf from routes in netlist.
        """
        self._netlist = netlist
        self._debug_level = debug_level

    def generate_route_elf(self, temp_dir: str):
        """
        Write CslIoRouting value to an ELF file.
        """
        if self._debug_level:
            print("Writing: MEMCPY_XY_ROUTES.elf")

        bits = []
        per_pe_sizes = []
        rectangles = []
        sym_addrs = []

        def add_memcpy_route_to_data(route_tuple, addr):
            # rectangle: (x, y), (w, h)
            dims = (route_tuple[1].width(), route_tuple[1].height())
            offset = (route_tuple[0].x(), route_tuple[0].y())
            rectangle = (offset, dims)
            data = route_tuple[2]

            np_arr = np.array(data, dtype=np.uint16)

            if self._debug_level:
                print(f"memcpy route: rect   ={rectangle}")
                vhex = np.vectorize(hex)
                print(f"memcpy route: np_arr ={vhex(np_arr)}")

            bits.append(np_arr.tobytes())
            per_pe_sizes.append(2)  # Each value of the traffic map is 2 bytes
            rectangles.append(rectangle)
            sym_addrs.append(addr)

        for pe_tuple in self._netlist.routes:
            memcpy_route = self._netlist.routes[pe_tuple]
            if memcpy_route.has_y_route():
                route_tuple = memcpy_route.y_route()
                # Multiply by 2 to convert to byte address
                addr = memcpy_route.traffic_map_address() * 2
                add_memcpy_route_to_data(route_tuple, addr)

            if memcpy_route.has_x_route():
                route_tuple = memcpy_route.x_route()
                # Multiply by 2 to convert to byte address
                addr = memcpy_route.traffic_map_address() * 2
                add_memcpy_route_to_data(route_tuple, addr)

            if memcpy_route.has_receiver_route():
                route_tuple = memcpy_route.receiver_route()
                # Multiply by 2 to convert to byte address
                addr = memcpy_route.traffic_map_address() * 2
                add_memcpy_route_to_data(route_tuple, addr)

        fab_w = self._netlist.FABRIC_DIMS.width()
        fab_h = self._netlist.FABRIC_DIMS.height()

        if len(bits) == 0:
            return

        result_elf = f"{temp_dir}/MEMCPY_XY_ROUTES.elf"

        poke_data(
            result_elf, bits, fab_w, fab_h, rectangles, sym_addrs, per_pe_sizes
        )

        return result_elf
