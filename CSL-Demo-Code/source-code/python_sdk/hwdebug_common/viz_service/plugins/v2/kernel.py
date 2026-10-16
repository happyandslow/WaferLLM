"""
v1 viz api

this plugin implements the v1 api /api/v1/*
"""
from cerebras.hwdebug_common.viz_service.base import PluginBase
from cerebras.pb.viz.viz_common_pb2 import VizSchemaType
from cerebras.pb.viz.viz_explorer_pb2 import (
    VizExplorerResponse,
    VizExplorerResponseType,
)


class Plugin(PluginBase):
    ALIAS = None

    def handle_get(self, handler, url, req, qs):
        self.logger.debug("V2/kernel GET: %s", url)
        self.handler = handler
        path = req.path
        ctx = self.get_das()

        body = {"pe": [qs['x'][0], qs['y'][0]], "session_id": qs['session_id']}

        resp = VizExplorerResponse()
        resp.meta.type = VizSchemaType.ST_VIZ_EXPLORER_RESPONSE
        resp.data.details.response_type = VizExplorerResponseType.JSON
        resp = self.msg_to_dict(resp)
        resp['data']['data'] = body
        self.return_json(resp)
        return True
        kgs = ctx.get_kernel_graph_service()

        kernels = []
        nodes = kgs.nodes()
        num_debug_kernel_ports = 0

        """
      .def("dimensions", &KernelGraphService::dimensions)
      .def("send_port_of", &KernelGraphService::get_immediate_driver_of_str)
      .def("receive_port_of", &KernelGraphService::get_immediate_receiver_of_str)
      .def("nodes", &KernelGraphService::get_top_kernels)
      .def("ports", &KernelGraphService::get_all_ports_str)
      .def("get_associated_ports", &KernelGraphService::get_associated_port)
      .def("get_kernel_rect", &KernelGraphService::get_rect)
      .def("has_output_port", &KernelGraphService::has_output_port)
      .def("delta_t", &KernelGraphService::delta_t)
      .def("kernel_tensor_shape", &KernelGraphService::port_shape)
      .def("kernel_tensor_type", &KernelGraphService::port_data_type)
      .def("kernel_tensor_format", &KernelGraphService::port_packet_type)
      .def("kernel_port_splits", &KernelGraphService::port_deprecated_split)
      .def("kernel_port_dataformat", &KernelGraphService::kernel_port_data_format)
      .def("get_immediate_subkernels", &KernelGraphService::get_immediate_subkernels)
      .def("get_elfs_for_kernel", &KernelGraphService::get_elfs_for_kernel)
      .def("get_cigar_file_path", &KernelGraphService::get_cigar_file_path)
      .def("get_port_attr", &KernelGraphService::get_port_attr)
      .def("get_outward_skip_rc", &KernelGraphService::get_outward_skip_rc)
      .def("get_input_ports", &KernelGraphService::get_input_ports_str)
      .def("get_output_ports", &KernelGraphService::get_output_ports_str)
      .def("get_rect", &KernelGraphService::get_rect)
      .def("get_kernels_at_pe", &KernelGraphService::get_kernels_at_pe)
      .def("get_port_connections", &KernelGraphService::get_port_connections)
      .def("get_kernels_of_flavor", &KernelGraphService::get_kernels_of_flavor)
      """

        i = 0
        for node in nodes:

            for port in kgs.ports(node):
                port_attr = kgs.get_port_attr(node, port)
                if len(port_attr.dbg_pes):
                    self.logger.debug(
                        f"port {node}:{port} --> {port_attr.dbg_node}:{port_attr.dbg_port}"
                    )
                    num_debug_kernel_ports += 1

            rect = kgs.get_rect(node)
            kernels[i] = {
                'rect': str(rect),
                'port': {
                    'uri': f"api/v2/kernel/ports?kernel={node}",
                    'text': "ports",
                },
            }
            ++i

        """
    kernels = kgs.get_kernels_of_flavor("UnpackCore2")
    if(len(kernels) != 1):
        logging.error("Only one UnpackCore2 expected per design. Found: "
                      + len(kernels))
    r = kgs.lookup_kernel_object(kernels[0])
    c = r.rect()
    self.logger.debug(str(c))
    self.logger.debug("x = %s,%s", c.x, c.y)

    pt = Coordinate(c.x(), c.y())
    self.logger.debug(str(pt))

    kerns = kgs.get_kernels_from_point(pt)
    self.logger.debug(str(kerns))
    self.logger.debug("kernels at point:")
    for x in kerns:
        self.logger.debug("subkernel %s, %s", x.name(), str(x.rect()))

    subk = kgs.get_immediate_subkernels("unpack")
    self.logger.debug("port subkernels = %s", str(subk))

    rch = root.get_children()
    self.logger.debug("root kernels = %s", str(rch))
    for x in rch:
        n = x.name()
        self.logger.debug("Getting elfs for kernel %s", n)
        flav = x.flavor()
        elfs = kgs.get_elfs_for_kernel(n)

        children = x.get_children()
        self.logger.debug("children of %s = %s", x.name(), children)
        for child in children:
            n1 = child.name()
            celfs = kgs.get_elfs_for_kernel(n1)
            if celfs:
                self.logger.debug("   - Child: %s Elfs: %s", n1, celfs)
                for e in celfs:
                    if e not in elfs:
                        elfs.append(e)

        self.logger.debug("Kernel: %s Flavor: %s elfs: %s", n, flav, elfs)
"""

        body = {"data": kernels, "meta": {"type": "EXPLORER_KERNELS",}}
        self.return_json(body)
        return True
