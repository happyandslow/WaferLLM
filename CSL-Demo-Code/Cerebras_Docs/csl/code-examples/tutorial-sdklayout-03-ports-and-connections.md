---
title: "SdkLayout 3: Ports and connections"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-sdklayout-03-ports-and-connections"
crawled_at: "2026-02-14 23:55:27"
---

# SdkLayout 3: Ports and connections

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-sdklayout-03-ports-and-connections](https://sdk.cerebras.net/csl/code-examples/tutorial-sdklayout-03-ports-and-connections)

This tutorial demonstrates how to attach ports to code regions
and then connect those ports together. It instantiates two
code regions that send data to a third code region. The receiving
code region adds the input streams element-wise and then sends
the result out and towards a fourth code region that saves the
result on device memory.

There are two kinds of ports: input ports and output ports. It is
only possible to connect an output port to an input port. When
we do that the `SdkLayout` compiler will automatically find and
encode a path between them.

## sender.csl

```csl
param size: u16;
param tx: color;

const out_q = @get_output_queue(0);

export var data = [10]u16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

const data_dsd = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{size} -> data[i]});

const output = @get_dsd(fabout_dsd, .{.extent = size,
                                      .fabric_color = tx,
                                      .output_queue = out_q});

const main_id = @get_local_task_id(8);
task main() void {
  @mov16(output, data_dsd, .{.async = true});
}

comptime {
  @bind_local_task(main, main_id);
  @activate(main_id);

  if (@is_arch("wse3")) {
    @initialize_queue(out_q, .{.color = tx});
  }
}

```

## receiver.csl

```csl
param size: u16;
param rx: color;

const in_q = @get_input_queue(0);

export var data: [size]u16;

const data_dsd = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{size} -> data[i]});

const input = @get_dsd(fabin_dsd, .{.extent = size,
                                    .fabric_color = rx,
                                    .input_queue = in_q});

const main_id = @get_local_task_id(8);
task main() void {
  @mov16(data_dsd, input, .{.async = true});
}

comptime {
  @bind_local_task(main, main_id);
  @activate(main_id);

  @initialize_queue(in_q, .{.color = rx});
}

```

## add2vec.csl

```csl
param size: u16;
param rx1: color;
param rx2: color;
param tx: color;

const in_q1 = @get_input_queue(0);
const in_q2 = @get_input_queue(1);
const out_q = @get_output_queue(0);

const input1 = @get_dsd(fabin_dsd, .{.extent = size,
                                     .fabric_color = rx1,
                                     .input_queue = in_q1});

const input2 = @get_dsd(fabin_dsd, .{.extent = size,
                                     .fabric_color = rx2,
                                     .input_queue = in_q2});

const output = @get_dsd(fabout_dsd, .{.extent = size,
                                      .fabric_color = tx,
                                      .output_queue = out_q});

// WSE3 does not allow multiple fabric inputs per DSD operation.
// Therefore, we introduce a FIFO for portability between WSE2
// and WSE3.
var buffer: [size]u16;
const fifo = @allocate_fifo(buffer);
const main_id = @get_local_task_id(8);
task main() void {
  @mov16(fifo, input2, .{.async = true});
  @add16(output, input1, fifo, .{.async = true});
}

comptime {
  @bind_local_task(main, main_id);
  @activate(main_id);

  @initialize_queue(in_q1, .{.color = rx1});
  @initialize_queue(in_q2, .{.color = rx2});

  if (@is_arch("wse3")) {
    @initialize_queue(out_q, .{.color = tx});
  }
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (
    Color,
    Edge,
    Route,
    RoutingPosition,
    SdkLayout,
    SdkTarget,
    SdkRuntime,
    SimfabConfig,
    get_platform,
)

parser = argparse.ArgumentParser()
parser.add_argument('--cmaddr', help='IP:port for CS system')
parser.add_argument(
    '--arch',
    choices=['wse2', 'wse3'],
    default='wse3',
    help='Target WSE architecture (default: wse3)'
)
args = parser.parse_args()

###########
### Layout
###########
# If 'cmaddr' is empty then we create a default simulation layout.
# If 'cmaddr' is not empty then 'config' and 'target' are ignored.
config = SimfabConfig(dump_core=True)
target = SdkTarget.WSE3 if (args.arch == 'wse3') else SdkTarget.WSE2
platform = get_platform(args.cmaddr, config, target)
layout = SdkLayout(platform)

######################
### Common invariants
######################
size = 10
sender_routes = RoutingPosition().set_input([Route.RAMP])
receiver_routes = RoutingPosition().set_output([Route.RAMP])

#################################
### Sender 1 and port 'tx1_port'
#################################
sender1 = layout.create_code_region('./sender.csl', 'sender1', 1, 1)
# Color 'tx1' is scoped because even though the name of the color is
# 'tx' for both senders, colors must be globally unique for the
# compiler to assign different values to them. By scoping colors like
# this we are effectively uniqueing them since code regions are unique
# (i.e., no two regions can have the same name).
tx1 = sender1.color('tx')
sender1.set_param_all('size', size)
sender1.set_param_all(tx1)
# A sender port is created using a color ('tx1'), an edge (in this
# example the edge doesn't matter since we have a 1x1 code region),
# a list of routing positions and a size. The routing positions for an
# output port must not contain output routes (if they do, an error will
# be emitted). That's because the compiler is free to chose any output
# route depending on what's globally optimal. Finally, the 'size' is
# used to verify compatibility between connected ports.
tx1_port = sender1.create_output_port(tx1, Edge.RIGHT, [sender_routes], size)

#################################
### Sender 2 and port 'tx2_port'
#################################
sender2 = layout.create_code_region('./sender.csl', 'sender2', 1, 1)
tx2 = sender2.color('tx')
sender2.set_param_all('size', size)
sender2.set_param_all(tx2)
tx2_port = sender2.create_output_port(tx2, Edge.RIGHT, [sender_routes], size)

#########################
### Placement of senders
#########################
# We place the senders in arbitrary locations in the layout as
# an example that demonstrates the ability of the framework to automatically
# produce paths between input and output ports.
sender1.place(2, 2)
sender2.place(4, 7)

############
### Add2vec
############
add2vec = layout.create_code_region('./add2vec.csl', 'add2vec', 1, 1)
rx1 = Color('rx1')
rx2 = Color('rx2')
tx = Color('tx')
add2vec.set_param_all('size', size)
add2vec.set_param_all(rx1)
add2vec.set_param_all(rx2)
add2vec.set_param_all(tx)
rx1_port = add2vec.create_input_port(rx1, Edge.RIGHT, [receiver_routes], size,)
rx2_port = add2vec.create_input_port(rx2, Edge.RIGHT, [receiver_routes], size,)
tx_port = add2vec.create_output_port(tx, Edge.LEFT, [sender_routes], size,)
add2vec.place(7, 4)

#############
### Receiver
#############
receiver = layout.create_code_region('./receiver.csl', 'receiver', 1, 1)
rx = Color('rx')
receiver.set_param_all('size', size)
receiver.set_param_all(rx)
rx_port = receiver.create_input_port(rx, Edge.LEFT, [receiver_routes], size,)
receiver.place(3, 3)

#####################
### Port connections
#####################
# This is the key part of this example. The ports defined above for
# each code region, are now connected. The physical location of the
# ports can be arbitrary because the SdkLayout compiler will find
# optimal paths automatically.
layout.connect(tx1_port, rx1_port)
layout.connect(tx2_port, rx2_port)
layout.connect(tx_port, rx_port)

#################
### Compilation
#################
# Compile the layout and use 'out' as the prefix for all
# produced artifacts.
compile_artifacts = layout.compile(out_prefix='out')

#############
### Runtime
#############
# Create the runtime using the compilation artifacts and the execution platform.
runtime = SdkRuntime(compile_artifacts, platform, memcpy_required=False)
runtime.load()
runtime.run()
runtime.stop()

#################
### Verification
#################
# Finally, once execution has stopped, read the result from the receiver's
# memory and compare with expected value.
expected = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=np.uint16)
actual = runtime.read_symbol(3, 3, 'data').view(np.uint16)
assert np.array_equal(expected, actual)
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cs_python run.py --arch=wse3

```
[previous

SdkLayout 2: Basic routing](csl/code-examples/tutorial-sdklayout-02-routing.md "previous page")
[next

SdkLayout 4: Host-to-device and device-to-host data streaming](csl/code-examples/tutorial-sdklayout-04-h2d-d2h.md "next page")
