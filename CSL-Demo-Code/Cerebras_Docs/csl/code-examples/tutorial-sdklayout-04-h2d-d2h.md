---
title: "SdkLayout 4: Host-to-device and device-to-host data streaming"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-sdklayout-04-h2d-d2h"
crawled_at: "2026-02-14 23:55:20"
---

# SdkLayout 4: Host-to-device and device-to-host data streaming

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-sdklayout-04-h2d-d2h](https://sdk.cerebras.net/csl/code-examples/tutorial-sdklayout-04-h2d-d2h)

This tutorial demonstrates how we can connect ports to the
host to allow us to stream data in and out of the WSE.

It uses the ‘add2vec’ code region that was also used in
tutorial [SdkLayout 3: Ports and connections](csl/code-examples/tutorial-sdklayout-03-ports-and-connections.md#sdkruntime-sdklayout-03-ports-and-connections) but instead of
using sender/receiver code regions it creates streams directly
to/from the host.

Similar to connections between input and output ports (see tutorial
[SdkLayout 3: Ports and connections](csl/code-examples/tutorial-sdklayout-03-ports-and-connections.md#sdkruntime-sdklayout-03-ports-and-connections)) paths to/from ports
to/from the edge of the wafer are produced automatically.

For now, it is only possible to create input/output streams
to/from single-PE ports. If a port consists of more than one PE then
an adaptor layer must be created explicitly to funnel the data
through a single PE port. The next tutorial shows an example
of such a configuration.

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

#############################
### Input and output streams
#############################
in_stream1 = layout.create_input_stream(rx1_port)
in_stream2 = layout.create_input_stream(rx2_port)
out_stream = layout.create_output_stream(tx_port)

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

# Data to be sent to the device and buffer to accept the result from
# the device.
data1 = np.random.randint(0, 1000, size=size, dtype=np.uint32)
data2 = np.random.randint(1000, 2000, size=size, dtype=np.uint32)
actual = np.empty(size, dtype=np.uint32)

# Send and receive to/from the device.
runtime.send(in_stream1, data1, nonblock=True)
runtime.send(in_stream2, data2, nonblock=True)
runtime.receive(out_stream, actual, size, nonblock=True)

runtime.stop()

#################
### Verification
#################
# Verify that the received data is correct.
expected = data1 + data2
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

SdkLayout 3: Ports and connections](csl/code-examples/tutorial-sdklayout-03-ports-and-connections.md "previous page")
[next

SdkLayout 5: Generalized matrix-vector multiplication (GEMV)](csl/code-examples/tutorial-sdklayout-05-gemv.md "next page")
