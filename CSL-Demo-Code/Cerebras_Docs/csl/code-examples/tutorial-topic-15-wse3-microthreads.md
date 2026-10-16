---
title: "Topic 13: WSE-3 Microthreads"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-topic-15-wse3-microthreads"
crawled_at: "2026-02-14 23:55:05"
---

# Topic 13: WSE-3 Microthreads

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-topic-15-wse3-microthreads](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-15-wse3-microthreads)

Unlike WSE-2, the WSE-3 architecture exposes microthread IDs.
This example demonstrates the use of explicit microthread IDS
on the WSE-3 architecture.

On WSE-2, the queue ID of an input or output fabric DSD corresponds to the
ID of the microthread in which that operation executes.
On WSE-3, queue IDs and microthreads can be decoupled, so that any
microthread ID 0 to 7 can be used with any of queues 0 to 7.

In this example, the left PE sends `M` wavelets to the right PE over
the color `send_color`.
These wavelets are sent in an asynchronous `@fmovs` operation which
copies from the `y` array via `y_dsd` into `out_dsd`.
`out_dsd` is a `fabout_dsd` associated with the color `send_color`,
and the output queue with ID 2.
The `@fmovs` operation is launched using microthread ID 4.

The right PE receives these `M` wavelets on the same color (called
`right_color` in `right_pe.csl`) via `in_dsd`, which uses input
queue with ID 2.
The asynchronous `@fmovs` operation which receives these wavelets
and copies them into `y` is launched using microthread ID 5.

Decoupling microthread IDs from queue IDs can provide valuable flexibility
in managing program resource usage, and conserve microthreads.

By using explicit microthread IDs, we allow CSL’s DSR allocator to use fewer
DSRs in situations where fabric DSD operands are not known at compile time.

Additionally, on the WSE-3, output queues cannot be re-used with a different
color if they have not yet been drained, and CSL does not yet support a
mechanism for guaranteeing that a given queue is empty.
This may force the programmer to use more output queues than needed, which in
turn can lead to overusing microthread IDs (if they are not explicitly
specified, they default to the respective queue IDs).
By allowing explicit microthread IDs, a programmer can share microthreads
between output queues, and thus conserve microthreads for other operations.
Note, however, that two operations cannot concurrently use the same microthread.

## layout.csl

```csl
// Colors
const send_color: color = @get_color(0); // Color used to send/recv data between PEs

// This example only uses 2 PEs
const memcpy = @import_module("<memcpy/get_params>", .{ .width = 2, .height = 1 });

layout {
  // PE coordinates are (column, row)
  @set_rectangle(2, 1);

  // Left PE (0, 0)
  @set_tile_code(0, 0, "left_pe.csl", .{
    .memcpy_params = memcpy.get_params(0), .send_color = send_color });

  // Left PE sends to the right
  @set_color_config(0, 0, send_color, .{.routes = .{ .rx = .{RAMP}, .tx = .{EAST} }});

  // Right PE (1, 0)
  @set_tile_code(1, 0, "right_pe.csl", .{
    .memcpy_params = memcpy.get_params(1), .recv_color = send_color });

  // Right PE receives from left PE
  @set_color_config(1, 0, send_color, .{.routes = .{ .rx = .{WEST}, .tx = .{RAMP} }});

  // export symbol names
  @export_name("y", [*]f32, true);
  @export_name("compute", fn()void);
}

```

## left\_pe.csl

```csl
param memcpy_params: comptime_struct;

param send_color: color;

const M: i16 = 10;

// Task IDs
const exit_task_id: local_task_id = @get_local_task_id(9);

// Queue and microthread IDs
const send_color_oq = @get_output_queue(2);
const send_color_ut = @get_ut_id(4);

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

var y: [M]f32;
var y_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> y[i] });
var y_ptr: [*]f32 = &y;

fn compute() void {
  const out_dsd = @get_dsd(fabout_dsd, .{
                    .fabric_color = send_color, .extent = M,
                    .output_queue = send_color_oq
                  });
  @fmovs(out_dsd, y_dsd, .{ .async = true, .ut_id = send_color_ut,
                            .activate = exit_task_id });
}

task exit_task() void {
  sys_mod.unblock_cmd_stream();
}

comptime {
  @bind_local_task(exit_task, exit_task_id);

  @initialize_queue(send_color_oq, .{ .color = send_color });

  @export_symbol(y_ptr, "y");
  @export_symbol(compute);
}

```

## right\_pe.csl

```csl
param memcpy_params: comptime_struct;

param recv_color: color;

const M: i16 = 10;

// Task IDs
const exit_task_id: local_task_id = @get_local_task_id(9);

// Queue and microthread IDs
const recv_color_iq = @get_input_queue(2);
const recv_color_ut = @get_ut_id(5);

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

var y: [M]f32;
var y_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> y[i] });
var y_ptr: [*]f32 = &y;

fn compute() void {
  const in_dsd = @get_dsd(fabin_dsd, .{
                   .fabric_color = recv_color, .extent = M,
                   .input_queue = recv_color_iq
                 });
  @fmovs(y_dsd, in_dsd, .{ .async = true, .ut_id = recv_color_ut,
                           .activate = exit_task_id });
}

task exit_task() void {
  sys_mod.unblock_cmd_stream();
}

comptime {
  @bind_local_task(exit_task, exit_task_id);

  @initialize_queue(recv_color_iq, .{ .color = recv_color });

  @export_symbol(y_ptr, "y");
  @export_symbol(compute);
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

M = 10
y = np.arange(M, dtype=np.float32)
y_expected = y

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for A, x, y on device
y_symbol = runner.get_id('y')

# Load and run the program
runner.load()
runner.run()

# Copy y into PE (0, 0)
runner.memcpy_h2d(y_symbol, y, 0, 0, 1, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Launch the compute function on device
runner.launch('compute', nonblock=False)

# Copy y back from PE (1, 0)
y_result = np.zeros([M], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, 1, 0, 1, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected, atol=0.01, rtol=0)
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=11,3 \
--fabric-offsets=4,1 -o out --memcpy --channels 1
cs_python run.py --name out

```
[previous

Topic 14: Color Swap](csl/code-examples/tutorial-topic-14-color-swap.md "previous page")
[next

Pipeline 1: Redirect fabin to fabout](csl/code-examples/tutorial-pipeline-01-basic.md "next page")
