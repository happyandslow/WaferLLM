---
title: "GEMV 7: Routes and Fabric DSDs, Part II"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-07-routes-2"
crawled_at: "2026-02-14 23:55:12"
---

# GEMV 7: Routes and Fabric DSDs, Part II

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-07-routes-2](https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-07-routes-2)

Continuing from the previous example, we now break up a single GEMV
computation among a 2 x 2 square of PEs.

The host program copies `b` into the `y` tensor of the left column of PEs,
with PE (0, 0) getting the first `M/2` values and PE (0, 1) getting the
last `M/2` values.

Each PE also gets a corresponding chunk of `A`.
The left PEs get the left `N/2` columns and the right PEs
get the right `N/2` columns,
while the upper PEs get the upper `M/2` rows and the lower PEs
get the lower `M/2` rows.
In other words, the northwest PE gets the northwest quadrant of `A`,
the northeast PE gets the northeast quadrant of `A`, and so on.

The host program also copies `x` into the upper row of PEs,
with PE (0, 0) getting the first `N/2` values and the PE (1, 0)
gettin the last `N/2` values.

When the `compute` function is launched, the PEs in the top row begin
sending their respective elements of `x` to their routers,
along the color `x_color`.
These PEs send the elements of `x` both to the `SOUTH` and back down
their own `RAMP`.

On all four PEs, receiving a wavelet along `x_color` activates
`recv_x`. This task is a wavelet-triggered task (WTT): the wavelet’s
data is fed in as an argument to `recv_x`.

When a PE receives an element of `x` in the `recv_x` task, it computes
the corresponding piece of `Ax` and adds it to its local `y` tensor.
When a PE has received all corresponding elements of `x` along `x_color`,
(i.e., the first `N/2` values of `x` for the two left PEs,
and the last `N/2` values of `x` for the two right PEs),
it has finished computing its local contribution to `y`.

At this point, the local task `reduce` is activated.
The left column of PEs send their partial `y` result along the color
`ax_color` to the `EAST`, and the right column of PEs receives these
partial `y` results, and increments their `y` tensors
by the received values.
At this point, the right column of PEs contain the final result `y`,
with the first `M/2` elements in PE (1, 0)
and the last `M/2` elements in PE (1, 1).

Last, the host copies `y` from the right column of PEs,
and checks that the result is correct.

Note that in this program’s layout file, we no longer assign a `pe_id`
as a compile-time parameter.
Instead, we use the `<layout>` module in `pe_program.csl`
to determine the coordinates of the PE at runtime.
This can reduce compilation time by reducing the
number of unique PE programs that need to be compiled.
Specifically, by parameterizing a PE’s code (i.e., passing
parameters through `@set_tile_code`) we are creating more
unique PE programs as opposed to relying on
runtime-evaluated values.

## layout.csl

```csl
// total matrix dimensions
param M: i16;
param N: i16;

// Colors
const ax_color: color = @get_color(0); // sends/recvs partial result Ax EAST
const x_color:  color = @get_color(1); // sends/recvs elems x

// This example uses 2x2 PEs
const memcpy = @import_module("<memcpy/get_params>", .{
  .width = 2,
  .height = 2
});

layout {
  // PE coordinates are (column, row)
  @set_rectangle(2, 2);

  for (@range(i16, 2)) |pe_x| {
    for (@range(i16, 2)) |pe_y| {
      @set_tile_code(pe_x, pe_y, "pe_program.csl", .{
        .memcpy_params = memcpy.get_params(pe_x),
        .M_per_PE = M / 2,
        .N_per_PE = N / 2,
        .ax_color = ax_color,
        .x_color = x_color
      });
    }
  }

  // Top left PE (0, 0)
  @set_color_config(0, 0, ax_color, .{.routes = .{ .rx = .{RAMP}, .tx = .{EAST}  }});
  @set_color_config(0, 0, x_color,  .{.routes = .{ .rx = .{RAMP}, .tx = .{RAMP, SOUTH} }});

  // Top right PE (1, 0)
  @set_color_config(1, 0, ax_color, .{.routes = .{ .rx = .{WEST},  .tx = .{RAMP} }});
  @set_color_config(1, 0, x_color,  .{.routes = .{ .rx = .{RAMP}, .tx = .{RAMP, SOUTH} }});

  // Bottom left PE (0, 1)
  @set_color_config(0, 1, ax_color, .{.routes = .{ .rx = .{RAMP}, .tx = .{EAST} }});
  @set_color_config(0, 1, x_color,  .{.routes = .{ .rx = .{NORTH}, .tx = .{RAMP} }});

  // Bottom right PE (1, 1)
  @set_color_config(1, 1, ax_color, .{.routes = .{ .rx = .{WEST}, .tx = .{RAMP} }});
  @set_color_config(1, 1, x_color,  .{.routes = .{ .rx = .{NORTH}, .tx = .{RAMP} }});

  // export symbol names
  @export_name("A", [*]f32, true);
  @export_name("x", [*]f32, true);
  @export_name("y", [*]f32, true);
  @export_name("compute", fn()void);
}

```

## pe\_program.csl

```csl
param memcpy_params: comptime_struct;

// Matrix dimensions
param M_per_PE: i16;
param N_per_PE: i16;

// Colors
param ax_color: color; // sends partial result Ax EAST
param x_color:  color; // sends elems x SOUTH/ recvs elems x from NORTH

// Queue IDs
const ax_color_oq: output_queue = @get_output_queue(2);
const ax_color_iq: input_queue  = @get_input_queue(2);
const x_color_oq:  output_queue = @get_output_queue(3);
const x_color_iq:  input_queue  = @get_input_queue(3);

// Task ID used by exit task to unblock cmd stream
const exit_task_id:   local_task_id = @get_local_task_id(9);

// Task ID used by reduce task
const reduce_task_id: local_task_id = @get_local_task_id(10);

// Data task ID for task recv_x, consumes x_color wlts
// On WSE-2, data task IDs are created from colors; on WSE-3, from input queues
const recv_x_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(x_color)
  else if (@is_arch("wse3")) @get_data_task_id(x_color_iq);

// memcpy module provides infrastructure for copying data
// and launching functions from the host
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// layout module provides PE coordinates at runtime
const layout_mod = @import_module("<layout>");

// 48 kB of global memory contain A, x, b, y
var A: [M_per_PE*N_per_PE]f32; // A is stored column major
var x: [N_per_PE]f32;
var y: [M_per_PE]f32;

// DSDs for accessing A, x, y
// A_dsd accesses column of A
var A_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &A, .extent = M_per_PE });
var x_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &x, .extent = N_per_PE });
var y_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = M_per_PE });

// ptrs to A, x, b, y will be advertised as symbols to host
var A_ptr: [*]f32 = &A;
var x_ptr: [*]f32 = &x;
var y_ptr: [*]f32 = &y;

fn is_top_row() bool {
  return (layout_mod.get_y_coord() == 0);
}

fn is_left_col() bool {
  return (layout_mod.get_x_coord() == 0);
}

task reduce() void {
  if (is_left_col()) {
    const out_dsd = @get_dsd(fabout_dsd, .{
                      .fabric_color = ax_color, .extent = M_per_PE,
                      .output_queue = ax_color_oq
                    });
    // After fmovs is done, activate exit_task to unblock cmd stream
    @fmovs(out_dsd, y_dsd, .{ .async = true, .activate = exit_task_id });

  } else {
    const in_dsd = @get_dsd(fabin_dsd, .{
                     .fabric_color = ax_color, .extent = M_per_PE,
                     .input_queue  = ax_color_iq
                   });
    // After fadds is done, activate exit_task to unblock cmd stream
    @fadds(y_dsd, y_dsd, in_dsd, .{ .async = true, .activate = exit_task_id });
  }
}

// Use to keep track of # of invocations of recv_x task
// when num_recv_x == N_per_PE, we are done receiving x elements
var num_recv_x: i16 = 0;

task recv_x(x_val: f32) void {
  @fmacs(y_dsd, y_dsd, A_dsd, x_val);
  A_dsd = @increment_dsd_offset(A_dsd, M_per_PE, f32);

  num_recv_x += 1;
  if (num_recv_x == N_per_PE) {
    @activate(reduce_task_id);
  }
}

// The top row sends x values along x_color to launch recv_x
fn compute() void {
  if (is_top_row()) {
    const send_x_dsd = @get_dsd(fabout_dsd, .{
                         .fabric_color = x_color, .extent = N_per_PE,
                         .output_queue = x_color_oq
                       });
    @fmovs(send_x_dsd, x_dsd, .{ .async = true });
  }
}

task exit_task() void {
  sys_mod.unblock_cmd_stream();
}

comptime {
  // When exit_task_id is activated, exit_task will execute
  @bind_local_task(exit_task, exit_task_id);

  // reduce is local task activated by ID reduce_task_ID
  @bind_local_task(reduce, reduce_task_id);

  // recv_x is wavelet-triggered task (WTT) activated by receiving
  // wavelets along color x_color, which corresponds to recv_x_task_id
  // On WSE-3, these wavelets are received in input queue x_color_iq
  @bind_data_task(recv_x, recv_x_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(ax_color_oq, .{ .color = ax_color });
    @initialize_queue(ax_color_iq, .{ .color = ax_color });
    @initialize_queue(x_color_oq,  .{ .color = x_color });
    @initialize_queue(x_color_iq,  .{ .color = x_color });
  }

  @export_symbol(A_ptr, "A");
  @export_symbol(x_ptr, "x");
  @export_symbol(y_ptr, "y");
  @export_symbol(compute);
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Matrix dimensions
N = int(compile_data['params']['N'])
M = int(compile_data['params']['M'])

# Construct A, x, b
A = np.arange(M*N, dtype=np.float32).reshape(M,N)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Size of N dimension on each PE
N_per_PE = N // 2
M_per_PE = M // 2

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for A, x, y on device
A_symbol = runner.get_id('A')
x_symbol = runner.get_id('x')
y_symbol = runner.get_id('y')

# Load and run the program
runner.load()
runner.run()

# Copy b into y of PEs (0, 0) and (0, 1)
# PE (0, 0) gets first M/2 elements; PE (0, 1) gets last M/2 elements
runner.memcpy_h2d(y_symbol, b, 0, 0, 1, 2, M_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Copy chunks of A into all PEs
# PE (0, 0) gets A[0:M/2,0:N/2], PE (1, 0) gets A[0:M/2][N/2:N]
# PE (0, 1) gets A[M/2:M,0:N/2], PE (1, 1) gets A[M/2:M][N/2:N]
# Each chunk on each PE is stored column major
A_prepared = A.reshape(2, M_per_PE, 2, N_per_PE).transpose(0, 2, 3, 1).ravel()
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, 2, 2, M_per_PE*N_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Copy x into PEs (0, 0) and (1, 0)
# PE (0, 0) gets first N/2 elements; PE (1, 0) gets last N/2 elements
runner.memcpy_h2d(x_symbol, x, 0, 0, 2, 1, N_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Launch the compute function on device
runner.launch('compute', nonblock=False)

# Copy y back from PEs (1, 0) and (1, 1)
# First M/2 elements from PE (1, 0); Last M/2 elements from PE (1, 1)
y_result = np.zeros([M], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, 1, 0, 1, 2, M_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected)
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=9,4 \
--fabric-offsets=4,1 --params=M:4,N:6 -o out --memcpy --channels 1
cs_python run.py --name out

```
[previous

GEMV 6: Routes and Fabric DSDs, Part I](csl/code-examples/tutorial-gemv-06-routes-1.md "previous page")
[next

GEMV 8: Routes and Fabric DSDs, Part III](csl/code-examples/tutorial-gemv-08-routes-3.md "next page")
