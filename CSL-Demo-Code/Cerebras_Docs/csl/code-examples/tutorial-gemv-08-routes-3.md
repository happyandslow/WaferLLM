---
title: "GEMV 8: Routes and Fabric DSDs, Part III"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-08-routes-3"
crawled_at: "2026-02-14 23:55:04"
---

# GEMV 8: Routes and Fabric DSDs, Part III

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-08-routes-3](https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-08-routes-3)

Continuing from the previous example, we now extend the GEMV computation
to a `kernel_x_dim x kernel_y_dim` rectangle of PEs.
We make one simplification,
enforcing that `M` is a multiple of `kernel_y_dim`,
and `N` is a multiple of `kernel_x_dim`.

The host program copies `b` into the `y` tensor of the left column of PEs,
with each PE getting the corresponding `M/kernel_y_dim` values.
Each PE also gets a corresponding chunk of `A`,
consisting of `M/kernel_y_dim x N/kernel_x_dim` elements.
Similarly, the host program copies `x` into the upper row of PEs,
with each PE getting `N/kernel_x_dim` values.

When the `compute` function is launched, the PEs in the top row begin
sending their respective elements of `x` to their routers,
along the color `x_color`.
These PEs send the elements of `x` both to the `SOUTH` and back down
their own `RAMP`.
All other rows receive elements of `x` along `x_color` from the `NORTH`
and transmit them down their `RAMP`, with all but the last row also
transmitting the elements further `SOUTH`.

On all PEs, receiving a wavelet along `x_color` activates
`recv_x`. This task is a wavelet-triggered task (WTT): the wavelet’s
data is fed in as an argument to `recv_x`.

When a PE receives an element of `x` in the `recv_x` task, it increments
its local `y` tensor by computing the corresponding piece of `Ax`.
When a PE has received all corresponding elements of `x` along `x_color`,
with each PE receiving `N/kernel_x_dim` values,
it has finished computing its local contribution to `y`.

At this point, the local task `reduce` is activated.
We use two colors in a *checkerboard pattern* to accumulate the partial
`y` results from `WEST` to `EAST`.
On the even columns, we use `ax_color_1` to receive partial results
from the `WEST` and `ax_color_2` to send partial results to the `EAST`;
on the odd columns, we use `ax_color_2` to receive partial results
from the `WEST` and `ax_color_1` to send partial results to the `EAST`.
We must use this checkerboard pattern because we cannot safely send
and receive multiple wavelets on the same color with a fixed routing.
In a future example, we will demonstrate the use of dynamic switching
to update the color routing, which will allow you to use a single color.

The leftmost column of PEs has nothing to receive from the `WEST`,
so these PEs only send their partial `y` results to the `EAST`.
The remaining columns, upon receiving a partial `y` result from the `WEST`,
increment their `y` tensors by the received values,
and all but the rightmost column sends those values to the `EAST`
The values in the rightmost column contain the final result `y`,
with each PE containing `M/kernel_y_dim` elements.

Last, the host copies `y` from the right column of PEs,
and checks that the result is correct.

## layout.csl

```csl
// kernel dimensions
param kernel_x_dim: i16;
param kernel_y_dim: i16;

// total matrix dimensions
param M: i16;
param N: i16;

// Colors
const ax_color_1: color = @get_color(0); // sends/recvs partial result Ax EAST/WEST
const ax_color_2: color = @get_color(1); // sends/recvs partial result Ax EAST/WEST
const x_color:    color = @get_color(2); // sends/recvs elems x

// This example uses kernel_x_dim x kernel_y_dim PEs
const memcpy = @import_module("<memcpy/get_params>", .{
  .width = kernel_x_dim,
  .height = kernel_y_dim
});

layout {
  // PE coordinates are (column, row)
  @set_rectangle(kernel_x_dim, kernel_y_dim);

  // PE rectangle dimensions should evenly divide matrix dimensions
  @comptime_assert(M % kernel_y_dim == 0);
  @comptime_assert(N % kernel_x_dim == 0);

  const common_params = .{
    .M_per_PE = M / kernel_y_dim,
    .N_per_PE = N / kernel_x_dim,
    .kernel_x_dim = kernel_x_dim,
    .kernel_y_dim = kernel_y_dim,
    .x_color = x_color
  };

  const even_params = @concat_structs(common_params, .{
    .send_east_color = ax_color_1, .recv_west_color = ax_color_2
  });

  const odd_params = @concat_structs(common_params, .{
    .send_east_color = ax_color_2, .recv_west_color = ax_color_1
  });

  for (@range(i16, kernel_x_dim)) |pe_x| {
    for (@range(i16, kernel_y_dim)) |pe_y| {
      if (pe_x % 2 == 0) {
        @set_tile_code(pe_x, pe_y, "pe_program.csl", @concat_structs(
          .{ .memcpy_params = memcpy.get_params(pe_x) }, even_params));
      } else {
        @set_tile_code(pe_x, pe_y, "pe_program.csl", @concat_structs(
          .{ .memcpy_params = memcpy.get_params(pe_x) }, odd_params));
      }
    }
  }

  // Create route values
  const RX_R_TX_RS = .{ .rx = .{RAMP},  .tx = .{RAMP, SOUTH} };
  const RX_N_TX_RS = .{ .rx = .{NORTH}, .tx = .{RAMP, SOUTH} };
  const RX_N_TX_R  = .{ .rx = .{NORTH}, .tx = .{RAMP} };

  const RX_W_TX_R  = .{ .rx = .{WEST},  .tx = .{RAMP} };
  const RX_R_TX_E  = .{ .rx = .{RAMP},  .tx = .{EAST} };

  for (@range(i16, kernel_x_dim)) |pe_x| {
    for (@range(i16, kernel_y_dim)) |pe_y| {
      if (pe_y == 0) {
        @set_color_config(pe_x, pe_y, x_color, .{ .routes = RX_R_TX_RS });
      } else if (pe_y == kernel_y_dim-1) {
        @set_color_config(pe_x, pe_y, x_color, .{ .routes = RX_N_TX_R  });
      } else {
        @set_color_config(pe_x, pe_y, x_color, .{ .routes = RX_N_TX_RS });
      }

      if (pe_x % 2 == 0) {
        @set_color_config(pe_x, pe_y, ax_color_1, .{ .routes = RX_R_TX_E });
        @set_color_config(pe_x, pe_y, ax_color_2, .{ .routes = RX_W_TX_R });
      } else {
        @set_color_config(pe_x, pe_y, ax_color_1, .{ .routes = RX_W_TX_R });
        @set_color_config(pe_x, pe_y, ax_color_2, .{ .routes = RX_R_TX_E });
      }
    }
  }

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

// Program rectangle dimensions
param kernel_x_dim: u16;
param kernel_y_dim: u16;

// Colors
param send_east_color: color; // sends partial result Ax EAST
param recv_west_color: color; // recvs partial result Ax WEST
param x_color:         color; // sends elems x SOUTH

// Queue IDs
const send_east_oq: output_queue = @get_output_queue(2);
const recv_west_iq: input_queue  = @get_input_queue(3);
const x_color_oq:   output_queue = @get_output_queue(4);
const x_color_iq:   input_queue  = @get_input_queue(4);

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

fn is_right_col() bool {
  return (layout_mod.get_x_coord() == kernel_x_dim-1);
}

task reduce() void {

  const out_dsd = @get_dsd(fabout_dsd, .{
                    .fabric_color = send_east_color, .extent = M_per_PE,
                    .output_queue = send_east_oq
                  });

  const in_dsd  = @get_dsd(fabin_dsd, .{
                    .fabric_color = recv_west_color, .extent = M_per_PE,
                    .input_queue = recv_west_iq
                  });

  // After fmovs is done, activate exit_task to unblock cmd_stream
  if (is_left_col()) {
    @fmovs(out_dsd, y_dsd, .{ .async = true, .activate = exit_task_id });
  } else if (is_right_col()) {
    @fadds(y_dsd, y_dsd, in_dsd, .{ .async = true, .activate = exit_task_id });
  } else {
    @fadds(out_dsd, y_dsd, in_dsd, .{ .async = true, .activate = exit_task_id });
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

  // recv_x is wavelet-triggered task (WTT)
  // activated by receiving wavelets along color x_color,
  // which corresponds to recv_x_task_id
  @bind_data_task(recv_x, recv_x_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(send_east_oq, .{ .color = send_east_color });
    @initialize_queue(recv_west_iq, .{ .color = recv_west_color });
    @initialize_queue(x_color_oq,   .{ .color = x_color });
    @initialize_queue(x_color_iq,   .{ .color = x_color });
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
N = int(compile_data['params']['N']) # columns
M = int(compile_data['params']['M']) # rows

# PE grid dimensions
kernel_x_dim = int(compile_data['params']['kernel_x_dim'])
kernel_y_dim = int(compile_data['params']['kernel_y_dim'])

np.random.seed(seed=7)

# Construct A, x, b
#
# In the previous examples, we used arange to initialize
# the elements of A. This example can support any number
# of PEs, and thus the matrix A could be quite large, so
# arange is no longer suitable. Instead, we use rand() to
# initialize the elements to random numbers between 0 and
# 1, to avoid numerical issues in our calculation."
#
# The upper bound of error estimate is proportional to
# (|A|*|x|+|b|). If the magnitude of A is large, any small
# mistake could be invisible. For example, if one entry is
# not calculated correctly, it may still pass because the
# error bound is too big. So rand() is better than arange().
A = np.random.rand(M, N).astype(np.float32)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Size of N dimension on each PE
N_per_PE = N // kernel_x_dim
M_per_PE = M // kernel_y_dim

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for A, x, y on device
A_symbol = runner.get_id('A')
x_symbol = runner.get_id('x')
y_symbol = runner.get_id('y')

# Load and run the program
runner.load()
runner.run()

# Copy b into y of PEs (0, 0) to (0, kernel_y_dim-1)
runner.memcpy_h2d(y_symbol, b, 0, 0, 1, kernel_y_dim, M_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Copy chunks of A into all PEs
# Each chunk on each PE is stored column major
A_prepared = A.reshape(kernel_y_dim, M_per_PE, kernel_x_dim, N_per_PE).transpose(0, 2, 3, 1).ravel()
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, kernel_x_dim, kernel_y_dim, M_per_PE*N_per_PE,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

# Copy x into PEs (0, 0) and (kernel_x_dim-1, 0)
# PE (0, 0) gets first N/2 elements; PE (1, 0) gets last N/2 elements
runner.memcpy_h2d(x_symbol, x, 0, 0, kernel_x_dim, 1, N_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Launch the compute function on device
runner.launch('compute', nonblock=False)

# Copy y back from PEs (kernel_x_dim-1, 0) and (kernel_x_dim-1, kernel_y_dim-1)
y_result = np.zeros([M], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, kernel_x_dim-1, 0, 1, kernel_y_dim, M_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
# The formula of assert_allclose() is
#   |a-b| <= atol + rtol * |b|
# where rtol = 1.e-5, atol=1.e-8
#
# However the magnitude of y_result or y_expected is (|A|*|x| + |b|), so
# it is likely to fail the absolute tolerance (atol) when |A| is large.
#
# Using the norm-wise error is perferred for large dimension.
#   |y_result - y_expected|/(|A|*|x| + |b|) < tol
#
r = y_result - y_expected
nrm_r = np.linalg.norm(r, np.inf)
nrm_x = np.linalg.norm(x, np.inf)
nrm_b = np.linalg.norm(b, np.inf)
nrm_A = np.linalg.norm(A, np.inf)
print(f"|y_result - y_expected| = {nrm_r}")
print(f"|x| = {nrm_x}")
print(f"|b| = {nrm_b}")
print(f"|A| = {nrm_A}")

relerr = nrm_r/(nrm_A*nrm_x + nrm_b)
print(f"|y_result - y_expected|/(|A|*|x| + |b|) = {relerr}")

assert relerr < 1.e-6, "norm-wise relative error is too big, something must be wrong"

# The norm-wise estimate sometimes over-estimates too much, we can also
# check the absolute error when the matrix is small.
#
# too large. If it fails (but norm-wise estimate can pass), try
#   np.testing.assert_allclose(y_result/nrm_A, y_expected/nrm_A)
if N < 10:
  np.testing.assert_allclose(y_result, y_expected)

print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=11,5 \
--fabric-offsets=4,1 --params=kernel_x_dim:4,kernel_y_dim:3,M:6,N:8 \
-o out --memcpy --channels 1
cs_python run.py --name out

```
[previous

GEMV 7: Routes and Fabric DSDs, Part II](csl/code-examples/tutorial-gemv-07-routes-2.md "previous page")
[next

GEMV 9: Memcpy Streaming Mode](csl/code-examples/tutorial-gemv-09-streaming.md "next page")
