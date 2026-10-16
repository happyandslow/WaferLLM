---
title: "GEMV 9: Memcpy Streaming Mode"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-09-streaming"
crawled_at: "2026-02-14 23:55:15"
---

# GEMV 9: Memcpy Streaming Mode

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-09-streaming](https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-09-streaming)

We present an alternative version of the previous example,
in which we use the `streaming` mode of `memcpy` to stream `x` and `b`
onto the device, and stream `y` off of the device.
All of the previous examples used the `copy` mode of `memcpy`.
This example is meant to simply present the basics of `streaming` mode,
and future tutorials will demonstrate some use cases for this mode.

The host code no longer includes an explicit kernel launch.
Instead, computation is started by the wavelet-triggered tasks that receive
elements of `x` and `b` along the top row and left column of PEs,
respectively.
We finish computation when the kernel streams back the result `y`
to the host.

The colors `MEMCPYH2D_DATA_1` and `MEMCPYH2D_DATA_2` are used
to stream `x` and `b` onto the device, respectively,
while `MEMCPYD2H_DATA_1` is used to stream `y` off the device.

Note that, because `memcpy` commands are serialized, the order of these
`streaming` mode `memcpy_h2d` calls in this example is important.
If the `b` values were streamed in before `x`, the program would hang.

## layout.csl

```csl
// kernel dimensions
param kernel_x_dim: i16;
param kernel_y_dim: i16;

// total matrix dimensions
param M: i16;
param N: i16;

// IDs for memcpy streaming colors
param MEMCPYH2D_DATA_1_ID: i16; // streams x from host to top row
param MEMCPYH2D_DATA_2_ID: i16; // streams b from host to left column
param MEMCPYD2H_DATA_1_ID: i16; // streams y from right column to host

// Colors
const ax_color_1:       color = @get_color(3); // sends/recvs partial result Ax EAST
const ax_color_2:       color = @get_color(4); // sends/recvs partial result Ax EAST
const x_color:          color = @get_color(5); // sends/recvs elems x
const MEMCPYH2D_DATA_1: color = @get_color(MEMCPYH2D_DATA_1_ID);
const MEMCPYH2D_DATA_2: color = @get_color(MEMCPYH2D_DATA_2_ID);
const MEMCPYD2H_DATA_1: color = @get_color(MEMCPYD2H_DATA_1_ID);

// Task IDs

// This example uses kernel_x_dim x kernel_y_dim PEs
const memcpy = @import_module("<memcpy/get_params>", .{
  .width = kernel_x_dim,
  .height = kernel_y_dim,
  .MEMCPYH2D_1 = MEMCPYH2D_DATA_1,
  .MEMCPYH2D_2 = MEMCPYH2D_DATA_2,
  .MEMCPYD2H_1 = MEMCPYD2H_DATA_1
});

layout {
  // PE coordinates are (column, row)
  @set_rectangle(kernel_x_dim, kernel_y_dim);

  const common_params = .{
    .M_per_PE = M / kernel_y_dim,
    .N_per_PE = N / kernel_x_dim,
    .kernel_x_dim = kernel_x_dim,
    .kernel_y_dim = kernel_y_dim,
    .x_color = x_color,
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

  const RX_R_TX_R  = .{ .rx = .{RAMP},  .tx = .{RAMP} };
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

      if (pe_x == 0) {
        @set_color_config(pe_x, pe_y, ax_color_1, .{ .routes = RX_R_TX_E });
        @set_color_config(pe_x, pe_y, ax_color_2, .{ .routes = RX_R_TX_R });
      } else if (pe_x % 2 == 0) {
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
}

```

## pe\_program.csl

```csl
param memcpy_params: comptime_struct;

// memcpy module provides infrastructure for copying data
// and launching functions from the host
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// layout module provides PE coordinates at runtime
const layout_mod = @import_module("<layout>");

// Matrix dimensions
param M_per_PE: i16;
param N_per_PE: i16;

// Program rectangle dimensions
param kernel_x_dim: u16;
param kernel_y_dim: u16;

// Colors
param send_east_color:  color; // sends partial result Ax EAST
param recv_west_color:  color; // recvs partial result Ax WEST
param x_color:          color; // sends elems x SOUTH

// Queue IDs
// These input queues are bound to tasks for WSE-3,
// so they do not have associated microthreaded operations
const h2d_x_iq:  input_queue  = @get_input_queue(2); // bound to memcpy_recv_x
const h2d_b_iq:  input_queue  = @get_input_queue(3); // bound to memcpy_recv_b
const recv_x_iq: input_queue  = @get_input_queue(4); // bound to recv_x

// These queues are used in microthreaded operations on both WSE-2 and WSE-3
const x_oq:      output_queue = @get_output_queue(2); // outputs to x_color
const recv_w_oq: output_queue = @get_output_queue(3); // outputs to recv_west_color
const d2h_oq:    output_queue = @get_output_queue(4); // outputs to MEMCPYD2H_1
const send_e_oq: output_queue = @get_output_queue(5); // outputs to send_east_color
const recv_w_iq: input_queue  = @get_input_queue(6);  // input from recv_west_color

// Task IDs
// On WSE-2, data task IDs are created from colors; on WSE-3, from input queues
// Task ID for data task that recvs x from memcpy
const memcpy_recv_x_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(sys_mod.MEMCPYH2D_1)
  else if (@is_arch("wse3")) @get_data_task_id(h2d_x_iq);

// Task ID for data task that recvs b from memcpy
const memcpy_recv_b_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(sys_mod.MEMCPYH2D_2)
  else if (@is_arch("wse3")) @get_data_task_id(h2d_b_iq);

// Task ID for data task recv_x, consumes x_color wlts
const recv_x_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(x_color)
  else if (@is_arch("wse3")) @get_data_task_id(recv_x_iq);

// Task ID for local task reduce_task
const reduce_task_id: local_task_id = @get_local_task_id(10);

// 48 kB of global memory contain A, x, b, y
var A: [M_per_PE*N_per_PE]f32; // A is stored column major
var y: [M_per_PE]f32;

// DSDs for accessing A, x, y
// A_dsd accesses column of A
var A_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &A, .extent = M_per_PE });
var y_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = M_per_PE });

// ptr to A will be advertised as symbol to host
var A_ptr: [*]f32 = &A;

fn is_right_col() bool {
  return (layout_mod.get_x_coord() == kernel_x_dim-1);
}

task reduce() void {

  const in_dsd  = @get_dsd(fabin_dsd, .{
                    .fabric_color = recv_west_color, .extent = M_per_PE,
                    .input_queue = recv_w_iq
                  });

  if (is_right_col()) {
    const out_dsd = @get_dsd(fabout_dsd, .{
                      .fabric_color = sys_mod.MEMCPYD2H_1, .extent = M_per_PE,
                      .output_queue = d2h_oq
                    });
    @fadds(out_dsd, y_dsd, in_dsd, .{ .async = true });

  } else {
    const out_dsd = @get_dsd(fabout_dsd, .{
                      .fabric_color = send_east_color, .extent = M_per_PE,
                      .output_queue = send_e_oq
                    });
    @fadds(out_dsd, y_dsd, in_dsd, .{ .async = true });
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

// buf stores an element in memory to be used by a microthreaded operation
var buf = @zeros([1]f32);
const mem_buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> buf[i] });

const memcpy_x_dsd = @get_dsd(fabout_dsd, .{
                       .fabric_color = x_color, .extent = 1,
                       .output_queue = x_oq
                     });

// 1st row receives x from MEMCPYH2D_DATA_1, then
// forwards data to the whole column, including itself, via color "x_color"
task memcpy_recv_x(data : f32) void {
  @block(memcpy_recv_x_task_id);
  buf[0] = data;
  @fmovs(memcpy_x_dsd, mem_buf_dsd, .{.async = true, .unblock = memcpy_recv_x_task_id });
}

const memcpy_b_dsd = @get_dsd(fabout_dsd, .{
                       .fabric_color = recv_west_color, .extent = 1,
                       .output_queue = recv_w_oq
                     });

// 1st column receives b from MEMCPYH2D_DATA_2, then
// forwards data to itself via color "recv_west_color"
task memcpy_recv_b(data : f32) void {
  @block(memcpy_recv_b_task_id);
  buf[0] = data;
  @fmovs(memcpy_b_dsd, mem_buf_dsd, .{.async = true, .unblock = memcpy_recv_b_task_id });
}

comptime {
  // These WTTs are activated by receiving wavelets streamed from host
  @bind_data_task(memcpy_recv_x, memcpy_recv_x_task_id);
  @bind_data_task(memcpy_recv_b, memcpy_recv_b_task_id);

  // recv_x is WTT activated by receiving wavelets along color x_color,
  // which corresponds to recv_x_task_id
  @bind_data_task(recv_x, recv_x_task_id);

  // reduce is local task activated by ID reduce_task_ID
  @bind_local_task(reduce, reduce_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    // These input queues are bound to tasks for WSE-3
    @initialize_queue(h2d_x_iq,  .{ .color = sys_mod.MEMCPYH2D_1 });
    @initialize_queue(h2d_b_iq,  .{ .color = sys_mod.MEMCPYH2D_2 });
    @initialize_queue(recv_x_iq, .{ .color = x_color });
    // These queues are used in microthreaded ops on WSE-2 and WSE-3
    @initialize_queue(x_oq,      .{ .color = x_color });
    @initialize_queue(recv_w_oq, .{ .color = recv_west_color });
    @initialize_queue(d2h_oq,    .{ .color = sys_mod.MEMCPYD2H_1 });
    @initialize_queue(send_e_oq, .{ .color = send_east_color });
    @initialize_queue(recv_w_iq, .{ .color = recv_west_color });
  }

  @export_symbol(A_ptr, "A");
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

# Colors used for memcpy streaming
MEMCPYH2D_DATA_1 = int(compile_data['params']['MEMCPYH2D_DATA_1_ID'])
MEMCPYH2D_DATA_2 = int(compile_data['params']['MEMCPYH2D_DATA_2_ID'])
MEMCPYD2H_DATA_1 = int(compile_data['params']['MEMCPYD2H_DATA_1_ID'])

# Construct A, x, b
A = np.arange(M*N, dtype=np.float32).reshape(M,N)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Size of N dimension on each PE
N_per_PE = N // kernel_x_dim
M_per_PE = M // kernel_y_dim

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbol for A on device
A_symbol = runner.get_id('A')

# Load and run the program
runner.load()
runner.run()

# Copy chunks of A into all PEs
# Each chunk on each PE is stored column major
A_prepared = A.reshape(kernel_y_dim, M_per_PE, kernel_x_dim, N_per_PE).transpose(0, 2, 3, 1).ravel()
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, kernel_x_dim, kernel_y_dim, M_per_PE*N_per_PE,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

# Stream x into PEs (0, 0) and (kernel_x_dim-1, 0)
# PE (0, 0) gets first N/2 elements; PE (1, 0) gets last N/2 elements
runner.memcpy_h2d(MEMCPYH2D_DATA_1, x, 0, 0, kernel_x_dim, 1, N_per_PE, streaming=True,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stream b into PEs (0, 0) to (0, kernel_y_dim-1)
runner.memcpy_h2d(MEMCPYH2D_DATA_2, b, 0, 0, 1, kernel_y_dim, M_per_PE, streaming=True,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stream y back from PEs (kernel_x_dim-1, 0) and (kernel_x_dim-1, kernel_y_dim-1)
y_result = np.zeros([M], dtype=np.float32)
runner.memcpy_d2h(y_result, MEMCPYD2H_DATA_1, kernel_x_dim-1, 0, 1, kernel_y_dim, M_per_PE,
  streaming=True, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

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

cslc --arch=wse3 ./layout.csl --fabric-dims=11,5 \
--fabric-offsets=4,1 --params=kernel_x_dim:4,kernel_y_dim:3,M:6,N:8 \
--params=MEMCPYH2D_DATA_1_ID:0 \
--params=MEMCPYH2D_DATA_2_ID:1 \
--params=MEMCPYD2H_DATA_1_ID:2 \
-o out --memcpy --channels 1
cs_python run.py --name out

```
[previous

GEMV 8: Routes and Fabric DSDs, Part III](csl/code-examples/tutorial-gemv-08-routes-3.md "previous page")
[next

Topic 1: Arrays and Pointers](csl/code-examples/tutorial-topic-01-arrays-and-pointers.md "next page")
