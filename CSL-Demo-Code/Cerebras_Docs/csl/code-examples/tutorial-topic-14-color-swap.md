---
title: "Topic 14: Color Swap"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-topic-14-color-swap"
crawled_at: "2026-02-14 23:55:40"
---

# Topic 14: Color Swap

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-topic-14-color-swap](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-14-color-swap)

This example demonstrates the color swap feature of WSE-2.
CSL currently does not support color swap on WSE-3, and support
is in development.

This program uses a row of four contiguous PEs.
Two colors, `red` (color 0) and `blue` (color 1), are used.
On all PEs, the routing associated with these colors receives
from the `WEST` and sends down the `RAMP` and `EAST`.
Additionally, for both colors, `swap_color_x` is set to `true`.
Because these colors differ only in their lowest bit, when a
`red` wavelet comes into a router from `WEST`, it leaves the
router to the `EAST` as a `blue` wavelet, and vice versa.

The host code sends four wavelets along the color `MEMCPYH2D_DATA_1`
into the first PE. The WTT of `MEMCPYH2D_DATA_1` forwards this data
to color `blue`. When a PE receives a `red` wavelet, the task
`red_task` is activated, and when a PE receives a `blue` wavelet,
the task `blue_task` is activated.

Each PE program contains a global variable named `sum`,
initialized to zero.
When a `red_task` is activated by an incoming wavelet `in_data`,
`sum` is incremented by an amount `in_data`.
When a `blue_task` is activated by an incoming wavelet `in_data`,
`sum` is incremented by an amount `2 * in_data`.

## layout.csl

```csl
// Color/ task ID map
//
//  ID var                ID var  ID var                ID var
//   0 red                 9      18                    27 reserved (memcpy)
//   1 blue               10      19                    28 reserved (memcpy)
//   2                    11      20                    29 reserved
//   3                    12      21 reserved (memcpy)  30 reserved (memcpy)
//   4                    13      22 reserved (memcpy)  31 reserved
//   5                    14      23 reserved (memcpy)  32
//   6 MEMCPY_H2D_DATA_1  15      24                    33
//   7                    16      25                    34
//   8                    17      26                    35

// ID for memcpy streaming color
param MEMCPYH2D_DATA_1_ID: i16;

// number of PEs in kernel
param width: u16;

// Colors
const MEMCPYH2D_DATA_1: color = @get_color(MEMCPYH2D_DATA_1_ID);
const red:              color = @get_color(0);
const blue:             color = @get_color(1);

const memcpy = @import_module("<memcpy/get_params>", .{
  .width = width,
  .height = 1,
  .MEMCPYH2D_1 = MEMCPYH2D_DATA_1,
});

layout {
  @set_rectangle(width, 1);

  for (@range(u16, width)) |pe_x| {

    const memcpy_params = memcpy.get_params(pe_x);

    @set_tile_code(pe_x, 0, "pe_program.csl", .{
      .memcpy_params = memcpy_params,
      .red = red,
      .blue = blue,
    });

    const start = .{ .rx = .{ RAMP }, .tx = .{ RAMP, EAST }, .color_swap_x = true };
    const routes = .{ .rx = .{ WEST }, .tx = .{ RAMP, EAST }, .color_swap_x = true };
    const end = .{ .rx = .{ WEST }, .tx = .{ RAMP }, .color_swap_x = true };

    if (pe_x == 0) {
      // 1st PE receives data from streaming H2D, then forwards it to color "red"
      // (WTT(H2D) forwards data to color "blue", not color "red")
      @set_color_config(pe_x, 0, blue, .{ .routes = start });
      @set_color_config(pe_x, 0, red, .{ .routes = start });
    } else if (pe_x == width - 1) {
      @set_color_config(pe_x, 0, blue, .{ .routes = end });
      @set_color_config(pe_x, 0, red, .{ .routes = end });
    } else {
      @set_color_config(pe_x, 0, blue, .{ .routes = routes });
      @set_color_config(pe_x, 0, red, .{ .routes = routes });
    }
  }

  // export symbol name
  @export_name("sum", [*]u32, true);
}

```

## pe\_program.csl

```csl
param memcpy_params: comptime_struct;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// Colors
param red:  color;
param blue: color;

// Queue IDs
const blue_oq: output_queue = @get_output_queue(2);

// Task IDs
// Task ID for data task that recvs from memcpy
const h2d_task_id: data_task_id = @get_data_task_id(sys_mod.MEMCPYH2D_1);

// Task ID for data task red, consumes red wlts
const red_task_id: data_task_id = @get_data_task_id(red);

// Task ID for data task blue, consumes blue wlts
const blue_task_id: data_task_id = @get_data_task_id(blue);

// Single-elem array to hold sum of received wlts
var sum = @zeros([1]u32);
var ptr_sum: [*]u32 = &sum;

// Task that will be triggered by red wavelet
task red_task(in_data : u32) void {
  sum[0] += in_data;
}

// Task that will be triggered by blue wavelet
task blue_task(in_data : u32) void {
  sum[0] += in_data * 2;
}

var buf = @zeros([1]u32);
const buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> buf[i] });

// PEs 0, 2 activate blue task; 1, 3 activate red task.
const out_dsd = @get_dsd(fabout_dsd, .{
  .extent = 1,
  .fabric_color = blue,
  .output_queue = blue_oq
});

// receive data from streaming H2D and forward it to color red
task wtt_h2d(data: u32) void {
  @block(h2d_task_id);
  buf[0] = data;
  @mov16(out_dsd, buf_dsd, .{ .async = true, .unblock = h2d_task_id });
}

comptime {
  @bind_data_task(red_task, red_task_id);
  @bind_data_task(blue_task, blue_task_id);
  @bind_data_task(wtt_h2d, h2d_task_id);

  @export_symbol(ptr_sum, "sum");
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
dirname = args.name

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
params = compile_data["params"]
MEMCPYH2D_DATA_1 = int(params["MEMCPYH2D_DATA_1_ID"])
width = int(params["width"])
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"width = {width}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sum_symbol = runner.get_id("sum")

runner.load()
runner.run()

num_elems = 5
x = np.arange(num_elems, dtype=np.uint32)

print("step 1: streaming H2D to 1st PE")
runner.memcpy_h2d(MEMCPYH2D_DATA_1, x, 0, 0, 1, 1, num_elems, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("step 2: copy mode D2H buf")
result = np.zeros(width, np.uint32)
runner.memcpy_d2h(result, sum_symbol, 0, 0, width, 1, 1, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

runner.stop()

# In order, the host streams in 0, 1, 2, 3, 4 from the West.
# Red tasks add values to running global sum on its PE.
# Blue tasks add 2*values to running global sum on its PE.
# PEs 0, 2 activate blue task; 1, 3 activate red task.
# Final value of sum var on even PEs will be: 20
# Final value of sum var on odd PEs will be: 10
oracle = np.zeros([width], dtype=np.uint32)
for i in range(width):
  oracle[i] = ((i+1) % 2 + 1) * num_elems * (num_elems-1) / 2

# Assert that all values of 'sum' are as expected
np.testing.assert_equal(result, oracle)
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse2 ./layout.csl --fabric-dims=11,3 \
--fabric-offsets=4,1 -o out  \
--params=width:4 \
--params=MEMCPYH2D_DATA_1_ID:6 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out

```
[previous

Topic 13: Simprint Library](csl/code-examples/tutorial-topic-13-simprint.md "previous page")
[next

Topic 13: WSE-3 Microthreads](csl/code-examples/tutorial-topic-15-wse3-microthreads.md "next page")
