---
title: "Pipeline 3: Add an artificial halo"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-pipeline-03-multiple"
crawled_at: "2026-02-14 23:55:34"
---

# Pipeline 3: Add an artificial halo

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-pipeline-03-multiple](https://sdk.cerebras.net/csl/code-examples/tutorial-pipeline-03-multiple)

The disadvantage of FIFO in the previous example is the resource consumption.
The FIFO requires two microthreads and a scratch buffer.

The simple workaround is to move such FIFO outside the kernel. We add another
halo, which we call an artificial halo, around the kernel (`pe_program.csl`).
The west side is `west.csl` and east side is `east.csl`.
The `west.csl` implements a FIFO to receive the data from H2D.
The `east.csl` implements a FIFO to receive the data from `pe_program.csl`
and redirect it to D2H.

There is no more FIFO in `pe_program.csl`. Instead, we replace the colors
`MEMCPYH2D_DATA_1` by `Cin` and `MEMCPYD2H_DATA_1` by `Cout`.
The color `Cin` receives data from the west to the ramp.
The color `Cout` sends the data from ramp to the east.

This example has the same property as `pipeline-02-fifo`: as long as the
parameter `size` does not exceed the capacity of the FIFO in `west.csl`,
H2D can always finish so the `@add16` can progress.

## layout.csl

```csl
// Color/ task ID map
//
//  ID var           ID var      ID var                ID var
//   0 MEMCPYH2D_1    9 STARTUP  18                    27 reserved (memcpy)
//   1 MEMCPYD2H_1   10          19                    28 reserved (memcpy)
//   2               11          20                    29 reserved
//   3               12          21 reserved (memcpy)  30 reserved (memcpy)
//   4               13          22 reserved (memcpy)  31 reserved
//   5               14          23 reserved (memcpy)  32
//   6 Cin           15          24                    33
//   7 Cout          16          25                    34
//   8 main_task_id  17          26                    35

// Number of elements sent through core program rectangle
param size: i16;

param MEMCPYH2D_DATA_1_ID: i16;
param MEMCPYD2H_DATA_1_ID: i16;

const MEMCPYH2D_DATA_1: color = @get_color(MEMCPYH2D_DATA_1_ID);
const MEMCPYD2H_DATA_1: color = @get_color(MEMCPYD2H_DATA_1_ID);

// Colors used for input/ output of core program rectangle
const Cin: color = @get_color(6);
const Cout: color = @get_color(7);

const memcpy = @import_module("<memcpy/get_params>", .{
    .width = 3,
    .height = 1,
    .MEMCPYH2D_1 = MEMCPYH2D_DATA_1,
    .MEMCPYD2H_1 = MEMCPYD2H_DATA_1
    });

layout {
  @set_rectangle(3, 1);

  // west.csl has a H2D
  @set_tile_code(0, 0, "memcpy_edge/west.csl", .{
    .USER_IN_1 = Cin,
    .memcpy_params = memcpy.get_params(0)
  });

  @set_tile_code(1, 0, "pe_program.csl", .{
    .size = size,
    .Cin = Cin,
    .Cout = Cout,
    .memcpy_params = memcpy.get_params(1)
  });

  // east.csl only hase a D2H
  @set_tile_code(2, 0, "memcpy_edge/east.csl", .{
    .USER_OUT_1 = Cout,
    .memcpy_params = memcpy.get_params(2)
  });
}

```

## pe\_program.csl

```csl
param memcpy_params: comptime_struct;

param size: i16;

param Cin: color;
param Cout: color;

// Queue IDs
const Cin_iq: input_queue = @get_input_queue(2);
const Cout_oq: output_queue = @get_output_queue(3);

// Task IDs
const main_task_id: local_task_id = @get_local_task_id(8);

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

const Cin_in_dsd = @get_dsd(fabin_dsd, .{
  .extent = size,
  .fabric_color = Cin,
  .input_queue = Cin_iq,
});

const Cout_out_dsd = @get_dsd(fabout_dsd, .{
  .extent = size,
  .fabric_color = Cout,
  .output_queue = Cout_oq,
});

const buf = [1]i16{ 1 };
const one_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{size} -> buf[0] });

task main_task() void {
  @add16(Cout_out_dsd, Cin_in_dsd, one_dsd, .{ .async = true });
}

comptime {
  // activate local task main_task at startup
  @bind_local_task(main_task, main_task_id);
  @activate(main_task_id);

  @set_local_color_config(Cin, .{ .routes = .{ .rx = .{ WEST }, .tx = .{ RAMP }}});
  @set_local_color_config(Cout, .{ .routes = .{ .rx = .{ RAMP }, .tx = .{ EAST }}});

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(Cin_iq, .{ .color = Cin });
    @initialize_queue(Cout_oq, .{ .color = Cout });
  }
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.sdk_utils import memcpy_view, input_array_to_u32
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
MEMCPYD2H_DATA_1 = int(params["MEMCPYD2H_DATA_1_ID"])
# Size of the input and output tensors; use this value when compiling the
# program, e.g. `cslc --params=size:12 --fabric-dims=8,3 --fabric-offsets=4,1`
size = int(params["size"])
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")
print(f"size = {size}")

memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

runner.load()
runner.run()

# Generate a random input tensor of the desired size
input_tensor = np.random.randint(256, size=size, dtype=np.int16)

print("step 1: streaming H2D to P0.0")
# The type of input_tensor is int16, so we need to extend to 32-bit for memcpy H2D
input_tensor_u32 = input_array_to_u32(input_tensor, 1, size)
runner.memcpy_h2d(MEMCPYH2D_DATA_1, input_tensor_u32, 0, 0, 1, 1, size, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("step 2: streaming D2H at P2.0")
# The memcpy D2H buffer must be 32-bit
output_tensor_u32 = np.zeros(size, np.uint32)
runner.memcpy_d2h(output_tensor_u32, MEMCPYD2H_DATA_1, 2, 0, 1, 1, size, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
# remove upper 16-bit of each u32
result_tensor = memcpy_view(output_tensor_u32, np.dtype(np.int16))

runner.stop()

np.testing.assert_equal(result_tensor, input_tensor + 1)
print("SUCCESS!")

```

## memcpy\_edge/memcpy\_edge.csl

```csl
// This is a template of memcpy over the edges.
// memcpy_edge.csl can be "north", "south", "west" or "east"
// of the following layout.
//        +---------+
//        |  north  |
// +------+---------+------+
// | west |  core   | east |
// +------+---------+------+
//        |  south  |
//        +---------+
// north.csl, south.csl, west.csl and east.csl instantiate
// memcpy_edge.csl with a proper direction.
//
// memcpy_edge.csl supports 2 streaming H2Ds and one
// streaming D2H. Such constraint depends on the design.
// The current implementation binds a FIFO for a H2D or D2H,
// so we can only support 3 in total.
// We choose 2 H2Ds and 1 D2H.
// if we replace FIFO by WTT, we could support more.
//
// However the user can instantiate memcpy_edge.csl for each
// edge. The maximum number of H2Ds is 2*4 = 8 and maximum
// number of D2Hs is 1*4 = 4.
//
// If the user only has a H2D at north, for example, he only
// needs to configure color USER_IN_1, i.e. only a single
// streaming H2D is used.
//
// For example,
//   @set_tile_code(pe_x, 0, "north.csl", .{
//     .USER_IN_1 = mainColor,
//     .memcpy_params = memcpy_params,
//     .MEMCPYH2D_DATA_1 = MEMCPYH2D_DATA_1,
//     .MEMCPYD2H_DATA_1 = MEMCPYD2H_DATA_1
//   });

// send data to the "core"
param USER_IN_1 = {};
param USER_IN_2 = {};

// receive data from the "core"
param USER_OUT_1 = {};

// ----------
// Every PE needs to import memcpy module otherwise the I/O cannot
// propagate the data to the destination.

param memcpy_params: comptime_struct;

// The direction of "core", for example
// north.csl has dir = SOUTH
// south.csl has dir = NORTH
// west.csl has dir = EAST
// east.csl has dir = WEST
param dir: direction;

// entrypoint
const STARTUP: local_task_id = @get_local_task_id(9);

// On WSE-2, memcpy module reserves input and output queue 0
// On WSE-3, memcpy module reserves queues 0 and 1
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
// ----------

const h2d_mod = @import_module("h2d.csl", .{
  .USER_IN_1 = USER_IN_1,
  .USER_IN_2 = USER_IN_2,
  .MEMCPYH2D_1 = memcpy_params.MEMCPYH2D_1,
  .MEMCPYH2D_2 = memcpy_params.MEMCPYH2D_2,
  .txdir = dir
});

const d2h_mod = @import_module("d2h.csl", .{
  .USER_OUT_1 = USER_OUT_1,
  .MEMCPYD2H_1 = memcpy_params.MEMCPYD2H_1,
  .rxdir = dir
});

task f_startup() void {
  h2d_mod.f_startup();
  d2h_mod.f_startup();
}

comptime {
  @bind_local_task(f_startup, STARTUP);
  @activate(STARTUP);
}

```

## memcpy\_edge/h2d.csl

```csl
// Two streaming H2Ds:
// 1st H2D: UT 2 and UT 5
// 2nd H2D: UT 3 and UT 6

param MEMCPYH2D_1 = {};
param MEMCPYH2D_2 = {};

// Color along which we send a wavelet to pe_program
param USER_IN_1 = {};
param USER_IN_2 = {};

param txdir: direction;

// Queue IDs
const h2d_1_iq: input_queue = @get_input_queue(5);
const h2d_2_iq: input_queue = @get_input_queue(6);
const USER_IN_1_oq: output_queue = @get_output_queue(2);
const USER_IN_2_oq: output_queue = @get_output_queue(3);

const max_fifo_len = 256*20; // maximum length of the fifo

var fifo1_buffer = @zeros([max_fifo_len]u32);
const fifo1 = @allocate_fifo(fifo1_buffer);

var fifo2_buffer = @zeros([max_fifo_len]u32);
const fifo2 = @allocate_fifo(fifo2_buffer);

const INFINITE_DSD_LEN: u16 = 0x7fff;

var fab_recv_wdsd_1 = @get_dsd(fabin_dsd, .{
  .extent = INFINITE_DSD_LEN,
  .fabric_color = MEMCPYH2D_1,
  .input_queue = h2d_1_iq,
});

var fab_trans_wdsd_1 = @get_dsd(fabout_dsd, .{
  .extent = INFINITE_DSD_LEN,
  .fabric_color = USER_IN_1,
  .output_queue = USER_IN_1_oq,
});

var fab_recv_wdsd_2 = @get_dsd(fabin_dsd, .{
   .extent = INFINITE_DSD_LEN,
   .fabric_color = MEMCPYH2D_2,
   .input_queue = h2d_2_iq,
});

var fab_trans_wdsd_2 = @get_dsd(fabout_dsd, .{
    .extent = INFINITE_DSD_LEN,
    .fabric_color = USER_IN_2,
    .output_queue = USER_IN_2_oq,
});

// if no user's color is defined, f_startup() is empty
fn f_startup() void {
  if (!@is_same_type(@type_of(MEMCPYH2D_1), void) and !@is_same_type(@type_of(USER_IN_1), void)) {
    // receive data from streaming H2D
    @mov32(fifo1, fab_recv_wdsd_1, .{ .async = true });

    // forward data to USER_IN_1
    @mov32(fab_trans_wdsd_1, fifo1, .{ .async = true });
  }

  if (!@is_same_type(@type_of(MEMCPYH2D_2), void) and !@is_same_type(@type_of(USER_IN_2), void)) {
    // receive data from streaming H2D
    @mov32(fifo2, fab_recv_wdsd_2, .{ .async = true });

    // forward data to USER_IN_1
    @mov32(fab_trans_wdsd_2, fifo2, .{ .async = true });
  }
}

comptime {
  if (!@is_same_type(@type_of(USER_IN_1), void)) {
    @set_local_color_config(USER_IN_1, .{ .routes = .{ .rx = .{ RAMP }, .tx = .{ txdir }}});
  }

  if (!@is_same_type(@type_of(USER_IN_2), void)) {
    @set_local_color_config(USER_IN_2, .{ .routes = .{ .rx = .{ RAMP }, .tx = .{ txdir }}});
  }

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    if (!@is_same_type(@type_of(USER_IN_1), void)) {
      @initialize_queue(h2d_1_iq, .{ .color = MEMCPYH2D_1 });
      @initialize_queue(USER_IN_1_oq, .{ .color = USER_IN_1 });
    }

    if (!@is_same_type(@type_of(USER_IN_2), void)) {
      @initialize_queue(h2d_2_iq, .{ .color = MEMCPYH2D_2 });
      @initialize_queue(USER_IN_2_oq, .{ .color = USER_IN_2 });
    }
  }
}

```

## memcpy\_edge/d2h.csl

```csl
// One streaming D2H:
// 1st D2H: UT 4 and UT 7

param MEMCPYD2H_1 = {};

// Color along which we expect a wavelet
param USER_OUT_1 = {};

param rxdir: direction;

// Queue IDs
const USER_OUT_1_iq: input_queue = @get_input_queue(7);
const d2h_oq: output_queue = @get_output_queue(4);

const max_fifo_len = 256*40; // maximum length of the fifo

var fifo1_buffer = @zeros([max_fifo_len]u32);
const fifo1 = @allocate_fifo(fifo1_buffer);

const INFINITE_DSD_LEN: u16 = 0x7fff;

var fab_recv_wdsd = @get_dsd(fabin_dsd, .{
  .extent = INFINITE_DSD_LEN,
  .fabric_color = USER_OUT_1,
  .input_queue = USER_OUT_1_iq
});

var fab_trans_wdsd = @get_dsd(fabout_dsd, .{
  .extent = INFINITE_DSD_LEN,
  .fabric_color = MEMCPYD2H_1,
  .output_queue = d2h_oq
});

// if USER_OUT_1 is not valid, f_startup() is empty
fn f_startup() void {
  if (!@is_same_type(@type_of(MEMCPYD2H_1), void) and !@is_same_type(@type_of(USER_OUT_1), void)) {
    // receive data from USER_OUT_1
    @mov32(fifo1, fab_recv_wdsd, .{ .async = true });

    // forward data to MEMCPYD2H_1
    @mov32(fab_trans_wdsd, fifo1, .{ .async = true });
  }
}

comptime {
  if (!@is_same_type(@type_of(USER_OUT_1), void)) {
    @set_local_color_config(USER_OUT_1, .{ .routes = .{ .rx = .{ rxdir }, .tx = .{ RAMP }}});

    // On WSE-3, we must explicitly initialize input and output queues
    if (@is_arch("wse3")) {
      @initialize_queue(d2h_oq, .{ .color = MEMCPYD2H_1 });
      @initialize_queue(USER_OUT_1_iq, .{ .color = USER_OUT_1 });
    }
  }
}

```

## memcpy\_edge/east.csl

```csl
// send data to the "core"
param USER_IN_1 = {};
param USER_IN_2 = {};

// receive data from the "core"
param USER_OUT_1 = {};

param memcpy_params: comptime_struct;

const edge_mod = @import_module("memcpy_edge.csl", .{
  .memcpy_params = memcpy_params,
  .USER_IN_1 = USER_IN_1,
  .USER_IN_2 = USER_IN_2,
  .USER_OUT_1 = USER_OUT_1,
  .dir = WEST
});

```

## memcpy\_edge/west.csl

```csl
// send data to the "core"
param USER_IN_1 = {};
param USER_IN_2 = {};

// receive data from the "core"
param USER_OUT_1 = {};

param memcpy_params: comptime_struct;

const edge_mod = @import_module("memcpy_edge.csl", .{
  .memcpy_params = memcpy_params,
  .USER_IN_1 = USER_IN_1,
  .USER_IN_2 = USER_IN_2,
  .USER_OUT_1 = USER_OUT_1,
  .dir = EAST
});

```

## memcpy\_edge/north.csl

```csl
// send data to the "core"
param USER_IN_1 = {};
param USER_IN_2 = {};

// receive data from the "core"
param USER_OUT_1 = {};

param memcpy_params: comptime_struct;

const edge_mod = @import_module("memcpy_edge.csl", .{
  .memcpy_params = memcpy_params,
  .USER_IN_1 = USER_IN_1,
  .USER_IN_2 = USER_IN_2,
  .USER_OUT_1 = USER_OUT_1,
  .dir = SOUTH
});

```

## memcpy\_edge/south.csl

```csl
// send data to the "core"
param USER_IN_1 = {};
param USER_IN_2 = {};

// receive data from the "core"
param USER_OUT_1 = {};

param memcpy_params: comptime_struct;

const edge_mod = @import_module("memcpy_edge.csl", .{
  .memcpy_params = memcpy_params,
  .USER_IN_1 = USER_IN_1,
  .USER_IN_2 = USER_IN_2,
  .USER_OUT_1 = USER_OUT_1,
  .dir = NORTH
});

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=10,3 \
--fabric-offsets=4,1 --params=size:32 -o out \
--params=MEMCPYH2D_DATA_1_ID:0 \
--params=MEMCPYD2H_DATA_1_ID:1 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out

```
[previous

Pipeline 2: Attach a FIFO to H2D](csl/code-examples/tutorial-pipeline-02-fifo.md "previous page")
[next

SdkLayout 1: Introduction](csl/code-examples/tutorial-sdklayout-01-introduction.md "next page")
