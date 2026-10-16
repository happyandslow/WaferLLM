---
title: "Topic 12: Debug Library"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-topic-12-debug-library"
crawled_at: "2026-02-14 23:55:30"
---

# Topic 12: Debug Library

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-topic-12-debug-library](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-12-debug-library)

This example shows a program that uses the tracing mechanism of the `<debug>`
library to record variable values and compile time strings as well as
timestamps, for inspection by the host code.

The program uses a row of four contiguous PEs.
The first PE sends an array of values to three receiver PEs.
Each PE program contains a global variable named `global`, initialized to
zero.
When the data task `recv_task` on the receiver PE is activated by an incoming
wavelet `in_data`, `global` is incremented by `2 * in_data`.

The programs running on each PE import two instances of the `<debug>` library.
On the receiver PEs, each time a task activates, the instance named `trace`
logs a compile time string noting that the task has begun execution, and the
updated value of `global`.
The instance named `times` logs a timestamp at the beginning of a task, and
at the end of a task.

The host code uses the function `read_trace` from
`cerebras.sdk.debug.debug_util` to read the logged values after execution of
the device code finishes.
Note that the PE coordinates passed to `read_trace` start from the northwest
corner of the fabric, not from the northwest corner of the program rectangle.

## layout.csl

```csl
// Color map
//
//  ID var   ID var  ID var                ID var
//   0 comm   9      18                    27 reserved (memcpy)
//   1       10      19                    28 reserved (memcpy)
//   2       11      20                    29 reserved
//   3       12      21 reserved (memcpy)  30 reserved (memcpy)
//   4       13      22 reserved (memcpy)  31 reserved
//   5       14      23 reserved (memcpy)  32
//   6       15      24                    33
//   7       16      25                    34
//   8       17      26                    35

// See task maps in sender.csl and receiver.csl

param width: u16;     // number of PEs in kernel
param num_elems: u16; // number of elements in each PE's buf

// Colors
const comm: color = @get_color(0);

const memcpy = @import_module("<memcpy/get_params>", .{
  .width = width,
  .height = 1,
});

layout {
  @set_rectangle(width, 1);

  // Sender
  @set_tile_code(0, 0, "sender.csl", .{
    .memcpy_params = memcpy.get_params(0),
    .comm = comm, .num_elems = num_elems
  });

  @set_color_config(0, 0, comm, .{ .routes = .{ .rx = .{ RAMP }, .tx = .{ EAST }}});

  // Receivers
  for (@range(u16, 1, width, 1)) |pe_x| {

    @set_tile_code(pe_x, 0, "receiver.csl", .{
      .memcpy_params = memcpy.get_params(pe_x),
      .comm = comm, .num_elems = num_elems
    });

    if (pe_x == width - 1) {
      @set_color_config(pe_x, 0, comm, .{ .routes = .{ .rx = .{ WEST }, .tx = .{ RAMP }}});
    } else {
      @set_color_config(pe_x, 0, comm, .{ .routes = .{ .rx = .{ WEST }, .tx = .{ RAMP, EAST }}});
    }
  }

  // export symbol name
  @export_name("buf", [*]u32, true);
  @export_name("main_fn", fn()void);
}

```

## sender.csl

```csl
// WSE-2 task ID map
// On WSE-2, data tasks are bound to colors (IDs 0 through 24)
//
//  ID var                ID var  ID var                ID var
//   0                     9      18                    27 reserved (memcpy)
//   1                    10      19                    28 reserved (memcpy)
//   2                    11      20                    29 reserved
//   3                    12      21 reserved (memcpy)  30 reserved (memcpy)
//   4                    13      22 reserved (memcpy)  31 reserved
//   5                    14      23 reserved (memcpy)  32
//   6                    15      24                    33
//   7                    16      25                    34
//   8 exit_task_id       17      26                    35

// WSE-3 task ID map
// On WSE-3, data tasks are bound to input queues (IDs 0 through 7)
//
//  ID var                ID var  ID var                ID var
//   0 reserved (memcpy)   9      18                    27 reserved (memcpy)
//   1 reserved (memcpy)  10      19                    28 reserved (memcpy)
//   2                    11      20                    29 reserved
//   3                    12      21 reserved (memcpy)  30 reserved (memcpy)
//   4                    13      22 reserved (memcpy)  31 reserved
//   5                    14      23 reserved (memcpy)  32
//   6                    15      24                    33
//   7                    16      25                    34
//   8 exit_task_id       17      26                    35

param memcpy_params: comptime_struct;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// Number of elements to be send to receivers
param num_elems: u16;

// Colors
param comm: color;

// Queue IDs
const comm_oq: output_queue = @get_output_queue(2);

// Task IDs
const exit_task_id: local_task_id = @get_local_task_id(8);

const trace = @import_module(
  "<debug>",
  .{ .key = "trace",
     .buffer_size = 100,
   }
);
const times = @import_module(
  "<debug>",
  .{ .key = "times",
     .buffer_size = 100,
   }
);

// Host copies values to this array
// We then send the values to the receives
var buf = @zeros([num_elems]u32);
var ptr_buf: [*]u32 = &buf;

const buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{num_elems} -> buf[i] });

const out_dsd = @get_dsd(fabout_dsd, .{
  .extent = 1,
  .fabric_color = comm,
  .output_queue = comm_oq
});

fn main_fn() void {
  trace.trace_string("Sender beginning main_fn");
  times.trace_timestamp(); // Record timestamp for main_fn start
  @fmovs(out_dsd, buf_dsd, .{ .async = true, .activate = exit_task });
}

task exit_task() void {
  trace.trace_string("Sender exiting");
  times.trace_timestamp(); // Record timestamp for exit
  sys_mod.unblock_cmd_stream();
}

comptime {
  @bind_local_task(exit_task, exit_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(comm_oq, .{ .color = comm });
  }

  @export_symbol(ptr_buf, "buf");
  @export_symbol(main_fn);
}

```

## receiver.csl

```csl
// WSE-2 task ID map
// On WSE-2, data tasks are bound to colors (IDs 0 through 24)
//
//  ID var                ID var  ID var                ID var
//   0 recv_task_id        9      18                    27 reserved (memcpy)
//   1                    10      19                    28 reserved (memcpy)
//   2                    11      20                    29 reserved
//   3                    12      21 reserved (memcpy)  30 reserved (memcpy)
//   4                    13      22 reserved (memcpy)  31 reserved
//   5                    14      23 reserved (memcpy)  32
//   6                    15      24                    33
//   7                    16      25                    34
//   8                    17      26                    35

// WSE-3 task ID map
// On WSE-3, data tasks are bound to input queues (IDs 0 through 7)
//
//  ID var                ID var  ID var                ID var
//   0 reserved (memcpy)   9      18                    27 reserved (memcpy)
//   1 reserved (memcpy)  10      19                    28 reserved (memcpy)
//   2 recv_task_id       11      20                    29 reserved
//   3                    12      21 reserved (memcpy)  30 reserved (memcpy)
//   4                    13      22 reserved (memcpy)  31 reserved
//   5                    14      23 reserved (memcpy)  32
//   6                    15      24                    33
//   7                    16      25                    34
//   8                    17      26                    35

param memcpy_params: comptime_struct;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// Number of elements expected from sender
param num_elems: u16;

// Colors
param comm: color;

// Queue IDs
const comm_iq: input_queue = @get_input_queue(2);
const comm_oq: output_queue = @get_output_queue(2);

// Task ID for recv_task, consumed wlts with color comm
// On WSE-2, data task IDs are created from colors; on WSE-3, from input queues
// Task ID for data task that recvs from memcpy
const recv_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(comm)
  else if (@is_arch("wse3")) @get_data_task_id(comm_iq);

// Import two instances of <debug>:
// `trace` records comptime string and value of 'global'
// `times` records timestamps at begin and end of tasks
const trace = @import_module(
  "<debug>",
  .{ .key = "trace",
     .buffer_size = 100,
   }
);
const times = @import_module(
  "<debug>",
  .{ .key = "times",
     .buffer_size = 100,
   }
);

// Variable whose value we update in recv_task
var global : u32 = 0;

// Array to store received values
var buf = @zeros([num_elems]u32);
var ptr_buf: [*]u32 = &buf;

// main_fn does nothing on the senders
fn main_fn() void {}

// Track number of wavelets received by recv_task
var num_wlts_recvd: u16 = 0;

task recv_task(in_data : u32) void {

  times.trace_timestamp(); // Record timestamp for task start in `times`
  trace.trace_string("Start recv_task"); // Record string in `trace`

  buf[num_wlts_recvd] = in_data; // Store recvd value in buf
  global += 2*in_data; // Increment global by 2x received value

  trace.trace_u32(global); // Record updated value of global in `trace`

  num_wlts_recvd += 1; // Increment number of received wavelets
  // Once we have received all wavelets, we unblock cmd stream
  if (num_wlts_recvd == num_elems) {
    sys_mod.unblock_cmd_stream();
  }

  times.trace_timestamp(); // Record timestamp for task end in `times`
}

comptime {
  @bind_data_task(recv_task, recv_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(comm_iq, .{ .color = comm });
    @initialize_queue(comm_oq, .{ .color = comm });
  }

  @export_symbol(ptr_buf, "buf");
  @export_symbol(main_fn);
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.debug.debug_util import debug_util
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
num_elems = int(params["num_elems"])
width = int(params["width"])
print(f"width = {width}")
print(f"num_elems = {num_elems}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_buf = runner.get_id("buf")

runner.load()
runner.run()

x = np.arange(num_elems, dtype=np.uint32)

print("step 1: H2D copy buf to sender PE")
runner.memcpy_h2d(sym_buf, x, 0, 0, 1, 1, num_elems, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

print("step 2: launch main_fn")
runner.launch('main_fn', nonblock=False)

print("step 3: D2H copy buf back from all PEs")
out_buf = np.arange(width*num_elems, dtype=np.uint32)
runner.memcpy_d2h(out_buf, sym_buf, 0, 0, width, 1, num_elems, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

runner.stop()

debug_mod = debug_util(dirname, cmaddr=args.cmaddr)
core_offset_x = 4
core_offset_y = 1
print(f"=== dump core: core rectangle starts at {core_offset_x}, {core_offset_y}")

result = np.zeros([width, num_elems])
for idx in range(width):
  # Get traces recorded in 'trace'
  trace_output = debug_mod.read_trace(core_offset_x + idx, core_offset_y, 'trace')

  # On receiver PEs, record value of 'global'
  if idx > 0:
    result[idx, :] = trace_output[1::2]

  # Get timestamp traces recorded in 'times'
  timestamp_output = debug_mod.read_trace(core_offset_x + idx, core_offset_y, 'times')

  # Print out all traces for PE
  print("PE (", idx, ", 0): ")
  print("Trace: ", trace_output)
  print("Times: ", timestamp_output)
  print()

# Receiver PEs adds 2*value to running global sum on its PE.
# Value of global var is recorded after each update.
# Trace values of global var will be: 0, 2, 6, 12, 20
oracle = np.zeros([width, num_elems])
for i in range(1, width, 1):
  for j in range(num_elems):
    oracle[i, j] = 2 * j * (j+1) / 2

# Assert that all trace values of 'global' are as expected
np.testing.assert_equal(result, oracle)
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=11,3 \
--fabric-offsets=4,1 --params=width:4,num_elems:5 -o out  \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out

```
[previous

Topic 11: Collective Communications](csl/code-examples/tutorial-topic-11-collectives.md "previous page")
[next

Topic 13: Simprint Library](csl/code-examples/tutorial-topic-13-simprint.md "next page")
