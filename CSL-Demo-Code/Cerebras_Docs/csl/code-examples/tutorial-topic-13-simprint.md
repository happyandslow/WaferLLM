---
title: "Topic 13: Simprint Library"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-topic-13-simprint"
crawled_at: "2026-02-14 23:55:32"
---

# Topic 13: Simprint Library

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-topic-13-simprint](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-13-simprint)

When running with the simulator, you can also print values directly to the
simulator logs (`sim.log`).
This example modifies the previous example to show the use of the
`<simprint>` library for printing comptime strings and values to the
simulator log.

Just like the previous example, this program uses a row of four contiguous PEs.
The first PE sends an array of values to three receiver PEs.
Each PE program contains a global variable named `global`, initialized to
zero.
When the data task `recv_task` on the receiver PE is activated by an incoming
wavelet `in_data`, `global` is incremented by an amount `2 * in_data`.

On the receiver PEs, each time a task activates, the program writes to
`sim.log` a string denoting that the task has started, along with the value
of the wavelet received, and the updated value of `global`.
The program also defines a helper function `simprint_pe_coords` to print out
the coordinates of the PE to the simulator log.
The output is flushed to `sim.log` whenever a newline is encountered, so you
must explicitly print `"\n"` to flush the output.

After running this example, open up `sim.log` to see the output.
The output from `<simprint>` should look something like this:

```default
@968 PE(0,0): sender beginning main_fn
@996 PE(0,0): sender exiting
@1156 PE(1,0): recv_task: in_data = 0, global = 0
@1158 PE(2,0): recv_task: in_data = 0, global = 0
@1160 PE(3,0): recv_task: in_data = 0, global = 0
@1338 PE(1,0): recv_task: in_data = 1, global = 2
@1340 PE(2,0): recv_task: in_data = 1, global = 2
@1342 PE(3,0): recv_task: in_data = 1, global = 2
@1520 PE(1,0): recv_task: in_data = 2, global = 6
@1522 PE(2,0): recv_task: in_data = 2, global = 6
@1524 PE(3,0): recv_task: in_data = 2, global = 6
@1702 PE(1,0): recv_task: in_data = 3, global = 12
@1704 PE(2,0): recv_task: in_data = 3, global = 12
@1706 PE(3,0): recv_task: in_data = 3, global = 12
@1884 PE(1,0): recv_task: in_data = 4, global = 20
@1886 PE(2,0): recv_task: in_data = 4, global = 20
@1888 PE(3,0): recv_task: in_data = 4, global = 20

```

Note that each line printed to `sim.log` is prepended with the cycle at which
the print is encountered.

`<simprint>` is particularly useful for debugging stalling programs.
The `<debug>` library shown in the previous example requires a program to
complete to parse its output, but the `<simprint>` library prints to
`sim.log` whenever a newline character is encountered.

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
const prt = @import_module("<simprint>");

// Number of elements to be send to receivers
param num_elems: u16;

// Colors
param comm: color;

// Queue IDs
const comm_oq: output_queue = @get_output_queue(2);

// Task IDs
const exit_task_id: local_task_id = @get_local_task_id(8);

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
  prt.print_string("PE(0,0): sender beginning main_fn\n");
  @fmovs(out_dsd, buf_dsd, .{ .async = true, .activate = exit_task });
}

task exit_task() void {
  prt.print_string("PE(0,0): sender exiting\n");
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
const layout_mod = @import_module("<layout>");
const prt = @import_module("<simprint>");

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

// Variable whose value we update in recv_task
var global : u32 = 0;

// Array to store received values
var buf = @zeros([num_elems]u32);
var ptr_buf: [*]u32 = &buf;

// main_fn does nothing on the senders
fn main_fn() void {}

// Track number of wavelets received by recv_task
var num_wlts_recvd: u16 = 0;

// No newline character, so these vals will not be
// printed to simlog until newline character is encountered
fn simprint_pe_coords() void {
  prt.print_string("PE(");
  prt.print_u16_decimal(layout_mod.get_x_coord());
  prt.print_string(",");
  prt.print_u16_decimal(layout_mod.get_y_coord());
  prt.print_string("): ");
}

task recv_task(in_data : u32) void {

  simprint_pe_coords();
  prt.print_string("recv_task: in_data = ");
  prt.print_u32_decimal(in_data);

  buf[num_wlts_recvd] = in_data; // Store recvd value in buf
  global += 2*in_data; // Increment global by 2x received value

  prt.print_string(", global = ");
  prt.print_u32_decimal(global);
  prt.print_string("\n");

  num_wlts_recvd += 1; // Increment number of received wavelets
  // Once we have received all wavelets, we unblock cmd stream
  if (num_wlts_recvd == num_elems) {
    sys_mod.unblock_cmd_stream();
  }
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

# Receiver PEs write received value to out_buf,
# so out_buf of each PE should be same as x
# Assert that out_buf of each PE matches input array x
np.testing.assert_equal(np.tile(x, (width,1)), out_buf.reshape(width, num_elems))
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

Topic 12: Debug Library](csl/code-examples/tutorial-topic-12-debug-library.md "previous page")
[next

Topic 14: Color Swap](csl/code-examples/tutorial-topic-14-color-swap.md "next page")
