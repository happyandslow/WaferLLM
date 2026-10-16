---
title: "Topic 8: Filters"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-topic-08-filters"
crawled_at: "2026-02-14 23:55:21"
---

# Topic 8: Filters

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-topic-08-filters](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-08-filters)

Fabric filters allow a PE to selectively accept incoming wavelets. This example
shows the use of so-called range filters, which specify the wavelets to allow to
be forwarded to the CE based on the upper 16 bits of the wavelet contents.
Specifically, PE #0 sends all 12 wavelets to the other PEs, while each recipient
PE receives and processes only a quarter of the incoming wavelets.
See [Filter Configuration Semantics](csl/language/builtins.md#language-builtins-filters) for other possible filter configurations.

## layout.csl

```csl
// Color map
//
//  ID var           ID var            ID var                ID var
//   0                9                18                    27 reserved (memcpy)
//   1 data_color    10                19                    28 reserved (memcpy)
//   2               11                20                    29 reserved
//   3               12                21 reserved (memcpy)  30 reserved (memcpy)
//   4               13                22 reserved (memcpy)  31 reserved
//   5               14                23 reserved (memcpy)  32
//   6               15                24                    33
//   7               16                25                    34
//   8               17                26                    35

// See task maps in send.csl and recv.csl

// Colors
const data_color: color = @get_color(1);

const memcpy = @import_module("<memcpy/get_params>", .{
  .width = 4,
  .height = 1,
});

layout {
  @set_rectangle(4, 1);

  // Sender
  @set_tile_code(0, 0, "send.csl", .{
    .exch_color = data_color, .memcpy_params = memcpy.get_params(0), .pe_id = 0,
  });
  @set_color_config(0, 0, data_color, .{ .routes = .{ .rx = .{ RAMP }, .tx = .{ EAST } } });

  // Receivers
  for (@range(u16, 1, 4, 1)) |pe_id| {
    const filter = .{
      // Each PE should only accept three wavelets starting with the one whose
      // index field contains the value pe_id * 3.
      .kind = .{ .range = true },
      .min_idx = pe_id * 3,
      .max_idx = pe_id * 3 + 2,
    };

    @set_tile_code(pe_id, 0, "recv.csl", .{
      .recv_color = data_color, .memcpy_params = memcpy.get_params(pe_id), .pe_id = pe_id,
    });

    if (pe_id == 3) {
      @set_color_config(pe_id, 0, data_color, .{
        .routes = .{ .rx = .{ WEST }, .tx = .{ RAMP }}, .filter = filter
      });
    } else {
      @set_color_config(pe_id, 0, data_color, .{
        .routes = .{ .rx = .{ WEST }, .tx = .{ RAMP, EAST }}, .filter = filter
      });
    }
  }

  // export symbol names
  @export_name("result", [*]f32, true);
  @export_name("main_fn", fn()void);
}

```

## send.csl

```csl
// WSE-2 task ID map
// On WSE-2, data tasks are bound to colors (IDs 0 through 24)
//
//  ID var                ID var           ID var                ID var
//   0                     9               18                    27 reserved (memcpy)
//   1                    10 exit_task_id  19                    28 reserved (memcpy)
//   2                    11               20                    29 reserved
//   3                    12               21 reserved (memcpy)  30 reserved (memcpy)
//   4                    13               22 reserved (memcpy)  31 reserved
//   5                    14               23 reserved (memcpy)  32
//   6                    15               24                    33
//   7                    16               25                    34
//   8                    17               26                    35

// WSE-3 task ID map
// On WSE-3, data tasks are bound to input queues (IDs 0 through 7)
//
//  ID var                ID var           ID var                ID var
//   0 reserved (memcpy)   9               18                    27 reserved (memcpy)
//   1 reserved (memcpy)  10 exit_task_id  19                    28 reserved (memcpy)
//   2                    11               20                    29 reserved
//   3                    12               21 reserved (memcpy)  30 reserved (memcpy)
//   4                    13               22 reserved (memcpy)  31 reserved
//   5                    14               23 reserved (memcpy)  32
//   6                    15               24                    33
//   7                    16               25                    34
//   8                    17               26                    35

param memcpy_params: comptime_struct;
param pe_id: u16;

// Colors
param exch_color: color;

// Queues
const tx_oq: output_queue = @get_output_queue(2);

// Task IDs
const exit_task_id: local_task_id = @get_local_task_id(10);

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// Helper function to pack 16-bit index and 16-bit float value into one 32-bit
// wavelet.
fn pack(index: u16, data: f16) u32 {
  return (@as(u32, index) << 16) | @as(u32, @bitcast(u16, data));
}

const size = 12;
const data = [size]u32 {
  pack(0, 10.0),  pack( 1, 11.0), pack( 2, 12.0),
  pack(3, 13.0),  pack( 4, 14.0), pack( 5, 15.0),
  pack(6, 16.0),  pack( 7, 17.0), pack( 8, 18.0),
  pack(9, 19.0),  pack(10, 20.0), pack(11, 21.0),
};

// Function sends all data values to all east neighbors.
fn main_fn() void {
  const in_dsd = @get_dsd(mem1d_dsd, .{
    .tensor_access = |i|{size} -> data[i]
  });

  const out_dsd = @get_dsd(fabout_dsd, .{
    .extent = size,
    .fabric_color = exch_color,
    .output_queue = tx_oq,
  });

  @mov32(out_dsd, in_dsd, .{ .async = true, .activate = exit_task_id });
}

task exit_task() void {
  sys_mod.unblock_cmd_stream();
}

comptime {
  @bind_local_task(exit_task, exit_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(tx_oq, .{ .color = exch_color });
  }

  @export_symbol(main_fn);
}

```

## recv.csl

```csl
// WSE-2 task ID map
// On WSE-2, data tasks are bound to colors (IDs 0 through 24)
//
//  ID var                ID var  ID var                ID var
//   0                     9      18                    27 reserved (memcpy)
//   1 recv_task_id       10      19                    28 reserved (memcpy)
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
param pe_id: u16;

// Colors
param recv_color: color;

// Queues
const rx_iq: input_queue = @get_input_queue(2);

// Task IDs
// Data task recv_task triggered by wlts along recv_color
// On WSE-2, data task IDs are created from colors; on WSE-3, from input queues
const recv_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(recv_color)
  else if (@is_arch("wse3")) @get_data_task_id(rx_iq);

const sys_mod = @import_module( "<memcpy/memcpy>", memcpy_params);

// Each PE will receive 3 elements
const NUM_TO_RECV = 3;

var result = @zeros([NUM_TO_RECV]f32);
var result_ptr: [*]f32 = &result;

// Keep track of number of activations of recv_task
var iter: i16 = 0;

task recv_task(data: f16) void {
  // Write to result buffer at current iteration
  result[iter] = @as(f32, data / 2.0);
  iter += 1;

  if (iter == NUM_TO_RECV) {
    sys_mod.unblock_cmd_stream();
  }
}

// main_fn does nothing on recv PEs
fn main_fn() void {}

comptime {
  @bind_data_task(recv_task, recv_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(rx_iq, .{ .color = recv_color });
  }

  @export_symbol(result_ptr, "result");
  @export_symbol(main_fn);
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
dirname = args.name

runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

result_symbol = runner.get_id('result')

runner.load()
runner.run()

num_recv_pes = 3 # 3 PEs receive from the sender
elems_per_pe = 3 # Each recv PE receives 3 elems after filtering

print("step 1: launch function to send data to neighbors")
runner.launch("main_fn", nonblock=False)

print("step 2: copy back data from receiving PEs")
result = np.zeros(num_recv_pes * elems_per_pe, np.float32)
runner.memcpy_d2h(result, result_symbol, 1, 0, num_recv_pes, 1, elems_per_pe, streaming=False, \
   data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

runner.stop()

oracle = [6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5]
np.testing.assert_allclose(result, oracle, atol=0.0001, rtol=0)
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=11,3 --fabric-offsets=4,1 -o out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out

```
[previous

Topic 7: Switches and Control Entrypoints](csl/code-examples/tutorial-topic-07-switches-entrypt.md "previous page")
[next

Topic 9: FIFOs](csl/code-examples/tutorial-topic-09-fifos.md "next page")
