---
title: "Topic 7: Switches and Control Entrypoints"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-topic-07-switches-entrypt"
crawled_at: "2026-02-14 23:55:32"
---

# Topic 7: Switches and Control Entrypoints

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-topic-07-switches-entrypt](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-07-switches-entrypt)

Following on from the last topic example, we can also encode a special
task ID inside of a control wavelet. When that control wavelet is forwarded
to the CE of the receiving PE, it will activate a task known as a control
task which is bound to that ID.

The lower 16 bits of the control wavelet can be used to store an optional
data payload for that control task. Here, we encode the same values
sent to the PEs as normal data wavelets in the previous example.

Note that a PE router will move to a new switch position only after the
control wavelet carrying the switch command passes through that PE.
Therefore all control wavelets will continue to be routed using the current
switch position setting and the new switch position will only affect
subsequent wavelets. Thus, the data payload of a control wavelet is received
by the PE connected by the current switch position, not the new position.

## layout.csl

```csl
// Color map
//
//  ID var           ID var  ID var                ID var
//   0 channel        9      18                    27 reserved (memcpy)
//   1               10      19                    28 reserved (memcpy)
//   2               11      20                    29 reserved
//   3               12      21 reserved (memcpy)  30 reserved (memcpy)
//   4               13      22 reserved (memcpy)  31 reserved
//   5               14      23 reserved (memcpy)  32
//   6               15      24                    33
//   7               16      25                    34
//   8               17      26                    35

// See task maps in send.csl and recv.csl

// Colors
const channel: color = @get_color(0);

const memcpy = @import_module("<memcpy/get_params>", .{
  .width = 3,
  .height = 3,
  });

layout {
  @set_rectangle(3, 3);

  const memcpy_params_0 = memcpy.get_params(0);
  const memcpy_params_1 = memcpy.get_params(1);
  const memcpy_params_2 = memcpy.get_params(2);

  // The core has 3-by-3 rectangle of PEs.
  // Out of the nine PEs, the PE in the center (PE #1,1) will send four
  // control wavelets to the PE's four adjacent neighbors.  These four
  // adjacent numbers are programmed to receive the control wavelets, whereas
  // all other PEs (i.e. the PEs at the corners of the rectangle) are
  // programmed to contain no instructions or routes.

  // Sender
  @set_tile_code(1, 1, "send.csl", .{
    .memcpy_params = memcpy_params_1, .tx_color = channel,
  });

  const sender_routes = .{
    // The default route, which is to receive from ramp and send to north
    .rx = .{ RAMP },
    .tx = .{ NORTH }
  };

  const sender_switches = .{
    // Upon a control wavelet, change the transmit direction to west
    .pos1 = .{ .tx = WEST },
    // Upon another control wavelet, change the transmit direction to east
    .pos2 = .{ .tx = EAST },
    // Upon yet another control wavelet, change the transmit direction to south
    .pos3 = .{ .tx = SOUTH },
    // Send to west PE first, then east PE, then south PE, and then north PE
    .current_switch_pos = 1,
    // Wrap around from position 3 to position 0 after receiving control wavelet
    .ring_mode = true,
  };

  @set_color_config(1, 1, channel, .{ .routes = sender_routes,
                                      .switches = sender_switches });

  // Receivers
  @set_tile_code(1, 0, "recv.csl", .{
    .memcpy_params = memcpy_params_1, .rx_color = channel,
  });
  @set_color_config(1, 0, channel, .{ .routes = .{ .rx = .{ SOUTH }, .tx = .{ RAMP }}});

  @set_tile_code(0, 1, "recv.csl", .{
    .memcpy_params = memcpy_params_0, .rx_color = channel,
  });
  @set_color_config(0, 1, channel, .{ .routes = .{ .rx = .{ EAST }, .tx = .{ RAMP }}});

  @set_tile_code(2, 1, "recv.csl", .{
    .memcpy_params = memcpy_params_2, .rx_color = channel,
  });
  @set_color_config(2, 1, channel, .{ .routes = .{ .rx = .{ WEST }, .tx = .{ RAMP }}});

  @set_tile_code(1, 2, "recv.csl", .{
    .memcpy_params = memcpy_params_1, .rx_color = channel,
  });
  @set_color_config(1, 2, channel, .{ .routes = .{ .rx = .{ NORTH }, .tx = .{ RAMP }}});

  // Empty PEs
  @set_tile_code(0, 0, "empty.csl", .{ .memcpy_params = memcpy_params_0 });
  @set_tile_code(2, 0, "empty.csl", .{ .memcpy_params = memcpy_params_2 });
  @set_tile_code(0, 2, "empty.csl", .{ .memcpy_params = memcpy_params_0 });
  @set_tile_code(2, 2, "empty.csl", .{ .memcpy_params = memcpy_params_2 });

  // export symbol names
  @export_name("result", [*]u32, false);
  @export_name("main_fn", fn()void);
}

```

## send.csl

```csl
// WSE-2 task ID map
// On WSE-2, data tasks are bound to colors (IDs 0 through 24)
//
//  ID var                ID var   ID var                ID var
//   0                     9       18                    27 reserved (memcpy)
//   1                    10       19                    28 reserved (memcpy)
//   2                    11       20                    29 reserved
//   3                    12       21 reserved (memcpy)  30 reserved (memcpy)
//   4                    13       22 reserved (memcpy)  31 reserved
//   5                    14       23 reserved (memcpy)  32
//   6                    15       24                    33
//   7                    16       25                    34
//   8                    17       26                    35

// WSE-3 task ID map
// On WSE-3, data tasks are bound to input queues (IDs 0 through 7)
//
//  ID var                ID var   ID var                ID var
//   0 reserved (memcpy)   9       18                    27 reserved (memcpy)
//   1 reserved (memcpy)  10       19                    28 reserved (memcpy)
//   2                    11       20                    29 reserved
//   3                    12       21 reserved (memcpy)  30 reserved (memcpy)
//   4                    13       22 reserved (memcpy)  31 reserved
//   5                    14       23 reserved (memcpy)  32
//   6                    15       24                    33
//   7                    16       25                    34
//   8                    17       26                    35

param memcpy_params: comptime_struct;

// Colors
param tx_color: color;

// Queues
const tx_oq: output_queue = @get_output_queue(2);

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
const ctrl = @import_module("<control>");

// fabout DSD used to send ctrl wavelet to fabric along tx_color
const tx_ctrl_dsd = @get_dsd(fabout_dsd, .{
  .extent = 1,
  .fabric_color = tx_color,
  .control = true,
  .output_queue = tx_oq
});

fn main_fn() void {
  // ID of control task activated on receivers
  const recv_ctrl_id = @get_control_task_id(40);

  // Now we can reuse a single color to send data to the four neighbors of this PE.
  // We forward the payload of this control wavelet to the CE, where the ctrl task
  // bound to recv_ctrl_id will be activated. data argument is 16-bit arg of ctrl task.

  @mov32(tx_ctrl_dsd, ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, false, recv_ctrl_id, 0));
  @mov32(tx_ctrl_dsd, ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, false, recv_ctrl_id, 2));
  @mov32(tx_ctrl_dsd, ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, false, recv_ctrl_id, 4));
  @mov32(tx_ctrl_dsd, ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, false, recv_ctrl_id, 6));

  sys_mod.unblock_cmd_stream();
}

comptime {
  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(tx_oq, .{ .color = tx_color });
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
//   1                    10      19                    28 reserved (memcpy)
//   2                    11      20                    29 reserved
//   3                    12      21 reserved (memcpy)  30 reserved (memcpy)
//   4                    13      22 reserved (memcpy)  31 reserved
//   5                    14      23 reserved (memcpy)  32
//   6                    15      24                    33
//   7                    16      25                    34
//   8                    17      26                    35
//   ...
//  40 recv_ctrl_id

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
//   8                    17      26                    35
//   ...
//  40 recv_ctrl_id

param memcpy_params: comptime_struct;

// Colors
param rx_color: color;

// Queues
const rx_iq: input_queue = @get_input_queue(2);

// Task IDs
const recv_ctrl_id: control_task_id = @get_control_task_id(40);

var result = @zeros([1]u32);
const result_ptr: [*]u32 = &result;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// main_fn does nothing on recv PEs
fn main_fn() void {
  sys_mod.unblock_cmd_stream();
}

// Control task receives 16-bit data from ctrl wavelet paylaod
task recv_ctrl_task(data: u16) void {
  result[0] = @as(u32, data);
}

comptime {
  @bind_control_task(recv_ctrl_task, recv_ctrl_id);

  // Since there is no data task receiving wlts along rx_color, color/ queue
  // must be explicitly unblocked for CE to receive and ctrl task to be activated
  // On WSE-2, we unblock color. On WSE-3, we unblock queue to which color is bound
  if (@is_arch("wse3")) @unblock(rx_iq) else @unblock(rx_color);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(rx_iq, .{ .color = rx_color });
  }

  @export_symbol(result_ptr, "result");
  @export_symbol(main_fn);
}

```

## empty.csl

```csl
// Every PE needs to import memcpy module otherwise the I/O cannot
// propagate the data to the destination.

param memcpy_params: comptime_struct;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

fn main_fn() void {
  sys_mod.unblock_cmd_stream();
}

comptime {
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

# Get symbol for copying recv results off device
result_symbol = runner.get_id('result')

runner.load()
runner.run()

runner.launch('main_fn', nonblock=False)

# Copy arr back from PEs that received wlts
west_result = np.zeros([1], dtype=np.uint32)
runner.memcpy_d2h(west_result, result_symbol, 0, 1, 1, 1, 1, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

east_result = np.zeros([1], dtype=np.uint32)
runner.memcpy_d2h(east_result, result_symbol, 2, 1, 1, 1, 1, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

south_result = np.zeros([1], dtype=np.uint32)
runner.memcpy_d2h(south_result, result_symbol, 1, 2, 1, 1, 1, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

north_result = np.zeros([1], dtype=np.uint32)
runner.memcpy_d2h(north_result, result_symbol, 1, 0, 1, 1, 1, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.stop()

print("West result: ", west_result)
print("East result: ", east_result)
print("South result: ", south_result)
print("North result: ", north_result)

np.testing.assert_equal(0, west_result)
np.testing.assert_equal(2, east_result)
np.testing.assert_equal(4, south_result)
np.testing.assert_equal(6, north_result)

print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=10,5 --fabric-offsets=4,1 -o out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out

```
[previous

Topic 6: Switches](csl/code-examples/tutorial-topic-06-switches.md "previous page")
[next

Topic 8: Filters](csl/code-examples/tutorial-topic-08-filters.md "next page")
