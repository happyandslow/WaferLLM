---
title: "Topic 9: FIFOs"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-topic-09-fifos"
crawled_at: "2026-02-14 23:54:55"
---

# Topic 9: FIFOs

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-topic-09-fifos](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-09-fifos)

A FIFO DSD is useful to buffer input going into or out of a PE, as a way to
extend the small hardware queues used for fabric communication. In particular,
this may prevent stalls in the communication fabric when input or output
happens in bursts. It is also possible to operate on the values while they flow
through the FIFO, as this code sample demonstrates.

This example illustrates a typical pattern in the use of FIFOs, where a
receiver receives wavelets from the fabric and forwards them to a task that
performs some computation. Specifically, incoming data from the host is stored
in the FIFO, thus relieving the sender from being blocked until the receiver
has received all wavelets. While the incoming wavelets are being asynchronously
received into the FIFO buffer, we also start a second asynchronous DSD
operation that pulls data from the FIFO and forwards it to a wavelet-triggered
task.

This example also illustrates another common pattern, where a PE starts a
wavelet-triggered task using its own wavelets, by sending them to the router
which immediately sends them back to the compute element. In our example, this
wavelet-triggered task simply computes the cube of the wavelet’s data, before
sending the result to the host.

Note that, to demonstrate the use of FIFOs in this program, we use `memcpy`
streaming mode to stream data from the host and receive in the PE program’s
FIFO, and to stream data out of the PE program back to the host. Because
`memcpy` calls are serialized, the `memcpy_h2d` must finish before the
`memcpy_d2h`. This places an artificial restriction on our FIFO: the input
size from the host cannot exceed the FIFO size, or the program will potentially
stall.

## layout.csl

```csl
// Color map/ WSE-2 task ID map
// On WSE-2, data tasks are bound to colors (IDs 0 through 24)
//
//  ID var                      ID var  ID var                ID var
//   0                           9      18                    27 reserved (memcpy)
//   1                          10      19                    28 reserved (memcpy)
//   2 loopback_color/_task_id  11      20                    29 reserved
//   3                          12      21 reserved (memcpy)  30 reserved (memcpy)
//   4 MEMCPY_H2D_DATA_1        13      22 reserved (memcpy)  31 reserved
//   5 MEMCPY_D2H_DATA_1        14      23 reserved (memcpy)  32
//   6                          15      24                    33
//   7                          16      25                    34
//   8 main_task_id             17      26                    35

// WSE-3 task ID map
// On WSE-3, data tasks are bound to input queues (IDs 0 through 7)
//
//  ID var                      ID var  ID var                ID var
//   0 reserved (memcpy)         9      18                    27 reserved (memcpy)
//   1 reserved (memcpy)        10      19                    28 reserved (memcpy)
//   2                          11      20                    29 reserved
//   3                          12      21 reserved (memcpy)  30 reserved (memcpy)
//   4 loopback_iq/_task_id     13      22 reserved (memcpy)  31 reserved
//   5                          14      23 reserved (memcpy)  32
//   6                          15      24                    33
//   7                          16      25                    34
//   8 main_task_id             17      26                    35

// IDs for memcpy streaming colors
param MEMCPYH2D_DATA_1_ID: i16;
param MEMCPYD2H_DATA_1_ID: i16;

// Nubmer of elements received from MEMCPY_H2D_1 into fifo
param num_elems_to_process: i16;

// Colors
const MEMCPYH2D_DATA_1: color = @get_color(MEMCPYH2D_DATA_1_ID);
const MEMCPYD2H_DATA_1: color = @get_color(MEMCPYD2H_DATA_1_ID);
const loopback_color:   color = @get_color(2);

const memcpy = @import_module("<memcpy/get_params>", .{
  .width = 1,
  .height = 1,
  .MEMCPYH2D_1 = MEMCPYH2D_DATA_1,
  .MEMCPYD2H_1 = MEMCPYD2H_DATA_1
});

layout {
  @set_rectangle(1,1);

  @set_tile_code(0, 0, "buffer.csl", .{
    .memcpy_params = memcpy.get_params(0),
    .loopback_color = loopback_color,
    .num_elems_to_process = num_elems_to_process
  });

  @set_color_config(0, 0, loopback_color, .{.routes = .{.rx = .{RAMP}, .tx = .{RAMP}}});
}

```

## buffer.csl

```csl
param memcpy_params: comptime_struct;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// Nubmer of elements received from MEMCPY_H2D_1
param num_elems_to_process: i16;

// Colors
param loopback_color: color;

// Queue IDs
const h2d_data_1_iq: input_queue  = @get_input_queue(2);
const d2h_data_1_oq: output_queue = @get_output_queue(3);
const loopback_iq:   input_queue  = @get_input_queue(4);
const loopback_oq:   output_queue = @get_output_queue(5);

// Task IDs
// Data task process_task triggered by wlts along loopback_color
// On WSE-2, data task IDs are created from colors; on WSE-3, from input queues
const process_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(loopback_color)
  else if (@is_arch("wse3")) @get_data_task_id(loopback_iq);

const main_task_id: local_task_id = @get_local_task_id(8);

var fifo_buffer = @zeros([1024]f32);
const fifo = @allocate_fifo(fifo_buffer);

const in_dsd = @get_dsd(fabin_dsd, .{
  .extent = num_elems_to_process,
  .fabric_color = sys_mod.MEMCPYH2D_1,
  .input_queue = h2d_data_1_iq
});

const loopback_dsd = @get_dsd(fabout_dsd, .{
  .extent = num_elems_to_process,
  .fabric_color = loopback_color,
  .output_queue = loopback_oq
});

const ten = [1]f32 { 10.0 };
const ten_dsd = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{num_elems_to_process} -> ten[0]});

task main_task() void {
  // Move from the fabric to the FIFO
  // adding 10.0 to each element at the same time
  @fadds(fifo, in_dsd, ten_dsd, .{ .async = true });

  // Move from the FIFO to a process_task
  // negating values at the same time
  @fnegs(loopback_dsd, fifo, .{ .async = true });
}

const out_dsd = @get_dsd(fabout_dsd, .{
  .extent = 1,
  .fabric_color = sys_mod.MEMCPYD2H_1,
  .output_queue = d2h_data_1_oq
});

var elem = @zeros([1]f32);
const elem_dsd = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{1} -> elem[0]});

// Receive element from loopback color,
// then cube element, copy to elem buffer, and send to MEMCPY_D2H_1 color
task process_task(element: f32) void {
  // Block task to prevent its execution while element is
  // asynchronously sending to MEMCPY_D2H_1,
  // unblock when async send is done
  @block(process_task_id);

  elem[0] = element * element * element;
  @fmovs(out_dsd, elem_dsd, .{ .async = true, .unblock = process_task });
}

comptime {
  @activate(main_task_id);
  @bind_local_task(main_task, main_task_id);
  @bind_data_task(process_task, process_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(h2d_data_1_iq, .{ .color = sys_mod.MEMCPYH2D_1 });
    @initialize_queue(d2h_data_1_oq, .{ .color = sys_mod.MEMCPYD2H_1 });
    @initialize_queue(loopback_iq,   .{ .color = loopback_color });
    @initialize_queue(loopback_oq,   .{ .color = loopback_color });
  }
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
MEMCPYD2H_DATA_1 = int(params["MEMCPYD2H_DATA_1_ID"])
size = int(params["num_elems_to_process"])
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")
print(f"size = {size}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

runner.load()
runner.run()

np.random.seed(seed=7)

input_tensor = np.random.random(size).astype(np.float32)
print("step 1: streaming H2D")
runner.memcpy_h2d(MEMCPYH2D_DATA_1, input_tensor, 0, 0, 1, 1, size, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("step 2: streaming D2H")
result_tensor = np.zeros(size, np.float32)
runner.memcpy_d2h(result_tensor, MEMCPYD2H_DATA_1, 0, 0, 1, 1, size, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

runner.stop()

add_ten_negate = -(input_tensor + 10.0)
expected = add_ten_negate * add_ten_negate * add_ten_negate

np.testing.assert_equal(result_tensor, expected)
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl \
--fabric-dims=8,3 --fabric-offsets=4,1 \
--params=num_elems_to_process:512 \
--params=MEMCPYH2D_DATA_1_ID:4 \
--params=MEMCPYD2H_DATA_1_ID:5 \
-o out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out

```
[previous

Topic 8: Filters](csl/code-examples/tutorial-topic-08-filters.md "previous page")
[next

Topic 10: @map Builtin](csl/code-examples/tutorial-topic-10-map-builtin.md "next page")
