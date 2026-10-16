---
title: "Pipeline 1: Redirect fabin to fabout"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-pipeline-01-basic"
crawled_at: "2026-02-14 23:55:35"
---

# Pipeline 1: Redirect fabin to fabout

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-pipeline-01-basic](https://sdk.cerebras.net/csl/code-examples/tutorial-pipeline-01-basic)

While wavelet-triggered tasks enable us to receive and operate on one wavelet at
a time, the programmer may need a way to receive a tensor comprised of multiple
wavelets using one instruction. This is enabled by fabric input DSDs.
Similarly, using fabric output DSDs, the programmer can send multiple wavelets
using one instruction.

This example illustrates two fabric DSDs, one for input and another for output.
Each fabric DSD requires a corresponding color.

Crucially, when using a fabric input DSD, it is important that the programmer
blocks the wavelet’s color, as this example does for the color
`MEMCPYH2D_DATA_1`.
Otherwise, wavelets of that color will attempt to activate the (empty) task
associated with the color, which in turn will consume the wavelet before it can
be consumed by the fabric input DSD.

This example only has a single PE, which receives data via H2D and sends it out
via D2H in one vector operation. Logically speaking it is NOT valid because H2D
and D2H are serialized. The host triggers D2H only if H2D is done. The hardware
has some internal queues to hold the data for I/O, so H2D finishes when it
pushes all data into the dedicated queues. This example still works if the size
does not exceed the capacity of such queues. Otherwise H2D stalls.

The parameter `size` controls the number of wavelets of H2D and D2H. The
program stalls when `size` exceeds 14.

Such programming paradigm is called pipelined approach: the kernel receives
input data without storing it into memory, instead redirecting the result to
the output. The microthread is necessary because the CE (compute engine) must
have some resources to run `memcpy` kernel. The kernel stalls if a blocking
instruction `@add16(outDsd, inDsd, 1)` is used. The simulation stalls, and
the instruction trace shows `@add16` repeatedly querying data from input
queue 1, which is still empty. The router receives the H2D command much later
than running `@add16`. The CE has no resource to run the H2D command received
by the router, so it stalls.

## layout.csl

```csl
// Color/ task ID map
//
//  ID var           ID var  ID var               ID var
//   0 MEMCPYH2D_1    9      18                   27 reserved (memcpy)
//   1 MEMCPYD2H_1   10      19                   28 reserved (memcpy)
//   2               11      20                   29 reserved
//   3               12      21 reserved (memcpy) 30 reserved (memcpy)
//   4               13      22 reserved (memcpy) 31 reserved
//   5               14      23 reserved (memcpy) 32
//   6               15      24                   33
//   7               16      25                   34
//   8 main_task_id  17      26                   35

param size: i16;

param MEMCPYH2D_DATA_1_ID: i16;
param MEMCPYD2H_DATA_1_ID: i16;

const MEMCPYH2D_DATA_1: color = @get_color(MEMCPYH2D_DATA_1_ID);
const MEMCPYD2H_DATA_1: color = @get_color(MEMCPYD2H_DATA_1_ID);

const memcpy = @import_module("<memcpy/get_params>", .{
    .width = 1,
    .height = 1,
    .MEMCPYH2D_1 = MEMCPYH2D_DATA_1,
    .MEMCPYD2H_1 = MEMCPYD2H_DATA_1
    });

layout {
  @set_rectangle(1, 1);

  @set_tile_code(0, 0, "pe_program.csl", .{
    .size = size,
    .memcpy_params = memcpy.get_params(0)
  });
}

```

## pe\_program.csl

```csl
param memcpy_params: comptime_struct;

// number of elements received from host
param size: i16;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// Queues
const h2d_1_iq: input_queue = @get_input_queue(2);
const d2h_1_oq: output_queue = @get_output_queue(3);

// Task IDs
const main_task_id: local_task_id = @get_local_task_id(8);

const in_dsd = @get_dsd(fabin_dsd, .{
  .extent = size,
  .fabric_color = sys_mod.MEMCPYH2D_1,
  .input_queue = h2d_1_iq
});

const out_dsd = @get_dsd(fabout_dsd, .{
  .extent = size,
  .fabric_color = sys_mod.MEMCPYD2H_1,
  .output_queue = d2h_1_oq
});

var buf = @zeros([1]i16);
const one_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{size} -> buf[0] });

task main_task() void {
  // WARNING: large size can stall.
  // H2D and D2H are serialized. It is NOT safe to run "send" and "recv"
  // involving memcpy at the same time on the same PE.
  //
  // It only works for a small vector because the HW has some internal
  // queues to hold those values from/to IO. If such queues are full,
  // I/O stalls.
  //
  // In this case, if the length exceeds certain amount,
  // H2D cannot finish and D2H has no chance to run.

  buf[0] = @as(i16, 1);
  @add16(out_dsd, in_dsd, one_dsd, .{ .async = true });
}

comptime {
  // activate local task main_task at startup
  @activate(main_task_id);
  @bind_local_task(main_task, main_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(h2d_1_iq,  .{ .color = sys_mod.MEMCPYH2D_1 });
    @initialize_queue(d2h_1_oq,  .{ .color = sys_mod.MEMCPYD2H_1 });
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
size = int(params["size"])
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")
print(f"size = {size}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

runner.load()
runner.run()

# Generate a random input tensor of the desired size
input_tensor = np.random.randint(256, size=size, dtype=np.int16)

print("step 1: streaming H2D")
# The type of input_tensor is int16, so we need to extend to 32-bit for memcpy H2D
input_tensor_u32 = input_array_to_u32(input_tensor, 1, size)
runner.memcpy_h2d(MEMCPYH2D_DATA_1, input_tensor_u32, 0, 0, 1, 1, size, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("step 2: streaming D2H")
# The memcpy D2H buffer must be 32-bit
output_tensor_u32 = np.zeros(size, np.uint32)
runner.memcpy_d2h(output_tensor_u32, MEMCPYD2H_DATA_1, 0, 0, 1, 1, size, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
# remove upper 16-bit of each u32
result_tensor = memcpy_view(output_tensor_u32, np.dtype(np.int16))

runner.stop()

np.testing.assert_equal(result_tensor, input_tensor + 1)
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=8,3 \
--fabric-offsets=4,1 --params=size:12 -o out \
--params=MEMCPYH2D_DATA_1_ID:0 \
--params=MEMCPYD2H_DATA_1_ID:1 \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out

```
[previous

Topic 13: WSE-3 Microthreads](csl/code-examples/tutorial-topic-15-wse3-microthreads.md "previous page")
[next

Pipeline 2: Attach a FIFO to H2D](csl/code-examples/tutorial-pipeline-02-fifo.md "next page")
