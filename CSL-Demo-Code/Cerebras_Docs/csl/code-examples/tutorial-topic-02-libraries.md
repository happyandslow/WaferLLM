---
title: "Topic 2: Libraries"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-topic-02-libraries"
crawled_at: "2026-02-14 23:55:25"
---

# Topic 2: Libraries

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-topic-02-libraries](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-02-libraries)

The CSL compiler comes bundled with a few standard libraries, which can be
imported into the user’s program using the `@import_module()` builtin. This
example shows three such compiler-bundled libraries:

- the `random` library for generating uniform random numbers,
- the `timestamp` library for reading the on-chip timestamp counter, and
- the `math` library for square root.

## layout.csl

```csl
// Color/ task ID map
//
//  ID var           ID var     ID var                ID var
//   0                9         18                    27 reserved (memcpy)
//   1               10         19                    28 reserved (memcpy)
//   2               11         20                    29 reserved
//   3               12         21 reserved (memcpy)  30 reserved (memcpy)
//   4               13         22 reserved (memcpy)  31 reserved
//   5               14         23 reserved (memcpy)  32
//   6               15         24                    33
//   7               16         25                    34
//   8               17         26                    35

param iterations: u32;

const memcpy = @import_module("<memcpy/get_params>", .{
  .width = 1,
  .height = 1,
});

layout {
  @set_rectangle(1, 1);

  @set_tile_code(0, 0, "pe_program.csl", .{
    .memcpy_params = memcpy.get_params(0),
    .iterations = iterations
  });

  // export symbol name
  @export_name("result", [*]f32, true);
  @export_name("start_timestamp", [*]u16, true);
  @export_name("finish_timestamp", [*]u16, true);
  @export_name("f_run", fn()void);
}

```

## pe\_program.csl

```csl
// Not a complete program; the top-level source file is layout.csl.
param memcpy_params: comptime_struct;

param iterations: u32;

const sys_mod = @import_module( "<memcpy/memcpy>", memcpy_params);

// Import compiler-bundled libraries, which are identified by names surrounded
// by angular brackets ('<' and '>').
const random = @import_module("<random>");
const tsc = @import_module("<time>");
const math = @import_module("<math>");

// Declare variables for storing the timestamp counter at the start and the end
// of the core computation.
var startBuffer = @zeros([tsc.tsc_size_words]u16);
var finishBuffer = @zeros([tsc.tsc_size_words]u16);

var start_ts_ptr: [*]u16 = &startBuffer;
var finish_ts_ptr: [*]u16 = &finishBuffer;

// Result to be copied back to the host
var result: [1]f32;
var result_ptr: [*]f32 = &result;

fn f_run() void {
  var idx: u32 = 0;
  var hitCount: u32 = 0;

  tsc.enable_tsc();
  tsc.get_timestamp(&startBuffer);

  // For each iteration, compute two random values between -1 and +1, and check
  // whether they are inside the circle of unit radius.
  while (idx < iterations) : (idx += 1) {
    var x = random.random_f32(-1.0, 1.0);
    var y = random.random_f32(-1.0, 1.0);
    var distanceFromOrigin = math.sqrt_f32(x * x + y * y);

    if (distanceFromOrigin <= 1.0) {
      hitCount += 1;
    }
  }

  tsc.get_timestamp(&finishBuffer);

  result[0] = 4.0 * @as(f32, hitCount) / @as(f32, iterations);

  sys_mod.unblock_cmd_stream();
}

comptime {
  @export_symbol(result_ptr, "result");
  @export_symbol(start_ts_ptr, "start_timestamp");
  @export_symbol(finish_ts_ptr, "finish_timestamp");
  @export_symbol(f_run);
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import numpy as np

from cerebras.sdk.sdk_utils import memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
parser.add_argument("--tolerance", type=float, help="tolerance for result")
args = parser.parse_args()
dirname = args.name

runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

result_symbol = runner.get_id('result')
start_ts_symbol = runner.get_id('start_timestamp')
finish_ts_symbol = runner.get_id('finish_timestamp')

runner.load()
runner.run()

print("step 1: call f_run to start computation")
runner.launch("f_run", nonblock=False)

print("step 2: copy back result")
# The D2H buffer must be of type u32
result = np.zeros(1, np.float32)
runner.memcpy_d2h(result, result_symbol, 0, 0, 1, 1, 1, \
    streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)

print("step 3: copy back timestamps")
# The D2H buffer must be of type u32
start_timestamps_u32 = np.zeros(3, np.uint32)
runner.memcpy_d2h(start_timestamps_u32, start_ts_symbol, 0, 0, 1, 1, 3, \
    streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)

finish_timestamps_u32 = np.zeros(3, np.uint32)
runner.memcpy_d2h(finish_timestamps_u32, finish_ts_symbol, 0, 0, 1, 1, 3, \
    streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)

# remove upper 16-bit of each u32
start_timestamps = memcpy_view(start_timestamps_u32, np.dtype(np.uint16))
finish_timestamps = memcpy_view(finish_timestamps_u32, np.dtype(np.uint16))

runner.stop()

# Helper functions for computing the delta in the cycle count
def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def subtract_timestamps(finish, start):
  return make_u48(finish) - make_u48(start)

cycles = subtract_timestamps(finish_timestamps, start_timestamps)
print("cycle count:", cycles)

print(f"result = {result}, np.pi = {np.pi}, tol = {args.tolerance}")
np.testing.assert_allclose(result, np.pi, atol=args.tolerance, rtol=0)
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 \
--params=iterations:200 -o out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out --tolerance 0.1

```
[previous

Topic 1: Arrays and Pointers](csl/code-examples/tutorial-topic-01-arrays-and-pointers.md "previous page")
[next

Topic 3: Streaming Wavelet Data](csl/code-examples/tutorial-topic-03-streaming-wavelet-data.md "next page")
