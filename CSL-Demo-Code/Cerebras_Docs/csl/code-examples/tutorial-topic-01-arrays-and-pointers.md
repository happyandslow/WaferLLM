---
title: "Topic 1: Arrays and Pointers"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-topic-01-arrays-and-pointers"
crawled_at: "2026-02-14 23:55:27"
---

# Topic 1: Arrays and Pointers

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-topic-01-arrays-and-pointers](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-01-arrays-and-pointers)

Arrays can only be passed to or returned from functions used at compile-time.
For functions used at runtime, pointers should be used instead. This example
demonstrates a function `increment_and_sum()`, which accepts a pointer to an
array and a pointer to a scalar. When declaring an array pointer, CSL requires
that the type specification contain the size of the array. CSL does not have
a null pointer.

Pointers are dereferenced using the `.*` syntax. Once dereferenced, they can
be used just like non-pointer variables like `(data_ptr.*)[0]` for indexing
into the first element of the array.

## layout.csl

```csl
// The core kernel must start at P4.1 so the memcpy infrastructure has enough
// resources to route the data between the host and the device.

// Color/ task ID map
//
//  ID var  ID var         ID var                ID var
//   0       9             18                    27 reserved (memcpy)
//   1      10             19                    28 reserved (memcpy)
//   2      11             20                    29 reserved
//   3      12             21 reserved (memcpy)  30 reserved (memcpy)
//   4      13             22 reserved (memcpy)  31 reserved
//   5      14             23 reserved (memcpy)  32
//   6      15             24                    33
//   7      16             25                    34
//   8      17             26                    35

const memcpy = @import_module("<memcpy/get_params>", .{
  .width = 1,
  .height = 1,
});

layout {
  @set_rectangle(1, 1);

  @set_tile_code(0, 0, "pe_program.csl", .{ .memcpy_params = memcpy.get_params(0) });

  // export symbol name
  @export_name("result", [*]i16, true);
  @export_name("f_run", fn()void);
}

```

## pe\_program.csl

```csl
// Not a complete program; the top-level source file is layout.csl

param memcpy_params: comptime_struct;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

var result: [1]i16;
var result_ptr: [*]i16 = &result;

fn increment_and_sum(data_ptr: *[3]i16, result_ptr: *i16) void {
  // Write an updated value to each element of the array
  (data_ptr.*)[0] += 1;
  (data_ptr.*)[1] += 1;
  (data_ptr.*)[2] += 1;

  // Read all array values, sum them, and write the result
  result_ptr.* = (data_ptr.*)[0] + (data_ptr.*)[1] + (data_ptr.*)[2];
}

fn f_run() void {
  var data = [3]i16 { 1, 2, 3 };

  increment_and_sum(&data, &result[0]);

  sys_mod.unblock_cmd_stream();
}

comptime {
  @export_symbol(result_ptr, "result");
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
args = parser.parse_args()
dirname = args.name

memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

result_symbol = runner.get_id('result')

runner.load()
runner.run()

runner.launch("f_run", nonblock=False)

# The D2H buffer must be of type u32
out_tensors_u32 = np.zeros(1, np.uint32)
runner.memcpy_d2h(out_tensors_u32, result_symbol, 0, 0, 1, 1, 1, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

# remove upper 16-bit of each u32
result_tensor = memcpy_view(out_tensors_u32, np.dtype(np.int16))

runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_equal(result_tensor, [9])
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 -o out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out

```
[previous

GEMV 9: Memcpy Streaming Mode](csl/code-examples/tutorial-gemv-09-streaming.md "previous page")
[next

Topic 2: Libraries](csl/code-examples/tutorial-topic-02-libraries.md "next page")
