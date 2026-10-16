---
title: "Topic 10: @map Builtin"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-topic-10-map-builtin"
crawled_at: "2026-02-14 23:55:44"
---

# Topic 10: @map Builtin

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-topic-10-map-builtin](https://sdk.cerebras.net/csl/code-examples/tutorial-topic-10-map-builtin)

The `@map` builtin can be used to perform custom operations on the data
elements of one or more DSDs. In other words, it is a
*customizable DSD operation* that allows us to go beyond the
[fixed list](csl/language/builtins.md#language-builtins-for-dsd-operations) of
natively supported DSD operations.

This example demonstrates three use-cases of the `@map` builtin:

1. In the first use-case, `@map` is used to compute the square-root of the
   diagonal elements of a 2D tensor.
2. In the second use-case `@map` is used to perform a custom calculation with
   a mix of input DSDs of various kinds (`mem1d_dsd` and `fabin_dsd`) and
   scalar values while the result is stored to a `mem1d_dsd`. It shows how we
   can use arbitrary callbacks combined with a variety of input and output DSDs.
3. Finally, we demonstrate how `@map` can be used to compute a reduction like
   the sum of all elements in a tensor.

Without `@map`, we would have to write explicit loops iterating over each
element involved in these computations. With `@map` we can avoid writing such
loops by utilizing the DSD descriptions which specify the loop structure
implicitly. Since DSDs are supported natively by the hardware, using `@map`
can lead to significant performance gains compared to writing explicit loops.

## layout.csl

```csl
// Color/ task ID map
//
//  ID var           ID var      ID var                ID var
//   0                9          18                    27 reserved (memcpy)
//   1               10          19                    28 reserved (memcpy)
//   2               11          20                    29 reserved
//   3               12          21 reserved (memcpy)  30 reserved (memcpy)
//   4               13          22 reserved (memcpy)  31 reserved
//   5               14          23 reserved (memcpy)  32
//   6               15          24                    33
//   7               16          25                    34
//   8               17          26                    35

param size: i16;

const memcpy = @import_module( "<memcpy/get_params>", .{
  .width = 1,
  .height = 1,
});

layout {
  @set_rectangle(1, 1);

  @set_tile_code(0, 0, "pe_program.csl", .{
    .memcpy_params = memcpy.get_params(0),
    .size = size,
  });

  // export symbol name
  @export_name("weight", [*]f32, true);
  @export_name("sqrt_diag_A", [*]f32, true);
  @export_name("A", [*]f32, true);
  @export_name("sum", [*]i32, true);
  @export_name("f_run", fn()void);
}

```

## pe\_program.csl

```csl
// Not a complete program; the top-level source file is layout.csl.
param memcpy_params: comptime_struct;

param size: i16;

// Task IDs
param main_task_id: local_task_id;

const sys_mod = @import_module( "<memcpy/memcpy>", memcpy_params);
const math_lib = @import_module("<math>");

// A transformed in place by @map operation 2
var A = @constants([size, size]f32, 42.0);
var ptr_A: [*]f32 = &A;

const B = [size]i32{10, 20, 30, 40, 50};

// Copied in from the host
var weight = @zeros([size]f32);
var ptr_weight: [*]f32 = &weight;

// sqrt_diag_A computed by @map operation 1
var sqrt_diag_A = @zeros([size]f32);
var ptr_sqrt_diag_A: [*]f32 = &sqrt_diag_A;

// sum computed by @map operation 3
var sum = @zeros([1]i32);
var ptr_sum: [*]i32 = &sum;

// The loop structure is implicitly specified by the memory DSD descriptions
const dsdA = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{size} -> A[i, i]});
const dsdB = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{size} -> B[i]});

const dsd_sqrt_diag_A = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{size} -> sqrt_diag_A[i]});
const dsd_weight = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{size} -> weight[i]});

fn transformation(value: f32, coeff1: f32, coeff2: f32, weight: f32) f32 {
  return value * (coeff1 + weight) + value * (coeff2 + weight);
}

fn reduction(value: i32, sum: *i32) i32 {
  return sum.* + value;
}

fn f_run() void {
  // @map operation 1
  // Compute the square-root of each element of `dsdA` and send it
  // to `dsd_sqrt_diag_A`. We avoid writing an explicit loop and rely
  // on the DSD description instead.
  @map(math_lib.sqrt_f32, dsdA, dsd_sqrt_diag_A);

  // @map operation 2
  // Transform tensor A in-place through a custom calculation.
  @map(transformation, dsdA, 2.0, 6.0, dsd_weight, dsdA);

  // @map operation 3
  // Compute the sum of all elements in tensor B.
  @map(reduction, dsdB, &sum[0], &sum[0]);

  // WARNING: the user must unblock cmd color for every PE
  sys_mod.unblock_cmd_stream();
}

comptime{
  @export_symbol(ptr_weight, "weight");
  @export_symbol(ptr_sqrt_diag_A, "sqrt_diag_A");
  @export_symbol(ptr_A, "A");
  @export_symbol(ptr_sum, "sum");
  @export_symbol(f_run);
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
parser.add_argument('--cmaddr', help='IP:port for CS system')
args = parser.parse_args()
dirname = args.name

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
params = compile_data["params"]
size = int(params["size"])
print(f"size = {size}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT

runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_weight = runner.get_id("weight")
sym_sqrt_diag_A = runner.get_id("sqrt_diag_A")
sym_A = runner.get_id("A")
sym_sum = runner.get_id("sum")

runner.load()
runner.run()

A = np.array([[42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0]]).astype(np.float32)
B = np.array([10, 20, 30, 40, 50]).astype(np.int32)

def transformation(value: np.array, coeff1: float, coeff2: float, weight: np.array):
  return np.multiply(value, coeff1 + weight) + np.multiply(value, coeff2 + weight)

def reduction(array):
  return sum(array)

np.random.seed(seed=7)

print("step 1: copy weights to device")
weights = np.random.random(size).astype(np.float32)
runner.memcpy_h2d(sym_weight, weights, 0, 0, 1, 1, size, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

print("step 2: call f_run to test @map")
runner.launch("f_run", nonblock=False)

print("step 3: copy results back to host")
sqrt_result = np.zeros(size, np.float32)
runner.memcpy_d2h(sqrt_result, sym_sqrt_diag_A, 0, 0, 1, 1, size, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

sum_result = np.zeros(1, np.int32)
runner.memcpy_d2h(sum_result, sym_sum, 0, 0, 1, 1, 1, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

A_trans_result = np.zeros(size*size, np.float32)
runner.memcpy_d2h(A_trans_result, sym_A, 0, 0, 1, 1, size*size, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

runner.stop()

# Sqrt example
sqrt_expected = np.sqrt(np.diag(A))
np.testing.assert_equal(sqrt_result, sqrt_expected)

# Transformation example
trans_expected = transformation(np.diag(A), 2.0, 6.0, weights)
np.fill_diagonal(A, trans_expected)
np.testing.assert_equal(A_trans_result.reshape((5, 5)), A)

# Reduction example
sum_expected = np.array([reduction(B)], dtype=np.int32)
np.testing.assert_equal(sum_expected, sum_result)

print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl \
--fabric-dims=8,3 --fabric-offsets=4,1 --params=size:5 \
-o out --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out

```
[previous

Topic 9: FIFOs](csl/code-examples/tutorial-topic-09-fifos.md "previous page")
[next

Topic 11: Collective Communications](csl/code-examples/tutorial-topic-11-collectives.md "next page")
