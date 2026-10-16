---
title: "GEMV 4: Parameters"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-04-params"
crawled_at: "2026-02-14 23:54:58"
---

# GEMV 4: Parameters

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-04-params](https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-04-params)

Parameter values are compile-time constants, which implies that the compiler
is fully aware of their precise value.
This enables the programmer to not just change the program’s behavior at
runtime, but it also enables the programmer to change the program’s
compilation.

Continuing on from the previous example, we add two compile-time parameters
to the `layout.csl` file that specify the dimensions `M` and `N` of our
problem, instead of hardcoding them in `pe_program.csl`.
When the program is compiled, the program specifies `M` and `N` in the
compile command. `layout.csl` also sets these parameter values in
`pe_program.csl` in its `@set_tile_code` call.

> **Note**
> Note
> See [GEMV Tutorial 4: Parameters](csl/tutorials/gemv-04-params.md#tut-gemv-04-params) for a step-by-step walkthrough
> of this example.

## layout.csl

```csl
param M: i16;
param N: i16;

const memcpy = @import_module("<memcpy/get_params>", .{ .width = 1, .height = 1 });

layout {
  @set_rectangle(1, 1);
  @set_tile_code(0, 0, "pe_program.csl", .{
    .memcpy_params = memcpy.get_params(0),
    .M = M,
    .N = N
  });

  // export symbol names
  @export_name("A", [*]f32, true);
  @export_name("x", [*]f32, true);
  @export_name("b", [*]f32, true);
  @export_name("y", [*]f32, false);
  @export_name("init_and_compute", fn()void);
}

```

## pe\_program.csl

```csl
param memcpy_params: comptime_struct;

// Matrix dimensions
param M: i16;
param N: i16;

// memcpy module provides infrastructure for copying data
// and launching functions from the host
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// 48 kB of global memory contain A, x, b, y
var A: [M*N]f32; // A is stored row major
var x: [N]f32;
var b: [M]f32;
var y = @zeros([M]f32); // Initialize y to zero

// DSDs for accessing A, b, y
// A_dsd accesses column of A
var A_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> A[i*N] });
var b_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &b, .extent = M });
var y_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = M });

// ptrs to A, x, b, y will be advertised as symbols to host
var A_ptr: [*]f32 = &A;
var x_ptr: [*]f32 = &x;
var b_ptr: [*]f32 = &b;
const y_ptr: [*]f32 = &y;

// Compute gemv
fn gemv() void {
  // Loop over all columns of A
  for (@range(i16, N)) |i| {
    // Calculate contribution to A*x from ith column of A, ith elem of x
    @fmacs(y_dsd, y_dsd, A_dsd, x[i]);
    // Move A_dsd to next column of A
    A_dsd = @increment_dsd_offset(A_dsd, 1, f32);
  }
  // Add b to A*x
  @fadds(y_dsd, y_dsd, b_dsd);
}

// Call initialize and gemv functions
fn init_and_compute() void {
  gemv();
  sys_mod.unblock_cmd_stream();
}

comptime {
  @export_symbol(A_ptr, "A");
  @export_symbol(x_ptr, "x");
  @export_symbol(b_ptr, "b");
  @export_symbol(y_ptr, "y");
  @export_symbol(init_and_compute);
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

N = int(compile_data['params']['N'])
M = int(compile_data['params']['M'])

# Construct A, x, b
A = np.arange(M*N, dtype=np.float32)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A.reshape(M,N)@x + b

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for A, b, x, y on device
A_symbol = runner.get_id('A')
x_symbol = runner.get_id('x')
b_symbol = runner.get_id('b')
y_symbol = runner.get_id('y')

# Load and run the program
runner.load()
runner.run()

# Copy A, x, b to device
runner.memcpy_h2d(A_symbol, A, 0, 0, 1, 1, M*N, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(x_symbol, x, 0, 0, 1, 1, N, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
runner.memcpy_h2d(b_symbol, b, 0, 0, 1, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Launch the init_and_compute function on device
runner.launch('init_and_compute', nonblock=False)

# Copy y back from device
y_result = np.zeros([M], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, 0, 0, 1, 1, M, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected, atol=0.01, rtol=0)
print("SUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=8,3 \
--fabric-offsets=4,1 --params=M:4,N:6 -o out --memcpy --channels 1
cs_python run.py --name out

```
[previous

GEMV 3: H2D and D2H Memcpy](csl/code-examples/tutorial-gemv-03-memcpy.md "previous page")
[next

GEMV 5: Multiple PEs](csl/code-examples/tutorial-gemv-05-multiple-pes.md "next page")
