---
title: "GEMV 1: A Complete Program"
url: "https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-01-complete-program"
crawled_at: "2026-02-14 23:55:32"
---

# GEMV 1: A Complete Program

> Source: [https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-01-complete-program](https://sdk.cerebras.net/csl/code-examples/tutorial-gemv-01-complete-program)

This example demonstrates a complete CSL program.

A complete program consists of a host program (a Python script, in this example)
and at least two CSL code files,
one of which defines the layout of the program across a collection of
processing elements (PEs) on the Wafer-Scale Engine (hereafter referred to
as “device”),
and one or more of which define the programs running on the individual PEs.
In this example, there is just one PE.

When executing the program, the user first compiles the CSL code files, and
then invokes the host program to copy data on and off the device and launch
functions on the device using a remote procedure call (RPC) mechanism.
The device used may be an actual CS system,
or it may be simulated without access to an actual CS system using the
Cerebras Fabric Simulator.

The host program here is defined in the `run.py` script, and the layout and
device code are defined in `layout.csl` and `pe_program.csl`.

The movement of data from host to device and back is done with memory to memory
copy semantics, which is provided by an SDK utility called `memcpy`.
The top of the `layout.csl` file imports a module which is used to
parameterize the program’s `memcpy` infrastructure.
This file also includes a layout block which specifies the number
and spatial arrangement of PEs used by this program, as well as the instructions
to execute on each PE.
Here, we instruct the compiler to produce executable code for 1 PE using the
code in `pe_program.csl`.

This program executes as follows.
The host code `run.py` uses the remote procedure call (RPC) mechanism to
launch a function called `init_and_compute` on the device.
This function initializes a 4 x 6 matrix `A`, stored in row major format,
a 6 x 1 vector `x`, and a 4 x 1 vector `b`.
Then, it computes the matrix-vector product of `Ax + b`
and stores it in `y`.

Once `init_and_compute` finishes on the device,
the host program performs a device-to-host memcpy with
the `memcpy_d2h` command to copy back the result stored in `y`,
and then checks that the answer is correct.
Notice the `unblock_cmd_stream` call in `pe_program.csl` that occurs
at the end of `init_and_compute`;
this call allows the device-to-host `memcpy_d2h` to proceed.

> **Note**
> Note
> See [GEMV Tutorial 1: A Complete Program](csl/tutorials/gemv-01-complete-program.md#tut-gemv-01-complete-program) for a step-by-step walkthrough
> of this example.

## layout.csl

```csl
// Import memcpy layout module for 1 x 1 grid of PEs
// This module defines parameters passed to program on the single PE
const memcpy = @import_module("<memcpy/get_params>", .{ .width = 1, .height = 1 });

layout {

  // Use just one 1 PE (columns=1, rows=1)
  @set_rectangle(1, 1);

  // The lone PE in this program should execute the code in "pe_program.csl"
  // We pass memcpy parameters as a parameter to the program. Note that
  // memcpy parameters are parameterized by the PE's column number.
  @set_tile_code(0, 0, "pe_program.csl", .{ .memcpy_params = memcpy.get_params(0) });

  // Export device symbol for array "y"
  // Last argument is mutability: host can read y, but not write to it
  @export_name("y", [*]f32, false);

  // Export host-callable device function
  @export_name("init_and_compute", fn()void);
}

```

## pe\_program.csl

```csl
// Struct containing parameters for memcpy layout
param memcpy_params: comptime_struct;

// memcpy module provides infrastructure for copying data
// and launching functions from the host
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// Constants definining dimensions of our matrix
const M: i16 = 4;
const N: i16 = 6;

// 48 kB of global memory contain A, x, b, y
var A: [M*N]f32; // A is stored row major
var x: [N]f32;
var b: [M]f32;
var y: [M]f32;

// Ptr to y will be exported as symbol to host
// Ptr is const, so host can read but not write to y
const y_ptr: [*]f32 = &y;

// Initialize matrix and vectors
fn initialize() void {
  // for loop with range syntax
  for (@range(i16, M*N)) |idx| {
    A[idx] = @as(f32, idx);
  }

  for (@range(i16, N)) |j| {
    x[j] = 1.0;
  }

  // while loop with iterator syntax
  var i: i16 = 0;
  while (i < M) : (i += 1) {
    b[i] = 2.0;
    y[i] = 0.0;
  }
}

// Compute gemv
fn gemv() void {
  for (@range(i16, M)) |i| {
    var tmp: f32 = 0.0;
    for (@range(i16, N)) |j| {
      tmp += A[i*N + j] * x[j];
    }
    y[i] = tmp + b[i];
  }
}

// Call initialize and gemv functions
fn init_and_compute() void {
  initialize();
  gemv();

  // After this function finishes, memcpy's cmd_stream must
  // be unblocked on all PEs for further memcpy commands
  // to execute
  sys_mod.unblock_cmd_stream();
}

comptime {
  // Export symbol pointing to y so it is host-readable
  @export_symbol(y_ptr, "y");

  // Export function so it is host-callable by RPC mechanism
  @export_symbol(init_and_compute);
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Matrix dimensions
M = 4
N = 6

# Construct A, x, b
A = np.arange(M*N, dtype=np.float32).reshape(M, N)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbol for copying y result off device
y_symbol = runner.get_id('y')

# Load and run the program
runner.load()
runner.run()

# Launch the init_and_compute function on device
runner.launch('init_and_compute', nonblock=False)

# Copy y back from device
# Arguments to memcpy_d2h:
# - y_result is array on host which will story copied-back array
# - y_symbol is symbol of device tensor to be copied
# - 0, 0, 1, 1 are (starting x-coord, starting y-coord, width, height)
#   of rectangle of PEs whose data is to be copied
# - M is number of elements to be copied from each PE
y_result = np.zeros([1*1*M], dtype=np.float32)
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
--fabric-offsets=4,1 -o out --memcpy --channels 1
cs_python run.py --name out

```
[previous

GEMV 0: Basic CSL Syntax](csl/code-examples/tutorial-gemv-00-basic-syntax.md "previous page")
[next

GEMV 2: Memory DSDs](csl/code-examples/tutorial-gemv-02-memory-dsds.md "next page")
