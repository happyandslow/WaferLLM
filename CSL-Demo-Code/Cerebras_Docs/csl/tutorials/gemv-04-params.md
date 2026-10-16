---
title: "GEMV Tutorial 4: Parameters"
url: "https://sdk.cerebras.net/csl/tutorials/gemv-04-params"
crawled_at: "2026-02-14 23:55:43"
---

# GEMV Tutorial 4: Parameters

> Source: [https://sdk.cerebras.net/csl/tutorials/gemv-04-params](https://sdk.cerebras.net/csl/tutorials/gemv-04-params)

We’ve written a complete program that copies data to and from the
device, but both our host and device still need to define the
dimensions `M` and `N`.
Now we introduce compile time parameters to set `M` and `N`
while compiling our code, and show how our host code can read
these values from the compile output.

## Learning objectives

After completing this tutorial, you should know how to:

- Define compile time parameters for your device code
- Set the value of compile time parameters when compiling
- Read the value of compile time parameters from compile
  output in your host code

## Example overview

Our program will run on a single processing element (PE).
Like the previous tutorials, we will demonstrate the program
with a simulated fabric consisting of an 8 x 3 block of PEs.

Our problem steps are identical to the previous tutorial.
We need to modify our device code to replace the constants
`M` and `N` with parameters, and modify our compile
command to set these parameter values.
Our host code must be modified to read these values from
compile output.

## Modifying the CSL

How must we modify our layout code to support compile time parmeters
for `M` and `N`?

1. We need to define top level parameters for `M` and `N` that
   will be set by the compile command
2. We need to pass these parameters along to our PE program

Let’s take a look at the modified `layout.csl`:

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

Notice that we’ve defined two parameters at the top of the file:

```csl
param M: i16;
param N: i16;

```

Additionally, we pass these parameters along to our PE program inside of
our `@set_tile_code` call:

```csl
@set_tile_code(0, 0, "pe_program.csl", .{
  .memcpy_params = memcpy.get_params(0),
  .M = M,
  .N = N
});

```

Now let’s take a look at the modified `pe_program.csl`:

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

`pe_program.csl` must also contain parameter declarations for `M` and `N`.
When this file is compiled, it uses the values passed to it by `layout.csl`’s
`@set_tile_code` call to bind them.

`M` and `N` are no longer hard-coded in this file.

## Compiling and running the program

We’ve shown how the device and host code use the compile time
parameters, but how do we set them?
Our compile command now includes a `--params` flag, which
specifies the values:

```bash
$ cslc layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 --params=M:4,N:6 --memcpy --channels=1 -o out
$ cs_python run.py --name out

```

We use the same command to run.
You should see a `SUCCESS!` message at the end of execution.

## Exercises

`A` is stored row-major in the above code.
How would you rewrite `A_dsd` and the `gemv` function
if `A` were stored column major instead?

## Next

We’ve gone over some basics for writing a complete program
using a single PE.
Now let’s move on to using multiple PEs.

[previous

GEMV Tutorial 3: Memcpy](csl/tutorials/gemv-03-memcpy.md "previous page")
[next

GEMV Tutorial 5: Multiple PEs](csl/tutorials/gemv-05-multiple-pes.md "next page")
