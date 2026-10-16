---
title: "GEMV Tutorial 2: Memory DSDs"
url: "https://sdk.cerebras.net/csl/tutorials/gemv-02-memory-dsds"
crawled_at: "2026-02-14 23:55:04"
---

# GEMV Tutorial 2: Memory DSDs

> Source: [https://sdk.cerebras.net/csl/tutorials/gemv-02-memory-dsds](https://sdk.cerebras.net/csl/tutorials/gemv-02-memory-dsds)

Now that we’ve written a complete program, let’s introduce
a central concept in CSL: memory Data Structure Descriptors (DSDs).
Memory DSDs provide an efficient mechanism for performing operations
on entire tensors.

## Learning objectives

After completing this tutorial, you should know how to:

- Define memory DSDs for tensor accesses
- Use memory DSDs in builtin operations on tensors
- Use builtins to initialize tensors

## Example overview

Our program will run on a single processing element (PE).
Like the previous tutorial, we will demonstrate the program
with a simulated fabric consisting of an 8 x 3 block of PEs.

Our problem steps are identical to the previous tutorial.
Our layout file, host code, and compile and run commands
are also identical.
We only need to modify `pe_program.csl`, and we’ll take
a closer look at changes to this file.

## Modifying the CSL

In the previous tutorial, we created a complete CSL program using a
single PE to initialize and compute `y = Ax + b`.
What do we need to do in `pe_program.csl` to take advantage of
memory DSDs and builtin operations on tensors?

1. We need to define DSDs for accessing our tensors
2. We need to rewrite the `gemv` function to operate on these DSDs

We previously walked through `layout.csl`, which is the same for
this tutorial as the previous one.
We include the new `pe_program.csl` below, and highlight the
changes in this code.

```csl
param memcpy_params: comptime_struct;

// memcpy module provides infrastructure for copying data
// and launching functions from the host
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

// Constants definining dimensions of our matrix
const M: i16 = 4;
const N: i16 = 6;

// 48 kB of global memory contain A, x, b, y
var A: [M*N]f32; // A is stored row major

// Initialize x, b, y using builtins
var x = @constants([N]f32, 1.0);
var b = @constants([M]f32, 2.0);
var y = @zeros([M]f32);

// DSDs for accessing A, b, y
// b_dsd uses tensor access expression to specify access to M consecutive elements of b
var b_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> b[i] });
// The above expression is equivalent to:
// var b_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &b, .extent = M });

// y_dsd uses base_address and extent fields to specify access to M consecutive elements of y
var y_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = M });
// The above expression is equivalent to:
// var y_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> y[i] });

// A_dsd accesses column of A
// A_dsd uses tensor access expression to specify access to every Nth element of A
var A_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> A[i*N] });
// The above expression is equivalent to:
// var A_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &A, .extent = M, .stride = N });

// ptr to y will be advertised as symbol to host
const y_ptr: [*]f32 = &y;

// Initialize A matrix
fn initialize() void {
  // for loop with range syntax
  for (@range(i16, M*N)) |idx| {
    A[idx] = @as(f32, idx);
  }
}

// Compute gemv
fn gemv() void {
  // Loop over all columns of A
  for (@range(u16, N)) |i| {
    // Calculate contribution to A*x from ith column of A, ith elem of x
    @fmacs(y_dsd, y_dsd, A_dsd, x[i]);
    A_dsd = @increment_dsd_offset(A_dsd, 1, f32);
  }
  // Add b to A*x
  @fadds(y_dsd, y_dsd, b_dsd);
}

// Call initialize and gemv functions
fn init_and_compute() void {
  initialize();
  gemv();
  sys_mod.unblock_cmd_stream();
}

comptime {
  @export_symbol(y_ptr, "y");
  @export_symbol(init_and_compute);
}

```

### Defining our memory DSDs

First, let’s take a look at the DSDs we define for accessing `b` and `y`:

```csl
var b_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> b[i] });
var y_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> y[i] });

```

`b_dsd` and `y_dsd` are the memory DSDs for
accessing `b`, and `y`, respectively.

The `tensor_access` field defines the access pattern of these DSDs.
`|i|` specifies the induction variable, and `{M}` specifies
the loop bound; i.e., these DSDs will access `M` elements.
After `->`, an expression is given for accessing a memory location
using the induction variable.
This expression must be **affine**, or linear plus a constant.

The access pattern for these DSDs is straightforward: these DSDs
loop over all `M` elements, in order, of their respective tensors.

Now let’s take a look at the DSD for accessing `A`:

```csl
var A_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> A[i*N] });

```

This DSD accesses `M` elements of `A`, but strided by `N` elements;
i.e., `A_dsd` accesses elements `0, N, 2*N, ... (M-1)*N`.
Because `A` is stored in row major format, this means that `A_dsd`
as defined here accesses the 0th column of `A`.

> **Note**
> Note
> These memory DSDs are of type `mem1d_dsd`, which are one-dimensional
> memory DSDs. CSL also provides `mem4d_dsd`, multidimensional memory
> DSDs for up to four dimensions.
> You can learn more about memory DSDs in our language reference guide
> [Data Structure Descriptors](csl/language/dsds.md#language-dsds).

### Using our DSDs to compute GEMV

Now that we’ve defined our DSDs, let’s take a look at how to use them to
compute GEMV.

Recall that our previous `gemv()` function was defined as follows:

```csl
fn gemv() void {
  for (@range(i16, M)) |i| {
    var tmp: f32 = 0.0;
    for (@range(i16, N)) |j| {
      tmp += A[i*N + j] * x[j];
    }
    y[i] = tmp + b[i];
  }
}

```

Now, our `gemv()` looks like this:

```csl
fn gemv() void {
  for (@range(u16, N)) |i| {
    @fmacs(y_dsd, y_dsd, A_dsd, x[i]);
    A_dsd = @increment_dsd_offset(A_dsd, 1, f32);
  }
  @fadds(y_dsd, y_dsd, b_dsd);
}

```

Notice that we now only have one explicit loop over `N`,
instead of two explicit loops.
At each iteration, this `@fmacs` operation does the following:

- performs a vector-scalar multiplication between the column of `A`
  referenced by `A_dsd` and the scalar `x[i]`,
- performs an elementwise vector addition between this result and
  the vector `y`,
- and stores this final result into `y`.

Thus, each `@fmacs` operation increments the `M` elements of `y`
by the vector-scalar product of column `i` of `A`
and element `i` of `x`.

The `@increment_dsd_offset` operation at each loop iteration increments
`A_dsd` to reference the next column of `A`.
This builtin operation takes `A_dsd` and creates a new DSD by offseting
its access by 1 `f32` element.

For instance, the first time this operation occurs, `A_dsd` will now
access elements `1, N+1, 2*N+1, ... (M-1)*N+1` of `A`.
Again, because `A` is stored row major,
this will access the 1st column of `A`.

Once this loop over the `N` columns of `A` is complete,
`y` contains the result of `A*x`.
The `@fadds` operation performs an elementwise vector addition between
`y` and `b`, storing the result back in `y`.
Now `y` contains the result of `A*x + b`.

### Using builtins to initialize tensors

You may have noticed one other slight change to this code.
Instead of initializing `x`, `b`, and `y`, in the `initialize` function,
we make use of builtins to provide values for them at declaration:

```csl
var x = @constants([N]f32, 1.0);
var b = @constants([M]f32, 2.0);
var y = @zeros([M]f32);

```

The `@constants` builtin returns a tensor of the specified type,
with all elements initialized to the specified value.
Thus, `x` is initialized as an `N` element tensor of all ones,
and `b` is initialized as an `M` element tensor of all twos.

The `@zeros` builtin is rather obvious. `y` is initialized as an `M`
element tensor of all zeros.

## Compiling and running the program

As with the previous tutorial, we compile and run this code using:

```bash
$ cslc layout.csl --fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 -o out
$ cs_python run.py --name out

```

You should see a `SUCCESS!` message at the end of execution.

## Exercises

`A` is stored row-major in the above code.
How would you rewrite `A_dsd` and the `gemv` function
if `A` were stored column major instead?

## Next

In the next tutorial, we’ll introduce host-to-device `memcpy`,
and copy host-initialized values for `A`, `x`, and `b` onto
the device.

[previous

GEMV Tutorial 1: A Complete Program](csl/tutorials/gemv-01-complete-program.md "previous page")
[next

GEMV Tutorial 3: Memcpy](csl/tutorials/gemv-03-memcpy.md "next page")
