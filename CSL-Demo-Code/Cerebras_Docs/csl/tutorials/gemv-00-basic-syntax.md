---
title: "GEMV Tutorial 0: Basic CSL Syntax"
url: "https://sdk.cerebras.net/csl/tutorials/gemv-00-basic-syntax"
crawled_at: "2026-02-14 23:55:14"
---

# GEMV Tutorial 0: Basic CSL Syntax

> Source: [https://sdk.cerebras.net/csl/tutorials/gemv-00-basic-syntax](https://sdk.cerebras.net/csl/tutorials/gemv-00-basic-syntax)

This tutorial is the first in a series of successive tutorials aimed
at teaching CSL and the SDK by implementing a general matrix-vector product,
or GEMV.

We start by illustrating the syntax of some of the core language constructs.
The code in this example is not a complete program, but it shows
some of the most commonly-used features of the language.

Additionally, note that the complete source code for these tutorial programs
can be found in the CSL examples bundled with the release tarball,
or in the [CSL examples GitHub repository](https://github.com/Cerebras/csl-examples).

## Before you start

These tutorials are intended for a beginner CSL programmer using the Cerebras SDK.
Before you proceed, make sure you installed the Cerebras SDK successfully.
See [Installation and Setup](installation-guide.md#install-guide).

As you proceed through these tutorials, you may find
[A Conceptual View](computing-with-cerebras.md#computing-with-cerebras) and [Host Runtime and Tensor Streaming](tensor-streaming.md#tensor-streaming) helpful references
for some of the basics of the CSL programming model and the `SdkRuntime`
host runtime.

## Learning objectives

After completing this tutorial, you should know:

- Basic syntax of the CSL language
- How to write for and while loops
- How to declare constants and variables
- CSL’s function syntax

## Introduction

CSL is a language for writing programs that run on the Cerebras Wafer Scale Engine (WSE).
The WSE consists of hundreds of thousands of independent processing elements (PEs).
Each PE has room for a small program and some data.
CSL is designed to help you handle the multi-PE nature of the WSE
and the specific challenges of writing dataflow programs for the Cerebras hardware.

The design of CSL is based heavily on [Zig](https://ziglang.org), a general-purpose
language with powerful compile-time programming constructs. Zig was chosen as the
basis for CSL since its compile-time facilities make it possible to write maintainable
yet highly performant code for PEs. Note, however, that while CSL’s syntax and
semantics are very similar to Zig, CSL is not 100% compatible with Zig. Some features
of Zig are not implemented in CSL, and CSL also includes some features that are not
present in Zig.

> **Note**
> Note
> The CSL language is not 100% compatible with Zig, and its compiler does not share
> any code with the Zig compiler. Some of the CSL documentation is derived from the
> Zig documentation. Any bug reports or other feedback on CSL, including its
> documentation, should be directed to Cerebras, and not to the maintainers of Zig
> or to Zig community forums.

### Types

CSL includes some basic types such as:

- `bool` for boolean values
- `i16` and `i32` for 16- and 32-bit signed integers
- `u16` and `u32` for 16- and 32-bit unsigned integers
- `f16` and `f32` for 16- and 32-bit IEEE-754 floating point numbers

In addition to the above, CSL also supports array types and pointer types.

### Functions

Functions are declared using the `fn` keyword. The compiler provides special
functions called *Builtins*, whose names start with `@` and whose
implementation is provided by the compiler. All CSL builtins are described in
[Builtins](csl/language/builtins.md#language-builtins).

### Conditional Statements and Loops

CSL includes support for `if` statements and `while` and `for` loops.

## Example overview

This simple code computes the general matrix-vector product
`y = Ax + b`, where `A` has dimensions `M x N`, `x` is `N x 1`,
and `b` and `y` are `M x 1`.

Our code will store `A` in a one-dimensional array of size `M*N`,
using a row-major ordering.
This computation will be performed with 32-bit arithmetic.

## Writing the CSL

What does our code need to do?

1. Define the dimensions of our matrix
2. Define arrays for holding `A`, `x`, `b`, and `y`
3. Define a function that initialize the arrays
4. Define a function to compute `A*x + b` and store the result in `y`

We explain the sections of our `code.csl` file containing this code below.
This code file, along with all tutorials and examples, are available
in the `csl-extras` directory contained within the SDK tarball.

```csl
// Not a complete program; we include it here for illustrating some syntax

// Every variable must be declared either "const" or "var"
// Const cannot be modified after declaration, but var can

// Constants defining dimensions of our matrix
const M: i16 = 4;
const N: i16 = 6;

// 48 kB of global memory contain A, x, b, y
var A: [M*N]f32; // A is stored in row-major order
var x: [N]f32;
var b: [M]f32;
var y: [M]f32;

// Initialize matrix and vectors
fn initialize() void {
  // for loop with range syntax
  // loops over 0, 1, ...., M*N-1
  // idx stores the loop index
  for (@range(i16, M*N)) |idx| {
    // @as casts idx from i16 to f32
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
}

```

Reading the above CSL code, we can see the following:

### Defining our variables and constants

We first define two constants, `N` and `M`, that give our matrix and
vector dimensions.
For the purposes of this and the next few tutorials, we’ll use `M = 4` and
`N = 6`.

We then declare the arrays `A`, `x`, `b`, and `y`, which hold our
matrix and vectors.
`A` stores `N*M` single-precision floating point elements.
We will store the matrix in `A` in a row-major fashion.
These constants and arrays are declared in global scope: they will be visible
to all functions in this code file.

Note that all data items must explicitly be declared as variables or constants,
with the `var` or `const` keywords.

### Defining our initialize function

Next, we define a function named `initialize` that we will call to
initialize the values stored in `A`, `x`, `b`, and `y`.
`A` will be initialized such that each element `i` holds the value `i`,
all values of `x` will be initialized to `1.0`,
and all values of `b` will be initialized to `2.0`.
`y` will be zero-initialized.

To initialize `A`, we use a `for` loop with the `range` syntax.
`@range(i16, M*N)` returns the sequence of integers
`0, 1, 2, ..., M*N-1`, in `i16`, or half-precision signed, format.
The `for` loop iterates over this sequence of integers, and the variable
`idx` stores the index of the current loop iteration.
On each loop iteration, `@as(f32, idx)` casts the integer
value `idx` to type `f32`, or single precision float, and assigns this
value to the element `A[idx]`.

We also use a range-for loop to initialize all elements of `x` to the value
`1.0`.

To initialize `y` and `b`, we demonstrate the syntax of a while loop with
an assignment expression that acts as a loop index.
At each loop iteration, the variable `i` is incremented by 1.
This assignment expression is executed at the end of each loop iteration.

### Defining our gemv function

The function `gemv` actually computes `y = A*x + b`.
The outer loop iterates over `M`, i.e., over the rows of the matrix.
For each row `i` in the matrix, the inner loop over `N` computes the dot
product of that row with the vector `x` by incrementing the variable `tmp`.
After completing the inner loop, the final value of `y[i]` is computed from
`tmp` and `b[i]`.

This code sample ends with a function named `init_and_compute`, which simply
calls `initialize` followed by `gemv`.

## Next

In the next tutorial, we’ll expand this code to a full program.

[previous

Tutorials](csl/tutorials.md "previous page")
[next

GEMV Tutorial 1: A Complete Program](csl/tutorials/gemv-01-complete-program.md "next page")
