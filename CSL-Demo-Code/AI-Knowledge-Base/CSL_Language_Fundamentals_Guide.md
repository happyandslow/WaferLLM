# Cerebras CSL Language Fundamentals Guide
## SDK Version: 1.4.0

This document covers the core language features of CSL: syntax, type system, compile-time computation, generics, modules, storage classes, and pointer/array usage.

---

## Table of Contents

1. [Variables and Declarations](#1-variables-and-declarations)
2. [Type System](#2-type-system)
3. [Functions and Tasks](#3-functions-and-tasks)
4. [Control Flow](#4-control-flow)
5. [Operators](#5-operators)
6. [Pointers and Arrays](#6-pointers-and-arrays)
7. [Structs and Enumerations](#7-structs-and-enumerations)
8. [Compile-Time Computation (comptime)](#8-compile-time-computation-comptime)
9. [Generics and anytype](#9-generics-and-anytype)
10. [Modules](#10-modules)
11. [Storage Classes (export/extern)](#11-storage-classes-exportextern)
12. [Complete Code Examples](#12-complete-code-examples)
13. [API Quick Reference](#13-api-quick-reference)
14. [Best Practices and Common Pitfalls](#14-best-practices-and-common-pitfalls)
15. [Related Topics](#15-related-topics)

---

## 1. Variables and Declarations

### 1.1 Variable Kinds

CSL has three kinds of variable declarations:

| Keyword | Mutable | Initializer | Notes |
|---------|---------|-------------|-------|
| `const` | No | Required | Immutable after initialization |
| `var` | Yes | Optional | Mutable variable |
| `param` | No | Optional | Compile-time parameter; if no initializer, must be set via module import |

```csl
const ten_i: i16 = 10;
var   ten_f: f16 = 10.0;
param size: i16;              // initialized later via @import_module param_binding
param default_val: f32 = 1.0; // has default, can be overridden
```

### 1.2 Type Inference

The type annotation is optional when an initializer is provided:

```csl
const ten_a: i16 = 10;
const ten_b       = ten_a;  // inferred as i16
```

### 1.3 Alignment and Link Section

```csl
const aligned_var: i16 align(32) = 10;           // 32-byte aligned
var global_var: i16 linksection(".mySection") = 0; // placed in named ELF section
var named_var: i16 linkname("different_name") = 0; // custom ELF symbol name
```

### 1.4 Global Variables

- Global variable initializers are always evaluated at compile time (implicitly comptime).
- Global variables can be used before their declaration (forward references allowed).
- Runtime initialization of global variables is not supported.

---

## 2. Type System

### 2.1 Basic Types

| Category | Types |
|----------|-------|
| Void | `void` |
| Signed integers | `i8`, `i16`, `i32`, `i64` |
| Unsigned integers | `u8`, `u16`, `u32`, `u64` |
| Floating point | `f16`, `f32`, `bf16`, `cb16` |
| Comptime-only | `comptime_int`, `comptime_float`, `type`, `comptime_string`, `comptime_struct` |
| Boolean | `bool` |
| Hardware | `color`, `direction`, `input_queue`, `output_queue`, `local_task_id`, `data_task_id`, `control_task_id`, `ut_id` |

### 2.2 comptime_int and comptime_float

- Integer literals (`10`, `0xff`) have type `comptime_int` — arbitrary precision, compile-time only.
- Float literals (`10.0`) have type `comptime_float` — IEEE double precision, compile-time only.
- Arithmetic on comptime values happens at compile time with no overflow.
- When mixed with fixed-width types, comptime values are coerced to the fixed-width type.

```csl
const thousand = 1000;                    // comptime_int
const ten: i16 = 10;                      // 10 coerced from comptime_int to i16
const hundred: i16 = thousand / 10;       // thousand coerced to i16
// const overflow: i16 = 10000000000000;  // ERROR: overflow
```

### 2.3 FP16 Formats

CSL supports three 16-bit float formats, but **only one can be used at runtime** per program (selected by `--fp16-format`, default `f16`):

| Type | Description | Exponent | Mantissa |
|------|-------------|----------|----------|
| `f16` | IEEE half-precision | 5 bits | 10 bits |
| `cb16` | Cerebras float16 | 6 bits (custom bias) | 9 bits |
| `bf16` | Brain float16 | 8 bits | 7 bits |

Values of non-selected FP16 types must be comptime-known.

### 2.4 The `type` Type

Types are first-class comptime values in CSL:

```csl
const my_type = i16;                // my_type has type `type`
const array = @zeros([10]my_type);  // use type value where type is expected
fn foo() my_type { ... }            // use as return type
```

### 2.5 Type Casting with @as

The `@as` builtin performs explicit type conversion:

```csl
const x: f32 = @as(f32, some_i16_value);
const y: i16 = @as(i16, some_enum_value);
@as(u16, 1) | 0xbeef   // force literal to specific width
```

---

## 3. Functions and Tasks

### 3.1 Function Syntax

```csl
fn foo(arg: i16) i32 { ... }
```

- All parameters are implicitly `const`.
- To modify arguments, use pointer parameters: `fn foo(arg: *i16) void { arg.* = 42; }`
- Pass-by-value vs pass-by-reference is unspecified.

### 3.2 Tasks

```csl
task my_task(arg: i16) void { ... }
```

- Tasks are declared with `task` keyword instead of `fn`.
- Tasks **cannot be called directly** like regular functions.
- Tasks are invoked via the task system (`@activate`, data tasks bound to colors, etc.).

### 3.3 Inline Functions

```csl
inline fn foo(a: i32, b: i32) i32 {
  return a + b;
}
```

- `inline` is **semantic inlining** (not a hint) — the body is expanded at callsite during semantic analysis.
- Comptime-ness of arguments propagates to the return value.
- Cannot take address of `inline fn`, cannot have `linkname`.
- Only use when: (1) you need comptime propagation, or (2) performance measurements demand it.

### 3.4 Function Pointers

```csl
fn foo(a: i16, b: f32) f32 { ... }
var foo_ptr: *const fn(i16, f32) f32 = foo;  // implicit coercion
foo_ptr(42, 3.14);                            // indirect call
```

- Function values coerce to `*const fn(...)` pointers.
- Address-of `&foo` is equivalent to implicit coercion.

### 3.5 comptime Parameters

```csl
fn copy(size: i16, comptime base_type: type,
        dest: [*]base_type, src: [*]base_type) void {
  for (@range(i16, size)) |idx| {
    dest[idx] = src[idx];
  }
}
```

- `comptime` parameters create specialized copies of the function for each value.
- The argument must be comptime-known at the call site.

---

## 4. Control Flow

### 4.1 If-Expression

```csl
if (condition) {
  // ...
} else if (other) {
  // ...
} else {
  // ...
}

// As ternary:
const x: i32 = if (cond) 0 else 1;
```

- If `condition` is comptime-known, the branch not taken is **not semantically checked** (only syntactically valid required).

### 4.2 For-Statement

```csl
// Iterate over a range
for (@range(i16, M)) |element| { ... }

// With index
for (my_array) |element, index| { ... }

// @range variants:
@range(i16, end)                // 0, 1, ..., end-1
@range(i16, start, step, end)   // start, start+step, ...
```

- `break` and `continue` supported.
- Labeled loops with `outer: for ... { break :outer; }`.

### 4.3 While-Statement

```csl
while (condition) : (i += 1) {
  // assignment expression runs at end of each iteration (including continue)
}
```

### 4.4 Switch-Statement

```csl
switch (input) {
  0, 1 => { ... },          // combined cases
  2    => expr,
  else => default_expr,
}
```

- No fall-through (combine cases with commas instead).
- Must cover all possible values or include `else`.
- Can be used as an expression.
- Case values must be comptime-known.

### 4.5 Labeled Blocks

```csl
const x = blk: {
  if (cond) break :blk value1;
  break :blk value2;
};
```

- Blocks can return values via labeled `break`.

---

## 5. Operators

### 5.1 Arithmetic

| Operator | Types | Notes |
|----------|-------|-------|
| `+`, `-`, `*`, `/` | integers, floats | Both operands must have same type (or one is comptime) |
| `%` | integers only | Remainder |
| `-a` (negation) | integers, floats | |

### 5.2 Bitwise

| Operator | Types | Notes |
|----------|-------|-------|
| `<<`, `>>` | integers | Shift amount must be unsigned. `>>` is arithmetic for signed. |
| `&`, `\|`, `^` | integers | AND, OR, XOR |
| `~a` | fixed-width integers | Bitwise NOT |

### 5.3 Logical and Comparison

| Operator | Types | Notes |
|----------|-------|-------|
| `and`, `or` | `bool` | Short-circuit evaluation |
| `!` | `bool` | Logical NOT |
| `==`, `!=` | integers, floats, bool, enum, color, task IDs, etc. | |
| `<`, `>`, `<=`, `>=` | integers, floats | |

> **Important**: Except for `and`/`or`, the evaluation order of operands is **undefined**.

---

## 6. Pointers and Arrays

### 6.1 Array Types

```csl
var arr1d = @zeros([10]u16);         // 1D array
var arr2d = @zeros([10, 10]u16);     // 2D array (no nested array types)
var literal = [3]i16 {1, 2, 3};      // array literal
```

- Element type must not be another array type — use multi-dimensional syntax: `[d1, d2]T`.

### 6.2 Single-Element Pointer (`*T`)

```csl
var x = [2]i16 {0, 1};
var ptr = &x;           // ptr is *[2]i16
ptr.*[1] = 42;          // dereference with .*
```

- Created via address-of operator `&`.
- Dereferenced with `.*`.
- `*const T` is a read-only pointer.

### 6.3 Many-Element Pointer (`[*]T`)

```csl
var array = @zeros([10]i32);
var ptr: [*]i32 = &array;   // coercion from *[10]i32 to [*]i32
ptr[0] = 42;                // indexed access (NOT .*)
```

- Created by coercion from `*[N]T` (1D arrays only).
- Cannot dereference with `.*` — use `[]` indexing.
- Out-of-bounds access is undefined behavior.
- Used extensively for host-exported symbols (`@export_name`, `@export_symbol`).

### 6.4 const Pointers

```csl
const arr = @zeros([10]i32);
var ptr = &arr;               // inferred as *const[10]i32
// ptr.*[0] = 1;              // ERROR: const qualified
```

- Taking address of a `const` variable produces a `*const` pointer.

---

## 7. Structs and Enumerations

### 7.1 Anonymous Structs

```csl
var s = .{.a = 10, .b = 1.0};   // named fields
var t = .{10, 1.0};              // tuple (unnamed fields), access with t[0]
```

### 7.2 Named Structs

```csl
const Complex = struct {
  real_part: f16,
  imag_part: f16,
};

const one = Complex { .real_part = 1.0, .imag_part = 0.0 };
```

- Two named structs are the same type only if defined at the **same source location**.
- Fields are mutable via `.field = value` syntax.

### 7.3 Parameterized Structs (via functions returning types)

```csl
fn Pair(comptime T1: type, comptime T2: type) type {
  return struct { first: T1, second: T2 };
}

const my_pair = Pair(i32, comptime_string) { .first = 42, .second = "hello" };
```

### 7.4 comptime_struct

- Supertype of all struct types; any struct value can be assigned to `comptime_struct`.
- Must be comptime-known. Used mainly for passing anonymous config structs (e.g., param_binding in `@import_module`).

### 7.5 Enumerations

```csl
const Colors = enum(u16) { red, white, blue };
const fav = Colors.red;

// Casting
@comptime_assert(@as(u16, Colors.red) == 0);
@comptime_assert(@as(Colors, 1) == Colors.white);
```

- Underlying type must be a fixed-precision integer.
- Values auto-assigned 0, 1, 2, ... unless explicitly set.
- Can cast to/from numeric types with `@as`.
- Two enum types are the same only if same structure **and** same source location.

---

## 8. Compile-Time Computation (comptime)

### 8.1 comptime Variables

```csl
task foo() void {
  comptime var bitmap: u16 = 0xffff;
  bitmap &= 0x0400;  // happens at compile time, no runtime cost
}
```

- Must be declared inside a block or function (global `const` is already implicitly comptime).
- No runtime footprint, address cannot be taken.

### 8.2 comptime Expressions

```csl
param bar: u16;
fn foo(arg: u16) u16 { return arg * 2; }

task goo() void {
  var myVar = comptime foo(bar);  // function call evaluated at compile time
}
```

### 8.3 comptime Control Flow

- If predicate is comptime-known, the not-taken branch is pruned (not semantically checked).
- Key pattern for WSE-2/WSE-3 conditional compilation:

```csl
if (@is_arch("wse3")) {
  @initialize_queue(iq, .{ .color = color });
}
// On WSE-2, @initialize_queue doesn't exist — but the branch is pruned
```

### 8.4 comptime Blocks and Loops

```csl
comptime {
  var sum: u16 = 0;
  for ([3]u16 {1, 2, 3}) |value| {
    sum += value;
  }
  @comptime_assert(sum == 6);
}
```

- All variables and expressions inside `comptime { }` blocks are implicitly comptime.

### 8.5 comptime Initialization Pattern

```csl
param size: u16;

// Global initializers are implicitly comptime
const identity = createIdentityMatrix();

fn createIdentityMatrix() [size, size]f16 {
  var result = @zeros([size, size]f16);
  var i: u16 = 0;
  while (i < size) : (i += 1) {
    result[i, i] = 1.0;
  }
  return result;
}
```

- No runtime cost — `createIdentityMatrix()` runs on host during compilation.

### 8.6 @comptime_assert

```csl
@comptime_assert(condition);
@comptime_assert(condition, "error message");
```

- Fails compilation if condition is false.
- Used for type constraints, parameter validation, and invariant checking.

---

## 9. Generics and anytype

### 9.1 Generic Functions via comptime type

```csl
fn identity(comptime T: type, x: T) T {
  return x;
}

var result = identity(i16, 42);
```

- Functions are monomorphized — a copy is compiled for each set of type arguments.

### 9.2 anytype (Implicit Generics)

```csl
fn abs(x: anytype) @type_of(x) {
  const T = @type_of(x);
  if (x < @as(T, 0)) return -x;
  return x;
}
```

- `anytype` can only appear as function parameter type.
- Creates a specialized version for each type the function is called with.
- Use `@type_of(x)` to relate parameter types and return type.

### 9.3 Constraining Type Parameters

```csl
fn safe_abs(x: anytype) @type_of(x) {
  const T = @type_of(x);
  @comptime_assert(
    @is_same_type(T, f16) or @is_same_type(T, f32),
    "x must be a float"
  );
  if (x < @as(T, 0)) return -x;
  return x;
}
```

### 9.4 Specializing Logic by Type

```csl
fn sign(x: anytype) @type_of(x) {
  const T = @type_of(x);
  if (comptime is_signed(T)) {
    if (x < @as(T, 0)) return @as(T, -1);
  }
  if (x > @as(T, 0)) return @as(T, 1);
  return x;
}
```

- Guard type-specific branches with `if (comptime ...)` to avoid compile errors for invalid types.

### 9.5 Generic Structs

```csl
fn Point(comptime T: type) type {
  return struct { x: T, y: T };
}

const origin = Point(u16) { .x = 0, .y = 0 };
```

---

## 10. Modules

### 10.1 @import_module

```csl
@import_module(filename)
@import_module(filename, param_binding)
```

- `filename`: comptime string, relative to importer location.
- `param_binding`: anonymous struct setting `param`/`color` values in the imported module.
- Returns `imported_module` type, must be assigned to a `const` global.

### 10.2 Standard Library Imports

Standard libraries use angle brackets:

```csl
const math   = @import_module("<math>");
const random = @import_module("<random>");
const tsc    = @import_module("<time>");
const debug  = @import_module("<debug>");
```

### 10.3 Layout-to-PE Module Import

The layout file passes parameters to PE programs:

```csl
// layout.csl
@set_tile_code(pe_x, pe_y, "pe_program.csl", .{
  .memcpy_params = memcpy.get_params(pe_x),
  .M = M,
  .N = N,
});
```

```csl
// pe_program.csl
param memcpy_params: comptime_struct;
param M: i16;
param N: i16;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
```

### 10.4 Key Rules

- Importing the same file twice creates **two independent copies**.
- Layout blocks in imported modules are ignored.
- Module members accessed via `.` operator: `math.sqrt_f32(x)`.
- Module values must be `comptime`.

---

## 11. Storage Classes (export/extern)

### 11.1 export

Makes a symbol accessible to other object files and the Python `ELFLoader`:

```csl
export var x: i16 = 42;
export fn f() void { ... }
```

- Guaranteed not to be eliminated from compiled object.
- Required for symbols accessed via `ELFLoader` in Python host code.

### 11.2 extern

Declares that a symbol will be defined in another linked object:

```csl
extern var y: i16;          // no initializer allowed
extern fn g(f16, f16) f16;  // no body allowed
```

### 11.3 Export-Compatible Types

Only certain types are allowed for `export`/`extern` declarations:

- `i8`-`i64`, `u8`-`u64`, `@fp16()`, `f32`, `bool`, `color`
- Arrays `[N]T` where `T` is export-compatible
- Pointers `*T`, `*const T` where `T` is export-compatible
- `*const fn(...)` (function pointers, must be const)
- `enum` types

**Not** export-compatible: comptime-only types, `struct`, DSD/DSR, `direction`, `void` (except as fn return type).

### 11.4 linkname with Modules

```csl
// Avoid symbol name clashes when importing same module twice
const m1 = @import_module("sub.csl", .{ .sym_name = "m1_x" });
const m2 = @import_module("sub.csl", .{ .sym_name = "m2_x" });

// sub.csl
param sym_name: comptime_string;
export var x: i16 linkname(sym_name);
```

---

## 12. Complete Code Examples

### 12.1 Basic Syntax — GEMV (gemv-00)

Pure syntax example (not a runnable program):

```csl
// Source: examples/tutorials/gemv-00-basic-syntax/code.csl
const M: i16 = 4;
const N: i16 = 6;

var A: [M*N]f32;
var x: [N]f32;
var b: [M]f32;
var y: [M]f32;

fn initialize() void {
  for (@range(i16, M*N)) |idx| {
    A[idx] = @as(f32, idx);      // @as for type casting
  }
  for (@range(i16, N)) |j| {
    x[j] = 1.0;
  }
  var i: i16 = 0;
  while (i < M) : (i += 1) {    // while with assignment expression
    b[i] = 2.0;
    y[i] = 0.0;
  }
}

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

**Key patterns**: `const`/`var` declarations, `for` with `@range`, `while` with iterator, `@as` casting.

### 12.2 Compile-Time Params (gemv-04)

Layout passes params to PE program:

```csl
// layout.csl
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

  @export_name("A", [*]f32, true);
  @export_name("y", [*]f32, false);
  @export_name("init_and_compute", fn()void);
}
```

```csl
// pe_program.csl
param memcpy_params: comptime_struct;
param M: i16;
param N: i16;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

var A: [M*N]f32;
var x: [N]f32;
var y = @zeros([M]f32);

// ... DSD operations ...

comptime {
  @export_symbol(A_ptr, "A");
  @export_symbol(init_and_compute);
}
```

**Key patterns**: `param` declarations, `@import_module` with param_binding, `comptime { @export_symbol(...) }`.

### 12.3 Pointers and Arrays (topic-01)

```csl
// Source: examples/tutorials/topic-01-arrays-and-pointers/pe_program.csl
fn increment_and_sum(data_ptr: *[3]i16, result_ptr: *i16) void {
  (data_ptr.*)[0] += 1;   // dereference then index
  (data_ptr.*)[1] += 1;
  (data_ptr.*)[2] += 1;
  result_ptr.* = (data_ptr.*)[0] + (data_ptr.*)[1] + (data_ptr.*)[2];
}

fn f_run() void {
  var data = [3]i16 { 1, 2, 3 };
  increment_and_sum(&data, &result[0]);
  sys_mod.unblock_cmd_stream();
}
```

**Key patterns**: `*[3]i16` (pointer to array), `.*` dereference, `&` address-of.

### 12.4 Standard Libraries (topic-02)

```csl
// Source: examples/tutorials/topic-02-libraries/pe_program.csl
const random = @import_module("<random>");
const tsc    = @import_module("<time>");
const math   = @import_module("<math>");

fn f_run() void {
  tsc.enable_tsc();
  tsc.get_timestamp(&startBuffer);

  while (idx < iterations) : (idx += 1) {
    var x = random.random_f32(-1.0, 1.0);
    var y = random.random_f32(-1.0, 1.0);
    var dist = math.sqrt_f32(x * x + y * y);
    if (dist <= 1.0) { hitCount += 1; }
  }

  tsc.get_timestamp(&finishBuffer);
  result[0] = 4.0 * @as(f32, hitCount) / @as(f32, iterations);
  sys_mod.unblock_cmd_stream();
}
```

**Key patterns**: library imports with `<>`, `@as` for u32→f32 casting, `while` loop.

---

## 13. API Quick Reference

### 13.1 Type System Builtins

| Builtin | Purpose |
|---------|---------|
| `@as(T, value)` | Cast value to type T |
| `@type_of(expr)` | Get the type of an expression |
| `@is_same_type(T1, T2)` | Check if two types are identical (returns bool) |
| `@bitcast(T, value)` | Reinterpret bits as type T |
| `@fp16()` | Returns the selected FP16 type |

### 13.2 Array/Memory Builtins

| Builtin | Purpose |
|---------|---------|
| `@zeros([N]T)` | Create zero-initialized array |
| `@constants([N]T, value)` | Create array filled with a constant value |
| `@range(T, end)` | Range 0..end-1 of type T |
| `@range(T, start, step, end)` | Range with start/step/end |
| `@get_array(comptime_string)` | Convert string to byte array |

### 13.3 Compile-Time Builtins

| Builtin | Purpose |
|---------|---------|
| `@comptime_assert(cond)` | Fail compilation if condition is false |
| `@comptime_assert(cond, msg)` | With custom error message |
| `@comptime_print(...)` | Print at compile time (debug) |
| `@is_arch("wse2"/"wse3")` | Check target architecture at comptime |
| `@strcat(s1, s2, ...)` | Concatenate comptime strings |
| `@strlen(s)` | Length of comptime string |
| `@concat_structs(s1, s2)` | Merge two comptime structs |

### 13.4 Module/Export Builtins

| Builtin | Purpose |
|---------|---------|
| `@import_module(filename)` | Import a module |
| `@import_module(filename, params)` | Import with parameter binding |
| `@export_name(name, type, writable)` | Export symbol name in layout |
| `@export_symbol(ptr, name)` | Export symbol in PE program |

---

## 14. Best Practices and Common Pitfalls

### Best Practices

1. **Use `param` for PE-configurable values**: Pass matrix dimensions, PE coordinates, colors, etc. as `param` from layout to PE programs.

2. **Prefer `const` over `var`**: Use `var` only when mutation is needed. `const` enables better compiler optimizations.

3. **Use `@range` for iteration**: Prefer `for (@range(i16, N)) |i| { ... }` over manual `while` loops for cleaner code.

4. **Use `comptime` for initialization**: Compute lookup tables, identity matrices, and routing configs at compile time.

5. **Use `@is_arch` for WSE-2/WSE-3 compatibility**: Guard architecture-specific code with `if (@is_arch("wse3")) { ... }`.

6. **Use `anytype` for reusable functions**: When the function logic is type-agnostic, use `anytype` + `@type_of` instead of duplicating code.

7. **Use `@comptime_assert` to constrain generics**: Provide clear error messages when generic functions receive unsupported types.

### Common Pitfalls

1. **Calling a task directly**: Tasks cannot be called like functions. Use the task system (`@activate`, `@bind_data_task`, etc.) instead.

2. **Mixing comptime_int with float operations**: `comptime_int` values don't auto-convert to floats. Use `@as(T, value)` explicitly:
   ```csl
   // ERROR: 0 is comptime_int, not comparable with f16
   if (x > 0) { ... }
   // CORRECT:
   if (x > @as(f16, 0)) { ... }
   ```

3. **Using non-selected FP16 types at runtime**: Only one FP16 type (default `f16`) can be used at runtime. Others must be comptime-known.

4. **Undefined operator evaluation order**: Except for `and`/`or`, operand evaluation order is undefined. Don't rely on side effects in sub-expressions.

5. **Multidimensional array syntax**: CSL uses `[d1, d2]T`, not `[d1][d2]T`. Nested array types are not allowed.

6. **Dereferencing `[*]T` with `.*`**: Many-element pointers use `[]` indexing, not `.*`. Only single-element pointers (`*T`) use `.*`.

7. **Forgetting `comptime` in generic specialization**: When guarding type-specific branches, the `if` condition must be `comptime` to prune the invalid branch:
   ```csl
   // ERROR: branch still compiled for unsigned types
   if (is_signed(T)) { if (x < 0) ... }
   // CORRECT:
   if (comptime is_signed(T)) { if (x < 0) ... }
   ```

8. **Global variable runtime initialization**: Global variable initializers are always comptime. You cannot call runtime functions in a global initializer.

9. **export/extern type restrictions**: Only export-compatible types (integers, floats, pointers, enums) can be used with `export`/`extern`. Structs and comptime types are not allowed.

---

## 15. Related Topics

- **DSDs and DSD operations**: See `CSL_DSD_Complete_Guide.md` — mem1d_dsd, fabin/fabout_dsd, @get_dsd, @fmacs, @fadds
- **Builtins reference**: See `CSL_Builtins_Reference_Guide.md` — comprehensive DSD operations, task builtins, queue/color config
- **Async programming and tasks**: See `CSL_Async_Programming_Guide.md` — task system, @activate, @block/@unblock, microthreads
- **Host-device communication**: See `CSL_Host_Device_Communication_Guide.md` — memcpy, SdkRuntime, @export_name/@export_symbol patterns
- **Routing and colors**: See `CSL_Routing_Colors_Switches_Guide.md` — @get_color, @set_color_config, switches, filters
- **Standard libraries**: `Cerebras_Docs/csl/language/libraries.md` — math, random, time, debug, collectives_2d
- **Inline assembly**: `Cerebras_Docs/csl/language/syntax.md` (Inline assembly section) — `asm volatile`, constraints, clobbers
