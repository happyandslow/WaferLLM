# Cerebras CSL Standard Libraries Complete Guide
## SDK Version: 1.4.0

This document covers all CSL standard libraries: their purpose, API, import patterns, usage examples, and key implementation details.

---

## Table of Contents

1. [Overview](#1-overview)
2. [math — Math Functions](#2-math--math-functions)
3. [random — Random Number Generation](#3-random--random-number-generation)
4. [time — Timestamp Counters](#4-time--timestamp-counters)
5. [debug — Trace Buffers](#5-debug--trace-buffers)
6. [simprint — Simulator Printing](#6-simprint--simulator-printing)
7. [layout — PE Coordinate Access](#7-layout--pe-coordinate-access)
8. [control — Control Wavelet Construction](#8-control--control-wavelet-construction)
9. [complex — Complex Arithmetic](#9-complex--complex-arithmetic)
10. [directions — Direction Manipulation](#10-directions--direction-manipulation)
11. [malloc — Arena Allocator](#11-malloc--arena-allocator)
12. [message_passing — Point-to-Point Communication (WSE-3)](#12-message_passing--point-to-point-communication-wse-3)
13. [collectives_2d — MPI-like Collectives](#13-collectives_2d--mpi-like-collectives)
14. [dsd_ops — Type-Dispatched DSD Operations](#14-dsd_ops--type-dispatched-dsd-operations)
15. [Other Libraries](#15-other-libraries)
16. [Best Practices](#16-best-practices)

---

## 1. Overview

CSL standard libraries are imported using angle brackets:

```csl
const math = @import_module("<math>");
const sys = @import_module("<memcpy/memcpy>", memcpy_params);
```

### Import Convention

| Library | Import | Params Required? |
|---|---|---|
| `math` | `@import_module("<math>")` | No |
| `random` | `@import_module("<random>")` | No |
| `time` | `@import_module("<time>")` | No |
| `debug` | `@import_module("<debug>", .{.key=..., .buffer_size=...})` | **Yes** |
| `simprint` | `@import_module("<simprint>")` | Optional `.enable` |
| `layout` | `@import_module("<layout>")` | No |
| `control` | `@import_module("<control>")` | No |
| `complex` | `@import_module("<complex>")` | No (use types directly) |
| `directions` | `@import_module("<directions>")` | No |
| `malloc` | `@import_module("<malloc>", .{.buffer_num_words=...})` | **Yes** |
| `message_passing` | `@import_module("<message_passing>", .{...})` | **Yes** (WSE-3 only) |
| `collectives_2d` | `@import_module("<collectives_2d/pe>")` | Via params module |
| `dsd_ops` | `@import_module("<dsd_ops>")` | No |
| `memcpy` | `@import_module("<memcpy/memcpy>", memcpy_params)` | **Yes** |

---

## 2. math — Math Functions

**Import**: `const math = @import_module("<math>");`

### 2.1 Constants

```csl
const PI: comptime_float = 3.1415926535897932384;
const E: comptime_float = 2.7182818284590452354;
```

### 2.2 IEEE Floating-Point Utilities

| Function | Signature | Description |
|---|---|---|
| `POSITIVE_INF` | `fn(comptime T: type) T` | Positive infinity for T (f16/f32) |
| `NEGATIVE_INF` | `fn(comptime T: type) T` | Negative infinity |
| `NaN` | `fn(comptime T: type) T` | Quiet NaN |
| `isNaN` | `fn(x: anytype) bool` | True if x is NaN |
| `isInf` | `fn(x: anytype) bool` | True if x is ±Inf |
| `isFinite` | `fn(x: anytype) bool` | True if x is finite |
| `isSignaling` | `fn(x: anytype) bool` | True if x is signaling NaN |
| `signbit` | `fn(x: anytype) bool` | True if sign bit is set |

All accept f16, cb16, bf16, or f32 arguments. Non-generic `_f16` and `_f32` suffixed versions also exist.

### 2.3 Rounding & Scaling

| Function | Supported Types | Description |
|---|---|---|
| `ceil(x)` | f16, cb16, bf16, f32 | Ceiling |
| `floor(x)` | f16, cb16, bf16, f32 | Floor |
| `fscale(f, s)` | f16/f32 + i16 | Scale by 2^s (hardware op) |

### 2.4 Arithmetic Utilities

| Function | Supported Types | Description |
|---|---|---|
| `abs(x)` | f16, cb16, bf16, f32, i8, i16, i32, i64, u8, u16, u32, u64 | Absolute value |
| `max(x, y)` | Same as abs | Maximum |
| `min(x, y)` | Same as abs | Minimum |
| `sign(x)` | Same as abs | Sign (-1, 0, or +1) |

### 2.5 Transcendental Functions

| Function | Supported Types | Runtime Only? | Notes |
|---|---|---|---|
| `sin(x)` | f16, f32 | Yes | **Inaccurate when abs(x) >= 16384π (~51472)** |
| `cos(x)` | f16, f32 | Yes | Same accuracy limitation |
| `tanh(x)` | f16, cb16, bf16, f32 | No | Piecewise polynomial approximation |
| `sig(x)` | f16, cb16, bf16, f32 | No | Sigmoid: 0.5 * tanh(0.5*x) + 0.5 |
| `exp(x)` | f16, f32 | Yes | Table + polynomial |
| `log(x)` | f16, f32 | Yes | Polynomial after normalization |
| `pow(x, y)` | f16, f32 | Yes | exp(y*log(x)) |
| `sqrt(x)` | f16, f32 | Yes | Newton's method via inv_sqrt |
| `invsqrt(x)` | f16, f32 | Yes | Fast inverse sqrt |
| `inv(x)` | f16, f32 | Yes | Newton's method reciprocal |

### 2.6 Usage Example

```csl
const math = @import_module("<math>");

fn distance(x0: f32, y0: f32, x1: f32, y1: f32) f32 {
  return math.sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
}
```

**Source**: `source-code/csl_libraries/math.csl` (1657 lines)

---

## 3. random — Random Number Generation

**Import**: `const random = @import_module("<random>");`

### 3.1 API

| Function | Signature | Description |
|---|---|---|
| `set_global_prng_seed` | `fn(seed: u32) void` | Initialize all hardware PRNGs with seed |
| `random_f16` | `fn(lower: @fp16(), upper: @fp16()) @fp16()` | Uniform random in [lower, upper) |
| `random_f32` | `fn(lower: f32, upper: f32) f32` | Uniform random in [lower, upper) |
| `random_pow_u32` | `fn(pow: u16) u32` | Random integer in [0, 2^pow) |
| `random_normal_f32` | `fn() f32` | Standard normal N(0,1) via Box-Muller |

### 3.2 Implementation Notes

- Uses hardware PRNG via `@random16()` intrinsic
- `random_f16` masks mantissa bits to construct uniform float in [1.0, 2.0) then scales
- `random_f32` combines two 16-bit random values into 32-bit mantissa
- `random_normal_f32` uses Box-Muller: `sqrt(-2*log(u1)) * cos(2π*u2)`

### 3.3 Usage Example

```csl
const random = @import_module("<random>");

fn monte_carlo_pi(iterations: u32) f32 {
  var hits: u32 = 0;
  var i: u32 = 0;
  while (i < iterations) : (i += 1) {
    var x = random.random_f32(-1.0, 1.0);
    var y = random.random_f32(-1.0, 1.0);
    if (x * x + y * y <= 1.0) hits += 1;
  }
  return 4.0 * @as(f32, hits) / @as(f32, iterations);
}
```

**Source**: `source-code/csl_libraries/random.csl`  
**Tutorial**: `examples/tutorials/topic-02-libraries/pe_program.csl`

---

## 4. time — Timestamp Counters

**Import**: `const tsc = @import_module("<time>");`

### 4.1 API

| Function | Signature | Description |
|---|---|---|
| `enable_tsc` | `fn() void` | Enable timestamp counter |
| `disable_tsc` | `fn() void` | Disable timestamp counter |
| `get_timestamp` | `fn(result: *[3]u16) void` | Read 48-bit timestamp into 3×u16 array |
| `reset_tsc_counter` | `fn() void` | Reset counter to zero |
| `get_perf_cntr` | `fn(result: *[3]u16, id: i16) void` | Read performance counter by ID |

### 4.2 Constants

```csl
const tsc_size_words: u16;  // Number of u16 words per timestamp (3)
```

### 4.3 Usage Pattern

```csl
const tsc = @import_module("<time>");

var startBuffer = @zeros([tsc.tsc_size_words]u16);
var finishBuffer = @zeros([tsc.tsc_size_words]u16);

fn timed_operation() void {
  tsc.enable_tsc();
  tsc.get_timestamp(&startBuffer);
  // ... work ...
  tsc.get_timestamp(&finishBuffer);
}
```

**Source**: `source-code/csl_libraries/time.csl`  
**Tutorial**: `examples/tutorials/topic-02-libraries/pe_program.csl`

---

## 5. debug — Trace Buffers

**Import**: `const trace = @import_module("<debug>", .{ .key = "trace_name", .buffer_size = 100 });`

### 5.1 Import Parameters

| Parameter | Type | Description |
|---|---|---|
| `key` | `comptime_string` | **Required**. Unique name for this trace buffer |
| `buffer_size` | `comptime_int` | **Required**. Max entries in trace buffer |

Multiple instances can be imported with different keys to create separate trace channels.

### 5.2 API

| Function | Signature | Description |
|---|---|---|
| `trace` | `fn(x: anytype) void` | Generic trace (auto-detect type) |
| `trace_bool` | `fn(x: bool) void` | Record boolean |
| `trace_u8/i8` | `fn(x: u8/i8) void` | Record 8-bit value |
| `trace_u16/i16` | `fn(x: u16/i16) void` | Record 16-bit value |
| `trace_f16` | `fn(x: @fp16()) void` | Record f16 |
| `trace_u32/i32` | `fn(x: u32/i32) void` | Record 32-bit value |
| `trace_f32` | `fn(x: f32) void` | Record f32 |
| `trace_string` | `fn(comptime str: comptime_string) void` | Record compile-time string |
| `trace_timestamp` | `fn() void` | Record current timestamp (uses `<time>`) |

### 5.3 Reading Traces from Host

```python
from cerebras.sdk.debug.debug_util import debug_util

debug_mod = debug_util(dirname, cmaddr=args.cmaddr)
core_offset_x = 4  # Fabric offset (default for memcpy programs)
core_offset_y = 1

# Read trace by key name
trace_output = debug_mod.read_trace(core_offset_x + pe_x, core_offset_y, 'trace_name')
```

### 5.4 Usage Example

```csl
const trace = @import_module("<debug>", .{ .key = "values", .buffer_size = 100 });
const times = @import_module("<debug>", .{ .key = "timing", .buffer_size = 50 });

task my_task(data: u32) void {
  times.trace_timestamp();
  trace.trace_string("Received data");
  trace.trace_u32(data);
  times.trace_timestamp();
}
```

**Source**: `source-code/csl_libraries/debug.csl`  
**Tutorial**: `examples/tutorials/topic-12-debug-library/`

---

## 6. simprint — Simulator Printing

**Import**: `const prt = @import_module("<simprint>");`

### 6.1 Import Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable` | `bool` | `true` | Set to `false` to disable all output (zero overhead) |

### 6.2 Basic Printing API

| Function | Description |
|---|---|
| `print_string(comptime s)` | Print compile-time string |
| `print_u16_binary/decimal/hex(x)` | Print u16 in different formats |
| `print_f16(x)` | Print f16 |
| `print_u32_binary/decimal/hex(x)` | Print u32 in different formats |
| `print_f32(x)` | Print f32 |

**Important**: Output is buffered and only flushed on newline character (`\n`).

### 6.3 Format String API

| Function | Description |
|---|---|
| `fmt(fstr, args)` | Print formatted string with newline |
| `fmt_no_nl(fstr, args)` | Print formatted string without newline |
| `fmt_with_coords(fstr, args)` | Print with PE coordinates prefix: `PE(x,y): ...` |

Format specifiers: `{d}` decimal, `{X}` hex, `{b}` binary, `{f}` float.

### 6.4 Usage Example

```csl
const prt = @import_module("<simprint>");

task recv_task(data: u32) void {
  prt.fmt("Received: {d}, hex: {X}", .{ data, data });
  // Output: "Received: 42, hex: 2A\n"
  
  prt.fmt_with_coords("got {d}", .{ data });
  // Output: "PE(3,2): got 42\n"
}
```

### 6.5 Disabling Output

```csl
// Zero-overhead disable — all calls compile to no-ops
const prt = @import_module("<simprint>", .{ .enable = false });
```

**Source**: `source-code/csl_libraries/simprint.csl`  
**Tutorial**: `examples/tutorials/topic-13-simprint/`

---

## 7. layout — PE Coordinate Access

**Import**: `const layout_mod = @import_module("<layout>");`

### 7.1 API

| Function | Return Type | Description |
|---|---|---|
| `get_x_coord()` | `u16` | PE's x-coordinate within the rectangle (0-based) |
| `get_y_coord()` | `u16` | PE's y-coordinate within the rectangle (0-based) |

### 7.2 Implementation

Reads hardware fabric coordinates and subtracts the rectangle offset:

```csl
fn get_x_coord() u16 {
  return tile_config.get_fabric_coord(tile_config.fabric_coord.X)
       - @get_rectangle().offsets.width;
}
```

### 7.3 Common Use

```csl
const layout_mod = @import_module("<layout>");

fn get_pe_id() u16 {
  return layout_mod.get_y_coord() * @get_rectangle().width + layout_mod.get_x_coord();
}
```

**Source**: `source-code/csl_libraries/layout.csl`  
**Tutorial**: `examples/tutorials/gemv-07-routes-2/`

---

## 8. control — Control Wavelet Construction

**Import**: `const ctrl = @import_module("<control>");`

### 8.1 Types and Constants

```csl
const MAX_CMDS = 8;  // Max switch commands per control wavelet

const opcode = enum(u32) {
  NOP = 0,         // No operation
  SWITCH_ADV = 1,  // Advance switch position
  SWITCH_RST = 2,  // Reset switch to initial position
  TEARDOWN = 3     // Teardown the color
};
```

### 8.2 API

| Function | Description |
|---|---|
| `encode_control_task_payload(entrypoint)` | Encode payload activating a control task on all receivers |
| `encode_single_payload(cmd, ce_ignore, entrypoint, data)` | Encode one switch command + optional control task + 16-bit data |
| `encode_payload(N, cmds, ce_ignore, ce_ignore_remaining, entrypoint)` | Encode up to 8 switch commands + optional control task |

### 8.3 Usage Example

```csl
const ctrl = @import_module("<control>");

const comm_out_dsd = @get_dsd(fabout_dsd, .{
  .extent = 1, .fabric_color = comm,
  .control = true, .output_queue = comm_oq,
});

// Send control wavelet that advances switch + activates task
@mov32(comm_out_dsd, ctrl.encode_single_payload(
  ctrl.opcode.SWITCH_ADV, false, my_ctrl_id, 0));
```

### 8.4 Key Points

- `ce_ignore = true`: Router processes command but doesn't forward to CE
- `ce_ignore = false`: CE receives the control wavelet (can activate task)
- Only `encode_single_payload` supports a 16-bit `data` argument
- `encode_payload` uses those bits for additional switch commands instead
- Pass `{}` as `entrypoint` if no control task activation is needed

**Source**: `source-code/csl_libraries/control.csl`  
**Tutorial**: `examples/tutorials/topic-06-switches/`

---

## 9. complex — Complex Arithmetic

**Import**: Use types and functions from `@import_module("<complex>")`

### 9.1 Types

```csl
fn complex(comptime T: type) type;        // Generic: struct { real: T, imag: T }
const complex_32 = complex(f16);           // f16 complex
const complex_64 = complex(f32);           // f32 complex
```

### 9.2 Constructors

```csl
fn get_complex(r: anytype, i: @type_of(r)) complex(@type_of(r));
fn get_complex_32(r: f16, i: f16) complex_32;
fn get_complex_64(r: f32, i: f32) complex_64;
```

### 9.3 Operations

| Function | Description |
|---|---|
| `add_complex(x, y)` | x + y (component-wise) |
| `subtract_complex(x, y)` | x - y (component-wise) |
| `multiply_complex(x, y)` | x × y = (re*re - im*im, re*im + im*re) |

Generic versions and `_32`/`_64` non-generic versions are provided.

### 9.4 Usage in FFT

```csl
// From examples/benchmarks/fft-1d-2d/fft.csl
// Complex multiplication broken down for DSD operations:
@fmuls(resultDSD, sourceDSD, c1a);                   // real part: re * twiddle_re
@fmacs(real_dstDSD, real_dstDSD, imag_srcDSD, c1b);  // real part: -= im * twiddle_im
@fmacs(imag_dstDSD, imag_dstDSD, real_srcDSD, -c1b); // imag part: += re * twiddle_im
```

**Source**: `source-code/csl_libraries/complex.csl`  
**Benchmark**: `examples/benchmarks/fft-1d-2d/`

---

## 10. directions — Direction Manipulation

**Import**: `const dirs = @import_module("<directions>");`

### 10.1 API

| Function | Description |
|---|---|
| `rotate_clockwise(d)` | N→E→S→W→N |
| `rotate_counterclockwise(d)` | N→W→S→E→N |
| `flip_vertical(d)` | N↔S (E,W unchanged) |
| `flip_horizontal(d)` | E↔W (N,S unchanged) |
| `flip(d)` | Both vertical + horizontal flip (N↔S, E↔W) |

All functions: `fn(d: direction) direction`. RAMP input returns RAMP for all operations.

**Source**: `source-code/csl_libraries/directions.csl`  
**Benchmark**: `examples/benchmarks/game-of-life/`

---

## 11. malloc — Arena Allocator

**Import**: `const mem = @import_module("<malloc>", .{ .buffer_num_words = 1024 });`

### 11.1 Import Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `buffer_num_words` | `comptime_int` | Required | Size of backing buffer in 16-bit words |
| `asserts_enabled` | `bool` | `false` | Enable runtime bounds checking |

### 11.2 API

| Function | Signature | Description |
|---|---|---|
| `malloc` | `fn(comptime T: type, num_values: u16) [*]T` | Allocate array of T |
| `malloc_i16/u16/f16/i32/u32/f32` | `fn(num_values: u16) [*]T` | Type-specific malloc |
| `has_enough_space` | `fn(comptime T: type, num_values: u16) bool` | Check if allocation would succeed |
| `has_enough_words` | `fn(num_words: u16) bool` | Check by raw word count |
| `free` | `fn() void` | Reset allocator (frees all) |

### 11.3 Key Details

- **Arena allocator**: All allocations come from a single contiguous buffer
- `free()` resets the entire arena — no individual deallocation
- Supported types: f16, f32, i16, i32, i64, u16, u32, u64
- Handles alignment automatically for 32-bit and 64-bit types

### 11.4 Usage Example

```csl
const mem = @import_module("<malloc>", .{ .buffer_num_words = 4096 });

fn allocate_workspace() void {
  var data = mem.malloc(f32, 256);  // [*]f32
  var indices = mem.malloc(u16, 128);  // [*]u16
  // ... use data and indices ...
  mem.free();  // Reset entire arena
}
```

**Source**: `source-code/csl_libraries/malloc.csl`  
**Benchmark**: `examples/benchmarks/spmv-hypersparse/`

---

## 12. message_passing — Point-to-Point Communication (WSE-3)

**Import**:
```csl
const mp = @import_module("<message_passing>", .{
  .input_queue_id = 2,
  .output_queue_id = 3,
  .enable_asserts = false,  // optional
});
```

### 12.1 Requirements

- **WSE-3 only** (`@comptime_assert(@is_arch("wse3"))`)
- Uses colors 14-18 by default (reserved for library use)
- Requires `init()` before any send/receive

### 12.2 API

| Function | Description |
|---|---|
| `init()` | Initialize library (must call before send/receive) |
| `send_message(dest_col, dest_row, buf, len, on_term)` | Send message to target PE with completion callback |
| `recv_message(buf, on_term)` | Receive message with completion callback |

### 12.3 Exported Constants

```csl
const mp_input_queue: input_queue;   // The allocated input queue
const mp_output_queue: output_queue; // The allocated output queue
const mp_comm_color: color;          // Communication color (bank_id + 2)
```

### 12.4 Completion Callbacks

Both `send_message` and `recv_message` accept `on_term` — a comptime struct specifying what to do on completion (e.g., `.activate = some_task_id`).

**Source**: `source-code/csl_libraries/message_passing.csl`  
**Doc**: `Cerebras_Docs/csl/language/libraries_wse3.md`

---

## 13. collectives_2d — MPI-like Collectives

**Import**:
```csl
const params = @import_module("<collectives_2d/params>", .{ /* layout params */ });
const collective = @import_module("<collectives_2d/pe>", .{ /* pe params */ });
```

### 13.1 Operations

| Operation | Description |
|---|---|
| `broadcast` | One PE broadcasts to all PEs in the group |
| `scatter` | One PE distributes different data to each PE |
| `gather` | All PEs send data to one PE |
| `reduce_fadds` | Sum reduction across PEs |

### 13.2 Typical Usage

Set up in layout with `<collectives_2d/params>`, then use the PE module for collective calls. The library handles routing and synchronization internally.

**Doc**: `Cerebras_Docs/csl/language/libraries.md`  
**Tutorial**: `examples/tutorials/topic-11-collectives-2d/`  
**Benchmark**: `examples/benchmarks/gemm-collectives_2d/`

---

## 14. dsd_ops — Type-Dispatched DSD Operations

**Import**: `const ops = @import_module("<dsd_ops>");`

### 14.1 API

Wrappers that select the appropriate builtin based on type arguments:

| Function | Maps To |
|---|---|
| `mov(T, dst, src, config)` | `@fmovs/@fmovh/@mov16/@mov32` |
| `add(Tdst, Tsrc, dst, src0, src1, config)` | `@fadds/@faddh/@add16` |
| `sub(T, dst, src0, src1, config)` | `@fsubs/@fsubh` |
| `mul(T, dst, src0, src1, config)` | `@fmuls/@fmulh` |
| `fmac(Tdst, Tsrc, dst, src0, src1, x, config)` | `@fmacs/@fmach` |
| `neg(T, dst, src, config)` | `@fnegs/@fnegh` |
| `abs(T, dst, src, config)` | `@fabss/@fabsh` |
| `max(T, dst, src0, src1, config)` | `@fmaxs/@fmaxh` |
| `scale(T, dst, src0, src1, config)` | `@fscales/@fscaleh` |
| `convert(Tdst, Tsrc, dst, src, config)` | Type conversion builtins |
| `norm(T, dst, src, config)` | `@fnorms/@fnormh` |

Each function returns `bool` — guaranteed to expand to a single DSD builtin call.

### 14.2 Use Case

Write generic library code that works with both f16 and f32:

```csl
const ops = @import_module("<dsd_ops>");

fn generic_add(comptime T: type, dst: anytype, a: anytype, b: anytype) bool {
  return ops.add(T, T, dst, a, b, .{});
}
```

---

## 15. Other Libraries

### 15.1 empty

```csl
const cache = @import_module(if (stage == 0) "<empty>" else cache_name);
```

An intentionally empty library for conditional imports.

### 15.2 timer

```csl
const timer = @import_module("<timer>", .{ .timerCount = 2 });
timer.start(0);
// ... work ...
timer.stop(0);
var elapsed: [3]u16;
timer.elapsed(0, &elapsed);
```

Higher-level wrapper around `<time>` with named timer slots.

### 15.3 tile_config

Low-level PE hardware configuration. Submodules: `color_config`, `control_transform`, `exceptions`, `filters`, `input_queue_status`, `output_queue_status`, `main_thread_priority`, `switch_config`, `task_priority`, `teardown`.

WSE-3 additions: `queue_flush` submodule with `exit()`, `iq_status()`, `oq_status()`, `is_flushed()`.

### 15.4 Kernel Libraries

| Library | Purpose |
|---|---|
| `<kernels/fft/fft3d_layout>` | 3D FFT layout module |
| `<kernels/fft/fft3d>` | 3D FFT computation |
| `<kernels/tally/layout>` | Two-phase tally layout |
| `<kernels/tally/pe>` | Two-phase tally PE code |

---

## 16. Best Practices

1. **Import math generically**: Use `math.sqrt(x)` instead of `math.sqrt_f32(x)` — the generic version auto-dispatches based on type.

2. **Disable simprint in production**: `@import_module("<simprint>", .{ .enable = false })` compiles all calls to no-ops.

3. **Multiple debug trace channels**: Import `<debug>` multiple times with different `.key` values to separate trace data logically.

4. **malloc free() is all-or-nothing**: Design allocation patterns as phases — allocate everything, use it, then `free()` the entire arena.

5. **message_passing reserves colors**: Colors 14-18 are taken by default. Plan color allocation accordingly.

6. **sin/cos accuracy**: For trigonometric functions, ensure input magnitude stays below ~51472 for reliable results.

7. **tsc timing pattern**: Always call `enable_tsc()` before `get_timestamp()`, wrap computation between two timestamp calls.

---

## Related Topics

- **DSD Operations**: See `CSL_DSD_Complete_Guide.md` for `mem1d_dsd`, `fabin_dsd`, `fabout_dsd`
- **Async Programming**: See `CSL_Async_Programming_Guide.md` for task system and async DSD ops
- **Builtins**: See `CSL_Builtins_Reference_Guide.md` for `@fmacs`, `@fadds`, `@map`, etc.
- **Host Communication**: See `CSL_Host_Device_Communication_Guide.md` for memcpy module usage
- **FFT Implementation**: See `CSL_FFT_Complex_Arithmetic_Guide.md` for FFT algorithm details
- **Debugging**: See `CSL_Debugging_Guide.md` for debug/simprint deep dive
