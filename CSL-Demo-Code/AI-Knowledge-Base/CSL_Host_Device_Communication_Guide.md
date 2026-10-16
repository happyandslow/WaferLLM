# Cerebras CSL Host-Device Communication Complete Guide
## SDK Version: 1.4.0

This document covers the complete host-device communication system in Cerebras CSL, including the memcpy infrastructure, SdkRuntime Python API, copy mode, streaming mode, kernel launching, and the `sys_mod.unblock_cmd_stream()` protocol.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Memcpy Infrastructure Setup](#2-memcpy-infrastructure-setup)
3. [Copy Mode (Direct Memory Access)](#3-copy-mode-direct-memory-access)
4. [Streaming Mode (Wavelet-Based)](#4-streaming-mode-wavelet-based)
5. [Kernel Launching and RPC](#5-kernel-launching-and-rpc)
6. [SdkRuntime Python Host API](#6-sdkruntime-python-host-api)
7. [Command Stream and unblock_cmd_stream()](#7-command-stream-and-unblock_cmd_stream)
8. [Complete Code Examples](#8-complete-code-examples)
9. [API Quick Reference](#9-api-quick-reference)
10. [Best Practices and Common Pitfalls](#10-best-practices-and-common-pitfalls)
11. [Related Topics](#11-related-topics)

---

## 1. Architecture Overview

### 1.1 I/O Channel Architecture

The Cerebras WSE uses physical I/O channels at the East and West edges of the fabric for host-device data transfer:
- **WSE-2**: 60 channels per edge
- **WSE-3**: 62 channels per edge
- Channels are spaced roughly 16 rows apart

The `memcpy` infrastructure adds support PEs around the user kernel:
- 3 columns on the West (for H2D routing)
- 2 columns on the East (for D2H routing)
- A halo around the kernel

### 1.2 Two Transfer Modes

| Mode | Mechanism | Use Case |
|---|---|---|
| **Copy** | Direct memory write/read | Simple data loading before computation |
| **Streaming** | Wavelet-triggered data tasks | Overlapping computation and communication |

### 1.3 Reserved Resources

The `memcpy` infrastructure reserves the following — **do not use these in your program**:
- **Colors**: 21, 22, 23
- **Local task IDs**: 27, 28, 30
- **Control task IDs**: 33, 34, 35, 36, 37
- **Microthread**: 0
- **Input queue**: 0 (WSE-3 also uses input queue 1)
- **Output queue**: 0

---

## 2. Memcpy Infrastructure Setup

### 2.1 Layout File Setup

Every program using host-device communication must set up memcpy in the layout file:

```csl
// layout.csl
const memcpy = @import_module("<memcpy/get_params>", .{
    .width = width,
    .height = height
});

layout {
    @set_rectangle(width, height);
    @set_tile_code(0, 0, "pe_program.csl", .{
        .memcpy_params = memcpy.get_params(0)  // parameterized by PE x-coordinate
    });
}
```

**Key rule**: `memcpy.get_params(pe_x)` must receive the PE's **x-coordinate** as argument.

### 2.2 PE Program Setup

```csl
// pe_program.csl
param memcpy_params: comptime_struct;
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
```

### 2.3 Compiler Flags

Required compiler flags:
```shell
cslc --memcpy --channels=k \
     --fabric-dims=dim_x,dim_y \
     --fabric-offsets=x,y \
     layout.csl
```

Constraints:
- `k`: 1 to 16 (number of I/O channels)
- `dim_x >= 7 + width`, `dim_y >= 2 + height`
- `x >= 4`, `y >= 1`

With buffers (`--width-west-buf=k`, `--width-east-buf=p`):
- `x >= 4 + k`, `dim_x >= x + width + 3 + p`

---

## 3. Copy Mode (Direct Memory Access)

Copy mode writes data directly to PE memory without notifying the kernel. Useful for loading data before computation starts.

### 3.1 Device Side: Export Symbols

In the **layout file**, declare exported symbols with name, type, and mutability:

```csl
// layout.csl
@export_name("A", [*]f32, true);   // writable=true: host can write
@export_name("y", [*]f32, false);  // writable=false: host can only read
@export_name("compute", fn()void); // export a callable function
```

In the **PE program**, create pointers and export them:

```csl
// pe_program.csl
var A: [M*N]f32;
var y = @zeros([M]f32);
var A_ptr: [*]f32 = &A;
const y_ptr: [*]f32 = &y;  // const ptr = read-only from host

comptime {
    @export_symbol(A_ptr, "A");
    @export_symbol(y_ptr, "y");
}
```

**Mutability rules**:
- `@export_name("name", [*]T, true)` → host can **write** (H2D) and **read** (D2H)
- `@export_name("name", [*]T, false)` → host can only **read** (D2H)
- `var ptr` = mutable pointer, `const ptr` = immutable pointer

### 3.2 Host Side: Copy Mode Transfer

```python
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
A_symbol = runner.get_id('A')
y_symbol = runner.get_id('y')

runner.load()
runner.run()

# H2D copy: host array → device symbol
runner.memcpy_h2d(A_symbol, host_data, px, py, w, h, elem_per_pe,
    streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
    order=MemcpyOrder.ROW_MAJOR, nonblock=False)

# D2H copy: device symbol → host array
result = np.zeros([w*h*elem_per_pe], dtype=np.float32)
runner.memcpy_d2h(result, y_symbol, px, py, w, h, elem_per_pe,
    streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
    order=MemcpyOrder.ROW_MAJOR, nonblock=False)

runner.stop()
```

### 3.3 ROI (Region of Interest)

The `(px, py, w, h)` parameters define a sub-rectangle within the program rectangle:
- Origin `(0,0)` is the **top-left PE** of the program rectangle
- `elem_per_pe` is the number of elements per PE to transfer

### 3.4 Data Ordering

| Order | Mapping | Behavior |
|---|---|---|
| `ROW_MAJOR` | `A[h][w][l]`, `l` fastest | Elements contiguous on PE are contiguous on host |
| `COL_MAJOR` | `A[h][w][l]`, `w` fastest | **Better bandwidth** than row-major |

---

## 4. Streaming Mode (Wavelet-Based)

Streaming mode sends data as wavelets via fabric colors, allowing overlap of computation and communication. The kernel must use data tasks or microthreads to receive the data.

### 4.1 Layout Setup with Streaming Colors

Define streaming color parameters and pass them to memcpy:

```csl
// layout.csl
param MEMCPYH2D_DATA_1_ID: i16;
param MEMCPYD2H_DATA_1_ID: i16;

const MEMCPYH2D_DATA_1: color = @get_color(MEMCPYH2D_DATA_1_ID);
const MEMCPYD2H_DATA_1: color = @get_color(MEMCPYD2H_DATA_1_ID);

const memcpy = @import_module("<memcpy/get_params>", .{
    .width = width,
    .height = height,
    .MEMCPYH2D_1 = MEMCPYH2D_DATA_1,  // up to MEMCPYH2D_4
    .MEMCPYD2H_1 = MEMCPYD2H_DATA_1,  // up to MEMCPYD2H_4
});
```

**Limits**: Up to 4 H2D colors (`MEMCPYH2D_1..4`) and 4 D2H colors (`MEMCPYD2H_1..4`).

### 4.2 PE Program: Receiving Streamed Data (Data Task)

```csl
// pe_program.csl
const h2d_iq: input_queue = @get_input_queue(2);

const recv_task_id: data_task_id =
    if      (@is_arch("wse2")) @get_data_task_id(sys_mod.MEMCPYH2D_1)
    else if (@is_arch("wse3")) @get_data_task_id(h2d_iq);

task recv_data(data: f32) void {
    // Process each incoming wavelet
    // data contains one element per invocation
}

comptime {
    @bind_data_task(recv_data, recv_task_id);
    if (@is_arch("wse3")) {
        @initialize_queue(h2d_iq, .{ .color = sys_mod.MEMCPYH2D_1 });
    }
}
```

**WSE-2 vs WSE-3 difference**: On WSE-2, data tasks are bound to **colors**; on WSE-3, they are bound to **input queues**.

### 4.3 PE Program: Sending Data to Host (fabout DSD)

```csl
const d2h_oq: output_queue = @get_output_queue(2);

const out_dsd = @get_dsd(fabout_dsd, .{
    .fabric_color = sys_mod.MEMCPYD2H_1,
    .extent = M,
    .output_queue = d2h_oq
});

// Send data from memory to host via streaming
@fmovs(out_dsd, mem_dsd, .{ .async = true });
```

### 4.4 Host Side: Streaming Transfer

```python
# Get streaming color IDs from compile metadata
MEMCPYH2D_DATA_1 = int(compile_data['params']['MEMCPYH2D_DATA_1_ID'])
MEMCPYD2H_DATA_1 = int(compile_data['params']['MEMCPYD2H_DATA_1_ID'])

# Stream H2D: pass color ID as dest (not symbol)
runner.memcpy_h2d(MEMCPYH2D_DATA_1, x, 0, 0, w, h, elem_per_pe,
    streaming=True, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stream D2H: pass color ID as src (not symbol)
y_result = np.zeros([w*h*elem_per_pe], dtype=np.float32)
runner.memcpy_d2h(y_result, MEMCPYD2H_DATA_1, px, py, w, h, elem_per_pe,
    streaming=True, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
```

### 4.5 Streaming Color ID Compiler Params

Pass color IDs to the compiler:
```shell
--params=MEMCPYH2D_DATA_1_ID:<color_id>
--params=MEMCPYD2H_DATA_1_ID:<color_id>
```

---

## 5. Kernel Launching and RPC

### 5.1 Device Side: Export Functions

```csl
// layout.csl
@export_name("my_func", fn()void);

// pe_program.csl
fn my_func() void {
    // kernel computation
    sys_mod.unblock_cmd_stream();  // MUST call when done
}

comptime {
    @export_symbol(my_func);
}
```

Functions can have arguments:
```csl
// layout.csl
@export_name("f2", fn()void);  // Note: layout always declares fn()void

// pe_program.csl
fn f2(my_arg: f32) void {
    // use my_arg
    sys_mod.unblock_cmd_stream();
}
comptime { @export_symbol(f2); }
```

### 5.2 Host Side: Launch

```python
# Simple launch (no args)
runner.launch('my_func', nonblock=False)

# Launch with arguments
runner.launch('f2', 3.14, nonblock=False)
```

---

## 6. SdkRuntime Python Host API

### 6.1 Lifecycle

```python
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
runner.load()    # Load ELFs to simfabric or WSE (~80+ sec on real hardware)
runner.run()     # Start execution, wait for commands

# ... operations (memcpy_h2d, memcpy_d2h, launch) ...

runner.stop()    # Wait for all pending commands, then stop. MUST be called.
```

### 6.2 Key Methods

| Method | Description |
|---|---|
| `get_id(symbol)` | Get integer ID for an exported symbol (tensor or function) |
| `load()` | Load binaries to simfabric or WSE |
| `run()` | Start the simfabric or WSE |
| `memcpy_h2d(dest, src, px, py, w, h, l, **kw)` | Host → Device transfer |
| `memcpy_d2h(dest, src, px, py, w, h, l, **kw)` | Device → Host transfer |
| `launch(symbol, *args, nonblock=False)` | Launch host-callable function |
| `stop()` | Stop execution. **Must be called.** |

### 6.3 Advanced Transfer Methods

| Method | Description |
|---|---|
| `memcpy_h2d_colbcast(dest, src, px, py, w, h, l, **kw)` | Broadcast row data down columns |
| `memcpy_h2d_rowbcast(dest, src, px, py, w, h, l, **kw)` | Broadcast column data across rows |
| `memcpy_h2d_stride(dest, src, px, py, w, h, l, row_stride, col_stride, **kw)` | Strided transfer |

### 6.4 Nonblocking Operations

All transfer and launch methods accept `nonblock=True`:
- Returns immediately before operation completes
- Returns a `Task` handle for checking completion
- Use `runner.task_wait(task_handle)` or `runner.is_task_done(task_handle)` to poll

**Warning**: Avoid race conditions — two nonblocking operations to/from the same memory can produce undefined results.

### 6.5 16-bit Data Transfer

For 16-bit data:
- **H2D**: Zero-extend 16-bit values to 32-bit before sending
- **D2H**: Returned array contains 32-bit values; strip upper 16 bits

Use `sdk_utils` helpers:
```python
from cerebras.sdk.sdk_utils import memcpy_view, input_array_to_u32

# Create a float16 view into a uint32 container
container = np.zeros(N, dtype=np.uint32)
x = memcpy_view(container, np.float16)
x.fill(0.5)

runner.memcpy_h2d(symbol, container, 0, 0, 1, 1, N,
    streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
    order=MemcpyOrder.ROW_MAJOR, nonblock=False)
```

---

## 7. Command Stream and unblock_cmd_stream()

### 7.1 Why It's Needed

When the host calls `runner.launch()`, the memcpy infrastructure blocks its internal command stream while the kernel function executes. The kernel **must** call `sys_mod.unblock_cmd_stream()` to signal completion, allowing subsequent host commands to proceed.

### 7.2 Basic Pattern

```csl
fn my_kernel() void {
    // ... do computation ...
    sys_mod.unblock_cmd_stream();  // Signal completion to host
}
```

### 7.3 With Async Operations

If the kernel starts async operations, `unblock_cmd_stream()` must be called **after** the async operation completes, not immediately:

```csl
const exit_task_id: local_task_id = @get_local_task_id(9);

fn compute() void {
    // Start async operation — completion triggers exit_task
    @fmovs(out_dsd, in_dsd, .{ .async = true, .activate = exit_task_id });
}

task exit_task() void {
    sys_mod.unblock_cmd_stream();  // Called when async op finishes
}

comptime {
    @bind_local_task(exit_task, exit_task_id);
}
```

### 7.4 Common Mistake

**Wrong**: Calling `unblock_cmd_stream()` before async operations complete:
```csl
fn bad_kernel() void {
    @fmovs(out_dsd, in_dsd, .{ .async = true });
    sys_mod.unblock_cmd_stream();  // BUG: async op may not be done!
}
```

**Correct**: Use `.activate` or `.unblock` callbacks to defer the unblock.

---

## 8. Complete Code Examples

### 8.1 Minimal Copy Mode Program (gemv-03-memcpy)

**layout.csl**:
```csl
const memcpy = @import_module("<memcpy/get_params>", .{ .width = 1, .height = 1 });

layout {
    @set_rectangle(1, 1);
    @set_tile_code(0, 0, "pe_program.csl", .{ .memcpy_params = memcpy.get_params(0) });

    @export_name("A", [*]f32, true);    // writable
    @export_name("x", [*]f32, true);    // writable
    @export_name("b", [*]f32, true);    // writable
    @export_name("y", [*]f32, false);   // read-only
    @export_name("init_and_compute", fn()void);
}
```

**pe_program.csl**:
```csl
param memcpy_params: comptime_struct;
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

var A: [M*N]f32;
var x: [N]f32;
var b: [M]f32;
var y = @zeros([M]f32);
var A_ptr: [*]f32 = &A;
var x_ptr: [*]f32 = &x;
var b_ptr: [*]f32 = &b;
const y_ptr: [*]f32 = &y;

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

**run.py**:
```python
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
A_symbol = runner.get_id('A')
y_symbol = runner.get_id('y')
runner.load()
runner.run()

# Copy data to device
runner.memcpy_h2d(A_symbol, A, 0, 0, 1, 1, M*N, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Launch kernel
runner.launch('init_and_compute', nonblock=False)

# Copy result back
y_result = np.zeros([M], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, 0, 0, 1, 1, M, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.stop()
```

### 8.2 Streaming Mode Program (gemv-09-streaming)

**Key differences from copy mode**:

1. Layout declares streaming color params and passes them to memcpy
2. PE program uses data tasks to receive streamed wavelets
3. Host uses color IDs (not symbol IDs) for streaming transfers

**Layout streaming setup**:
```csl
param MEMCPYH2D_DATA_1_ID: i16;
param MEMCPYD2H_DATA_1_ID: i16;
const MEMCPYH2D_DATA_1: color = @get_color(MEMCPYH2D_DATA_1_ID);
const MEMCPYD2H_DATA_1: color = @get_color(MEMCPYD2H_DATA_1_ID);

const memcpy = @import_module("<memcpy/get_params>", .{
    .width = kernel_x_dim, .height = kernel_y_dim,
    .MEMCPYH2D_1 = MEMCPYH2D_DATA_1,
    .MEMCPYH2D_2 = MEMCPYH2D_DATA_2,
    .MEMCPYD2H_1 = MEMCPYD2H_DATA_1
});
```

**PE receiving and forwarding**:
```csl
// Receive from H2D stream, forward to internal color
task memcpy_recv_x(data: f32) void {
    @block(memcpy_recv_x_task_id);
    buf[0] = data;
    @fmovs(internal_out_dsd, buf_dsd, .{.async = true, .unblock = memcpy_recv_x_task_id});
}
```

**Host streaming**:
```python
# Read color IDs from compile metadata
MEMCPYH2D_DATA_1 = int(compile_data['params']['MEMCPYH2D_DATA_1_ID'])

# Stream to device
runner.memcpy_h2d(MEMCPYH2D_DATA_1, x, 0, 0, w, 1, N_per_PE,
    streaming=True, order=MemcpyOrder.ROW_MAJOR,
    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
```

---

## 9. API Quick Reference

### 9.1 Device-Side Exports

| Syntax | Purpose |
|---|---|
| `@export_name("sym", [*]T, true)` | Declare writable tensor symbol |
| `@export_name("sym", [*]T, false)` | Declare read-only tensor symbol |
| `@export_name("fn", fn()void)` | Declare callable function |
| `@export_symbol(ptr, "sym")` | Export pointer as symbol |
| `@export_symbol(fn_name)` | Export function |
| `sys_mod.unblock_cmd_stream()` | Signal function completion to host |

### 9.2 Host-Side API

| Call | Purpose |
|---|---|
| `SdkRuntime(bindir, cmaddr=addr)` | Create runtime |
| `runner.get_id("symbol")` | Get symbol/function ID |
| `runner.load()` | Load ELFs |
| `runner.run()` | Start execution |
| `runner.memcpy_h2d(dest, src, px, py, w, h, l, ...)` | Host → Device |
| `runner.memcpy_d2h(dest, src, px, py, w, h, l, ...)` | Device → Host |
| `runner.launch("fn", *args, nonblock=False)` | Launch function |
| `runner.stop()` | Stop (mandatory) |

### 9.3 Memcpy Keyword Arguments

| Keyword | Values | Notes |
|---|---|---|
| `streaming` | `True` / `False` | Streaming vs copy mode |
| `data_type` | `MEMCPY_32BIT` / `MEMCPY_16BIT` | Element size |
| `order` | `ROW_MAJOR` / `COL_MAJOR` | Host tensor layout |
| `nonblock` | `True` / `False` | Blocking vs nonblocking |

---

## 10. Best Practices and Common Pitfalls

### Best Practices

1. **Use copy mode for simple data loading**, streaming mode when you need computation-communication overlap
2. **Always call `sys_mod.unblock_cmd_stream()`** at the end of every host-launched function
3. **Use `COL_MAJOR` order** for better bandwidth when possible
4. **Read compile metadata** (`out.json`) for streaming color IDs rather than hardcoding
5. **Use blocking mode** (`nonblock=False`) unless you specifically need async host operations
6. **For multi-PE programs**, ensure `memcpy.get_params(pe_x)` receives the correct x-coordinate

### Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Missing `unblock_cmd_stream()` | Host hangs after `launch()` | Add `sys_mod.unblock_cmd_stream()` at end of function |
| Calling unblock before async completes | Data corruption or race | Use `.activate` callback to defer unblock |
| Using reserved colors/queues | Undefined behavior | Avoid colors 21-23, task IDs 27-28-30, queue 0 (and IQ1 on WSE-3) |
| Wrong `get_params(pe_x)` argument | Routing errors | Always pass the PE's x-coordinate |
| Wrong `streaming` flag | Copy to wrong location | `streaming=True` uses color ID, `streaming=False` uses symbol ID |
| Forgetting `runner.stop()` | Runtime error | Always call `stop()` |
| Mutability mismatch | Cannot write to device | Ensure `@export_name` has `true` for writable symbols |
| Missing `@initialize_queue` on WSE-3 | WSE-3 runtime error | Initialize all queues in `comptime` block with `@is_arch("wse3")` guard |

---

## 11. Related Topics

- **DSDs & DSRs**: `CSL_DSD_Complete_Guide.md` — `fabin_dsd` and `fabout_dsd` used in streaming mode
- **Async Programming**: `CSL_Async_Programming_Guide.md` — `.async`, `.activate`, `.unblock` callbacks
- **Builtins**: `CSL_Builtins_Reference_Guide.md` — `@export_name`, `@export_symbol`, `@get_color`
- **Routing & Colors**: `CSL_Routing_Colors_Switches_Guide.md` — Streaming colors and fabric routing
- **SdkLayout API**: `Cerebras_Docs/api-docs/sdklayout-api.md` — Python-based layout alternative
- **Tutorials**: `gemv-01` (basic), `gemv-03` (copy mode), `gemv-09` (streaming), `topic-15` (WSE-3 microthreads)
