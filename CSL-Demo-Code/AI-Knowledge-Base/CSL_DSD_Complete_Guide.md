# CSL DSD (Data Structure Descriptors) Complete Guide
## SDK Version: 1.4.0

This document covers all DSD types, DSR management, runtime DSD manipulation, async DSD operations, and FIFO usage in Cerebras CSL.

---

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [Memory DSDs (mem1d_dsd)](#2-memory-dsds-mem1d_dsd)
3. [Multi-Dimensional Memory DSDs (mem4d_dsd)](#3-multi-dimensional-memory-dsds-mem4d_dsd)
4. [Fabric Input DSDs (fabin_dsd)](#4-fabric-input-dsds-fabin_dsd)
5. [Fabric Output DSDs (fabout_dsd)](#5-fabric-output-dsds-fabout_dsd)
6. [FIFO DSDs](#6-fifo-dsds)
7. [DSD Runtime Manipulation](#7-dsd-runtime-manipulation)
8. [Data Structure Registers (DSRs)](#8-data-structure-registers-dsrs)
9. [Async DSD Operations](#9-async-dsd-operations)
10. [Code Patterns](#10-code-patterns)
11. [Common Pitfalls](#11-common-pitfalls)
12. [API Quick Reference](#12-api-quick-reference)

---

## 1. Core Concepts

**Data Structure Descriptors (DSDs)** are compact representations of memory chunks or fabric wavelets. They work with DSD operations (builtins like `@fmovs`, `@fadds`, etc.) to express repeated operations using single hardware instructions.

**Key benefits**:
- Non-contiguous memory access patterns (strided, diagonal)
- Hardware-accelerated element-wise operations
- Concurrent asynchronous operations via fabric DSDs
- FIFO buffering for producer-consumer decoupling

**DSD Types**:
| Type | Purpose |
|---|---|
| `mem1d_dsd` | 1D memory vector access |
| `mem4d_dsd` | Up to 4D memory access (comptime only) |
| `fabin_dsd` | Fabric input (receive wavelets) |
| `fabout_dsd` | Fabric output (send wavelets) |
| FIFO DSD | Hardware FIFO buffer (via `@allocate_fifo`) |

---

## 2. Memory DSDs (mem1d_dsd)

### 2.1 Creation

```csl
// Using tensor_access (preferred, comptime-known)
var data_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{N} -> data[i] });

// Using explicit fields
var data_dsd = @get_dsd(mem1d_dsd, .{
    .base_address = &data,
    .extent = N,
    .stride = 1,       // optional, default=1
    .offset = 0        // optional, default=0
});
```

### 2.2 Properties

| Property | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_address` | pointer | Yes* | — | Starting memory address |
| `extent` | u16 | Yes* | — | Number of elements to access |
| `stride` | i8 | No | 1 | Step size between elements |
| `offset` | i16 | No | 0 | Initial offset from base |
| `tensor_access` | comptime expr | Yes* | — | Tensor access expression |
| `wavelet_index_offset` | bool | No | false | Enable dynamic index mode |

\* Either `tensor_access` or `base_address`+`extent` is required.

### 2.3 Tensor Access Expression

```
|<induction_variable>| {<length>} -> <base_address>[<expression>]
```

**Examples**:
```csl
// Linear access: data[0], data[1], ..., data[N-1]
|i|{N} -> data[i]

// Strided access (column of row-major matrix): A[0], A[N], A[2N], ...
|i|{M} -> A[i * N]

// Diagonal of 2D array: matrix[0,0], matrix[1,1], ..., matrix[19,19]
|i|{20} -> matrix[i, i]

// Scalar repeat (same element for every iteration)
|i|{N} -> buf[0]
```

### 2.4 Pointer Destinations

When the destination is a pointer rather than a DSD, each iteration writes to the same address (last value wins):

```csl
var result: f32 = 0;
@fmovs(&result, src_dsd);  // result = last element of src_dsd
```

---

## 3. Multi-Dimensional Memory DSDs (mem4d_dsd)

### 3.1 Creation

```csl
const dsd = @get_dsd(mem4d_dsd, .{
    .tensor_access = |i, j| {2, 3} -> array[i, j]
});
```

### 3.2 Properties

All properties must be comptime-known. Supports up to 4 dimensions.

| Property | Type | Required | Description |
|---|---|---|---|
| `base_address` | comptime pointer | Yes* | Starting address |
| `extent` | comptime tuple | Yes* | Size per dimension |
| `stride` | comptime tuple | No | Step per dimension |
| `tensor_access` | comptime expr | Yes* | Multi-dim access |

### 3.3 Access Order

Iteration is row-major (rightmost index varies fastest):
```csl
// |i,j| {2,2} -> array[i, j]
// Access order: [0,0], [0,1], [1,0], [1,1]
```

### 3.4 Stride Convention

**Critical**: Stride values are read left-to-right, but access dimensions are read right-to-left (innermost first).

For `stride = {1, -2}` with `extent = {2, 4}`:
- Innermost (rightmost): stride=1, extent=4
- Outer (leftmost): stride=-2, extent=2

When the inner dimension resets, the stride accounts for the "step back":
- address_at_i1_j0 - address_at_i0_j_last = stride value

---

## 4. Fabric Input DSDs (fabin_dsd)

### 4.1 Creation

```csl
const in_dsd = @get_dsd(fabin_dsd, .{
    .fabric_color = recv_color,
    .extent = N,
    .input_queue = recv_iq
});
```

### 4.2 Properties

| Property | Type | Required | Description |
|---|---|---|---|
| `fabric_color` | color | Yes | Color of incoming wavelets |
| `extent` | u16 | Yes | Number of wavelets expected |
| `input_queue` | input_queue | Required for async | Hardware input queue |
| `priority` | struct | No | WSE-3: `.{.high=true}`, `.{.medium=true}`, `.{.low=true}` |
| `control_transform` | bool | No | Preserve control bit through FIFO |

### 4.3 Usage Pattern

```csl
const recv_iq = @get_input_queue(2);

const in_dsd = @get_dsd(fabin_dsd, .{
    .fabric_color = recv_color,
    .extent = M,
    .input_queue = recv_iq
});

// WSE-3: must initialize queue
comptime {
    if (@is_arch("wse3")) {
        @initialize_queue(recv_iq, .{ .color = recv_color });
    }
}

// Use in async receive
@fadds(y_dsd, y_dsd, in_dsd, .{ .async = true, .activate = done_task });
```

---

## 5. Fabric Output DSDs (fabout_dsd)

### 5.1 Creation

```csl
const out_dsd = @get_dsd(fabout_dsd, .{
    .fabric_color = send_color,
    .extent = N,
    .output_queue = send_oq
});
```

### 5.2 Properties

| Property | Type | Required | Description |
|---|---|---|---|
| `fabric_color` | color | Yes | Color for outgoing wavelets |
| `extent` | u16 | Yes | Number of wavelets to send |
| `output_queue` | output_queue | Required for async | Hardware output queue |
| `control` | bool | No | Send control wavelets (end-of-stream) |
| `wavelet_index_offset` | bool | No | Carry index in high 16 bits |
| `simd_mode` | struct | No | `.{.simd_32=true}`, `.{.simd_64=true}` |
| `zero` | struct | No | `.{.first_source=true}` or `.{.second_source=true}` |
| `advance_switch` | bool | No | Auto-advance switch after last wavelet |
| `control_transform` | bool | No | Preserve control bit through FIFO |

### 5.3 Advanced Features

**Control wavelets** (end-of-stream signal):
```csl
const ctrl_dsd = @get_dsd(fabout_dsd, .{
    .fabric_color = color,
    .extent = 1,
    .output_queue = oq,
    .control = true  // Mark as control wavelet
});
```

**Zero source after send** (auto-reset memory):
```csl
const out_dsd = @get_dsd(fabout_dsd, .{
    .fabric_color = color,
    .extent = N,
    .output_queue = oq,
    .zero = .{ .first_source = true }  // Zero first source memory after op
});
```

**Auto-switch advance** (ping-pong patterns):
```csl
const out_dsd = @get_dsd(fabout_dsd, .{
    .fabric_color = color,
    .extent = N,
    .output_queue = oq,
    .advance_switch = true  // Advance switch position after last wavelet
});
```

---

## 6. FIFO DSDs

### 6.1 Allocation

```csl
var fifo_buffer = @zeros([1024]f32);
const fifo = @allocate_fifo(fifo_buffer);
```

**Buffer requirements**:
- Must be `var` global array
- Element type: `i16`, `u16`, `f16`, `i32`, `u32`, or `f32`
- Must be comptime-known const variable
- Compiler auto-aligns unless explicitly declared

### 6.2 Optional Configuration

```csl
const fifo = @allocate_fifo(fifo_buffer, .{
    .activate_pop = pop_task_id,    // Activate when FIFO goes from full to has-space
    .activate_push = push_task_id,  // Activate when FIFO goes from empty to has-data
    .priority = .{ .high = true }   // WSE-3: .low (default), .medium, .high
});
```

### 6.3 FIFO with DSRs

```csl
@allocate_fifo(buffer, .{
    .dest = @get_dsr(dsr_dest, 4),
    .src = @get_dsr(dsr_src1, 4),
    .xdsr = @get_xdsr(1)
});
// Note: dest and src DSR IDs must match; all three must be specified together
```

### 6.4 Runtime Length Control

```csl
@set_fifo_write_length(fifo, length);  // Set before write operation
@set_fifo_read_length(fifo, length);   // Set before read operation
```

### 6.5 FIFO Constraints

- Cannot be the first operand if using multiple sources
- Cannot be used with `@map` builtin
- At most one FIFO per operation
- Hardware resource — use sparingly

---

## 7. DSD Runtime Manipulation

### 7.1 @increment_dsd_offset

Advance the base address of a memory DSD by a given offset.

```csl
// Signature
@increment_dsd_offset(input_dsd, offset, elem_type) → mem1d_dsd

// Parameters:
//   input_dsd: mem1d_dsd or mem4d_dsd
//   offset: i16, number of elements to advance
//   elem_type: u16, i16, f16, i32, u32, or f32
```

**Canonical pattern** (column iteration in GEMV):
```csl
var A_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> A[i * N] });

fn gemv() void {
    for (@range(u16, N)) |i| {
        @fmacs(y_dsd, y_dsd, A_dsd, x[i]);
        A_dsd = @increment_dsd_offset(A_dsd, M, f32);  // Next column
    }
}
```

### 7.2 @set_dsd_base_addr

Clone a DSD with a new base address:
```csl
new_dsd = @set_dsd_base_addr(input_dsd, &new_buffer);
```

### 7.3 @set_dsd_length

Clone a DSD with a new extent:
```csl
new_dsd = @set_dsd_length(input_dsd, new_length);
// Works on all DSD types except mem4d_dsd
```

### 7.4 @set_dsd_stride

Clone a 1D DSD with a new stride:
```csl
new_dsd = @set_dsd_stride(input_dsd, new_stride);
// mem1d_dsd only
```

---

## 8. Data Structure Registers (DSRs)

### 8.1 Overview

DSRs are physical registers that store DSD values. Three register files exist:

| DSR Type | Purpose | Can be Dest | Can be Src |
|---|---|---|---|
| `dsr_dest` | Destination only | Yes | No |
| `dsr_src0` | Source or destination | Yes | Yes |
| `dsr_src1` | Source only | No | Yes |
| `dsr_fifo_dest` | FIFO destination | Yes | No |
| `dsr_fifo_src1` | FIFO source | No | Yes |

### 8.2 @get_dsr

```csl
const dsr1 = @get_dsr(dsr_dest, 0);    // Destination DSR #0
const dsr2 = @get_dsr(dsr_src0, 1);    // Source/dest DSR #1
const dsr3 = @get_dsr(dsr_src1, 6);    // Source DSR #6

// Convert to FIFO type
const dsr4 = @get_dsr(dsr_fifo_dest, @get_dsr(dsr_dest, 4));
```

Comptime-only. Each ID must be unique within its DSR type.

### 8.3 @get_xdsr

Extended DSRs for multi-dimensional or complex operations:
```csl
const my_xdsr = @get_xdsr(4);
```

### 8.4 @load_to_dsr

Load a DSD into a DSR for use in operations:

```csl
@load_to_dsr(dsr_value, dsd_value);
@load_to_dsr(dsr_value, dsd_value, config);
```

**Config options**:
| Option | Type | Description |
|---|---|---|
| `.async` | bool | Async load (fabric DSDs only) |
| `.activate` | local_task_id | Activate task on completion |
| `.unblock` | id | Unblock task/color/queue on completion |
| `.save_address` | bool | Auto-update base address after operation (mem DSDs) |
| `.single_step` | bool | For @map operations |
| `.on_control` | struct | `.{.terminate=true}` or `.{.activate=task_id}` |

### 8.5 save_address Feature

Automatically advances the base address after each operation, enabling chunked processing without manual `@increment_dsd_offset`:

```csl
const CHUNK = 4;
const N_CHUNKS = 3;
var data_in = [12]i16{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
var data_out = @zeros([12]i16);

const in_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{CHUNK} -> data_in[i] });
const out_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{CHUNK} -> data_out[i] });

const out_dsr = @get_dsr(dsr_dest, 0);
const in_dsr = @get_dsr(dsr_src0, 1);

comptime {
    @load_to_dsr(out_dsr, out_dsd, .{ .save_address = true });
    @load_to_dsr(in_dsr, in_dsd, .{ .save_address = true });
}

task main() void {
    // Each iteration automatically advances base address by CHUNK
    for (@range(i16, N_CHUNKS)) |_| {
        @mov16(out_dsr, in_dsr);
    }
}
```

---

## 9. Async DSD Operations

### 9.1 When Operations Become Async

An operation is async when:
1. `.async = true` is explicitly set, OR
2. At least one operand DSR was loaded with `.async = true`

### 9.2 Hardware Resources

Async operations consume:
- **Queues** (0-7): Input or output hardware queues
- **Microthreads** (0-7, WSE-3): Execution units

**No two concurrent async operations can share the same queue or microthread.**

### 9.3 Queue Depths

| Queue | WSE-2 Input | WSE-2 Output | WSE-3 Input | WSE-3 Output |
|---|---|---|---|---|
| 0, 1 | 6 words | 2 words | 8 words | 8 words |
| 2, 3 | 4 words | 6 words | 4 words | 8 words |
| 4, 5 | 2 words | 2 words | 4 words | 8 words |
| 6, 7 | 2 words | — | 4 words | 8 words |

### 9.4 Completion Callbacks

```csl
// Activate task on completion
@fmovs(dest, src, .{ .async = true, .activate = task_id });

// Unblock task on completion
@fmovs(dest, src, .{ .async = true, .unblock = task_id });

// On control wavelet
@fmovs(dest, src, .{ .async = true, .on_control = .{ .terminate = true } });
@fmovs(dest, src, .{ .async = true, .on_control = .{ .activate = task_id } });
```

**Note**: `.activate` and `.unblock` are mutually exclusive.

### 9.5 INFINITE_DSD_LEN

Special extent value for streaming until explicit termination:

```csl
const INFINITE_DSD_LEN: u16 = 0x7fff;  // 32767

const h2d_dsd = @get_dsd(fabin_dsd, .{
    .extent = INFINITE_DSD_LEN,
    .fabric_color = sys_mod.MEMCPYH2D_1,
    .input_queue = h2d_iq
});
```

Use case: H2D/D2H streaming pipelines where data size is determined at runtime.

---

## 10. Code Patterns

### Pattern 1: Simple Linear Memory Access

```csl
var data: [N]f32;
const data_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{N} -> data[i] });
```

### Pattern 2: Strided Column Access (GEMV)

```csl
// From examples/tutorials/gemv-02-memory-dsds/pe_program.csl
var A: [M * N]f32;
var A_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> A[i * N] });

fn gemv() void {
    for (@range(u16, N)) |i| {
        @fmacs(y_dsd, y_dsd, A_dsd, x[i]);
        A_dsd = @increment_dsd_offset(A_dsd, M, f32);
    }
}
```

### Pattern 3: PE-to-PE Fabric Communication

```csl
// From examples/tutorials/gemv-06-routes-1/pe_program.csl
const send_color_oq = @get_output_queue(2);
const send_color_iq = @get_input_queue(2);

fn send_right() void {
    const out_dsd = @get_dsd(fabout_dsd, .{
        .fabric_color = send_color,
        .extent = M,
        .output_queue = send_color_oq
    });
    @fmovs(out_dsd, y_dsd, .{ .async = true, .activate = exit_task_id });
}

fn recv_left() void {
    const in_dsd = @get_dsd(fabin_dsd, .{
        .fabric_color = send_color,
        .extent = M,
        .input_queue = send_color_iq
    });
    @fadds(y_dsd, y_dsd, in_dsd, .{ .async = true, .activate = exit_task_id });
}

comptime {
    if (@is_arch("wse3")) {
        @initialize_queue(send_color_oq, .{ .color = send_color });
        @initialize_queue(send_color_iq, .{ .color = send_color });
    }
}
```

### Pattern 4: H2D → FIFO → Compute → D2H Pipeline

```csl
// From examples/tutorials/pipeline-02-fifo/pe_program.csl
const INFINITE_DSD_LEN: u16 = 0x7fff;

var fifo_buffer = @zeros([1024]i16);
const fifo = @allocate_fifo(fifo_buffer);

const h2d_in_dsd = @get_dsd(fabin_dsd, .{
    .extent = INFINITE_DSD_LEN,
    .fabric_color = sys_mod.MEMCPYH2D_1,
    .input_queue = h2d_1_iq
});

const C1_out_dsd = @get_dsd(fabout_dsd, .{
    .extent = INFINITE_DSD_LEN,
    .fabric_color = C1,
    .output_queue = C1_oq
});

task main_task() void {
    @mov16(fifo, h2d_in_dsd, .{ .async = true });       // H2D → FIFO
    @mov16(C1_out_dsd, fifo, .{ .async = true });       // FIFO → fabric
    @add16(d2h_out_dsd, C1_in_dsd, one_dsd, .{ .async = true }); // Compute
}
```

### Pattern 5: Scalar Reduction via Pointer Destination

```csl
const src = [8]f16{ 0, 1, 2, 3, 4, 5, 6, 7 };
const src_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{4} -> src[2*i] });

var result: f16 = 0;
@fmovh(&result, src_dsd);  // result = 6 (last element accessed)
```

### Pattern 6: Control Wavelet Passthrough via FIFO

```csl
const in_dsd = @get_dsd(fabin_dsd, .{
    .fabric_color = recv_color,
    .extent = 100,
    .input_queue = @get_input_queue(0),
    .control_transform = true
});

const out_dsd = @get_dsd(fabout_dsd, .{
    .extent = 100,
    .fabric_color = send_color,
    .output_queue = @get_output_queue(1),
    .control_transform = true
});

var buf = @zeros([5]u32);
const fifo = @allocate_fifo(buf);

task buffer() void {
    @mov32(fifo, in_dsd, .{ .async = true });
    @mov32(out_dsd, fifo, .{ .async = true });
}
```

---

## 11. Common Pitfalls

### Sharing Queues Between Concurrent Async Operations

```csl
// WRONG: Same output queue for two concurrent operations
task t() void {
    @mov16(fabric_out_dsd, mem1, .{ .async = true });
    @mov16(fabric_out_dsd, mem2, .{ .async = true });  // ERROR: same queue!
}
```

### Missing wavelet_index_offset Index

```csl
// WRONG: DSD declares wavelet_index_offset but no .index provided
const dsd = @get_dsd(mem1d_dsd, .{
    .tensor_access = |i|{10} -> array[i],
    .wavelet_index_offset = true
});
@add16(dsd, dsd, data);  // Undefined behavior!

// CORRECT:
@add16(dsd, dsd, data, .{ .index = my_index });
```

### FIFO Position Restrictions

```csl
// WRONG: FIFO as first operand with multiple sources
@fmacs(fifo, src1, src2);  // Error

// WRONG: FIFO with @map
@map(fn, fifo, out);  // Error

// CORRECT: FIFO as non-first operand
@fmacs(dest, mem_dsd, fifo);  // OK
```

### WSE-3 Missing Queue Initialization

```csl
// WRONG on WSE-3:
const fab_dsd = @get_dsd(fabin_dsd, .{
    .fabric_color = color,
    .input_queue = @get_input_queue(0)
});
// Will fail at runtime on WSE-3!

// CORRECT:
comptime {
    if (@is_arch("wse3")) {
        @initialize_queue(@get_input_queue(0), .{ .color = color });
    }
}
```

### Stride vs Access Dimension Order Confusion

```csl
// For stride {1, -2} and extent {2, 4}:
// Innermost (rightmost) stride=1, extent=4
// Outer (leftmost) stride=-2, extent=2
// Stride reads LEFT-to-RIGHT, access reads RIGHT-to-LEFT!
```

---

## 12. API Quick Reference

### DSD Creation

| Function | Returns | Description |
|---|---|---|
| `@get_dsd(mem1d_dsd, config)` | mem1d_dsd | 1D memory descriptor |
| `@get_dsd(mem4d_dsd, config)` | mem4d_dsd | Up to 4D memory descriptor |
| `@get_dsd(fabin_dsd, config)` | fabin_dsd | Fabric input descriptor |
| `@get_dsd(fabout_dsd, config)` | fabout_dsd | Fabric output descriptor |
| `@allocate_fifo(buffer)` | FIFO DSD | Allocate hardware FIFO |
| `@allocate_fifo(buffer, config)` | FIFO DSD | FIFO with task activation |

### DSD Manipulation

| Function | Description |
|---|---|
| `@increment_dsd_offset(dsd, offset, type)` | Advance base address |
| `@set_dsd_base_addr(dsd, addr)` | Clone with new base |
| `@set_dsd_length(dsd, length)` | Clone with new extent |
| `@set_dsd_stride(dsd, stride)` | Clone with new stride (mem1d only) |
| `@set_fifo_read_length(fifo, len)` | Set FIFO read length |
| `@set_fifo_write_length(fifo, len)` | Set FIFO write length |

### DSR Management

| Function | Description |
|---|---|
| `@get_dsr(type, id)` | Allocate DSR register |
| `@get_xdsr(id)` | Allocate extended DSR |
| `@load_to_dsr(dsr, dsd)` | Load DSD into DSR |
| `@load_to_dsr(dsr, dsd, config)` | Load with async/callback config |

### Queue Management

| Function | Description |
|---|---|
| `@get_input_queue(id)` | Get input queue (0-7) |
| `@get_output_queue(id)` | Get output queue (0-7) |
| `@initialize_queue(q, config)` | Initialize queue (WSE-3 required) |

---

## Related Topics

- **Async Programming**: See `CSL_Async_Programming_Guide.md` for task system, block/unblock patterns, microthreads
- **Builtins**: See `CSL_Builtins_Reference_Guide.md` for all DSD operations (@fmovs, @fadds, @map, etc.)
- **Official DSD docs**: `Cerebras_Docs/csl/language/dsds.md`
- **Official DSR docs**: `Cerebras_Docs/csl/language/dsrs.md`
- **Tutorials**: `examples/tutorials/gemv-02-memory-dsds/`, `gemv-06-routes-1/`, `pipeline-02-fifo/`
- **Benchmarks**: `examples/benchmarks/benchmark-libs/stencil_3d_7pts/` (advanced DSR usage)

---

**Document Version**: 1.0
**Last Updated**: 2026-02-18
**Applicable SDK**: Cerebras CSL 1.4.0
