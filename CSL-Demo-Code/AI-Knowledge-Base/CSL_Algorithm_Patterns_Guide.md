# Cerebras CSL Algorithm Patterns Guide
## SDK Version: 1.4.0

This document catalogs common algorithmic patterns in CSL: state machines, BLAS operations, stencil computations, neighbor exchange, and multi-PE synchronization.

---

## Table of Contents

1. [Overview](#1-overview)
2. [State Machine Pattern](#2-state-machine-pattern)
3. [BLAS Operations with @map](#3-blas-operations-with-map)
4. [Stencil / Neighbor Exchange](#4-stencil--neighbor-exchange)
5. [Counter-Based Synchronization](#5-counter-based-synchronization)
6. [Block/Unblock Synchronization](#6-blockunblock-synchronization)
7. [Allreduce / Collective Operations](#7-allreduce--collective-operations)
8. [Iterative Solver Framework](#8-iterative-solver-framework)
9. [Best Practices and Common Pitfalls](#9-best-practices-and-common-pitfalls)

---

## 1. Overview

CSL programs on the wafer-scale engine are event-driven, not thread-based. Algorithms are decomposed into:

| Pattern | Use For | Trigger |
|---|---|---|
| **State machine** | Multi-phase algorithms (solvers, pipelines) | `@activate(task_id)` |
| **@map reduction** | Element-wise ops, dot products, norms | DSD + user function |
| **Neighbor exchange** | Stencils, halo exchange | Fabric colors + data_tasks |
| **Counter sync** | Waiting for N async completions | Counter + conditional `@activate` |
| **Block/unblock** | Back-pressure, single-element processing | `@block`/`@unblock` on task |

---

## 2. State Machine Pattern

The canonical approach for multi-phase algorithms in CSL. A single `local_task` dispatches to different functions based on `cur_state`.

### 2.1 Structure

```csl
param STATE: local_task_id;  // The dispatcher task

// Define states as named constants
const STATE_INIT: i16 = 0;
const STATE_COMPUTE: i16 = 1;
const STATE_REDUCE: i16 = 2;
const STATE_CHECK: i16 = 3;
const STATE_EXIT: i16 = 4;

var cur_state: i16 = 0;
var next_state: i16 = 0;

// Callback for async modules — advances to next state
fn f_trigger_state_machine() void {
    cur_state = next_state;
    @activate(STATE);
}

// The dispatcher
task f_state() void {
    if (STATE_INIT == cur_state) {
        next_state = STATE_COMPUTE;
        f_init();                        // sync → calls f_trigger_state_machine()
    } else if (STATE_COMPUTE == cur_state) {
        next_state = STATE_REDUCE;
        f_compute();                     // async → module calls f_trigger_state_machine()
    } else if (STATE_REDUCE == cur_state) {
        next_state = STATE_CHECK;
        f_reduce();                      // async → allreduce callback triggers
    } else if (STATE_CHECK == cur_state) {
        if (converged) {
            next_state = STATE_EXIT;
        } else {
            next_state = STATE_COMPUTE;  // Loop back
        }
        f_trigger_state_machine();
    } else if (STATE_EXIT == cur_state) {
        sys_mod.unblock_cmd_stream();    // Signal host
    }
}

comptime {
    @bind_local_task(f_state, STATE);
}
```

### 2.2 Launching the State Machine

From the host:
```python
runner.launch("f_start", nonblock=False)  # f_start sets cur_state and @activate(STATE)
```

From CSL:
```csl
fn f_start() void {
    cur_state = STATE_INIT;
    @activate(STATE);
}
comptime { @export_symbol(f_start); }
```

### 2.3 Async Module Callback Integration

When calling async modules (allreduce, stencil), pass `f_trigger_state_machine` as the callback:

```csl
const reduce_mod = @import_module("allreduce/pe.csl", @concat_structs(reduceParams, .{
    .f_callback = f_trigger_state_machine,
    // ... queue assignments
}));
```

When the module completes, it calls `f_callback`, advancing the state machine.

### 2.4 Key Rules

1. **Set `next_state` before calling the async function**, not after
2. **Every path must eventually call `f_trigger_state_machine()`** or the machine stalls
3. **Sync functions** should call `f_trigger_state_machine()` at the end
4. **Async modules** should be configured with `f_trigger_state_machine` as callback
5. **Exit state** must `unblock_cmd_stream()` to return control to host

**Canonical example**: `examples/benchmarks/conjugate-gradient/src/kernel_cg.csl`

---

## 3. BLAS Operations with @map

### 3.1 @map Basics

`@map` applies a user-defined function element-wise over DSD operands:

```csl
@map(kernel_fn, dsd_arg1, dsd_arg2, scalar_arg, &accumulator, &result);
```

- DSD arguments are streamed element by element
- Scalar and pointer arguments are passed directly to every invocation
- The return value is written to the corresponding element of the last DSD (or stored via pointer)

### 3.2 Dot Product

```csl
fn reduce_dot(xval: f32, yval: f32, sum: *f32) f32 {
    return sum.* + xval * yval;
}

fn dot(n: i16, x: [*]f32, y: [*]f32) f32 {
    mem_x_dsd = @set_dsd_base_addr(mem_x_dsd, x);
    mem_x_dsd = @set_dsd_length(mem_x_dsd, @bitcast(u16, n));
    mem_y_dsd = @set_dsd_base_addr(mem_y_dsd, y);
    mem_y_dsd = @set_dsd_length(mem_y_dsd, @bitcast(u16, n));
    var sum: f32 = @as(f32, 0);
    @map(reduce_dot, mem_x_dsd, mem_y_dsd, &sum, &sum);
    return sum;
}
```

### 3.3 Norm-2 (Overflow-Safe)

Uses `approx()` to find a power-of-2 scaling factor, avoiding overflow in squaring:

```csl
fn nrm2(n: i16, y: [*]f32) f32 {
    var alpha: f32;
    var inv_alpha: f32;

    // Step 1: ymax = max(|y|)
    var ymax: f32 = @as(f32, 0);
    @map(reduce_fabs, mem_y_dsd, &ymax, &ymax);

    // Step 2: Find scaling factor
    approx(ymax, &alpha, &inv_alpha);

    // Step 3: sum = Σ(y_i / alpha)²
    var sum: f32 = @as(f32, 0);
    @map(reduce_scale_square, mem_y_dsd, inv_alpha, &sum, &sum);

    // Step 4: nrm2 = alpha * sqrt(sum)
    return math_lib.sqrt(sum) * alpha;
}
```

### 3.4 AXPY and MAC

Use hardware DSD builtins for maximum performance:

```csl
// y = y + alpha * x  (fused multiply-accumulate)
@fmacs(mem_y_dsd, mem_y_dsd, mem_x_dsd, alpha);

// y = a - b (element-wise subtraction)
@fsubs(mem_r_dsd, mem_b_dsd, mem_w_dsd);

// y = -x (negate)
@fnegs(mem_y_dsd, mem_x_dsd);
```

**Source**: `examples/benchmarks/conjugate-gradient/src/blas.csl`

---

## 4. Stencil / Neighbor Exchange

### 4.1 Communication Pattern

Each PE exchanges data with 4 neighbors (N/S/E/W). Each direction uses a separate color pair (send + receive):

```
         North
          ↑↓
West ←→ [PE] ←→ East
          ↑↓
         South
```

8 colors total: `c_send_east`, `c_recv_east`, `c_send_west`, `c_recv_west`, `c_send_north`, `c_recv_north`, `c_send_south`, `c_recv_south`.

### 4.2 Send/Receive with Counter Sync

```csl
var count_send_recv: i16 = 0;

// COMM task: initiate all 4 sends + 4 receives simultaneously
task f_comm() void {
    count_send_recv = 0;

    // Send east (async)
    @fmovs(fabout_east_dsd, mem_east_dsd, .{
        .async = true,
        .activate = f_send  // Callback when send completes
    });

    // Receive from west (async)
    @fmovs(west_buf_dsd, fabin_west_dsd, .{
        .async = true,
        .activate = f_recv  // Callback when recv completes
    });
    // ... repeat for other 3 directions
}

// Count completions
task f_send() void {
    count_send_recv += 1;
    if (count_send_recv == 8) {  // 4 sends + 4 receives
        f_laplacian();  // All exchanges done → compute
    }
}

task f_recv() void {
    count_send_recv += 1;
    if (count_send_recv == 8) {
        f_laplacian();
    }
}
```

### 4.3 Boundary PE Handling

Edge PEs skip sends/receives in non-existent directions:

```csl
param first_px: bool;  // No west neighbor
param last_px: bool;   // No east neighbor
param first_py: bool;  // No south neighbor
param last_py: bool;   // No north neighbor

fn f_send_east() void {
    if (last_px) {
        count_send_recv += 1;  // Skip send, just count
    } else {
        @fmovs(fabout_east, data_dsd, .{ .async = true, .activate = f_send });
    }
    // Check completion
    if (count_send_recv == 8) f_laplacian();
}
```

### 4.4 Stencil Computation

After receiving neighbor data:

```csl
fn f_laplacian() void {
    // w = c_center * x
    @fmuls(mem_w_dsd, mem_x_dsd, c_center);
    // w += c_west * west_buf
    @fmacs(mem_w_dsd, mem_w_dsd, west_buf_dsd, c_west);
    // w += c_east * east_buf
    @fmacs(mem_w_dsd, mem_w_dsd, east_buf_dsd, c_east);
    // ... south, north, bottom, top
    f_callback();  // Return to state machine
}
```

**Source**: `examples/benchmarks/benchmark-libs/stencil_3d_7pts/wse3/pe.csl`

---

## 5. Counter-Based Synchronization

When N async operations must complete before proceeding:

```csl
var completion_count: i16 = 0;
const TOTAL_OPS: i16 = 4;

task on_op_complete() void {
    completion_count += 1;
    if (completion_count == TOTAL_OPS) {
        completion_count = 0;  // Reset for next round
        @activate(NEXT_TASK);  // All done
    }
}
```

Used heavily in stencil neighbor exchange (wait for 4 sends + 4 receives = 8 completions).

### Key Rules

1. **Reset counter before starting new round** to avoid stale counts
2. **Use a single counter** for both send and receive callbacks
3. **Do NOT use counter across PEs** — counters are PE-local

---

## 6. Block/Unblock Synchronization

For processing one element at a time from a stream:

```csl
const task_id: data_task_id = @get_data_task_id(input_queue);

task process(datum: f32) void {
    @block(task_id);                    // Prevent next element from arriving
    // ... process datum ...
    @fmovs(out_dsd, result_dsd, .{
        .async = true,
        .unblock = process              // Re-enable task when output completes
    });
}
```

### When to Use

- **Per-element processing** with side effects that need serialization
- **Back-pressure** when consumer is slower than producer
- **Multi-step per-element pipelines** where each element needs several ops

**Tutorial**: `examples/tutorials/topic-09-fifos/buffer.csl`

---

## 7. Allreduce / Collective Operations

### 7.1 Row-Column Allreduce

The benchmark-libs provide a reusable allreduce module:

```csl
const reduce_mod = @import_module("allreduce/pe.csl", @concat_structs(reduceParams, .{
    .f_callback = f_trigger_state_machine,
    .queues = [1]u16{2},
    .dest_dsr_ids = [1]u16{1},
    .src0_dsr_ids = [1]u16{1},
    .src1_dsr_ids = [1]u16{1},
}));

// Usage: allreduce 1 element using ADD
reduce_mod.allreduce(1, &rho, reduce_mod.TYPE_BINARY_OP.ADD);
```

### 7.2 Collectives_2d Library

For broadcast, scatter, gather across rows/columns:

```csl
const collective = @import_module("<collectives_2d/pe>");

// Broadcast from column 0 along the row
collective.broadcast_along_x(data, size);

// Gather along column
collective.gather_along_y(local_data, gathered_output, local_size);
```

**Tutorial**: `examples/tutorials/topic-11-collectives/`
**Benchmark**: `examples/benchmarks/gemm-collectives_2d/pe.csl`

---

## 8. Iterative Solver Framework

Combining all patterns into a complete solver:

### 8.1 Conjugate Gradient State Flow

```
INIT → SPMV_AX → RESIDUAL → CONV_CHECK ──→ EXIT
                                    │
                               UPDATE_P → SPMV_AP → ETA → UPDATE_X_R_RHO → CONV_CHECK
```

### 8.2 Typical Module Composition

```csl
// 1. memcpy for host communication (queues 0-1)
const sys_mod = @import_module("<memcpy/memcpy>", memcpyParams);

// 2. allreduce for global reductions (queue 2)
const reduce_mod = @import_module("allreduce/pe.csl", @concat_structs(reduceParams, .{
    .f_callback = f_trigger_state_machine,
    .queues = [1]u16{2},
}));

// 3. stencil for SpMV (queues 4-7)
const stencil_mod = @import_module("stencil_3d_7pts/pe.csl", @concat_structs(stencilParams, .{
    .f_callback = f_trigger_state_machine,
    .input_queues = [4]u16{4, 5, 6, 7},
    .output_queues = [4]u16{4, 5, 6, 7},
}));
```

### 8.3 Host Driver

```python
runner.load()
runner.run()

# Transfer data
runner.memcpy_h2d(b_symbol, b_data, ...)
runner.memcpy_h2d(x_symbol, x0_data, ...)

# Run CG
runner.launch("f_cg", n, tol, max_iter, nonblock=False)

# Read results
runner.memcpy_d2h(x_result, x_symbol, ...)
runner.stop()
```

**Source**: `examples/benchmarks/conjugate-gradient/`

---

## 9. Best Practices and Common Pitfalls

### Best Practices

1. **State machine for multi-phase algorithms**: Cleanly separates phases and handles async callbacks.
2. **Use hardware DSD ops (@fmacs, @fsubs)** for standard linear algebra — faster than @map.
3. **Use @map for custom reductions** (dot, nrm2) where no DSD builtin exists.
4. **Assign queues non-overlapping**: memcpy (0-1), allreduce (2-3), stencil (4-7).
5. **Use `@concat_structs`** to compose module parameters with local configuration.
6. **Reset counters** before starting a new round of async operations.

### Common Pitfalls

1. **Setting `next_state` after the async call**: The callback may fire before `next_state` is set. Always set `next_state` first.
2. **Missing `f_trigger_state_machine()` in sync path**: If a function is synchronous, it must call the trigger itself.
3. **Queue conflicts between modules**: Each module needs dedicated queues. Plan queue allocation at the layout level.
4. **Forgetting `unblock_cmd_stream()` in EXIT**: Host will hang waiting for completion.
5. **Overflow in nrm2**: Always use the scaling `approx()` technique for squared sums.
6. **Deadlock in neighbor exchange**: Boundary PEs must still count skipped operations.

---

## Related Topics

- **DSD Operations**: See `CSL_DSD_Complete_Guide.md` for DSD creation and manipulation
- **Async Programming**: See `CSL_Async_Programming_Guide.md` for task binding, .async, .activate, .unblock
- **Queue & FIFO**: See `CSL_Queue_FIFO_Guide.md` for queue allocation and FIFO pipelines
- **Routing**: See `CSL_Routing_Colors_Switches_Guide.md` for color-based PE-to-PE communication setup
- **Builtins**: See `CSL_Builtins_Reference_Guide.md` for @fmacs, @fadds, @map, etc.
- **Standard Libraries**: See `CSL_Standard_Libraries_Guide.md` for math, collectives_2d, time
