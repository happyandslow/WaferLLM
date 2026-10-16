# Cerebras CSL Async Programming Complete Guide
## SDK Version: 1.4.0

This document covers the core concepts, syntax rules, and practical patterns for asynchronous programming, task management, microthreads, and inter-PE communication in Cerebras CSL.

---

## Table of Contents

1. [WSE-3 Microthread IDs](#1-wse-3-microthread-ids)
2. [Async DSD Operations](#2-async-dsd-operations)
3. [Task Identifiers and Execution Model](#3-task-identifiers-and-execution-model)
4. [Async Completion Callback Mechanisms](#4-async-completion-callback-mechanisms)
5. [FIFO and Task Activation](#5-fifo-and-task-activation)
6. [Queue Initialization (WSE-3 Required)](#6-queue-initialization-wse-3-required)
7. [Complete Code Examples](#7-complete-code-examples)
8. [Common Patterns and Best Practices](#8-common-patterns-and-best-practices)

---

## 1. WSE-3 Microthread IDs

### 1.1 Core Concepts

On the WSE-3 architecture, **microthread IDs (ut_id)** allow multiple concurrent async operations on the same hardware queue.

#### WSE-2 vs WSE-3
- **WSE-2**: Queue IDs and microthread IDs are tightly coupled — one queue can only handle one active async operation
- **WSE-3**: Queue IDs and microthread IDs are **decoupled** — any microthread ID (0-7) can be paired with any queue (0-7)

### 1.2 Syntax

```csl
// Get a microthread ID
const ut_id = @get_ut_id(id);  // id must be an integer between 0-7

// Use in DSD operations
@fmovs(out_dsd, in_dsd, .{ .async = true, .ut_id = ut_id });

// Use in @load_to_dsr
@load_to_dsr(dest_dsr, fab_dsd, .{ .async = true, .ut_id = ut_id });
```

### 1.3 Priority Rules

When multiple operands specify `ut_id`, the priority order is:
```
Destination > First Source > Second Source
```

### 1.4 Resource Reuse Example

```csl
const send_color_oq = @get_output_queue(2);  // Same output queue
const ut0 = @get_ut_id(0);
const ut1 = @get_ut_id(1);

// Two operations share the same output queue but run concurrently via different ut_ids
@mov16(out_dsd, in_dsd_1, .{ .async = true, .ut_id = ut0 });
@mov16(out_dsd, in_dsd_2, .{ .async = true, .ut_id = ut1 });
```

**Important constraint**: Two operations must not use the same microthread ID simultaneously.

### 1.5 Practical Application

```csl
// From examples/benchmarks/benchmark-libs/stencil_3d_7pts/wse3/pe.csl
const out_ut: ut_id = @get_ut_id(output_ut_id);

// Use in async send
@load_to_dsr(dest_dsr_send, fab_trans_east_wdsd,
    .{.async=true, .activate = f_send, .ut_id = out_ut });
```

---

## 2. Async DSD Operations

### 2.1 Triggering Conditions

A DSD operation becomes async when either condition is met:

1. **Explicitly specified** `.async = true`
2. **Operand is already marked async** (loaded via `@load_to_dsr` with `.async = true`)

### 2.2 Hardware Resource Constraints

Async operations consume two types of resources:

| Resource Type | Description | WSE-3 Details |
|---|---|---|
| **Queues** | Input/output queues (0-7) | Different queues have different buffer depths |
| **Microthreads** | Execution units (0-7) | Reuse queue resources via `ut_id` |

### 2.3 Queue Depths (WSE-3)

```
Input queues:
  - Queue 0,1: depth 6 wavelets
  - Queue 2,3: depth 4 wavelets
  - Queue 4-7: depth 2 wavelets

Output queues:
  - Queue 0,1: depth 2 wavelets
  - Queue 2,3: depth 6 wavelets
  - Queue 4,5: depth 2 wavelets
```

### 2.4 Async Operation Examples

```csl
// Basic async DSD operation
@fmovs(dest_dsd, src_dsd, .{ .async = true });

// Async operation + microthread ID
@mov32(dest_dsd, src_dsd, .{ .async = true, .ut_id = ut0 });

// Async operation + completion callback
@fmovs(out_dsd, in_dsd, .{
    .async = true,
    .ut_id = ut0,
    .activate = task_id  // Activate task upon completion
});

// DSR load can also be async
@load_to_dsr(src1_dsr, fab_recv_wdsd, .{
    .async = true,
    .activate = recv_task
});
```

---

## 3. Task Identifiers and Execution Model

### 3.1 Three Task Types

| Task Type | Identifier Type | WSE-3 ID Range | Activation Method |
|---|---|---|---|
| **Local Task** | `local_task_id` | 8-30 | `@activate` or async completion callback |
| **Data Task** | `data_task_id` | Depends on color/queue | Fabric data arrival (wavelet) |
| **Control Task** | `control_task_id` | Any ID | Control wavelet received |

### 3.2 Local Task

#### Definition and Binding

```csl
// Get a local task ID
const my_task_id: local_task_id = @get_local_task_id(8);  // WSE-3: 8-30

// Define the task function (no parameters)
task my_task() void {
    // Task logic
}

// Bind in a comptime block
comptime {
    @bind_local_task(my_task, my_task_id);
}
```

#### Task State

Local tasks have two independent state bits:

1. **Activated state**: Whether the task is in the scheduling queue
2. **Blocked state**: Whether the task is blocked

**Execution condition**: A task must be both **Activated** and **Unblocked** to run.

### 3.3 Data Task

```csl
// WSE-2: Create from color
const task_id: data_task_id = @get_data_task_id(my_color);

// WSE-3: Create from input_queue
const iq = @get_input_queue(2);
const task_id: data_task_id = @get_data_task_id(iq);

// Bind data task (accepts one parameter)
task process_data(value: f32) void {
    // Process received data
}

comptime {
    @bind_data_task(process_data, task_id);
}
```

### 3.4 Control Task

```csl
const ctrl_task_id: control_task_id = @get_control_task_id(42);

task control_handler(data: i16) void {
    // Handle control signal
}

comptime {
    @bind_control_task(control_handler, ctrl_task_id);
}
```

---

## 4. Async Completion Callback Mechanisms

### 4.1 `.activate` vs `.unblock`

In async DSD operation config, you can specify **only one** of:

```csl
// Option 1: Activate task
@mov32(dest, src, .{
    .async = true,
    .activate = local_task_id  // Add task to scheduling queue upon completion
});

// Option 2: Unblock task
@mov32(dest, src, .{
    .async = true,
    .unblock = local_task_id  // Clear the blocked flag upon completion
});
```

**Mutual exclusion rule**: You cannot set both `.activate` and `.unblock` simultaneously.

### 4.2 Multi-Task Synchronization Pattern

Using the dual-state mechanism, you can implement a "wait for multiple async operations" synchronization pattern:

```csl
const sync_task_id: local_task_id = @get_local_task_id(10);

task sync_task() void {
    // Only executes after both async operations complete
}

comptime {
    @bind_local_task(sync_task, sync_task_id);
    @block(sync_task_id);  // Initial state: blocked
}

fn launch_async_ops() void {
    // Async operation A: unblock upon completion (but don't activate)
    @mov32(dest1, src1, .{
        .async = true,
        .unblock = sync_task_id
    });

    // Async operation B: activate upon completion (if already unblocked, starts executing)
    @mov32(dest2, src2, .{
        .async = true,
        .activate = sync_task_id
    });
}
```

**Execution timeline**:
1. Initial: Task is blocked (`@block`)
2. Operation A completes: Task is unblocked, but not yet activated → **does not run**
3. Operation B completes: Task is activated, and already unblocked → **starts executing**

**Important characteristics**:
- Block state is **single-bit logic**, not reference-counted
- Multiple `@block` calls are equivalent to a single `@block`
- A single `@unblock` clears the blocked state

### 4.3 Builtins for Direct Task State Manipulation

```csl
// @block: Block a task or microthread
@block(task_id);          // Can be called at comptime or runtime
@block(@get_ut_id(0));    // Block a microthread

// @unblock: Unblock
@unblock(task_id);        // Only clears the blocked bit, does not activate

// @activate: Activate task
@activate(local_task_id); // Add task to scheduling queue
```

#### Call Sites
- **comptime blocks**: Set initial state
- **Inside runtime tasks**: Dynamic task state control

#### Example: Comptime Initialization

```csl
comptime {
    @bind_local_task(my_task, task_id);
    @block(task_id);  // Initial state is blocked, waiting for async operation to unblock
}
```

---

## 5. FIFO and Task Activation

### 5.1 FIFO Allocation

```csl
var fifo_buffer = @zeros([1024]f32);
const fifo = @allocate_fifo(fifo_buffer);
```

### 5.2 FIFO State Monitoring (Theoretical — rarely seen in practice)

```csl
// Note: The following is theoretical syntax, not found in current codebase
const fifo = @allocate_fifo(fifo_buffer, .{
    .activate_pop = pop_task_id,   // Activate when FIFO transitions from full to has-space
    .activate_push = push_task_id  // Activate when FIFO transitions from empty to has-data
});
```

### 5.3 Common FIFO Usage Patterns

Based on code analysis, the common pattern is:

```csl
// Async write to FIFO
@fadds(fifo, in_dsd, const_dsd, .{ .async = true });

// Async read from FIFO
@fnegs(out_dsd, fifo, .{ .async = true });

// FIFO serves as a buffer, decoupling producer and consumer
```

---

## 6. Queue Initialization (WSE-3 Required)

### 6.1 Why Initialization Is Needed

On WSE-3, when using explicit DSR allocation or async operations, queues must be manually initialized.

### 6.2 Initialization Syntax

```csl
comptime {
    // Input queue initialization
    @initialize_queue(input_queue_id, .{ .color = recv_color });

    // Output queue initialization (WSE-3 specific)
    if (@is_arch("wse3")) {
        @initialize_queue(output_queue_id, .{ .color = send_color });
    }
}
```

### 6.3 Complete Example

```csl
const recv_iq = @get_input_queue(2);
const send_oq = @get_output_queue(2);
const ut = @get_ut_id(4);

param recv_color: color;
param send_color: color;

comptime {
    // WSE-3 requires queue initialization
    @initialize_queue(recv_iq, .{ .color = recv_color });
    @initialize_queue(send_oq, .{ .color = send_color });
}
```

---

## 7. Complete Code Examples

### 7.1 Simple Async Communication (topic-15-wse3-microthreads)

#### Sender PE

```csl
param memcpy_params: comptime_struct;
param send_color: color;

const M: i16 = 10;

// Task IDs
const exit_task_id: local_task_id = @get_local_task_id(9);

// Queue and Microthread IDs
const send_color_oq = @get_output_queue(2);
const send_color_ut = @get_ut_id(4);

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

var y: [M]f32;
var y_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> y[i] });

fn compute() void {
    const out_dsd = @get_dsd(fabout_dsd, .{
        .fabric_color = send_color,
        .extent = M,
        .output_queue = send_color_oq
    });

    // Async send, activate exit_task upon completion
    @fmovs(out_dsd, y_dsd, .{
        .async = true,
        .ut_id = send_color_ut,
        .activate = exit_task_id
    });
}

task exit_task() void {
    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task(exit_task, exit_task_id);
    @initialize_queue(send_color_oq, .{ .color = send_color });
    @export_symbol(compute);
}
```

#### Receiver PE

```csl
param memcpy_params: comptime_struct;
param recv_color: color;

const M: i16 = 10;

const exit_task_id: local_task_id = @get_local_task_id(9);
const recv_color_iq = @get_input_queue(2);
const recv_color_ut = @get_ut_id(5);  // Different microthread ID

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

var y: [M]f32;
var y_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> y[i] });

fn compute() void {
    const in_dsd = @get_dsd(fabin_dsd, .{
        .fabric_color = recv_color,
        .extent = M,
        .input_queue = recv_color_iq
    });

    @fmovs(y_dsd, in_dsd, .{
        .async = true,
        .ut_id = recv_color_ut,
        .activate = exit_task_id
    });
}

task exit_task() void {
    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task(exit_task, exit_task_id);
    @initialize_queue(recv_color_iq, .{ .color = recv_color });
    @export_symbol(compute);
}
```

### 7.2 Data Task + Block/Unblock Pattern (topic-09-fifos)

```csl
const process_task_id: data_task_id =
    if (@is_arch("wse2")) @get_data_task_id(loopback_color)
    else if (@is_arch("wse3")) @get_data_task_id(loopback_iq);

var elem = @zeros([1]f32);
const elem_dsd = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{1} -> elem[0]});

const out_dsd = @get_dsd(fabout_dsd, .{
    .extent = 1,
    .fabric_color = sys_mod.MEMCPYD2H_1,
    .output_queue = d2h_data_1_oq
});

// Data task: receives one element at a time
task process_task(element: f32) void {
    // Block self to prevent re-triggering during async send
    @block(process_task_id);

    elem[0] = element * element * element;

    // Async send, unblock upon completion
    @fmovs(out_dsd, elem_dsd, .{
        .async = true,
        .unblock = process_task_id
    });
}

comptime {
    @bind_data_task(process_task, process_task_id);
}
```

### 7.3 Complex Synchronization Pattern (stencil example, simplified)

```csl
const COMM: local_task_id = @get_local_task_id(12);
const SEND: local_task_id = @get_local_task_id(13);
const RECV: local_task_id = @get_local_task_id(14);

var count_send_recv: i16 = 0;

task f_send() void {
    // Send logic...
    if (send_done) {
        count_send_recv += 1;
        check_and_continue();
    }
}

task f_recv() void {
    // Receive logic...
    if (recv_done) {
        count_send_recv += 1;
        check_and_continue();
    }
}

fn check_and_continue() void {
    if (count_send_recv >= 2) {
        // Both operations complete, proceed to next step
        compute_result();
        count_send_recv = 0;
        @activate(COMM);
    }
}

task f_comm() void {
    count_send_recv = 0;
    @activate(SEND);
    @activate(RECV);
}

comptime {
    @bind_local_task(f_send, SEND);
    @bind_local_task(f_recv, RECV);
    @bind_local_task(f_comm, COMM);
}
```

---

## 8. Common Patterns and Best Practices

### 8.1 Naming Conventions

```csl
// Task IDs
const main_task_id: local_task_id = @get_local_task_id(8);
const exit_task_id: local_task_id = @get_local_task_id(9);

// Queue IDs
const input_queue_id = @get_input_queue(2);
const output_queue_id = @get_output_queue(3);

// Microthread IDs
const ut0 = @get_ut_id(0);
const ut1 = @get_ut_id(1);
```

### 8.2 Queue and Microthread Allocation Strategy

#### Queue Selection
```csl
// Choose queues based on buffering needs
// Large buffer: IQ0,1 (depth 6) or OQ2,3 (depth 6)
// Normal buffer: IQ2,3 (depth 4)
// Small buffer: IQ4-7 (depth 2) or OQ0,1,4,5 (depth 2)
```

#### Microthread Reuse
```csl
// When queue resources are tight, reuse queues via microthread IDs
const oq = @get_output_queue(2);
const ut0 = @get_ut_id(0);
const ut1 = @get_ut_id(1);

// Same queue, different microthreads
@fmovs(out_dsd, data1_dsd, .{ .async = true, .ut_id = ut0 });
@fmovs(out_dsd, data2_dsd, .{ .async = true, .ut_id = ut1 });
```

### 8.3 Error Avoidance Checklist

#### Don't Do This

```csl
// Error 1: Setting both activate and unblock simultaneously
@mov32(dest, src, .{
    .async = true,
    .activate = task1,
    .unblock = task2  // WRONG — mutually exclusive!
});

// Error 2: Expecting @block to be reference-counted
@block(task_id);
@block(task_id);  // WRONG — equivalent to a single @block

// Error 3: Forgetting to initialize queues on WSE-3
const oq = @get_output_queue(2);
// WRONG — missing @initialize_queue(oq, .{ .color = my_color });

// Error 4: Two operations using the same microthread ID concurrently
const ut = @get_ut_id(0);
@mov32(dest1, src1, .{ .async = true, .ut_id = ut });
@mov32(dest2, src2, .{ .async = true, .ut_id = ut });  // WRONG — conflict!
```

#### Do This Instead

```csl
// Correct 1: Choose one callback method
@mov32(dest, src, .{ .async = true, .activate = task_id });

// Correct 2: Use a counter for multi-way synchronization
var sync_count: i16 = 0;
// Increment counter in each async completion callback

// Correct 3: Initialize queues on WSE-3
comptime {
    @initialize_queue(oq, .{ .color = my_color });
}

// Correct 4: Use different microthread IDs
const ut0 = @get_ut_id(0);
const ut1 = @get_ut_id(1);
@mov32(dest1, src1, .{ .async = true, .ut_id = ut0 });
@mov32(dest2, src2, .{ .async = true, .ut_id = ut1 });
```

### 8.4 Architecture Compatibility Checks

```csl
comptime {
    // Choose implementation based on architecture
    if (@is_arch("wse2")) {
        // WSE-2 specific code
        const task_id = @get_data_task_id(color);
    } else if (@is_arch("wse3")) {
        // WSE-3 specific code
        const task_id = @get_data_task_id(input_queue);
        @initialize_queue(output_queue, .{ .color = color });
    }
}
```

### 8.5 Debugging Tips

```csl
// Use @assert to verify state
@assert(stack_last == expected_value);
@assert(2 <= count);

// Use compile-time assertions
comptime {
    @comptime_assert(output_queue_id != input_queue_id);
}
```

---

## Appendix: API Quick Reference

### Task Management

| Function | Purpose | Call Site |
|---|---|---|
| `@get_local_task_id(n)` | Get local task ID (8-30) | comptime |
| `@get_data_task_id(color/queue)` | Get data task ID | comptime |
| `@get_control_task_id(n)` | Get control task ID | comptime |
| `@bind_local_task(fn, id)` | Bind local task | comptime |
| `@bind_data_task(fn, id)` | Bind data task | comptime |
| `@bind_control_task(fn, id)` | Bind control task | comptime |
| `@activate(task_id)` | Activate task | comptime/runtime |
| `@block(task_id/ut_id)` | Block task or microthread | comptime/runtime |
| `@unblock(task_id/ut_id)` | Unblock | runtime |

### Queues and Microthreads

| Function | Purpose |
|---|---|
| `@get_input_queue(n)` | Get input queue (0-7) |
| `@get_output_queue(n)` | Get output queue (0-7) |
| `@get_ut_id(n)` | Get microthread ID (0-7) |
| `@initialize_queue(q, .{.color=c})` | Initialize queue (required on WSE-3) |

### Async Configuration Parameters

| Parameter | Type | Description |
|---|---|---|
| `.async` | `bool` | Enable async execution |
| `.ut_id` | `ut_id` | Specify microthread ID |
| `.activate` | `local_task_id` | Activate task upon completion |
| `.unblock` | `local_task_id/input_queue/data_task_id` | Unblock upon completion |

### DSD Operation Examples

```csl
// Synchronous operation
@fmovs(dest_dsd, src_dsd);

// Async operation
@fmovs(dest_dsd, src_dsd, .{ .async = true });

// Async + microthread
@fmovs(dest_dsd, src_dsd, .{ .async = true, .ut_id = ut0 });

// Async + callback
@fmovs(dest_dsd, src_dsd, .{
    .async = true,
    .activate = task_id
});

// DSR load (async)
@load_to_dsr(dest_dsr, fab_dsd, .{
    .async = true,
    .activate = task_id,
    .ut_id = ut0
});
```

---

## References

- **Official Documentation**: https://sdk.cerebras.net/
  - Microthreads (WSE-3): https://sdk.cerebras.net/csl/language/microthreads_wse3
  - Async DSD Operations: https://sdk.cerebras.net/csl/language/dsds#asynchronous-dsd-operations
  - Task IDs: https://sdk.cerebras.net/csl/language/task-ids

- **Code Example Locations**:
  - `examples/tutorials/topic-15-wse3-microthreads/`
  - `examples/tutorials/topic-09-fifos/`
  - `examples/benchmarks/benchmark-libs/stencil_3d_7pts/wse3/`
  - `source-code/csl_libraries/memcpy/wse3/`

---

**Document Version**: 1.1
**Last Updated**: 2026-02-18
**Applicable SDK**: Cerebras CSL 1.4.0
