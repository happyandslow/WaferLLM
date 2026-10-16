# Cerebras CSL WSE-2 vs WSE-3 Migration Guide
## SDK Version: 1.4.0

This document catalogs all differences between WSE-2 and WSE-3 and provides concrete migration patterns for porting CSL code.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Reference Table](#2-quick-reference-table)
3. [Queue Initialization](#3-queue-initialization)
4. [Data Task ID Creation](#4-data-task-id-creation)
5. [Microthreads](#5-microthreads)
6. [Multi-Fabric-Input Restriction](#6-multi-fabric-input-restriction)
7. [Queue/Microthread Decoupling](#7-queuemicrothread-decoupling)
8. [Queue Depths](#8-queue-depths)
9. [Task ID Ranges](#9-task-id-ranges)
10. [Control Task Table Separation](#10-control-task-table-separation)
11. [Queue Flush](#11-queue-flush)
12. [Portable Code Patterns](#12-portable-code-patterns)
13. [Migration Checklist](#13-migration-checklist)
14. [Common Migration Errors](#14-common-migration-errors)

---

## 1. Overview

WSE-3 is a superset of WSE-2 in many ways, but introduces **breaking changes** that require code updates:

- **Mandatory queue initialization** (`@initialize_queue`)
- **data_task_id from input_queue** (not color)
- **No multi-fabric-input DSD operations** (requires FIFO workaround)
- **New microthread model** (decoupled from queues)

The `@is_arch("wse2")` / `@is_arch("wse3")` comptime check enables writing portable code.

---

## 2. Quick Reference Table

| Feature | WSE-2 | WSE-3 |
|---|---|---|
| **Queue init** | Implicit | `@initialize_queue()` **required** |
| **Queue ↔ Microthread** | Coupled (same ID) | **Decoupled**: any `ut_id(0-7)` + any `queue(0-7)` |
| **data_task_id from** | `@get_data_task_id(color)` | `@get_data_task_id(input_queue)` |
| **@get_ut_id** | N/A | `@get_ut_id(0-7)` |
| **Local task IDs** | 8–24 | 8–30 |
| **Arch check** | `@is_arch("wse2")` | `@is_arch("wse3")` |
| **Input queue depths** | Uniform | IQ 0,1: 6; IQ 2,3: 4; IQ 4-7: 2 |
| **Output queue depths** | Uniform | OQ 0,1: 2; OQ 2,3: 6; OQ 4,5: 2 |
| **Multi fabric inputs/op** | Allowed | **Not allowed** (use FIFO workaround) |
| **@queue_flush** | N/A | Available |
| **@set_control_task_table** | N/A | Available |
| **Instruction decode pipe** | Stage 2 | Stage 3 |
| **Instruction execute pipe** | Stage 4 | Stage 6 |
| **Max SIMD width** | SIMD-4 | SIMD-8 |
| **Half wavelets** | N/A | Supported (`half=0/1` in logs) |
| **memcpy input queues** | IQ 0 | IQ 0, IQ 1 |

---

## 3. Queue Initialization

### WSE-2 (implicit)

```csl
const iq: input_queue = @get_input_queue(2);
const oq: output_queue = @get_output_queue(3);
// Just use them — no initialization needed
```

### WSE-3 (required)

```csl
const iq: input_queue = @get_input_queue(2);
const oq: output_queue = @get_output_queue(3);

comptime {
    @initialize_queue(iq, .{ .color = my_color });
    @initialize_queue(oq, .{ .color = my_color });
}
```

### Portable Pattern

```csl
comptime {
    if (@is_arch("wse3")) {
        @initialize_queue(iq, .{ .color = my_color });
        @initialize_queue(oq, .{ .color = my_color });
    }
}
```

Or simply call `@initialize_queue` unconditionally — it's a no-op on WSE-2:

```csl
comptime {
    @initialize_queue(iq, .{ .color = my_color });
    @initialize_queue(oq, .{ .color = my_color });
}
```

---

## 4. Data Task ID Creation

### WSE-2: From Color

```csl
const recv_task_id: data_task_id = @get_data_task_id(my_color);
```

### WSE-3: From Input Queue

```csl
const recv_task_id: data_task_id = @get_data_task_id(my_iq);
```

### Portable Pattern

```csl
const recv_task_id: data_task_id =
    if      (@is_arch("wse2")) @get_data_task_id(my_color)
    else if (@is_arch("wse3")) @get_data_task_id(my_iq);
```

This is required everywhere a `data_task_id` is created for `@bind_data_task` or `@block`/`@unblock`.

---

## 5. Microthreads

### WSE-2

- No explicit microthread control
- Queue ID implicitly determines microthread

### WSE-3

- Microthreads are **decoupled** from queues
- Use `@get_ut_id(id)` to create microthread handles
- Assign to async DSD ops via `.ut_id` setting
- Can block/unblock microthreads independently

```csl
const ut0 = @get_ut_id(0);
const ut1 = @get_ut_id(1);

// Two concurrent async ops on different microthreads
@mov16(out_dsd, in_dsd, .{ .async = true, .ut_id = ut0 });
@mov16(out_dsd2, in_dsd2, .{ .async = true, .ut_id = ut1 });
```

### Microthread Priority

When multiple DSR operands specify `.ut_id`, priority order is:
1. Destination operand
2. First source operand
3. Second source operand

### Blocking/Unblocking Microthreads

```csl
const ut0 = @get_ut_id(0);

comptime {
    @block(ut0);  // Start blocked
}

task main() void {
    @unblock(ut0);  // Unblock at runtime
}
```

---

## 6. Multi-Fabric-Input Restriction

### WSE-2: Allowed

```csl
// Two fabric inputs in one operation — works on WSE-2
@add16(output, fabin_dsd_1, fabin_dsd_2, .{ .async = true });
```

### WSE-3: NOT Allowed

```csl
// ERROR on WSE-3: two fabric inputs
@add16(output, fabin_dsd_1, fabin_dsd_2, .{ .async = true });
```

### WSE-3: FIFO Workaround

Buffer one fabric input through a FIFO first:

```csl
var buffer: [size]u16;
const fifo = @allocate_fifo(buffer);

// Step 1: Move one fabric input to FIFO (memory)
@mov16(fifo, fabin_dsd_2, .{ .async = true });

// Step 2: Use FIFO (now memory source) + fabric input
@add16(output, fabin_dsd_1, fifo, .{ .async = true });
```

### Portable Pattern

The FIFO workaround works on both WSE-2 and WSE-3, so you can use it unconditionally for portable code.

---

## 7. Queue/Microthread Decoupling

### WSE-2

- Each queue is implicitly bound to a microthread of the same ID
- Queue 2 always runs on microthread 2

### WSE-3

- Queues and microthreads are independent
- Any queue can use any microthread via `.ut_id`
- Default: microthread ID matches the queue ID of the first fabric operand

```csl
const iq2 = @get_input_queue(2);
const ut5 = @get_ut_id(5);

// Queue 2 data, running on microthread 5
const in_dsd = @get_dsd(fabin_dsd, .{
    .extent = 10,
    .fabric_color = my_color,
    .input_queue = iq2,
});
@mov16(mem_dsd, in_dsd, .{ .async = true, .ut_id = ut5 });
```

---

## 8. Queue Depths

WSE-3 has asymmetric queue depths (WSE-2 depths are uniform):

### Input Queues

| Queue | Depth (wavelets) | Recommendation |
|---|---|---|
| IQ 0, 1 | 6 | memcpy uses IQ 0 (and IQ 1 on WSE-3) |
| IQ 2, 3 | 4 | Good for user data tasks |
| IQ 4, 5, 6, 7 | 2 | Low-throughput or control |

### Output Queues

| Queue | Depth (wavelets) | Recommendation |
|---|---|---|
| OQ 0, 1 | 2 | memcpy uses OQ 0 |
| OQ 2, 3 | 6 | Best for user high-throughput output |
| OQ 4, 5 | 2 | Low-throughput or control |

### Migration Impact

- On WSE-2, queue depth was uniform, so queue assignment didn't matter for performance
- On WSE-3, **match queue depth to data rate** — high-throughput paths need deep queues

---

## 9. Task ID Ranges

| Task Type | WSE-2 Range | WSE-3 Range |
|---|---|---|
| **Data tasks** | Bound to colors (0–24) | Bound to input queues (0–7) |
| **Local tasks** | 8–24 | 8–30 |
| **memcpy reserved local** | 27, 28, 30 | 27, 28, 30 |
| **memcpy reserved control** | 33–37 | 33–37 |
| **memcpy data task** | T22 (color 22) | T01 (input queue 1) |

### Migration Impact

- On WSE-2, data tasks are indexed by color → up to 25 data tasks
- On WSE-3, data tasks are indexed by input queue → only 8 data tasks
- If you have > 8 data tasks on WSE-2, refactor to use local tasks or multiplex

---

## 10. Control Task Table Separation

WSE-3 introduces `@set_control_task_table()` to decouple control tasks from data/local tasks:

```csl
task foo() void {}   // control task
task bar() void {}   // local task

comptime {
    @bind_control_task(foo, @get_control_task_id(10));
    @bind_local_task(bar, @get_local_task_id(10));

    // Same ID (10) — no conflict because separate tables
    @set_control_task_table(.{ .instructions = 8, .stride = 4 });
}
```

Options:
- `instructions`: Number of instructions per entry point (2, 4, or 8; default 4)
- `stride`: Stride per input queue's local control table index (1–7; default 1)

---

## 11. Queue Flush

WSE-3 adds `@queue_flush(queue_id)` — activates the teardown task when a queue becomes empty:

```csl
const out_q = @get_output_queue(2);

task foo() void {
    @mov16(fabout, mem_dsd, .{ .async = true });

    // Activate teardown task once out_q drains
    @queue_flush(out_q);
}
```

Useful for ensuring all data has been transmitted before teardown.

---

## 12. Portable Code Patterns

### 12.1 Complete Portable Template

```csl
param my_color: color;

// Queues
const my_iq: input_queue = @get_input_queue(2);
const my_oq: output_queue = @get_output_queue(2);

// Data task ID — architecture-dependent
const recv_task_id: data_task_id =
    if      (@is_arch("wse2")) @get_data_task_id(my_color)
    else if (@is_arch("wse3")) @get_data_task_id(my_iq);

task recv(datum: u32) void {
    // ... process datum ...
}

comptime {
    @bind_data_task(recv, recv_task_id);

    // Queue initialization (WSE-3 only, but calling unconditionally is fine)
    if (@is_arch("wse3")) {
        @initialize_queue(my_iq, .{ .color = my_color });
        @initialize_queue(my_oq, .{ .color = my_color });
    }
}
```

### 12.2 Portable Output Queue Assignment

For stencil modules that need different queue assignments:

```csl
const stencil_mod = @import_module("stencil/pe.csl", @concat_structs(stencilParams, .{
    .input_queues = [4]u16{4, 5, 6, 7},
    .output_queues = if (@is_arch("wse3")) [4]u16{4, 5, 6, 7} else [1]u16{3},
}));
```

---

## 13. Migration Checklist

### Mandatory Changes (code will break without these)

- [ ] Add `@initialize_queue` for every input/output queue in `comptime` blocks
- [ ] Change `@get_data_task_id(color)` → `@get_data_task_id(input_queue)`
- [ ] Remove or FIFO-wrap any DSD operation with two fabric inputs
- [ ] Verify queue IDs: memcpy uses IQ 0 **and IQ 1** on WSE-3

### Recommended Changes (performance/correctness)

- [ ] Assign high-throughput data to deep queues (IQ 0-1 depth 6, OQ 2-3 depth 6)
- [ ] Add explicit `.ut_id` for concurrent async operations
- [ ] Consider `@queue_flush` for clean termination
- [ ] Consider `@set_control_task_table` if control/local task IDs conflict

### Portable Code Practice

- [ ] Use `@is_arch("wse2"/"wse3")` guards for architecture-specific code
- [ ] Test on both architectures in the simulator
- [ ] Use the FIFO workaround unconditionally (works on both)

---

## 14. Common Migration Errors

| Error | Cause | Fix |
|---|---|---|
| Compilation failure: "queue not initialized" | Missing `@initialize_queue` on WSE-3 | Add to `comptime` block |
| Wrong task fires or no task fires | `@get_data_task_id(color)` on WSE-3 | Change to `@get_data_task_id(input_queue)` |
| "cannot have two fabric inputs" | Two `fabin_dsd` in one DSD op on WSE-3 | Use FIFO workaround |
| Hang after memcpy | Using IQ 1 which is now reserved by memcpy on WSE-3 | Start user queues at IQ 2 |
| Performance drop | High-throughput data on shallow queue (IQ 4-7, depth 2) | Move to IQ 0-1 or IQ 2-3 |
| Task ID collision | Control and local task use same ID on WSE-3 | Use `@set_control_task_table()` |

---

## Related Topics

- **Queue & FIFO**: See `CSL_Queue_FIFO_Guide.md` for FIFO allocation and multi-input workaround
- **Routing**: See `CSL_Routing_Colors_Switches_Guide.md` for color-queue-task relationships
- **Builtins**: See `CSL_Builtins_Reference_Guide.md` for `@is_arch`, `@initialize_queue`, `@get_ut_id`
- **Async Programming**: See `CSL_Async_Programming_Guide.md` for `.async`, `.ut_id` settings
- **SdkLayout**: See `CSL_SdkLayout_Guide.md` for the new layout API (replaces `layout.csl`)
