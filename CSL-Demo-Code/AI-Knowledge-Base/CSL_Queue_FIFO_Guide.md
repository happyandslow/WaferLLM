# Cerebras CSL Queue & FIFO Complete Guide
## SDK Version: 1.4.0

This document provides a comprehensive reference for hardware queues, FIFOs, queue initialization, and related synchronization patterns in CSL.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Hardware Queues](#2-hardware-queues)
3. [Queue Initialization (WSE-3 Required)](#3-queue-initialization-wse-3-required)
4. [FIFO Allocation and Usage](#4-fifo-allocation-and-usage)
5. [FIFO + Loopback Color Pattern](#5-fifo--loopback-color-pattern)
6. [FIFO as WSE-3 Multi-Input Workaround](#6-fifo-as-wse-3-multi-input-workaround)
7. [INFINITE_DSD_LEN for Streaming](#7-infinite_dsd_len-for-streaming)
8. [Queue Depth Limits (WSE-3)](#8-queue-depth-limits-wse-3)
9. [Complete Code Examples](#9-complete-code-examples)
10. [Best Practices and Common Pitfalls](#10-best-practices-and-common-pitfalls)

---

## 1. Overview

CSL uses two types of data buffering mechanisms:

| Mechanism | Purpose | Allocation |
|---|---|---|
| **Hardware Queues** | Interface between fabric and CE | `@get_input_queue(id)`, `@get_output_queue(id)` |
| **FIFOs** | Software-managed circular buffer in PE memory | `@allocate_fifo(buffer)` |

Queues connect colors to the compute element. FIFOs provide decoupling between async producer/consumer operations.

---

## 2. Hardware Queues

### 2.1 Getting Queue Handles

```csl
const iq: input_queue = @get_input_queue(id);   // id: 0-7
const oq: output_queue = @get_output_queue(id);  // id: 0-7
```

### 2.2 Queue Usage in DSDs

Queues are specified in fabric DSDs:

```csl
const in_dsd = @get_dsd(fabin_dsd, .{
  .extent = 10,
  .fabric_color = my_color,
  .input_queue = iq,
});

const out_dsd = @get_dsd(fabout_dsd, .{
  .extent = 10,
  .fabric_color = my_color,
  .output_queue = oq,
});
```

### 2.3 Queue-Color Binding

- Each queue must be bound to exactly one color
- On WSE-2: binding is implicit (queue ID = color's physical position)
- On WSE-3: binding is explicit via `@initialize_queue`

---

## 3. Queue Initialization (WSE-3 Required)

### 3.1 The Rule

**On WSE-3, every queue must be explicitly initialized before use:**

```csl
comptime {
  @initialize_queue(iq, .{ .color = my_color });
  @initialize_queue(oq, .{ .color = my_color });
}
```

On WSE-2, queue initialization is implicit and `@initialize_queue` is not needed.

### 3.2 Portable Pattern

```csl
comptime {
  if (@is_arch("wse3")) {
    @initialize_queue(my_iq, .{ .color = my_color });
    @initialize_queue(my_oq, .{ .color = my_color });
  }
}
```

Or unconditionally (works on both architectures if using WSE-3 style):

```csl
comptime {
  @initialize_queue(my_iq, .{ .color = my_color });
  @initialize_queue(my_oq, .{ .color = my_color });
}
```

### 3.3 What Happens Without Initialization

On WSE-3, failing to initialize queues results in compilation errors or undefined runtime behavior. Always initialize in a `comptime` block.

---

## 4. FIFO Allocation and Usage

### 4.1 Allocating a FIFO

```csl
var fifo_buffer = @zeros([1024]i16);  // Backing storage (type determines element type)
const fifo = @allocate_fifo(fifo_buffer);
```

The FIFO type (i16, f32, etc.) is determined by the buffer element type.

### 4.2 Using FIFOs in DSD Operations

FIFOs can be used as source or destination in DSD operations:

```csl
// Write to FIFO (producer)
@mov16(fifo, input_dsd, .{ .async = true });

// Read from FIFO (consumer)
@mov16(output_dsd, fifo, .{ .async = true });
```

### 4.3 FIFO as Decoupling Buffer

FIFOs decouple producers from consumers. The producer writes data into the FIFO at its own pace; the consumer reads when data is available:

```csl
// Pipeline: H2D → FIFO → fabric output
@mov16(fifo, h2d_in_dsd, .{ .async = true });     // H2D fills FIFO
@mov16(fabric_out_dsd, fifo, .{ .async = true });  // FIFO drains to fabric
```

Both operations run concurrently, with the FIFO absorbing timing differences.

---

## 5. FIFO + Loopback Color Pattern

This pattern converts streaming data into wavelet-triggered tasks:

```
H2D → FIFO → (process in FIFO) → fabout (loopback color) → fabin → data_task
```

### 5.1 How It Works

1. Data arrives via H2D into a FIFO through async DSD ops
2. Processed data is sent out on a **loopback color** (a color routed from RAMP back to RAMP on the same PE)
3. Each wavelet on the loopback color triggers a `data_task`
4. The data task processes one element at a time with `@block`/`@unblock` for back-pressure

### 5.2 Code Pattern

```csl
param loopback_color: color;

const loopback_iq: input_queue = @get_input_queue(4);
const loopback_oq: output_queue = @get_output_queue(5);

const process_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(loopback_color)
  else if (@is_arch("wse3")) @get_data_task_id(loopback_iq);

var fifo_buffer = @zeros([1024]f32);
const fifo = @allocate_fifo(fifo_buffer);

// Producer: H2D + add 10 → FIFO → negate → loopback
task main_task() void {
  @fadds(fifo, in_dsd, ten_dsd, .{ .async = true });
  @fnegs(loopback_dsd, fifo, .{ .async = true });
}

// Consumer: triggered per-element by loopback wavelet
task process_task(element: f32) void {
  @block(process_task_id);
  elem[0] = element * element * element;
  @fmovs(out_dsd, elem_dsd, .{ .async = true, .unblock = process_task });
}
```

The loopback color routing in `layout.csl`:
```csl
@set_color_config(pe_x, 0, loopback_color, .{
  .routes = .{ .rx = .{ RAMP }, .tx = .{ RAMP } }
});
```

**Tutorial**: `examples/tutorials/topic-09-fifos/buffer.csl`

---

## 6. FIFO as WSE-3 Multi-Input Workaround

### 6.1 The Problem

**WSE-3 restriction**: A DSD operation cannot have multiple fabric inputs. This code fails on WSE-3:

```csl
// ERROR on WSE-3: two fabric inputs
@add16(output, fabin_dsd_1, fabin_dsd_2, .{ .async = true });
```

### 6.2 The Solution

Buffer one fabric input through a FIFO first:

```csl
var buffer: [size]u16;
const fifo = @allocate_fifo(buffer);

// Step 1: Move second input to FIFO
@mov16(fifo, fabin_dsd_2, .{ .async = true });

// Step 2: Use FIFO (memory source) + first fabric input
@add16(output, fabin_dsd_1, fifo, .{ .async = true });
```

### 6.3 Portable Multi-Input Pattern

```csl
// Works on both WSE-2 and WSE-3
var buffer: [size]u16;
const fifo = @allocate_fifo(buffer);

task main() void {
  @mov16(fifo, input2, .{.async = true});    // Buffer second input
  @add16(output, input1, fifo, .{.async = true});  // Combine
}
```

**Tutorial**: `examples/tutorials/sdklayout-04-h2d-d2h/add2vec.csl`

---

## 7. INFINITE_DSD_LEN for Streaming

### 7.1 Definition

```csl
const INFINITE_DSD_LEN: u16 = 0x7fff;  // 32767
```

When used as the DSD extent, the operation runs "infinitely" — it won't stop until the data source/sink is exhausted.

### 7.2 Use Case: FIFO Pipeline

```csl
// H2D has unknown length at compile time → use INFINITE
const h2d_in_dsd = @get_dsd(fabin_dsd, .{
  .extent = INFINITE_DSD_LEN,
  .fabric_color = sys_mod.MEMCPYH2D_1,
  .input_queue = h2d_iq,
});

const fabric_out_dsd = @get_dsd(fabout_dsd, .{
  .extent = INFINITE_DSD_LEN,
  .fabric_color = C1,
  .output_queue = c1_oq,
});

// Pipeline: H2D → FIFO → fabric (both sides use INFINITE)
@mov16(fifo, h2d_in_dsd, .{ .async = true });
@mov16(fabric_out_dsd, fifo, .{ .async = true });

// Then a finite-extent operation on the other end
const c1_in_dsd = @get_dsd(fabin_dsd, .{
  .extent = size,  // Finite: knows exact count
  .fabric_color = C1,
  .input_queue = c1_iq,
});
@add16(d2h_out_dsd, c1_in_dsd, one_dsd, .{ .async = true });
```

### 7.3 Key Rules

- At least one side of a FIFO pipeline should have finite extent to ensure termination
- INFINITE_DSD_LEN is typically used with FIFO-backed H2D/D2H streaming
- The actual data length is determined by the host-side transfer size

**Tutorial**: `examples/tutorials/pipeline-02-fifo/pe_program.csl`

---

## 8. Queue Depth Limits (WSE-3)

### 8.1 Input Queue Depths

| Queue ID | Depth (wavelets) |
|---|---|
| IQ 0, 1 | 6 |
| IQ 2, 3 | 4 |
| IQ 4, 5, 6, 7 | 2 |

### 8.2 Output Queue Depths

| Queue ID | Depth (wavelets) |
|---|---|
| OQ 0, 1 | 2 |
| OQ 2, 3 | 6 |
| OQ 4, 5 | 2 |

### 8.3 Queue Assignment Strategy

- **High-throughput data**: Use IQ 0-1 (depth 6) for input, OQ 2-3 (depth 6) for output
- **Memcpy reserves**: IQ 0 and OQ 0 are typically used by the memcpy library
- **Application queues**: Start from IQ 2/OQ 2 for user operations

---

## 9. Complete Code Examples

### 9.1 Basic FIFO Pipeline (pipeline-02)

```csl
// H2D → FIFO → loopback color → add 1 → D2H
param memcpy_params: comptime_struct;
param size: i16;
param C1: color;

const h2d_iq: input_queue = @get_input_queue(2);
const d2h_oq: output_queue = @get_output_queue(3);
const C1_iq: input_queue = @get_input_queue(4);
const C1_oq: output_queue = @get_output_queue(5);

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

var fifo_buffer = @zeros([1024]i16);
const fifo = @allocate_fifo(fifo_buffer);

const INFINITE_DSD_LEN: u16 = 0x7fff;

task main_task() void {
  const h2d_dsd = @get_dsd(fabin_dsd, .{
    .extent = INFINITE_DSD_LEN,
    .fabric_color = sys_mod.MEMCPYH2D_1,
    .input_queue = h2d_iq,
  });
  const c1_out = @get_dsd(fabout_dsd, .{
    .extent = INFINITE_DSD_LEN,
    .fabric_color = C1,
    .output_queue = C1_oq,
  });
  const c1_in = @get_dsd(fabin_dsd, .{
    .extent = size,
    .fabric_color = C1,
    .input_queue = C1_iq,
  });
  const d2h_dsd = @get_dsd(fabout_dsd, .{
    .extent = size,
    .fabric_color = sys_mod.MEMCPYD2H_1,
    .output_queue = d2h_oq,
  });
  const one = [1]i16{1};
  const one_dsd = @get_dsd(mem1d_dsd, .{.tensor_access = |i|{size} -> one[0]});

  @mov16(fifo, h2d_dsd, .{ .async = true });       // H2D → FIFO
  @mov16(c1_out, fifo, .{ .async = true });         // FIFO → C1
  @add16(d2h_dsd, c1_in, one_dsd, .{ .async = true });  // C1 + 1 → D2H
}

comptime {
  @bind_local_task(main_task, @get_local_task_id(8));
  @activate(@get_local_task_id(8));
  if (@is_arch("wse3")) {
    @initialize_queue(h2d_iq, .{ .color = sys_mod.MEMCPYH2D_1 });
    @initialize_queue(d2h_oq, .{ .color = sys_mod.MEMCPYD2H_1 });
    @initialize_queue(C1_iq, .{ .color = C1 });
    @initialize_queue(C1_oq, .{ .color = C1 });
  }
}
```

### 9.2 Block/Unblock with FIFO (topic-09)

```csl
// Process elements one-at-a-time using data_task + block/unblock
task process_task(element: f32) void {
  @block(process_task_id);              // Prevent re-entry
  elem[0] = element * element * element;
  @fmovs(out_dsd, elem_dsd, .{
    .async = true,
    .unblock = process_task             // Auto-unblock on completion
  });
}
```

---

## 10. Best Practices and Common Pitfalls

### Best Practices

1. **Always initialize queues on WSE-3**: Every `input_queue` and `output_queue` needs `@initialize_queue` in a `comptime` block.

2. **Size FIFO buffers appropriately**: Larger FIFOs (1024+) prevent stalls in producer-consumer pipelines. Too small may cause deadlocks.

3. **Use `@is_arch` for portable code**: Wrap WSE-3-specific `@initialize_queue` and `@get_data_task_id` calls.

4. **Match queue IDs to depth needs**: Use deep queues (IQ 0-1, OQ 2-3) for high-throughput paths.

5. **Memcpy queue reservation**: Queues 0-1 are typically reserved for memcpy. Start user queues at 2.

### Common Pitfalls

1. **Missing `@initialize_queue` on WSE-3**: Most common WSE-3 compilation error.

2. **Two fabric inputs in one DSD op on WSE-3**: Use FIFO workaround.

3. **FIFO buffer type mismatch**: The FIFO element type must match the DSD operation type (i16 for `@mov16`, f32 for `@fmovs`).

4. **Forgetting `.async = true`**: FIFO operations must be async to enable concurrent producer/consumer.

5. **Queue exhaustion**: Only 8 input + 8 output queues available. Plan allocation carefully when composing multiple libraries.

6. **Deadlock from blocking send/receive**: Always use `nonblock=True` in Python when sending to multiple streams.

---

## 11. Queue Flush & Queue Reuse (WSE-3)

WSE-3 provides only 8 input queues and 8 output queues, each binding to exactly one color.
When an application needs more colors than available queues, `@queue_flush` allows **reusing
a queue** by rebinding it to a different color after the previous operation completes.

### Key Builtins

| Builtin | Signature | Description |
| --- | --- | --- |
| `@queue_flush` | `@queue_flush(queue)` | Monitors an output queue; fires T29 when queue becomes empty |
| `@set_empty_queue_handler` | `@set_empty_queue_handler(fn, queue)` | Registers `fn` as the T29 handler for the given output queue |

### Library: `queue_flush`

Import with `@import_module("<queue_flush/queue_flush>")`.

| Function | Description |
| --- | --- |
| `queue_flush.exit(oq)` | **Required** after reconfig — clears flush state so subsequent D2H teardown wavelets do not re-trigger the handler and stall the kernel |

### Output Queue Reuse Pattern (Sender)

Use `@queue_flush` + a handler when the **sender** needs to send on a second color after finishing on the first.

```csl
const queue_flush = @import_module("<queue_flush/queue_flush>");
const output_queue_config = @import_module("<queue_flush/output_queue_config>");

const output_queue_id: i16 = 2;
const oq = @get_output_queue(output_queue_id);
const out_color1 = @get_color(10);
const out_color2 = @get_color(12);
var switch_color: i16 = 0;

comptime {
  // 1. Init queue to first color
  @initialize_queue(oq, .{ .color = out_color1 });
  // 2. Register T29 handler
  @set_empty_queue_handler(foo, oq);
}

// 3. After first send completes, arm the flush monitor
fn done_transmit1() void {
  switch_color = @get_int(out_color2);
  @queue_flush(oq);  // T29 fires when queue drains
}

// 4. T29 handler: reconfigure → activate next → clear flush state
fn foo() void {
  output_queue_config.encode_output_queue(output_queue_id, switch_color);
  @activate(transmit2_id);
  queue_flush.exit(oq);  // MUST call — prevents D2H teardown from re-triggering
}
```

**Critical timing**: Call `@queue_flush(oq)` only **after** the async send (`@mov32` / `@fmovs`) has been posted — i.e., in the `.activate` callback of the send. If called while the queue is already empty, T29 fires immediately before any data is enqueued.

### Input Queue Reuse Pattern (Receiver)

No `@queue_flush` is needed on the receiver side. After an async receive completes, the input queue is naturally empty, so it is safe to immediately reconfigure it.

```csl
const input_queue_config = @import_module("<queue_flush/input_queue_config>");

const input_queue_id: i16 = 2;
const iq = @get_input_queue(input_queue_id);

comptime { @initialize_queue(iq, .{ .color = in_color1 }); }

// After receive1 completes (in its .activate callback):
fn done_receive1() void {
  // Input queue is empty after @mov32 — safe to rebind directly
  input_queue_config.encode_input_queue(input_queue_id, @get_int(in_color2));
  @activate(receive2_id);
}
```

### Canonical Example

`examples/updates/queue_flush/` — full 2-PE (sender + receiver) demonstration:

| File | Role |
| --- | --- |
| `queue_flush_layout.csl` | 2×1 layout, colors 10 & 12 |
| `queue_flush_sender.csl` | Output queue reuse with `@queue_flush` |
| `queue_flush_receiver.csl` | Input queue reuse without flush |
| `run.py` | `--arch=wse3`; validates `result1 == [42]*15`, `result2 == [43]*15` |

### Pitfalls (Queue Flush)

| Pitfall | Consequence | Fix |
| --- | --- | --- |
| Call `@queue_flush` before async send is posted | T29 fires on empty queue, skips data send | Call in `.activate` callback of the send, not before |
| Omit `queue_flush.exit(oq)` | D2H teardown wavelets re-trigger T29 → kernel stall | Always call `queue_flush.exit(oq)` at end of handler |
| Reconfigure input queue before receive completes | Race condition / incorrect data routing | Only reconfigure in `.activate` callback of the receive |

---

## Related Topics

- **DSD Operations**: See `CSL_DSD_Complete_Guide.md` for fabric DSD types and manipulation
- **Async Programming**: See `CSL_Async_Programming_Guide.md` for `.async`, `.activate`, `.unblock`
- **Routing**: See `CSL_Routing_Colors_Switches_Guide.md` for color routing and loopback setup
- **WSE-3 Migration**: See `CSL_WSE2_vs_WSE3_Migration_Guide.md` for queue constraints and init requirements
- **SdkLayout**: See `CSL_SdkLayout_Guide.md` for SdkLayout streaming (alternative to memcpy)
