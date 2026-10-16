# Cerebras CSL Debugging Guide
## SDK Version: 1.4.0

This document covers all debugging tools for CSL programs: the `<debug>` library, `<simprint>`, CSDB debugger, simulator logs, and the SDK GUI visualizer.

---

## Table of Contents

1. [Overview](#1-overview)
2. [<debug> Library — Trace Buffers](#2-debug-library--trace-buffers)
3. [<simprint> Library — Simulator Console](#3-simprint-library--simulator-console)
4. [CSDB Debugger](#4-csdb-debugger)
5. [Simulator Logs (sim.log)](#5-simulator-logs-simlog)
6. [SDK GUI Visualizer](#6-sdk-gui-visualizer)
7. [Interpreting Logs with memcpy](#7-interpreting-logs-with-memcpy)
8. [Best Practices and Common Pitfalls](#8-best-practices-and-common-pitfalls)

---

## 1. Overview

| Tool | When to Use | Runs On |
|---|---|---|
| **`<debug>`** | Record values/timestamps into PE-local buffers | Simulator + Hardware |
| **`<simprint>`** | Printf-style output to simulator console | Simulator only |
| **CSDB** | Interactive post-mortem debugging, memory inspection | Hardware (coredump) + Simulator |
| **sim.log** | Cycle-by-cycle wavelet/instruction/router traces | Simulator only |
| **SDK GUI** | Visual timeline, routing, wavelet trace inspection | Post-simulation |

---

## 2. `<debug>` Library — Trace Buffers

### 2.1 Import and Configuration

Each import creates an independent trace buffer with a unique `key`:

```csl
const trace = @import_module("<debug>", .{
    .key = "trace",          // Unique key for this buffer
    .buffer_size = 100,      // Number of entries (u32 words)
});

const times = @import_module("<debug>", .{
    .key = "times",
    .buffer_size = 100,
});
```

### 2.2 API

| Function | Description |
|---|---|
| `trace.trace_string("msg")` | Record a comptime string marker |
| `trace.trace_u16(val)` | Record a u16 value |
| `trace.trace_u32(val)` | Record a u32 value |
| `trace.trace_f32(val)` | Record an f32 value |
| `trace.trace_timestamp()` | Record current cycle counter |

### 2.3 Example: Sender with Debug Traces

```csl
const trace = @import_module("<debug>", .{ .key = "trace", .buffer_size = 100 });
const times = @import_module("<debug>", .{ .key = "times", .buffer_size = 100 });

fn main_fn() void {
    trace.trace_string("Sender beginning main_fn");
    times.trace_timestamp();
    @fmovs(out_dsd, buf_dsd, .{ .async = true, .activate = exit_task });
}

task exit_task() void {
    trace.trace_string("Sender exiting");
    times.trace_timestamp();
    sys_mod.unblock_cmd_stream();
}
```

### 2.4 Example: Receiver with Per-Wavelet Tracing

```csl
const trace = @import_module("<debug>", .{ .key = "trace", .buffer_size = 100 });
const times = @import_module("<debug>", .{ .key = "times", .buffer_size = 100 });

task recv_task(in_data: u32) void {
    times.trace_timestamp();
    trace.trace_string("Start recv_task");

    global += 2 * in_data;
    trace.trace_u32(global);

    num_wlts_recvd += 1;
    if (num_wlts_recvd == num_elems) {
        sys_mod.unblock_cmd_stream();
    }
    times.trace_timestamp();
}
```

### 2.5 Viewing Debug Output

After simulation, debug buffer contents are accessible via the SDK GUI or CSDB:
- SDK GUI: Select PE → view debug traces in the debug pane
- CSDB: Use `memory read` to extract raw buffer contents

**Tutorial**: `examples/tutorials/topic-12-debug-library/`

---

## 3. `<simprint>` Library — Simulator Console

### 3.1 Import

```csl
const prt = @import_module("<simprint>");
```

**Important**: `<simprint>` only produces output in the **simulator** (not hardware).

### 3.2 API

| Function | Description |
|---|---|
| `prt.print_string("msg")` | Print string literal (no newline until `\n`) |
| `prt.print_u16_decimal(val)` | Print u16 in decimal |
| `prt.print_u32_decimal(val)` | Print u32 in decimal |
| `prt.print_f32(val)` | Print f32 value |

### 3.3 Buffering Rules

- Output is **buffered** — no output appears until `\n` is printed
- Build complex messages by chaining multiple `print_*` calls before a `\n`
- Each PE has its own output stream — interleaving is possible

### 3.4 Example: Print PE Coordinates

```csl
const prt = @import_module("<simprint>");
const layout_mod = @import_module("<layout>");

fn simprint_pe_coords() void {
    prt.print_string("PE(");
    prt.print_u16_decimal(layout_mod.get_x_coord());
    prt.print_string(",");
    prt.print_u16_decimal(layout_mod.get_y_coord());
    prt.print_string("): ");
}

task recv_task(in_data: u32) void {
    simprint_pe_coords();
    prt.print_string("recv_task: in_data = ");
    prt.print_u32_decimal(in_data);
    prt.print_string(", global = ");
    prt.print_u32_decimal(global);
    prt.print_string("\n");  // Flush output
}
```

Output in simulator:
```
PE(1,0): recv_task: in_data = 42, global = 84
PE(2,0): recv_task: in_data = 42, global = 84
```

**Tutorial**: `examples/tutorials/topic-13-simprint/`

---

## 4. CSDB Debugger

### 4.1 Overview

CSDB (Cerebras Software Language Debugger) provides interactive post-mortem debugging:
- Inspect PE memory from coredumps
- Read register values
- Trace wavelet flows
- Works with both simulator and hardware coredumps

### 4.2 Generating Coredumps

Add to `run.py` before `runner.stop()`:

```python
runner.dump_core("corefile.cs1")
```

This produces files: `corefile.cs1_0`, `corefile.cs1_1`, etc.

### 4.3 Starting CSDB

```bash
$ csdb .
(csdb) context select out
(csdb) target create --core-file=corefile.cs1
```

### 4.4 Key Commands

| Command | Usage | Description |
|---|---|---|
| `context select <dir>` | `context select out` | Select compile context (ELF directory) |
| `target create` | `--core-file=corefile.cs1` | Load coredump for inspection |
| `rectangle select` | `rectangle select 4,1,1,1` | Select PE rectangle (x,y,w,h) |
| `memory read` | `--address 0x9e0 --length 4` | Read PE memory (address/length in 16-bit units) |
| `trace select` | `trace select .` | Load simulation traces |
| `wavelet` | (after trace select) | Inspect wavelet traces |
| `settings` | | Show current configuration |

### 4.5 Interactive Hardware Debugging

```bash
(csdb) target create --cmaddr 12.34.56.78:9000
(csdb) target save-core my_coredump
```

### 4.6 Terminology

| Term | Meaning |
|---|---|
| **Compilation context** | Directory generated by `cslc` (default: `out`) |
| **Trace** | Directory with simulation traces (default: `simfab_traces`) |
| **Working directory** | Where CSDB writes output files |

---

## 5. Simulator Logs (sim.log)

### 5.1 Enabling Logs

Set the environment variable before running the simulator:

```bash
# Landing logs (wavelet arrivals)
SINGULARITYENV_SIMFABRIC_DEBUG=landing

# Instruction trace
SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace

# Router state / switch advances
SINGULARITYENV_SIMFABRIC_DEBUG=router
```

### 5.2 Landing Logs

Show wavelet arrivals at each PE's router:

```
@53 P6.1 (hwtile) landing C3 from link R, ctrl=0, idx=0000, data=0000 (+0.000(-15)), half=0, ident=00000E0300000000, lf=0
@55 P5.1 (hwtile) landing C3 from link E, ctrl=0, idx=0000, data=0000 (+0.000(-15)), half=0, ident=00000E0300000000, lf=0
```

**Field reference**:

| Field | Meaning |
|---|---|
| `@53` | Cycle number (0-based) |
| `P6.1` | PE coordinates (X=6, Y=1) |
| `(hwtile)` | Tile type: `hwtile` (compute) or `iotile` (I/O edge) |
| `C3` | Color ID |
| `link R` | Arrived from: `E`(East), `W`(West), `N`(North), `S`(South), `R`(Ramp/CE) |
| `ctrl=0` | 0=data wavelet, 1=control wavelet |
| `idx=...` | 16-bit index field |
| `data=...` | 16-bit data field (also shown as fp16) |
| `half=0` | WSE-3 only: half-wavelet flag |
| `ident=...` | Unique wavelet ID — same across PEs for tracking flow |
| `lf=0` | Local flip bit (CE signals switch advance) |

### 5.3 Instruction Trace Logs

Show per-cycle instruction execution:

```
@497 P4.1: Id: 12, Instr: 225, Seq: 0, Pipe: 3, Msg: [IS OP] 0x021c: T01 FMACS Dest:[DDS1] Src0:[S0DS1] Src1:[S1DS1] Src2:R13,R12
@500 P4.1: Id: 12, Instr: 225, Seq: 0, Pipe: 6, Msg: [EX OP] 0x021c: T01 FMACS Dest:3f800000 Src0:00000000 Src1:3f800000 Src2:3f800000 [     ] U0
```

**Field reference**:

| Field | Meaning |
|---|---|
| `Id: 12` | PE position in 1D array |
| `Instr: 225` | Unique instruction ID (same from decode to execution) |
| `Seq: 0` | Sequence number (vector element index, 0 to N-1) |
| `Pipe: 3/6` | Pipeline stage: 3=decode (WSE-3), 6=execute (WSE-3); 2/4 on WSE-2 |
| `[IS OP]` | Instruction decode stage |
| `[EX OP]` | Instruction execution stage |
| `T01` | Task ID; `.UT4` suffix = microthread ID |
| `FMACS` | Disassembled instruction name |
| `[     ]` | Error flags: `u`=underflow, `o`=overflow, `x`=inexact, `i`=invalid, `z`=div-by-zero |
| `U0` | SIMD unit (SIMD-1 to SIMD-8 on WSE-3) |

### 5.4 Router Logs

Show switch state and advances:

```
@0 P5.2 (hwtile) router: C0 : input switch: 1=/    R/ -> 1=/    R/ (init)
@0 P5.2 (hwtile) router: C0 : output switch 1=/  W  / -> 1=/  W  / (init)
@376 P5.2 (hwtile) router: C0 : input switch: 1=/    R/ -> 2=/    R/ (advance)
@376 P5.2 (hwtile) router: C0 : output switch 1=/  W  / -> 2=/E    / (advance)
```

Shows initial switch positions and positions after control wavelet advances.

---

## 6. SDK GUI Visualizer

### 6.1 Launch

```bash
$ sdk_debug_shell visualize --artifact_dir /path/to/run/directory
```

Opens a web-based GUI for visual inspection.

### 6.2 Features

| Panel | Shows |
|---|---|
| **Fabric view** | PE grid with color routes overlaid |
| **Routes and colors** | Select colors to display routing arrows |
| **Instruction trace** | Instructions, Tasks, Statistics, or Micro-threads per PE |
| **Source code** | Click an instruction to jump to source line |
| **Wavelet trace** | Wavelet data per color and PE |
| **Timeline (wavelet)** | Horizontal bars: green=active, yellow=delayed, red=backpressure |
| **Timeline (instruction)** | Per-task execution bars, color-coded by task |
| **Timeline (combined)** | Both wavelet + instruction traces overlaid |
| **Debug pane** | Hover over timeline items for detail |

### 6.3 Key Interactions

- **Double-click PE**: Open instruction trace, source code, wavelet trace panels
- **Mouse scroll**: Zoom fabric view
- **Drag**: Navigate fabric
- **Re-center**: Click Re-center button to reset view
- **PE Coordinate**: Enter specific coordinates to jump to a PE
- **Timeline navigation**: Drag the red bar to scroll; +/- buttons to zoom

### 6.4 Timeline Legend

| Status | Color | Meaning |
|---|---|---|
| Active | Green | Wavelet sent without backpressure |
| Delayed | Yellow | Router output congested (multiple inputs) |
| Backpressure | Red | Destination PE busy, sender blocked |
| Idle | Empty | No wavelet activity |
| Control | Blue | Control wavelet (switch advance, etc.) |

### 6.5 Settings Dialog

Accessed via Submit button after entering a new work directory:
- **Compile directory** (required): Contains ELF binaries
- **Run directory**: Contains simulation traces (needed for timeline)
- **Core file path**: For route display from actual execution
- **Source directory**: For source code navigation

### 6.6 Scalable Fabric View

For fabrics > 20×20:
- Right panel: Full dimensional view
- Left panel: Zoomed view of the dark square selected on right
- Navigation: Left/Right/Top/Bottom buttons + Re-corner button

---

## 7. Interpreting Logs with memcpy

Programs using `memcpy` have additional hidden infrastructure:

| Resource | Used By memcpy |
|---|---|
| **Fabric offset** | Top-left PE of user rectangle is typically at `P4.1` |
| **Colors** | 21 (D2H), 22 (command), 23 (H2D) |
| **Local task IDs** | 27, 28, 30 |
| **Control task IDs** | 33, 34, 35, 36, 37 |
| **Microthread** | UT0 |
| **Input queue** | 0 (also 1 on WSE-3) |
| **Output queue** | 0 |

### Task ID Mapping

- **WSE-2**: Functions launched via memcpy execute within `T22` (bound to color 22)
- **WSE-3**: Functions execute within `T01` (bound to input queue 1)

### User Resource Availability

| Resource | User-Safe Range (WSE-2) | User-Safe Range (WSE-3) |
|---|---|---|
| **Colors** | 0-20, 24 | 0-20, 24 |
| **Local task IDs** | 8-26, 29, 31-32 | 8-26, 29, 31-32 |
| **Input queues** | 1-7 | 2-7 |
| **Output queues** | 1-7 | 1-7 |

---

## 8. Best Practices and Common Pitfalls

### Best Practices

1. **Use `<debug>` for hardware debugging**: Unlike `<simprint>`, it works on real hardware too.
2. **Multiple debug instances**: Use separate keys (`"trace"`, `"times"`, `"perf"`) for different concerns.
3. **Start with landing logs**: Best first step to diagnose wavelet-level issues.
4. **Use `ident` to trace wavelets**: Same `ident` value across PEs = same physical wavelet.
5. **Check `[EX OP]` error flags**: `o`=overflow and `z`=div-by-zero are common bugs.
6. **Use SDK GUI combined view**: Correlates wavelet arrivals with instruction execution.

### Common Pitfalls

1. **Using `<simprint>` on hardware**: No output — simulator only. Use `<debug>` instead.
2. **Forgetting `\n` with simprint**: Output is buffered; nothing prints without newline.
3. **Large `buffer_size` in `<debug>`**: Consumes precious PE memory. Start with 100-200.
4. **Misreading PE coordinates**: Remember memcpy offsets; your (0,0) is typically `P4.1`.
5. **Expecting `csdb` on legacy CS-2**: Not supported on Cerebras software version 1.6 or lower.
6. **Forgetting to dump core**: Must call `runner.dump_core("corefile.cs1")` before `runner.stop()`.

---

## Related Topics

- **Routing & Colors**: See `CSL_Routing_Colors_Switches_Guide.md` for color routing visible in SDK GUI
- **Builtins**: See `CSL_Builtins_Reference_Guide.md` for `@assert` and trap instructions
- **Standard Libraries**: See `CSL_Standard_Libraries_Guide.md` for `<debug>`, `<simprint>`, `<time>` library details
- **Host Communication**: See `CSL_Host_Device_Communication_Guide.md` for memcpy infrastructure
- **Async Programming**: See `CSL_Async_Programming_Guide.md` for understanding task execution in traces
