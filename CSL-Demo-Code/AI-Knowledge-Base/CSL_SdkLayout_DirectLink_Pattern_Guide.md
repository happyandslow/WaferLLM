# CSL SdkLayout — Direct-Link Pattern Guide
## SDK Version: 1.4.0

This document covers **advanced SdkLayout usage**: the direct-link (point-to-point) streaming paradigm, the demux/mux fan-out/fan-in pattern for distributing H2D data across N PEs and collecting results, multi-host runtime setup, and router state management. These topics are NOT covered in `CSL_SdkLayout_Guide.md`.

**Source**: Derived from the `sdklayout-05-gemv` tutorial and the `bandwidth-test-parallel` implementation.

---

## Table of Contents

1. [Direct Link vs Memcpy — Two Paradigms](#1-direct-link-vs-memcpy)
2. [Demux/Mux Fan-Out/Fan-In Pattern](#2-demuxmux-pattern)
3. [Router State Machine — paint_all and SWITCH_ADV](#3-router-state-machine)
4. [Port Routing Position Semantics](#4-port-routing-position-semantics)
5. [Multi-Host SdkRuntime Setup](#5-multi-host-sdkruntime)
6. [Loopback Kernel and Deadlock Prevention](#6-loopback-and-deadlock)
7. [Single-Pass Constraint and Re-running](#7-single-pass-constraint)
8. [Color Scoping — region.color() vs Color()](#8-color-scoping)
9. [Complete Working Example — Loopback Bandwidth Test](#9-complete-example)
10. [Placement Coordinates](#10-placement-coordinates)

---

## 1. Direct Link vs Memcpy — Two Paradigms

The Cerebras SDK offers **two mutually exclusive** host-device communication paradigms:

### Memcpy Paradigm (traditional)
- CSL layout uses `@import_module("<memcpy/memcpy>")` and `@export_name`/`@export_symbol`
- Compilation uses `--memcpy --channels=N` flags
- Host calls `runtime.memcpy_h2d()` / `runtime.memcpy_d2h()`
- Reserves 3 west + 2 east PE columns for memcpy infrastructure
- Timing via on-device TSC (`sys_mod.unblock_cmd_stream()`) can be retrieved via `memcpy_d2h`
- **Single host only**: all I/O goes through one host process

### Direct Link Paradigm (SdkLayout)
- Layout is defined in Python using `SdkLayout` (no `layout.csl` file)
- Compilation via `layout.compile(out_prefix='out')`
- Host calls `runtime.send()` / `runtime.receive()` with `nonblock=True`
- **No memcpy infrastructure overhead**: more fabric columns available for compute
- **Multi-host capable**: each host can own separate FPGA links and `SdkRuntime` instances
- Runtime created with `memcpy_required=False`

```python
# Direct link runtime — key difference
runtime = SdkRuntime(artifacts, platform, memcpy_required=False)
```

---

## 2. Demux/Mux Pattern

To stream data from a single host I/O port to **N PEs** (and back), use the demux/mux infrastructure from `sdklayout-05-gemv`:

```
host ──[h2d_stream]──► in_adaptor(1×1) ──► in_demux(1×N) ──► core(1×N) ──► out_mux(1×N) ──[d2h_stream]──► host
```

### Roles of Each Region

| Region | Size | Role |
|---|---|---|
| `in_adaptor` | 1×1 | Receives full stream from host; injects `SWITCH_ADV` after each `batch_size` wavelets |
| `in_demux` | 1×N | Distributes: PE[0] gets first `batch_size`, PE[1] next, etc. |
| `core` | 1×N | Compute or loopback: each PE processes its `batch_size` wavelets independently |
| `out_mux` | 1×N | Serialises: PE[0] sends first `batch_size` to host, then PE[1], etc. |

### Why the Adaptor is Needed

A host I/O stream always enters through a **single PE port** (1 PE wide). To fan out to N PEs:
1. The adaptor forwards `batch_size` wavelets to PE[0] of the demux
2. Then sends a `SWITCH_ADV` control wavelet on its output color
3. The demux PE[0] receives the `SWITCH_ADV` on its `in_color` and advances its router from `NORTH→RAMP` (deliver to self) to `NORTH→SOUTH` (pass down)
4. Now the next `batch_size` wavelets flow to PE[1], etc.

### Python Helpers

```python
from demux import get_demux_adaptor, get_b_demux  # vertical distribution
from mux   import get_mux

# in_adaptor: 1×1, batch_size wavelets per batch, N batches total
(h2d_port, adaptor_out_port, adaptor) = get_demux_adaptor(layout, 'in_adaptor', batch_size, N)
adaptor.place(1, 0)

# in_demux: 1×N vertical fan-out
(demux_in_port, demux_out_port, demux) = get_b_demux(layout, 'in_demux', batch_size, 1, N)
demux.place(2, 0)
layout.connect(adaptor_out_port, demux_in_port)

# core: 1×N compute/loopback
# (your region)
core.place(3, 0)
layout.connect(demux_out_port, core_in_port)

# out_mux: 1×N vertical fan-in
(mux_in_port, d2h_port, mux) = get_mux(layout, 'out_mux', batch_size, 1, N)
mux.place(4, 0)
layout.connect(core_out_port, mux_in_port)

# Streams
h2d_stream = layout.create_input_stream(h2d_port)   # adaptor's LEFT port
d2h_stream = layout.create_output_stream(d2h_port)  # mux's TOP port
```

### Key Routing Details for `get_b_demux` (vertical)

```python
# in_color routing:
#   pos1 (initial):   NORTH → RAMP   (PE delivers to itself)
#   pos2 (after ADV): NORTH → SOUTH  (PE passes data down to next PE)
# Bottom PE is pinned to pos1 via get_edge_routing(Edge.BOTTOM, [pos1]) override
core_out_route = RoutingPosition().set_input([Route.NORTH]).set_output([Route.RAMP])
forward_route  = RoutingPosition().set_input([Route.NORTH]).set_output([Route.SOUTH])
edge_route     = get_edge_routing(Edge.BOTTOM, [core_out_route])
demux.paint_all(in_color, [core_out_route, forward_route], [edge_route])
```

### Key Routing Details for `get_mux` (vertical)

```python
# out_color routing:
#   pos1 (initial):   RAMP  → NORTH  (PE sends its own data northward to host)
#   pos2 (after ADV): SOUTH → NORTH  (PE forwards lower PEs' data northward)
core_out_route = RoutingPosition().set_input([Route.RAMP]).set_output([Route.NORTH])
forward_route  = RoutingPosition().set_input([Route.SOUTH]).set_output([Route.NORTH])
mux.paint_all(out_color, [core_out_route, forward_route])
```

The mux PE advances its own router after sending `batch_size` wavelets by sending `SWITCH_ADV` on `out_color` to itself (`ctrlOurDSD` with the same `out_color`).

---

## 3. Router State Machine — paint_all and SWITCH_ADV

`paint_all(color, [pos1, pos2], [edge_overrides])` defines a **router FSM** per PE:

- Each PE starts at `pos1` (initial routing state)
- When a `SWITCH_ADV` control wavelet arrives on `color`, the PE advances to `pos2`
- If there's a `pos3`, a second `SWITCH_ADV` advances to `pos3`
- Edge overrides (`get_edge_routing`) pin specific border PEs to fixed states

**Critical**: `SWITCH_ADV` is a control wavelet (`.control = true` on the `fabout_dsd`) carrying the opcode:
```csl
ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, true, {}, 0)
```

### Who sends SWITCH_ADV?

| Source | When | To whom |
|---|---|---|
| `demux_adaptor.csl` | After every `batch_size` data wavelets | Next PE's `in_color` (downstream in demux) |
| `mux.csl` | After forwarding its own `batch_size` wavelets | Itself (`ctrlOurDSD` on its own `out_color`) |

---

## 4. Port Routing Position Semantics

The `RoutingPosition` list passed to `create_input_port` / `create_output_port` describes **router states at the port boundary**.

### `create_input_port(color, edge, [routes...], size)`

The `routes` list describes routing states at the **boundary PE** (the PE at the edge where data enters). A list with 1 element means a fixed state; a list with 2+ elements means an FSM:

| Position | Meaning |
|---|---|
| `routes[0]` | Initial state: how data entering the edge is routed |
| `routes[1]` | State after first SWITCH_ADV |
| ... | etc. |

Example from `get_b_demux` (input at `Edge.TOP`):
```python
input_routes = RoutingPosition().set_output([Route.RAMP])   # data goes to RAMP
blah         = RoutingPosition().set_output([Route.SOUTH])  # after SWITCH_ADV: pass south
in_port = demux.create_input_port(in_color, Edge.TOP, [input_routes, blah], size)
```

### `create_output_port(color, edge, [routes...], size)`

Routes describe how data exits at the **output boundary PE**:

```python
output_routes       = RoutingPosition().set_input([Route.RAMP])   # from RAMP → exit
forward_port_routes = RoutingPosition().set_input([Route.SOUTH])  # from SOUTH → exit
out_port = mux.create_output_port(out_color, Edge.TOP, [output_routes, forward_port_routes], size)
```

### `layout.connect()` — Non-adjacent Connections

`layout.connect(tx_port, rx_port)` works regardless of physical adjacency:
- `tx_port` and `rx_port` can be on different edges of non-adjacent regions
- The SDK inserts fabric routing through unoccupied PEs automatically
- Connected ports share a single fabric color (the SDK unifies them)

**Example**: `adaptor.RIGHT` at (1,0) connecting to `demux.TOP` at (2,0) — geometrically a corner meeting, not a shared edge — works fine via `layout.connect()`.

---

## 5. Multi-Host SdkRuntime

When multiple hosts should each own separate I/O streams (future: parallel bandwidth testing), each host must run its own `SdkRuntime` instance:

1. **One host compiles** using `SdkLayout` → produces compiled artifacts
2. **Each host gets its own copy** of the compiled artifacts, configured with only the streams it owns
3. **One host calls `load()`** to program the wafer
4. **Each host calls `run()`** independently
5. **Each host calls `send()`/`receive()`** only on the streams it owns

```python
# Host A (H2D sender)
runtime_a = SdkRuntime(artifacts_a, platform, memcpy_required=False)
runtime_a.load()     # Only one host calls load()!
runtime_a.run()
runtime_a.send(h2d_stream, data, nonblock=True)
runtime_a.stop()

# Host B (D2H receiver — different process)
runtime_b = SdkRuntime(artifacts_b, platform, memcpy_required=False)
# runtime_b.load() — NOT called; A already loaded
runtime_b.run()
result = np.empty(size, dtype=np.float32)
runtime_b.receive(d2h_stream, result, size, nonblock=True)
runtime_b.stop()
```

**CRITICAL**: If two `SdkRuntime` instances try to claim the **same FPGA link (stream)**, the hardware will fail. Each host must compile a different artifact that declares only its own streams.

The "config file" mentioned by Cerebras engineers = the compiled artifact directory. It declares which FPGA links (streams) that host owns.

---

## 6. Loopback Kernel and Deadlock Prevention

### CSL Loopback Kernel

A simple loopback kernel buffers `pe_length` wavelets from `in_color` and echoes them on `out_color`:

```csl
param in_color:  color;
param out_color: color;
param pe_length: u16;

const input_q  = @get_input_queue(0);
const output_q = @get_output_queue(0);

var buf = @zeros([pe_length]f32);

const mem_dsd = @get_dsd(mem1d_dsd, .{.base_address = &buf, .extent = pe_length});
const in_dsd  = @get_dsd(fabin_dsd,  .{.extent = pe_length, .fabric_color = in_color,
                                        .input_queue = input_q});
const out_dsd = @get_dsd(fabout_dsd, .{.extent = pe_length, .fabric_color = out_color,
                                        .output_queue = output_q});

const recv_id = @get_local_task_id(8);
const send_id = @get_local_task_id(9);

task recv() void {
  @mov32(mem_dsd, in_dsd, .{.async = true, .activate = send_id});
}

task send() void {
  @mov32(out_dsd, mem_dsd, .{.async = true});
}

comptime {
  @bind_local_task(recv, recv_id);
  @bind_local_task(send, send_id);
  @activate(recv_id);
  @initialize_queue(input_q, .{.color = in_color});
  if (@is_arch("wse3")) {
    @initialize_queue(output_q, .{.color = out_color});
  }
}
```

### Deadlock Prevention — ALWAYS Use nonblock=True

In any loopback scenario, **both `send` and `receive` MUST use `nonblock=True`**:

```python
# CORRECT — both non-blocking
runtime.send(h2d_stream, data, nonblock=True)
runtime.receive(d2h_stream, result, total, nonblock=True)
runtime.stop()   # blocks here until both complete

# WRONG — DEADLOCK
runtime.send(h2d_stream, data)       # blocks until H2D completes
# But H2D can't complete because the device is waiting for the host
# to be ready on d2h_stream → DEADLOCK
```

**Why**: In a loopback, the device can only send D2H data after it receives H2D data. If the host blocks on `send()`, it never calls `receive()`, so the device never gets started — deadlock.

### Python Core Helper

```python
from cerebras.sdk.runtime.sdkruntimepybind import Edge, Route, RoutingPosition

def get_loopback_core(layout, name, pe_length, height):
    core = layout.create_code_region('./src/bw_loopback_kernel.csl', name, 1, height)
    core.set_param_all('pe_length', pe_length)

    in_color  = core.color('in_color')
    out_color = core.color('out_color')
    core.set_param_all(in_color)
    core.set_param_all(out_color)

    # Routing: WEST → RAMP (receive from demux to right)
    core.paint_all(in_color,  [RoutingPosition().set_input([Route.WEST]).set_output([Route.RAMP])])
    # Routing: RAMP → EAST (send to mux on right)
    core.paint_all(out_color, [RoutingPosition().set_input([Route.RAMP]).set_output([Route.EAST])])

    total    = pe_length * height
    in_port  = core.create_input_port( in_color,  Edge.LEFT,  [RoutingPosition().set_output([Route.RAMP])], total)
    out_port = core.create_output_port(out_color, Edge.RIGHT, [RoutingPosition().set_input([Route.RAMP])],  total)
    return (in_port, out_port, core)
```

---

## 7. Single-Pass Constraint and Re-running

**After one complete demux/mux pass, the router switch states are consumed:**
- All demux PEs except the bottom are in `forward_route` (NORTH→SOUTH = pass-through)
- All mux PEs are in `forward_route` (SOUTH→NORTH = pass-through)

**A second pass over the same router state would not distribute data correctly.**

### Re-running Correctly

`runtime.run()` **restarts the device program from its initial state**, including all `comptime` initializations and `@activate` calls. This resets the router states.

```python
runtime.load()              # load compiled binary once

for i in range(num_iterations):
    runtime.run()           # restarts the program; resets router states
    runtime.send(h2d_stream, data, nonblock=True)
    runtime.receive(d2h_stream, result, total, nonblock=True)
    runtime.stop()          # wait for this iteration to complete
    # measure timing, record results
```

This pattern gives **N independent timing measurements** without recompiling.

---

## 8. Color Scoping — region.color() vs Color()

Two ways to create colors for SdkLayout use:

### `Color('name')` — Global standalone
```python
my_color = Color('my_color')
region.set_param_all(my_color)         # uses 'my_color' as the CSL param name
# or:
region.set_param_all('param_name', my_color)  # explicit CSL param name
```

### `region.color('param_name')` — Region-scoped
```python
in_color = region.color('in_color')   # allocates a color AND associates 'in_color' as its name
region.set_param_all(in_color)        # sets CSL param named 'in_color' to this color value
```

**Key**: `set_param_all(color_obj)` uses `color_obj.name` to determine which CSL `param name: color` to set. When you call `region.color('in_color')`, the object's name is `'in_color'`, which must match the CSL param name.

**Practical rule**: For ports used in `layout.connect()`, always use `region.color('name')` and `set_param_all(color)` immediately afterward. The SDK will unify the physical color IDs at compile time to make connected ports share a color.

---

## 9. Complete Example — Loopback Bandwidth Test

Full working example: `bandwidth-test-parallel/` (in `examples/benchmarks/`).

### Architecture
```
host──[h2d]──► in_adaptor(1×1) @ (1,0)
                     │ SWITCH_ADV every pe_length wavelets
                     ▼
               in_demux(1×H)   @ (2,0)   ← vertical fan-out
                     │ pe_length wavelets per PE
                     ▼
               core(1×H)       @ (3,0)   ← loopback: recv→buf→send
                     │
                     ▼
               out_mux(1×H)    @ (4,0)   ← vertical fan-in
                     │
              ◄──[d2h]──── host
```

### `run_single.py` skeleton

```python
platform = get_platform(args.cmaddr, SimfabConfig(dump_core=True),
                        SdkTarget.WSE3 if args.arch=='wse3' else SdkTarget.WSE2)
layout = SdkLayout(platform)

(h2d_port, adaptor_out_port, adaptor) = get_demux_adaptor(layout, 'in_adaptor', pe_length, H)
adaptor.place(1, 0)

(demux_in_port, demux_out_port, demux) = get_b_demux(layout, 'in_demux', pe_length, 1, H)
demux.place(2, 0)
layout.connect(adaptor_out_port, demux_in_port)

(core_in_port, core_out_port, core) = get_loopback_core(layout, 'core', pe_length, H)
core.place(3, 0)
layout.connect(demux_out_port, core_in_port)

(mux_in_port, d2h_port, mux) = get_mux(layout, 'out_mux', pe_length, 1, H)
mux.place(4, 0)
layout.connect(core_out_port, mux_in_port)

h2d_stream = layout.create_input_stream(h2d_port)
d2h_stream = layout.create_output_stream(d2h_port)

artifacts = layout.compile(out_prefix='out')

runtime = SdkRuntime(artifacts, platform, memcpy_required=False)
runtime.load()
runtime.run()

data = np.arange(H * pe_length, dtype=np.float32)
result = np.empty(H * pe_length, dtype=np.float32)

t0 = time.perf_counter()
runtime.send(h2d_stream, data, nonblock=True)
runtime.receive(d2h_stream, result, H * pe_length, nonblock=True)
runtime.stop()
t1 = time.perf_counter()
```

---

## 10. Placement Coordinates

`region.place(x, y)` uses **layout-space coordinates**, NOT fabric coordinates.

The fabric offset (typically `(4, 1)` for WSE-3) is applied internally:
- Layout `(1, 0)` → fabric `(4+1, 1+0)` = `(5, 1)`
- Layout `(2, 0)` → fabric `(6, 1)`

**`SdkLayout.compile()` automatically calculates the required fabric dimensions** based on all placed regions. No need to specify `--fabric-dims` manually.

Minimum layout width for a 4-region column design (H PEs each, W=1):
- Columns: 1 (adaptor) + 1 (demux) + 1 (core) + 1 (mux) = 4 layout units → fabric width ≈ 10

---

## CSL Demux/Mux Kernel Reference

### `demux_adaptor.csl` — 1×1 SWITCH_ADV injector

```
Params: batch_size, num_batches, in_color, out_color
Logic:  For each batch [0..num_batches-1]:
          @mov32(outDSD, inDSD, async)    ; forward batch_size wavelets
          if not last: send SWITCH_ADV on out_color to advance demux
Uses:   local_task IDs 8 (main), 9 (send_ctrl)
```

### `demux.csl` — per-PE router with optional sentinel

```
Params: size, in_color, out_color, entry_point, has_sentinel
Logic:  @mov32(outDSD, inDSD, async, activate=send_sentinel)
        optionally sends a control task payload as sentinel
Uses:   local_task IDs 8 (main), 9 (send_sentinel)
```

### `mux.csl` — per-PE router with self-SWITCH_ADV

```
Params: size, in_color, out_color
Logic:  @mov32(outDSD, inDSD, async, activate=send_ctrl)
        send_ctrl: @mov32(ctrlOurDSD, SWITCH_ADV)  ; advance self
Note:   Uses output_q=1 (not 0, to separate from in_color's input_q=0)
Uses:   local_task IDs 8 (main), 9 (send_ctrl)
```

---

## Common Pitfalls

| Pitfall | Cause | Fix |
|---|---|---|
| Deadlock on loopback | `send()` blocking prevents `receive()` | Always `nonblock=True` on both |
| Wrong data on second run | Router state consumed after one pass | Call `runtime.run()` before each iteration |
| Two hosts claim same stream | Both `SdkRuntime` instances try to own the same FPGA link | Each host must use a different artifact that declares only its own streams |
| SWITCH_ADV not arriving | `adaptor.out_color` and `demux.in_color` are different fabric colors | `layout.connect()` unifies them at compile time; don't set explicit color values |
| Mux output queue conflict | mux.csl uses `output_q=1`; mismatching in CSL kernel | Follow the pattern: mux input_q=0, mux output_q=1 |
| pe_length too large | Memory per PE overflow | Keep `pe_length ≤ 4096` for f32 buffers (~16KB) |

---

## Related Topics

- `CSL_SdkLayout_Guide.md` — SdkLayout API fundamentals (prerequisite)
- `CSL_Routing_Colors_Switches_Guide.md` — Color routing, SWITCH_ADV, paint_all
- `CSL_Host_Device_Communication_Guide.md` — Memcpy paradigm (the alternative)
- Source: `examples/tutorials/sdklayout-05-gemv/` (demux/mux reference implementation)
- Source: `examples/benchmarks/bandwidth-test-parallel/` (loopback BW test)
