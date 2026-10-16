# Cerebras CSL Routing, Colors & Switches Complete Guide
## SDK Version: 1.4.0

This document covers the fabric routing system in Cerebras CSL, including colors, route configuration, switches, filters, control wavelets, sentinels, color swapping, and torus routing patterns.

---

## Table of Contents

1. [Colors and Route Basics](#1-colors-and-route-basics)
2. [Route Configuration](#2-route-configuration)
3. [Checkerboard Color Alternation](#3-checkerboard-color-alternation)
4. [Switches](#4-switches)
5. [Control Wavelets](#5-control-wavelets)
6. [Sentinels](#6-sentinels)
7. [Filters](#7-filters)
8. [Color Swapping](#8-color-swapping)
9. [Torus Routing](#9-torus-routing)
10. [Complete Code Examples](#10-complete-code-examples)
11. [API Quick Reference](#11-api-quick-reference)
12. [Best Practices and Common Pitfalls](#12-best-practices-and-common-pitfalls)
13. [Related Topics](#13-related-topics)

---

## 1. Colors and Route Basics

### 1.1 What Are Colors?

Colors are the fundamental routing abstraction in CSL. Each color represents a logical communication channel on the fabric. Wavelets (data packets carrying 32 bits of payload) travel along colors between PEs.

### 1.2 Color Allocation

```csl
// Allocate a color by ID
const my_color: color = @get_color(0);

// Colors from compile-time params (used for memcpy streaming)
param MY_COLOR_ID: i16;
const my_color: color = @get_color(MY_COLOR_ID);
```

**Reserved colors**: 21, 22, 23 are reserved by the memcpy infrastructure. Do not use them.

### 1.3 Direction Constants

CSL provides five direction constants for routing:
- `NORTH` — toward row 0 (up)
- `SOUTH` — toward higher rows (down)
- `EAST` — toward higher columns (right)
- `WEST` — toward lower columns (left)
- `RAMP` — into/out of the PE's compute element (CE)

### 1.4 Fabric DSDs

Colors are used through fabric DSDs:

```csl
// Send data out through the fabric
const out_dsd = @get_dsd(fabout_dsd, .{
    .fabric_color = my_color,
    .extent = N,
    .output_queue = my_oq
});

// Receive data from the fabric
const in_dsd = @get_dsd(fabin_dsd, .{
    .fabric_color = my_color,
    .extent = N,
    .input_queue = my_iq
});
```

---

## 2. Route Configuration

### 2.1 @set_color_config in Layout

Routes are configured **per-PE, per-color** in the layout file using `@set_color_config`:

```csl
@set_color_config(pe_x, pe_y, color, .{
    .routes = .{
        .rx = .{ DIRECTION },      // Where wavelets come FROM
        .tx = .{ DIRECTION, ... }  // Where wavelets go TO
    }
});
```

### 2.2 Common Route Patterns

```csl
// Receive from RAMP (CE), send EAST
.routes = .{ .rx = .{ RAMP }, .tx = .{ EAST } }

// Receive from WEST, deliver to RAMP (CE)
.routes = .{ .rx = .{ WEST }, .tx = .{ RAMP } }

// Receive from RAMP, send to RAMP and SOUTH (broadcast down column)
.routes = .{ .rx = .{ RAMP }, .tx = .{ RAMP, SOUTH } }

// Receive from NORTH, forward to RAMP and SOUTH (relay + consume)
.routes = .{ .rx = .{ NORTH }, .tx = .{ RAMP, SOUTH } }

// Receive from NORTH, deliver to RAMP only (last in column)
.routes = .{ .rx = .{ NORTH }, .tx = .{ RAMP } }

// Pass-through: WEST → EAST (no RAMP, PE doesn't consume)
.routes = .{ .rx = .{ WEST }, .tx = .{ EAST } }

// Pass-through with consumption: WEST → RAMP + EAST
.routes = .{ .rx = .{ WEST }, .tx = .{ RAMP, EAST } }
```

### 2.3 Queue Initialization (WSE-3)

On WSE-3, all queues used with fabric DSDs must be explicitly initialized:

```csl
const my_oq: output_queue = @get_output_queue(2);
const my_iq: input_queue  = @get_input_queue(2);

comptime {
    if (@is_arch("wse3")) {
        @initialize_queue(my_oq, .{ .color = my_color });
        @initialize_queue(my_iq, .{ .color = my_color });
    }
}
```

### 2.4 Simple Two-PE Routing Example

```csl
// layout.csl — Left PE sends EAST, right PE receives from WEST
const send_color: color = @get_color(0);

layout {
    @set_rectangle(2, 1);
    @set_tile_code(0, 0, "pe_program.csl", .{ .send_color = send_color, .pe_id = 0 });
    @set_tile_code(1, 0, "pe_program.csl", .{ .send_color = send_color, .pe_id = 1 });

    // Left PE: CE → EAST
    @set_color_config(0, 0, send_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
    // Right PE: WEST → CE
    @set_color_config(1, 0, send_color, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } });
}
```

---

## 3. Checkerboard Color Alternation

### 3.1 The Problem

When data flows EAST through multiple PEs on the same color, a PE cannot simultaneously receive from WEST and send from RAMP on the same color. The solution: use **two alternating colors**.

### 3.2 The Pattern

Even-column PEs use `color_1` to send, `color_2` to receive.
Odd-column PEs use `color_2` to send, `color_1` to receive.

```csl
// layout.csl
const ax_color_1: color = @get_color(3);
const ax_color_2: color = @get_color(4);

const even_params = .{ .send_east_color = ax_color_1, .recv_west_color = ax_color_2 };
const odd_params  = .{ .send_east_color = ax_color_2, .recv_west_color = ax_color_1 };

for (@range(i16, kernel_x_dim)) |pe_x| {
    for (@range(i16, kernel_y_dim)) |pe_y| {
        if (pe_x == 0) {
            // First column: only sends, no receive from west
            @set_color_config(pe_x, pe_y, ax_color_1, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
            @set_color_config(pe_x, pe_y, ax_color_2, .{ .routes = .{ .rx = .{RAMP}, .tx = .{RAMP} } });
        } else if (pe_x % 2 == 0) {
            @set_color_config(pe_x, pe_y, ax_color_1, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
            @set_color_config(pe_x, pe_y, ax_color_2, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } });
        } else {
            @set_color_config(pe_x, pe_y, ax_color_1, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } });
            @set_color_config(pe_x, pe_y, ax_color_2, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
        }
    }
}
```

### 3.3 Column Broadcast Pattern

For broadcasting data down a column (e.g., vector x in GEMV):

```csl
// Top row: receive from RAMP, send to RAMP + SOUTH
@set_color_config(pe_x, 0, x_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{RAMP, SOUTH} } });

// Middle rows: receive from NORTH, send to RAMP + SOUTH
@set_color_config(pe_x, pe_y, x_color, .{ .routes = .{ .rx = .{NORTH}, .tx = .{RAMP, SOUTH} } });

// Bottom row: receive from NORTH, deliver to RAMP only
@set_color_config(pe_x, last_y, x_color, .{ .routes = .{ .rx = .{NORTH}, .tx = .{RAMP} } });
```

### 3.4 2D Grid Example (gemv-07)

A simple 2x2 grid routing without checkerboard (when only one hop EAST):

```csl
// Left column sends EAST, right column receives from WEST
@set_color_config(0, py, ax_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
@set_color_config(1, py, ax_color, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } });

// Top row broadcasts x SOUTH, bottom row receives from NORTH
@set_color_config(px, 0, x_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{RAMP, SOUTH} } });
@set_color_config(px, 1, x_color, .{ .routes = .{ .rx = .{NORTH}, .tx = .{RAMP} } });
```

---

## 4. Switches

### 4.1 Overview

Switches allow **dynamic route changes** at runtime. A color's transmit direction can be changed by receiving control wavelets. This enables sending data to different neighbors using a single color.

### 4.2 Switch Configuration

Switches are configured in `@set_color_config` alongside routes:

```csl
const sender_routes = .{
    .rx = .{ RAMP },
    .tx = .{ NORTH }    // Default (position 0) direction
};

const sender_switches = .{
    .pos1 = .{ .tx = WEST },    // After 1st control wavelet → send WEST
    .pos2 = .{ .tx = EAST },    // After 2nd control wavelet → send EAST
    .pos3 = .{ .tx = SOUTH },   // After 3rd control wavelet → send SOUTH
    .current_switch_pos = 1,     // Start at position 1 (WEST)
    .ring_mode = true,           // Wrap from pos3 back to pos0
};

@set_color_config(1, 1, channel, .{
    .routes = sender_routes,
    .switches = sender_switches
});
```

### 4.3 Switch Positions

There are 4 switch positions (0-3):
- **Position 0**: Default route (from `.routes.tx`)
- **Position 1**: `.pos1.tx` direction
- **Position 2**: `.pos2.tx` direction
- **Position 3**: `.pos3.tx` direction

Each `SWITCH_ADV` control wavelet advances the position by 1.

### 4.4 Ring Mode

- `.ring_mode = true`: After position 3, wraps back to position 0
- `.ring_mode = false`: Stays at position 3 (default)

### 4.5 Advancing Switches

To advance a switch, send a `SWITCH_ADV` control wavelet:

```csl
const ctrl = @import_module("<control>");

const tx_ctrl_dsd = @get_dsd(fabout_dsd, .{
    .extent = 1,
    .fabric_color = channel,
    .control = true,        // This DSD sends CONTROL wavelets
    .output_queue = tx_oq
});

const tx_data_dsd = @get_dsd(fabout_dsd, .{
    .extent = 1,
    .fabric_color = channel,
    .output_queue = tx_oq
});

// Send: advance switch, then send data. Repeat for each direction.
fn send_to_neighbors() void {
    const switch_adv = ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, true, {}, 0);

    @mov32(tx_ctrl_dsd, switch_adv);  // Advance to next position
    @mov32(tx_data_dsd, 0);           // Send data in new direction

    @mov32(tx_ctrl_dsd, switch_adv);  // Advance again
    @mov32(tx_data_dsd, 2);           // Send data

    @mov32(tx_ctrl_dsd, switch_adv);
    @mov32(tx_data_dsd, 4);

    @mov32(tx_ctrl_dsd, switch_adv);
    @mov32(tx_data_dsd, 6);
}
```

---

## 5. Control Wavelets

### 5.1 Overview

Control wavelets are special wavelets that modify router behavior rather than carrying data to the CE. They use the same color infrastructure but are flagged with `.control = true`.

### 5.2 Control DSD

```csl
const ctrl_dsd = @get_dsd(fabout_dsd, .{
    .extent = 1,
    .fabric_color = my_color,
    .control = true,          // Mark as control wavelet
    .output_queue = my_oq
});
```

### 5.3 Control Opcodes

From `<control>` library:
- `SWITCH_ADV`: Advance switch position by 1
- `TEARDOWN`: Reset switch to position 0
- Others: see `source-code/csl_libraries/control.csl`

### 5.4 Encoding Control Payloads

```csl
const ctrl = @import_module("<control>");

// Simple SWITCH_ADV with no CE forwarding
const payload = ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, true, {}, 0);

// Control wavelet that triggers a control task on the receiver
const sentinel_task_id: control_task_id = @get_control_task_id(42);
const payload = ctrl.encode_control_task_payload(sentinel_task_id);
```

---

## 6. Sentinels

### 6.1 Overview

Sentinels are control wavelets used to signal **end-of-stream**. A sending PE sends data wavelets followed by a sentinel control wavelet. The receiving PE binds a control task to detect the sentinel and take action (e.g., start computation, send results).

### 6.2 Sentinel Setup

**Layout**:
```csl
const end_computation: u16 = 43;  // Arbitrary sentinel ID

@set_tile_code(0, 0, "sender.csl", .{ .sentinel = end_computation });
@set_tile_code(1, 0, "receiver.csl", .{ .sentinel = end_computation });
```

**Sender**:
```csl
param sentinel: u16;

const ctrl = @import_module("<control>");
const sentinel_task_id: control_task_id = @get_control_task_id(sentinel);

const fab_data_dsd = @get_dsd(fabout_dsd, .{
    .extent = 1, .fabric_color = main_color, .output_queue = main_oq
});

const fab_ctrl_dsd = @get_dsd(fabout_dsd, .{
    .extent = 1, .fabric_color = main_color, .output_queue = main_oq,
    .control = true
});

// After sending all data, send a sentinel
task send_data(data: u32) void {
    @mov32(fab_data_dsd, data);
    index += 1;
    if (index >= num_wvlts) {
        const ctrl_wvlt = ctrl.encode_control_task_payload(sentinel_task_id);
        @mov32(fab_ctrl_dsd, ctrl_wvlt);
    }
}
```

**Receiver** (in pe_program.csl):
```csl
param sentinel: u16;
const sentinel_task_id: control_task_id = @get_control_task_id(sentinel);

task sentinel_handler() void {
    // All data received — now process or send results
}

comptime {
    @bind_control_task(sentinel_handler, sentinel_task_id);
}
```

---

## 7. Filters

### 7.1 Overview

Filters allow PEs to selectively accept wavelets based on an **index field** embedded in each wavelet. This enables a single stream to deliver different data to different PEs.

### 7.2 Filter Configuration

Filters are set in `@set_color_config`:

```csl
const filter = .{
    .kind = .{ .range = true },
    .min_idx = pe_id * 3,       // Accept wavelets with index >= min_idx
    .max_idx = pe_id * 3 + 2,   // Accept wavelets with index <= max_idx
};

@set_color_config(pe_id, 0, data_color, .{
    .routes = .{ .rx = .{WEST}, .tx = .{RAMP, EAST} },
    .filter = filter
});
```

### 7.3 Filter Types

| Kind | Behavior |
|---|---|
| `.range = true` | Accept wavelets with index in `[min_idx, max_idx]` |

### 7.4 Example: Distributing Data to Multiple PEs (topic-08)

One sender sends 9 wavelets EAST. Each of 3 receivers filters 3 wavelets:

```csl
// layout.csl
@set_tile_code(0, 0, "send.csl", .{ .exch_color = data_color });
@set_color_config(0, 0, data_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });

for (@range(u16, 1, 4, 1)) |pe_id| {
    const filter = .{
        .kind = .{ .range = true },
        .min_idx = pe_id * 3,
        .max_idx = pe_id * 3 + 2,
    };

    if (pe_id == 3) {
        // Last PE: no further forwarding
        @set_color_config(pe_id, 0, data_color, .{
            .routes = .{ .rx = .{WEST}, .tx = .{RAMP} },
            .filter = filter
        });
    } else {
        // Middle PEs: consume + forward EAST
        @set_color_config(pe_id, 0, data_color, .{
            .routes = .{ .rx = .{WEST}, .tx = .{RAMP, EAST} },
            .filter = filter
        });
    }
}
```

---

## 8. Color Swapping

### 8.1 Overview

Color swapping allows incoming data on one color to trigger a **different color's** data task. Two colors are swappable if they differ only in the low bit (e.g., colors 0 and 1, colors 2 and 3).

### 8.2 How It Works

Given a swappable pair (e.g., `red=0`, `blue=1`) with `.color_swap_x = true`:

- Wavelet arriving from EAST/WEST on `red` → received on **both** `red` and `blue`
- If **both** colors have swap enabled → `red` wavelets go to `blue` task, `blue` wavelets go to `red` task (full swap)

### 8.3 Configuration

```csl
const red:  color = @get_color(0);
const blue: color = @get_color(1);

// Enable color_swap_x on both colors
@set_color_config(pe_x, 0, blue, .{
    .routes = .{ .rx = .{WEST}, .tx = .{RAMP, EAST}, .color_swap_x = true }
});
@set_color_config(pe_x, 0, red, .{
    .routes = .{ .rx = .{WEST}, .tx = .{RAMP, EAST}, .color_swap_x = true }
});
```

### 8.4 Use Case: Ping-Pong Communication

Color swap enables alternating data processing across PEs without additional routing logic. Even PEs activate `blue_task`, odd PEs activate `red_task`, using the same wavelet stream.

### 8.5 Swap Axes

| Setting | Triggers on wavelets from |
|---|---|
| `.color_swap_x = true` | EAST or WEST |
| `.color_swap_y = true` | NORTH or SOUTH |
| Both enabled | Also applies to RAMP (CE) |

### 8.6 Example (topic-14)

```csl
// pe_program.csl — Both tasks bound, color swap decides which fires
task red_task(in_data: u32) void {
    sum[0] += in_data;
}
task blue_task(in_data: u32) void {
    sum[0] += in_data * 2;
}

comptime {
    @bind_data_task(red_task, red_task_id);
    @bind_data_task(blue_task, blue_task_id);
}
```

---

## 9. Torus Routing

### 9.1 Overview

Torus routing creates wrap-around connections where the last PE in a row/column connects back to the first, forming a ring topology. This is useful for algorithms like histograms where data needs to circulate.

### 9.2 Canonical Example

See `benchmarks/histogram-torus/routes.csl` for a complete torus routing implementation.

### 9.3 Key Concept

In a torus, the rightmost PE sends EAST, and the routing wraps around so the leftmost PE receives from EAST (or equivalently, the data re-enters from WEST after wrapping). The same applies vertically.

---

## 10. Complete Code Examples

### 10.1 Basic Two-PE Communication (gemv-06)

**Layout**: Left PE sends result EAST, right PE receives from WEST.

```csl
// layout.csl
const send_color: color = @get_color(0);

layout {
    @set_rectangle(2, 1);
    @set_color_config(0, 0, send_color, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST} } });
    @set_color_config(1, 0, send_color, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } });
}
```

**PE Program**: Send via `fabout_dsd`, receive via `fabin_dsd`.

```csl
// pe_program.csl
fn send_right() void {
    const out_dsd = @get_dsd(fabout_dsd, .{
        .fabric_color = send_color, .extent = M,
        .output_queue = send_color_oq
    });
    @fmovs(out_dsd, y_dsd, .{ .async = true, .activate = exit_task_id });
}

fn recv_left() void {
    const in_dsd = @get_dsd(fabin_dsd, .{
        .fabric_color = send_color, .extent = M,
        .input_queue = send_color_iq
    });
    @fadds(y_dsd, y_dsd, in_dsd, .{ .async = true, .activate = exit_task_id });
}
```

### 10.2 Switches: Sending to 4 Neighbors (topic-06)

Center PE sends data to N/W/E/S neighbors using switches:

```csl
// layout.csl — switch config
const sender_switches = .{
    .pos1 = .{ .tx = WEST },
    .pos2 = .{ .tx = EAST },
    .pos3 = .{ .tx = SOUTH },
    .current_switch_pos = 1,
    .ring_mode = true,
};
@set_color_config(1, 1, channel, .{
    .routes = .{ .rx = .{RAMP}, .tx = .{NORTH} },
    .switches = sender_switches
});

// send.csl — advance switch before each data wavelet
const switch_adv = ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, true, {}, 0);
@mov32(tx_ctrl_dsd, switch_adv);
@mov32(tx_data_dsd, value);
```

### 10.3 Filters: Selective Wavelet Reception (topic-08)

```csl
// Each PE accepts 3 wavelets from a stream of 9
const filter = .{
    .kind = .{ .range = true },
    .min_idx = pe_id * 3,
    .max_idx = pe_id * 3 + 2,
};
@set_color_config(pe_id, 0, data_color, .{
    .routes = .{ .rx = .{WEST}, .tx = .{RAMP, EAST} },
    .filter = filter
});
```

---

## 11. API Quick Reference

### 11.1 Layout-Side Configuration

| API | Purpose |
|---|---|
| `@get_color(id)` | Allocate a color by numeric ID |
| `@set_color_config(x, y, color, config)` | Set routes/switches/filters for a color on a PE |
| `.routes = .{ .rx = .{...}, .tx = .{...} }` | Define receive and transmit directions |
| `.switches = .{ .pos1, .pos2, .pos3, .ring_mode, .current_switch_pos }` | Configure switch positions |
| `.filter = .{ .kind, .min_idx, .max_idx }` | Configure wavelet filtering |
| `.color_swap_x = true` | Enable E-W color swapping |
| `.color_swap_y = true` | Enable N-S color swapping |

### 11.2 PE-Side Fabric DSDs

| DSD Type | Purpose | Key Fields |
|---|---|---|
| `fabout_dsd` | Send wavelets to fabric | `.fabric_color`, `.extent`, `.output_queue`, `.control` |
| `fabin_dsd` | Receive wavelets from fabric | `.fabric_color`, `.extent`, `.input_queue` |

### 11.3 Control Library

| API | Purpose |
|---|---|
| `@import_module("<control>")` | Import control wavelet library |
| `ctrl.encode_single_payload(opcode, ...)` | Encode a control wavelet payload |
| `ctrl.encode_control_task_payload(task_id)` | Encode sentinel/control task payload |
| `ctrl.opcode.SWITCH_ADV` | Advance switch position |
| `ctrl.opcode.TEARDOWN` | Reset switch to position 0 |

### 11.4 Directions

| Direction | Meaning |
|---|---|
| `NORTH` | Toward row 0 (up) |
| `SOUTH` | Toward higher rows (down) |
| `EAST` | Toward higher columns (right) |
| `WEST` | Toward lower columns (left) |
| `RAMP` | Into/out of PE compute element |

---

## 12. Best Practices and Common Pitfalls

### Best Practices

1. **Use checkerboard alternation** for multi-hop horizontal data flow to avoid routing conflicts
2. **Always terminate a column broadcast** — the last PE should NOT have `SOUTH` in `.tx`
3. **Use sentinels** to signal end-of-stream instead of counting wavelets when the stream length varies
4. **Use switches sparingly** — they add complexity; prefer separate colors when possible
5. **Test with 2 PEs first**, then scale to NxM grids
6. **On WSE-3, always initialize queues** for all fabric DSDs

### Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Using memcpy-reserved colors (21-23) | Undefined behavior | Use other color IDs |
| Missing `RAMP` in `.tx` | PE doesn't consume its own data | Add `RAMP` to transmit when PE needs to process data |
| Sending and receiving on same color without checkerboard | Routing deadlock or corruption | Use two alternating colors (checkerboard pattern) |
| Forgetting `@initialize_queue` on WSE-3 | WSE-3 runtime error | Initialize all queues in `comptime` with `@is_arch("wse3")` guard |
| Not terminating broadcast chain | Wavelets leak off fabric edge | Ensure last PE has only `RAMP` in `.tx` |
| Control wavelet without `.control = true` | Data corrupted or switch not advancing | Set `.control = true` on the fabout DSD |
| Wrong `current_switch_pos` | Data goes to wrong neighbor first | Count positions carefully; 0 = default route |
| Color swap on non-adjacent IDs | No swap occurs | Colors must differ only in low bit (e.g., 0&1, 2&3, 4&5) |
| Mixing data and control on same DSD | Corruption | Use separate DSDs for data and control wavelets |

---

## 13. Related Topics

- **DSDs & DSRs**: `CSL_DSD_Complete_Guide.md` — `fabin_dsd`, `fabout_dsd`, DSD properties
- **Async Programming**: `CSL_Async_Programming_Guide.md` — `.async`, `.activate`, `.unblock` callbacks for fabric operations
- **Builtins**: `CSL_Builtins_Reference_Guide.md` — `@get_color`, `@set_color_config`, `@get_input_queue`, `@get_output_queue`
- **Host Communication**: `CSL_Host_Device_Communication_Guide.md` — Memcpy streaming colors
- **Tutorials**: `gemv-06` (basic routing), `gemv-07` (2D grid), `gemv-08` (NxM scalable), `topic-05` (sentinels), `topic-06` (switches), `topic-08` (filters), `topic-14` (color swap)
- **Benchmarks**: `histogram-torus` (torus routing), `stencil_3d_7pts` (4-dir neighbor exchange), `game-of-life` (neighbor communication)
