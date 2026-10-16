# Cerebras CSL SdkLayout Complete Guide
## SDK Version: 1.4.0

This document covers the SdkLayout Python API for defining PE grid layouts, routing, port connections, and H2D/D2H streaming — all without writing `layout.csl`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Workflow](#2-core-workflow)
3. [SdkLayout Class](#3-sdklayout-class)
4. [CodeRegion](#4-coderegion)
5. [Colors and Routing](#5-colors-and-routing)
6. [Ports and Connections](#6-ports-and-connections)
7. [Input/Output Streams (H2D/D2H)](#7-inputoutput-streams-h2dd2h)
8. [Parameters and Symbols](#8-parameters-and-symbols)
9. [Placement](#9-placement)
10. [Compilation and Runtime](#10-compilation-and-runtime)
11. [Complete Examples](#11-complete-examples)
12. [Best Practices](#12-best-practices)

---

## 1. Overview

`SdkLayout` is a Python API that replaces `layout.csl` for defining:
- PE grid regions and their source code
- Color routing between PEs
- Inter-region port connections
- Host-to-device and device-to-host data streams

**Key advantage**: Layout configuration is done in Python, enabling dynamic/programmatic setup, complex multi-region designs, and automatic color allocation.

**Import**:
```python
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkLayout, SdkTarget, SdkRuntime, SimfabConfig,
    Color, Edge, Route, RoutingPosition,
    get_platform, get_edge_routing,
)
from cerebras.geometry.geometry import IntVector, IntRectangle
```

---

## 2. Core Workflow

```
1. Create platform → SdkLayout
2. Create CodeRegions (PE grids with .csl source)
3. Define Colors and routing
4. Create input/output Ports on region edges
5. Connect ports between regions
6. Create I/O streams (H2D/D2H)
7. Set parameters and symbols
8. Compile → artifacts
9. Create SdkRuntime → load → run → send/receive → stop
```

### Minimal Example

```python
platform = get_platform(cmaddr, SimfabConfig(), SdkTarget.WSE3)
layout = SdkLayout(platform)

# Create a 1×1 PE region
region = layout.create_code_region('./kernel.csl', 'compute', 1, 1)

# Define colors and ports
rx = Color('rx')
tx = Color('tx')
region.set_param_all(rx)
region.set_param_all(tx)

rx_port = region.create_input_port(rx, Edge.RIGHT, [receiver_routes], size)
tx_port = region.create_output_port(tx, Edge.LEFT, [sender_routes], size)

# Create H2D/D2H streams
in_stream = layout.create_input_stream(rx_port)
out_stream = layout.create_output_stream(tx_port)

# Compile and run
artifacts = layout.compile(out_prefix='out')
runtime = SdkRuntime(artifacts, platform, memcpy_required=False)
runtime.load()
runtime.run()
runtime.send(in_stream, data, nonblock=True)
runtime.receive(out_stream, result, size, nonblock=True)
runtime.stop()
```

---

## 3. SdkLayout Class

### Constructor

```python
# From platform (recommended)
layout = SdkLayout(platform)

# From fabric file
layout = SdkLayout(fabric_file='path/to/fabric.json')

# From target
layout = SdkLayout(target=SdkTarget.WSE3)
```

Optional `msg_level` keyword: `'DEBUG'`, `'INFO'`, `'WARNING'` (default), `'ERROR'`.

### Key Methods

| Method | Description |
|---|---|
| `create_code_region(source, name, width, height)` | Create a rectangular PE region |
| `connect(tx_port, rx_port)` | Auto-route between two ports |
| `create_input_stream(port, io_loc?, io_buffer_size?)` | Set up H2D stream to a port |
| `create_output_stream(port, io_loc?, io_buffer_size?)` | Set up D2H stream from a port |
| `create_input_stream_from_loc(loc, color, prefix?)` | H2D stream to specific PE/color |
| `create_output_stream_from_loc(loc, color, prefix?)` | D2H stream from specific PE/color |
| `hstack(children, origin?)` | Place regions horizontally |
| `vstack(children, origin?)` | Place regions vertically |
| `compile(out_prefix)` | Compile and produce artifacts |

---

## 4. CodeRegion

Created via `layout.create_code_region(source, name, width, height)`.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `source` | `str` | Path to CSL source file |
| `name` | `str` | Unique region name |
| `width` | `int` | Number of PEs horizontally |
| `height` | `int` | Number of PEs vertically |

### Routing Methods

| Method | Description |
|---|---|
| `paint(coord, color, routes)` | Set routing on single PE |
| `paint_all(color, routes)` | Set routing on all PEs |
| `paint_all(color, routes, edge_routes)` | Set routing with edge overrides |
| `paint_range(rect, color, routes)` | Set routing on rectangular sub-region |

### Parameter Methods

| Method | Description |
|---|---|
| `set_param(coord, name, value)` | Set int param on one PE |
| `set_param(coord, color)` | Set color param on one PE |
| `set_param(coord, name, color)` | Set named color param on one PE |
| `set_param_all(name, value)` | Set int param on all PEs |
| `set_param_all(color)` | Set color param on all PEs  |
| `set_param_all(name, color)` | Set named color param on all PEs |
| `set_param_range(rect, name, value)` | Set param on rectangle |
| `set_symbol_all(symbol, data, width, height)` | Set symbol data distributed across PEs |

### Placement

```python
region.place(x, y)  # Place at absolute coordinates
```

---

## 5. Colors and Routing

### Color

```python
# Auto-allocated color (recommended)
my_color = Color('my_color')

# Or with explicit physical value (0-23)
my_color = Color('my_color', 5)

# Or scoped to a region
my_color = region.color('my_color')
my_color = region.color('my_color', 5)
```

### RoutingPosition

A routing position specifies input routes and output routes:

```python
# Receive from RAMP, output nowhere
ramp_in = RoutingPosition().set_input([Route.RAMP])

# Receive from nothing, output to RAMP
ramp_out = RoutingPosition().set_output([Route.RAMP])

# East-to-West pass-through with RAMP tap
passthrough = RoutingPosition()
passthrough.set_input([Route.EAST])
passthrough.set_output([Route.WEST, Route.RAMP])
```

### Route Enum

| Value | Direction |
|---|---|
| `Route.RAMP` | To/from the CE (compute element) |
| `Route.EAST` | Right |
| `Route.WEST` | Left |
| `Route.NORTH` | Up |
| `Route.SOUTH` | Down |

### Painting Routes

```python
# All PEs: color flows East→West with RAMP tap
routes = RoutingPosition()
routes.set_input([Route.EAST])
routes.set_output([Route.WEST, Route.RAMP])
region.paint_all(my_color, [routes])

# Single PE override
region.paint(IntVector(0, 0), my_color, [special_routes])

# With edge-specific routing
from cerebras.sdk.runtime.sdkruntimepybind import get_edge_routing
edge_route = get_edge_routing(Edge.LEFT, [left_edge_routes])
region.paint_all(my_color, [interior_routes], [edge_route])
```

---

## 6. Ports and Connections

### Creating Ports

Ports define data entry/exit points on region edges:

```python
# Input port: data enters the region
rx_port = region.create_input_port(
    color,          # Color for this port
    Edge.RIGHT,     # Which edge data enters from
    [routes],       # Routing positions inside the region
    data_size,      # Number of elements (for compatibility check)
    prefix=''       # Optional: disambiguate same-color ports
)

# Output port: data leaves the region
tx_port = region.create_output_port(
    color,          # Color for this port
    Edge.LEFT,      # Which edge data exits from
    [routes],       # Routing positions inside the region
    data_size,      # Number of elements
    prefix=''
)
```

### Connecting Ports

```python
# Auto-route from output port to input port
layout.connect(tx_port, rx_port)
```

The SDK automatically handles the routing between regions.

---

## 7. Input/Output Streams (H2D/D2H)

### Creating Streams

```python
# H2D: host → device (through a port)
in_stream = layout.create_input_stream(
    port,                   # Input port to receive data
    io_loc=IntVector(x,y),  # Optional: I/O PE location
    io_buffer_size=1024     # Optional: buffer size
)

# D2H: device → host (through a port)
out_stream = layout.create_output_stream(
    port,                   # Output port to send data
    io_loc=IntVector(x,y),  # Optional: I/O PE location
    io_buffer_size=1024
)

# Direct location-based streams (no port needed)
in_stream = layout.create_input_stream_from_loc(
    IntVector(x, y), color, prefix=''
)
out_stream = layout.create_output_stream_from_loc(
    IntVector(x, y), color, prefix=''
)
```

### Sending/Receiving Data

```python
runtime.send(in_stream, data, nonblock=True)
runtime.receive(out_stream, buffer, num_elements, nonblock=True)
```

**Important**: Use `nonblock=True` for send/receive to avoid deadlocks when multiple streams are interdependent.

---

## 8. Parameters and Symbols

### Setting Compile-Time Parameters

```python
# Integer parameter
region.set_param_all('size', 10)
region.set_param(IntVector(0, 0), 'my_param', 42)

# Color parameter (CSL `param name: color`)
my_color = Color('my_color')
region.set_param_all(my_color)              # Uses color's name as param name
region.set_param_all('color_name', my_color)  # Explicit param name
```

### Setting Symbol Data

```python
import numpy as np

# Distribute matrix A across a pe_width × pe_height grid
A = np.random.rand(data_height, data_width).astype(np.float32)
region.set_symbol_all('A', A, pe_width, pe_height)
```

The 2D shape of data must be a multiple of the region dimensions. Data is distributed uniformly.

---

## 9. Placement

### Manual Placement

```python
region.place(x, y)  # Absolute coordinates
```

### Automatic Stacking

```python
# Horizontal: place regions left-to-right
layout.hstack([region_a, region_b, region_c])

# Vertical: place regions top-to-bottom
layout.vstack([region_a, region_b])

# With explicit origin
layout.hstack([region_a, region_b], IntVector(5, 5))
```

---

## 10. Compilation and Runtime

### Compilation

```python
artifacts = layout.compile(out_prefix='out')
```

Produces compilation artifacts in `out/` directory.

### Runtime (SdkLayout mode)

```python
# Note: memcpy_required=False when using SdkLayout streams
runtime = SdkRuntime(artifacts, platform, memcpy_required=False)
runtime.load()
runtime.run()

# Data transfer via streams
runtime.send(in_stream, data, nonblock=True)
runtime.receive(out_stream, result, size, nonblock=True)

runtime.stop()
```

**Key difference from memcpy mode**: SdkLayout uses `send()/receive()` with stream names instead of `memcpy_h2d()/memcpy_d2h()` with symbol IDs.

---

## 11. Complete Examples

### 11.1 Add Two Vectors (sdklayout-04)

**CSL Kernel** (`add2vec.csl`):
```csl
param size: u16;
param rx1: color;
param rx2: color;
param tx: color;

const in_q1 = @get_input_queue(0);
const in_q2 = @get_input_queue(1);
const out_q = @get_output_queue(0);

const input1 = @get_dsd(fabin_dsd, .{.extent=size, .fabric_color=rx1, .input_queue=in_q1});
const input2 = @get_dsd(fabin_dsd, .{.extent=size, .fabric_color=rx2, .input_queue=in_q2});
const output = @get_dsd(fabout_dsd, .{.extent=size, .fabric_color=tx, .output_queue=out_q});

// WSE-3: FIFO workaround for multi-fabric-input restriction
var buffer: [size]u16;
const fifo = @allocate_fifo(buffer);

const main_id = @get_local_task_id(8);
task main() void {
  @mov16(fifo, input2, .{.async = true});
  @add16(output, input1, fifo, .{.async = true});
}

comptime {
  @bind_local_task(main, main_id);
  @activate(main_id);
  @initialize_queue(in_q1, .{.color = rx1});
  @initialize_queue(in_q2, .{.color = rx2});
  if (@is_arch("wse3")) {
    @initialize_queue(out_q, .{.color = tx});
  }
}
```

**Host** (`run.py`):
```python
platform = get_platform(args.cmaddr, SimfabConfig(dump_core=True), SdkTarget.WSE3)
layout = SdkLayout(platform)

size = 10
sender_routes = RoutingPosition().set_input([Route.RAMP])
receiver_routes = RoutingPosition().set_output([Route.RAMP])

# Create region
add2vec = layout.create_code_region('./add2vec.csl', 'add2vec', 1, 1)
rx1 = Color('rx1')
rx2 = Color('rx2')
tx = Color('tx')
add2vec.set_param_all('size', size)
add2vec.set_param_all(rx1)
add2vec.set_param_all(rx2)
add2vec.set_param_all(tx)

# Create ports
rx1_port = add2vec.create_input_port(rx1, Edge.RIGHT, [receiver_routes], size)
rx2_port = add2vec.create_input_port(rx2, Edge.RIGHT, [receiver_routes], size)
tx_port = add2vec.create_output_port(tx, Edge.LEFT, [sender_routes], size)
add2vec.place(7, 4)

# Create streams
in_stream1 = layout.create_input_stream(rx1_port)
in_stream2 = layout.create_input_stream(rx2_port)
out_stream = layout.create_output_stream(tx_port)

# Compile and run
artifacts = layout.compile(out_prefix='out')
runtime = SdkRuntime(artifacts, platform, memcpy_required=False)
runtime.load()
runtime.run()

data1 = np.random.randint(0, 1000, size=size, dtype=np.uint32)
data2 = np.random.randint(1000, 2000, size=size, dtype=np.uint32)
actual = np.empty(size, dtype=np.uint32)

runtime.send(in_stream1, data1, nonblock=True)
runtime.send(in_stream2, data2, nonblock=True)
runtime.receive(out_stream, actual, size, nonblock=True)
runtime.stop()

assert np.array_equal(data1 + data2, actual)
```

### 11.2 Multi-Region GEMV (sdklayout-05)

Complex multi-region design with:
- **x_demux_adaptor** + **x_demux**: Distribute vector x across PE columns
- **b_demux_adaptor** + **b_demux**: Distribute vector b across PE rows
- **gemv** (4×4 PE grid): Matrix-vector multiply
- **y_mux**: Collect results

```python
layout = SdkLayout(platform)

# Create multiple code regions
x_adaptor = layout.create_code_region('./x_demux_adaptor.csl', 'x_demux_adaptor', ...)
x_demux = layout.create_code_region('./x_demux.csl', 'x_demux', ...)
gemv = layout.create_code_region('./gemv.csl', 'gemv', pe_width, pe_height)
y_mux = layout.create_code_region('./y_mux.csl', 'y_mux', ...)

# Place regions
x_adaptor.place(1, 0)
x_demux.place(5, 0)
gemv.place(5, 2)
y_mux.place(10, 2)

# Connect ports between regions
layout.connect(x_adaptor_port, x_demux_port)
layout.connect(x_out_port, gemv_x_port)
layout.connect(gemv_y_port, y_in_port)

# Set matrix data on gemv region
gemv.set_symbol_all('A', A, pe_width, pe_height)

# I/O streams
x_stream = layout.create_input_stream(x_port)
b_stream = layout.create_input_stream(b_port)
y_stream = layout.create_output_stream(y_port)

# Must use nonblock=True to prevent deadlock
runtime.send(x_stream, x, nonblock=True)
runtime.send(b_stream, b, nonblock=True)
runtime.receive(y_stream, y, data_height, nonblock=True)
```

---

## 12. Best Practices

1. **Use `nonblock=True`** for all `send()`/`receive()` calls to avoid deadlocks when data flows are interdependent.

2. **FIFO workaround for WSE-3**: When a PE kernel needs two fabric inputs in one DSD operation, first `@mov16` one input into a FIFO, then use the FIFO as a source. See `add2vec.csl` example.

3. **Color auto-allocation**: Prefer `Color('name')` without explicit values — let the compiler assign physical colors. Use explicit values only when needed for compatibility.

4. **Region placement**: Use `place()` for explicit control, or `hstack()`/`vstack()` for automatic adjacent placement.

5. **`memcpy_required=False`**: Always pass this when using SdkLayout streams. The SdkLayout system replaces the memcpy infrastructure.

6. **Port data_size**: Ensure input and output ports that connect have compatible `data_size` values, or compilation will fail.

7. **CSL kernel params**: In SdkLayout mode, CSL kernels use `param` for colors and sizes — these are set from Python via `set_param_all()`.

8. **Edge routing**: Use `get_edge_routing()` + `paint_all()` overload for different routing behavior at region boundaries.

---

## Related Topics

- **Host Communication**: See `CSL_Host_Device_Communication_Guide.md` for memcpy-based H2D/D2H
- **Routing**: See `CSL_Routing_Colors_Switches_Guide.md` for color/route concepts
- **Queue & FIFO**: See `CSL_Queue_FIFO_Guide.md` for FIFO workaround details
- **WSE-3 Migration**: See `CSL_WSE2_vs_WSE3_Migration_Guide.md` for multi-input restrictions
