---
title: "Cholesky"
url: "https://sdk.cerebras.net/csl/code-examples/benchmark-cholesky"
crawled_at: "2026-02-14 23:55:28"
---

# Cholesky

> Source: [https://sdk.cerebras.net/csl/code-examples/benchmark-cholesky](https://sdk.cerebras.net/csl/code-examples/benchmark-cholesky)

If we have a symmetric positive-definite matrix `A`, then we can use
Cholesky decomposition to find a lower-triangular matrix `L` such that
`LL^T = A`.

This benchmark implements the Cholesky decomposition algorithm using the
“right-looking” approach. We can write out the algorithm as:

```default
N = A.shape[0]
for i in range(N):
  pivot = A[i,i]
  A[:,i] /= sqrt(pivot)
  A[i+1:,i+1:] -= np.outer(A[i+1:,i], A[i+1:,i])

```

Note that because `A` is symmetric, we need only store its lower triangle,
and indeed do the whole computation on the lower triangle.

To implement this in CSL, we tile `A` over the lower triangle of our grid of
PEs. We run routes with color `row_color` across the rows of PEs and routes
with color `col_color` down the columns of PEs. We can visualize our triangle
of PEs as follows:

```default
  ┌───┐
┌─┤P00│
│ └───┘
│
│   ┌───────┐
│   │       │
│ ┌─┴─┐   ┌─▼─┐
├─►P01│ ┌─┤P11│
│ └───┘ │ └───┘
│       │
│   ┌─► │ ──┬───────┐
│   │   │   │       │
│ ┌─┴─┐ │ ┌─▼─┐   ┌─▼─┐
├─►P02│ ├─►P12│ ┌─┤P22│
│ └───┘ │ └───┘ │ └───┘
│       │       │
│   ┌─► │ ──┬─► │ ──┬───────┐
│   │   │   │   │   │       │
│ ┌─┴─┐ │ ┌─▼─┐ │ ┌─▼─┐   ┌─▼─┐
├─►P03│ ├─►P13│ ├─►P23│ ┌─┤P33│
│ └───┘ │ └───┘ │ └───┘ │ └───┘
│       │       │       │
│   ┌─► │ ──┬─► │ ──┬─► │ ──┬───────┐
│   │   │   │   │   │   │   │       │
│ ┌─┴─┐ │ ┌─▼─┐ │ ┌─▼─┐ │ ┌─▼─┐   ┌─▼─┐
└─►P04│ └─►P14│ └─►P24│ └─►P34│   │P44│
  └───┘   └───┘   └───┘   └───┘   └───┘

```

For clarity, each PE stores a an `Nt x Nt` sized tile of A. For PEs on the
diagonal, only the lower triangle of the tile is actually stored.

Recall from the code above that the algorithm will need to run for `N`
iterations. Let’s look at what happens in a given outer-loop iteration:

> 1. The top left PE (P00) computes the inverse square root of the pivot, and
> multiplies that value by the first column of its tile. It then sends its
> first column down the row color.
>
> 2. PEs along the left column receive P00’s chunk of the first column and use
> it to update their first column (multiply by `invsqrt`). Then, they compute
> an outer product of this updated first column with the chunk received from
> P00. Finally, they send their updated first columns `EAST` along the
> `row_color`.
>
> 3. When row tile reaches PEs along the diagonal (P11, P22, P33, P44), those
> PEs
> subtract an outer product of that row chunk with itself from their own tile’s
> values. They then send their received row chunk (unmodified) down the
> `col_color`
>
> 4. Interior PEs (P12, P13, P23, P14, P24, P34) receive a row chunk along the
> `row_color` and a column chunk along the `col_color`. They subtract
> the outer product of these chunks from their local tiles.

We can then move onto the next outer loop iteration.

The interesting transition
happens once we have done `Nt` iterations. At this point, the left-most
column of PEs no longer participates in the algorithm, and column 1 becomes
the next left-most column. P11 assumes the “top left” role previously held by
P00. Importantly, P12, P13, P14 now need to *send* on `row_col` instead of
receiving. This means that we need to reconfigure some routes!

Fortunately, this can be achieved using fabric switches on `row_color`. After
they have finished processing their tile’s final column, PEs P02, P03, and
P04 send control wavelets to flip their neighbors’ fabric switches to allow
them to send on `row_color`. Note that P01 does not need to do this because
P11 will never send values on `row_color`.

This process will repeat again as column 2, then column 3, then finally column 4
become the left-most columns. For the last `Nt` many iterations, all PEs other
than P44 will be idle.

## layout.csl

```csl
// The core kernel must start at P4.1 so the memcpy infrastructure has enough
// resources to route the data between the host and the device.
//
// color/ task ID map
//
//  ID var        ID var           ID var                ID var
//   0 row_color   9               18 cont_task_id       27 reserved (memcpy)
//   1 col_color  10               19                    28 reserved (memcpy)
//   2            11               20                    29 reserved
//   3            12               21 reserved (memcpy)  30 reserved (memcpy)
//   4            13               22 reserved (memcpy)  31 reserved
//   5            14               23 reserved (memcpy)  32
//   6            15               24                    33
//   7            16               25                    34
//   8            17 main_task_id  26                    35

param P : i16;
param Nt: i16;

// Colors
const row_color: color = @get_color(0);
const col_color: color = @get_color(1);

// Task IDs
const main_task_id: local_task_id = @get_local_task_id(17);
const cont_task_id: local_task_id = @get_local_task_id(18);

const memcpy = @import_module( "<memcpy/get_params>", .{
    .width = P,
    .height = P
    });

layout {
  @set_rectangle(P, P);

  var x = 0;
  while (x < P) : (x += 1) {
    var y = 0;
    while (y < P) : (y += 1) {
      const memcpy_params = memcpy.get_params(x);
      const params = .{ .px = x, .py = y, .Nt = Nt,
                        .row_color = row_color, .col_color = col_color,
                        .main_task_id = main_task_id, .cont_task_id = cont_task_id };

      if (x <= y) {
        @set_tile_code(x, y, "pe.csl", @concat_structs( .{
             .memcpy_params = memcpy_params,
        }, params));
      } else {
        const launch_params = .{ .Nt = Nt };
        @set_tile_code(x, y, "launch.csl", @concat_structs( .{
             .memcpy_params = memcpy_params,
        }, launch_params));
      }
    }
  }

  // Setup column routes (straightforward)
  x = 0;
  while (x < (P - 1)) : (x += 1) {
    @set_color_config(x, x, col_color, .{ .routes = .{ .rx = .{ RAMP }, .tx = .{ SOUTH } } });

    var y = x + 1;
    while (y < P) : (y += 1) {
      const tx = if (y == P - 1) .{ RAMP } else .{ RAMP, SOUTH };
      @set_color_config(x, y, col_color, .{ .routes = .{ .rx = .{ NORTH }, .tx = tx } });
    }
  }

  // Setup row routes (requires switches)
  var y = 1;
  while (y < P) : (y += 1) {
    @set_color_config(0, y, row_color, .{ .routes = .{ .rx = .{ RAMP }, .tx = .{ EAST } } });
    x = 1;
    while (x < y) : (x += 1) {
      const routes = .{
        .rx = .{ WEST },
        .tx = .{ RAMP, EAST },
      };
      const switches = .{
        .pos1 = .{ .tx = EAST },
        .pos2 = .{ .rx = RAMP },
        .pop_mode = .{ .pop_on_advance = true }
      };
      @set_color_config(x, y, row_color, .{ .routes = routes, .switches = switches });
    }
    @set_color_config(y, y, row_color, .{ .routes = .{ .rx = .{ WEST }, .tx = .{ RAMP } } });
  }

  // export symbol name
  @export_name("tile", [*]f32, true);
  @export_name("f_chol", fn()void);
}

```

## pe.csl

```csl
// This benchmark implements right-looking Cholesky factorization

param memcpy_params: comptime_struct;
param px: i16;
param py: i16;
param Nt: i16;

// Colors
param row_color: color;
param col_color: color;

// Task IDs
param main_task_id: local_task_id;
param cont_task_id: local_task_id;

// Queue IDs
const col_color_iq = @get_input_queue(2);
const col_color_oq = @get_output_queue(3);
const row_color_iq = @get_input_queue(4);
const row_color_oq = @get_output_queue(5);

const P: i16 = @as(i16, @get_rectangle().height);

const math = @import_module("<math>");

const sys_mod = @import_module( "<memcpy/memcpy>", memcpy_params);

// Memory buffers
// tile is Nt-by-Nt in row-major order
var tile = @zeros([Nt*Nt]f32);
var col_buf = @zeros([Nt]f32);
var row_buf = @zeros([Nt]f32);

var ptr_tile : [*]f32 = &tile;

// // Memory DSDs
var tile_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> tile[i*Nt+0] });
var col_buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> col_buf[i] });
var row_buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> row_buf[i] });

// // Column in/out DSDs
var col_in = @get_dsd(fabin_dsd, .{ .fabric_color = col_color, .extent = Nt, .input_queue = col_color_iq });
var col_out = @get_dsd(fabout_dsd, .{ .fabric_color = col_color, .extent = Nt, .output_queue = col_color_oq });

// // Row in/out DSDs
const row_in = @get_dsd(fabin_dsd, .{ .fabric_color = row_color, .extent = Nt, .input_queue = row_color_iq });
const row_out = @get_dsd(fabout_dsd, .{ .fabric_color = row_color, .extent = Nt, .output_queue = row_color_oq });

// // Two wavelets of (ADV + NO_CE, NOP + NO_CE)
const ctrl_buf = [2]u32 { (5 << 22) | (4 << 25), (5 << 22) | (4 << 25) };
const ctrl_mem = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{2} -> ctrl_buf[i] });
const row_ctrl_out = @get_dsd(fabout_dsd, .{ .fabric_color = row_color, .extent = 2,
                                             .output_queue = row_color_oq, .control = true });

var P_left: i16;
var ti: i16;

var iter: i16 = 0;
task main() void {

  P_left = iter / Nt;
  ti = iter % Nt;

  if (px < P_left) {
    // If the fringe has moved on, we need to flip a switch to allow the next
    // fringe to send out data
    if (px != py) {
      @mov32(row_ctrl_out, ctrl_mem);
    }
    // WARNING: the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
    return; // PE at left of current column is done
            // right-bottom PE is done when iter = Nt*P
            // i.e. all PEs are done when iter = Nt*P
  }

  if (px == P_left and py == P_left) {

    // Top left of the current fringe
    //const pivot = tile[ti, ti];
    const pivot = tile[ti*Nt + ti];
    const invsqrt = math.invsqrt_f32(pivot);

    @fmuls(tile_dsd, tile_dsd, invsqrt);

    // If we're the top left PE for the current fringe, we send values down
    // our column
    if (px < P - 1) {
      @mov32(col_out, tile_dsd, .{ .async = true, .activate = cont_task_id });
    } else {
      // Unless we don't even have a column because we're the bottom-right
      // PE
      @activate(cont_task_id);
    }

    var left_col_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> tile[i*Nt + 0] });
    var dest_row_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> tile[0*Nt + i] });

    left_col_dsd = @increment_dsd_offset(left_col_dsd,  @as(i16, Nt * (ti + 1) + (ti)), f32);
    dest_row_dsd = @increment_dsd_offset(dest_row_dsd, @as(i16, (ti +1) * Nt + (ti + 1)), f32);

    for (@range(i16, ti+1, Nt, 1)) |i| {
      dest_row_dsd = @set_dsd_length(dest_row_dsd, @as(u16,i - ti));
      @fmacs(dest_row_dsd, dest_row_dsd, left_col_dsd, -1.0 * tile[i*Nt + ti]);
      dest_row_dsd = @increment_dsd_offset(dest_row_dsd, Nt, f32);
    }

  } else if (px == P_left) {
    // Left edge of the current fringe
    @mov32(col_buf_dsd, col_in, .{ .async = true, .activate = cont_task_id });

  } else if (px == py) {
    // Non-fringe diagonal
    @mov32(row_buf_dsd, row_in, .{ .async = true, .activate = cont_task_id });

  } else {
    // Non-fringe interior block
    @block(cont_task_id);
    @mov32(row_buf_dsd, row_in, .{ .async = true, .activate = cont_task_id });
    @mov32(col_buf_dsd, col_in, .{ .async = true, .unblock = cont_task_id });
  }
}

// // Continuation task
task cont() void {

  if (px == P_left and py == P_left) {

    tile_dsd = @increment_dsd_offset(tile_dsd, 1, f32);
    @activate(main_task_id);

  } else if (px == P_left) {

    var invsqrt = 1.0 / col_buf[ti];
    @fmuls(tile_dsd, tile_dsd, invsqrt);

    var dest_col_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> tile[i*Nt + 0] });
    dest_col_dsd = @increment_dsd_offset(dest_col_dsd, @as(i16, ti + 1), f32);

    for (@range(i16, ti+1, Nt, 1)) |j| {
      @fmacs(dest_col_dsd, dest_col_dsd, tile_dsd, -1.0 * col_buf[j]);
      dest_col_dsd = @increment_dsd_offset(dest_col_dsd, 1, f32);
    }
    // for (@range(u16, Nt)) |i| {
    //    for (@range(u16, ti+1, Nt, 1)) |j| {
    //      tile[i,j] -= col_buf[j] * tile[i,ti];
    //    }
    // }

    @mov32(row_out, tile_dsd, .{ .async = true, .activate = main_task_id });
    tile_dsd = @increment_dsd_offset(tile_dsd, 1, f32);

  } else if (px == py) {
    @assert(px > P_left);

    var tile_row = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> tile[0*Nt + i] });
    for (@range(u16, Nt)) |i| {
      tile_row = @set_dsd_length(tile_row, i + 1);
      @fmacs(tile_row, tile_row, row_buf_dsd, -1.0 * row_buf[i]);
      tile_row = @increment_dsd_offset(tile_row, Nt, f32);
    }

    if (py != P - 1) {
      // If we're on the diagonal, our job is to take values we received along
      // our row and send them down our column
      @mov32(col_out, row_buf_dsd, .{ .async = true, .activate = main_task_id });
    } else {
      // Unless we're the bottom-right corner PE... in which case we don't
      // have a column to send values down!
      @activate(main_task_id);
    }

  } else {
    @assert(px > P_left);

    for (@range(u16, Nt)) |i| {
      @fmacs(tile_dsd, tile_dsd, row_buf_dsd, -1.0 * col_buf[i]);
      tile_dsd = @increment_dsd_offset(tile_dsd, 1, f32);
    }
    tile_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> tile[i*Nt + 0] });
    @activate(main_task_id);
  }

  // Next time we go to the main task, we've moved up an iteration
  iter += 1;
}

fn f_chol() void {
  @activate(main_task_id);
}

comptime {
  @comptime_assert(px <= py);

  @bind_local_task(main, main_task_id);
  @bind_local_task(cont, cont_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(col_color_iq, .{ .color = col_color });
    @initialize_queue(col_color_oq, .{ .color = col_color });
    @initialize_queue(row_color_iq, .{ .color = row_color });
    @initialize_queue(row_color_oq, .{ .color = row_color });
  }

  @export_symbol(ptr_tile, "tile");
  @export_symbol(f_chol);
}

```

## launch.csl

```csl
param memcpy_params: comptime_struct;
param Nt: u16;

var tile = @zeros([Nt*Nt]f32);

var ptr_tile : [*]f32 = &tile;

const sys_mod = @import_module( "<memcpy/memcpy>", memcpy_params);

fn f_chol() void {
  // WARNING: the user must unblock cmd color for every PE
  sys_mod.unblock_cmd_stream();
}

comptime{
  @export_symbol(ptr_tile, "tile");

  @export_symbol(f_chol);
}

```

## run.py

```python
#!/usr/bin/env cs_python

from itertools import product
import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
dirname = args.name

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
compile_params = compile_data["params"]
P = int(compile_params["P"])
Nt = int(compile_params["Nt"])
print(f"P = {P}, Nt = {Nt}")

print("WARNING: The simfab may take 90 sec")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_tile = runner.get_id("tile")

runner.load()
runner.run()

# Initialize the input matrices
N = P * Nt

counter = 1
L = np.zeros((N, N), dtype=np.float32)
for i in range(N):
  for j in range(i+1):
    L[i, j] = counter
    counter += 1

# M = LL^T except we only store the upper triangle
M = np.dot(L, L.T)
for i in range(N):
  for j in range(i+1, N):
    M[i, j] = 0

# Split it up into tiles that can be mapped to each PE
M_tiles_xy = np.array([np.vsplit(s, P) for s in np.hsplit(M, P)])

print("step 1: copy mode H2D prepares data in non-upper of A")
# Write tiles to PEs
for px, py in product(range(P), range(P)):
  if px > py:
    continue

  M_tile = M_tiles_xy[px, py]
  assert M_tile.size == Nt*Nt
  runner.memcpy_h2d(sym_tile, M_tile.ravel(), px, py, 1, 1, Nt*Nt, \
    streaming=False, data_type=memcpy_dtype, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)

print("stpe 2: call f_chol to compute A = L*L**T")
runner.launch("f_chol", nonblock=False)

print("step 3: copy mode D2H gather L")
# collect results
result_tiles = np.zeros(M_tiles_xy.shape, dtype=M_tiles_xy.dtype)
for px, py in product(range(P), range(P)):
  if px > py:
    continue

  tile = np.zeros(Nt*Nt, np.float32)
  runner.memcpy_d2h(tile, sym_tile, px, py, 1, 1, Nt*Nt,\
    streaming=False, data_type=memcpy_dtype, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)
  result_tiles[px, py] = tile.reshape(Nt, Nt)

runner.stop()

# reassemble result
result = result_tiles.transpose(1, 2, 0, 3).reshape(N, N)

np.testing.assert_almost_equal(result, L, decimal=2)

print("SUCCESS")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=17,12 --fabric-offsets=4,1 \
--params=P:10,Nt:4 -o out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python run.py --name out

```
[previous

Residual](csl/code-examples/benchmark-residual.md "previous page")
[next

25-Point Stencil](csl/code-examples/benchmark-25-pt-stencil.md "next page")
