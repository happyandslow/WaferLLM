---
title: "GEMM with Collective Operations"
url: "https://sdk.cerebras.net/csl/code-examples/benchmark-gemm-collectives"
crawled_at: "2026-02-14 23:55:16"
---

# GEMM with Collective Operations

> Source: [https://sdk.cerebras.net/csl/code-examples/benchmark-gemm-collectives](https://sdk.cerebras.net/csl/code-examples/benchmark-gemm-collectives)

This program implements the SUMMA matrix multiplication algorithm and serves
as an example of using the `collectives_2d` library together with
`SdkRuntime` and the `memcpy` framework.

The host code first copies tiles of `A` and `B` onto their corresponding
PEs. It then uses the remote procedure call (RPC) mechanism to launch the
function `main`, at which point the GEMM computation begins.

We perform GEMM in `P` many steps on a grid of `P x P` processors.
At each step `i`, PEs in the ith column broadcast their home tiles of `A`
to other PEs in their row, and PEs in the ith row broadcast their home
tiles of `B` to other PEs in their column. Once both broadcasts are complete
as determined by `x_done()` and `y_done()` both being activated,
each PE computes `C_tile += Ap * Bp` where `Ap` and `Bp` are pointers to
either the PE’s home tile or the tile it received through broadcasts.

When computation is complete the host copies back the resulting tiles of
`C` from the device.

## layout.csl

```csl
// Color/ task ID map
//
//  ID var              ID var              ID var                ID var
//   0 c2d_x_color_0     9 c2d_x_entrypt_1  18                    27 reserved (memcpy)
//   1 c2d_x_color_1    10 c2d_y_entrypt_0  19                    28 reserved (memcpy)
//   2                  11 c2d_y_entrypt_1  20                    29 reserved
//   3                  12 EXIT             21 reserved (memcpy)  30 reserved (memcpy)
//   4 c2d_y_color_0    13 compute_task_id  22 reserved (memcpy)  31 reserved
//   5 c2d_y_color_1    14 x_task_id        23 reserved (memcpy)  32
//   6                  15 y_task_id        24                    33
//   7                  16                  25                    34
//   8 c2d_x_entrypt_0  17                  26                    35

// Program rectangle is P x P
param P: u16;

// Matrix dimensions on one PE
param Mt: u16;
param Kt: u16;
param Nt: u16;

const memcpy = @import_module("<memcpy/get_params>", .{
  .width = P,
  .height = P
});

const c2d = @import_module("<collectives_2d/params>");

layout {
  @set_rectangle(P, P);

  var Px: u16 = 0;
  while (Px < P) : (Px += 1) {
    var Py: u16 = 0;
    const memcpy_params = memcpy.get_params(Px);
    while (Py < P) : (Py += 1) {
      const c2d_params = c2d.get_params(Px, Py, .{
        .x_colors      = .{ @get_color(0),         @get_color(1) },
        .x_entrypoints = .{ @get_local_task_id(8), @get_local_task_id(9) },
        .y_colors      = .{ @get_color(4),         @get_color(5) },
        .y_entrypoints = .{ @get_local_task_id(10), @get_local_task_id(11) },
      });
      @set_tile_code(Px, Py, "pe.csl", .{
        .memcpy_params = memcpy_params,
        .c2d_params = c2d_params,
        .Mt = Mt, .Kt = Kt, .Nt = Nt,
      });
    }
  }

  // export symbol names
  @export_name("A", [*]f32, true);
  @export_name("B", [*]f32, true);
  @export_name("C", [*]f32, true);
  @export_name("main", fn()void);
}

```

## pe.csl

```csl
// This program implements the SUMMA matrix multiplication algorithm and is
// written as an example to show how to use the `collectives_2d` library.

// We perform GEMM in `P` many steps on a grid of `P x P` processors.
// At each step `i`, PEs in the `i`th column broadcast their home tiles of `A`
// to other PEs in their row, and PEs in the `i`th row broadcast their home
// tiles of `B` to other PEs in their column. Once both broadcasts are complete
// as determined by `x_done()` and `y_done()` both being activated,
// each PE computes `C_tile += Ap * Bp` where `Ap` and `Bp` are pointers to
// either the PE's home tile or the tile it received through broadcasts.

param c2d_params: comptime_struct;
param memcpy_params: comptime_struct;

// Matrix size params
param Mt: i16;
param Kt: i16;
param Nt: i16;

// Task IDs
const EXIT:            local_task_id = @get_local_task_id(12);
const compute_task_id: local_task_id = @get_local_task_id(13);
const x_task_id:       local_task_id = @get_local_task_id(14);
const y_task_id:       local_task_id = @get_local_task_id(15);

const mpi_x = @import_module("<collectives_2d/pe>", .{
    .dim_params = c2d_params.x,
    .queues = [2]u16{2,4},
    .dest_dsr_ids = [1]u16{1},
    .src0_dsr_ids = [1]u16{1},
    .src1_dsr_ids = [1]u16{1}
    });
const mpi_y = @import_module("<collectives_2d/pe>", .{
    .dim_params = c2d_params.y,
    .queues = [2]u16{3,5},
    .dest_dsr_ids = [1]u16{2},
    .src0_dsr_ids = [1]u16{2},
    .src1_dsr_ids = [1]u16{2}
    });

// On WSE-2, memcpy uses input/output queue 0
// On WSE-3, memcpy uses input/output queues 0 and 1
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

const P = @get_rectangle().width;

// This PE's home tile of A, B, C
// `A_tile` and `B_tile` will be populated with initial values by run.py
// These arrays are stored in a column major format.
var A_tile = @zeros([Mt*Kt]f32);
var B_tile = @zeros([Kt*Nt]f32);
var C_tile = @zeros([Mt*Nt]f32);

var ptr_A : [*]f32 = &A_tile;
var ptr_B : [*]f32 = &B_tile;
var ptr_C : [*]f32 = &C_tile;

// Temporary buffers for storing in-flight tiles of A and B
var A_buffer = @zeros([Mt*Kt]f32);
var B_buffer = @zeros([Kt*Nt]f32);

var px: u16;
var py: u16;

task x_done() void {
  @activate(compute_task_id);
}

task y_done() void {
  @unblock(compute_task_id);
}

var step: u16 = 0;
fn main() void {
  @assert(step < P);

  // The first time through we need to initialize our state
  if (step == 0) {
    mpi_x.init();
    mpi_y.init();
    px = mpi_x.pe_id;
    py = mpi_y.pe_id;
  }

  // Communicate along both rows and columns
  const Ap = if (px == step) &A_tile else &A_buffer;
  const Bp = if (py == step) &B_tile else &B_buffer;
  mpi_x.broadcast(step, @ptrcast([*]u32, Ap), Mt * Kt, x_task_id);
  mpi_y.broadcast(step, @ptrcast([*]u32, Bp), Kt * Nt, y_task_id);
}

task compute() void {
  const Ap = if (px == step) &A_tile else &A_buffer;
  const Bp = if (py == step) &B_tile else &B_buffer;

  // Do an fmacs based local GEMM
  var A_dsd  = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Mt} -> A_tile[i] });
  A_dsd = @set_dsd_base_addr(A_dsd, Ap);

  for (@range(i16, Kt)) |k| {
    var C_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Mt} -> C_tile[i] });

    for (@range(i16, Nt)) |j| {
      const b = Bp.*[j*Kt + k];
      @fmacs(C_dsd, C_dsd, A_dsd, b);
      C_dsd = @increment_dsd_offset(C_dsd, Mt, f32);
    }
    A_dsd = @increment_dsd_offset(A_dsd, Mt, f32);
  }

  step += 1;
  @block(compute_task_id);

  if (step != P) {
    main();
  } else {
    @activate(EXIT);
  }
}

task f_exit() void {
  // the user must unblock cmd color for every PE
  sys_mod.unblock_cmd_stream();
}

comptime {
  @bind_local_task(f_exit, EXIT);
  @bind_local_task(compute, compute_task_id);
  @bind_local_task(x_done, x_task_id);
  @bind_local_task(y_done, y_task_id);
  @block(compute_task_id);

  @export_symbol(ptr_A, "A");
  @export_symbol(ptr_B, "B");
  @export_symbol(ptr_C, "C");
  @export_symbol(main);
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder    # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()

# Get params from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Kernel rectangle and per-PE matrix dimensions
P = int(compile_data['params']['P'])
Mt = int(compile_data['params']['Mt'])
Kt = int(compile_data['params']['Kt'])
Nt = int(compile_data['params']['Nt'])

# Full matrix dimensions
# A is M x K, B is K x N, C is M x N
M = Mt * P
K = Kt * P
N = Nt * P

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_A = runner.get_id("A")
sym_B = runner.get_id("B")
sym_C = runner.get_id("C")

runner.load()
runner.run()

w = P # number of columns PEs in the core rectangle
h = P # number of row PEs in the core rectangle

# How to transform a 2-D tensor into a cliff distribution with
# column-major local tensor
#
# Example: w=2, h=2, A is 4-by-4 (lh-by-lw)
# A = |  0  1  2  3 |
#     |  4  5  6  7 |
#     |  8  9 10 11 |
#     | 12 13 14 15 |
# A1 = A.reshape(2,2,2,2) of the form (h,lh,w,lw)
# A1 = | | 0  1|  | 4  5| |
#      | | 2  3|, | 6  7| |
#      |                  |
#      | | 8  9|  |12 13| |
#      | |10 11|, |14 15| |
# A2 = A1.transpose(0, 2, 3, 1) of the form (h, w, lw, lh)
# so the local tensor lh-by-lw is col-major
# A2 = | | 0  4|  | 2  6| |
#      | | 1  5|, | 3  7| |
#      |                  |
#      | | 8 12|  |10 14| |
#      | | 9 13|, |11 15| |
# A3 = A2.reshape(2,2,4)
# A3 = |  0  4  1  5 |
#      |  2  6  3  7 |
#      |  8 12  9 13 |
#      | 10 14 11 15 |
# A3 is h-w-l

A1 = A.reshape(h, Mt, w, Kt)
A2 = A1.transpose(0, 2, 3, 1)
A3 = A2.reshape(h, w, Mt*Kt)
runner.memcpy_h2d(sym_A, A3.ravel(), 0, 0, w, h, Mt*Kt, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

B1 = B.reshape(h, Kt, w, Nt)
B2 = B1.transpose(0, 2, 3, 1)
B3 = B2.reshape(h, w, Kt*Nt)
runner.memcpy_h2d(sym_B, B3.ravel(), 0, 0, w, h, Kt*Nt, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

runner.launch("main", nonblock=False)

C3_1d_u32 = np.zeros(h*w*Mt*Nt, np.uint32)
runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, w, h, Mt*Nt, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
# C3 is h-by-w-l or
# C3 is of the form (h, w, Nt, Mt) where local tensor Mt-by-Nt is column-major
C3 = C3_1d_u32.reshape((h, w, Nt, Mt))
# C2 is of the form (h, Mt, w, Nt)
C2 = C3.transpose(0, 3, 1, 2)
# C1 is of the form (M, N)
C1 = C2.reshape(M, N)
# C has the correct data type
C = C1.view(np.float32)

runner.stop()

# Check the result
C_expected = np.dot(A, B)

# absolute(a - b) <= (atol + rtol * absolute(b))
np.testing.assert_allclose(C_expected, C, rtol=1e-05, atol=1e-06)

print("SUCCESS")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=P:4,Mt:14,Kt:14,Nt:14 \
--memcpy --channels=1 -o out
cs_python run.py --name out

```
[previous

GEMV with Collective Communications](csl/code-examples/benchmark-gemv-collectives.md "previous page")
[next

Residual](csl/code-examples/benchmark-residual.md "next page")
