---
title: "3D 7-Point Stencil SpMV"
url: "https://sdk.cerebras.net/csl/code-examples/benchmark-7pt-stencil-spmv"
crawled_at: "2026-02-14 23:55:13"
---

# 3D 7-Point Stencil SpMV

> Source: [https://sdk.cerebras.net/csl/code-examples/benchmark-7pt-stencil-spmv](https://sdk.cerebras.net/csl/code-examples/benchmark-7pt-stencil-spmv)

This example evaluates the performance of 7-point stencil. The kernel records
the `start` and `end` of `spmv` by tsc counter. In addition the tsc
counters of all PEs are not sychronized in the beginning. To avoid the timing
variation among those PEs, `sync()` synchronizes all PEs and samples the
reference clock.

The kernel `kernel.csl` defines a couple of host-callable functions,
`f_sync()`, `f_tic()` and `f_toc()` in order to synchronize the PEs and
record the timing of `spmv`.

The kernel `allreduce/pe.csl` performs a reduction over the whole rectangle
to synchronize the PEs, then the bottom-right PE sends a signal to other PEs
to sample the reference clock.

The kernel `stencil_3d_7pts/pe.csl` performs a matrix-vector product (spmv)
where the matrix has 7 diagonals corresponding to 7 point stencil. The stencil
coefficients can vary per PE, but must be the same for the local vector. The
user can change the coefficients based on the boundary condition or curvilinear
coordinate transformation.

The script `run.py` has the following parameters:

- `-k=<int>` specifies the maximum size of local vector.
- `--zDim=<int>` specifies how many elements per PE are computed.
- `--channels=<int>` specifies the number of I/O channels, no bigger than 16.

The `tic()` samples “time\_start” and `toc()` samples “time\_end”. The
`sync()` samples “time\_ref” which is used to adjust “time\_start” and
“time\_end”. The elapsed time (unit: cycles) is measured by
`cycles_send = max(time_end) - min(time_start)`

The overall runtime (us) is computed via the following formula
`time_send = (cycles_send / 0.85) * 1.e-3 us`

The bandwidth is calculated by
`bandwidth = ((6*w*h*4)/time_send)`

## layout.csl

```csl
// c0,c1,c2,c3,c4,c5,c6,c7 are internal colors of 7-point stencil
param C0_ID: i16;
param C1_ID: i16;
param C2_ID: i16;
param C3_ID: i16;
param C4_ID: i16;
param C5_ID: i16;
param C6_ID: i16;
param C7_ID: i16;
// c8 is an internal color of allreduce
param C8_ID: i16;

param MAX_ZDIM: i16; // maximum size of local vector x and y
param width: i16 ; // width of the core
param height: i16 ; // height of the core

param BLOCK_SIZE: i16; // size of temporary buffers for communication

const C0: color = @get_color(C0_ID);
const C1: color = @get_color(C1_ID);
const C2: color = @get_color(C2_ID);
const C3: color = @get_color(C3_ID);
const C4: color = @get_color(C4_ID);
const C5: color = @get_color(C5_ID);
const C6: color = @get_color(C6_ID);
const C7: color = @get_color(C7_ID);
const C8: color = @get_color(C8_ID);

// entrypoints of sync module
const STARTUP: local_task_id = @get_local_task_id(13);

// entrypoints of 7-point stenil
const EN_STENCIL_1: local_task_id = @get_local_task_id(14);
const EN_STENCIL_2: local_task_id = @get_local_task_id(15);
const EN_STENCIL_3: local_task_id = @get_local_task_id(16);

// entrypoints of allreduce
const EN_REDUCE_1: local_task_id = @get_local_task_id(17);
const EN_REDUCE_2: local_task_id = @get_local_task_id(18);
const EN_REDUCE_3: local_task_id = @get_local_task_id(19);
const EN_REDUCE_4: local_task_id = @get_local_task_id(20);

const stencil = @import_module( "../../benchmark-libs/stencil_3d_7pts/layout.csl", .{
    .colors = [8]color{C0, C1, C2, C3, C4, C5, C6, C7},
    .entrypoints = [3]local_task_id{EN_STENCIL_1, EN_STENCIL_2, EN_STENCIL_3},
    .width = width,
    .height = height
    });

const reduce = @import_module( "../../benchmark-libs/allreduce/layout.csl", .{
    .colors = [1]color{C8},
    .entrypoints = [4]local_task_id{EN_REDUCE_1, EN_REDUCE_2, EN_REDUCE_3, EN_REDUCE_4},
    .width = width,
    .height = height
    });

const memcpy = @import_module( "<memcpy/get_params>", .{
    .width = width,
    .height = height,
    });

layout{

    @comptime_assert(C0_ID < C1_ID);
    @comptime_assert(C1_ID < C2_ID);
    @comptime_assert(C2_ID < C3_ID);
    @comptime_assert(C3_ID < C4_ID);
    @comptime_assert(C4_ID < C5_ID);
    @comptime_assert(C5_ID < C6_ID);
    @comptime_assert(C6_ID < C7_ID);
    @comptime_assert(C7_ID < C8_ID);

    // step 1: configure the rectangle which does not include halo
    @set_rectangle( width, height );

    // step 2: compile csl code for a set of PEx.y and generate out_x_y.elf
    //   format: @set_tile_code(x, y, code.csl, param_binding);

    var py: i16 = 0;
    while(py < height) : (py +=1) {
        var px: i16 = 0;
        while(px < width) : (px +=1) {

            const memcpyParams = memcpy.get_params(px);
            const stencilParams = stencil.get_params(px, py);
            const reduceParams = reduce.get_params(px, py);
            var params: comptime_struct = .{
                .memcpyParams = memcpyParams,
                .reduceParams = reduceParams,
                .MAX_ZDIM = MAX_ZDIM,
                .BLOCK_SIZE = BLOCK_SIZE,
                .STARTUP = STARTUP,
                .stencilParams = stencilParams
            };

            @set_tile_code(px, py, "kernel.csl", params);
        }
    }

    @export_name("x", [*]f32, true);
    @export_name("y", [*]f32, true);
    @export_name("stencil_coeff", [*]f32, true);
    @export_name("time_buf_u16", [*]u16, true);
    @export_name("time_ref", [*]u16, true);

    @export_name("f_tic", fn()void);
    @export_name("f_toc", fn()void);
    @export_name("f_memcpy_timestamps", fn()void);
    @export_name("f_spmv", fn(i16)void);
    @export_name("f_sync", fn(i16)void);
    @export_name("f_reference_timestamps", fn()void);
} // end of layout

```

## kernel.csl

```csl
param memcpyParams: comptime_struct;

param reduceParams: comptime_struct;

param stencilParams: comptime_struct;

param MAX_ZDIM: i16; // size of vector x

param BLOCK_SIZE: i16; // size of temporary buffers for communication

param STARTUP: local_task_id;

const timestamp = @import_module("<time>");

// memcpy module reserves
// - input/output queue 0 and 1
const sys_mod = @import_module( "<memcpy/memcpy>", memcpyParams);

// allreduce uses input queue/output queue 2
const reduce_mod = @import_module( "../../benchmark-libs/allreduce/pe.csl", @concat_structs(reduceParams, .{
     .f_callback = sys_mod.unblock_cmd_stream,
     .queues = [1]u16{2},
     .dest_dsr_ids = [1]u16{1},
     .src0_dsr_ids = [1]u16{1},
     .src1_dsr_ids = [1]u16{1}
     }));

// output queue cannot overlap input queues
const stencil_mod = @import_module( "../../benchmark-libs/stencil_3d_7pts/pe.csl", @concat_structs(stencilParams, .{
     .f_callback = sys_mod.unblock_cmd_stream,
     .input_queues = [4]u16{4, 5, 6, 7},
     .output_queues = if (@is_arch("wse3")) [4]u16{4, 5, 6, 7} else [1]u16{3},
     .output_ut_id = 3,
     .BLOCK_SIZE = BLOCK_SIZE,
     .dest_dsr_ids = [2]u16{2,3},
     .src0_dsr_ids = [1]u16{2},
     .src1_dsr_ids = [2]u16{2,3}
     }));

// tsc_size_words = 3
// starting time of H2D/D2H
var tscStartBuffer = @zeros([timestamp.tsc_size_words]u16);
// ending time of H2D/D2H
var tscEndBuffer = @zeros([timestamp.tsc_size_words]u16);

////////////////////////////////////////////////////////////////////////////////
// Main memory (48KB)
////////////////////////////////////////////////////////////////////////////////

var x = @zeros([MAX_ZDIM]f32);
var y = @zeros([MAX_ZDIM]f32);

var dot = @zeros([1]f32);

// stencil coefficients are organized as
// {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
//
// The formula is
//    c_west * x[i-1][j][k] + c_east * x[i+1][j][k] +
//    c_south * x[i][j-1][k] + c_north * x[i][j+1][k] +
//    c_bottom * x[i][j][k-1] + c_top * x[i][j][k+1] +
//    c_center * x[i][j][k]
var stencil_coeff = @zeros([7]f32);

// time_buf_u16[0:5] = {tscStartBuffer, tscEndBuffer}
var time_buf_u16 = @zeros([timestamp.tsc_size_words*2]u16);

// reference clock inside allreduce module
var time_ref_u16 = @zeros([timestamp.tsc_size_words]u16);

var ptr_x: [*]f32 = &x;
var ptr_y: [*]f32 = &y;
var ptr_stencil_coeff: [*]f32 = &stencil_coeff;
var ptr_time_buf_u16: [*]u16 = &time_buf_u16;
var ptr_time_ref: [*]u16 = &time_ref_u16;

////////////////////////////////////////////////////////////////////////////////
// Tasks
// syntax
//     task_begin(name, entrypoint, color)
////////////////////////////////////////////////////////////////////////////////

fn f_tic() void {
    timestamp.get_timestamp(&tscStartBuffer);

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

fn f_toc() void {
    timestamp.get_timestamp(&tscEndBuffer);

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

fn f_memcpy_timestamps() void {

    time_buf_u16[0] = tscStartBuffer[0];
    time_buf_u16[1] = tscStartBuffer[1];
    time_buf_u16[2] = tscStartBuffer[2];

    time_buf_u16[3] = tscEndBuffer[0];
    time_buf_u16[4] = tscEndBuffer[1];
    time_buf_u16[5] = tscEndBuffer[2];

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

// stencil coefficients are organized as
// {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
fn f_spmv(n:i16) void {
    stencil_mod.spmv(n, &stencil_coeff, &x, &y);
}

fn f_sync( n: i16 ) void {
   reduce_mod.allreduce(n, &dot, reduce_mod.TYPE_BINARY_OP.ADD);
}

fn f_reference_timestamps() void {

    time_ref_u16[0] = reduce_mod.tscRefBuffer[0];
    time_ref_u16[1] = reduce_mod.tscRefBuffer[1];
    time_ref_u16[2] = reduce_mod.tscRefBuffer[2];

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

task f_startup() void {
    timestamp.enable_tsc();
}

comptime {

    @activate(STARTUP);

    @bind_local_task(f_startup, STARTUP);
}

comptime {
    @export_symbol(ptr_x, "x");
    @export_symbol(ptr_y, "y");
    @export_symbol(ptr_stencil_coeff, "stencil_coeff");
    @export_symbol(ptr_time_buf_u16, "time_buf_u16");
    @export_symbol(ptr_time_ref, "time_ref");
}

comptime{
    @export_symbol(f_tic);
    @export_symbol(f_toc);
    @export_symbol(f_memcpy_timestamps);
    @export_symbol(f_spmv);
    @export_symbol(f_sync);
    @export_symbol(f_reference_timestamps);
}

```

## run.py

```python
#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" test 7-point stencil

    The Laplacian operator L on 3-dimensional domain can be represented by 7-point
  stencil based on the standard 2nd order Finite Difference Method. The operator form
  with Dirichlet boundary conditions can be written by
         L[u](i,j,k) = u(i+1, j,  k  ) + u(i-1, j,   k  ) +
                       u(i,   j+1,k  ) + u(i,   j-1, k  ) +
                       u(i,   j,  k+1) + u(i,   j,   k-1) +
                      -6*u(i, j, k)
  In general the coefficients of those 7 points can vary. To minimize the memory
  consumption, this example assumes the coefficients are independent of index k and
  whole vector u(i,j,:) is placed in one PE (px=j, py=i).
  The above formula can be re-written by
     c_west   * x[i-1][j  ][k  ] + c_east  * x[i+1][j  ][k  ] +
     c_south  * x[i  ][j-1][k  ] + c_north * x[i  ][j+1][k  ] +
     c_bot    * x[i  ][j  ][k-1] + c_top   * x[i  ][j  ][k+1] +
     c_center * x[i][j][k]
  Each PE only holds 7 coefficients organized by c_west, c_east, c_south, c_north,
  c_bot, c_top and c_center.

  This example provides two modules, one is allreduce and the other is stencil_3d_7pts.
  "allreduce" module can synchronize all PEs to form a reference clock.
  "stencil_3d_7pts" module can compute y = A*x where A is the matrix from 7-point stencil

  The framework is
  ---
       sync()      // synchronize all PEs to sample the reference clock
       tic()       // record start time
       spmv(zdim)  // compute y = A*x
       toc()       // record end time
  ---

  The tic() samples "time_start" and toc() samples "time_end". The sync() samples
  "time_ref" which is used to shift "time_start" and "time_end".
  The elapsed time is measured by
       cycles_send = max(time_end) - min(time_start)

  The overall runtime is computed via the following formula
       time_send = (cycles_send / 0.85) *1.e-3 us
  where a PE runs with clock speed 850MHz

  Each PE needs to gather six f32 from six neighbors, the cost of the communication is
        6*h*w*zDim*4 bytes
  where w-by-h is the core rectangle and zDim is the length of local vector.

  Here is the list of parameters:
    -m=<int> is the height of the core
    -n=<int> is the width of the core
    -k=<int> is size of x and y allocated in the core
    --zDim=<int> is the number of f32 per PE, computed by y = A*x
                 zDim must be not greater than k
    --channels=<int> specifies the number of I/O channels, no bigger than 16
"""

import random
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from cmd_parser import parse_args
from util import hwl_2_oned_colmajor, laplacian, oned_to_hwl_colmajor

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

def float_to_hex(f):
  return hex(struct.unpack("<I", struct.pack("<f", f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def csl_compile_core(
    cslc: str,
    width: int,  # width of the core
    height: int,  # height of the core
    pe_length: int,
    blockSize: int,
    file_config: str,
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,  # fabric-offsets of the core
    core_fabric_offset_y: int,
    use_precompile: bool,
    arch: Optional[str],
    C0: int,
    C1: int,
    C2: int,
    C3: int,
    C4: int,
    C5: int,
    C6: int,
    C7: int,
    C8: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
):
  if not use_precompile:
    args = []
    args.append(cslc)  # command
    args.append(file_config)
    args.append(f"--fabric-dims={fabric_width},{fabric_height}")
    args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
    args.append(f"--params=width:{width},height:{height},MAX_ZDIM:{pe_length}")
    args.append(f"--params=BLOCK_SIZE:{blockSize}")
    args.append(f"--params=C0_ID:{C0}")
    args.append(f"--params=C1_ID:{C1}")
    args.append(f"--params=C2_ID:{C2}")
    args.append(f"--params=C3_ID:{C3}")
    args.append(f"--params=C4_ID:{C4}")
    args.append(f"--params=C5_ID:{C5}")
    args.append(f"--params=C6_ID:{C6}")
    args.append(f"--params=C7_ID:{C7}")
    args.append(f"--params=C8_ID:{C8}")

    args.append(f"-o={elf_dir}")
    if arch is not None:
      args.append(f"--arch={arch}")
    args.append("--memcpy")
    args.append(f"--channels={channels}")
    args.append(f"--width-west-buf={width_west_buf}")
    args.append(f"--width-east-buf={width_east_buf}")

    print(f"subprocess.check_call(args = {args}")
    subprocess.check_call(args)
  else:
    print("\tuse pre-compile ELFs")

def main():
  """Main method to run the example code."""

  random.seed(127)

  args, dirname = parse_args()

  cslc = "cslc"
  if args.driver is not None:
    cslc = args.driver

  print(f"cslc = {cslc}")

  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  channels = args.channels
  assert channels <= 16, "only support up to 16 I/O channels"
  assert channels >= 1, "number of I/O channels must be at least 1"

  print(f"width_west_buf = {width_west_buf}")
  print(f"width_east_buf = {width_east_buf}")
  print(f"channels = {channels}")

  height = args.m
  width = args.n
  pe_length = args.k
  zDim = args.zDim
  blockSize = args.blockSize

  print(
      f"width={width}, height={height}, pe_length={pe_length}, zDim={zDim}, blockSize={blockSize}"
  )
  assert pe_length >= 2, "the maximum size of z must be greater than 1"
  assert zDim <= pe_length, "[0, zDim) cannot exceed the storage"

  np.random.seed(2)
  # A is h-by-w-by-l
  x = (np.arange(height * width * pe_length).reshape(height, width, pe_length).astype(np.float32) +
       100)

  x_1d = hwl_2_oned_colmajor(height, width, pe_length, x, np.float32)

  # stencil coefficients has the following order
  # {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
  stencil_coeff = np.zeros((height, width, 7), dtype=np.float32)
  for i in range(height):
    for j in range(width):
      stencil_coeff[(i, j, 0)] = -1  # west
      stencil_coeff[(i, j, 1)] = -2  # east
      stencil_coeff[(i, j, 2)] = -3  # south
      stencil_coeff[(i, j, 3)] = -4  # north
      stencil_coeff[(i, j, 4)] = -5  # bottom
      stencil_coeff[(i, j, 5)] = -6  # top
      stencil_coeff[(i, j, 6)] = 6  # center

  stencil_coeff_1d = hwl_2_oned_colmajor(height, width, 7, stencil_coeff, np.float32)

  y_ref = np.zeros((height, width, pe_length), dtype=np.float32)

  laplacian(stencil_coeff, zDim, x, y_ref)

  # fabric-offsets = 1,1
  fabric_offset_x = 1
  fabric_offset_y = 1
  # starting point of the core rectangle = (core_fabric_offset_x, core_fabric_offset_y)
  # memcpy framework requires 3 columns at the west of the core rectangle
  # memcpy framework requires 2 columns at the east of the core rectangle
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y
  # (min_fabric_width, min_fabric_height) is the minimal dimension to run the app
  min_fabric_width = core_fabric_offset_x + width + 2 + 1 + width_east_buf
  min_fabric_height = core_fabric_offset_y + height + 1

  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)

  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height

  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  # prepare the simulation
  print("store ELFs and log files in the folder ", dirname)

  # layout of a rectangle
  code_csl = "src/layout.csl"

  C0 = 0
  C1 = 1
  C2 = 2
  C3 = 3
  C4 = 4
  C5 = 5
  C6 = 6
  C7 = 7
  C8 = 8

  csl_compile_core(
      cslc,
      width,
      height,
      pe_length,
      blockSize,
      code_csl,
      dirname,
      fabric_width,
      fabric_height,
      core_fabric_offset_x,
      core_fabric_offset_y,
      args.run_only,
      args.arch,
      C0,
      C1,
      C2,
      C3,
      C4,
      C5,
      C6,
      C7,
      C8,
      channels,
      width_west_buf,
      width_east_buf,
  )
  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

  symbol_x = runner.get_id("x")
  symbol_y = runner.get_id("y")
  symbol_stencil_coeff = runner.get_id("stencil_coeff")
  symbol_time_buf_u16 = runner.get_id("time_buf_u16")
  symbol_time_ref = runner.get_id("time_ref")

  runner.load()
  runner.run()

  print("copy vector x of type f32")
  # the size of x per PE is pe_length
  runner.memcpy_h2d(
      symbol_x,
      x_1d,
      0,
      0,
      width,
      height,
      pe_length,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  print("copy coefficients of type f32")
  # each PE holds 7 coefficients
  runner.memcpy_h2d(
      symbol_stencil_coeff,
      stencil_coeff_1d,
      0,
      0,
      width,
      height,
      7,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  print("step 1: sync all PEs")
  runner.launch("f_sync", np.int16(1), nonblock=False)

  print("step 2: tic() records time_start")
  runner.launch("f_tic", nonblock=True)

  print(f"step 3: compute y = A*x with zDim = {zDim}")
  # positive zDim can be smaller than pe_length
  runner.launch("f_spmv", np.int16(zDim), nonblock=False)

  print("step 4: toc() records time_end")
  runner.launch("f_toc", nonblock=False)

  print("step 5: prepare (time_start, time_end)")
  runner.launch("f_memcpy_timestamps", nonblock=False)

  print("step 6: D2H (time_start, time_end)")
  time_memcpy_hwl_1d = np.zeros(height * width * 6, np.uint32)
  runner.memcpy_d2h(
      time_memcpy_hwl_1d,
      symbol_time_buf_u16,
      0,
      0,
      width,
      height,
      6,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_16BIT,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  time_memcpy_hwl = oned_to_hwl_colmajor(height, width, 6, time_memcpy_hwl_1d, np.uint16)

  print("step 7: D2H y of type f32")
  y_1d = np.zeros(height * width * pe_length, np.float32)
  runner.memcpy_d2h(
      y_1d,
      symbol_y,
      0,
      0,
      width,
      height,
      pe_length,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  y_wse = np.reshape(y_1d, (height, width, pe_length), order="F")

  print("step 8: prepare reference clock")
  runner.launch("f_reference_timestamps", nonblock=False)

  print("step 9: D2H reference clock")
  time_ref_1d = np.zeros(height * width * 3, np.uint32)
  runner.memcpy_d2h(
      time_ref_1d,
      symbol_time_ref,
      0,
      0,
      width,
      height,
      3,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_16BIT,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  time_ref_hwl = oned_to_hwl_colmajor(height, width, 3, time_ref_1d, np.uint16)

  runner.stop()

  if args.simulator:
    # move simulation log and core dump to the given folder
    dst_log = Path(f"{dirname}/sim.log")
    src_log = Path("sim.log")
    if src_log.exists():
      shutil.move(src_log, dst_log)

    dst_trace = Path(f"{dirname}/simfab_traces")
    src_trace = Path("simfab_traces")
    if dst_trace.exists():
      shutil.rmtree(dst_trace)
    if src_trace.exists():
      shutil.move(src_trace, dst_trace)

  # time_start = start time of spmv
  time_start = np.zeros((height, width)).astype(int)
  # time_end = end time of spmv
  time_end = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_memcpy_hwl[(h, w, 0)]
      word[1] = time_memcpy_hwl[(h, w, 1)]
      word[2] = time_memcpy_hwl[(h, w, 2)]
      time_start[(h, w)] = make_u48(word)
      word[0] = time_memcpy_hwl[(h, w, 3)]
      word[1] = time_memcpy_hwl[(h, w, 4)]
      word[2] = time_memcpy_hwl[(h, w, 5)]
      time_end[(h, w)] = make_u48(word)

  # time_ref = reference clock
  time_ref = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_ref_hwl[(h, w, 0)]
      word[1] = time_ref_hwl[(h, w, 1)]
      word[2] = time_ref_hwl[(h, w, 2)]
      time_ref[(h, w)] = make_u48(word)

  # adjust the reference clock by the propagation delay
  # the right-bottom PE signals other PEs, the propagation delay is
  #     (h-1) - py + (w-1) - px
  for py in range(height):
    for px in range(width):
      time_ref[(py, px)] = time_ref[(py, px)] - ((width + height - 2) - (px + py))

  # shift time_start and time_end by time_ref
  time_start = time_start - time_ref
  time_end = time_end - time_ref

  # cycles_send = time_end[(h,w)] - time_start[(h,w)]
  # 850MHz --> 1 cycle = (1/0.85) ns = (1/0.85)*1.e-3 us
  # time_send = (cycles_send / 0.85) *1.e-3 us
  #
  # each PE needs to gather six f32 from six neighbors, the cost of the communication is
  #      6*h*w*zDim*4 bytes
  #
  # bandwidth = (((wvlts-1) * 4)/time_send) MBS
  wvlts = 6 * height * width * zDim
  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) * 1.0e-3
  bandwidth = (wvlts * 4) / time_send
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")
  print(f"bandwidth = {bandwidth} MB/S ")

  z = y_ref.ravel() - y_wse.ravel()
  nrm_z = np.linalg.norm(z, np.inf)
  print(f"|y_ref - y_wes| = {nrm_z}")
  np.testing.assert_allclose(y_ref.ravel(), y_wse.ravel(), 1.0e-5)
  print("\nSUCCESS!")

if __name__ == "__main__":
  main()

```

## cmd\_parser.py

```python
"""command parser for bandwidthTest

   -m <int>     number of rows of the core rectangle
   -n <int>     number of columns of the core rectangle
   -k <int>     number of elements of local tensor
   --zDim <int>   number of elements to compute y=A*x
   --blockSize <int>  the size of temporary buffers for communication
   --latestlink   working directory
   --driver     path to CSL compiler
   --fabric-dims  fabric dimension of a WSE
   --cmaddr       IP address of a WSE
   --channels        number of I/O channels, 1 <= channels <= 16
   --width-west-buf  number of columns of the buffer in the west of the core rectangle
   --width-east-buf  number of columns of the buffer in the east of the core rectangle
   --compile-only    compile ELFs
   --run-only        run the test with precompiled binary
"""

import argparse
import os

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", default=1, type=int, help="number of rows")
  parser.add_argument("-n", default=1, type=int, help="number of columns")
  parser.add_argument("--simulator", action="store_true", help="Runs on simulator")
  parser.add_argument("-k", default=1, type=int, help="size of local tensor, no less than 2")
  parser.add_argument(
      "--zDim",
      default=2,
      type=int,
      help="[0 zDim-1) is the domain of Laplacian",
  )
  parser.add_argument("--latestlink", help="folder to contain the log files (default: latest)")
  parser.add_argument("-d", "--driver", help="The path to the CSL compiler")
  parser.add_argument("--compile-only", help="Compile only", action="store_true")
  parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")
  parser.add_argument("--cmaddr", help="CM address and port, i.e. <IP>:<port>")
  parser.add_argument("--run-only", help="Run only", action="store_true")
  parser.add_argument("--arch", help="wse2 or wse3. Default is wse2 when not supplied.")
  parser.add_argument("--width-west-buf", default=0, type=int, help="width of west buffer")
  parser.add_argument("--width-east-buf", default=0, type=int, help="width of east buffer")
  parser.add_argument(
      "--channels",
      default=1,
      type=int,
      help="number of I/O channels, between 1 and 16",
  )
  parser.add_argument(
      "--blockSize",
      default=2,
      type=int,
      help="the size of temporary buffers for communication",
  )

  args = parser.parse_args()

  logs_dir = "latest"
  if args.latestlink:
    logs_dir = args.latestlink

  dir_exist = os.path.isdir(logs_dir)
  if dir_exist:
    print(f"{logs_dir} already exists")
  else:
    print(f"create {logs_dir} to store log files")
    os.mkdir(logs_dir)

  if args.cmaddr is None:
    args.simulator = False

  return args, logs_dir

```

## benchmark-libs/stencil\_3d\_7pts/layout.csl

```csl
param colors:[8]color;
param entrypoints:[3]local_task_id;
param width : i16 ;   // width of the core
param height: i16 ;   // height of the core

const C0 : color = colors[0];
const C1 : color = colors[1];
const C2 : color = colors[2];
const C3 : color = colors[3];
const C4 : color = colors[4];
const C5 : color = colors[5];
const C6 : color = colors[6];
const C7 : color = colors[7];

// entrypoints of sync module
const SEND: local_task_id = entrypoints[0];
const RECV: local_task_id = entrypoints[1];
const COMM: local_task_id = entrypoints[2];

fn get_params(px:i16, py:i16) comptime_struct {

    var first_py: bool = (0 == py);
    var last_py: bool = ((height-1) == py);
    var is_py_even: bool = (0 == (py % 2));

    var first_px: bool = (0 == px);
    var last_px: bool = ((width-1) == px);
    var is_px_even: bool = (0 == (px % 2));

    // C0, C1:recv_west, send_east
    //         C0     C1     C0     C1     C0
    // West P0 --> P1 --> P2 --> P3 --> P4 --> P5 East
    //
    var c_recv_west: color = C1;
    var c_send_east: color = C0;
    if (is_px_even){
        c_recv_west = C1;
        c_send_east = C0;
    }else{
        c_recv_west = C0;
        c_send_east = C1;
    }

    // C2, C3: recv_east, send_west
    //          C2     C3     C2     C3     C2
    // West P0 <-- P1 <-- P2 <-- P3 <-- P4 <-- P5 East
    //
    var c_recv_east: color = C2;
    var c_send_west: color = C3;
    if (is_px_even){
        c_recv_east = C2;
        c_send_west = C3;
    }else{
        c_recv_east = C3;
        c_send_west = C2;
    }

    // C4, C5: recv_south, send_north
    //           C4     C5     C4     C5     C4
    // North P0 <-- P1 <-- P2 <-- P3 <-- P4 <-- P5 south
    //
    var c_recv_south: color = C4;
    var c_send_north: color = C5;
    if (is_py_even){
        c_recv_south = C4;
        c_send_north = C5;
    }else{
        c_recv_south = C5;
        c_send_north = C4;
    }

    // C6, C7: recv_north, send_south
    //           C6     C7     C6     C7     C6
    // North P0 --> P1 --> P2 --> P3 --> P4 --> P5 south
    //
    var c_recv_north: color = C7;
    var c_send_south: color = C6;
    if (is_py_even){
        c_recv_north = C7;
        c_send_south = C6;
    }else{
        c_recv_north = C6;
        c_send_south = C7;
    }

    return .{
        .c_recv_west = c_recv_west,
        .c_send_east = c_send_east,
        .c_recv_east = c_recv_east,
        .c_send_west = c_send_west,
        .c_recv_south = c_recv_south,
        .c_send_north = c_send_north,
        .c_recv_north = c_recv_north,
        .c_send_south = c_send_south,

        .SEND = SEND,
        .RECV = RECV,
        .COMM = COMM,

        .first_px = first_px,
        .last_px = last_px,
        .first_py = first_py,
        .last_py = last_py,
    };
}

```

## benchmark-libs/stencil\_3d\_7pts/pe.csl

```csl
param c_recv_west: color;
param c_send_east: color;
param c_recv_east: color;
param c_send_west: color;

param c_recv_south: color;
param c_send_north: color;
param c_recv_north: color;
param c_send_south: color;

param COMM: local_task_id; // entrypoint f_comm
param SEND: local_task_id; // entrypoint f_send
param RECV: local_task_id; // entrypoint f_recv

param first_px: bool;
param last_px: bool;
param first_py: bool;
param last_py: bool;

// To continue next command, f_callback = sys_mod.unblock_cmd_stream
param f_callback : fn ()void;

param input_queues:[4]u16;
// WSE2:
//   param output_queues:[1]u16;
// WSE3:
//   param output_queues:[4]u16;
param output_queues = {};

// only WSE3 needs output_ut_id
param output_ut_id = {};

param BLOCK_SIZE: i16; // size of temporary buffers for communication

// explicit DSR allocation
param dest_dsr_ids:[2]u16;
param src0_dsr_ids:[1]u16;
param src1_dsr_ids:[2]u16;

const api_wse3 = @is_arch("wse3");

// The user must specify --import-path=<path to benchmark-libs>
fn get_stencil_module() comptime_string {
  if (api_wse3) {
    return "../../benchmark-libs/stencil_3d_7pts/wse3/pe.csl";
  }else{
    return "../../benchmark-libs/stencil_3d_7pts/wse2/pe.csl";
  }
}

const stencilParams = .{
  .c_recv_west = c_recv_west,
  .c_send_east = c_send_east,
  .c_recv_east = c_recv_east,
  .c_send_west = c_send_west,

  .c_recv_south = c_recv_south,
  .c_send_north = c_send_north,
  .c_recv_north = c_recv_north,
  .c_send_south = c_send_south,

  .COMM = COMM,
  .SEND = SEND,
  .RECV = RECV,

  .first_px = first_px,
  .last_px = last_px,
  .first_py = first_py,
  .last_py = last_py,

  .f_callback = f_callback,

  .input_queues = input_queues,
  .output_queues = output_queues,
  .output_ut_id = output_ut_id,

  .BLOCK_SIZE = BLOCK_SIZE,

  .dest_dsr_ids = dest_dsr_ids,
  .src0_dsr_ids = src0_dsr_ids,
  .src1_dsr_ids = src1_dsr_ids
};

const stencil_mod = @import_module(get_stencil_module(), stencilParams);

fn spmv(n: i16, coeff: *[7]f32, x: [*]f32, y: [*]f32) void {
    stencil_mod.spmv(n, coeff, x, y);
}

```

## benchmark-libs/allreduce/layout.csl

```csl
param colors: [1]color;
param entrypoints: [4]local_task_id;
param width: i16 ;   // width of the core
param height: i16 ;  // height of the core

const C0: color = colors[0];

// entrypoints of allreduce module
const SEND_CTRL: local_task_id = entrypoints[0];
const SEND_DATA: local_task_id = entrypoints[1];
const STATE_ENTRY: local_task_id = entrypoints[2];
// LOCK runs only if teardown is received and the operation is done
// LOCK performs the state transition
// teardown handler activates LOCK
// the operation blocks LOCK in the beginning and unblocks it when it finishes
const C_LOCK: local_task_id = entrypoints[3];

fn get_params(px:i16, py:i16) comptime_struct {

    var first_py: bool = (0 == py);
    var last_py: bool = ((height-1) == py);

    var first_px: bool = (0 == px);
    var last_px: bool = ((width-1) == px);

    return .{
        .first_px = first_px,
        .last_px = last_px,
        .first_py = first_py,
        .last_py = last_py,
        .C_ROUTE = C0,
        .C_SEND_CTRL = SEND_CTRL,
        .C_SEND_DATA = SEND_DATA,
        .C_STATE_ENTRY = STATE_ENTRY,
        .C_LOCK = C_LOCK,
        .width = width,
        .height = height
    };
}

```

## benchmark-libs/allreduce/pe.csl

```csl
// allreduce module has the following three operations
//  - row reduction
//  - column reduction
//  - broadcasting
//
// It only uses a single routable color, three entrypoints and a single
// input/output queue. Any user's kernel can combine this module with other
// modules without running out of resources.
//
// 1. row reduction
//   The reduction is from left to right. The last PE (px=width-1) receives
//   all data from the neighbors one by one. The result is stored in the last
//   PE, other PEs do not change the content.
//
// 2. column reduction
//   The reduction is from north to south. The last PE (py=height-1) receives
//   all data from the neighbors one by one. The result is stored in the last
//   PE, other PEs do not change the content.
//
// 3. broadcast
//   The right-bottom PE (px=width-1, py=height-1) broadcasts the data upwards to
//   whole column, then each PE in this column broadcasts data to its west neighbors.
//

// The portal allreduce_nrm2() computes nrm2(x) by using allreduce(MAX)
// x must be a scalar (for simplicity)
// Here is the sequence of operations
// 1. allreduce(MAX, |x|)
//    xmax = max(|x|) overwrites |x|
// 2. SCALE_AND_SQUARE
//    alpha = approx(xmax)
//    |x| = |x|/alpha
//    |x| = |x| * |x|
// 3. allreduce(ADD, |x|)
//    |x| = sum{ (xj/alpha)^2 }
// 4. NRM2
//    |x| = alpha * sqrt(|x|)
//    All PEs perform NRM2 because of broadcasting, so we don't need to broadcast
//    the final result to all PEs.
//
// The state machine has 9 states
//   # (1) allreduce(MAX)
//   wvlts_per_pe = 1
//   functorop = MAX
//   state_seq[0] = STATE_ROW_REDUCE;
//   state_seq[1] = STATE_COL_REDUCE;
//   state_seq[2] = STATE_BCAST;
//   # (2) SCALE_AND_SQUARE
//   state_seq[3] = STATE_SCALE_AND_SQUARE; // next operation is ADD
//   # (3) allreduce(ADD)
//   state_seq[4] = STATE_ROW_REDUCE;
//   state_seq[5] = STATE_COL_REDUCE;
//   state_seq[6] = STATE_BCAST;
//   # (4) NRM2
//   state_seq[7] = STATE_NRM2;
//   # (5) END
//   state_seq[8] = STATE_DONE;
//

// How to assign explicit DSRs
//
// reduction:
//  last PE: f_send_data --> @fadds(mem_x_buf_dsd, mem_x_buf_dsd, fab_recv_wdsd, .{.async=true, .activate=f_send_data} );
//                 ^                          |
//                 |--------------------------+
//  others: f_send_data --> @mov32(fab_trans_x_wdsd, mem_x_buf_dsd, .{.async=true, .activate=f_send_ctrl} );
//          --> @mov32(fab_trans_ctrl_wdsd, mem_ctrl_buf_dsd, .{.async=true, .activate=f_send_data } );
//          --> f_send_data
//          1st PE: @mov32(fab_trans_ctrl_wdsd, mem_buf_td_dsd, .{.async=true} );
//
// bcast:
//  right-bottom PE: @mov32(fab_trans_x_wdsd, mem_x_buf_dsd, .{.async=true, .activate=f_send_ctrl} );
//                   --> @mov32(fab_trans_ctrl_wdsd, mem_buf_td_dsd, .{.async=true} );
//  others: @mov32(mem_x_buf_dsd, fab_recv_wdsd, .{.async=true} );
//
// Only one dest DSR, one src0 DSR and one src1 DSR are enough because
// - the teardown separates different operations
// - when TD arrives, sender has sent out the data/ctrl
//   the receiver has received all data because there is only one color
// - all DSD operations are serialized
//
// For example:
//   dest_dsr = @get_dsr(dsr_dest, 1);
//   src0_dsr = @get_dsr(dsr_src0, 1);
//   src1_dsr = @get_dsr(dsr_src1, 1);
//

// The sequence of LOCK of { row_reduce, col_reduce, bcast}
//
//  row_reduce blocks LOCK
//  T29 activates LOCK
//  row_reduce unblocks LOCK when it finishes
//
//  LOCK goes to next state
//
//  col_reduce blocks LOCK
//  T29 activates LOCK
//  col_reduce unblocks LOCK when it finishes
//
//  LOCK goes to next state
//
//  bcast blocks LOCK
//  T29 activates LOCK
//  bcast unblocks LOCK when it finishes
//
//  LOCK goes to next state (done)
//

param C_ROUTE: color;

param C_SEND_CTRL: local_task_id;  // send switch advance
param C_SEND_DATA: local_task_id;  // send data
param C_STATE_ENTRY: local_task_id; // state machine
// LOCK runs only if teardown is received and the operation is done
// LOCK performs the state transition
// teardown handler activates LOCK
// the operation blocks LOCK in the beginning and unblocks it when it finishes
param C_LOCK: local_task_id;

param first_px: bool; // (0 == px)
param last_px: bool;  // ((width-1) == px)
param first_py: bool; // (0 == py)
param last_py: bool;  // ((height-1) == py)

// row reduction needs to receive width-1 neighbors
// column reduction needs to receive height-1 neighbors
param width: i16;
param height: i16;

// f_callback = sys_mod.unblock_cmd_stream, to continue next command
param f_callback: fn ()void;

// last PE uses this ID as the input queue
// others use this ID as the output queue
param queues:[1]u16;

// explicit DSR allocation
param dest_dsr_ids: [1]u16;
param src0_dsr_ids: [1]u16;
param src1_dsr_ids: [1]u16;

const timestamp = @import_module("<time>");

const math_lib = @import_module("<math>");

// A new type for binary operators
// compiler assigns ADD=0 and MAX=1
const TYPE_BINARY_OP = enum(u16) { ADD, MAX };

// tsc_size_words = 3
var tscRefBuffer = @zeros([timestamp.tsc_size_words]u16);

////////////////////////////////////////////////////////////////////////////////
// Main memory (48KB)
////////////////////////////////////////////////////////////////////////////////

var x: [*]f32;

var functor: TYPE_BINARY_OP = TYPE_BINARY_OP.ADD;

const STATE_ROW_REDUCE: i16 = 0;
const STATE_COL_REDUCE: i16 = 1;
const STATE_BCAST: i16 = 2;
const STATE_SCALE_AND_SQUARE: i16 = 3;
const STATE_NRM2: i16 = 4;
const STATE_DONE: i16 = 5;

// allreduce(ADD/MAX) has four states
// allreduce_nrm2 has 9 states
// "+1" is to avoid out-of-bound if
// STATE_DONE also dereference next state
var state_seq = @zeros([9+1]i16);
var state_idx: i16 = 0;
var cur_state: i16 = 0;
var next_state: i16 = 0;

// record the reduction length from the caller
var wvlts_per_pe: u16 = 0;

// number of PEs involed in the reduction: last PE needs to count number of received neighbors
// WARNING: reduce_pes only records number of received PEs
//   row reduction: width-1
//   column reduction: height-1
// If reduce_pes is wrong, simfab shows re-entry error of UT when row reduction and col reduction
// are combined because row reduction has extra UT1 waiting for wavelets
var reduce_pes: i16 = 0;
// 1st PE during the reduction: send TD to others
//   row reduction: {px = 0}
//   column reduction: {py = 0}
var reduce_first_pe: bool;
// last PE during the reduction: receive data from w-1 or h-1 neighbors
//   row reduction: {px = w-1}
//   column reduction: {py = h-1}
var reduce_last_pe: bool;

// last PE uses count_recv_or_send to receive data from w-1 neighbors
// other PEs use count_recv_or_send to send data and control
var count_recv_or_send: i16 = 0;

const dest_dsr = @get_dsr(dsr_dest, dest_dsr_ids[0]);
const src0_dsr = @get_dsr(dsr_src0, src0_dsr_ids[0]);
const src1_dsr = @get_dsr(dsr_src1, src1_dsr_ids[0]);

const iq_route = @get_input_queue(queues[0]);
const oq_route = @get_output_queue(queues[0]);

// The portal function of allreduce(ADD/MAX)
//
// How to use:
//  reduce_mod = = @import_module( "<allreduce/pe>");
//  reduce_mod.allreduce(n, x);
//  The user has to prepare input vector x.
//
//  When allreduce() finishes, it will call user's callback.
//
// case 1: row reduction
//   state_seq = {STATE_ROW_REDUCE, STATE_DONE}
// case 2: column reduction
//   state_seq = {STATE_COL_REDUCE, STATE_DONE}
// case 3: row + column reduction
//   state_seq = {STATE_ROW_REDUCE, STATE_COL_REDUCE, STATE_DONE}
// case 4: broadcast
//   state_seq = {STATE_BCAST, STATE_DONE}
//
fn allreduce( n: i16, in_tensor: [*]f32, op: TYPE_BINARY_OP ) void {

   x = in_tensor;
   functor = op;

   @assert(n > 0);

   wvlts_per_pe = @bitcast(u16, n);

   // setup state sequence
   state_seq[0] = STATE_ROW_REDUCE;
   state_seq[1] = STATE_COL_REDUCE;
   state_seq[2] = STATE_BCAST;
   state_seq[3] = STATE_DONE;

   state_idx = 0;
   cur_state = state_seq[0];
   @activate(C_STATE_ENTRY);
}

// nrm2_x_copy keeps a copy of x during the nrm2 because
// the x is used by the allreduce
// After allreduce(MAX,x), all PEs have the same x = max(|xj|)
// nrm2_x_copy is used in scale_and_square:
//    alpha = approx(x[0])
//    x[0] <- (nrm2_x_copy / alpha)^2
// Then allreduce(ADD, x) updates x[0] = (|x|_2/alpha)^2
var nrm2_x_copy: f32;

// The portal function rnm2
//
// It only computes nrm2(x[0]) because
// - common case is n = 1
// - no SIMD on sqrt
//
fn allreduce_nrm2(in_tensor: [*]f32) void {

    x = in_tensor;
    functor = TYPE_BINARY_OP.MAX;
    wvlts_per_pe = 1; // nrm2 of x[0]

    // x <-- |xj|
    var xreg = x[0];
    xreg = math_lib.abs(xreg);
    x[0] = xreg;
    // nrm2_x_copy can keep either xj or |xj|
    nrm2_x_copy = xreg;

    // setup state sequence
    // (1) allreduce(MAX)
    state_seq[0] = STATE_ROW_REDUCE;
    state_seq[1] = STATE_COL_REDUCE;
    state_seq[2] = STATE_BCAST;
    // x[0] = max(|xj|)
    // (2) SCALE_AND_SQUARE
    // x[0] = (|xj|/alpha)^2
    state_seq[3] = STATE_SCALE_AND_SQUARE; // next operation is ADD
    // (3) allreduce(ADD)
    state_seq[4] = STATE_ROW_REDUCE;
    state_seq[5] = STATE_COL_REDUCE;
    state_seq[6] = STATE_BCAST;
    // x[0] = sum{(|xj|/alpha)^2}
    // (4) NRM2
    // x[0] = |x|_2
    state_seq[7] = STATE_NRM2;
    // (5) END
    state_seq[8] = STATE_DONE;

    state_idx = 0;
    cur_state = state_seq[0];
    @activate(C_STATE_ENTRY);
}

////////////////////////////////////////////////////////////////////////////////
// DSDs
// data-structure descriptors (DSDs), loaded into data-structure registers (DSRs)
//
// Queues 0,1: input depth 6 wavelets
// Queues 2,3: input depth 4 wavelets
// Queues 4-7: input depth 2 wavelets
//
// queues 0,1: output depth 2 wavelets
// queues 2,3: output depth 6 wavelets
// queues 4,5: output depth 2 wavelets
//
// Length of an operand:
// The length of all other types of DSRs is specified by the length field of its DSD. When
// the bits encoding the length are 0x7fff, the length is infinite.
//
// Length of the vector instruction is then determined in the following order:
// 1. If src0 has a non-zero length, that length is used
// 2. If src1 has a non-zero length, that length is used
// 3. If dst has a non-zero length, that length is used
// 4. if no operands have length (all operands are GPR), length = 1
////////////////////////////////////////////////////////////////////////////////

const dummy = @zeros([1]i16);

// rowReduce() binds mem_x_buf_dsd to pointer x and resets its length to n (given by the caller)
// Last PE adds data from neighbors to mem_x_buf_dsd
// Other PEs send mem_x_buf_dsd to the east
var mem_x_buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> dummy[i] });

// other PE (not last PE) uses this DSD to send x
var fab_trans_x_wdsd = @get_dsd(fabout_dsd, .{
    .extent = 1,
    .fabric_color = C_ROUTE,
    .output_queue = oq_route
});

// WARNING: control wavelet must be sent with the same microthread, via the same output buffer,
// otherwise, we may see only one data wavelet, then 2nd is the control wavelet, then
// the remaining data cannot be sent out because the routing is back to {.rx=WEST, .tx=EAST},
// there is no path from RAMP to the router.
const fab_trans_ctrl_wdsd = @get_dsd(fabout_dsd, .{
    .extent = 1,
    .control = true,
    .fabric_color = C_ROUTE,
    .output_queue = oq_route
});

// row reduction: the last PE receives the data from its w-1 neighbors,
// the receiving sequence is p0, p1, ..., p{w-1}.
// It uses the same queue ID because it does not send, only receives.
// It does not receive ctrl wavelets because of NOCE.
// f_send_data() receives data (w-1) times
//
var fab_recv_wdsd =  @get_dsd(fabin_dsd, .{
   .extent = 1,
   .fabric_color = C_ROUTE,
   .input_queue = iq_route
});

////////////////////////////////////////////////////////////////////////////////
// Tasks
// syntax
//     task_begin(name, entrypoint, color)
////////////////////////////////////////////////////////////////////////////////

const switches = @import_module("<memcpy/memcpy_switches>");

// The following arrays define values for control wavelets, which update the
// switch position at the recipient PEs.
// All are comptime constants
//
// ctrl_11 is for other PEs which changes switch of two consecutive PEs
var ctrl_11 = [1]u32 { switches.ctrl(switches.switch_cmd_11()) };

var mem_ctrl_buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> ctrl_11[i] });

// teardown from casm
// teardown_buf[0] = (1 << 31) | (0b111 << 22)
//
// teardown from csl
// from cslang/test-e2e/dynamic_filters/sender.csl
//  31=0x1f = no entrypoint
//
// teardown wavelet = 0x1df 0000
//const teardown_buf = [1]u32{(31 << 16) | 0b111 << 22};
// teardown wavelet = 0x9df 9249
const teardown_buf = [1]u32 { switches.ctrl(switches.teardown_cmd_1()) };

const mem_buf_td_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> teardown_buf[i] });

// the color C_ROUTE is in teardown mode at comptime by specifying {.teardown = true}
// reduce() and broadcast_to_all() block LOCK in the beginning and unblock it
// when the operation finishes
//
// WARNING: we don't block LOCK in the xxx_configure() because it is sequential, and
// it is more intuitive to unblock LOCK in reduce() and broadcast_to_all()
//
task f_state_entry() void {

    if (STATE_ROW_REDUCE == cur_state){
        // rowReduce_configure() reconfigures the pos0, pos1 and clears TIP
        rowReduce_configure();
        // perform row reduction and store the result to last PE
        // 1st PE will send TD to turn C_ROUTE back to teardown mode
        reduce_pes = width-1;
        reduce_first_pe = first_px;
        reduce_last_pe = last_px;
        reduce( wvlts_per_pe );

        // prefetch next state which will be copied into cur_state in teardown handler
        state_idx += 1;
        next_state = state_seq[state_idx];

    }else if (STATE_COL_REDUCE == cur_state){
        // colReduce_configure() reconfigures the pos0, pos1 and clears TIP
        colReduce_configure();
        // perform column reduction and store the result to last PE
        // 1st PE will send TD to turn C_ROUTE back to teardown mode
        reduce_pes = height-1;
        reduce_first_pe = first_py;
        reduce_last_pe = last_py;
        reduce( wvlts_per_pe );

        // prefetch next state which will be copied into cur_state in teardown handler
        state_idx += 1;
        next_state = state_seq[state_idx];

    }else if (STATE_BCAST == cur_state){
        // bcast_configure() reconfigures pos0 and disables pos1
        bcast_configure();
        // right-bottom PE broadcasts the data to others and also sends TD
        broadcast_to_all( wvlts_per_pe );

        // prefetch next state which will be copied into cur_state in teardown handler
        state_idx += 1;
        next_state = state_seq[state_idx];

    }else if (STATE_SCALE_AND_SQUARE == cur_state){
        // Assume allreduce(MAX) is done
        // x[0] = xmax = max({|xj|})

        // Update x[0] by (xmax/alpha)^2
        scale_and_square();
        // next reduction is allreduce(ADD)
        functor = TYPE_BINARY_OP.ADD;

        // prefetch next state which will be copied into cur_state in teardown handler
        // sequential code: f_lock is not triggered to assign (cur_state = next_state)
        // update cur_state directly
        state_idx += 1;
        cur_state = state_seq[state_idx];

        @activate(C_STATE_ENTRY);

    }else if (STATE_NRM2 == cur_state){
        // Assume allreduce(ADD) is done
        // x[0] = sum({(|xj|/alpha)^2})
        // Update x[0] by |x|_2
        nrm2_postprocessing();

        // prefetch next state which will be copied into cur_state in teardown handler
        // sequential code: f_lock is not triggered to assign (cur_state = next_state)
        // update cur_state directly
        state_idx += 1;
        cur_state = state_seq[state_idx];

        @activate(C_STATE_ENTRY);

    }else if (STATE_DONE == cur_state){
        // state machine is done, return control back to the caller
        timestamp.get_timestamp(&tscRefBuffer);

        f_callback();
    }else{
        @assert(false); // Error: unknown state
        // assert() is ignored by HW, it could hang here
        // To avoid a stall, trigger callback (the caveat is the wrong result)
        f_callback();
    }
}

fn reduce( n: u16 ) void {

    // WARNING: block LOCK in the beginning and only
    // unblock LOCK when "reduce" finishes
    @block(C_LOCK);

    count_recv_or_send = 0;

    // changes switch of of itself and its neighbor
    // The last PE does not call f_send_ctrl(), so this op is DON'T care
    mem_ctrl_buf_dsd =  @set_dsd_base_addr(mem_ctrl_buf_dsd, ctrl_11);

    mem_x_buf_dsd = @set_dsd_base_addr(mem_x_buf_dsd, x);
    mem_x_buf_dsd = @set_dsd_length(mem_x_buf_dsd, n);

    fab_recv_wdsd = @set_dsd_length(fab_recv_wdsd, n);
    fab_trans_x_wdsd = @set_dsd_length(fab_trans_x_wdsd, n);

    // last PE receives data from w-1 neighbors
    // other PEs send data and control to the east/south
    @activate(C_SEND_DATA);  // triggers f_send_data
}

fn broadcast_to_all( n: u16 ) void {

    // WARNING: block LOCK in the beginning and only
    // unblock LOCK when "broadcast" finishes
    @block(C_LOCK);

    // No PE sends switch advance
    // mem_ctrl_buf_dsd =  @set_dsd_base_addr(mem_ctrl_buf_dsd, ctrl_11);

    mem_x_buf_dsd = @set_dsd_base_addr(mem_x_buf_dsd, x);
    mem_x_buf_dsd = @set_dsd_length(mem_x_buf_dsd, n);
    fab_recv_wdsd = @set_dsd_length(fab_recv_wdsd, n);
    fab_trans_x_wdsd = @set_dsd_length(fab_trans_x_wdsd, n);

    if ( last_px and last_py ){
        // Pw-1,h-1 sends data and then f_send_ctrl sends a TD
        // f_send_ctrl() will unblock LOCK
        //@mov32(fab_trans_x_wdsd, mem_x_buf_dsd, .{.async=true, .activate=f_send_ctrl} );
        @load_to_dsr(dest_dsr, fab_trans_x_wdsd, .{.async=true, .activate=f_send_ctrl} );
        @load_to_dsr(src1_dsr, mem_x_buf_dsd);
        @mov32(dest_dsr, src1_dsr, .{.async=true} );
    }else{
        // other PEs receive data and wait for TD
        // unblock LOCK after data is received, T29 will activate LOCK
        //@mov32(mem_x_buf_dsd, fab_recv_wdsd, .{.async=true, .unblock=C_LOCK} );
        @load_to_dsr(dest_dsr, mem_x_buf_dsd);
        @load_to_dsr(src1_dsr, fab_recv_wdsd, .{.async=true, .unblock=C_LOCK} );
        @mov32(dest_dsr, src1_dsr, .{.async=true} );
    }
}

var alpha: f32;
var inv_alpha: f32;

// Assume the caller finishes allreduce(MAX,|x|) so x[0] = max({|xj|})
// Update x[0] by (x[0]/alpha)^2
// where
//     alpha = 2^(E-127) approximates x[0]
//
fn scale_and_square() void {
    var xreg: f32 = x[0];
    // (1) compute alpha
    approx(xreg, &alpha, &inv_alpha);
    // (2) scale x by x/alpha
    xreg = nrm2_x_copy;
    xreg = xreg * inv_alpha;
    // (3) square x
    // xreg is O(1), SQUARE does not overflow
    x[0] = xreg * xreg;
}

// Assume the caller has computed
//   x[0] = allreduce(ADD, (xj/alpha)^2)
//
// Update x[0] by |x|_2
//
fn nrm2_postprocessing() void{
    // x[0] = sum({(xj/alpha)^2}) = |x|^2 / alpha^2
    var xreg: f32 = x[0];
    xreg = math_lib.sqrt(xreg);
    x[0] = xreg * alpha;
}

// last PE does not send data, it only receives data
// row-reduce sequence: f_send_data() --> f_send_ctrl()
//                      ^                  |
//                      |------------------+
//
// f_send_data() is the last call when the reduction finishes
// unblock LOCK here when the operation is done
task f_send_data() void {
    if (reduce_last_pe){
        // last PE receives data from reduce_pes neighbors
        if (count_recv_or_send < reduce_pes){
            //@fadds(mem_x_buf_dsd, mem_x_buf_dsd, fab_recv_wdsd, .{.async=true, .activate=f_send_data} );
            @load_to_dsr(src1_dsr, fab_recv_wdsd, .{.async=true, .activate=f_send_data} );
            @load_to_dsr(src0_dsr, mem_x_buf_dsd);
            @load_to_dsr(dest_dsr, mem_x_buf_dsd);
            if (TYPE_BINARY_OP.ADD == functor){
                @fadds(dest_dsr, src0_dsr, src1_dsr, .{.async=true} );
            }else{
                @fmaxs(dest_dsr, src0_dsr, src1_dsr, .{.async=true} );
            }
            count_recv_or_send += 1;
        }else{
            // last PE has received all data from the reduce_pes neighbors
            // wait for TD from 1st PE
            // unblock LOCK, T29 will activate LOCK
            @unblock(C_LOCK);
        }
    }else{
        // other PE (not last PE) sends data and control
        if (count_recv_or_send < 1){
            //@mov32(fab_trans_x_wdsd, mem_x_buf_dsd, .{.async=true, .activate=f_send_ctrl} );
            @load_to_dsr(dest_dsr, fab_trans_x_wdsd, .{.async=true, .activate=f_send_ctrl} );
            @load_to_dsr(src1_dsr, mem_x_buf_dsd);
            @mov32(dest_dsr, src1_dsr, .{.async=true} );
            count_recv_or_send += 1;
        }else{
            // sending is done (including data wavelets and control wavelets)
            @unblock(C_LOCK);
            // only 1st PE sends TD to other PEs
            // T29 will activate LOCK
            if (reduce_first_pe){
                //@mov32(fab_trans_ctrl_wdsd, mem_buf_td_dsd, .{.async=true} );
                @load_to_dsr(dest_dsr, fab_trans_ctrl_wdsd, .{.async=true} );
                @load_to_dsr(src1_dsr, mem_buf_td_dsd);
                @mov32(dest_dsr, src1_dsr, .{.async=true} );
            }
        }
    }
}

task f_send_ctrl() void{
    if (STATE_BCAST == cur_state){
        //broadcast: Pw-1,h-1 only sends the TD
        // unblock LOCK after TD is sent out
        //@mov32(fab_trans_ctrl_wdsd, mem_buf_td_dsd, .{.async=true, .unblock=C_LOCK} );
        @load_to_dsr(dest_dsr, fab_trans_ctrl_wdsd, .{.async=true, .unblock=C_LOCK} );
        @load_to_dsr(src1_dsr, mem_buf_td_dsd);
        @mov32(dest_dsr, src1_dsr, .{.async=true} );
    }else{
        // reduction: other PEs (not last PE) sends switch advance
        //   last PE does not trigger f_send_ctrl because it only receives data
        // f_send_data() will unblock LOCK
        //@mov32(fab_trans_ctrl_wdsd, mem_ctrl_buf_dsd, .{.async=true, .activate=f_send_data } );
        @load_to_dsr(dest_dsr, fab_trans_ctrl_wdsd, .{.async=true, .activate=f_send_data } );
        @load_to_dsr(src1_dsr, mem_ctrl_buf_dsd);
        @mov32(dest_dsr, src1_dsr, .{.async=true} );
    }
}

// LOCK runs only if TD is received and the operation (*) finishes
//
// Here is the sequence
// - the operation blocks LOCK in the beginning
// - teardown handler activates LOCK
// - the operation unblocks LOCK when it finishes
// - LOCK is picked by the HW scheduler to perform the state transition
//
// (*) operation is {row_reduce, col_reduce, bcast}
//
task f_lock() void {
    cur_state = next_state; // go to next state
    @activate(C_STATE_ENTRY);
}

// (alpha, inv_alpha) = approx(x) approximates x by positive alpha such that
//     x = alpha * (x/alpha)
// where alpha = 2^(exp) and (x/alpha) has no precision loss.
//
// If x is a normal number, |x| = 2^(exp) * r, then alpha = 2^(exp)
//
// The purpose of this approximation is for nrm2(x).
// nrm2(x) can hit overflow if we just do square-sum.
// The simple workaround is to square-sum of x/max(x).
// However the division is very expensive, about 50 cycles.
// We just need a number alpha close to max(x) such that x/alpha = O(1).
// The cost of approx is about 11 instructions, much cheaper than div.
//
// Assume x = sign * 2^(E-127) * mantissa, "approx" handles the following
// four cases:
//
// case 1: x is a normal number
//    0 < E < 255
//    x is normal
//    x = sign * 2^(E-127) * 1.b22b21... b1b0
//    min(x) = 0x0080 0000
//           = 2^(-126) = 1.1754943508 x 10^(-38)
//    max(x) = 0x7f7f ffff
//           = 2^127 x (2 - 2^(-23)) = 3.4028234664 * 10^38
//
// case 2: x is a subnormal number
//    E = 0 and mantissa > 0
//    x = sign * 2^(-127) * b22.b21... b1b0
//      = sign * 2^(-126) * 0.b22b21... b1b0
//    min(x) = 0x000 0001
//           = 2^(-126) x 2^(-23) = 2^(-149) = 1.4*10^(-45)
//    max(x) = 007f ffff
//           = 2^(-126) x (1 - 2^(-23)) = 1.17 x 10^(-38)
//
// case 3: x = 0
//    E = 0 and mantissa = 0
//
// case 4: x = inf or nan
//    inf: E = 255 and mantissa = 0
//    nan: E = 255 and mantissa > 0
//
// Example 1: x = 14.0
//    alpha_u32 = 0x41000000
//    inv_alpha_u32 = 0x3e000000
//    alpha = 8.
//    inv_alpha = 0.125
// Example 2: x = 0.15625
//    alpha_u32 = 0x3e000000
//    inv_alpha_u32 = 0x41000000
//    alpha = 0.125
//    inv_alpha = 8.0
// Example 3: x = 1.e-43
//    alpha_u32 = 0x800000
//    inv_alpha_u32 = 0x7e800000
//    alpha = 1.1754943508222875e-38
//    inv_alpha = 8.507059173023462e+37
// Example 4: x = 1.0/0.0 (np.float32(np.inf))
//    alpha_u32 = 0x3f800000
//    inv_alpha_u32 = 0x3f800000
//    alpha = 1.0
//    inv_alpha = 1.0
// Example 5: x = 0.0/0.0 (np.float32(np.nan))
//    alpha_u32 = 0x3f800000
//    inv_alpha_u32 = 0x3f800000
//    alpha = 1.0
//    inv_alpha = 1.0
//
fn approx(x: f32, alpha: *f32, inv_alpha: *f32) void {
   const MASK_EXPONENT: u32 = 0x7F800000;
   const MASK_MANTISSA: u32 = 0x007FFFFF;
   const x_u32: u32 = @bitcast(u32, x);
   // x is presented by (sign | E | mantissa)
   // sign has 1 bit, E has 8 bits and mantissa has 23 bits
   // E = (x & MASK_EXPONEN) >> 23
   const exp: u32 = (x_u32 & MASK_EXPONENT);
   // mantissa = b22b21...b1b0 has 23-bit, need u32
   const mantissa: u32 = (x_u32) & MASK_MANTISSA;
   // E has 8-bit, use u16
   var E: u16 = @as(u16, (exp >> 23));

   // case 1: 0 < E < 255, x is normal
   // the following if-clause handles case 2, 3 and 4
   if (0 == E){
        if (0 == mantissa){
            // case 3: x = 0
            // reset alpha = 1
            E = 127;
        }else{
            // case 2: x is subnormal
            // reset alpha= 2^(-126)
            E = 1;
        }
    }
    if (255 == E){
        // case 4: x = inf or NAN
        // reset alpha = 1
        E = 127;
    }
    // alpha and inv_alpha are u32
    // alpha = 2^(E - 127)
    // inv_alpha = 1/alpha = 2^(127 - E)
    var alpha_u32: u32 = (@as(u32, E) << 23);
    var inv_alpha_u32: u32 = @as(u32, (254 - E)) << 23;

    alpha.* = @bitcast(f32, alpha_u32);
    inv_alpha.* = @bitcast(f32, inv_alpha_u32);
}

comptime {
    @bind_local_task(f_send_ctrl, C_SEND_CTRL);
    @bind_local_task(f_send_data, C_SEND_DATA);
    @bind_local_task(f_state_entry, C_STATE_ENTRY);
    @bind_local_task(f_lock, C_LOCK);
}

//----------------- the following is the routing of C_ROUTE

const tile_config = @import_module("<tile_config>");
const color_config = tile_config.color_config;
const switch_config = tile_config.switch_config;

fn rowReduce_configure() void {

    // (1) setup the new routing first because the pos1 needs to inherit from pos0
    const addr = color_config.get_color_config_addr(C_ROUTE);

    if (first_px){
        // 1st PE must has {rx = RAMP} to send out the data
        // .rx = .{ RAMP },.tx = .{ EAST },
        color_config.reset_routes(addr, .{.tx = EAST, .rx = RAMP});
    }else if (last_px){
        // last PE only receives data
        // .rx = .{ WEST }, .tx = .{ RAMP },
        color_config.reset_routes(addr, .{.tx = RAMP, .rx = WEST});
    }else{
        // 0 < px < width-1
        // .rx = .{ WEST }, .tx = .{ EAST },
        color_config.reset_routes(addr, .{.tx = EAST, .rx = WEST});
    }

    // (2) setup switch according to config parameters
    // 1. pos0 (color config reg)
    // 2. pos1 (switch config reg)
    //    pos1 = {.rx = RAMP} for all PEs except last PE

    // reset switch position to pos0
    // WARNING: if switch config register does not reset the switch position back to pos0,
    // it is possible that some PE is at pos1 after the switch is reconfigured and the sending
    // pattern is messed up, for example, the PE sends data first, then forwards the data from
    // the west.
    switch_config.clear_current_position(C_ROUTE);

    // invalidate pos1, pos2, pos3 but keep ring mode and pop mode
    switch_config.set_invalid_for_all_switch_positions(C_ROUTE);

    // WARNING: all PEs are configured by
    //   - ".pos1 = .{ .rx = RAMP }"  --> bit 3 is 1
    //   - ".ring_mode = true"  --> bit 12 is 1
    //   - ".pop_mode = .{ .pop_on_advance = true }" --> bits 13:12 of fabric per-color config
    if (last_px){
        // last PE does not have pos1
    }else{
        // others have ".pos1 = .{ .rx = RAMP }"
        switch_config.set_rx_switch_pos1(C_ROUTE, RAMP);
    }

    // (3) clear teardown-in-progress bit
    // config_reg[c] ^= mask where mask = 1 << 14
    tile_config.teardown.exit(C_ROUTE);
}

fn colReduce_configure() void {

    // (1) setup the new routing first because the pos1 needs to inherit from pos0
    const addr = color_config.get_color_config_addr(C_ROUTE);

    if (first_py){
        // 1st PE must has {rx = RAMP} to send out the data
        // .rx = .{ RAMP },.tx = .{ SOUTH },
        color_config.reset_routes(addr, .{.tx = SOUTH, .rx = RAMP});
    }else if (last_py){
        // last PE only receives data
        // .rx = .{ NORTH }, .tx = .{ RAMP },
        color_config.reset_routes(addr, .{.tx = RAMP, .rx = NORTH});
    }else{
        // 0 < py < width-1
        // .rx = .{ NORTH }, .tx = .{ SOUTH },
        color_config.reset_routes(addr, .{.tx = SOUTH, .rx = NORTH});
    }

    // (2) setup switch according to config parameters
    // 1. pos0 (color config reg)
    // 2. pos1 (switch config reg)
    //    pos1 = {.rx = RAMP} for all PEs except last PE

    // reset switch position to pos0
    // WARNING: if switch config register does not reset the switch position back to pos0,
    // it is possible that some PE is at pos1 after the switch is reconfigured and the sending
    // pattern is messed up, for example, the PE sends data first, then forwards the data from
    // the west.
    switch_config.clear_current_position(C_ROUTE);

    // invalidate pos1, pos2, pos3 but keep ring mode and pop mode
    switch_config.set_invalid_for_all_switch_positions(C_ROUTE);

    // WARNING: all PEs are configured by
    //   - ".pos1 = .{ .rx = RAMP }"  --> bit 3 is 1
    //   - ".ring_mode = true"  --> bit 12 is 1
    //   - ".pop_mode = .{ .pop_on_advance = true }" --> bits 13:12 of fabric per-color config
    if (last_py){
        // last PE does not have pos1
    }else{
        // others have ".pos1 = .{ .rx = RAMP }"
        switch_config.set_rx_switch_pos1(C_ROUTE, RAMP);
    }

    // (3) clear teardown-in-progress bit
    // config_reg[c] ^= mask where mask = 1 << 14
    tile_config.teardown.exit(C_ROUTE);
}

// w > 1 and h > 1
//  x <-- x <-- x
//              ^
//              |
//  x <-- x <-- x
//              ^
//              |
//  x <-- x <-- x
//
fn bcast_configure() void {

    // (1) setup the new routing first because the pos1 needs to inherit from pos0
    const addr = color_config.get_color_config_addr(C_ROUTE);

    if (last_px){
        // px = w-1
        if (last_py){
            // Pw-1,h-1: send to west and north, { .rx = .{RAMP}, .tx = .{WEST, NORTH} } }
            color_config.reset_routes(addr, .{.tx = [2]direction{WEST, NORTH}, .rx = RAMP});
        }else{
            if (first_py){
                // Pw-1,0: { .rx = .{SOUTH}, .tx = .{WEST, RAMP} }
                color_config.reset_routes(addr, .{.tx = [2]direction{WEST, RAMP}, .rx = SOUTH});
            }else{
                // Pw-1,py: 0 < py < h-1, { .rx = .{SOUTH}, .tx = .{WEST, RAMP, NORTH} }
                color_config.reset_routes(addr, .{.tx = [3]direction{WEST, RAMP, NORTH}, .rx = SOUTH});
            }
        }
    }else{
        if (first_px){
            // px = 0, {.rx = .{EAST}, .tx = .{RAMP}}
            color_config.reset_routes(addr, .{.tx = RAMP, .rx = EAST});
        }else{
            // 0 < px < w-1, { .rx = .{EAST}, .tx = .{WEST, RAMP} }
            color_config.reset_routes(addr, .{.tx = [2]direction{WEST, RAMP}, .rx = EAST});
        }
    }

    // (2) setup switch according to config parameters
    // 1. pos0 (color config reg)
    // 2. pos1 (switch config reg)
    //    pos1 = {invalid} for all PEs

    // reset switch position to pos0
    // WARNING: if switch config register does not reset the switch position back to pos0,
    // it is possible that some PE is at pos1 after the switch is reconfigured and the sending
    // pattern is messed up, for example, the PE sends data first, then forwards the data from
    // the west.
    switch_config.clear_current_position(C_ROUTE);

    // WARNING: all PEs have pos0 only, so disable pos1
    //   no change for ring_mode and pop_mode
    //   - ".ring_mode = true"  --> bit 12 is 1
    //   - ".pop_mode = .{ .pop_on_advance = true }" --> bits 13:12 of fabric per-color config
    // invalidate pos1, pos2, pos3 but keep ring mode and pop mode
    switch_config.set_invalid_for_all_switch_positions(C_ROUTE);

    // (3) clear teardown-in-progress bit
    // config_reg[c] ^= mask where mask = 1 << 14
    tile_config.teardown.exit(C_ROUTE);
}

// state 1: row-reduce
// state 2: col-reduce
// state 3: bcast
//
fn teardown_allreduce() void {
    // turn C_ROUTE back to teardown mode
    // LOCK can be picked only when the operation finishes
    @activate(C_LOCK);
}

comptime {
    @set_teardown_handler(teardown_allreduce, C_ROUTE);
}

//
// routing of C_ROUTE (send data to west, from leftmost)
//    -->   --->-->   -->-->
//    ^
//    |
//   sw_adv
//    -->      -->   -->-->
//    ^        ^
//    |        |
//   data     data
//            sw_adv
//    -->     --> -->     -->
//    ^                   ^
//    |                   |
//   sw_adv              data
//                      sw_adv
//    -->       -->    --> -->
//    ^         ^
//    |         |
//             data
//             sw_adv
//
comptime {

    // The switch must work for different operations, including
    //   - row reduction
    //   - column reduction
    //   - broadcasting
    // The initial setting must be universal so we can reconfigure the
    // switch for these three operations at runtime
    //
    // We need to set invariant parameters at comptime (runtime does not alter):
    // 1. teardown mode at comptime
    //   {.teardown = true} implies color is in teardown mode at comptime
    // 2. ring mode
    //   ".ring_mode = true"  --> fabric switch config reg sets bit 12 as 1
    // 3. pop on advance
    //   ".pop_mode = .{ .pop_on_advance = true }" --> fabric per-color config reg sets bits 13:12
    // 4. position 1
    //   ".pos1 = .{ .rx = RAMP }"  --> fabric switch config reg sets bit 3 as 1
    //
    // The following (last) PEs do not have position 1:
    //   - "px = width" for row reduction
    //   - "py = height" for column reduction
    //   - all for broadcasting
    // The runtime resets position 1 (bits 2:0 of fabric switch config) to either
    //   SWITCH_POS1_INVALID to disable position 1 or
    //   SWITCH_POS1_RAMP to reset position 1 back to ".pos1 = .{ .rx = RAMP }"
    // The bit 3 of fabric switch config is always 1 (position 1 switch select is "1=input")
    // If position 1 is disabled, bit 3 is don't care
    // If position 1 is disabled, pop mode is also don't care because of NOCE
    // If position 1 is disabled, ring mode is also don't care
    //
    // Remark: we don't use ".pop_mode = .{ .always_pop = true }" because there is no need
    // to propagate the TD to mux. All PEs have a teardown handler to deal with this TD, so
    // we do not need to pop out an instruction in TD wavelet, for example
    //     0x9df 9249 --> 0x91f 9249
    // (The instruction is NOT teardown 0b111, but 0b100 (NOCE, NOP))
    // (teardown = 0x9df,9249  (cmd1=teardown+NOCE, others=NOP+NOCE))
    //
    // The original setting of row reduction
    // 1st PE: px = 0
    //   .pos0 = .{ .rx = .{ RAMP }, .tx = .{ EAST }}
    //   .pop_mode = .{ .pop_on_advance = true }
    //   .pos1 = .{ .rx = RAMP }
    //   .ring_mode = true
    //   .teardown = true
    // middle: 1st PE < px < last PE
    //   .pos0 = .{ .rx = .{ WEST }, .tx = .{ EAST }}
    //   .pop_mode = .{ .pop_on_advance = true }
    //   .pos1 = .{ .rx = RAMP }
    //   .ring_mode = true
    //   .teardown = true
    // last PE: px = w-1
    //   .pos0 = .{ .rx = .{ WEST }, .tx = .{ RAMP }}
    //   .teardown = true
    //
    // The original setting of column reduction
    // 1st PE: py = 0
    //   .pos0 = .{ .rx = .{ RAMP }, .tx = .{ SOUTH }}
    //   .pop_mode = .{ .pop_on_advance = true }
    //   .pos1 = .{ .rx = RAMP }
    //   .ring_mode = true
    //   .teardown = true
    // middle: 1st PE < py < last PE
    //   .pos0 = .{ .rx = .{ NORTH }, .tx = .{ SOUTH }}
    //   .pop_mode = .{ .pop_on_advance = true }
    //   .pos1 = .{ .rx = RAMP }
    //   .ring_mode = true
    //   .teardown = true
    // last PE: py = h-1
    //   .pos0 = .{ .rx = .{ NORTH }, .tx = .{ RAMP }}
    //   .teardown = true
    //
    const universalConfig = .{
        .routes= .{
            .rx = .{ WEST },
            .tx = .{ EAST },
        },
        .switches=.{
            .pos1 = .{ .rx = RAMP },
            .ring_mode = true,
            .pop_mode = .{ .pop_on_advance = true },
        },
        .teardown = true
    };

    if (1 == width){
        @comptime_assert(1 < width);
    }else{
        @set_local_color_config(C_ROUTE, universalConfig);
    }
}

// binding a color to an input queue.
// This is necessary when an explicit DSR binds to a fabin DSD because
// the compiler no longer can generate the instruction to set up the
// config register of input queue.
comptime {
    @initialize_queue(iq_route, .{.color = C_ROUTE} );
}

comptime {
    // necessary, otherwise the data is not sent into the output queue
    if (@is_arch("wse3")){
        @initialize_queue(oq_route, .{.color = C_ROUTE });
    }
}

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc ./src/layout.csl --arch wse3 --fabric-dims=12,7 --fabric-offsets=4,1 \
--params=width:5,height:5,MAX_ZDIM:5 --params=BLOCK_SIZE:2 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py -m=5 -n=5 -k=5 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=5 --run-only

```
[previous

Bandwidth Test](csl/code-examples/benchmark-bandwidth-test.md "previous page")
[next

Power Method](csl/code-examples/benchmark-power-method.md "next page")
