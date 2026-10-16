---
title: "Host-to-Device Broadcast Test"
url: "https://sdk.cerebras.net/csl/code-examples/benchmark-row-col-broadcast"
crawled_at: "2026-02-14 23:55:11"
---

# Host-to-Device Broadcast Test

> Source: [https://sdk.cerebras.net/csl/code-examples/benchmark-row-col-broadcast](https://sdk.cerebras.net/csl/code-examples/benchmark-row-col-broadcast)

This example shows how to use row or column broadcast. For example if the user
wants to broadcast a column of data [1.0, 2.0, 3.0, 4.0] to a region of interest
starting from (1,1) with width 3 and height 4, one element per PE, the H2D API
requires the user to prepare the following 3-by-4 tensor,

```default
| 1.0  1.0  1.0 |
| 2.0  2.0  2.0 |
| 3.0  3.0  3.0 |
| 4.0  4.0  4.0 |

```

and use `memcpy_h2d()` API to stream 12 elements into the device. This
operation wastes host bandwidth by 3x.
Now the user can use the new API, `memcpy_h2d_rowbcast()`, to stream 4
elements only.

The same for column broadcasting, the user only needs to provide data of one
row and uses `memcpy_h2d_colbcast()` API.

The new broadcasting scheme only supports H2D, not D2H.

The kernel of `row-col-broadcast` is the same as `bandwidth-test`.
The `run.py` calculates the bandwidth as well.
The formula of the bandwidth calculation is the same as `bandwidth-test`,
so the user can see how much time this new API can save.

## layout.csl

```csl
// c0,c1,c2,c3,c4 are internal colors of sync module
param C0_ID: i16;
param C1_ID: i16;
param C2_ID: i16;
param C3_ID: i16;
param C4_ID: i16;

param pe_length: i16; // number of wavelets per PE
param width : i16 ;   // width of the core
param height: i16 ;   // height of the core

const C0 : color = @get_color(C0_ID);
const C1 : color = @get_color(C1_ID);
const C2 : color = @get_color(C2_ID);
const C3 : color = @get_color(C3_ID);
const C4 : color = @get_color(C4_ID);

// entrypoints of sync module
const STARTUP: local_task_id = @get_local_task_id(15);
const SYNC_Y: local_task_id = @get_local_task_id(16);
const SYNC_BCAST: local_task_id = @get_local_task_id(17);
const EXIT: local_task_id = @get_local_task_id(18);

const memcpy = @import_module( "<memcpy/get_params>", .{
    .width = width,
    .height = height,
    });

const sync = @import_module( "sync/layout.csl", .{
    .colors = [5]color{C0, C1, C2, C3, C4},
    .entrypoints = [4]local_task_id{STARTUP, SYNC_Y, SYNC_BCAST, EXIT},
    .width = width,
    .height = height
    });

layout{

    // H2D or D2H colors must be less than 15 (smallest color of entrypoints)
    @comptime_assert( C0_ID < C1_ID);
    @comptime_assert( C1_ID < C2_ID);
    @comptime_assert( C2_ID < C3_ID);
    @comptime_assert( C3_ID < C4_ID);

    // step 1: configure the rectangle which does not include halo
    @set_rectangle( width, height );

    // step 2: compile csl code for a set of PEx.y and generate out_x_y.elf
    //   format: @set_tile_code(x, y, code.csl, param_binding);

    var py: i16 = 0;
    while(py < height) : (py +=1) {
        var px: i16 = 0;
        while( px < width) : (px +=1) {

            const memcpyParams = memcpy.get_params(px);
            const syncParams = sync.get_params(px, py);

            var params: comptime_struct = .{
                .memcpyParams = memcpyParams,
                .pe_length = pe_length,

                .syncParams = syncParams,
            };

            @set_tile_code(px, py, "kernel.csl", params);
        }
    }

    @export_name("A", [*]f32, true);
    @export_name("time_memcpy", [*]f32, true);
    @export_name("time_ref", [*]f32, true);

    @export_name("f_tic", fn()void);
    @export_name("f_toc", fn()void);
    @export_name("f_memcpy_timestamps", fn()void);
    @export_name("f_sync", fn()void);
    @export_name("f_reference_timestamps", fn()void);
} // end of layout

```

## kernel.csl

```csl
// contraints: input/output queue ID = 0 is reserved for memcpy module
// only use microthread 2,3,4,5,6,7

param memcpyParams: comptime_struct;

param syncParams: comptime_struct;

param pe_length: i16;

const timestamp = @import_module("<time>");
// starting time of H2D/D2H
var tscStartBuffer = @zeros([timestamp.tsc_size_words]u16);
// ending time of H2D/D2H
var tscEndBuffer = @zeros([timestamp.tsc_size_words]u16);

const sys_mod = @import_module( "<memcpy/memcpy>", memcpyParams);

const sync_mod = @import_module( "sync/pe.csl", @concat_structs(syncParams, .{
     .f_callback = sys_mod.unblock_cmd_stream,
     .input_queues=[3]u16{2, 3, 4},
     .output_queues=[3]u16{2, 3, 4},
     }));

////////////////////////////////////////////////////////////////////////////////
// Main memory (48KB)
////////////////////////////////////////////////////////////////////////////////

const size : i16 = 1024*4;

var A = @zeros([size]f32);
// time_buf_f32[0:2] = {tscStartBuffer, tscEndBuffer}
var time_buf_f32 = @zeros([3]f32);
// reference clock inside sync module
var time_ref_f32 = @zeros([2]f32);

var ptr_A : [*]f32 = &A;
var ptr_time_memcpy: [*]f32 = &time_buf_f32;
var ptr_time_ref: [*]f32 = &time_ref_f32;

////////////////////////////////////////////////////////////////////////////////
// Tasks
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
    // time_buf_f32[0] = {tscStartBuffer[1], tscStartBuffer[0]}
    // time_buf_f32[1] = {tscEndBuffer[0], tscStartBuffer[2]}
    // time_buf_f32[2] = {tscEndBuffer[2], tscEndBuffer[1]}
    var lo_ : u16 = 0;
    var hi_ : u16 = 0;
    var word : u32 = 0;

    lo_ = tscStartBuffer[0];
    hi_ = tscStartBuffer[1];
    time_buf_f32[0] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

    lo_ = tscStartBuffer[2];
    hi_ = tscEndBuffer[0];
    time_buf_f32[1] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

    lo_ = tscEndBuffer[1];
    hi_ = tscEndBuffer[2];
    time_buf_f32[2] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

fn f_sync() void {
    // sync all PEs and record the reference clock
    sync_mod.f_sync();
}

fn f_reference_timestamps() void {
    // time_ref_f32[0] = {tscRefBuffer[1], tscRefBuffer[0]}
    // time_ref_f32[1] = {0, tscRefBuffer[2]}
    var lo_ : u16 = 0;
    var hi_ : u16 = 0;

    lo_ = sync_mod.tscRefBuffer[0];
    hi_ = sync_mod.tscRefBuffer[1];
    time_ref_f32[0] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

    lo_ = sync_mod.tscRefBuffer[2];
    hi_ = 0;
    time_ref_f32[1] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

comptime {

    @comptime_assert( pe_length <= size );
}

comptime {
    @export_symbol(ptr_A, "A");
    @export_symbol(ptr_time_memcpy, "time_memcpy");
    @export_symbol(ptr_time_ref, "time_ref");
}

comptime{
    @export_symbol(f_tic);
    @export_symbol(f_toc);
    @export_symbol(f_memcpy_timestamps);
    @export_symbol(f_sync);
    @export_symbol(f_reference_timestamps);
}

```

## run.py

```python
#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" Test row or column broadcast
    The kernel is the same as bandwidthTest.
    The bandwidth calculation follows bandwidthTest.

    Here is the list of parameters:
    -m=<int> specifies the height of the core.
    -n=<int> specifies the width of the core.
    -k=<int> specifies the maximum number of elements per PE in the core.
    --roi_px=<int> specifies the starting column index of region of interest
    --roi_py=<int> specifies the starting row index of region of interest
    --roi_w=<int> specifies the width of region of interest
    --roi_h=<int> specifies the height of region of interest
    --channels specifies the number of I/O channels, no bigger than 16.
"""

import random
import struct

import numpy as np
from cmd_parser import parse_args

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

def float_to_hex(f):
  return hex(struct.unpack("<I", struct.pack("<f", f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def main():
  """Main method to run the example code."""

  random.seed(127)

  args, dirname = parse_args()

  height = args.m
  width = args.n
  pe_length = args.k
  use_col_major = args.use_col_major
  is_row_bcast = args.is_row_bcast
  loop_count = args.loop_count

  print(f"core: width = {width}, height = {height}, pe_length={pe_length}")

  np.random.seed(2)
  if is_row_bcast:
    print("row broadcast mode: only prepare data for 1 column")
    # A is h-by-1-by-l
    A = (np.arange(height * 1 * pe_length).reshape(height, 1, pe_length).astype(np.uint32))
  else:
    print("column broadcast mode: only prepare data for 1 row")
    # A is 1-by-w-by-l
    A = (np.arange(1 * width * pe_length).reshape(1, width, pe_length).astype(np.uint32))
  print(f"shape(A) = {A.shape}")
  print(f"A = {A}")

  px = args.roi_px
  py = args.roi_py
  pw = args.roi_w
  ph = args.roi_h

  print(f"ROI: px = {px}, py = {py}, pw = {pw}, ph = {ph}")

  assert px >= 0, "px must be non-negative"
  assert py >= 0, "px must be non-negative"
  assert pw <= width, "pw must not be greater than width"
  assert ph <= height, "ph must not be greater than height"

  # extract ROI from A
  if is_row_bcast:
    B = A[py:(py + ph), 0:, 0:]
  else:
    B = A[0:, px:(px + pw), 0:]
  print(f"shape(B) = {B.shape}")
  print(f"B = {B}")

  bx, by, bz = B.shape
  if is_row_bcast:
    assert bx == ph
    assert by == 1
    assert bz == pe_length
  else:
    assert bx == 1
    assert by == pw
    assert bz == pe_length

  print(f"use_col_major = {use_col_major}")
  if use_col_major:
    B_1d = B.T.ravel()
  else:
    B_1d = B.ravel()

  print("store ELFs and log files in the folder ", dirname)

  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT

  runner = SdkRuntime(
      dirname,
      suppress_simfab_trace=True,
      # msg_level="DEBUG",
      cmaddr=args.cmaddr,
  )

  symbol_A = runner.get_id("A")
  symbol_time_memcpy = runner.get_id("time_memcpy")
  symbol_time_ref = runner.get_id("time_ref")

  runner.load()
  runner.run()

  print("step 1: sync() synchronizes all PEs and records reference clock")
  runner.call("f_sync", [], nonblock=True)

  print("step 2: tic() records time_start")
  runner.call("f_tic", [], nonblock=True)

  print(f"len(B_1d) = {len(B_1d)}")
  print(f"B_1d = {B_1d}")
  for _ in range(loop_count):
    if is_row_bcast:
      print("step 1: memcpy_h2d_rowbcast(B)")
      runner.memcpy_h2d_rowbcast(
          symbol_A,
          B_1d,
          px,
          py,
          pw,
          ph,
          pe_length,
          streaming=False,
          data_type=memcpy_dtype,
          order=(MemcpyOrder.COL_MAJOR if use_col_major else MemcpyOrder.ROW_MAJOR),
          nonblock=True,
      )
    else:
      print("step 1: memcpy_h2d_colbcast(B)")
      runner.memcpy_h2d_colbcast(
          symbol_A,
          B_1d,
          px,
          py,
          pw,
          ph,
          pe_length,
          streaming=False,
          data_type=memcpy_dtype,
          order=(MemcpyOrder.COL_MAJOR if use_col_major else MemcpyOrder.ROW_MAJOR),
          nonblock=True,
      )

  print("step 4: toc() records time_end")
  runner.call("f_toc", [], nonblock=False)

  print("step 5: prepare (time_start, time_end)")
  runner.call("f_memcpy_timestamps", [], nonblock=False)

  print("step 6: D2H (time_start, time_end)")
  # time_start/time_end is of type u16[3]
  # {time_start, time_end} is packed into three f32
  time_memcpy_1d_f32 = np.zeros(height * width * 3, np.float32)
  runner.memcpy_d2h(
      time_memcpy_1d_f32,
      symbol_time_memcpy,
      0,
      0,
      width,
      height,
      3,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.ROW_MAJOR,
      nonblock=False,
  )
  time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (height, width, 3), order="C")

  print("step 7: prepare reference clock")
  runner.call("f_reference_timestamps", [], nonblock=False)

  print("step 8: D2H reference clock")
  # time_ref is of type u16[3], packed into two f32
  time_ref_1d_f32 = np.zeros(height * width * 2, np.float32)
  runner.memcpy_d2h(
      time_ref_1d_f32,
      symbol_time_ref,
      0,
      0,
      width,
      height,
      2,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.ROW_MAJOR,
      nonblock=False,
  )
  time_ref_hwl = np.reshape(time_ref_1d_f32, (height, width, 2), order="C")

  print("step 9: D2H(A)")
  E_1d = np.zeros(height * width * pe_length, A.dtype)
  runner.memcpy_d2h(
      E_1d,
      symbol_A,
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

  runner.stop()

  print("DONE")

  # E is h-by-w-by-l
  E_hwl = np.reshape(E_1d, (height, width, pe_length), order="F")
  print(f"E_hwl (from device) = {E_hwl}")

  # B_ext is the expected result
  B_ext = (np.zeros(height * width * pe_length).reshape(height, width, pe_length).astype(A.dtype))
  if is_row_bcast:
    # copy B to each column of ROI
    for w in range(pw):
      B_ext[py:(py + ph), (px + w):(px + w + 1), 0:] = B
  else:
    # copy B to each row of ROI
    for h in range(ph):
      B_ext[(py + h):(py + h + 1), px:(px + pw), 0:] = B
  print(f"B_ext = {B_ext}")

  print("check E_hwl == B_ext")
  assert np.allclose(E_hwl.ravel(), B_ext.ravel(), 0)

  # time_start = start time of H2D/D2H
  time_start = np.zeros((height, width)).astype(int)
  # time_end = end time of H2D/D2H
  time_end = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      hex_t0 = int(float_to_hex(time_memcpy_hwl[(h, w, 0)]), base=16)
      hex_t1 = int(float_to_hex(time_memcpy_hwl[(h, w, 1)]), base=16)
      hex_t2 = int(float_to_hex(time_memcpy_hwl[(h, w, 2)]), base=16)
      word[0] = hex_t0 & 0x0000FFFF
      word[1] = (hex_t0 >> 16) & 0x0000FFFF
      word[2] = hex_t1 & 0x0000FFFF
      time_start[(h, w)] = make_u48(word)
      word[0] = (hex_t1 >> 16) & 0x0000FFFF
      word[1] = hex_t2 & 0x0000FFFF
      word[2] = (hex_t2 >> 16) & 0x0000FFFF
      time_end[(h, w)] = make_u48(word)

  # time_ref = reference clock
  time_ref = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
      hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
      word[0] = hex_t0 & 0x0000FFFF
      word[1] = (hex_t0 >> 16) & 0x0000FFFF
      word[2] = hex_t1 & 0x0000FFFF
      time_ref[(h, w)] = make_u48(word)
  # adjust the reference clock by the propagation delay
  for py in range(height):
    for px in range(width):
      time_ref[(py, px)] = time_ref[(py, px)] - (px + py)

  # shift time_start and time_end by time_ref
  time_start = time_start - time_ref
  time_end = time_end - time_ref

  # cycles_send = time_end[(h,w)] - time_start[(h,w)]
  # 850MHz --> 1 cycle = (1/0.85) ns = (1/0.85)*1.e-3 us
  # time_send = (cycles_send / 0.85) *1.e-3 us
  # bandwidth = (((wvlts-1) * 4)/time_send) MBS
  wvlts = pw * ph * pe_length
  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) * 1.0e-3
  bandwidth = ((wvlts * 4) / time_send) * loop_count
  print(f"ROI: pw = {pw}, ph= {ph}, pe_length={pe_length}")
  print(f"wvlts = {wvlts}, loop_count = {loop_count}")
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")
  print(f"bandwidth = {bandwidth} MB/S ")

if __name__ == "__main__":
  main()

```

## \_cmd\_parser.py

```python
"""command parser for broadcast

   -m <int>      number of rows of the core rectangle
   -n <int>      number of columns of the core rectangle
   -k <int>      number of elements of local tensor
   --latestlink  working directory
   --cmaddr      IP address of a WSE
   --roi_px      starting column index of region of interest
   --roi_py      starting row index of region of interest
   --roi_w       width of region of interest
   --roi_h       height of region of interest
"""

import argparse
import os

def parse_args():
  """command parser"""
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", default=1, type=int, help="number of rows")
  parser.add_argument("-n", default=1, type=int, help="number of columns")
  parser.add_argument("-k", default=1, type=int, help="size of local tensor")
  parser.add_argument("--latestlink", help="folder to contain the log files (default: latest)")
  parser.add_argument("--cmaddr", help="CM address and port, i.e. <IP>:<port>")
  parser.add_argument("--arch", help="wse2 or wse3. Default is wse2 when not supplied.")
  parser.add_argument("--channels", default=1, type=int, help="number of channels")
  parser.add_argument("--roi_px", default=1, type=int, help="starting column index of ROI")
  parser.add_argument("--roi_py", default=1, type=int, help="starting row index of ROI")
  parser.add_argument("--roi_w", default=3, type=int, help="width of ROI")
  parser.add_argument("--roi_h", default=3, type=int, help="height of ROI")
  parser.add_argument(
      "--use_col_major",
      action="store_true",
      help="use column major to send the row or column broadcast",
  )
  parser.add_argument(
      "--is_row_bcast",
      action="store_true",
      help="row broadcast or column broadcast",
  )
  parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")
  parser.add_argument(
      "--loop_count",
      default=1,
      type=int,
      help="number of back-to-back H2D/D2H",
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

  return args, logs_dir

```

## sync/layout.csl

```csl
param colors:[5]color;
param entrypoints:[4]local_task_id;
param width : i16 ;   // width of the core
param height: i16 ;   // height of the core

const C0 : color = colors[0];
const C1 : color = colors[1];
const C2 : color = colors[2];
const C3 : color = colors[3];
const C4 : color = colors[4];

const STARTUP: local_task_id = entrypoints[0];
const SYNC_Y: local_task_id = entrypoints[1];
const SYNC_BCAST: local_task_id = entrypoints[2];
const EXIT: local_task_id = entrypoints[3];

fn get_params(px:i16, py:i16) comptime_struct {

    var first_py: bool = (0 == py);
    var last_py: bool = ((height-1) == py);
    var is_py_even: bool = (0 == (py % 2));

    var first_px: bool = (0 == px);
    var last_px: bool = ((width-1) == px);
    var is_px_even: bool = (0 == (px % 2));

    var c_recv_px: color = C0;
    var c_send_px: color = C1;
    if (is_px_even){
        c_recv_px = C0;
        c_send_px = C1;
    }else{
        c_recv_px = C1;
        c_send_px = C0;
    }

    var c_recv_py: color = C2;
    var c_send_py: color = C3;
    if (is_py_even){
        c_recv_py = C2;
        c_send_py = C3;
    }else{
        c_recv_py = C3;
        c_send_py = C2;
    }

    return .{
        .c_recv_px = c_recv_px,
        .c_send_px = c_send_px,
        .c_recv_py = c_recv_py,
        .c_send_py = c_send_py,
        .c_bcast = C4,

        .STARTUP = STARTUP,
        .SYNC_Y = SYNC_Y,
        .SYNC_BCAST = SYNC_BCAST,
        .EXIT = EXIT,

        .first_px = first_px,
        .last_px = last_px,
        .first_py = first_py,
        .last_py = last_py,
    };
}

```

## sync/pe.csl

```csl
param c_recv_px: color;
param c_send_px: color;
param c_recv_py: color;
param c_send_py: color;
param c_bcast: color;

param STARTUP: local_task_id;
param SYNC_Y: local_task_id;
param SYNC_BCAST: local_task_id;
param EXIT: local_task_id;

param first_px: bool;
param last_px: bool;
param first_py: bool;
param last_py: bool;

// f_callback = sys_mod.unblock_cmd_stream, to continue next command
param f_callback : fn ()void;

// input_queues={2,3,4}
// output_queues={2,3,4}
param input_queues:[3]u16;
param output_queues:[3]u16;

const c_recv_px_iq = @get_input_queue(input_queues[0]);
const c_send_px_oq = @get_output_queue(output_queues[0]);

const c_recv_py_iq = @get_input_queue(input_queues[1]);
const c_send_py_oq = @get_output_queue(output_queues[1]);

const c_bcast_iq = @get_input_queue(input_queues[2]);
const c_bcast_oq = @get_output_queue(input_queues[2]);

const timestamp = @import_module("<time>");

// tsc_size_words = 3
var tscRefBuffer = @zeros([timestamp.tsc_size_words]u16);

////////////////////////////////////////////////////////////////////////////////
// Main memory (48KB)
////////////////////////////////////////////////////////////////////////////////

var buf = @zeros([1]f32);

////////////////////////////////////////////////////////////////////////////////
// Tasks
// syntax
//     task_begin(name, entrypoint, color)
////////////////////////////////////////////////////////////////////////////////

const mem_buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> buf[i] });

var fab_recv_data_px_wdsd =  @get_dsd(fabin_dsd, .{
   .extent = 1,
   .fabric_color = c_recv_px,
   .input_queue = c_recv_px_iq
});

var fab_trans_data_px_wdsd = @get_dsd(fabout_dsd, .{
    .extent = 1,
    .fabric_color = c_send_px,
    .output_queue = c_send_px_oq
});

var fab_recv_data_py_wdsd =  @get_dsd(fabin_dsd, .{
   .extent = 1,
   .fabric_color = c_recv_py,
   .input_queue = c_recv_py_iq
});

var fab_trans_data_py_wdsd = @get_dsd(fabout_dsd, .{
    .extent = 1,
    .fabric_color = c_send_py,
    .output_queue = c_send_py_oq
});

var fab_recv_data_bcast_wdsd =  @get_dsd(fabin_dsd, .{
   .extent = 1,
   .fabric_color = c_bcast,
   .input_queue = c_bcast_iq
});

var fab_trans_data_bcast_wdsd = @get_dsd(fabout_dsd, .{
    .extent = 1,
    .fabric_color = c_bcast,
    .output_queue = c_bcast_oq
});

// Each row performs a sync from the last PE to first PE
fn f_sync() void {
    // sync a row
    if (last_px){
        // px = width-1: send sync signal
        @mov32(fab_trans_data_px_wdsd, mem_buf_dsd, .{.async=true, .activate = f_sync_y });
    }else{
        if (first_px){
            // px = 0: receive signal
            @mov32(mem_buf_dsd, fab_recv_data_px_wdsd, .{.async=true, .activate = f_sync_y });
        }else{
            // 0 < px < width-1: receive signal and forward it
            @mov32(fab_trans_data_px_wdsd, fab_recv_data_px_wdsd, .{.async=true, .activate = f_sync_y });
        }
    }
}

// prerequisite: row synchronization is done
//   the first PE is the last one to receive the signal
// The first column performs a sync from last PE to first PE
// other PEs wait for bcast signal
task f_sync_y() void {
    if (first_px){
        // 1st column performs a sync
        if (last_py){
            // py = height-1: send sync signal
            @mov32(fab_trans_data_py_wdsd, mem_buf_dsd, .{.async=true, .activate = f_sync_bcast });
        }else{
            if (first_py){
                // py = 0: receive signal
                @mov32(mem_buf_dsd, fab_recv_data_py_wdsd, .{.async=true, .activate = f_sync_bcast });
            }else{
                // 0 < py < height-1: receive signal and forward it
                @mov32(fab_trans_data_py_wdsd, fab_recv_data_py_wdsd, .{.async=true, .activate = f_sync_bcast });
            }
        }
    }else{
        // other PEs wait for bcast signal
        @activate(SYNC_BCAST); // trigger f_sync_bcast
    }
}

// prerequisite: sync is done, P0.0 is the last one to receive the sync
// P0.0 broadcasts the signal, others wait for the bcast signal from P0.0
task f_sync_bcast() void {

    if ( first_px and first_py ){
        // P0.0 sends the signal
        @mov32(fab_trans_data_bcast_wdsd, mem_buf_dsd, .{.async=true, .activate = f_exit });
    }else{
        // others wait for bcast from P0.0
        @mov32(mem_buf_dsd, fab_recv_data_bcast_wdsd, .{.async=true, .activate = f_exit });
    }
}

// record reference clock T
// T is regarded as clock 0 because all PEs sync with P0.0
task f_exit() void {

    timestamp.get_timestamp(&tscRefBuffer);

    //sys_mod.unblock_cmd_stream();
    f_callback();
}

task f_startup() void {
    timestamp.enable_tsc();
}

comptime {
    @activate(STARTUP);

    @bind_local_task(f_startup, STARTUP);
    @bind_local_task(f_sync_y, SYNC_Y);
    @bind_local_task(f_sync_bcast, SYNC_BCAST);
    @bind_local_task(f_exit, EXIT);

    // On WSE-3, we must explicitly initialize input and output queues
    if (@is_arch("wse3")) {
        @initialize_queue(c_recv_px_iq, .{ .color = c_recv_px });
        @initialize_queue(c_send_px_oq, .{ .color = c_send_px });

        @initialize_queue(c_recv_py_iq, .{ .color = c_recv_py });
        @initialize_queue(c_send_py_oq, .{ .color = c_send_py });

        @initialize_queue(c_bcast_iq, .{ .color = c_bcast });
        @initialize_queue(c_bcast_oq, .{ .color = c_bcast });
    }
}

// sync a row with C0 and C1
//
//     C0     C1     C0     C1
// P0 <-- P1 <-- P2 <-- P3 <-- P4
//
//     C0     C1     C0     C1     C0
// P0 <-- P1 <-- P2 <-- P3 <-- P4 <-- P5
//
// P0: recv C0
// P_even: recv C0, send C1
// P_odd: recv C1, send C0
// P_last: send C0 if odd; send C1 if even
comptime {
    if (first_px){
        // px = 0: receive from east
        @set_local_color_config(c_recv_px, .{ .routes = .{ .rx = .{EAST}, .tx = .{RAMP} } } );
    }else{
        if (last_px){
           // px = width-1: send to west
           @set_local_color_config(c_send_px, .{ .routes = .{ .rx = .{RAMP}, .tx = .{WEST} } } );
        }else{
           // 0 < px < width-1: receive from east, send to west
           @set_local_color_config(c_recv_px, .{ .routes = .{ .rx = .{EAST}, .tx = .{RAMP} } } );
           @set_local_color_config(c_send_px, .{ .routes = .{ .rx = .{RAMP}, .tx = .{WEST} } } );
        }
    }
}

// sync a col with C2 and C3
//     C2     C3     C2     C3
// P0 <-- P1 <-- P2 <-- P3 <-- P4
//
//     C2     C3     C2     C3     C2
// P0 <-- P1 <-- P2 <-- P3 <-- P4 <-- P5
//
// P0: recv C2
// P_even: recv C2, send C3
// P_odd: recv C3, send C2
// P_last: send C2 if odd; send C3 if even
comptime {
    if (first_py){
        // py = 0 (even): receive from south
        @set_local_color_config(c_recv_py, .{ .routes = .{ .rx = .{SOUTH}, .tx = .{RAMP} } } );
    }else{
        if (last_py){
           // py = height-1: send to north
           @set_local_color_config(c_send_py, .{ .routes = .{ .rx = .{RAMP}, .tx = .{NORTH} } } );
        }else{
           // 0 < py < height-1: receive from south, send to north
           @set_local_color_config(c_recv_py, .{ .routes = .{ .rx = .{SOUTH}, .tx = .{RAMP} } } );
           @set_local_color_config(c_send_py, .{ .routes = .{ .rx = .{RAMP}, .tx = .{NORTH} } } );
        }
    }
}

// w > 1 and h > 1
//  x --> x --> x
//  |
//  V
//  x --> x --> x
//  |
//  V
//  x --> x --> x
//
// WARNING: corner case for w=1 or h=1
comptime {
    if (first_px){
        // px = 0
        if (first_py){
            // P0,0: send to east and south
            @set_local_color_config(c_bcast, .{ .routes = .{ .rx = .{RAMP}, .tx = .{EAST, SOUTH} } } );
        }else{
            if (last_py){
                // P0,h-1
                @set_local_color_config(c_bcast, .{ .routes = .{ .rx = .{NORTH}, .tx = .{EAST, RAMP} } } );
            }else{
                // P0,py: 0 < py < height-1
                @set_local_color_config(c_bcast, .{ .routes = .{ .rx = .{NORTH}, .tx = .{EAST, RAMP, SOUTH} } } );
            }
        }
    }else{
        if (last_px){
            // px = width-1
           @set_local_color_config(c_bcast, .{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP} } } );
        }else{
            // 0 < px < width-1
           @set_local_color_config(c_bcast, .{ .routes = .{ .rx = .{WEST}, .tx = .{EAST, RAMP} } } );
        }
    }
}

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc ./src/layout.csl --arch wse3 --fabric-dims=12,7 --fabric-offsets=4,1 \
--params=width:5,height:5,pe_length:5 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 -o=out \
--memcpy --channels=2 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py -m=5 -n=5 -k=5 --latestlink out --is_row_bcast --loop_count=1

```
[previous

Single Tile Matvec](csl/code-examples/benchmark-single-tile-matvec.md "previous page")
[next

3D FFT](csl/code-examples/benchmark-fft-3d.md "next page")
