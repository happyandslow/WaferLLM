---
title: "Bandwidth Test"
url: "https://sdk.cerebras.net/csl/code-examples/benchmark-bandwidth-test"
crawled_at: "2026-02-14 23:54:56"
---

# Bandwidth Test

> Source: [https://sdk.cerebras.net/csl/code-examples/benchmark-bandwidth-test](https://sdk.cerebras.net/csl/code-examples/benchmark-bandwidth-test)

This example evaluates the bandwidth between the host and the device (WSE). The
kernel records the `start` and `end` of H2D or D2H by tsc counter. This is
better than host timer because the runtime may not send the command right after
the user issues it. The runtime can aggregate multiple nonblocking commands
together to reduce TCP overhead. In addition the tsc counters of all PEs are
not sychronized in the beginning. To avoid the timing variation among those PEs
, we add a sync() to synchronize all PEs and sample the reference clock.

The kernel `bw_sync_kernel.csl` defines a couple of host-callable functions,
`f_sync()`, `f_tic()` and `f_toc()` in order to synchronize the PEs and
record the timing of H2D or D2H.

The kernel `sync/pe.csl` performs a reduction over the whole rectangle to sync
the PEs, then the top-left PE sends a signal to other PEs to sample the
reference clock.

The script `run.py` has the following parameters:

- `--loop_count=<int>` decides how many H2Ds/D2Hs are called.
- `--d2h` measures the bandwidth of D2H, otherwise H2D is measured.
- `--channels=<int>` specifies the number of I/O channels, no bigger than 16.

The tic() samples “time\_start” and toc() samples “time\_end”. The sync() samples
“time\_ref” which is used to adjust “time\_start” and “time\_end”.
The elapsed time (unit: cycles) is measured by
`cycles_send = max(time_end) - min(time_start)`

The overall runtime (us) is computed via the following formula
`time_send = (cycles_send / 0.85) * 1.e-3 us`

The bandwidth is calculated by
`bandwidth = ((wvlts * 4)/time_send)*loop_count`

## bw\_sync\_layout.csl

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

            @set_tile_code(px, py, "bw_sync_kernel.csl", params);
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

## bw\_sync\_kernel.csl

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
""" test bandwidth between host and device

    The host connects the device via 100Gbps ethernets. The data is distributed
  from/to couple of I/O channels. The maximum bandwidth of a single channel is
  around 7Gbps (Giga bit per second). In addition, the overhead of TCP is about
  200 us, a non-negligible cost when the transaction is small.

  The bandwidth is affected by the following factors:
  (1) number of I/O channels
      The number of I/O channels is controlled by the flag --channels=<int>
      The more channels, the higher bandwidth
  (2) buffers to hold input/output data to hide the long latency of I/O
      Although The I/O channelsand the core are independent, if the core has a
      heavy computation such that it cannot respond to the I/O request, there is
      a backpressure from the core upstream to the I/O channels. The backpressure
      stalls the data transfer and the host can no longer push the data.
      I/O channel will resume only when the core responds the request,however
      there is a long latency before the core can receive the data.
      To overlap the computaton and communication (or to avoid this long latency)
      , we can insert buffers to hold the data from the I/O channels while the
      core is busy for something else.
      The user can use flag --width-west-buf=<int> to set a buffer for the input
      and the flag --width-east-buf to set a buffer for the output.
      Each PE in the buffer has 46KB fifo to store the data, if a H2D/D2H has
      "pe_length" u32 per PE and "width" PEs per row, it needs
      (pe_length*width)*4/46K columns
  (3) blocking (sync) or nonblocking (async)
      The long latency of I/O can be amortized if multiple requests are combined
      together into one TCP transfer (200 us overhead per TCP transaction). The
      runtime can aggregate multiple nonblocking H2D/D2H commands implicitly.
      The user can set paramerer 'nonblock=True' to enable async operations.

  The framework of bandwidthTest is
  ---
       sync   // synchronize all PEs to sample the reference clock
       tic()  // record start time
       for j = 0:loop_count
          H2D or D2H (sync or async)
       end
       toc()  // record end time
  ---

  To record the elapsed time on host may not show the true timing because the
  runtime may not start the transaction when the user calls a H2D/D2H command.
  For example, the runtime can aggregate multiple nonblocking commands together.
  Instead, this bandwidhTest samples the timing on the device.

  The strategy is to record "start" time and "end" time of H2D/D2H on each PE and
  to compute the elapsed time by the different of max/min of these two numbers.
  However the tsc timer is not synchronized and could differ a lot if we take max
  or min operation on the timer. To obtain the reliable timing info, we need to
  synchronize all PEs and use one PE to trigger the timer such that all PEs can
  start at "the same" time. The "sync" operation can sample the reference clock
  which is the initial time t0 for all PEs.
  Even we shift the "start clock" by the "reference clock", each PE does not have
  the same number because of the propagation delay of the signal. The delay of
  "start clock" is about the dimension of the WSE.

  Here is the list of parameters:
    The flag --loop_count=<int> decides how many H2Ds/D2Hs are called.
    The flag --d2h measures the bandwidth of D2H, otherwise bandwidth of H2D is
        measured.
    The flag --channels specifies the number of I/O channels, no bigger than 16.

  The tic() samples "time_start" and toc() samples "time_end". The sync() samples
  "time_ref" which is used to shift "time_start" and "time_end".
  The elapsed time is measured by
       cycles_send = max(time_end) - min(time_start)

  The overall runtime is computed via the following formula
       time_send = (cycles_send / 0.85) *1.e-3 us
  where a PE runs with clock speed 850MHz

  The bandwidth is calculated by
       bandwidth = ((wvlts * 4)/time_send)*loop_count
"""

import random
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from bw_cmd_parser import parse_args

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

def float_to_hex(f):
  return hex(struct.unpack("<I", struct.pack("<f", f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def cast_uint32(x):
  if isinstance(x, (np.float16, np.int16, np.uint16)):
    z = x.view(np.uint16)
    val = np.uint32(z)
  elif isinstance(x, (np.float32, np.int32, np.uint32)):
    val = x.view(np.uint32)
  elif isinstance(x, int):
    val = np.uint32(x)
  elif isinstance(x, float):
    z = np.float32(x)
    val = z.view(np.uint32)
  else:
    raise RuntimeError(f"type of x {type(x)} is not supported")

  return val

def csl_compile_core(
    cslc: str,
    width: int,  # width of the core
    height: int,  # height of the core
    pe_length: int,
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
    args.append(f"--params=width:{width},height:{height},pe_length:{pe_length}")
    args.append(f"--params=C0_ID:{C0}")
    args.append(f"--params=C1_ID:{C1}")
    args.append(f"--params=C2_ID:{C2}")
    args.append(f"--params=C3_ID:{C3}")
    args.append(f"--params=C4_ID:{C4}")

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

def hwl_2_oned_colmajor(height: int, width: int, pe_length: int, A_hwl: np.ndarray):
  """
    Given a 3-D tensor A[height][width][pe_length], transform it to
    1D array by column-major
    """
  A_1d = np.zeros(height * width * pe_length, np.float32)
  idx = 0
  for l in range(pe_length):
    for w in range(width):
      for h in range(height):
        A_1d[idx] = A_hwl[(h, w, l)]
        idx = idx + 1
  return A_1d

# How to compile:
#  <path/to/cslc> src/bw_sync_layout.csl --fabric-dims=12,7 --fabric-offsets=4,1 \
#    --params=width:5,height:5,pe_length:5 \
#    --params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 \
#    --params=C3_ID:3 --params=C4_ID:4 \
#    -o=latest --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
# or
#  python run.py -m=5 -n=5 -k=5 --latestlink latest --channels=1 \
#    --width-west-buf=0 --width-east-buf=0 \
#    --compile-only --driver=<path/to/cslc>
#
# How to run:
#  python run.py -m=5 -n=5 -k=5 --latestlink latest --channels=1 \
#   --width-west-buf=0 --width-east-buf=0 \
#   --run-only --loop_count=1
#
# To run a WSE, add --cmaddr=<IP address of WSE>
#
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
  loop_count = args.loop_count

  print(f"width = {width}, height = {height}, pe_length={pe_length}, loop_count = {loop_count}")

  np.random.seed(2)
  # A is h-by-w-by-l
  A = (np.arange(height * width * pe_length).reshape(height, width, pe_length).astype(np.float32))

  A_1d = hwl_2_oned_colmajor(height, width, pe_length, A)

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

  # core dump after execution is complete
  # core_path = os.path.join(dirname, "core.out")

  # layout of a rectangle
  code_csl = "src/bw_sync_layout.csl"

  C0 = 0
  C1 = 1
  C2 = 2
  C3 = 3
  C4 = 4

  csl_compile_core(
      cslc,
      width,
      height,
      pe_length,
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
      channels,
      width_west_buf,
      width_east_buf,
  )
  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  # output tensor via D2H
  E_1d = np.zeros(height * width * pe_length, np.float32)

  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

  symbol_A = runner.get_id("A")
  symbol_time_memcpy = runner.get_id("time_memcpy")
  symbol_time_ref = runner.get_id("time_ref")

  runner.load()
  runner.run()

  print("step 1: sync() synchronizes all PEs and records reference clock")
  runner.call("f_sync", [], nonblock=True)

  print("step 2: tic() records time_start")
  runner.call("f_tic", [], nonblock=True)

  if args.d2h:
    for j in range(loop_count):
      print(f"step 3: measure D2H with loop_count = {loop_count}, {j}-th")
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
  else:
    for j in range(loop_count):
      print(f"step 3: measure H2D with loop_count = {loop_count}, {j}-th")
      runner.memcpy_h2d(
          symbol_A,
          A_1d,
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

  # runner.stop(core_path)
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
  wvlts = height * width * pe_length
  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) * 1.0e-3
  bandwidth = ((wvlts * 4) / time_send) * loop_count
  print(f"wvlts = {wvlts}, loop_count = {loop_count}")
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")
  print(f"bandwidth = {bandwidth} MB/S ")

if __name__ == "__main__":
  main()

```

## bw\_cmd\_parser.py

```python
"""command parser for bandwidthTest

   -m <int>     number of rows of the core rectangle
   -n <int>     number of columns of the core rectangle
   -k <int>     number of elements of local tensor
   --latestlink   working directory
   --driver     path to CSL compiler
   --fabric-dims  fabric dimension of a WSE
   --cmaddr       IP address of a WSE
   --channels        number of I/O channels, 1 <= channels <= 16
   --width-west-buf  number of columns of the buffer in the west of the core rectangle
   --width-east-buf  number of columns of the buffer in the east of the core rectangle
   --loop_count      number of H2D/D2H to be measured
   --d2h             measure bandwidth of D2H (default: H2D)
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
  parser.add_argument("-k", default=1, type=int, help="size of local tensor")
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
  parser.add_argument("--d2h", help="measure D2H", action="store_true")
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

  if args.cmaddr is None:
    args.simulator = False

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

cslc ./src/bw_sync_layout.csl --arch wse3 --fabric-dims=12,7 --fabric-offsets=4,1 \
--params=width:5,height:5,pe_length:5 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py -m=5 -n=5 -k=5 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --run-only --loop_count=1

```
[previous

25-Point Stencil](csl/code-examples/benchmark-25-pt-stencil.md "previous page")
[next

3D 7-Point Stencil SpMV](csl/code-examples/benchmark-7pt-stencil-spmv.md "next page")
