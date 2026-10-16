---
title: "Single Tile Matvec"
url: "https://sdk.cerebras.net/csl/code-examples/benchmark-single-tile-matvec"
crawled_at: "2026-02-14 23:55:42"
---

# Single Tile Matvec

> Source: [https://sdk.cerebras.net/csl/code-examples/benchmark-single-tile-matvec](https://sdk.cerebras.net/csl/code-examples/benchmark-single-tile-matvec)

This program performs single tile matrix-vector products y = A\*x,
where A has dimension N x N, and the data type is `fp32`.
This program produces memory bandwidth information and FLOPS.

To compile and run for N=10 to N=100 on an actual CS-2, use:
`./sweep.py --dims 750,994 --cmaddr <CS IP ADDR>`

Dims here refers to the dimensions of the program rectangle.
The program above will perform 750\*994 matvecs, with one matvec
occurring on each PE, for each value of N.

There is also an `iters` flag, which allows you to average
cycle counts over multiple runs. Here is an example for a
10 x 10 program rectangle run in the simulator:
`./sweep.py --dims 10,10 --iters 10`.

You can also compile and run separately, and also run in a verificaiton
mode that verifies the result on each PE.

To compile:

```default
cslc layout_matvec.csl --fabric-dims=17,12 --fabric-offsets=4,1 \
--params=width:10,height:10,tile_size:25,iters:1 -o out --memcpy --channels=1

```

where `fabric-dims` is the dimension of the simfabric, `width`, `height`
are the dimensions of the program rectangle, `tile_size` is N,
and `iters` is the number of iterations over which we average.

Note that the width must be no bigger than 7 less than the x-dimension of the
fabric, and the height must be no bigger than 2 less than the y-dimension of
the fabric.

Additionally, if you are running on a real CS-2, `fabric-dims` must
be `750,994`.

To run:

```default
cs_python run.py --name out --verify

```

where the `--verify` flag verifies the result on each PE.
This flag is incompatible with any number of iterations greater than 1.

Again, pass `--cmaddr` to run on a real CS-2.
If you’re just running on simfabric, there’s no need to use a width or height
greater than one. The cycle counts will be the same across all simulated PEs.

The program will report a “relative” and “absolute” memory bandwidth. Relative
refers to the bandwidth obtained if we calculate based on the number of memory
accesses that would occur to implement a matrix-vector product on a more
traditional architecture, where caching allows us to write the y vector
for example, only once to memory. “Absolute” refers to the actual number of
memory accesses required to run this program on the WSE. See comments in the
code for more details.

## layout\_matvec.csl

```csl
param width: u16;
param height: u16;
param tile_size: u16;
param iters: u16;

const memcpy = @import_module("<memcpy/get_params>", .{
  .width = width,
  .height = height,
});

layout {
  @set_rectangle(width, height);

  for (@range(u16, width)) |px| {
    const memcpy_params = memcpy.get_params(px);
    for (@range(u16, height)) |py| {
      @set_tile_code(px, py, "pe_matvec.csl", .{ .memcpy_params = memcpy_params,
        .nb = tile_size, .iters = iters});
    }
  }

  // export symbol names
  @export_name("A", [*]f32, true);
  @export_name("x", [*]f32, true);
  @export_name("y", [*]f32, true);
  @export_name("maxmin_time", [*]f32, true);
  @export_name("compute", fn()void);
}

```

## pe\_matvec.csl

```csl
param memcpy_params: comptime_struct;

param nb: u16;    // array tile size, corresponds to matrix dimension
param iters: u16; // num iterations to run matvec

const EXIT:   local_task_id = @get_local_task_id(9);  // entrypoint to leave RPC

// alignment calculation
const pad_align:   u16 = 16;
const elem_size:   u16 = 4;
const align_ratio: u16 = pad_align / elem_size;
const padded_nb:   u16 = if ((nb / align_ratio) * align_ratio == nb) nb
                         else (nb / align_ratio + 1) * align_ratio;

const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);

const timestamp = @import_module("<time>");
var tsc_end_buf = @zeros([timestamp.tsc_size_words]u16);
var tsc_start_buf = @zeros([timestamp.tsc_size_words]u16);

var timer_buf = @zeros([3]f32);
var ptr_timer_buf: [*]f32 = &timer_buf;

var A_array: [nb*padded_nb+1]f32 align(16) = @zeros([nb*padded_nb+1]f32);
var x_array: [nb]f32 align(16) = @zeros([nb]f32);
var y_array: [padded_nb]f32 align(16) = @zeros([padded_nb]f32);
var ptr_A: [*]f32 = &A_array;
var ptr_x: [*]f32 = &x_array;
var ptr_y: [*]f32 = &y_array;

var A_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{padded_nb} -> A_array[i+1] });
var x_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{nb} -> x_array[i] });
var y_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{padded_nb} -> y_array[i] });

const y_dest_dsr = @get_dsr(dsr_dest, 0);
const y_src0_dsr = @get_dsr(dsr_src0, 0);
const A_src1_dsr = @get_dsr(dsr_src1, 0);

fn gemv_static_step_A(curX: f32) void {
  @fmacs(y_dest_dsr, y_src0_dsr, A_src1_dsr, curX);
}

fn gemv_map() void {

  // COMPUTE //
  /////////////

  // A * X = Y
  ////////////

  var local_A_dsd: mem1d_dsd = A_dsd;
  var local_y_dsd: mem1d_dsd = y_dsd;
  @load_to_dsr(y_dest_dsr, local_y_dsd, .{ .save_address = false });
  @load_to_dsr(y_src0_dsr, local_y_dsd, .{ .save_address = false });
  @load_to_dsr(A_src1_dsr, local_A_dsd, .{ .save_address = true });
  @map(gemv_static_step_A, x_dsd);
}

fn compute() void {

  timestamp.enable_tsc();
  timestamp.get_timestamp(&tsc_start_buf);

  for (@range(u16, iters)) |iter| {
    gemv_map();
  }

  timestamp.get_timestamp(&tsc_end_buf);
  timestamp.disable_tsc();

  var lo_: u16 = 0;
  var hi_: u16 = 0;
  var word: u32 = 0;

  lo_ = tsc_start_buf[0];
  hi_ = tsc_start_buf[1];
  timer_buf[0] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

  lo_ = tsc_start_buf[2];
  hi_ = tsc_end_buf[0];
  timer_buf[1] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

  lo_ = tsc_end_buf[1];
  hi_ = tsc_end_buf[2];
  timer_buf[2] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

  @activate(EXIT);
}

task f_exit() void {
  // the user must unblock cmd color for every PE
  sys_mod.unblock_cmd_stream();
}

comptime {
  @bind_local_task(f_exit, EXIT);

  @export_symbol(ptr_A, "A");
  @export_symbol(ptr_x, "x");
  @export_symbol(ptr_y, "y");
  @export_symbol(ptr_timer_buf, "maxmin_time");
  @export_symbol(compute);
}

```

## run.py

```python
#!/usr/bin/env cs_python
# pylint: disable=line-too-long,too-many-function-args

import argparse
import csv
import json
import math
import struct
import time

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

def parse_args():
  """parse the command line"""

  parser = argparse.ArgumentParser(description="single tile matvec run parameters")
  parser.add_argument("--name", required=False, default="out", help="prefix of ELF files")
  parser.add_argument("--cmaddr", required=False, default="", help="IP:port for CS system")
  parser.add_argument("--verify", action="store_true", help="Verify Y computation")

  args = parser.parse_args()
  return args

def float_to_hex(f):
  return hex(struct.unpack("<I", struct.pack("<f", f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def sub_ts(words):
  return make_u48(words[3:]) - make_u48(words[0:3])

def main():
  """Main method to run the example code."""

  args = parse_args()

  name = args.name
  cmaddr = args.cmaddr
  verify = args.verify

  # Parse the compile metadata
  with open(f"{name}/out.json", encoding="utf-8") as json_file:
    compile_data = json.load(json_file)

  nb = int(compile_data["params"]["tile_size"])
  width = int(compile_data["params"]["width"])
  height = int(compile_data["params"]["height"])
  iters = int(compile_data["params"]["iters"])

  print(f"nb = {nb}")
  print(f"width = {width}")
  print(f"height = {height}")
  print(f"iters = {iters}")

  # Calculate alignment and padding to avoid bank conflicts
  align = 16
  multiple = int(align / 4)
  padded_nb = math.ceil(nb / multiple) * multiple

  #############
  # Run
  #############

  start = time.time()

  # Instantiate runner
  runner = SdkRuntime(name, cmaddr=cmaddr)

  # Device symbols for memcpy
  A_symbol = runner.get_id("A")
  x_symbol = runner.get_id("x")
  y_symbol = runner.get_id("y")
  symbol_maxmin_time = runner.get_id("maxmin_time")

  # Load and begin run
  runner.load()
  runner.run()

  # Construct A data and copy random A matrix PE (0,0) for verification
  A_mat = np.random.rand(nb, nb)
  A_data = np.zeros(width * height * (nb * padded_nb + 1), dtype=np.float32)

  for w in range(width):
    for h in range(height):
      for i in range(nb):
        for j in range(nb):
          A_data[(h * width + w) * (nb * padded_nb + 1) + j * padded_nb + i + 1] = A_mat[i, j]

  print()
  print("Beginning run.")
  print("Copy A matrices to device...")
  runner.memcpy_h2d(
      A_symbol,
      A_data,
      0,
      0,
      width,
      height,
      nb * padded_nb + 1,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_32BIT,
      order=MemcpyOrder.ROW_MAJOR,
      nonblock=False,
  )

  # Construct x data and copy random x vector to PE (0,0) for verification
  x_vec = np.random.rand(nb)
  x_data = np.zeros(width * height * nb, dtype=np.float32)
  for w in range(width):
    for h in range(height):
      x_data[(h * width + w) * nb:(h * width + w) * nb + nb] = x_vec

  print("Copy x vectors to device...")
  runner.memcpy_h2d(
      x_symbol,
      x_data,
      0,
      0,
      width,
      height,
      nb,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_32BIT,
      order=MemcpyOrder.ROW_MAJOR,
      nonblock=False,
  )

  # Launch the compute kernel
  print("Launch kernel...")
  runner.call("compute", [], nonblock=False)

  # Copy back timestamps from device
  data = np.zeros((width * height * 3, 1), dtype=np.uint32)
  runner.memcpy_d2h(
      data,
      symbol_maxmin_time,
      0,
      0,
      width,
      height,
      3,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_32BIT,
      order=MemcpyOrder.ROW_MAJOR,
      nonblock=False,
  )
  maxmin_time_hwl = data.view(np.float32).reshape((height, width, 3))
  print("Copied back timestamps.")

  # Copy back data array from device
  if verify:
    data = np.zeros((width * height * padded_nb, 1), dtype=np.uint32)
    runner.memcpy_d2h(
        data,
        y_symbol,
        0,
        0,
        width,
        height,
        padded_nb,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    y_device_array = data.view(np.float32).reshape((height, width, padded_nb))
    print("Copied back Y array.")

  print("Done.")

  # End walltime timer
  runner.stop()
  end = time.time()
  walltime = end - start

  ###########
  # Verify
  ###########

  if verify:
    print("Test y result is as expected on each PE...")
    expected = A_mat @ x_vec
    for w in range(width):
      for h in range(height):
        np.testing.assert_allclose(y_device_array[h, w, :nb], expected, atol=0.0001, rtol=0)
    print("SUCCESS!")

  #################################
  # Calculate mem accesses and FLOP
  #################################

  # STANDARD read/writes
  # Read full x, read each column of V stack, write full y = nb + nb*nb + nb
  # = nb*nb + 2*nb
  #
  # 4 bytes per elem. Mem = 4 * (nb*nb + 2*nb)
  #                       = 4*nb*nb + 8*nb

  # ACTUAL read/writes
  # Read full x; read each col of V stack; read, write full y nb times
  # = nb + nb*nb + 2*nb*nb
  # = 3*nb*nb + nb
  #
  # 4 bytes per elem. Mem = 4 * (3*nb*nb + nb)
  #                       = 12*nb*nb + 4*nb

  # Floating point operations
  # Compute A_ij * x_j for each i, j = nb * nb
  # For each row of A, reduction uses nb - 1 adds = nb * (nb-1)
  # = nb * nb + nb * (nb-1)
  # = 2*nb*nb - nb

  total_relative_accesses = width * height * (4 * nb * nb + 8 * nb)
  total_absolute_accesses = width * height * (12 * nb * nb + 4 * nb)
  total_flop = width * height * (2 * nb * nb - nb)

  #######################
  # Calculate cycle count
  #######################

  tsc_tensor_d2h = np.zeros(6).astype(np.uint16)
  min_cycles = math.inf
  max_cycles = 0

  for w in range(width):
    for h in range(height):
      hex_t0 = int(float_to_hex(maxmin_time_hwl[(h, w, 0)]), base=16)
      hex_t1 = int(float_to_hex(maxmin_time_hwl[(h, w, 1)]), base=16)
      hex_t2 = int(float_to_hex(maxmin_time_hwl[(h, w, 2)]), base=16)
      tsc_tensor_d2h[0] = hex_t0 & 0x0000FFFF
      tsc_tensor_d2h[1] = (hex_t0 >> 16) & 0x0000FFFF
      tsc_tensor_d2h[2] = hex_t1 & 0x0000FFFF
      tsc_tensor_d2h[3] = (hex_t1 >> 16) & 0x0000FFFF
      tsc_tensor_d2h[4] = hex_t2 & 0x0000FFFF
      tsc_tensor_d2h[5] = (hex_t2 >> 16) & 0x0000FFFF

      cycles = sub_ts(tsc_tensor_d2h)
      if cycles < min_cycles:
        min_cycles = cycles
        min_w = w
        min_h = h
      if cycles > max_cycles:
        max_cycles = cycles
        max_w = w
        max_h = h

  #####################
  # Calculate bandwidth
  #####################

  # Calculate in bytes/sec and FLOP/sec for program rectangle
  secs = max_cycles / 850000000.0
  relative_bw = total_relative_accesses / secs * iters
  absolute_bw = total_absolute_accesses / secs * iters
  flops_sec = total_flop / secs

  # Convert to Petabytes/sec and PetaFLOPS
  relative_bw /= 1.0e15
  absolute_bw /= 1.0e15
  flops_sec /= 1.0e15

  # Scale to program rectangle
  scale_factor = (994.0 * 750.0) / (width * height)
  scale_relative_bw = relative_bw * scale_factor
  scale_absolute_bw = absolute_bw * scale_factor
  scale_flops_sec = flops_sec * scale_factor

  #################
  # Generate output
  #################

  print()
  print(f"Real walltime: {walltime}s")
  print()
  print("Cycle Counts:")
  print("Min cycles (", min_w, ", ", min_h, "): ", min_cycles)
  print("Max cycles (", max_w, ", ", max_h, "): ", max_cycles)
  print()
  print("Accesses and FLOP Information:")
  print("Relative accesses (bytes): ", total_relative_accesses)
  print("Absolute accesses (bytes): ", total_absolute_accesses)
  print("FP operations:             ", total_flop)
  print()
  print("Bandwidth and FLOPS Information:")
  print("Relative BW (PB/s): ", relative_bw)
  print("Absolute BW (PB/s): ", absolute_bw)
  print("PetaFLOPS:          ", flops_sec)
  print()
  print("Scaled (", width, ",", height, ") to (750,994)...")
  print("Scaled relative BW (PB/s): ", scale_relative_bw)
  print("Scaled absolute BW (PB/s): ", scale_absolute_bw)
  print("Scaled PetaFLOPS:          ", scale_flops_sec)

  # Write a CSV
  csv_name = name + ".csv"
  with open(csv_name, encoding="utf-8", mode="a") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        cmaddr,
        width,
        height,
        iters,
        nb,
        padded_nb,
        min_cycles,
        max_cycles,
        total_relative_accesses,
        total_absolute_accesses,
        relative_bw,
        absolute_bw,
        scale_relative_bw,
        scale_absolute_bw,
        total_flop,
        flops_sec,
        scale_flops_sec,
        walltime,
    ])

if __name__ == "__main__":
  main()

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc ./src/layout_matvec.csl --arch wse3 --fabric-dims=9,4 \
--fabric-offsets=4,1 \
--params=width:2,height:2,tile_size:25,iters:1 \
-o out --memcpy --channels=1
cs_python ./run.py --name out --verify

```

---

## sweep.py

```python
#!/usr/bin/env python

import argparse
import subprocess

def parse_args():
  """parse the command line"""

  parser = argparse.ArgumentParser(description="Sweep single tile matvec size parameter")
  parser.add_argument("--name", required=False, default="out", help="Prefix of ELF files")
  parser.add_argument("--cmaddr", required=False, default="", help="IP:port for CS system")
  parser.add_argument("--dims", help="Fabric and program dimension, i.e. <W>,<H>")
  parser.add_argument(
      "--iters",
      required=False,
      type=int,
      default=1,
      help="Number of iterations for each matvec",
  )

  args = parser.parse_args()

  return args

def cslc_compile(width: int, height: int, tile_size: int, iters: int, name: str):
  """Generate ELFs for the layout"""

  args = []
  args.append("cslc")  # command
  args.append("layout_matvec.csl")  # file
  args.append(f"--fabric-dims={width+7},{height+2}")  # options
  args.append("--fabric-offsets=4,1")
  args.append(f"--params=width:{width},height:{height},tile_size:{tile_size},iters:{iters}")

  args.append(f"-o={name}")
  args.append("--arch=wse2")
  args.append("--memcpy")
  args.append("--channels=1")
  print(f"subprocess.check_call(args = {args}")
  subprocess.check_call(args)

def cs_run(name: str, cmaddr: str):
  """Run with cs_python"""

  args = []
  args.append("cs_python")
  args.append("run.py")
  args.append(f"--name={name}")
  args.append(f"--cmaddr={cmaddr}")
  subprocess.check_call(args)

def compile_and_run(width: int, height: int, tile_size: int, iters: int, name: str, cmaddr: str):
  """Compile and run program."""

  cslc_compile(width, height, tile_size, iters, name)

  cs_run(name, cmaddr)

def main():
  """Main method to run the example code."""

  args = parse_args()

  w, h = args.dims.split(",")
  width = int(w)
  height = int(h)

  name = args.name  # compilation output
  cmaddr = args.cmaddr
  iters = args.iters

  for tile_size in range(10, 101, 10):
    compile_and_run(width, height, tile_size, iters, name, cmaddr)

if __name__ == "__main__":
  main()

```
[previous

BiCGSTAB](csl/code-examples/benchmark-bicgstab.md "previous page")
[next

Host-to-Device Broadcast Test](csl/code-examples/benchmark-row-col-broadcast.md "next page")
