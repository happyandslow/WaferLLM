---
title: "3D FFT"
url: "https://sdk.cerebras.net/csl/code-examples/benchmark-fft-3d"
crawled_at: "2026-02-14 23:55:30"
---

# 3D FFT

> Source: [https://sdk.cerebras.net/csl/code-examples/benchmark-fft-3d](https://sdk.cerebras.net/csl/code-examples/benchmark-fft-3d)

This example implements a 3D Discrete Fourier Transform by using a pencil
decomposition, in which the input data is viewed as a 2D array of 1D pencils,
and each PE stores a small subarray of the 2D array of pencils.

The algorithm proceeds in steps. First, the 1D FFT of the pencils on each PE
are performed. Then, the data is transposed along a coordinate axis among
all PEs. This process happens two more times, resulting in three local
operations in which 1D FFTs are performed independently on each PE, and three
transpose operations in which all PEs commmunicate to change which axis of
the data is stored in memory.

The algorithm used to compute the 1D FFTs is Cooley-Tukey,
Decimation in Time (DIT), radix 2, with the
slight tweak that we use iteration instead of recursion.

## FFT Compilation Parameters

- `N`: Size of 3D FFT along one dimension. The full problem size is
  `N x N x N`.
- `NUM_PENCILS_PER_DIM`: Number of pencils along a given dimension on each PE.
  For instance, `NUM_PENCILS_PER_DIM == 2` means that each PE stores
  `2 x 2` pencils.
- `FP`: Floating point precision. Valid values are `1` or `2`, specifying
  IEEE fp16 or fp32, respectively.

## FFT Runtime Parameters

- `--inverse`: With this flag set, perform an inverse Fourier transform.
- `--real`: With this flag set, compute Fourier transform with real input
  data. Without this flag, complex Fourier transform is computed.
- `--norm`: Normalization strategy. Valid values are `0`, `1`, or `2`,
  specifying `forward`, `backward`, or `orthonormal`, respectively.

## layout.csl

```csl
param FP: i16;   // Precision: 0 == float16, 1 == float32

param N: u16;   // FFT size in each dimension
param NUM_PENCILS_PER_DIM: u16; // Pencils in each dimension per PE

// Number of PEs for FFT in both X and Y dimension
param WIDTH: i16 = N / NUM_PENCILS_PER_DIM;

const tensor_type: type = if (FP == 0) f16 else f32;

const memcpy = @import_module("<memcpy/get_params>", .{
  .width = WIDTH,
  .height = WIDTH,
});

const fft_helper = @import_module("<kernels/fft/fft3d_layout>", .{
  .width = WIDTH,
  .memcpy = memcpy,
});

layout {
  @set_rectangle(WIDTH, WIDTH);
  fft_helper.FFT_kernel(WIDTH, N, tensor_type);
}

```

## run.py

```python

import argparse
import json
import time
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder  # pylint: disable=no-name-in-module
from cerebras.sdk import sdk_utils  # pylint: disable=no-name-in-module

def make_u48_array(b: np.ndarray):
  return b[..., 0] + (np.left_shift(b[..., 1], 16, dtype=int)) \
                   + (np.left_shift(b[..., 2], 32, dtype=int))

def arguments():
  parser = argparse.ArgumentParser(description="FFT parameters.")
  parser.add_argument(
    "-no_check",
    "--no-check-outcome",
    action="store_true",
    help="Do not validate outcome against numpy implementation",
  )
  parser.add_argument(
    "-n",
    "--name",
    required=True,
    default="out",
    help="Compile output directory.",
  )
  parser.add_argument(
    "-i",
    "--inverse",
    action="store_true",
    help="Compute the inverse FFT.",
    default=False,
  )
  parser.add_argument(
    "-r",
    "--real",
    action="store_true",
    help="Compute real FFT.",
    default=False,
  )
  parser.add_argument(
    "--norm",
    help="Normalization (0=backward, 1=ortho, 2=forward.",
    type=int,
    default=0,
  )
  parser.add_argument(
    "--save-output",
    action="store_true",
    help="Save result of FFT to .npy file.",
    default=False
  )
  parser.add_argument("--cmaddr", help="IP:port for CS system")
  return parser.parse_args()

args = arguments()

CHECK_RES = not args.no_check_outcome
REAL = args.real
NORM = np.int16(args.norm)
INVERSE = np.int16(args.inverse)
np_norm = "backward" if NORM == 0 else "forward" if NORM == 2 else "ortho"

# Parse the compile metadata
with open(f"{args.name}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
compile_params = compile_data["params"]

N = int(compile_params["N"])
T = int(compile_params["NUM_PENCILS_PER_DIM"])
FP = int(compile_params["FP"])

# number of PEs per dimension is FFT size along dimension divided by
# number of FFT pencils per dimension
width = N // T

precision_type = np.dtype(np.float16 if FP == 0 else np.float32)
print(f"N: {N}, num_pencils_per_dim: {T}, kernel width: {width}, data type: {precision_type}")

# element has real and imaginary parts
ELEM_SIZE = 2

LOCAL_TENSOR_ELEM = T * T * N
LOCAL_TENSOR_LEN = LOCAL_TENSOR_ELEM * ELEM_SIZE

GLOBAL_TENSOR_ELEM = N * N * N
GLOBAL_TENSOR_LEN = GLOBAL_TENSOR_ELEM * ELEM_SIZE

# create twiddle factors
f_container = np.zeros(N, dtype=np.uint32)
f = sdk_utils.memcpy_view(f_container, precision_type)

Nh = N >> 1
exponent = np.pi * np.arange(Nh) / Nh
fI = np.sin(exponent).astype(precision_type)
fR = np.cos(exponent).astype(precision_type)
f[0::2] = fR
f[1::2] = fI

# Set the seed so that CI results are deterministic
np.random.seed(seed=7)

# Create random array if CHECK_RES set, or a fixed array if not
if CHECK_RES:
  random_array = np.random.random(GLOBAL_TENSOR_LEN).astype(precision_type)
else:
  random_array = np.arange(GLOBAL_TENSOR_LEN).astype(precision_type)

X_pre = random_array.reshape((N, N, ELEM_SIZE*N))

X_container = np.zeros((width, width, LOCAL_TENSOR_LEN), dtype=np.uint32)
X_res_container = np.zeros((width, width, LOCAL_TENSOR_LEN), dtype=np.uint32)

X = sdk_utils.memcpy_view(X_container, precision_type)
X_res = sdk_utils.memcpy_view(X_res_container, precision_type)

# Reshuffle input to expected order
# On each PE, pencils are interleaved
for x in range(N):
  for y in range(N):
    for z in range(N):
      offset = z * T * T + (y % T) * T + x % T
      X[y // T][x // T][offset*2] = X_pre[y][x][2*z]
      if REAL:
        X[y // T][x // T][offset*2+1] = 0
      else:
        X[y // T][x // T][offset*2+1] = X_pre[y][x][2*z+1]

#########################################
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT if precision_type == np.float32 \
  else MemcpyDataType.MEMCPY_16BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Create SdkRuntime and load and run program
runner = SdkRuntime(args.name, cmaddr=args.cmaddr, suppress_simfab_trace=True)
runner.load()
runner.run()

# Get device symbols for data array, twiddle factors, and timestamps
symbol_X = runner.get_id("X")
symbol_twiddle = runner.get_id("twiddle_array")
symbol_timestamps = runner.get_id("fft_time")

# Write twiddle factors 'f' to every PE
f_fill = np.full((width, width, N), f_container, dtype=np.uint32)
runner.memcpy_h2d(symbol_twiddle, f_fill.ravel(), 0, 0, width, width, N,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)

tstart = time.time()
# Copy input data to device
runner.memcpy_h2d(symbol_X, X_container.ravel(), 0, 0, width, width, LOCAL_TENSOR_LEN,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)
if REAL:
  if INVERSE:
    runner.launch("csfftExecC2R", NORM, nonblock=False)
  else:
    runner.launch("csfftExecR2C", NORM, nonblock=False)
else:
  runner.launch("csfftExecC2C", NORM, INVERSE, nonblock=False)

# Copy back result from device
runner.memcpy_d2h(X_res_container.ravel(), symbol_X, 0, 0, width, width, LOCAL_TENSOR_LEN,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)
tstop = time.time()

print(f"Time to compute FFT and transfer result back: {tstop - tstart}s")

# Copy back timestamps from device
timestamps = np.zeros((width, width, 2), dtype=np.uint32)
runner.memcpy_d2h(timestamps.ravel(), symbol_timestamps, 0, 0, width, width, 2,
                  streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                  order=memcpy_order, nonblock=False)

# Compute worst PE time
timestamps = np.frombuffer(timestamps.tobytes(), dtype=np.uint16).reshape((width, width, 4))
u48cycles_array = make_u48_array(timestamps)
cycles = u48cycles_array[:, :].max()
cs2_freq = 850000000.0
compute_time = cycles / cs2_freq

print(f"Compute time on WSE: {compute_time}s, {cycles} cycles")

# Stop device program
runner.stop()

# Create result arrays to check result and write to file
result_array_pre = np.zeros((width, width, LOCAL_TENSOR_ELEM), dtype=complex)
result_array = np.zeros((N, N, N), dtype=complex)

# Create complex array out of real and imaginary parts
for row in range(width):
  for i in range(width):
    for j in range(LOCAL_TENSOR_ELEM):
      result_array_pre[row][i][j] = complex(
          X_res[row][i][j * 2], X_res[row][i][j * 2 + 1])

# Unshuffle result from N/T x N/T x N*T*T array to N x N x N array
for x in range(N):
  for y in range(N):
    for z in range(N):
      offset = z * T * T + (y % T) * T + x % T
      result_array[y][x][z] = result_array_pre[y // T][x // T][offset]

if args.save_output:
  np.save("result_array.npy", result_array)

# Reshape input array to match np.fft format
if REAL:
  random_array_sq = random_array.reshape((N, N, N, 2))[:, :, :, 0].reshape((N, N, N))
else:
  # For the float16 case, you must first cast array to float32. Otherwise
  # the view will interpret the four float16s as a single complex number.
  random_array_sq = random_array.astype(np.float32).reshape((N, N, N*2)).view(np.csingle)

# Compute numpy reference
if INVERSE:
  reference_array = np.fft.ifftn(random_array_sq, norm=np_norm)
else:
  reference_array = np.fft.fftn(random_array_sq, norm=np_norm)

# Check result against numpy reference
if CHECK_RES:
  # 16-bit calculation can have large relative errors for entries close to 0
  rtol = 0.5 if (precision_type == np.float16) else 0.01
  np.testing.assert_allclose(
    result_array, reference_array, rtol=rtol, atol=0)
  print("\nSUCCESS!")

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=N:16,NUM_PENCILS_PER_DIM:4,FP:1 --memcpy --channels=1 -o out
cs_python run.py --name out --real --norm 1
cs_python run.py --inverse --name out --norm 1
cslc --arch=wse3 ./layout.csl --fabric-dims=11,6 --fabric-offsets=4,1 \
--params=N:16,NUM_PENCILS_PER_DIM:4,FP:0 --memcpy --channels=1 -o out
cs_python run.py --name out
cs_python run.py --inverse --name out

```
[previous

Host-to-Device Broadcast Test](csl/code-examples/benchmark-row-col-broadcast.md "previous page")
[next

Conway’s Game of Life](csl/code-examples/benchmark-game-of-life.md "next page")
