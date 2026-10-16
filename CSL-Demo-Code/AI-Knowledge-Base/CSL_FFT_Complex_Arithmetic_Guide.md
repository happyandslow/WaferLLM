# Cerebras CSL FFT & Complex Arithmetic Guide
## SDK Version: 1.4.0

This document covers complex number types, arithmetic, and FFT implementation patterns on the Cerebras WSE using CSL.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Complex Number Library](#2-complex-number-library)
3. [Complex Arithmetic with DSDs](#3-complex-arithmetic-with-dsds)
4. [Data Layout: Interleaved Real/Imaginary](#4-data-layout-interleaved-realimaginary)
5. [FFT Algorithm Implementation](#5-fft-algorithm-implementation)
6. [Inverse FFT (IFFT)](#6-inverse-fft-ifft)
7. [Precision: f16 vs f32](#7-precision-f16-vs-f32)
8. [Twiddle Factors](#8-twiddle-factors)
9. [Best Practices and Common Pitfalls](#9-best-practices-and-common-pitfalls)

---

## 1. Overview

| Feature | Support |
|---|---|
| **Complex type** | `<complex>` library: `complex_32` (f16 pair), `complex_64` (f32 pair) |
| **DSD operations** | Hardware `@fmulh/@fmuls`, `@fmach/@fmacs`, `@faddh/@fadds`, `@fsubh/@fsubs` |
| **FFT** | Iterative Cooley-Tukey, in-place, parameterized by `tensor_type` (f16 or f32) |
| **Data layout** | Interleaved [re₀, im₀, re₁, im₁, ...] in flat arrays |

---

## 2. Complex Number Library

### 2.1 Import

```csl
const complex_lib = @import_module("<complex>");
```

### 2.2 Types

```csl
const complex_32 = complex_lib.complex_32;  // struct { real: f16, imag: f16 }
const complex_64 = complex_lib.complex_64;  // struct { real: f32, imag: f32 }
```

### 2.3 Construction

```csl
const z = complex_lib.get_complex_64(@as(f32, 3.0), @as(f32, 4.0));
// z.real == 3.0, z.imag == 4.0

const w = complex_lib.get_complex_32(@as(f16, 1.0), @as(f16, -2.0));
```

### 2.4 Arithmetic API

| Function | Signature | Formula |
|---|---|---|
| `add_complex_64(x, y)` | `(complex_64, complex_64) → complex_64` | `(x.re+y.re, x.im+y.im)` |
| `subtract_complex_64(x, y)` | `(complex_64, complex_64) → complex_64` | `(x.re-y.re, x.im-y.im)` |
| `multiply_complex_64(x, y)` | `(complex_64, complex_64) → complex_64` | `(x.re*y.re - x.im*y.im, x.re*y.im + x.im*y.re)` |
| `add_complex_32(x, y)` | `(complex_32, complex_32) → complex_32` | Same, f16 |
| `subtract_complex_32(x, y)` | `(complex_32, complex_32) → complex_32` | Same, f16 |
| `multiply_complex_32(x, y)` | `(complex_32, complex_32) → complex_32` | Same, f16 |

### 2.5 Generic Functions

```csl
// Works on any complex type (complex_32 or complex_64)
const z = complex_lib.add_complex(x, y);         // Generic add
const z = complex_lib.subtract_complex(x, y);    // Generic subtract
const z = complex_lib.multiply_complex(x, y);     // Generic multiply
const z = complex_lib.get_complex(real_val, imag_val);  // Generic constructor
```

**Source**: `source-code/csl_libraries/complex.csl`

---

## 3. Complex Arithmetic with DSDs

For high-performance vector operations, use DSD builtins instead of the scalar `<complex>` API.

### 3.1 Complex Multiply via DSD Ops

Complex multiplication $(a + bi)(c + di) = (ac - bd) + (ad + bc)i$ decomposes into real operations:

```csl
// Given interleaved arrays: X[2k] = real, X[2k+1] = imag
// Multiply by twiddle factor (c1a, c1b):

// Step 1: result = X * c1a  (scale both real and imag by c1a)
@fmuls(resultDSD, sourceDSD, c1a);

// Step 2: Fix real part: real -= imag * c1b
@fmacs(real_dstDSD, real_dstDSD, imag_srcDSD, c1b);

// Step 3: Fix imaginary part: imag += real * (-c1b)
@fmacs(imag_dstDSD, imag_dstDSD, real_srcDSD, -c1b);
```

### 3.2 Butterfly Operation

The FFT butterfly computes:
- $X_{even} = X_{even} + T$ (add twiddle-multiplied odd element)
- $X_{odd} = X_{even} - T$ (subtract)

```csl
// T is in Xaux after twiddle multiply
@fsubs(X_odd_dsd, X_even_dsd, Xaux_dsd);   // X_odd = X_even - T
@fadds(X_even_dsd, X_even_dsd, Xaux_dsd);  // X_even = X_even + T
```

For f16 precision, replace `@fsubs`→`@fsubh`, `@fadds`→`@faddh`, etc.

---

## 4. Data Layout: Interleaved Real/Imaginary

### 4.1 Layout Convention

Complex arrays are stored in **interleaved** format:

```
Index:  0     1     2     3     4     5     ...  2N-2  2N-1
Value:  re₀   im₀   re₁   im₁   re₂   im₂  ...  reN₋₁  imN₋₁
```

Array size = 2×N for N complex numbers.

### 4.2 Accessing Real and Imaginary Parts

Use stride-2 DSDs to access only real or only imaginary elements:

```csl
// All elements (stride 1)
const full_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{2*N} -> X[i] });

// Real parts only (stride 2, offset 0)
var real_dsd = @set_dsd_stride(full_dsd, 2);   // Accesses X[0], X[2], X[4], ...

// Imaginary parts only (stride 2, offset 1)
var imag_dsd = @increment_dsd_offset(real_dsd, 1, tensor_type);  // X[1], X[3], X[5], ...
```

### 4.3 Even/Odd Splitting

For FFT decomposition into even and odd halves:

```csl
const X_even_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{N} -> X[i] });      // First N
const X_odd_dsd  = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{N} -> X[i+N] });    // Second N
```

---

## 5. FFT Algorithm Implementation

### 5.1 Parameterized Design

The FFT module is parameterized for precision and buffer sizes:

```csl
param N: u16;                  // FFT size
param FP: i16;                 // 1 = f16, 2 = f32
param tensor_type: type;       // f16 or f32
param ARRAY_LEN: u16;         // Input array length (≥ 2*N)
param X: *[ARRAY_LEN]tensor_type;  // Pointer to input/output data
```

### 5.2 Iterative Cooley-Tukey

The implementation uses an iterative (bottom-up) approach:

```csl
fn fft(f_twiddle: *[N]tensor_type) void {
    var subproblems = N;
    while (subproblems > 1) {
        var stride = subproblems;
        subproblems >>= 1;

        // For each sub-FFT at this stage:
        var offset: u16 = 0;
        while (offset < N) : (offset += stride) {
            // 1. Apply twiddle factors to odd elements
            var c1a = f_twiddle.*[offset];
            var c1b = f_twiddle.*[offset + 1];

            if (FP == 1) {
                @fmulh(resultDSD, sourceDSD, c1a);
                @fmach(real_dstDSD, real_dstDSD, imag_srcDSD, c1b);
                @fmach(imag_dstDSD, imag_dstDSD, real_srcDSD, -c1b);
            } else {
                @fmuls(resultDSD, sourceDSD, c1a);
                @fmacs(real_dstDSD, real_dstDSD, imag_srcDSD, c1b);
                @fmacs(imag_dstDSD, imag_dstDSD, real_srcDSD, -c1b);
            }

            // Advance DSDs for next sub-FFT
            sourceDSD = @increment_dsd_offset(sourceDSD, @as(i16, stride), tensor_type);
            // ... other DSDs similarly
        }

        // 2. Butterfly: even/odd combination
        if (FP == 1) {
            @fsubh(X_odd_dsd, X_even_dsd, Xaux_dsd);
            @faddh(X_even_dsd, X_even_dsd, Xaux_dsd);
        } else {
            @fsubs(X_odd_dsd, X_even_dsd, Xaux_dsd);
            @fadds(X_even_dsd, X_even_dsd, Xaux_dsd);
        }

        // 3. Reshape for next stage
        if (subproblems > 1) {
            reshape_x.reshape(...);
            reshape_aux.reshape(...);
        }
    }
}
```

### 5.3 DSD Offset Manipulation

The FFT heavily uses DSD offset/base manipulation to walk through sub-arrays:

```csl
// Move DSD to point to next sub-FFT group
sourceDSD = @increment_dsd_offset(sourceDSD, @as(i16, stride), tensor_type);

// Set new base address for a DSD
sourceDSD = @set_dsd_base_addr(Xaux_dsd, X);

// Set length for the current stage
sourceDSD = @set_dsd_length(sourceDSD, stride);
```

---

## 6. Inverse FFT (IFFT)

The IFFT reuses the FFT with conjugated twiddle factors and a 1/N scaling:

```csl
fn ifft(f_twiddle: *[N]tensor_type) void {
    // Step 1: Negate imaginary parts of twiddle factors (conjugate)
    var fDSD = @set_dsd_stride(
        @increment_dsd_offset(@set_dsd_base_addr(Xaux_dsd, f_twiddle), 1, tensor_type),
        2
    );
    fDSD = @set_dsd_length(fDSD, N);

    if (FP == 1) { @fnegh(fDSD, fDSD); }
    else         { @fnegs(fDSD, fDSD); }

    // Step 2: Forward FFT with conjugated twiddles
    fft(f_twiddle);

    // Step 3: Scale result by 1/N
    var resultDSD = @set_dsd_base_addr(@set_dsd_length(Xaux_dsd, 2 * N), X);
    var invN: tensor_type = @as(tensor_type, 1.0) / @as(tensor_type, N);

    if (FP == 1) { @fmulh(resultDSD, resultDSD, invN); }
    else         { @fmuls(resultDSD, resultDSD, invN); }

    // Step 4: Restore original twiddle factors
    if (FP == 1) { @fnegh(fDSD, fDSD); }
    else         { @fnegs(fDSD, fDSD); }
}
```

---

## 7. Precision: f16 vs f32

### 7.1 Compile-Time Selection

The `FP` parameter controls precision throughout:

| `FP` value | `tensor_type` | DSD ops | SIMD width |
|---|---|---|---|
| 1 | `f16` | `@fmulh`, `@faddh`, `@fsubh`, `@fmach`, `@fnegh`, `@fmovh` | Up to SIMD-8 |
| 2 | `f32` | `@fmuls`, `@fadds`, `@fsubs`, `@fmacs`, `@fnegs`, `@fmovs` | Up to SIMD-4 (WSE-3) |

### 7.2 Pattern

```csl
if (FP == 1) {
    @fmulh(dst, src1, src2);   // f16 path
} else {
    @fmuls(dst, src1, src2);   // f32 path
}
```

### 7.3 Trade-offs

| | f16 | f32 |
|---|---|---|
| **Precision** | ~3.3 decimal digits | ~7.2 decimal digits |
| **Throughput** | 2× SIMD width | 1× SIMD width |
| **Memory** | 2 bytes/element | 4 bytes/element |
| **Recommended** | Inference/preview | Training/scientific |

---

## 8. Twiddle Factors

### 8.1 Pre-computation

Twiddle factors are pre-computed on the host and transferred to the PE:

```python
import numpy as np

N = 64
twiddle = np.zeros(2*N, dtype=np.float32)
for k in range(N):
    angle = -2.0 * np.pi * k / N
    twiddle[2*k]   = np.cos(angle)   # Real part
    twiddle[2*k+1] = np.sin(angle)   # Imaginary part

# Transfer to PE via memcpy_h2d
runner.memcpy_h2d(twiddle_symbol, twiddle, ...)
```

### 8.2 On-PE Format

Twiddle factors are stored in the same interleaved format as complex data:

```
twiddle[2k]   = cos(-2πk/N)    (real)
twiddle[2k+1] = sin(-2πk/N)    (imaginary)
```

### 8.3 Indexed Access

In the FFT inner loop, twiddle factors are accessed by offset:

```csl
var c1a: tensor_type = f_twiddle.*[offset];      // cos component
var c1b: tensor_type = f_twiddle.*[offset + 1];  // sin component
```

---

## 9. Best Practices and Common Pitfalls

### Best Practices

1. **Use DSD ops for vector complex arithmetic**: `@fmacs`/`@fmach` with stride-2 DSDs beats scalar `<complex>` by orders of magnitude.
2. **Use `<complex>` for scalar/comptime work**: Good for single-value operations and comptime initialization.
3. **Pre-compute twiddle factors on host**: Avoids expensive trig on the PE.
4. **Use reshape modules**: For FFT bit-reversal and data rearrangement between stages.
5. **Parameterize precision**: The `FP`/`tensor_type` pattern allows a single codebase for f16 and f32.

### Common Pitfalls

1. **Overflow in f16 FFT**: f16 range is ±65504; large inputs will overflow. Scale inputs before FFT.
2. **Stride-2 DSD off-by-one**: Forgetting `@increment_dsd_offset` for imaginary parts skips to wrong data.
3. **Twiddle factor ordering**: The interleaved [cos, sin, cos, sin, ...] must match the algorithm's expected access pattern.
4. **Forgetting to restore twiddle factors after IFFT**: The IFFT modifies twiddle factors in-place (conjugation); must negate back.
5. **Mixing f16/f32 DSD ops**: Using `@fmuls` on f16 data or `@fmulh` on f32 data causes undefined behavior.

---

## Related Topics

- **DSD Operations**: See `CSL_DSD_Complete_Guide.md` for DSD types, stride, offset manipulation
- **Builtins**: See `CSL_Builtins_Reference_Guide.md` for `@fmacs`, `@fadds`, `@fnegs` reference
- **Standard Libraries**: See `CSL_Standard_Libraries_Guide.md` for `<complex>` and `<math>` imports
- **Algorithm Patterns**: See `CSL_Algorithm_Patterns_Guide.md` for state machine and @map patterns
- **Host Communication**: See `CSL_Host_Device_Communication_Guide.md` for H2D transfer of twiddle factors
