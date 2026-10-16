# GEMM Output is Column-Major Due to save_address Ordering — 2026-03-17

**Date**: 2026-03-17
**Context**: Investigating layout mismatch in fused GEMM→GEMV pipeline (WaferServe/fused_ops/gemm_gemv)
**Category**: gotcha

## Problem / Observation

When fusing GEMM and GEMV to share an on-PE buffer (`shared_tile`), the GEMV produces wrong results. The GEMM output is **column-major** but GEMV expects **row-major** input.

## Root Cause

The memory layout is determined by which DSR has `save_address = true` in the `@map` + `@fmach` pattern:

```csl
// GEMM inner loop:
// out_dsd has extent Mt, W_dsd has extent Nt
@load_to_dsr(out_dest_dsr, out_dsd, .{ .save_address = true });   // ← advances by Mt
@load_to_dsr(out_src0_dsr, out_dsd, .{ .save_address = true });
@load_to_dsr(X_src1_dsr, X_dsd, .{ .save_address = false });      // ← stays fixed
@map(gemv_static_step_X, W_dsd);  // iterates over Nt elements
```

The `@map` iterates Nt times. Each iteration, `@fmach` writes Mt elements to `out`, then `save_address=true` advances the out pointer by Mt. Result:

- Iteration n=0: writes `shared_tile[0..Mt-1]`
- Iteration n=1: writes `shared_tile[Mt..2*Mt-1]`
- → `shared_tile[n*Mt + m] = result[m][n]` — **column-major**

GEMV reads with `W_dsd = |i|{Nt} -> ptr_shared[i]`, `save_address=true` on W_src1 (advances by Nt per @map iteration):
- Row m reads `shared_tile[m*Nt .. m*Nt+Nt-1]`
- → Expects `shared_tile[m*Nt + n] = W[m][n]` — **row-major**

## Key Insight

**The `save_address` flag on the output DSR determines the memory layout orientation, not the DSD declaration.** When out advances by Mt for each of Nt iterations, columns fill first → column-major. The host readback reshape `(P, P, Nt, Mt)` confirms this: Nt groups of Mt.

## Diagnostic Technique

**Check the host readback reshape to determine on-PE layout:**
- `reshape((P, P, Nt, Mt))` → column-major (Nt columns of Mt)
- `reshape((P, P, Mt, Nt))` → row-major (Mt rows of Nt)

## Fix Options

1. **On-PE transpose** in `reconfig()` using a temp buffer (cleanest, no algorithm changes)
2. **Strided DSD** in GEMV: `|i|{Nt} -> ptr_shared[i * Mt]` (no extra memory, but save_address interaction with strided DSDs needs verification)
3. **Restructure GEMM output** (most invasive)

## Related Resources

- Full analysis: `WaferServe/fused_ops/gemm_gemv/LAYOUT_MISMATCH_ANALYSIS.md`
- GEMM computation: `meshgemm.csl:mm_compute()`
- GEMV computation: `meshgemv.csl:mv_compute()`
