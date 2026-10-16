# GQA Implementation Comparison: WSE-3-GQA vs WSE-3-GQA-cheng

Both implementations derive from the same base decoder (`Decode/WSE-3/`). This document compares how each adapts the original MHA (Multi-Head Attention) design to support GQA (Grouped Query Attention), focusing on the deltas from the shared baseline.

---

## 1. Executive Summary

| | WSE-3 (Base) | WSE-3-GQA | WSE-3-GQA-cheng (full) |
|--|-------------|-----------|------------------------|
| **GQA strategy** | None (MHA) | Dimensional (asymmetric K/V dims) | Replicated-heads (runtime offset) |
| **decode.csl** | 965 lines | 1282 lines (**+33%**) | 942 lines (**-2%**) |
| **comm_pe.csl** | 1035 lines | 1708 lines (**+65%**) | 1086 lines (**+5%**) |
| **comm_layout.csl** | 405 lines | 575 lines (**+42%**) | 406 lines (**+0.2%**) |
| **Total CSL delta** | baseline (2519) | +1046 lines (+42%) | -85 lines (-3%) |
| **K/V weight shape** | `[dim_p_pe x dim_p_pe]` | `[dim_p_pe x kv_dim_p_pe]` | `[n_kv_heads x dim_p_pe x dim_p_pe]` |
| **QKV tile layout** | Symmetric `[Q\|K\|V]` (3 x dim) | Asymmetric `[Q\|K\|V]` (dim + 2*kv_dim) | Symmetric `[Q\|K\|V]` (3 x dim) |
| **Score tensor** | `[bsz x seq_len]` | `[gqa_group_size x bsz x seq_len]` | `[bsz x seq_len]` |
| **KV cache mgmt** | None (pre-filled) | Full (`process_kv()`, var-len, prefill) | None (pre-filled) |
| **GEMV variants** | 1 | 3 | 2 |
| **Comm reduce fns** | 7 | ~12 | ~8 |
| **Axis modes** | 2 (0, 1) | 3 (0, 1, 3) | 2 (0, 1) |
| **RoPE** | Enabled | Commented out | Enabled |
| **SILU bug** | Present | Inherited | Fixed |

**In short**: WSE-3-GQA is a heavy refactor (+42% code) that restructures dimensions, communication, and attention loops. WSE-3-GQA-cheng is a lightweight patch (-3% code) that bolts GQA onto the base with minimal structural changes.

---

## 2. The Base Decoder (WSE-3)

The base implementation assumes **full multi-head attention** where Q, K, V all have the same per-PE dimension:

```
Q_weight: [dim_p_pe x dim_p_pe]     K_weight: [dim_p_pe x dim_p_pe]     V_weight: [dim_p_pe x dim_p_pe]
QKV_tile: [bsz * 3 * dim_p_pe]      score: [bsz * seq_len_p_pe]
XKCache:  [dim_p_pe * seq_len_p_pe]  XVCache: [seq_len_p_pe * dim_p_pe]
```

Pipeline: `RMSNorm → QKV Proj → QKV Reduce → RoPE → Score GEMV → Softmax → Output GEMV → O Proj → Residual → RMSNorm → UP+GATE → SILU → Mul → DOWN → Residual`

Key characteristics:
- **One GEMV variant** (`vecmat_computation()`) — standard fixed-size
- **No KV cache management** — cache pre-filled by host, read-only during decode
- **No variable-length attention** — always operates over full `seq_len_p_pe`
- **7 communication functions**, 2 axis modes (X=0, Y=1)
- **`pes_p_kv_head` declared but unused** — GQA parameters exist in config but are inert

---

## 3. Delta Overview: What Each Implementation Changes

### WSE-3-GQA: Heavy Refactor (+42% CSL code)

Restructures the dimension model so K/V are inherently smaller than Q:

| Change category | What changed | Lines added |
|----------------|-------------|-------------|
| **New parameter** | `kv_dim_p_pe = (n_kv_heads * head_dim) / P` | Pervasive |
| **New parameter** | `gqa_group_size = dim_p_pe / kv_dim_p_pe` | Pervasive |
| **New parameter** | `prefill_len_p_pe` (initial cache fill) | ~10 |
| **K/V weight reshape** | `[dim_p_pe x dim_p_pe]` → `[dim_p_pe x kv_dim_p_pe]` | ~10 |
| **QKV tile reshape** | `[3*dim]` → `[dim + 2*kv_dim]` (asymmetric) | ~20 |
| **KV cache reshape** | `[dim x seq]` → `[kv_dim x seq]` + add `process_kv()` | ~50 |
| **Score reshape** | `[bsz x seq]` → `[gqa_group_size x bsz x seq]` | ~20 |
| **Multi-group score loop** | New nested `for(b) for(g)` in `score_matvec_mult()` | ~80 |
| **Multi-group softmax** | `softmax_score_all_groups()` with group-extended scalars | ~60 |
| **Multi-group output loop** | `output_matvec_mult_all_groups()` with V cache reuse | ~50 |
| **2 new GEMV variants** | `vecmat_computation_var_right()`, `_var_right_2d_strided()` | ~40 |
| **KV-head-scoped reduction** | New `axis=3` mode in comm_pe + comm_layout routing | ~170 + 170 |
| **Group-extended reductions** | `all_reduce_bsz_g()`, `all_reduceMax_bsz_g()`, var-len variants | ~330 |
| **Debug export symbols** | 7 intermediate snapshot buffers for validation | ~30 |
| **Extensive host launcher** | Reference Python impl, validation, prefill init | ~1600 |

### WSE-3-GQA-cheng: Lightweight Patch (-3% CSL code)

Keeps the base dimension model intact but replicates all KV heads per PE:

| Change category | What changed | Lines added |
|----------------|-------------|-------------|
| **New parameter** | `n_kv_heads` passed to CSL as compile-time param | ~5 |
| **New runtime variable** | `kv_head_id = quotient_x % n_kv_heads` | ~5 |
| **K/V weight resize** | `[dim x dim]` → `[n_kv_heads x dim x dim]` (replicated) | ~10 |
| **KV cache resize** | `[dim x seq]` → `[n_kv_heads x dim x seq]` (replicated) | ~10 |
| **DSD offset selection** | `@increment_dsd_offset(dsd, kv_head_id * block_size)` in K/V/score/output ops | ~20 |
| **New GEMV variant** | `vecmat_computation_dsd()` (DSD snapshot for offset preservation) | ~25 |
| **SILU bug fix** | Correct iteration count + correct z2 offset | ~10 |
| **DOWN weight fix** | Semantic shape correction `[ffn x dim]` | ~2 |
| **QKV input fix** | `ptr_X` → `ptr_X_norm` (full variant) | ~3 |
| **Comm lib additions** | Minor: `all_reduce_bsz_g` placeholder (~50 lines) | ~50 |

---

## 4. GQA Design: Dimensional vs Replicated

### WSE-3-GQA: Dimensional GQA

The GQA ratio is baked into tensor dimensions at compile time:

```
kv_dim_p_pe = (n_kv_heads * head_dim) / P     // Smaller than dim_p_pe
gqa_group_size = dim_p_pe / kv_dim_p_pe        // How many Q groups share one KV
```

For Llama-8B (P=256): `dim_p_pe=16`, `kv_dim_p_pe=4`, `gqa_group_size=4`.

```
Base:     Q:[16x16]  K:[16x16]  V:[16x16]    QKV:[3*16=48]    Score:[1*seq]
                                               ↓
WSE-3-GQA: Q:[16x16]  K:[16x4]   V:[16x4]    QKV:[16+2*4=24]  Score:[4*1*seq]
```

The Q output vector (`dim_p_pe=16`) is conceptually split into `gqa_group_size=4` groups of `kv_dim_p_pe=4` each. Each group independently computes attention against the **same** (smaller) K/V cache. No runtime head mapping needed — the sharing is structural.

**Impact on base code**: Nearly every function that touches K, V, score, or output must change to handle the asymmetric dimensions and multi-group loops. The communication library needs entirely new reduction scopes.

### WSE-3-GQA-cheng: Replicated-Heads GQA

The dimension model stays identical to the base. Instead, all `n_kv_heads` weight blocks are stored per PE, and the correct one is selected at runtime:

```
Base:      Q:[16x16]  K:[16x16]  V:[16x16]    QKV:[3*16=48]    Score:[1*seq]
                                               ↓
GQA-cheng: Q:[16x16]  K:[8*16x16] V:[8*16x16] QKV:[3*16=48]    Score:[1*seq]
                       ↑ replicated              (same layout!)
                       select via offset
```

At init: `kv_head_id = quotient_x % n_kv_heads`. At operation time:
```csl
var k_offset: i16 = kv_head_id * dim_p_pe * dim_p_pe;
right_matrix_dsd = @increment_dsd_offset(right_matrix_dsd, k_offset, f16);
```

**Impact on base code**: Minimal structural changes. The QKV tile, score tensor, softmax, output GEMV, and communication functions all retain their base shapes. Only K/V-touching functions gain an offset computation (~4 lines each), and a new `vecmat_computation_dsd()` preserves the offset across batch rows.

---

## 5. Per-PE Memory Comparison (Llama-8B, P=256, n_kv_heads=8, 4:1 GQA)

| Buffer | Base (MHA) | WSE-3-GQA | GQA-cheng | GQA vs Base | cheng vs Base |
|--------|-----------|-----------|-----------|-------------|---------------|
| Q weight | 256 f16 | 256 f16 | 256 f16 | — | — |
| K weight | 256 f16 | **64 f16** | **2048 f16** | **4x smaller** | **8x larger** |
| V weight | 256 f16 | **64 f16** | **2048 f16** | **4x smaller** | **8x larger** |
| KV cache (K) | 320 f16 | **80 f16** | **2560 f16** | **4x smaller** | **8x larger** |
| KV cache (V) | 320 f16 | **80 f16** | **2560 f16** | **4x smaller** | **8x larger** |
| Score | 20 f16 | **80 f16** | 20 f16 | 4x larger | — |
| QKV tile | 48 f16 | **24 f16** | 48 f16 | 2x smaller | — |

(dim_p_pe=16, kv_dim_p_pe=4, seq_len_p_pe=20, n_kv_heads=8)

WSE-3-GQA **reduces** K/V memory by the GQA ratio (4x for 4:1 GQA) but expands the score buffer by `gqa_group_size`.
WSE-3-GQA-cheng **increases** K/V memory by `n_kv_heads` (8x) because it replicates all heads.

---

## 6. Delta: QKV Projection & Tile Layout

### Base → WSE-3-GQA (Asymmetric reshape)

The QKV tile structure is fundamentally altered:

```diff
- var QKV_tile: [bsz * 3 * dim_p_pe]f16;
+ var QKV_tile: [bsz * (dim_p_pe + 2 * kv_dim_p_pe)]f16;

- var K_dsd: offset=bsz*dim_p_pe,   extent=bsz*dim_p_pe;      // [bsz x 16]
+ var K_dsd: offset=bsz*dim_p_pe,   extent=bsz*kv_dim_p_pe;   // [bsz x 4]

  fn xk_matvec_mult() {
-     Kt = dim_p_pe;  Nt = dim_p_pe;    // 16 → 16
+     Kt = dim_p_pe;  Nt = kv_dim_p_pe; // 16 → 4
  }
```

The fused QKV reduction changes from `3*dim_p_pe` to `dim_p_pe + 2*kv_dim_p_pe` — **less communication volume** when `kv_dim < dim`.

### Base → WSE-3-GQA-cheng (Same layout, different weights)

QKV tile is **unchanged** from base:

```diff
  var QKV_tile: [bsz * 3 * dim_p_pe]f16;  // Same as base

  fn xk_matvec_mult() {
+     var k_offset: i16 = kv_head_id * dim_p_pe * dim_p_pe;
+     right_matrix_dsd = @set_dsd_base_addr(right_matrix_dsd, ptr_K_weight);
+     right_matrix_dsd = @increment_dsd_offset(right_matrix_dsd, k_offset, f16);
      Kt = dim_p_pe;  Nt = dim_p_pe;    // Still 16 → 16
-     vecmat_computation();
+     vecmat_computation_dsd();           // DSD-snapshot variant
  }
```

The QKV reduce stays at `3*dim_p_pe` — **same communication volume** as base.

---

## 7. Delta: Attention Score Computation

### Base: Single Fixed-Length GEMV

```csl
fn score_matvec_mult() void {
    Kt = dim_p_pe;  Nt = seq_len_p_pe;
    vecmat_computation();                      // Q @ K_cache^T
    all_reduce_bsz_seq_len(px, ...);           // Global X-reduce
    score *= alpha;
}
```

### Base → WSE-3-GQA (+80 lines: multi-group, variable-length, head-scoped)

The most heavily modified function:

```diff
  fn score_matvec_mult() void {
+     // Multi-group: gqa_group_size scores per batch
+     for (b in bsz) {
+         for (g in gqa_group_size) {
+             // Q[b,g] is kv_dim_p_pe elements at offset g*kv_dim_p_pe
+             // K[b] is the SAME kv_dim_p_pe x iter_num cache
+             score[g*bsz+b, :iter_num] = Q[b,g,:] @ K_cache[b,:,:]^T
+         }
+     }
-     Kt = dim_p_pe;  Nt = seq_len_p_pe;
+     Kt = kv_dim_p_pe;  Nt = iter_num;       // Reduced dim, variable length
-     vecmat_computation();
+     vecmat_computation_var_right_2d_strided(max_seq_len_p_pe);  // New GEMV variant
-     all_reduce_bsz_seq_len(px, ...);
+     all_reduce_bsz_g_var_seq_len_kv_head_scoped(...);  // New: KV-head-scoped reduce
  }
```

This delta introduces:
- Nested batch+group loops
- Variable-length strided GEMV (new kernel)
- KV-head-scoped reduction (new axis=3 mode, new reduce function)
- Score tensor grows to `[gqa_group_size x bsz x seq_len]`

### Base → WSE-3-GQA-cheng (+4 lines: DSD offset)

```diff
  fn score_matvec_mult() void {
+     var xk_offset: i16 = kv_head_id * dim_p_pe * seq_len_p_pe;
+     right_matrix_dsd = @set_dsd_base_addr(right_matrix_dsd, ptr_XKCache);
+     right_matrix_dsd = @increment_dsd_offset(right_matrix_dsd, xk_offset, f16);
      Kt = dim_p_pe;  Nt = seq_len_p_pe;     // Same dims as base
-     vecmat_computation();
+     vecmat_computation_dsd();                // DSD-snapshot variant
      all_reduce_bsz_seq_len(px, ...);        // Same global reduce
  }
```

Minimal change: offset into replicated cache, swap to DSD-snapshot GEMV.

---

## 8. Delta: Softmax

### Base → WSE-3-GQA (new function name, group-extended scalars)

```diff
- fn softmax_score() void {
+ fn softmax_score_all_groups() void {
+     // Resize DSDs to iter_num (variable-length)
-     all_reduceMax_bsz(py, ...);
+     all_reduceMax_bsz_g(py, ...);  // NEW: reduces gqa_group_size * bsz scalars
-     all_reduce_bsz(py, ...);
+     all_reduce_bsz_g(py, ...);     // NEW: reduces gqa_group_size * bsz scalars
  }
```

New `local_max_g[gqa_group_size * bsz]` and `local_sum_g[gqa_group_size * bsz]` buffers, plus two new communication functions.

### Base → WSE-3-GQA-cheng (unchanged)

Softmax is **identical to the base** — no delta.

---

## 9. Delta: Output GEMV

### Base → WSE-3-GQA (new function, multi-group)

```diff
- fn output_matvec_mult() void {
+ fn output_matvec_mult_all_groups() void {
+     for (b in bsz) {
+         for (g in gqa_group_size) {
+             output[b, g*kv_dim_p_pe:(g+1)*kv_dim_p_pe] = score[g,b,:] @ V[b,:,:]
+         }
+     }
-     Kt = seq_len_p_pe;
+     Kt = iter_num;  // Variable-length
  }
```

Each group produces `kv_dim_p_pe` elements; concatenated = `dim_p_pe`. V cache reused across groups.

### Base → WSE-3-GQA-cheng (+3 lines: DSD offset)

```diff
  fn output_matvec_mult() void {
+     var xv_offset: i16 = kv_head_id * seq_len_p_pe * dim_p_pe;
+     right_matrix_dsd = @set_dsd_base_addr(right_matrix_dsd, ptr_XVCache);
+     right_matrix_dsd = @increment_dsd_offset(right_matrix_dsd, xv_offset, f16);
-     vecmat_computation();
+     vecmat_computation_dsd();
  }
```

---

## 10. Delta: KV Cache Management

### Base: No Cache Management

K/V cache is pre-filled by the host via H2D memcpy. Decode reads it but never writes.

### Base → WSE-3-GQA (+50 lines: full autoregressive caching)

```diff
+ var step: i16 = 0;
+ var iter_num: i16 = prefill_len_p_pe;  // NEW: start from prefill
+
+ fn process_kv() void {
+     if (py == step % P) {
+         // Write kv_dim_p_pe elements to cache at position iter_num
+         range_copy_2d_strided(&XKCache_tile[iter_num], &QKV_tile[K_offset],
+             kv_dim_p_pe, ...);
+         range_copy_2d_strided(&XVCache_tile[iter_num*kv_dim_p_pe], &QKV_tile[V_offset],
+             kv_dim_p_pe, ...);
+         iter_num += 1;
+     }
+ }
+
+ fn range_copy_2d_strided(...) void { ... }  // ~30 lines helper
```

Also changes all attention GEMVs to use `Nt = iter_num` instead of `seq_len_p_pe`, and adds `all_reduce_bsz_var_seq_len_strided()`.

### Base → WSE-3-GQA-cheng (no delta)

No KV cache management. Same as base: cache pre-filled, read-only.

---

## 11. Delta: Communication Library

### Base comm_pe.csl (1035 lines, 7 functions, 2 axes)

| Function | Data size |
|----------|-----------|
| `all_reduce_bsz` | `[bsz]` |
| `all_reduceMax_bsz` | `[bsz]` |
| `all_reduce_bsz_dim` | `[bsz * dim_p_pe]` |
| `all_reduce_bsz_dim_QKV_fusion` | `[bsz * 3 * dim_p_pe]` |
| `all_reduce_bsz_seq_len` | `[bsz * seq_len_p_pe]` |
| `all_reduce_bsz_ffn_dim_ZZ_fusion` | `[bsz * 2 * ffn_dim_p_pe]` |
| `reconfig_allreduce_axis` | axis 0 (X), 1 (Y) |

### Base → WSE-3-GQA (+673 lines, +5 functions, +1 axis)

| Added function | Data size | Purpose |
|---------------|-----------|---------|
| `all_reduce_bsz_g` | `[gqa_group_size * bsz]` | Group-extended softmax sum |
| `all_reduceMax_bsz_g` | `[gqa_group_size * bsz]` | Group-extended softmax max |
| `all_reduce_bsz_g_var_seq_len_kv_head_scoped` | `[gqa_group_size * bsz * iter_num]` | Head-scoped variable-length score |
| `all_reduce_bsz_var_seq_len_strided` | `[bsz * iter_num]` | Variable-length strided |
| `reconfig_allreduce_axis(3)` | — | KV-head-scoped X-axis mode |

Also modified: `all_reduce_bsz_dim_QKV_fusion` handles asymmetric `[dim + 2*kv_dim]` size.

The `comm_layout.csl` grows by 170 lines (+42%) for KV-head-scoped routing topology.

### Base → WSE-3-GQA-cheng (+51 lines, +1 function)

| Added function | Data size | Purpose |
|---------------|-----------|---------|
| `all_reduce_bsz_g` | `[gqa_group_size * bsz]` | (placeholder, ~50 lines) |

The communication library is essentially **unchanged** from the base. `comm_layout.csl` is +1 line.

---

## 12. Delta: GEMV Primitives

### Base: 1 variant

```csl
fn vecmat_computation() void {
    for (b in bsz) {
        right_matrix_dsd = @set_dsd_base_addr(right_matrix_dsd, ptr_right_matrix);
        @map(gemv_static_step, left_vector_dsd);
        // advance offsets
    }
}
```

### Base → WSE-3-GQA: +2 variants

1. **`vecmat_computation_var_right(width_right)`**: Right matrix advances by `width_right` per batch (not `Nt`). For output GEMV where `Kt = iter_num` varies.

2. **`vecmat_computation_var_right_2d_strided(right_stride)`**: Processes 1 left-vector element at a time, stepping K cache with stride `max_seq_len_p_pe`. For score GEMV (`Q @ K^T`).

### Base → WSE-3-GQA-cheng: +1 variant

**`vecmat_computation_dsd()`**: Snapshots the pre-offset DSD and restores per batch row:

```csl
fn vecmat_computation_dsd() void {
    right_matrix_dsd_snap = right_matrix_dsd;    // Preserve runtime offset
    for (b in bsz) {
        right_matrix_dsd = right_matrix_dsd_snap; // Restore (not reset to base ptr)
        @map(gemv_static_step, left_vector_dsd);
    }
}
```

This is needed because the standard `vecmat_computation()` does `@set_dsd_base_addr(right_matrix_dsd, ptr_right_matrix)` which resets to the array base, losing the runtime `kv_head_id` offset.

---

## 13. Delta: RoPE

### Base: Enabled

`xq_rope()` and `xk_rope()` are called in `decode_struct()`.

### Base → WSE-3-GQA: Commented out

```csl
// xq_rope();  // TODO: strided-DSD @fmulh bug
// xk_rope();
```

Functions still defined but not called. A regression from the base.

### Base → WSE-3-GQA-cheng (full): No change

RoPE remains enabled. Same as base.

---

## 14. Delta: Bug Fixes

### SILU Iteration Bug (inherited from base)

The base has this bug:
```csl
fn z2_silu() void {
    for (@range(i16, bsz*dim_p_pe)) |i| {  // BUG: should be bsz*ffn_dim_p_pe
        z2_val = ZZ_tile[i];                 // BUG: index 0 is z1 region, not z2
        ZZ_tile[i] = z2_val / (1.0 + fast_exp(-z2_val));
    }
}
```

- **WSE-3-GQA**: Bug **inherited unchanged**
- **WSE-3-GQA-cheng (full)**: Bug **fixed**:
  ```csl
  const offset: i16 = bsz * ffn_dim_p_pe;
  for (@range(i16, bsz * ffn_dim_p_pe)) |i| {
      var idx: i16 = offset + i;
      ZZ_tile[idx] = silu(ZZ_tile[idx]);
  }
  ```

### DOWN Weight Shape (inherited from base)

Base: `var DOWN_weight_tile: [dim_p_pe * ffn_dim_p_pe]f16;`

- **WSE-3-GQA**: Inherited unchanged
- **WSE-3-GQA-cheng (full)**: Fixed to `[ffn_dim_p_pe * dim_p_pe]f16` (semantic, same buffer size)

### QKV Input

Base: uses `ptr_X_norm` (correct). Both GQA variants use `ptr_X_norm` in their main variants. GQA-cheng's draft variant (`decode_gqa.csl`) has a regression using `ptr_X` instead.

---

## 15. Decode Pipeline: Three-Way Comparison

```
Step  Base (WSE-3)                 WSE-3-GQA                          WSE-3-GQA-cheng (full)
----  -----------                  ----------                         ----------------------
 1    rmsnorm_x()                  rmsnorm_x()                        rmsnorm_x()
 2    xq [Kt=dim, Nt=dim]         xq [Kt=dim, Nt=dim]               xq [Kt=dim, Nt=dim]
 3    xk [Kt=dim, Nt=dim]         xk [Kt=dim, Nt=KV_DIM]            xk [Kt=dim, Nt=dim, OFFSET]
 4    xv [Kt=dim, Nt=dim]         xv [Kt=dim, Nt=KV_DIM]            xv [Kt=dim, Nt=dim, OFFSET]
 5    reduce QKV [3*dim]           reduce QKV [dim+2*kv_dim]          reduce QKV [3*dim]
 6    reconfig(0)                  reconfig(0)                        reconfig(0)
 7    xq_rope()                    -- (commented out)                 xq_rope()
 8    xk_rope()                    -- (commented out)                 xk_rope()
 9    --                           PROCESS_KV() [NEW]                 --
10    --                           RECONFIG(3) [NEW: head-scoped]     --
11    score [dim x seq, fixed]     score [MULTI-GRP, VAR-LEN, SCOPED] score [dim x seq, OFFSET]
12    reconfig(1)                  reconfig(1)                        reconfig(1)
13    softmax [bsz scalars]        softmax [GRP*BSZ SCALARS, VAR-LEN] softmax [bsz scalars]
14    output [seq x dim, fixed]    output [MULTI-GRP, VAR-LEN]        output [seq x dim, OFFSET]
15    reconfig(0)                  reconfig(0)                        reconfig(0)
16    o_matvec                     o_matvec                           o_matvec
17    residual                     residual                           residual
18    reconfig(1)                  reconfig(1)                        reconfig(1)
19    rmsnorm_z                    rmsnorm_z                          rmsnorm_z
20    up + gate                    up + gate                          up + gate
21    reduce ZZ                    reduce ZZ                          reduce ZZ
22    z2_silu [BUG]                z2_silu [BUG inherited]            z2_silu [FIXED]
23    z3, down, residual           z3, down, residual                 z3, down, residual
24    reconfig(1)                  reconfig(1)                        reconfig(1)
25    --                           step += 1                          --
```

**CAPS** = changed from base. Steps 9-14 show the fundamental divergence.

---

## 16. Configuration & Host-Side

### Base
- `launch_sim.py`: 336 lines, minimal validation
- Configs: Llama-8B, Qwen-72B (all with `n_heads ≈ n_kv_heads`, i.e., no actual GQA)
- No prefill parameter

### WSE-3-GQA
- `launch_sim.py`: **~2000 lines** (+1664, 5x larger)
  - Full Python reference implementations for every CSL function
  - Intermediate snapshot validation (7 export symbols)
  - GQA-aware tensor reshaping with asymmetric K/V dims
  - Prefill cache initialization
- Configs: Llama-8B with **true 4:1 GQA** (`n_heads=32, n_kv_heads=8`)
- New validation: `group_num % n_heads == 0`, `group_num % n_kv_heads == 0`

### WSE-3-GQA-cheng
- `launch_sim_gqa_full.py`: **~100 lines** (smaller than base)
  - Minimal: load config, init runtime, run
  - No reference validation, no intermediate snapshots
- 1 test config: `P=8, n_heads=4, n_kv_heads=2` (2:1 GQA)

---

## 17. Code Complexity Summary

| Metric | Base | WSE-3-GQA | Delta | GQA-cheng | Delta |
|--------|------|-----------|-------|-----------|-------|
| decode.csl | 965 | 1282 | **+317 (+33%)** | 942 | **-23 (-2%)** |
| comm_pe.csl | 1035 | 1708 | **+673 (+65%)** | 1086 | **+51 (+5%)** |
| comm_layout.csl | 405 | 575 | **+170 (+42%)** | 406 | **+1 (+0%)** |
| layout.csl | 114 | 131 | +17 (+15%) | 120 | +6 (+5%) |
| **Total CSL** | **2519** | **3696** | **+1177 (+47%)** | **2554** | **+35 (+1%)** |
| launch_sim.py | 336 | ~2000 | +1664 | ~100 | -236 |
| GEMV variants | 1 | 3 | +2 | 2 | +1 |
| Reduce functions | 7 | ~12 | +5 | ~8 | +1 |
| Axis modes | 2 | 3 | +1 | 2 | 0 |
| Model configs | MHA only | True GQA (4:1) | — | Test GQA (2:1) | — |
| Debug exports | 1 | 7 | +6 | 1 | 0 |

---

## 18. Host-Side Data Structure Handling (Dimensional Approach)

The dimensional approach (WSE-3-GQA) requires significant host-side preprocessing that goes well beyond the base's straightforward reshape-transpose-reshape tiling. This section documents what changes and why.

### 18.1 The Base Pattern: Uniform Tiling

In the base implementation, **all** weight matrices follow an identical tiling pattern. Q, K, V, and O weights are all `[dim, dim]`, and each is distributed to PEs using:

```python
# Base: every weight uses the same pattern
tensor = np.random.rand(dim, dim).astype(np.float16)
tiled = tensor.reshape(P, dim_p_pe, P, dim_p_pe)    # split into PE grid
tiled = tiled.transpose(0, 2, 1, 3)                  # [P, P, dim_p_pe, dim_p_pe]
tiled = tiled.reshape(P, P, dim_p_pe * dim_p_pe)     # flatten per-PE block
# → PE(py, px) gets a [dim_p_pe × dim_p_pe] square block
```

KV caches are tiled the same way: `XKCache [dim, seq_len]` becomes `[P, P, dim_p_pe * seq_len_p_pe]`. Every PE stores the same-shaped block. No special logic needed.

### 18.2 Dimensional GQA: Asymmetric K/V Weight Tiling

With GQA, K and V weight matrices shrink from `[dim, dim]` to `[dim, kv_dim]` where `kv_dim = n_kv_heads * head_dim`. The tiling function generalizes to handle asymmetric dimensions:

```python
def tile_weight_row(W, rows, cols):
    """W: [rows*P, cols*P] → PE tile [P, P, rows*cols]"""
    return W.reshape(P, rows, P, cols).transpose(0, 2, 1, 3).reshape(P, P, rows * cols)

Q_tile = tile_weight_row(W_Q_perm,        dim_p_pe, dim_p_pe)     # [P, P, dim_p_pe²]
K_tile = tile_weight_row(tensor_k_weight,  dim_p_pe, kv_dim_p_pe)  # [P, P, dim_p_pe * kv_dim_p_pe]
V_tile = tile_weight_row(tensor_v_weight,  dim_p_pe, kv_dim_p_pe)  # [P, P, dim_p_pe * kv_dim_p_pe]
O_tile = tile_weight_row(W_O_perm,         dim_p_pe, dim_p_pe)     # [P, P, dim_p_pe²]
```

**Why this is needed**: On each PE, the K and V projection output has only `kv_dim_p_pe` columns (not `dim_p_pe`). If you tiled K/V as `[dim_p_pe × dim_p_pe]` blocks like the base, each PE would hold 4× too much data (for 4:1 GQA), waste memory, and the GEMV output dimensions wouldn't match the score/cache dimensions.

For Llama-8B (P=256): K/V tiles are `[16 × 4]` = 64 elements per PE, vs `[16 × 16]` = 256 in the base — a 4× reduction.

### 18.3 Q and O Weight Offline Column Permutation

This is the most subtle and important host-side change. The dimensional approach applies an **offline column permutation** to both Q and O weight matrices before tiling.

**The problem**: In standard Llama weight layout, Q columns are ordered by Q-head: `[head_0 | head_1 | ... | head_31]`. Each head occupies `head_dim` = 128 contiguous columns. But in the dimensional approach, the on-chip Q output vector (`dim_p_pe` elements per PE) is conceptually split into `gqa_group_size` groups of `kv_dim_p_pe` each. Each group must align with the corresponding KV head's cache. The standard column order doesn't satisfy this — columns belonging to Q-heads that share a KV-head may be scattered across different PE column ranges.

**The permutation** (lines 479–489 in `launch_sim.py`):

```python
W_Q_perm = np.zeros((dim, dim), dtype=np.float16)
W_O_perm = np.zeros((dim, dim), dtype=np.float16)
for _h in range(n_heads):
    _kv_head = _h // gqa_group_size        # which KV head this Q head maps to
    _g = _h % gqa_group_size               # position within the sharing group
    for _s in range(pes_p_kv_head):
        _old_col = _h * head_dim + _s * kv_dim_p_pe
        _new_col = _kv_head * pes_p_kv_head * dim_p_pe + _s * dim_p_pe + _g * kv_dim_p_pe
        W_Q_perm[:, _new_col:_new_col + kv_dim_p_pe] = tensor_q_weight[:, _old_col:_old_col + kv_dim_p_pe]
        W_O_perm[_new_col:_new_col + kv_dim_p_pe, :] = tensor_o_weight[_old_col:_old_col + kv_dim_p_pe, :]
```

**What this achieves**: After permutation, within each PE's `dim_p_pe`-element Q output vector, the elements are arranged as `[group_0 | group_1 | ... | group_{gqa_group_size-1}]`, each of size `kv_dim_p_pe`. Group `g` contains the Q elements from Q-head `kv_head * gqa_group_size + g`, restricted to this PE's column range. This means the nested `for(b) for(g)` loop in the on-chip `score_matvec_mult()` can simply slice contiguous `kv_dim_p_pe`-element chunks from Q — no runtime index mapping needed.

**Concrete example** (n_heads=4, n_kv_heads=2, gqa_group_size=2, P=4):

```
Standard Q layout:   [Q_h0 | Q_h1 | Q_h2 | Q_h3]     (each head_dim=128 cols)
                     KV0 shares h0,h1    KV1 shares h2,h3

Permuted Q layout:   [KV0: h0_seg0,h1_seg0 | h0_seg1,h1_seg1 | KV1: h2_seg0,h3_seg0 | h2_seg1,h3_seg1]
                                ↑                    ↑
                          PE px=0's portion     PE px=1's portion
```

After tiling, PE(py, px) holds `dim_p_pe` Q-output elements where groups `g=0..gqa_group_size-1` are contiguous `kv_dim_p_pe`-element slices — exactly matching the K cache layout on that PE.

**Why O needs the same permutation**: The output projection `O_proj = attention_output @ W_O` is the inverse operation. The attention output vector is structured the same way as Q (groups of `kv_dim_p_pe` per PE), so O's rows must be reordered to match. This is the transpose of the Q column permutation.

**Why GQA-cheng doesn't need this**: In the replicated approach, Q/K/V all keep `dim_p_pe` per PE. The PE selects the correct KV head via runtime offset (`kv_head_id = quotient_x % n_kv_heads`), so the standard weight layout works unchanged.

### 18.4 KV Cache: Interleaved Tiling with Partial Pre-fill

The base tiles KV caches with the same reshape-transpose-reshape as weights. The dimensional approach uses custom interleaved tiling functions because:

1. **Asymmetric cache dimensions**: K cache is `[bsz, kv_dim, max_seq_len]` (not `[dim, seq_len]`), requiring `kv_dim_p_pe` per PE instead of `dim_p_pe`.
2. **Batch dimension**: Each batch element has its own cache slice.
3. **Partial pre-fill**: Only the first `prefill_len` tokens are filled; the rest are zeros (for autoregressive generation).
4. **Interleaved token distribution**: Token `t` is assigned to PE row `py = t % P`. Tokens are not contiguous per PE — they are strided by `P`.

```python
def tile_kcache_interleaved(K_cache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, prefill_len_p_pe):
    """
    K_cache: [bsz, kv_dim, max_seq_len]
    On-chip layout per PE: [bsz, kv_dim_p_pe, max_seq_len_p_pe]
    PE(py, px) gets tokens {py, py+P, py+2P, ...} and kv features [px*kv_dim_p_pe : (px+1)*kv_dim_p_pe]
    """
    tile = np.zeros((P, P, bsz * kv_dim_p_pe * max_seq_len_p_pe), dtype=np.float16)
    for b in range(bsz):
        for py in range(P):
            for px in range(P):
                for s in range(prefill_len_p_pe):
                    token = py + s * P    # interleaved: PE py owns tokens py, py+P, py+2P, ...
                    k_row_start = px * kv_dim_p_pe
                    for k in range(kv_dim_p_pe):
                        tile[py, px, b*kv_dim_p_pe*max_seq_len_p_pe + k*max_seq_len_p_pe + s] = \
                            K_cache[b, k_row_start + k, token]
    return tile
```

**Why interleaving**: In the score GEMV (`Q @ K^T`), the reduction is over the `kv_dim` axis (X-axis PEs). Each PE row `py` must hold a distinct set of sequence positions so that the Y-axis reduction for softmax operates on different tokens. Interleaving by `P` ensures maximum parallelism: PE row 0 has tokens {0, P, 2P, ...}, PE row 1 has {1, P+1, 2P+1, ...}, etc.

**Why the base doesn't need this**: The base uses `XKCache[dim, seq_len]` which tiles cleanly with reshape-transpose because both axes divide evenly by P and there's no batch dimension or partial fill.

### 18.5 V Cache Layout Difference

V cache uses a transposed layout compared to K:

```
K cache on-chip: [bsz, kv_dim_p_pe, max_seq_len_p_pe]  — feature-major (for Q @ K^T)
V cache on-chip: [bsz, max_seq_len_p_pe, kv_dim_p_pe]  — sequence-major (for score @ V)
```

This matches the GEMV access pattern: score computation reads K row-by-row along the feature dimension, while output computation reads V row-by-row along the sequence dimension.

### 18.6 Host-Side Memcpy Size Differences

The asymmetric dimensions propagate to H2D transfer sizes:

| Symbol | Base `count_per_pe` | WSE-3-GQA `count_per_pe` | Ratio |
|--------|-------------------|-------------------------|-------|
| Q_weight | `dim_p_pe * dim_p_pe` | `dim_p_pe * dim_p_pe` | 1x |
| K_weight | `dim_p_pe * dim_p_pe` | `dim_p_pe * kv_dim_p_pe` | **1/4** (for 4:1 GQA) |
| V_weight | `dim_p_pe * dim_p_pe` | `dim_p_pe * kv_dim_p_pe` | **1/4** |
| O_weight | `dim_p_pe * dim_p_pe` | `dim_p_pe * dim_p_pe` | 1x |
| XKCache | `dim_p_pe * seq_len_p_pe` | `bsz * kv_dim_p_pe * max_seq_len_p_pe` | varies |
| XVCache | `seq_len_p_pe * dim_p_pe` | `bsz * max_seq_len_p_pe * kv_dim_p_pe` | varies |

### 18.7 Summary: Where the Complexity Lives

The dimensional approach trades **host-side preprocessing complexity** for **on-chip memory efficiency**:

| Host-side operation | Base | WSE-3-GQA | Why needed |
|---|---|---|---|
| Weight tiling | Uniform reshape-transpose | Generalized asymmetric `tile_weight_row()` | K/V have fewer output columns than Q |
| Q/O weight permutation | None | Offline column reorder (4 nested loops) | Align Q group structure with KV-head PE mapping |
| KV cache tiling | Standard reshape-transpose | Custom `tile_kcache_interleaved()` / `tile_vcache_interleaved()` | Asymmetric dims + batch + partial fill + interleaved tokens |
| Memcpy sizes | All symmetric | K/V transfers smaller than Q/O | Per-PE K/V buffers are genuinely smaller |

The key insight is that this complexity is **paid once at initialization** (offline, on the host CPU) but yields **continuous benefits during inference** (smaller on-chip buffers, less communication volume, and GQA-ratio memory savings on every PE).

---

## 19. Key Findings

1. **Radically different delta sizes**: WSE-3-GQA adds +47% CSL code to the base; WSE-3-GQA-cheng adds +1%. The approaches are not incrementally different — they represent fundamentally different engineering trade-offs.

2. **WSE-3-GQA restructures dimensions** (compile-time), requiring changes across the entire stack: decode logic, GEMV kernels, communication functions, routing topology, and the host launcher. The payoff is optimal memory efficiency — K/V storage shrinks by the GQA ratio.

3. **WSE-3-GQA-cheng preserves structure** (runtime offset), requiring only local changes to K/V-touching functions plus one new GEMV variant. The cost is replicated memory — every PE stores all KV heads.

4. **Communication library is the biggest delta**: WSE-3-GQA's comm_pe.csl grows by 673 lines (+65%) for KV-head-scoped reductions and group-extended scalar operations. This is the single largest engineering investment. WSE-3-GQA-cheng's comm_pe.csl grows by only 51 lines (+5%).

5. **WSE-3-GQA-cheng fixes base bugs** (SILU iteration, DOWN weight shape) that WSE-3-GQA inherits. These fixes should be backported.

6. **WSE-3-GQA adds features absent from both the base and GQA-cheng**: KV cache management (`process_kv()`), variable-length attention, prefill support, and comprehensive debug instrumentation. These are orthogonal to the GQA strategy and represent separate engineering effort.

7. **WSE-3-GQA regresses RoPE** (commented out due to strided-DSD bug) while WSE-3-GQA-cheng preserves it from the base.

8. **For production**: WSE-3-GQA's dimensional approach is the right long-term strategy (memory efficiency scales with GQA ratio), but WSE-3-GQA-cheng's bug fixes and RoPE should be integrated. WSE-3-GQA-cheng is valuable as a rapid prototyping path — its minimal delta makes it easy to test GQA correctness before investing in the full dimensional refactor.
