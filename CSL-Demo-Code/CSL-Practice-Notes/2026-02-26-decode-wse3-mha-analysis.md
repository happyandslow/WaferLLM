# WaferLLM Decode WSE-3 MHA — Deep Analysis
## Date: 2026-02-26

---

## Overview

This document analyzes the MHA decode kernel at `Decode/WSE-3` in detail, covering:
1. PE grid and parameter derivation
2. Data partitioning across PEs and in-memory tensor layouts
3. Attention algorithm steps and communication pattern
4. CSL primitives used and why they are efficient
5. What changes for GQA

---

## 1. PE Grid & Parameter Derivation

The system places a **P×P grid** of Processing Elements on the chip.
Each PE has integer coordinates `(px, py)` where `0 <= px, py < P`.

From the JSON config (`run_sim.sh` derivation):

| Compile-time param | Formula | Meaning |
|----|----|----|
| `dim_p_pe`       | `dim / P`           | Hidden-dim slice per PE |
| `pes_p_head`     | `P / n_heads`       | PEs assigned to one Q head |
| `pes_p_kv_head`  | `P / n_kv_heads`    | PEs assigned to one KV head |
| `head_dim_p_pe`  | `head_dim / P`      | Head-dim slice per PE |
| `seq_len_p_pe`   | `seq_len / P`       | Sequence-length slice per PE |
| `ffn_dim_p_pe`   | `ffn_dim / P`       | FFN-dim slice per PE |
| `pe_num_p_group` | `P / group_num`     | PEs per communication group |
| `root_1st_phase` | `pe_num_p_group / 2` | Intra-group reduce root |
| `root_2nd_phase` | `(group_num/2) * pe_num_p_group + root_1st_phase` | Inter-group reduce root |

**Key constraint**: All dimensions (`dim`, `seq_len`, `ffn_dim`, `head_dim`) must be divisible by P.

---

## 2. Data Layout per PE

Every PE holds a **slice** of all tensors, identified by `px` and `py`.
All data is f16. Arrays are 1-D in memory; index order matters.

### 2a. Weight Matrices (model parameters, loaded once from host)

Each weight matrix `[in_dim, out_dim]` is partitioned **both input and output** directions across PEs:
- Each PE holds a `[dim_p_pe × dim_p_pe]` tile of the full `[dim × dim]` weight matrix.

```
Q_weight_tile : [dim_p_pe * dim_p_pe]f16
K_weight_tile : [dim_p_pe * dim_p_pe]f16   <- same shape as Q in MHA
V_weight_tile : [dim_p_pe * dim_p_pe]f16   <- same shape as Q in MHA
O_weight_tile : [dim_p_pe * dim_p_pe]f16
UP_weight_tile  : [dim_p_pe * ffn_dim_p_pe]f16
GATE_weight_tile: [dim_p_pe * ffn_dim_p_pe]f16
DOWN_weight_tile: [dim_p_pe * ffn_dim_p_pe]f16   (note: [ffn_dim_p_pe * dim_p_pe])
```

**In-memory layout (row-major)**: `[input_features_p_pe, output_features_p_pe]`
i.e., consecutive elements in memory are adjacent output features for the same input feature.

### 2b. KV Cache (loaded from host each decode step in the no-KV-management version)

```
XKCache_tile : [dim_p_pe * seq_len_p_pe]f16   <- K transposed: K^T[dim, seq_len]
XVCache_tile : [seq_len_p_pe * dim_p_pe]f16   <- V natural:    V[seq_len, dim]
```

- `XKCache` is stored **transposed**: layout is `[dim_p_pe, seq_len_p_pe]`, i.e., for each of
  the `dim_p_pe` key-feature dimensions, we have `seq_len_p_pe` cached values across tokens.
  This allows `Q[1, dim_p_pe] @ K^T[dim_p_pe, seq_len_p_pe]` as a contiguous row-major GEMV.

- `XVCache` is stored **naturally**: layout is `[seq_len_p_pe, dim_p_pe]`, i.e., for each of
  the `seq_len_p_pe` tokens, we have `dim_p_pe` value features.
  This allows `score[1, seq_len_p_pe] @ V[seq_len_p_pe, dim_p_pe]` as a contiguous row-major GEMV.

### 2c. Activations (per decode step)

```
X_tile      : [bsz * dim_p_pe]f16    <- input token (dim partitioned along both px and py axes)
X_norm_tile : [bsz * dim_p_pe]f16    <- RMSNorm-normalized X
X_tmp       : [bsz * dim_p_pe]f16    <- temporary for norm computation
W_tile      : [dim_p_pe]f16          <- RMSNorm weights (same shard as X)

QKV_tile    : [bsz * 3 * dim_p_pe]f16  <- fused Q, K, V buffer
  Q  @ offset 0            : [bsz, dim_p_pe]
  K  @ offset bsz*dim_p_pe : [bsz, dim_p_pe]
  V  @ offset 2*bsz*dim_p_pe: [bsz, dim_p_pe]

score       : [bsz * seq_len_p_pe]f16  <- Q*K^T attention scores
score_tmp   : [bsz * seq_len_p_pe]f16  <- temp for softmax computation
output_tile : [bsz * dim_p_pe]f16      <- attention output (after score @ V)
h1_tile     : [bsz * dim_p_pe]f16      <- output after O-proj
Z_tile      : [bsz * dim_p_pe]f16      <- residual stream

ZZ_tile     : [bsz * 2 * ffn_dim_p_pe]f16  <- fused gate+up projections
  z1 (up)   @ offset 0
  z2 (gate) @ offset bsz*ffn_dim_p_pe
z3_tile     : [bsz * ffn_dim_p_pe]f16   <- z1 * silu(z2)
z_norm_tile : [bsz * dim_p_pe]f16       <- FFN RMSNorm output
h2_tile     : [bsz * dim_p_pe]f16       <- down-proj output
```

---

## 3. The GEMV Primitive: `vecmat_computation()`

This is the core computation primitive. It implements:
```
out[1, N] += left[1, K] @ right[K, N]
```

**Mechanism** (decode.csl lines 184–202):

```csl
@load_to_dsr(dest_dsr_1, out_vector_dsd, .{ .save_address = false });
@load_to_dsr(src0_dsr_1, out_vector_dsd, .{ .save_address = false });
@load_to_dsr(src1_dsr_1, right_matrix_dsd, .{ .save_address = true });
@map(gemv_static_step, left_vector_dsd);  // iterates over K elements

fn gemv_static_step(cur: f16) void {
    @fmach(dest_dsr_1, src0_dsr_1, src1_dsr_1, cur);
}
// @fmach(dest, src0, src1, scalar):  dest[i] = src0[i] + scalar * src1[i]
```

For each scalar `cur` from the left vector:
- `@fmach` performs vector fused-multiply-add: `out[i] += cur * right[i]` for all N output positions.
- `src1_dsr_1` has `save_address=true` so it auto-advances by N after each call.
- This is **outer-product accumulation** (rank-1 updates), applied K times.

**Why efficient**: The MFU (Matrix Functional Unit) processes the entire N-element right-matrix row
in one SIMD instruction (`simd_64`), so effective throughput is `K × N` FMAs per call with
minimal loop overhead.

---

## 4. Attention Algorithm: Step-by-Step (`decode_struct()`)

The full decode step is in `decode_struct()` (lines 765–814). The axis of the all-reduce
is reconfigured dynamically between Y (column-wise) and X (row-wise) operations.

### Step 1: RMSNorm on X  (`rmsnorm_x()`)
- Compute `sum(x_i^2)` locally on each PE
- **All-reduce along Y-axis**: accumulate partial sums → global norm
- Normalize: `x_norm = x * W / sqrt(sum/dim + eps)`

### Step 2: Q, K, V projections + all-reduce
```
xq_matvec_mult():  QKV[0,  bsz*dim_p_pe) = X_norm @ Q_weight  (partial, along input dim)
xk_matvec_mult():  QKV[bsz*dim_p_pe, 2*bsz*dim_p_pe) = X_norm @ K_weight
xv_matvec_mult():  QKV[2*bsz*dim_p_pe, 3*bsz*dim_p_pe) = X_norm @ V_weight
```
- Each PE holds a `dim_p_pe`-slice of the input dimension; the weight tile connects that
  input slice to a `dim_p_pe` output slice. The dot product across the full input dimension
  is split across PEs; we need to **all-reduce along Y-axis** to sum partial products.
- **Optimization**: Q, K, V all-reduces are **fused into one** `all_reduce_bsz_dim_QKV_fusion`
  call operating on the entire `3*dim_p_pe` buffer at once, saving 2/3 of the reduce overhead.

After this step, each PE holds the **full** QKV for its output-dim shard.

### Step 3: Reconfigure all-reduce axis to X, apply RoPE
```
reconfig_allreduce_axis(0)  <- switch to X-axis for subsequent reductions
xq_rope()                   <- in-place RoPE on Q portion of QKV_tile
xk_rope()                   <- in-place RoPE on K portion of QKV_tile
```

RoPE applies: `q_out[2i] = q[2i]*cos - q[2i+1]*sin`,  `q_out[2i+1] = q[2i]*sin + q[2i+1]*cos`
This is done using strided DSDs (stride=2, even/odd interleaving) with 4 DSRs in parallel.

### Step 4: Attention scores = Q @ K^T  (`score_matvec_mult()`)
```
score[bsz, seq_len_p_pe] += Q[bsz, dim_p_pe] @ KCache[dim_p_pe, seq_len_p_pe]
comm_mod.all_reduce_bsz_seq_len(px, ...)   <- sum across X-axis
score *= alpha  (= 1/sqrt(head_dim))
```

- The Q vector is distributed across both X and Y axes (dim = P × dim_p_pe). After the
  Y-axis all-reduce in step 2, each PE has the same Q shard. But KCache is partitioned
  along dim (X-axis), so we reduce along X to get the full dot product.
- Result: each PE holds `score[bsz, seq_len_p_pe]` — the partial sequence scores.

### Step 5: Reconfigure to Y-axis, Softmax  (`softmax_score()`)
```
reconfig_allreduce_axis(1)
local max → all_reduceMax along Y-axis → shift and exp
local sum → all_reduce along Y-axis → normalize
```

- Scores are partitioned along seq_len (each PE holds `seq_len_p_pe` scores).
- To compute numerically stable softmax, need global max and sum across all PEs.
- Both reductions are **along Y-axis** (since scores for the same query head span Y).

### Step 6: Attention output = softmax(scores) @ V  (`output_matvec_mult()`)
```
output[bsz, dim_p_pe] += score[bsz, seq_len_p_pe] @ VCache[seq_len_p_pe, dim_p_pe]
comm_mod.all_reduce_bsz_dim(py, ...)   <- sum across Y-axis
```

- Each PE has a `seq_len_p_pe` slice of scores, and the corresponding V slice.
- Reduce along Y-axis to sum partial V contributions.

### Step 7: Reconfigure to X-axis, Output projection  (`o_matvec_mult()`)
```
reconfig_allreduce_axis(0)
h1[bsz, dim_p_pe] += output[bsz, dim_p_pe] @ O_weight[dim_p_pe, dim_p_pe]
comm_mod.all_reduce_bsz_dim(px, ...)   <- sum across X-axis
```

### Step 8: Residual add
```
Z = X + h1
```

---

## 5. Hierarchical All-Reduce Communication Pattern

The all-reduce operates over a 1-D strip of PEs (either X-row or Y-column).
It uses a **two-phase tree reduce + broadcast**:

### Phase 1: Intra-group reduce (within `pe_num_p_group` PEs)
- PEs are split into groups of `pe_num_p_group`
- Each group reduces to the `root_1st_phase` PE within that group
- Uses **2 alternating colors** (0 and 1) to enable pipelining and avoid routing conflicts
  - Even-indexed PEs within a group use color 0 for one direction, color 1 for the other
  - Odd-indexed PEs use reversed colors
  - This "butterfly" pattern enables all PEs in the group to communicate simultaneously

### Phase 2: Inter-group reduce (across groups)
- Only group roots (`remainder == root_1st_phase`) participate
- Reduce to `root_2nd_phase` across the P-wide axis
- Again uses 2 alternating colors

### Broadcast
- `root_2nd_phase` broadcasts result back to all PEs using a spanning tree (EAST+WEST or NORTH+SOUTH)

### Axis reconfiguration
Routes are set at compile-time as named structs (`X_1st_RD_0`, `Y_2nd_RD_1`, `X_BD`, etc.),
and `color_config.reset_routes()` swaps between X and Y configurations at runtime.
This is a WSE-3 feature (`@initialize_queue` + dynamic route reset).

---

## 6. CSL Primitives Reference

| Primitive | Operation | Notes |
|---|---|---|
| `@fmach(dst, src0, src1, scalar)` | `dst[i] = src0[i] + scalar * src1[i]` | Core GEMV step |
| `@fmulh(dst, src0, src1)` | `dst[i] = src0[i] * src1[i]` | Element-wise multiply |
| `@faddh(dst, src0, src1)` | `dst[i] = src0[i] + src1[i]` | Element-wise add |
| `@fsubh(dst, src0, src1)` | `dst[i] = src0[i] - src1[i]` | Element-wise subtract |
| `@fmaxh(ptr, init, src)` | `*ptr = max(init, max(src[i]))` | Reduction max |
| `@faddh(ptr, init, src)` | `*ptr = sum(src[i])` | Reduction sum (scalar) |
| `@fmovh(dst, scalar)` | `dst[i] = scalar` | Fill with scalar |
| `@map(fn, src_dsd)` | `fn(src[i])` for each i | Element-wise with DSR side effects |
| `@map(fn, src_dsd, dst_dsd)` | `dst[i] = fn(src[i])` | Map with explicit output |
| `@mov16(dst, src)` | Copy between mem/fabric DSDs | Used for fabric send/receive |
| `@load_to_dsr(dsr, dsd)` | Bind DSR to DSD | Pre-loading avoids per-iteration overhead |

**simd_64**: All fabric DSDs use `.simd_mode = .{ .simd_64 = true }` — processes 2 f16 per cycle.

**DSR preloading pattern**: Inner loops preload DSRs once before `@map()`, letting the hardware
auto-advance addresses rather than recomputing per element. This is crucial for GEMV efficiency.

---

## 7. MHA vs GQA: What Must Change

### Current MHA assumptions:
- `n_heads == n_kv_heads` → `pes_p_head == pes_p_kv_head`
- `K_weight_tile: [dim_p_pe * dim_p_pe]` — same output dim as Q
- `V_weight_tile: [dim_p_pe * dim_p_pe]` — same output dim as Q
- `XKCache_tile: [dim_p_pe * seq_len_p_pe]` — kv_dim == dim
- `XVCache_tile: [seq_len_p_pe * dim_p_pe]` — kv_dim == dim

### GQA changes (n_heads > n_kv_heads, e.g., 32 Q heads, 8 KV heads):

Define: `kv_dim = n_kv_heads * head_dim` and `kv_dim_p_pe = kv_dim / P`

**Tensors that must change size:**
| Tensor | MHA | GQA |
|---|---|---|
| `K_weight_tile` | `[dim_p_pe * dim_p_pe]` | `[dim_p_pe * kv_dim_p_pe]` |
| `V_weight_tile` | `[dim_p_pe * dim_p_pe]` | `[dim_p_pe * kv_dim_p_pe]` |
| `XKCache_tile`  | `[dim_p_pe * seq_len_p_pe]` | `[kv_dim_p_pe * seq_len_p_pe]` |
| `XVCache_tile`  | `[seq_len_p_pe * dim_p_pe]` | `[seq_len_p_pe * kv_dim_p_pe]` |
| `QKV_tile`      | `[bsz * 3 * dim_p_pe]` | Q: `[bsz * dim_p_pe]`, KV: `[bsz * kv_dim_p_pe]` each |

**Computation steps that must change:**

1. `xk_matvec_mult()`: output size changes from `dim_p_pe` to `kv_dim_p_pe`
   - `Nt = kv_dim_p_pe` instead of `dim_p_pe`
2. `xv_matvec_mult()`: similarly `Nt = kv_dim_p_pe`
3. `all_reduce_bsz_dim_QKV_fusion`: cannot fuse Q+K+V in one reduce since sizes differ.
   - Option A: 2 separate reduces: one for Q (`dim_p_pe`), one fused KV (`2*kv_dim_p_pe`)
   - Option B: 3 separate reduces
4. `xk_rope()`: applies to K of size `kv_dim_p_pe` (fewer freqs needed)
5. `score_matvec_mult()`:
   - Input Q still has size `dim_p_pe`
   - K cache has size `kv_dim_p_pe × seq_len_p_pe`
   - **Problem**: Q and K have different dims — Q is the full-head-dim vector, K is the kv-head-dim vector
   - Resolution: the attention is computed PER HEAD. In MHA the entire `dim_p_pe` slice corresponds
     to one head's portion. In GQA, groups of Q-heads share one KV head.
   - The Q→score computation must be done `n_heads/n_kv_heads` times per KV head, OR
     the data must be rearranged so each subset of Q heads maps to the right KV head.
6. `output_matvec_mult()`: score @ V now uses `kv_dim_p_pe` instead of `dim_p_pe` for V

**Communication changes:**
- The QKV all-reduce fusion must handle the new sizes
- New fabric DSDs needed for `kv_dim_p_pe` extents
- The score computation may need more careful partitioning across heads

**What stays the same:**
- RMSNorm on X: no change
- O projection, FFN block: completely unchanged
- Score scaling, softmax, residual adds: unchanged
- All-reduce algorithm structure: unchanged (just need new DSD sizes)
- RoPE for Q: unchanged
- Layout, fabric routing, communication infrastructure: unchanged

---

## 8. WSE-3-KV: Incremental KV Cache Management

This section documents the changes introduced in `Decode/WSE-3-KV` versus the base `Decode/WSE-3`.
The KV variant **keeps the KV cache on-chip and updates it incrementally** rather than loading it
from the host each step. Everything unrelated to the KV cache (FFN, norms, O-proj, comm
infrastructure) is **identical**.

### 8a. Core Design Idea: Distributed KV Cache across Y-dimension

Each PE in a Y-column stores only **every P-th token's K and V**.
At decode step `t`, only the PE column `py = t % P` writes the new K/V pair into its cache.

```
step 0: py=0 PEs write token 0's K,V  → iter_num becomes 1 at py=0
step 1: py=1 PEs write token 1's K,V  → iter_num becomes 1 at py=1
...
step P-1: py=P-1 PEs write token P-1's K,V
step P:   py=0 PEs write token P's K,V   → iter_num becomes 2 at py=0
...
```

After T total tokens, PE column py holds tokens `{py, py+P, py+2P, ...}` → `T/P` tokens each.

**Why this works**: attention is `softmax(Q @ K^T / sqrt(d)) @ V`. The sum over all tokens
is linear (both numerator and denominator of softmax can be split across PE rows). The key
properties exploited:
- Each PE's score computation covers its owned tokens.
- Softmax max and sum are all-reduced across Y (global correctness).
- Each PE's output contribution `score_i * V_i` is summed via Y-axis all-reduce.

### 8b. New Runtime State

```csl
var step: i16 = 0;     // current global decode step (monotonically increasing)
var iter_num: i16 = 0; // how many tokens this PE's cache holds (only incremented when py == step % P)
```

### 8c. Parameter Change

| WSE-3 | WSE-3-KV | Meaning |
|---|---|---|
| `seq_len_p_pe` | `max_seq_len_p_pe` | Now the MAXIMUM allocated capacity, not the actual current length |

The config key also changes: `seq_len` → `max_seq_len`.

### 8d. KV Cache Layout (now includes bsz, allocated at max capacity)

```csl
// WSE-3 (no bsz, full allocation):
XKCache_tile : [dim_p_pe * seq_len_p_pe]f16           // [dim_p_pe, seq_len_p_pe]  (K^T)
XVCache_tile : [seq_len_p_pe * dim_p_pe]f16           // [seq_len_p_pe, dim_p_pe]  (V)

// WSE-3-KV (with bsz, max allocation):
XKCache_tile : [bsz * dim_p_pe * max_seq_len_p_pe]f16 // [bsz, dim_p_pe, max_seq_len_p_pe]  (K^T per batch)
XVCache_tile : [bsz * max_seq_len_p_pe * dim_p_pe]f16 // [bsz, max_seq_len_p_pe, dim_p_pe]  (V per batch)
```

The innermost arrangement is unchanged (K is still transposed, V is still row-major), but now:
- A batch dimension is prepended to both caches.
- All `max_seq_len_p_pe` slots are allocated upfront; only `iter_num` are valid at runtime.

### 8e. `process_kv()` — The KV Write Function

Called once per decode step, **after the QKV all-reduce** (so all PEs have the complete K and V):

```csl
fn process_kv() void {
    var remain: i16 = step % P;
    if (py == remain) {

        // Write K (from QKV_tile[bsz*dim_p_pe:]) to column `iter_num` of KCache
        // KCache layout: [bsz, dim_p_pe, max_seq_len_p_pe] → write one column (stride max_seq_len_p_pe)
        range_copy_2d_strided(
            &XKCache_tile[iter_num],             // dest: column iter_num (stride=max_seq_len_p_pe)
            &QKV_tile[bsz * dim_p_pe],           // src: K in QKV buffer
            dim_p_pe,                             // copy dim_p_pe elements per batch
            dim_p_pe * max_seq_len_p_pe,          // dest outer stride: next batch's K^T block
            max_seq_len_p_pe,                     // dest inner stride: column-stride in K^T
            dim_p_pe,                             // src outer stride: next batch's K chunk
            1                                     // src inner stride: contiguous
        );

        // Write V (from QKV_tile[2*bsz*dim_p_pe:]) to row `iter_num` of VCache
        // VCache layout: [bsz, max_seq_len_p_pe, dim_p_pe] → write one row (stride=1)
        range_copy_2d_strided(
            &XVCache_tile[iter_num * dim_p_pe],   // dest: row iter_num (offset iter_num*dim_p_pe)
            &QKV_tile[2 * bsz * dim_p_pe],        // src: V in QKV buffer
            dim_p_pe,                              // copy dim_p_pe elements per batch
            dim_p_pe * max_seq_len_p_pe,           // dest outer stride: next batch's V block
            1,                                     // dest inner stride: row-contiguous in V
            dim_p_pe,                              // src outer stride: next batch's V chunk
            1                                      // src inner stride: contiguous
        );
        iter_num += 1;
    }
}
```

`range_copy_2d_strided` uses strided DSDs to copy with different source and destination strides,
iterating over `bsz` batches in an outer loop.

### 8f. Two New GEMV Variants for Variable-Length Cache

**Problem**: After T decode steps, each PE has `iter_num ≤ max_seq_len_p_pe` valid cache entries.
Score computation must use exactly `iter_num`, not the full allocation.

#### `vecmat_computation_var_right_2d_strided(right_stride)` — for K^T strided access

Used in `score_matvec_mult()`. K is stored as `[bsz, dim_p_pe, max_seq_len_p_pe]`, so consecutive
K-dimension entries are `max_seq_len_p_pe` elements apart in memory (column-strided).

```csl
// Nt = iter_num, Kt = dim_p_pe, right_stride = max_seq_len_p_pe
for b in range(bsz):
    for k in range(dim_p_pe):           // outer loop over K dimension
        src1 = KCache[b, k, :iter_num]  // iter_num contiguous elements
        @fmach(score[b, :iter_num], score[b, :iter_num], src1, Q[b, k])
        left_vector += 1                // next Q element
        right_matrix += max_seq_len_p_pe // next K^T row (stride jump)
```

#### `vecmat_computation_var_right(width_right)` — for V batched access

Used in `output_matvec_mult()`. V is stored as `[bsz, max_seq_len_p_pe, dim_p_pe]`.
For each batch, the first `iter_num` rows are valid and contiguous in memory.
After each batch, advance by the full `max_seq_len_p_pe * dim_p_pe` to the next batch block.

```csl
// Kt = iter_num, Nt = dim_p_pe, width_right = max_seq_len_p_pe * dim_p_pe
for b in range(bsz):
    @fmach(output[b, :], output[b, :], V[b, :iter_num, :], score[b, :iter_num])
    right_matrix += width_right  // skip to next batch's V block
```

### 8g. Variable-Length All-Reduce: `all_reduce_bsz_var_seq_len_strided()`

**Problem**: fabric DSDs are sized at compile time to `max_seq_len_p_pe`. We cannot create
a new DSD with a runtime-determined length.

**Solution**: The function takes `var_seq_len: i16` at runtime and uses `@set_dsd_length`
to resize all send/receive/buffer DSDs before executing the reduce:

```csl
fn all_reduce_bsz_var_seq_len_strided(..., var_seq_len: i16) void {
    var var_reduce_1st_0_recv = @set_dsd_length(reduce_1st_0_recv_bsz_seq_len, bsz * var_seq_len);
    // ... resize all 10 DSDs (4 phases × 2 colors + broadcast × 2) ...
    vector_buf_dsd = @set_dsd_length(vector_buf_dsd, bsz * var_seq_len);
    // then execute the same hierarchical reduce pattern as the static version
}
```

This efficiently transfers only the valid `iter_num` scores, avoiding unnecessary bandwidth.

### 8h. `decode_struct()` Changes Summary

```
WSE-3:                                WSE-3-KV:
──────────────────────────────        ────────────────────────────────
rmsnorm_x()                           rmsnorm_x()
xq_matvec_mult()  [uses X_norm]       xq_matvec_mult()  [uses X — dev simplification]
xk_matvec_mult()  [uses X_norm]       xk_matvec_mult()  [uses X]
xv_matvec_mult()  [uses X_norm]       xv_matvec_mult()  [uses X]
all_reduce QKV (Y)                    all_reduce QKV (Y)
reconfig_axis(0)                      reconfig_axis(0)
xq_rope()                             [OMITTED — commented out in KV version]
xk_rope()                             [OMITTED]
                                      process_kv()          ← NEW: write K,V to cache
score_matvec_mult()                   score_matvec_mult()   ← uses iter_num
  all_reduce_bsz_seq_len (X)            all_reduce_bsz_var_seq_len_strided(X, iter_num)
reconfig_axis(1)                      reconfig_axis(1)
softmax_score()                       softmax_score()       ← uses iter_num DSDs
output_matvec_mult()                  output_matvec_mult()  ← uses iter_num, strided V
  all_reduce_bsz_dim (Y)                all_reduce_bsz_dim (Y)
reconfig_axis(0)                      reconfig_axis(0)
o_matvec_mult()                       o_matvec_mult()       [IDENTICAL]
attn_residual_add()                   attn_residual_add()   [IDENTICAL]
reconfig_axis(1)                      reconfig_axis(1)
rmsnorm_z()                           rmsnorm_z()           [IDENTICAL]
up_matvec_mult()                      up_matvec_mult()      [IDENTICAL]
gate_matvec_mult()                    gate_matvec_mult()    [IDENTICAL]
all_reduce ZZ (Y)                     all_reduce ZZ (Y)     [IDENTICAL]
z2_silu()                             z2_silu()             [IDENTICAL]
z3_mat()                              z3_mat()              [IDENTICAL]
reconfig_axis(0)                      reconfig_axis(0)
down_matvec_mult()                    down_matvec_mult()    [IDENTICAL]
ffn_residual_add()                    ffn_residual_add()    [IDENTICAL]
reconfig_axis(1)                      reconfig_axis(1)
decode_entry()                        step += 1             ← new counter
```

### 8i. New Exported Functions

```csl
fn decode_once() void { decode_struct(); }
fn finalize() void { decode_struct(); /* + take timestamps */ }
```

These allow the host to drive individual decode steps (for multi-step generation).

---

## 9. Key "Gotchas"

1. **QKV fusion**: The current code fuses Q+K+V all-reduces into a single `3*dim_p_pe` transfer.
   For GQA this optimization is broken by size mismatch — must split or implement Q+KV fusion.

2. **`pes_p_kv_head` is computed in run_sim.sh but not used in decode.csl** — it's a placeholder
   left for GQA implementation.

3. **`_dim_p_pe = (dim_p_pe / 2) * 2`**: RoPE operates on even-sized vectors; this ensures the
   size is even. Similar adjustment needed for `kv_dim_p_pe` in GQA.

4. **RoPE frequencies**: `freqs_sin` and `freqs_cos` are sized `[_dim_p_pe / 2]`. For GQA,
   K has different dimension (`kv_dim_p_pe`), so K needs separate/subset frequency arrays.

5. **Score all-reduce axis**: Score all-reduce is along X (`px`) because Q·K^T dot product
   accumulates over the hidden dim which spans the X-axis shards. This depends on how
   Q and K are partitioned — must verify for GQA.

6. **`freqs_sin/cos` are shared between Q and K RoPE** — valid in MHA since both have
   the same dim-per-PE. For GQA, K has `kv_dim_p_pe` so we need `kv_dim_p_pe/2` frequencies
   for K's rope, or use a subset of the Q frequencies.
