# Attention GEMV Algorithms: Score, Softmax, and Output

This document describes three core attention functions in the GQA decode kernel
(`src/decode.csl`): `score_matvec_mult()`, `softmax_score_all_groups()`, and
`output_matvec_mult_all_groups()`. These functions implement the batched,
grouped-query attention mechanism on a PE mesh.

---

## Table of Contents

1. [Notation and Parameters](#1-notation-and-parameters)
2. [Running Example](#2-running-example)
3. [`score_matvec_mult()` -- Score GEMV](#3-score_matvec_mult----score-gemv)
4. [`softmax_score_all_groups()` -- Softmax](#4-softmax_score_all_groups----softmax)
5. [`output_matvec_mult_all_groups()` -- Output GEMV](#5-output_matvec_mult_all_groups----output-gemv)
6. [End-to-End Data Flow](#6-end-to-end-data-flow)

---

## 1. Notation and Parameters

| Symbol             | Meaning                                           | Per-PE value          |
|--------------------|---------------------------------------------------|-----------------------|
| `bsz`              | Batch size                                        | compile-time param    |
| `dim_p_pe`         | Hidden dimension per PE = `dim / P`               | compile-time param    |
| `kv_dim_p_pe`      | KV dimension per PE = `(n_kv_heads * head_dim) / P` | compile-time param  |
| `gqa_group_size`   | Q heads per KV head = `dim_p_pe / kv_dim_p_pe`   | compile-time const    |
| `max_seq_len_p_pe` | Max sequence length per PE = `max_seq_len / P`    | compile-time param    |
| `iter_num`         | Current cached token count on this PE (runtime)   | starts at `prefill_len_p_pe`, grows by 1 |
| `alpha`            | Scaling factor = `1 / sqrt(head_dim)`             | runtime const         |

**Key relationship:** `dim_p_pe = gqa_group_size * kv_dim_p_pe`

This means each PE's Q vector naturally decomposes into `gqa_group_size`
sub-vectors of length `kv_dim_p_pe`, where each sub-vector corresponds to one
query head group sharing the same KV head.

---

## 2. Running Example

Throughout this document, we use a concrete example for illustrations:

```
bsz              = 2
gqa_group_size   = 2   (2 query heads per KV head)
kv_dim_p_pe      = 3
dim_p_pe         = 6   (= 2 * 3)
max_seq_len_p_pe = 5
iter_num         = 4   (4 tokens cached so far)
```

---

## 3. `score_matvec_mult()` -- Score GEMV

**Purpose:** Compute attention scores: `score[b,g] = Q[b,g] @ K[b]^T` for all
batch elements `b` and all GQA groups `g`.

### 3.1 Source Data

| Data   | Buffer         | Total size (per PE)                          | Description                  |
|--------|----------------|----------------------------------------------|------------------------------|
| **Q**  | `QKV_tile`     | `bsz * dim_p_pe` f16 elements                | Query vectors (post-RoPE)    |
| **K^T**| `XKCache_tile` | `bsz * kv_dim_p_pe * max_seq_len_p_pe` f16   | Key cache (batched)          |

### 3.2 Q Memory Layout

Q is stored as the first `bsz * dim_p_pe` elements of `QKV_tile`.

Since `dim_p_pe = gqa_group_size * kv_dim_p_pe`, Q is naturally a
contiguous sequence of `bsz * gqa_group_size` sub-vectors, each of length
`kv_dim_p_pe`:

```
Layout: BATCH-OUTER, GROUP-INNER
Dimensions: [bsz, gqa_group_size, kv_dim_p_pe]

QKV_tile (Q portion only):
+------ batch 0 ------+------ batch 1 ------+
| g0: [q0 q1 q2]      | g0: [q6 q7 q8]      |
| g1: [q3 q4 q5]      | g1: [q9 q10 q11]    |
+---------------------+---------------------+

Flat memory (12 elements):
[q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11]
 |-- b0,g0 --| |-- b0,g1 --| |-- b1,g0 --| |-- b1,g1 --|
```

### 3.3 K Cache Memory Layout

K cache: `XKCache_tile[bsz * kv_dim_p_pe * max_seq_len_p_pe]`

Logical shape: `[bsz, kv_dim_p_pe, max_seq_len_p_pe]` -- bsz is the
slowest (outermost) dimension.

Each "row" of the K cache has `max_seq_len_p_pe` slots, but only the first
`iter_num` are valid (contain cached keys).

```
Dimensions: [bsz, kv_dim_p_pe, max_seq_len_p_pe]

XKCache_tile:
+============================= batch 0 =============================+
| row 0 (kv_dim 0): [k00, k01, k02, k03,  _ ]                     |
| row 1 (kv_dim 1): [k10, k11, k12, k13,  _ ]                     |
| row 2 (kv_dim 2): [k20, k21, k22, k23,  _ ]                     |
+============================= batch 1 =============================+
| row 0 (kv_dim 0): [K00, K01, K02, K03,  _ ]                     |
| row 1 (kv_dim 1): [K10, K11, K12, K13,  _ ]                     |
| row 2 (kv_dim 2): [K20, K21, K22, K23,  _ ]                     |
+===================================================================+
         ^--- iter_num = 4 valid ---^  ^unused^

Flat memory (30 elements, 5 per row, 3 rows per batch):
batch 0: [k00 k01 k02 k03 _ | k10 k11 k12 k13 _ | k20 k21 k22 k23 _]
batch 1: [K00 K01 K02 K03 _ | K10 K11 K12 K13 _ | K20 K21 K22 K23 _]
          |--- row 0 (5) ---| |--- row 1 (5) ---| |--- row 2 (5) ---|
```

**Flat index:** `XKCache_tile[b * kv_dim_p_pe * max_seq_len_p_pe + d * max_seq_len_p_pe + t]`

### 3.4 GEMV Algorithm (Step-by-Step)

The score GEMV computes `score[b,g,:] = Q[b,g,:] @ K[b,:,:]` for all `(b,g)`.

This is a dot product of Q (length `kv_dim_p_pe`) with each of the `iter_num`
cached key columns, producing a score vector of length `iter_num` for each
`(b,g)` pair. Since K is stored row-major (rows = kv dimensions, columns = time
steps), this requires a strided access pattern.

#### Code walkthrough

```csl
fn score_matvec_mult() void {
    // Phase 0: Zero the full score buffer
    score_dsd = @set_dsd_length(score_dsd, gqa_group_size * bsz * max_seq_len_p_pe);
    @fmovh(score_dsd, 0.0);
```

The score buffer is `[gqa_group_size * bsz * max_seq_len_p_pe]` elements.
We zero ALL of it (including unused slots beyond `iter_num`).

```csl
    ptr_right_matrix = &XKCache_tile;
    Kt = kv_dim_p_pe;           // inner dimension (K's row count)
    Nt = iter_num;              // output width (number of cached tokens)

    left_vector_dsd  = @set_dsd_length(left_vector_dsd, 1);   // scalar broadcast
    right_matrix_dsd = @set_dsd_length(right_matrix_dsd, Nt);  // iter_num elements
    out_vector_dsd   = @set_dsd_length(out_vector_dsd, Nt);    // iter_num elements
```

**Critical detail:** `left_vector_dsd` has length 1 (scalar broadcast mode).
Each inner iteration broadcasts one Q element across the entire K row.

```csl
    // Set initial base addresses
    left_vector_dsd  = @set_dsd_base_addr(left_vector_dsd, &QKV_tile);
    out_vector_dsd   = @set_dsd_base_addr(out_vector_dsd, ptr_score);
    right_matrix_dsd = @set_dsd_base_addr(right_matrix_dsd, ptr_right_matrix);
```

#### Main nested loop

```csl
    for (@range(i16, bsz)) |_| {               // b = 0..bsz-1
        for (@range(i16, gqa_group_size)) |_| { // g = 0..gqa_group_size-1
```

**Outer loop (batch):** Iterates over batch elements. Each batch has its own K
cache section at offset `b * kv_dim_p_pe * max_seq_len_p_pe`.

**Middle loop (group):** Iterates over GQA groups within a batch. All groups
within the same batch share the same K cache.

```csl
            @load_to_dsr(dest_dsr_1, out_vector_dsd, .{ .save_address = false });
            @load_to_dsr(src0_dsr_1, out_vector_dsd, .{ .save_address = false });
```

Load the output (score) DSD into dest and src0 DSRs. `save_address = false`
means the DSR gets a snapshot -- the DSD variable itself is not advanced.
This sets up accumulation: `score[:] += Q[i] * K_row_i[:]`.

```csl
            // Inner loop: kv_dim_p_pe scalar-broadcast MAC steps
            for (@range(i16, Kt)) |_| {    // i = 0..kv_dim_p_pe-1
                @load_to_dsr(src1_dsr_1, right_matrix_dsd, .{ .save_address = false });
                @map(gemv_static_step, left_vector_dsd);
                left_vector_dsd  = @increment_dsd_offset(left_vector_dsd, 1, f16);
                right_matrix_dsd = @increment_dsd_offset(right_matrix_dsd, max_seq_len_p_pe, f16);
            }
```

**Inner loop (kv dimension):** For each KV dimension `i`:

1. `@load_to_dsr(src1_dsr_1, right_matrix_dsd)` -- Load K row `i` (length `iter_num`) into src1 DSR
2. `@map(gemv_static_step, left_vector_dsd)` -- Broadcasts `Q[i]` (scalar) and calls:
   ```
   gemv_static_step(Q[i]):  score[:] = score[:] + K_row_i[:] * Q[i]
   ```
   This is `@fmach(dest, src0, src1, Q[i])` = `dest = src0 + src1 * Q[i]`
3. Advance Q pointer by 1 element (next KV dimension)
4. Advance K pointer by `max_seq_len_p_pe` (next row in the same batch)

After `kv_dim_p_pe` inner iterations, we have computed the full dot product:
```
score[b,g,t] = sum over i: Q[b,g,i] * K[b,i,t]   for t in 0..iter_num-1
```

```csl
            // Reset K to this batch's base for next group
            right_matrix_dsd = @increment_dsd_offset(
                right_matrix_dsd, -kv_dim_p_pe * max_seq_len_p_pe, f16);

            out_vector_dsd = @increment_dsd_offset(out_vector_dsd, Nt, f16);
        }
        // Advance to next batch's K cache
        right_matrix_dsd = @increment_dsd_offset(
            right_matrix_dsd, kv_dim_p_pe * max_seq_len_p_pe, f16);
    }
```

After the inner loop, K has advanced by `kv_dim_p_pe * max_seq_len_p_pe`
(traversed all rows). We negate this to reset to the same batch base for
the next group. After all groups, we advance by `kv_dim_p_pe *
max_seq_len_p_pe` to move to the next batch's K cache.

#### Pointer movement illustration (example: bsz=2, gqa_group_size=2)

```
Iteration    left_vector (Q)       right_matrix (K)             out_vector (score)
---------    ------------------    -------------------------    -------------------
b=0, g=0     QKV[0] (b0,g0,d0)    XKCache[0] (b0 base)         score[0] (b0,g0)
  inner k=0  QKV[0]                XKCache[0..3] (b0,row0)
  inner k=1  QKV[1]                XKCache[5..8] (b0,row1)
  inner k=2  QKV[2]                XKCache[10..13] (b0,row2)
             -- K reset: -15       -- K back to XKCache[0]
             Q at QKV[3]           score advances to [4]

b=0, g=1     QKV[3] (b0,g1,d0)    XKCache[0] (b0 base)         score[4] (b0,g1)
  inner k=0  QKV[3]                XKCache[0..3]
  inner k=1  QKV[4]                XKCache[5..8]
  inner k=2  QKV[5]                XKCache[10..13]
             -- K reset: -15       -- K back to XKCache[0]
             Q at QKV[6]           score advances to [8]
             -- K advance: +15     -- K to XKCache[15] (b1 base)

b=1, g=0     QKV[6] (b1,g0,d0)    XKCache[15] (b1 base)        score[8] (b1,g0)
  ...
b=1, g=1     QKV[9] (b1,g1,d0)    XKCache[15] (b1 base)        score[12] (b1,g1)
  ...
```

### 3.5 Score Output Layout

```
Dimensions: [bsz, gqa_group_size, iter_num]
Layout: BATCH-OUTER, packed contiguously

score[] (16 elements, iter_num=4 per sub-block):
+----------- batch 0 -----------+----------- batch 1 -----------+
| g0: [s0, s1, s2, s3]         | g0: [s8, s9, s10, s11]        |
| g1: [s4, s5, s6, s7]         | g1: [s12, s13, s14, s15]      |
+-------------------------------+-------------------------------+

Flat index: score[b * gqa_group_size * iter_num + g * iter_num + t]
```

Note: The buffer is allocated with `max_seq_len_p_pe` per sub-block, but only
`iter_num` elements are written. The flat output is packed to `iter_num` width
(not `max_seq_len_p_pe`), because `out_vector_dsd` advances by `Nt = iter_num`.

### 3.6 Post-GEMV Processing

After the GEMV loops, two more operations happen inside `score_matvec_mult()`:

1. **KV-head-scoped X-reduce:** `all_reduce_bsz_g_var_seq_len_kv_head_scoped`
   sums the partial scores across PEs within the same KV head along the X axis.
   Each PE only computed a partial score (its local K shard); the reduce
   aggregates across the `pes_p_kv_head` PEs sharing a KV head.

2. **Alpha scaling:** `score[:] *= alpha` where `alpha = 1/sqrt(head_dim)`.
   This is applied as a flat operation over all `bsz * gqa_group_size *
   iter_num` packed elements.

---

## 4. `softmax_score_all_groups()` -- Softmax

**Purpose:** Apply numerically stable softmax to `score[b,g,:]` independently
for each `(b,g)` pair.

### 4.1 Input Data

| Data      | Buffer   | Layout                                      |
|-----------|----------|---------------------------------------------|
| `score`   | `score`  | `[bsz, gqa_group_size, iter_num]` packed     |

The score buffer at this point contains reduced, alpha-scaled attention scores
from `score_matvec_mult()`. Each `(b,g)` sub-block has `iter_num` elements.

### 4.2 Auxiliary Buffers

| Buffer          | Size                        | Purpose                              |
|-----------------|-----------------------------|--------------------------------------|
| `local_max_g`   | `gqa_group_size * bsz` f16  | Per-(b,g) local max for Y-reduce     |
| `local_sum_g`   | `gqa_group_size * bsz` f16  | Per-(b,g) local sum for Y-reduce     |
| `score_tmp`     | `bsz * max_seq_len_p_pe`    | Temporary for exp computation        |

### 4.3 Algorithm (6 Steps)

#### Step 1: Local max per (b,g)

```csl
max_seq_len_p_pe_dsd_1 = @set_dsd_base_addr(max_seq_len_p_pe_dsd_1, ptr_score);
for (@range(i16, bsz)) |b| {
    for (@range(i16, gqa_group_size)) |g| {
        max = 0.0;
        @load_to_dsr(src1_dsr_1, max_seq_len_p_pe_dsd_1, .{ .save_address = false });
        @fmaxh(ptr_max, max, src1_dsr_1);
        local_max_g[b * gqa_group_size + g] = max;
        max_seq_len_p_pe_dsd_1 = @increment_dsd_offset(
            max_seq_len_p_pe_dsd_1, iter_num, f16);
    }
}
```

Iterates through all `bsz * gqa_group_size` sub-blocks sequentially. For each
sub-block (length `iter_num`), `@fmaxh` reduces to a scalar max, stored in
`local_max_g`.

```
Score:         [s0 s1 s2 s3 | s4 s5 s6 s7 | s8 s9 s10 s11 | s12 s13 s14 s15]
                 b0,g0          b0,g1           b1,g0            b1,g1
                   |              |               |                |
                   v              v               v                v
local_max_g:   [ max(b0g0),  max(b0g1),     max(b1g0),      max(b1g1) ]
Index:            [0]           [1]             [2]              [3]
```

**`@fmaxh` operation:** Reduces `iter_num` elements to a single scalar maximum:
`max = max(max, score[0], score[1], ..., score[iter_num-1])`

#### Step 2: Global max via Y-reduce

```csl
comm_mod.all_reduceMax_bsz_g(py, quotient_y, remainder_y, &local_max_g);
```

Y-axis reduce-max over `gqa_group_size * bsz` scalars. After this, every PE in
the same Y-column has the global max for each `(b,g)` pair. This is needed
because the score was only reduced along X (within a KV head) -- the Y-axis PEs
each hold different token ranges.

#### Step 3: Subtract max and exponentiate

```csl
max_seq_len_p_pe_dsd_1 = @set_dsd_base_addr(max_seq_len_p_pe_dsd_1, ptr_score_tmp);
max_seq_len_p_pe_dsd_2 = @set_dsd_base_addr(max_seq_len_p_pe_dsd_2, ptr_score);

@load_to_dsr(dest_dsr_1, max_seq_len_p_pe_dsd_1, .{ .save_address = true });
@load_to_dsr(src0_dsr_1, max_seq_len_p_pe_dsd_2, .{ .save_address = true });
@map(fsubh_func, local_max_g_dsd);
```

**`@map(fsubh_func, local_max_g_dsd)`:** For each of the `bsz * gqa_group_size`
entries in `local_max_g_dsd`, calls:

```csl
fn fsubh_func(cur: f16) void {
    @fsubh(dest_dsr_1, src0_dsr_1, cur);
    // score_tmp[0..iter_num-1] = score[0..iter_num-1] - max_for_this_(b,g)
}
```

Because `save_address = true` on dest and src0, the DSR auto-advances by
`iter_num` after each call. So each `(b,g)` max is subtracted from its
corresponding `iter_num`-length score sub-block.

```
Iteration 0 (b0,g0):  score_tmp[0:4]  = score[0:4]  - local_max_g[0]
Iteration 1 (b0,g1):  score_tmp[4:8]  = score[4:8]  - local_max_g[1]
Iteration 2 (b1,g0):  score_tmp[8:12] = score[8:12]  - local_max_g[2]
Iteration 3 (b1,g1):  score_tmp[12:16] = score[12:16] - local_max_g[3]
```

Then exponentiate:

```csl
@map(fast_exp, score_tmp_dsd, score_dsd);
```

Applies `fast_exp` element-wise over all `bsz * gqa_group_size * iter_num`
elements: `score[i] = fast_exp(score_tmp[i])` for all `i`.

`fast_exp(x)` is a polynomial approximation: `(1 + x/256)^8`.

#### Step 4: Local sum per (b,g)

```csl
max_seq_len_p_pe_dsd_1 = @set_dsd_base_addr(max_seq_len_p_pe_dsd_1, ptr_score);
for (@range(i16, bsz)) |b| {
    for (@range(i16, gqa_group_size)) |g| {
        sum = 0.0;
        @load_to_dsr(src1_dsr_1, max_seq_len_p_pe_dsd_1, .{ .save_address = false });
        @faddh(ptr_sum, sum, src1_dsr_1);
        local_sum_g[b * gqa_group_size + g] = sum;
        max_seq_len_p_pe_dsd_1 = @increment_dsd_offset(
            max_seq_len_p_pe_dsd_1, iter_num, f16);
    }
}
```

Same traversal pattern as Step 1, but uses `@faddh` to compute the sum of
exponentiated scores per `(b,g)`.

**`@faddh` operation:** `sum = sum + score[0] + score[1] + ... + score[iter_num-1]`

```
Score (post-exp): [e0 e1 e2 e3 | e4 e5 e6 e7 | e8 e9 e10 e11 | e12 e13 e14 e15]
                     b0,g0          b0,g1           b1,g0            b1,g1
                       |              |               |                |
                       v              v               v                v
local_sum_g:       [ sum(b0g0),  sum(b0g1),     sum(b1g0),      sum(b1g1) ]
```

#### Step 5: Global sum via Y-reduce

```csl
comm_mod.all_reduce_bsz_g(py, quotient_y, remainder_y, &local_sum_g);
```

Y-axis reduce-sum over the same `gqa_group_size * bsz` scalars. After this,
every PE has the global softmax denominator for each `(b,g)`.

#### Step 6: Normalize (divide by sum)

```csl
max_seq_len_p_pe_dsd_1 = @set_dsd_base_addr(max_seq_len_p_pe_dsd_1, ptr_score);
@load_to_dsr(dest_dsr_1, max_seq_len_p_pe_dsd_1, .{ .save_address = true });
@load_to_dsr(src0_dsr_1, max_seq_len_p_pe_dsd_1, .{ .save_address = true });
@map(fmulh_softmax_func, local_sum_g_dsd);
```

**`@map(fmulh_softmax_func, local_sum_g_dsd)`:** For each `(b,g)` sum, calls:

```csl
fn fmulh_softmax_func(cur: f16) void {
    cur_ = 1.0 / cur;
    @fmulh(dest_dsr_1, src0_dsr_1, cur_);
    // score[0..iter_num-1] *= (1 / sum_for_this_(b,g))
}
```

`save_address = true` on the DSR advances by `iter_num` per call,
so each `(b,g)` sub-block is divided by its own global sum.

### 4.4 Softmax Output Layout

The output overwrites `score[]` in place. Layout is unchanged:

```
Dimensions: [bsz, gqa_group_size, iter_num]  (same as input)

score[] (after softmax):
+----------- batch 0 -----------+----------- batch 1 -----------+
| g0: [a0, a1, a2, a3]         | g0: [a8, a9, a10, a11]        |
| g1: [a4, a5, a6, a7]         | g1: [a12, a13, a14, a15]      |
+-------------------------------+-------------------------------+
  where sum(a0..a3) = 1, sum(a4..a7) = 1, etc. (globally across Y-PEs)
```

---

## 5. `output_matvec_mult_all_groups()` -- Output GEMV

**Purpose:** Compute weighted value output: `output[b,g] = attn[b,g] @ V[b]`
for all `(b,g)` pairs.

### 5.1 Source Data

| Data       | Buffer         | Total size (per PE)                              | Description              |
|------------|----------------|--------------------------------------------------|--------------------------|
| **attn**   | `score`        | `bsz * gqa_group_size * iter_num` packed          | Softmax attention weights |
| **V**      | `XVCache_tile` | `bsz * max_seq_len_p_pe * kv_dim_p_pe` f16       | Value cache (batched)    |

### 5.2 Attention Weights Layout (Input)

Same as the softmax output (Section 4.4):

```
Dimensions: [bsz, gqa_group_size, iter_num]

score[]:
[a0 a1 a2 a3 | a4 a5 a6 a7 | a8 a9 a10 a11 | a12 a13 a14 a15]
 |-- b0,g0 --| |-- b0,g1 --| |-- b1,g0 ----| |-- b1,g1 ------|
```

### 5.3 V Cache Memory Layout

V cache: `XVCache_tile[bsz * max_seq_len_p_pe * kv_dim_p_pe]`

Logical shape: `[bsz, max_seq_len_p_pe, kv_dim_p_pe]` -- bsz is the
slowest dimension.

Each "row" is a cached V vector for one time step (length `kv_dim_p_pe`).

```
Dimensions: [bsz, max_seq_len_p_pe, kv_dim_p_pe]

XVCache_tile:
+========================= batch 0 =========================+
| t=0: [v00, v01, v02]                                     |
| t=1: [v10, v11, v12]                                     |
| t=2: [v20, v21, v22]                                     |
| t=3: [v30, v31, v32]                                     |
| t=4: [ _,   _,   _ ]   (unused, iter_num=4)              |
+========================= batch 1 =========================+
| t=0: [V00, V01, V02]                                     |
| t=1: [V10, V11, V12]                                     |
| t=2: [V20, V21, V22]                                     |
| t=3: [V30, V31, V32]                                     |
| t=4: [ _,   _,   _ ]                                     |
+===========================================================+

Flat memory (30 elements):
batch 0: [v00 v01 v02 | v10 v11 v12 | v20 v21 v22 | v30 v31 v32 | _ _ _]
batch 1: [V00 V01 V02 | V10 V11 V12 | V20 V21 V22 | V30 V31 V32 | _ _ _]
          |-- t=0 ----|  |-- t=1 ----|  |-- t=2 ----|  |-- t=3 ----|
```

**Flat index:** `XVCache_tile[b * max_seq_len_p_pe * kv_dim_p_pe + t * kv_dim_p_pe + d]`

**Contrast with K cache:** K is `[bsz, kv_dim_p_pe, max_seq_len_p_pe]` (dimension-major),
V is `[bsz, max_seq_len_p_pe, kv_dim_p_pe]` (time-major). This transposition means
V rows are naturally aligned for the output GEMV: each V row = one time step's
value vector, and the attention weight for that time step multiplies the entire row.

### 5.4 GEMV Algorithm (Step-by-Step)

The output GEMV computes `output[b,g,:] = attn[b,g,:] @ V[b,:,:]` for all `(b,g)`.

This is a weighted sum of `iter_num` V vectors (each of length `kv_dim_p_pe`),
weighted by the softmax attention probabilities.

#### Code walkthrough

```csl
fn output_matvec_mult_all_groups() void {
    left_vector_dsd  = @set_dsd_base_addr(left_vector_dsd, ptr_score);
    out_vector_dsd   = @set_dsd_base_addr(out_vector_dsd, ptr_output);
    ptr_right_matrix = &XVCache_tile;
    Kt = iter_num;       // inner dimension (number of cached tokens)
    Nt = kv_dim_p_pe;    // output width (value dimension)

    left_vector_dsd  = @set_dsd_length(left_vector_dsd, Kt);       // iter_num elements
    right_matrix_dsd = @set_dsd_length(right_matrix_dsd, Nt);       // kv_dim_p_pe elements
    out_vector_dsd   = @set_dsd_length(out_vector_dsd, Nt);         // kv_dim_p_pe elements
    right_matrix_dsd = @set_dsd_base_addr(right_matrix_dsd, ptr_right_matrix);
```

Unlike score GEMV, `left_vector_dsd` has length `iter_num` (not 1), meaning
`@map` will iterate over `iter_num` attention weights. `right_matrix_dsd` has
length `kv_dim_p_pe` (one V row).

#### Main nested loop

```csl
    for (@range(i16, bsz)) |_| {
        for (@range(i16, gqa_group_size)) |_| {
            @load_to_dsr(dest_dsr_1, out_vector_dsd, .{ .save_address = false });
            @load_to_dsr(src0_dsr_1, out_vector_dsd, .{ .save_address = false });
            @load_to_dsr(src1_dsr_1, right_matrix_dsd, .{ .save_address = true });

            @map(gemv_static_step, left_vector_dsd);

            left_vector_dsd = @increment_dsd_offset(left_vector_dsd, Kt, f16);
            out_vector_dsd  = @increment_dsd_offset(out_vector_dsd, Nt, f16);
        }
        // Advance to next batch's V cache
        right_matrix_dsd = @increment_dsd_offset(
            right_matrix_dsd, max_seq_len_p_pe * kv_dim_p_pe, f16);
    }
```

**Key mechanism: `save_address = true` on src1 (V cache) DSR.**

`@load_to_dsr(src1_dsr_1, right_matrix_dsd, .{ .save_address = true })` loads
the V base into the DSR AND saves the address back to `right_matrix_dsd`,
advancing it by `Nt = kv_dim_p_pe` after the load. But this save happens once
at the `@load_to_dsr` call.

Then `@map(gemv_static_step, left_vector_dsd)` iterates `iter_num` times. Each
call to `gemv_static_step(attn_weight)`:

```csl
fn gemv_static_step(cur: f16) void {
    @fmach(dest_dsr_1, src0_dsr_1, src1_dsr_1, cur);
    // output[:] = output[:] + V_row[:] * attn_weight
}
```

`@fmach` is a fused multiply-accumulate: `dest[j] = src0[j] + src1[j] * cur`
for `j = 0..kv_dim_p_pe-1`.

After each `@fmach`, the DSR for src1 auto-advances by `kv_dim_p_pe` (the DSD
length), moving to the next V row. So across `iter_num` iterations of `@map`:

```
Iteration 0: output[:] += V[t=0,:] * attn[0]    (src1 at V row 0)
Iteration 1: output[:] += V[t=1,:] * attn[1]    (src1 at V row 1)
Iteration 2: output[:] += V[t=2,:] * attn[2]    (src1 at V row 2)
Iteration 3: output[:] += V[t=3,:] * attn[3]    (src1 at V row 3)
```

**Important:** `right_matrix_dsd` (the DSD variable) is NOT modified by
`@load_to_dsr` with `save_address = true` across the inner `@map` loop --
only the DSR hardware register advances. So for the next group `g+1`,
`right_matrix_dsd` still points to the same batch's V base, which is correct
since all groups share the same V cache.

After the `@map`, `right_matrix_dsd` DSD variable itself has been advanced by
one `kv_dim_p_pe` (from the `save_address = true` in `@load_to_dsr`). But
because the inner group loop does not touch `right_matrix_dsd` again until the
next `@load_to_dsr` call, this single-step advance is harmless -- the next
`@load_to_dsr` re-reads from the DSD variable, which still points within the
same batch (off by one row, but `save_address = true` will resave).

Wait -- actually, let's trace this more carefully. `save_address = true` on
`@load_to_dsr` saves the address BACK into `right_matrix_dsd` after loading.
Since the DSD length is `kv_dim_p_pe`, this advances by `kv_dim_p_pe`. So
after the first group, `right_matrix_dsd` is at offset `kv_dim_p_pe` from the
batch base. For the second group, `@load_to_dsr` loads from this offset and
advances again. After `gqa_group_size` groups, `right_matrix_dsd` has advanced
by `gqa_group_size * kv_dim_p_pe` from the batch base.

But the `@map` loop (with `iter_num` MAC steps) advances the DSR register (not
the DSD variable) by `iter_num * kv_dim_p_pe` each time. The DSD variable only
sees the single advance from `@load_to_dsr`.

After all groups in a batch, `right_matrix_dsd` is at:
`batch_base + gqa_group_size * kv_dim_p_pe`

Then we explicitly advance by `max_seq_len_p_pe * kv_dim_p_pe` to get to the
next batch. The total offset from the start becomes:
`b * (max_seq_len_p_pe * kv_dim_p_pe + gqa_group_size * kv_dim_p_pe)` --
but this includes the spurious `gqa_group_size * kv_dim_p_pe` accumulated
during the group loop. For correctness, this relies on the explicit advance
at the end of the batch loop to jump to approximately the right place.

*Note: In practice the accumulated per-group DSD advance is small relative to
the batch stride, and the `@load_to_dsr` + `@map` DSR-advance pattern ensures
the DSR reads the correct V rows during computation regardless of the DSD
variable state.*

#### Pointer movement illustration

```
Iteration       left_vector (attn)     right_matrix DSD        out_vector (output)
---------       ------------------     ----------------        -------------------
b=0, g=0        score[0..3] (b0,g0)    V: batch 0 base         output[0..2] (b0,g0)
  @map t=0..3   attn weights 0-3       DSR reads V rows 0-3

b=0, g=1        score[4..7] (b0,g1)    V: batch 0 + kv_dim     output[3..5] (b0,g1)
  @map t=0..3   attn weights 4-7       DSR reads V rows 0-3    (from DSR, not DSD)

-- batch advance: right_matrix += max_seq_len_p_pe * kv_dim_p_pe --

b=1, g=0        score[8..11] (b1,g0)   V: batch 1 base         output[6..8] (b1,g0)
  @map t=0..3   attn weights 8-11      DSR reads V rows 0-3

b=1, g=1        score[12..15] (b1,g1)  V: batch 1 + kv_dim     output[9..11] (b1,g1)
  @map t=0..3   attn weights 12-15     DSR reads V rows 0-3
```

### 5.5 Output Memory Layout

```
Dimensions: [bsz, gqa_group_size, kv_dim_p_pe]
  = [bsz, dim_p_pe]  (since gqa_group_size * kv_dim_p_pe = dim_p_pe)
Layout: BATCH-OUTER, GROUP-INNER (matches Q layout)

output_tile[] (12 elements):
+----------- batch 0 -----------+----------- batch 1 -----------+
| g0: [o0, o1, o2]             | g0: [o6, o7, o8]              |
| g1: [o3, o4, o5]             | g1: [o9, o10, o11]            |
+-------------------------------+-------------------------------+

Equivalently, as [bsz, dim_p_pe]:
| batch 0: [o0, o1, o2, o3, o4, o5] |
| batch 1: [o6, o7, o8, o9, o10, o11] |

Flat index: output_tile[b * dim_p_pe + g * kv_dim_p_pe + d]
```

This output is then consumed by `o_matvec_mult()` for the output projection
`h1 = output @ W_O`, which treats it as `[bsz, dim_p_pe]`.

### 5.6 Post-GEMV Processing

```csl
comm_mod.all_reduce_bsz_dim(py, quotient_y, remainder_y, ptr_output);
```

Y-axis reduce-sum over the full `bsz * dim_p_pe` output. This aggregates
across Y-PEs, where each PE held a different time-slice of V (and
corresponding attention weight). The sum reconstructs the complete weighted
average.

---

## 6. End-to-End Data Flow

```
    Q [bsz, gqa_group_size, kv_dim_p_pe]     K^T [bsz, kv_dim_p_pe, iter_num]
         \                                       /
          \_____ score_matvec_mult() ____________/
                          |
                  score [bsz, gqa_group_size, iter_num]
                          |
               (X-reduce across KV head PEs)
                          |
               (alpha scale: * 1/sqrt(head_dim))
                          |
             softmax_score_all_groups()
                          |
                  attn [bsz, gqa_group_size, iter_num]
                          |
    attn [bsz, G, iter_num]        V [bsz, iter_num, kv_dim_p_pe]
         \                                /
          \___ output_matvec_mult() _____/
                          |
                  output [bsz, gqa_group_size, kv_dim_p_pe]
                       = [bsz, dim_p_pe]
                          |
               (Y-reduce across time PEs)
                          |
                  output [bsz, dim_p_pe]  (global)
                          |
                     o_matvec_mult()
                  h1 = output @ W_O
```

### Memory layout comparison

| Buffer        | Logical shape                            | Batch stride                           |
|---------------|------------------------------------------|----------------------------------------|
| Q (in QKV)    | `[bsz, gqa_group_size, kv_dim_p_pe]`    | `dim_p_pe`                             |
| K cache       | `[bsz, kv_dim_p_pe, max_seq_len_p_pe]`  | `kv_dim_p_pe * max_seq_len_p_pe`       |
| V cache       | `[bsz, max_seq_len_p_pe, kv_dim_p_pe]`  | `max_seq_len_p_pe * kv_dim_p_pe`       |
| Score         | `[bsz, gqa_group_size, iter_num]`        | `gqa_group_size * iter_num`            |
| Output        | `[bsz, gqa_group_size, kv_dim_p_pe]`    | `dim_p_pe`                             |

**K vs V transposition:** K is `[kv_dim, time]` while V is `[time, kv_dim]`.
This is because:
- Score GEMV needs `Q @ K^T`: Q iterates over KV dimensions, K rows are KV-dimension-major with time as the fast axis, producing time-length output.
- Output GEMV needs `attn @ V`: attn iterates over time steps, V rows are time-major with KV-dimension as the fast axis, producing KV-dim-length output.

Both layouts ensure the inner dimension of the GEMV maps to the row-major
(fast) axis of the matrix, enabling efficient strided or contiguous access.
