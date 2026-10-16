# Per-PE Memory Layout: WSE-3-GQA-cheng

This document describes the memory layout of Q (after projection), KCache, VCache, and Score buffers within each PE in the `WSE-3-GQA-cheng` implementation.

All buffers are 1D arrays of `f16`. Dimensions are listed **outermost → innermost** (i.e., the innermost dimension is contiguous in memory).

---

## Parameters

| Symbol | Meaning | Example (Llama-8B, P=256) |
|--------|---------|---------------------------|
| `bsz` | Batch size | 1 |
| `dim_p_pe` | Hidden dim per PE = `dim / P` | 16 |
| `seq_len_p_pe` | Sequence length per PE = `seq_len / P` | 16 |
| `n_kv_heads` | Number of KV heads (GQA parameter) | 8 |
| `head_dim` | Dimension per head | 128 |

---

## QKV Tile (after projection, before reduce)

```
QKV_tile: [bsz * 3 * dim_p_pe] f16
```

Layout (outermost → innermost): **`[Q | K | V]`**, each section is `[bsz, dim_p_pe]`.

```
Memory:  [Q_b0(dim_p_pe), Q_b1(dim_p_pe), ...,
          K_b0(dim_p_pe), K_b1(dim_p_pe), ...,
          V_b0(dim_p_pe), V_b1(dim_p_pe), ...]
         |<---- bsz*dim_p_pe ---->|<---- bsz*dim_p_pe ---->|<---- bsz*dim_p_pe ---->|
```

- **Q region**: offset `0`, extent `bsz * dim_p_pe`
- **K region**: offset `bsz * dim_p_pe`, extent `bsz * dim_p_pe`
- **V region**: offset `2 * bsz * dim_p_pe`, extent `bsz * dim_p_pe`

### Xq layout within QKV_tile

**`[bsz, dim_p_pe]`** — batch is outer, dimension shard is inner (contiguous).

```
Memory: [Q_b0_d0, Q_b0_d1, ..., Q_b0_d{dim_p_pe-1},   Q_b1_d0, ..., Q_b1_d{dim_p_pe-1}, ...]
        |<---------- batch 0 (dim_p_pe) ---------->|   |<---------- batch 1 ------------>|
```

Each batch element's `dim_p_pe` contiguous values are a shard of the full Q vector — specifically elements `[px*dim_p_pe : (px+1)*dim_p_pe]` of the global Q output. These elements may span parts of one or more attention heads depending on how `dim_p_pe` relates to `head_dim`.

After the fused QKV X-reduce (`all_reduce_bsz_dim_QKV_fusion`), each PE has the complete Q/K/V values for its `dim_p_pe` output elements. The Q portion is used directly for score computation. The K and V portions are written into the KV cache by the host at init time (no on-device cache update in this implementation).


Global Q: [head0(128) | head1(128) | head2(128) | head3(128)]

  PE px=0: Q[0:64]    → part of head 0 → quotient_x=0 → kv_head_id = 0 % 2 = 0
  PE px=1: Q[64:128]  → part of head 0 → quotient_x=0 → kv_head_id = 0 % 2 = 0
  PE px=2: Q[128:192] → part of head 1 → quotient_x=1 → kv_head_id = 1 % 2 = 1
  PE px=3: Q[192:256] → part of head 1 → quotient_x=1 → kv_head_id = 1 % 2 = 1
  PE px=4: Q[256:320] → part of head 2 → quotient_x=2 → kv_head_id = 2 % 2 = 0  ← same KV head as PE 0,1
  PE px=5: Q[320:384] → part of head 2 → quotient_x=2 → kv_head_id = 2 % 2 = 0
  PE px=6: Q[384:448] → part of head 3 → quotient_x=3 → kv_head_id = 3 % 2 = 1  ← same KV head as PE 2,3
  PE px=7: Q[448:512] → part of head 3 → quotient_x=3 → kv_head_id = 3 % 2 = 1



---



## XKCache (K Cache)

```
XKCache_tile: [n_kv_heads * dim_p_pe * seq_len_p_pe] f16
```

Layout (outermost → innermost): **`[kv_head, dim_p_pe, seq_len_p_pe]`**

```
Memory:  [head0: [row0(seq_len_p_pe), row1(seq_len_p_pe), ..., row_{dim_p_pe-1}(seq_len_p_pe)],
          head1: [row0(seq_len_p_pe), row1(seq_len_p_pe), ..., row_{dim_p_pe-1}(seq_len_p_pe)],
          ...
          head_{n_kv_heads-1}: [...]]
```

Each KV head block has size `dim_p_pe * seq_len_p_pe`. The PE selects the correct block via:
```csl
xk_offset = kv_head_id * (dim_p_pe * seq_len_p_pe);
ptr_right_matrix = @ptrcast([*]f16, &XKCache_tile[xk_offset]);
```

**Why this layout?** The score GEMV computes `Q[1, dim_p_pe] @ K[dim_p_pe, seq_len_p_pe]^T`, implemented as an outer-product accumulation. The inner loop iterates over `dim_p_pe` rows of K, each row being a contiguous `seq_len_p_pe`-length vector. The `right_matrix_dsd` advances by `seq_len_p_pe` (one row stride) at each step of the inner GEMV loop (`@increment_dsd_offset(right_matrix_dsd, Nt, f16)` where `Nt = seq_len_p_pe`). This makes each K row contiguous for efficient DSD access.

**Host packing** (`launch_sim_gqa.py`): The global K cache `[n_kv_heads, dim, seq_len]` is sharded per PE as:
```python
for h in range(n_kv_heads):
    XKCache_reshape = tensor_XKCache[h].reshape(P, dim_p_pe, P, seq_len_p_pe)
    XKCache_transpose = XKCache_reshape.transpose(0, 2, 1, 3)  # [P, P, dim_p_pe, seq_len_p_pe]
    xk_blocks.append(XKCache_transpose.reshape(P, P, dim_p_pe * seq_len_p_pe))
XKCache_packed = np.concatenate(xk_blocks, axis=2)  # [P, P, n_kv_heads * dim_p_pe * seq_len_p_pe]
```

PE `(px, py)` receives rows `[px*dim_p_pe : (px+1)*dim_p_pe]` and columns `[py*seq_len_p_pe : (py+1)*seq_len_p_pe]` for each KV head, concatenated across all `n_kv_heads` blocks.

---

## XVCache (V Cache)

```
XVCache_tile: [n_kv_heads * seq_len_p_pe * dim_p_pe] f16
```

Layout (outermost → innermost): **`[kv_head, seq_len_p_pe, dim_p_pe]`**

```
Memory:  [head0: [tok0(dim_p_pe), tok1(dim_p_pe), ..., tok_{seq_len_p_pe-1}(dim_p_pe)],
          head1: [tok0(dim_p_pe), tok1(dim_p_pe), ..., tok_{seq_len_p_pe-1}(dim_p_pe)],
          ...
          head_{n_kv_heads-1}: [...]]
```

Each KV head block has size `seq_len_p_pe * dim_p_pe`. The PE selects the correct block via:
```csl
xv_offset = kv_head_id * (seq_len_p_pe * dim_p_pe);
ptr_right_matrix = @ptrcast([*]f16, &XVCache_tile[xv_offset]);
```

**Why this layout (transposed vs K)?** The output GEMV computes `score[1, seq_len_p_pe] @ V[seq_len_p_pe, dim_p_pe]`, which is also an outer-product accumulation. The inner loop iterates over `seq_len_p_pe` rows of V, each row being a contiguous `dim_p_pe`-length vector. The `right_matrix_dsd` advances by `dim_p_pe` per step. This means V is stored as `[seq_len_p_pe, dim_p_pe]` — the **transpose** of K's layout — so that the GEMV's "right matrix" rows are contiguous in both cases.

**Host packing** (`launch_sim_gqa.py`):
```python
for h in range(n_kv_heads):
    XVCache_reshape = tensor_XVCache[h].reshape(P, seq_len_p_pe, P, dim_p_pe)
    XVCache_transpose = XVCache_reshape.transpose(0, 2, 1, 3)  # [P, P, seq_len_p_pe, dim_p_pe]
    xv_blocks.append(XVCache_transpose.reshape(P, P, seq_len_p_pe * dim_p_pe))
XVCache_packed = np.concatenate(xv_blocks, axis=2)  # [P, P, n_kv_heads * seq_len_p_pe * dim_p_pe]
```

---

## Score

```
score: [bsz * seq_len_p_pe] f16
```

Layout (outermost → innermost): **`[bsz, seq_len_p_pe]`**

```
Memory:  [score_b0(seq_len_p_pe), score_b1(seq_len_p_pe), ...]
```

**Only one score vector per PE** — no `gqa_group_size` multiplier. This is possible because each PE's Q shard corresponds to a single query head (or partial head), and the GQA head sharing is handled by multiple PEs selecting the same KV head block via `kv_head_id` offset. Different Q heads that share a KV head live on **different PEs**, each computing one independent score vector.

---

## Score GEMV Access Pattern

```
score[j] = Σ_i  Q[i] * KCache[kv_head_id, i, j]     for j ∈ [0, seq_len_p_pe)
                                                       for i ∈ [0, dim_p_pe)
```

The `vecmat_computation()` inner loop (per batch element):
1. `left_vector_dsd` points to Q (1 element at a time, scalar broadcast)
2. `right_matrix_dsd` points to K row `i` (contiguous `seq_len_p_pe` elements)
3. `out_vector_dsd` points to score (contiguous `seq_len_p_pe` elements, accumulated)
4. After each GEMV step: `right_matrix_dsd += Nt` (advance to next K row)

After the local GEMV, `all_reduce_bsz_seq_len` along X sums partial scores across PEs that hold different dimension shards of the same head, producing the complete attention score.

---

## Output GEMV Access Pattern

```
output[j] = Σ_i  score[i] * VCache[kv_head_id, i, j]   for j ∈ [0, dim_p_pe)
                                                          for i ∈ [0, seq_len_p_pe)
```

Same `vecmat_computation()` structure but with:
- `Kt = seq_len_p_pe`, `Nt = dim_p_pe`
- `left_vector_dsd` → score, `right_matrix_dsd` → V cache, `out_vector_dsd` → output

After the local GEMV, `all_reduce_bsz_dim` along Y sums partial outputs across PEs that hold different sequence shards of the same head.

---

## Summary Table

| Buffer | Array Size | Logical Shape (outer → inner) | Per-Head Block Size |
|--------|-----------|-------------------------------|---------------------|
| QKV_tile | `bsz * 3 * dim_p_pe` | `[Q\|K\|V], [bsz, dim_p_pe]` | — |
| XKCache | `n_kv_heads * dim_p_pe * seq_len_p_pe` | `[kv_head, dim_p_pe, seq_len_p_pe]` | `dim_p_pe * seq_len_p_pe` |
| XVCache | `n_kv_heads * seq_len_p_pe * dim_p_pe` | `[kv_head, seq_len_p_pe, dim_p_pe]` | `seq_len_p_pe * dim_p_pe` |
| Score | `bsz * seq_len_p_pe` | `[bsz, seq_len_p_pe]` | — |
| Output | `bsz * dim_p_pe` | `[bsz, dim_p_pe]` | — |

**Key insight**: KCache is `[dim, seq_len]` and VCache is `[seq_len, dim]` — they are transposed relative to each other. This ensures that in both the score GEMV (`Q @ K^T`) and the output GEMV (`score @ V`), the right matrix's rows are contiguous in memory, enabling efficient DSD-based row access in the inner GEMV loop.
