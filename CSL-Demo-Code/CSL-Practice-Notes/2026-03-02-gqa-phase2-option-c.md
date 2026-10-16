# GQA Phase 2 (Option C) Implementation — 2026-03-02

## Finding
GQA Phase 2 (Option C) adds `kv_dim_p_pe = n_kv_heads * head_dim / P` as a CSL compile-time
param and achieves compact K/V storage + offline W_Q/W_O column permutation (no runtime scatter).
Key constraints: (1) `group_num % n_kv_heads == 0` required, (2) `bsz=1` for correctness of the
multi-group score GEMV (with bsz>1 and gqa_group_size>1 the batch indexing in score_matvec_mult
needs a different layout — see note below).

## Key Design Decisions

- **QKV fusion buffer** resized: `bsz*(dim_p_pe + 2*kv_dim_p_pe)` — Q stays `dim_p_pe`, K/V compact.
- **score buffer** sized `gqa_group_size * bsz * seq_len_p_pe`; sub-head g at offset `g*bsz*seq_len_p_pe`.
- **output buffer** `bsz*dim_p_pe` = `gqa_group_size * bsz * kv_dim_p_pe`; sub-head g at offset `g*bsz*kv_dim_p_pe`.
- **Single Y-reduce** covers all `gqa_group_size` sub-head outputs after the output loop.
- **KV-head route structs**: new `X_KV_HEAD_2nd_RD_0/1`, `X_KV_HEAD_BD` in comm_layout.csl, using
  `px_in_kv_head`, `quotient_x_in_kv_head`, `root_kv_head_2nd_in_head`, `pes_p_kv_head` boundary.
- **axis=3** added to `reconfig_allreduce_axis` in comm_pe.csl for KV-head-scoped X reduce.

## W_Q/W_O Permutation (host-side, Python)
```python
for h in range(n_heads):
    kv_head = h // gqa_group_size
    g = h % gqa_group_size
    for s in range(pes_p_kv_head):
        old_col = h * head_dim + s * kv_dim_p_pe
        new_col = kv_head * pes_p_kv_head * dim_p_pe + s * dim_p_pe + g * kv_dim_p_pe
        W_Q_perm[:, new_col:new_col+kv_dim_p_pe] = W_Q[:, old_col:old_col+kv_dim_p_pe]
        W_O_perm[new_col:new_col+kv_dim_p_pe, :] = W_O[old_col:old_col+kv_dim_p_pe, :]
```
After permutation, PE(px) holds Q groups g=0..gqa_group_size-1 interleaved: offsets `0, kv_dim_p_pe, 2*kv_dim_p_pe, ...` within `QKV_tile[0:bsz*dim_p_pe]`.

## KCache/VCache Tiling (host-side)
- KCache `[kv_dim, seq_len]` → `tile_weight_row(XKCache, kv_dim_p_pe, seq_len_p_pe).swapaxes(0,1)`
  (swapaxes needed: K-dim goes by px, seq-len goes by py)
- VCache `[seq_len, kv_dim]` → `tile_weight_row(XVCache, seq_len_p_pe, kv_dim_p_pe)` (no swap)

## bsz>1 Limitation with gqa_group_size>1
In `score_matvec_mult()`, `ptr_q_g = &QKV_tile[g * bsz * kv_dim_p_pe]`. For bsz=1 this
correctly points to the g-th sub-block. For bsz>1, the vecmat loop increments by Kt=kv_dim_p_pe
per batch but crosses into the next group's region. Fix requires restructuring the Q buffer
or a custom GEMV that knows the per-batch stride = dim_p_pe (not kv_dim_p_pe).

## Test Config (gqa_test.json)
P=8, n_heads=4, n_kv_heads=2, group_num=4, head_dim=16, dim=64, seq_len=64, ffn_dim=64.
Gives: gqa_group_size=2, kv_dim_p_pe=4, pes_p_kv_head=4, groups_per_kv_head=2.

## Source
Trial/design: Decode/WSE-3-GQA session 2026-03-02
