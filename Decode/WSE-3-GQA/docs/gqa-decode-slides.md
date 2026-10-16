---
theme: default
title: GQA Decode on WSE-3
info: |
  Design plan for multi-head and GQA decode on Cerebras WSE-3.
  Three-phase implementation: MHA → GQA + Option C → Incremental KV Cache.
highlighter: shiki
---

# GQA Decode on WSE-3

**Three-Phase Implementation Plan**

`Decode/WSE-3-GQA/` — 2026-02-26

---

## The Problem: WSE-3 Only Works for n_heads = 1

WSE-3 has parameter stubs for multi-head attention — but they are **dead code**.

| Evidence | File | Conclusion |
|----------|------|------------|
| `param pes_p_head` declared, never referenced | `decode.csl:11-12` | head param unused |
| `// param pes_p_head` commented out | `comm_pe.csl:4-5` | comm ignores head count |
| `pes_p_head` computed but never used | `launch_sim.py:58-59` | host ignores it too |
| K/V weights always `[dim, dim]` | `launch_sim.py:76-89` | no per-head tensor shapes |
| Score reduce spans all P PEs | `comm_pe.csl:823+` | **mixes all heads — wrong** |

Only `head_dim_p_pe` is used — solely for the attention scale factor `α = 1/√head_dim`.

---

## Why Full-P Reduce is Wrong for MHA

Score computation requires `Q @ K^T` — a **partial dot-product per PE** that must be summed only within one Q-head.

```
n_heads=4, P=32, pes_p_head=8

PE  0–7:  Q_head0 partial scores  →  must reduce only these 8 PEs ✓
PE  8–15: Q_head1 partial scores  →  must reduce only these 8 PEs ✓
...

Current code: all-reduce over all 32 PEs
→ computes Q[0:dim] @ K^T[0:dim, seq]   ← WRONG
```

**Fix:** X-reduce for scores must be **scoped to `pes_p_head` PEs**.

Y-reduces for softmax/output are naturally head-isolated (each px-column = one head). No scoping change needed there.

---

## PE Grid Layout

```
X-axis (px): partitions dim
  head 0       head 1       head 2       head 3
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│PE 0..7   │ │PE 8..15  │ │PE 16..23 │ │PE 24..31 │
│pes_p_head│ │pes_p_head│ │pes_p_head│ │pes_p_head│
└──────────┘ └──────────┘ └──────────┘ └──────────┘

Y-axis (py): partitions seq_len
  PE (px, py) holds K^T rows aligned with Q, at seq positions
  [py·seq_len_p_pe : (py+1)·seq_len_p_pe]
```

**Key constraint:** `n_heads | group_num`
→ head boundaries align with communication group boundaries
→ `groups_per_head = group_num / n_heads`

---

## Three-Phase Plan

```
Phase 1: MHA n_heads > 1
  Fix score X-reduce to be head-scoped (pes_p_head PEs)
  gqa_group_size = 1 degenerate case of Phase 2

          ↓

Phase 2: GQA + Option C
  Compact K^T/V (kv_dim_p_pe rows per PE, not replicated)
  W_Q/W_O offline column permutation → interleaved Q layout
  KV-head-scoped X-reduce (pes_p_kv_head PEs)

          ↓

Phase 3: Incremental KV Cache
  On-chip cache management (process_kv, iter_num)
  Inherits Phase 2 compute layout unchanged
  Adds variable-length GEMV variants
```

**Rationale:** Phase 2 degenerates to Phase 1 when `gqa_group_size = 1`. Phase 1 tests validate Phase 2 code for free.

---

## Phase 1: Head-Scoped Reduce

**New route structs** in `comm_layout.csl` that terminate at head boundaries:

```c
const head_id:          i16 = px / pes_p_head;
const px_in_head:       i16 = px % pes_p_head;
const quotient_x_in_head: i16 = px_in_head / pe_num_p_group;
const root_head_2nd_abs: i16 = head_id * pes_p_head + root_head_2nd_in_head;
```

| Struct | Purpose |
|--------|---------|
| `X_HEAD_2nd_RD_0/1` | Phase-2 reduce within head (stops at head boundary) |
| `X_HEAD_BD` | Broadcast from head-local root to `pes_p_head` PEs only |

**New `axis=2`** in `reconfig_allreduce_axis`:
```c
@set_color_config(reduce_2nd_color_0, X_HEAD_2nd_RD_0);
@set_color_config(reduce_2nd_color_1, X_HEAD_2nd_RD_1);
@set_color_config(broadcast_color,    X_HEAD_BD);
```

---

## Phase 1: Code Changes

| File | Change |
|------|--------|
| `comm_layout.csl` | Add `X_HEAD_2nd_RD_0/1`, `X_HEAD_BD` route structs |
| `comm_pe.csl` | Add `axis=2` to `reconfig_allreduce_axis`; add `all_reduce_bsz_seq_len_head_scoped` |
| `decode.csl` | `score_matvec_mult()`: call head-scoped reduce instead of full-P reduce |
| `run_sim.sh` | Validate `group_num % n_heads == 0` |
| `launch_sim.py` | Fix RoPE freqs: wrap per head (`px % pes_p_head`) not global `px` |

**New params:** `pes_p_head`, `groups_per_head`, `root_head_2nd_in_head`

**Test config** (`mha_test.json`): `P=8, n_heads=4, group_num=4`
→ `pes_p_head=2`, `groups_per_head=1`, `gqa_group_size=1`

---

## Phase 2: The GQA Alignment Problem

After standard QKV projection + Y-reduce:

```
PE px holds Q at Q-head-local offset:  r = px % pes_p_head
PE px holds K^T rows at KV-head-local: s = px % pes_p_kv_head
```

When `gqa_group_size > 1`, **`r·dim_p_pe ≠ s·kv_dim_p_pe`** at most PEs.

```
Example: n_heads=4, n_kv_heads=2, gqa_group_size=2, P=8

PE 0: Q_head0[0:8]  /  K^T[0:4]   → Q at d=0–7, K at d=0–3  ← mismatch
PE 1: Q_head0[8:16] /  K^T[4:8]   → Q at d=8–15, K at d=4–7 ← mismatch
```

Direct GEMV produces **wrong partial scores**. The all-reduce cannot correct this.

---

## Phase 2: Option C — Offline W_Q Permutation

**Core insight:** eliminate Q misalignment at *compile time* by permuting W_Q columns so that after projection + Y-reduce, each PE's Q already matches its local K^T rows.

**W_Q permutation formula (host, one-time):**
```python
for h in range(n_heads):
    kv_head = h // gqa_group_size
    for s in range(pes_p_kv_head):
        old_col = h * head_dim + s * kv_dim_p_pe
        new_col = kv_head * pes_p_kv_head * dim_p_pe \
                + s * dim_p_pe \
                + (h % gqa_group_size) * kv_dim_p_pe
        W_Q_perm[:, new_col:new_col+kv_dim_p_pe] = W_Q[:, old_col:old_col+kv_dim_p_pe]
# W_O: same permutation on row dimension
```

**Result (interleaved Q layout):**
```
PE 0: Q_head0[0:4] | Q_head1[0:4]   /  K^T[0:4]   ← both match ✓
PE 1: Q_head0[4:8] | Q_head1[4:8]   /  K^T[4:8]   ← both match ✓
```

→ **gqa_group_size GEMVs per PE** (one per Q-head sub-block), `Kt = kv_dim_p_pe`.

---

## Phase 2: GQA Cache Design

**Compact cache** — `kv_dim_p_pe` rows per PE (NOT replicated):

```python
kv_head     = px // pes_p_kv_head
s           = px %  pes_p_kv_head
k_row_start = kv_head * head_dim + s * kv_dim_p_pe

XKCache_pe[px, py] = K_T[k_row_start : k_row_start+kv_dim_p_pe,
                         py*seq_len_p_pe : (py+1)*seq_len_p_pe]
```

**Memory comparison** (LLaMA-3 70B: `n_heads=64, n_kv_heads=8, head_dim=128, P=256, seq=4096`):

| Option | K^T cache per PE | Reduction |
|--------|-----------------|-----------|
| Option A (Q-aligned, replicated) | 256 KB | — |
| **Option C (compact, preferred)** | **32 KB** | **8× smaller** |

At `gqa_group_size=8`, Option A may **exceed PE SRAM**. Option C is the only viable choice.

---

## Phase 2: What Changes vs. Phase 1

```
rmsnorm_x()                        UNCHANGED
xq_matvec_mult()                   UNCHANGED (W_Q permuted offline)
xk_matvec_mult()                   Nt = kv_dim_p_pe  [CHANGED]
xv_matvec_mult()                   Nt = kv_dim_p_pe  [CHANGED]
all_reduce_bsz_dim_QKV_fusion()    extent = bsz*(dim_p_pe + 2*kv_dim_p_pe)  [CHANGED]
xq_rope()                          loop gqa_group_size sub-blocks, interleaved freqs  [CHANGED]
xk_rope()                          extent kv_dim_p_pe/2, new freqs_k arrays  [CHANGED]
reconfig_allreduce_axis(3)         KV-head-scoped X (new axis=3)  [CHANGED]
score_matvec_mult()                Kt=kv_dim_p_pe, gqa_group_size GEMVs  [CHANGED]
softmax_score()                    called gqa_group_size times  [CHANGED: loop]
output_matvec_mult()               Vt=kv_dim_p_pe, gqa_group_size GEMVs  [CHANGED]
o_matvec_mult()                    UNCHANGED (W_O permuted offline)
attn_residual_add() ... FFN        ALL UNCHANGED
```

**New route structs:** `X_KV_HEAD_2nd_RD_0/1`, `X_KV_HEAD_BD` — KV-head scoped (pes_p_kv_head boundary).

---

## Phase 3: Incremental KV Cache

Ports WSE-3-KV pattern to WSE-3-GQA — **Phase 2 compute is completely unchanged**.

**Distribution scheme:** Token at step `t` cached at PE column `py = t % P`. PE (px, py) writes only when `py == t % P`. `iter_num` tracks tokens per PE.

**Tensor changes:**

| Tensor | WSE-3-KV | WSE-3-GQA Phase 3 |
|--------|----------|-------------------|
| `XKCache_tile` | `[bsz × dim_p_pe × max_seq_len_p_pe]` | `[bsz × kv_dim_p_pe × max_seq_len_p_pe]` |
| `XVCache_tile` | `[bsz × max_seq_len_p_pe × dim_p_pe]` | `[bsz × max_seq_len_p_pe × kv_dim_p_pe]` |

**New functions only:**
- `process_kv()` — write new K/V token to cache at `iter_num`
- `vecmat_computation_var_right_2d_strided(max_seq_len_p_pe)` — variable-length K^T GEMV
- `all_reduce_bsz_var_seq_len_kv_head_scoped(iter_num)` — KV-head-scoped reduce with `@set_dsd_length`

---

## Summary: What Each Phase Adds

```
Base WSE-3         Phase 1 (MHA)          Phase 2 (GQA)          Phase 3 (KV Cache)
─────────────────  ─────────────────────  ─────────────────────  ──────────────────
n_heads = 1 only   n_heads > 1            n_kv_heads < n_heads   on-chip KV storage
                   head-scoped X-reduce   compact K^T/V          process_kv()
                   axis=2, X_HEAD routes  W_Q/W_O permutation    variable-length GEMV
                   pes_p_head param       interleaved Q layout    @set_dsd_length
                   groups_per_head        axis=3, KV-HEAD routes  iter_num tracking
                   head-local RoPE        gqa_group_size loops
```

---

## Constraints

| Constraint | Reason |
|-----------|--------|
| `n_heads \| group_num` | Head boundaries must align with comm group boundaries |
| `P % n_heads == 0` | `pes_p_head` must be integer |
| `P % n_kv_heads == 0` | `pes_p_kv_head` must be integer |
| `(n_kv_heads × head_dim) % P == 0` | `kv_dim_p_pe` must be integer |
| `n_heads % n_kv_heads == 0` | `gqa_group_size` must be integer |
| `dim % P == 0` | Standard PE sharding |
| `n_heads=1, group_num=1` degenerate | Must match original WSE-3 output |

**Test configs:**
- `mha_test.json`: `P=8, n_heads=4, n_kv_heads=4, group_num=4`
- `gqa_test.json`: `P=8, n_heads=4, n_kv_heads=2, group_num=4`
- Degenerate: `n_heads=4, n_kv_heads=4` through Phase 2 code → must match Phase 1 output
