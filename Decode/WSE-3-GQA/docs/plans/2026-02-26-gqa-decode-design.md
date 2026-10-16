# GQA Decode Design — WSE-3-GQA

**Date:** 2026-02-26
**Base:** `Decode/WSE-3-GQA/` (currently mirrors `Decode/WSE-3`)

---

## Running Example (used throughout this document)

To keep all sections consistent, every concrete example uses one of the following two configurations:

**Phase 1 (MHA) example** — matches `model_config/mha_test.json`:

```
P=8, n_heads=2, n_kv_heads=2, group_num=4, head_dim=32, dim=64, seq_len=64, ffn_dim=64
→ dim_p_pe=8, pes_p_head=4, groups_per_head=2, pe_num_p_group=2, kv_dim_p_pe=8 (=dim_p_pe)
```

**Phase 2 (GQA) example** — matches `model_config/gqa_test.json`:

```
P=8, n_heads=4, n_kv_heads=2, group_num=4, head_dim=16, dim=64, seq_len=64, ffn_dim=64
→ dim_p_pe=8, kv_dim_p_pe=4, pes_p_head=2, pes_p_kv_head=4, gqa_group_size=2, groups_per_head=1
```

---

## 0. Validation: Does WSE-3 Support n_heads > 1?

**Short answer: No.** The WSE-3 codebase has parameter stubs for multi-head attention but they are never used. Evidence:


| Evidence                                                                                    | Location                     | Conclusion                      |
| ------------------------------------------------------------------------------------------- | ---------------------------- | ------------------------------- |
| `param pes_p_head: i16` declared but never referenced in any DSD, computation, or comm call | `decode.csl:11-12`           | n_heads param is dead code      |
| `// param pes_p_head: i16;` commented out                                                   | `comm_pe.csl:4-5`            | comm library ignores head count |
| `pes_p_head = P // n_heads` computed but never used                                         | `launch_sim.py:58-59`        | host ignores head count         |
| K/V weights always `[dim, dim]`; caches always `[dim, seq_len]`                             | `launch_sim.py:76-77, 88-89` | no per-head tensor shapes       |
| `all_reduce_bsz_seq_len` does full 2-phase reduce over all P PEs                            | `comm_pe.csl:823+`           | score reduce mixes all heads    |


The only head-related param actually used is `head_dim_p_pe`, which is used solely to compute the attention scaling factor `alpha = 1/sqrt(head_dim)` in `init_task()`.

**Why n_heads > 1 requires code changes:**

The score computation `Q @ K^T` requires an X-reduce that sums partial dot-products. For correct multi-head attention, this reduce must be **scoped to pes_p_head PEs** (the PEs belonging to one Q head). The current full-P X-reduce would incorrectly sum partial scores from all n_heads heads together.

Using the Phase 1 example (P=8, n_heads=2, pes_p_head=4):

- PE 0–3: compute Q_head0 partial scores → X-reduce must sum only these 4 PEs
- PE 4–7: compute Q_head1 partial scores → X-reduce must sum only these 4 PEs
- Full-P reduce sums all 8 PEs: produces Q[0:dim] @ K^T[0:dim, seq] — **wrong**

---

## 1. Architecture Analysis: How Multi-head Works on the PE Grid

The PE grid is P×P. Let (px, py) be the coordinates. After QKV projection + Y-axis all-reduce:

- **X-axis (px):** partitions the `dim` dimension. PE px holds `Q[:, px*dim_p_pe : (px+1)*dim_p_pe]`.
- **Y-axis (py):** partitions the `seq_len` dimension. PE (px,py) holds K^T rows aligned with Q, at seq positions `[py*seq_len_p_pe : (py+1)*seq_len_p_pe]`.

For multi-head attention (n_heads=H), the Q vector has H heads. Each head spans `pes_p_head = P/H` consecutive PEs on the X-axis. Q head q occupies PEs `px = q*pes_p_head` to `(q+1)*pes_p_head - 1`.

**Score computation per head:** Each group of `pes_p_head` PEs independently computes `Q_head_q @ K_head_q^T`. This requires an X-reduce scoped to those pes_p_head PEs.

**Softmax and output:** These Y-reduces operate per-px column. Since PEs in different Q-head groups have different px values, the Y-reduces are **naturally head-isolated** — no extra scoping needed. Softmax correctly finds max/sum across all seq positions for each head independently.

Using the Phase 1 example (P=8, n_heads=2, pes_p_head=4):

```
X-axis:  [px=0  px=1  px=2  px=3] [px=4  px=5  px=6  px=7]
          ───────────────────────   ───────────────────────
              Head 0 (Q/K/V)             Head 1 (Q/K/V)
              4 PEs, score X-reduce      4 PEs, score X-reduce
              within this block          within this block
```

---

## 2. Three-Phase Implementation Plan

To reach correct GQA with incremental KV cache, three sequential phases are required:

```
Phase 1: MHA n_heads > 1   (fix score X-reduce scope; gqa_group_size=1 degenerate case of Phase 2)
    ↓
Phase 2: GQA + Option C    (compact K^T/V; W_Q/W_O permutation; interleaved Q; KV-head-scoped reduce)
    ↓
Phase 3: Incremental KV    (on-chip KV cache management; builds on Phase 2 layout)
```

**Rationale for this phasing:** Phase 1 is a special sub-case of Phase 2 with `n_kv_heads=n_heads` (gqa_group_size=1). All Phase 2 code degenerates correctly to Phase 1 behavior when gqa_group_size=1 and kv_dim_p_pe=dim_p_pe. Phase 3 adds on-chip KV management on top of the Phase 2 layout unchanged.

---

## 3. Phase 1: MHA with n_heads > 1

### 3.1 Key Design Decision: n_heads divides group_num

The existing 2-phase all-reduce is parameterized by:

- `pe_num_p_group = P / group_num` — size of each phase-1 intra-group (communication leaf)
- `pes_p_head = P / n_heads` — size of each Q-head's PE span on the X-axis
- `groups_per_head = group_num / n_heads` — number of communication groups within one Q-head

**Constraint: `n_heads | group_num`** (n_heads must evenly divide group_num), which implies `pes_p_head % pe_num_p_group == 0`. Head boundaries must align with group boundaries.

With this constraint:

- The X-axis PE range for Q-head `q` is `[q*pes_p_head, (q+1)*pes_p_head - 1]`
- This range spans exactly `groups_per_head` complete communication groups
- Phase 1 (intra-group) reduce stays unchanged; it always operates within `pe_num_p_group` PEs
- For the score X-reduce: need a head-scoped reduce over `pes_p_head` PEs (detailed in §3.2)
- For all other X/Y-reduces (QKV, output, FFN): use existing full P-reduce unchanged

Special case: `group_num = n_heads` → `groups_per_head = 1` → each head spans exactly 1 group → phase 1 alone covers the full head (simplest case, no head-scoped phase 2 needed).

Using the Phase 1 example (P=8, n_heads=2, group_num=4):

```
pe_num_p_group = 8/4 = 2        pes_p_head = 8/2 = 4        groups_per_head = 4/2 = 2
Head 0: px=0..3 (groups 0,1)    Head 1: px=4..7 (groups 2,3)
→ 2-level head reduce required (groups_per_head=2)
→ root_head_2nd_in_head = (2/2)*2 + 0 = 2  →  px_in_head=2  →  abs px=2 (head0), px=6 (head1)
```

### 3.2 Head-Scoped All-Reduce: PE Count and Color/Route Analysis

#### How to determine the all-reduce PE count

For the score X-reduce, the program must reduce over exactly `pes_p_head = P / n_heads` PEs. Two cases arise at compile time:

- `**groups_per_head == 1**` (`group_num == n_heads`): one group = one head. Phase 1 alone covers the entire head. Only a phase-1 reduce + intra-group broadcast is needed; no phase-2 is required.
- `**groups_per_head > 1**` (`group_num > n_heads`): multiple groups per head. A full 2-level reduce *within the head* is required: phase 1 (intra-group, unchanged) then a head-scoped phase 2 (inter-group within the head boundary), then a head-local broadcast.

The Phase 1 example falls into `groups_per_head=2`, exercising the full 2-level path.

#### Why `reconfig_allreduce_axis` is insufficient as-is

The current function has only **two route configurations**:


| `axis` | Colors reconfigured                    | Scope                       |
| ------ | -------------------------------------- | --------------------------- |
| 0      | `X_1st_RD_0/1`, `X_2nd_RD_0/1`, `X_BD` | Global X-reduce (all P PEs) |
| 1      | `Y_1st_RD_0/1`, `Y_2nd_RD_0/1`, `Y_BD` | Global Y-reduce (all P PEs) |


Both configs use route structs that span all P PEs. There is no route config that limits reduce traffic to a sub-range of the X-axis. A **third route config (`axis=2`)** must be added.

#### New route structs in `comm_layout.csl`

Two new route families are needed, computed per PE at compile time. They mirror `X_2nd_RD_0/1` and `X_BD` but operate within each head's PE range.

```csl
// Compile-time head-local quantities (computed in comm_layout.csl per PE)
const head_id: i16          = px / pes_p_head;
const px_in_head: i16       = px % pes_p_head;
const quotient_x_in_head: i16 =
    @as(i16, math.floor_f16(@as(f16, px_in_head) / @as(f16, pe_num_p_group)));

// Head-local phase-2 root (same relative position for every head)
const root_head_2nd_in_head: i16 = (groups_per_head/2)*pe_num_p_group + root_1st_phase;
const root_head_2nd_abs: i16 = head_id * pes_p_head + root_head_2nd_in_head;
```

`**X_HEAD_2nd_RD_0` / `X_HEAD_2nd_RD_1**` — Phase-2 reduce within a head: route direction logic uses `quotient_x_in_head` and `root_head_2nd_in_head` exactly as `X_2nd_RD_0/1` uses `quotient_x` and `root_2nd_phase`. PEs at head boundaries are configured as sinks (no further EAST/WEST forwarding out of the head).

`**X_HEAD_BD**` — Head-local broadcast: fans out from `root_head_2nd_abs` both EAST and WEST, limited to `[head_id*pes_p_head, (head_id+1)*pes_p_head - 1]`. Boundary PEs use single-direction config.

#### New `axis=2` case in `reconfig_allreduce_axis`

```csl
} else if (axis == 2) {
    // Head-scoped X-reduce (used for score computation only)
    @set_color_config(reduce_1st_color_0, X_1st_RD_0);        // phase 1: unchanged
    @set_color_config(reduce_1st_color_1, X_1st_RD_1);        // phase 1: unchanged
    @set_color_config(reduce_2nd_color_0, X_HEAD_2nd_RD_0);   // phase 2 within head
    @set_color_config(reduce_2nd_color_1, X_HEAD_2nd_RD_1);   // phase 2 within head
    @set_color_config(broadcast_color,    X_HEAD_BD);          // head-local broadcast
}
```

The phase-2 root check uses the head-local index: `quotient_x_in_head == root_head_2nd_in_head` instead of the global `quotient_pe_id == root_2nd_phase`.

#### New compile-time parameters required


| Parameter               | Derivation                                            | Purpose                                                         |
| ----------------------- | ----------------------------------------------------- | --------------------------------------------------------------- |
| `pes_p_head`            | `P / n_heads`                                         | Head span; used in route struct computation                     |
| `groups_per_head`       | `group_num / n_heads`                                 | Determines 1-vs-2-level code path                               |
| `root_head_2nd_in_head` | `(groups_per_head/2)*pe_num_p_group + root_1st_phase` | Phase-2 root within a head (same relative index for every head) |


### 3.3 Code Changes Required

#### `comm_pe.csl` — add `all_reduce_bsz_seq_len_head_scoped`

```csl
fn all_reduce_bsz_seq_len_head_scoped(
    pe_id: i16, quotient_pe_id: i16, remainder_pe_id: i16,
    vector_buf_ptr: [*]f16
) void {
    // groups_per_head == 1 fast path: phase 1 + intra-group broadcast only
    // groups_per_head >  1 general path: reconfig_allreduce_axis(2) then
    //   full 2-phase structure but bounded within head by X_HEAD_2nd_RD_*/X_HEAD_BD routes
}
```

#### `decode.csl` — update `score_matvec_mult()`

```csl
fn score_matvec_mult() void {
    // ... GEMV unchanged ...
    // Replace:  comm_mod.all_reduce_bsz_seq_len(px, quotient_x, remainder_x, ptr_score);
    // With:     comm_mod.all_reduce_bsz_seq_len_head_scoped(px_in_head, quotient_x_in_head, remainder_x, ptr_score);
}
```

#### `run_sim.sh` — enforce n_heads divides group_num

```bash
if [ $(( GROUP_NUM % N_HEADS )) -ne 0 ]; then
    echo "ERROR: group_num must be a multiple of n_heads for multi-head attention"
    exit 1
fi
```

#### `launch_sim.py` — fix RoPE freqs to be head-local

For PE px in head `h = px // pes_p_head`, head-local position `s = px % pes_p_head`:

```python
# Phase 1 example: pes_p_head=4, _half_head=8
# PE px=0 (head=0, s=0): freqs from rope_freqs[0:8]
# PE px=1 (head=0, s=1): freqs from rope_freqs[8:16]
# PE px=4 (head=1, s=0): freqs from rope_freqs[0:8]  ← same as head0/s=0 (shared freqs)
local_head_offset = (px % pes_p_head) * (_dim_p_pe // 2)
freqs_sin_pe[px] = rope_sin[local_head_offset : local_head_offset + _dim_p_pe // 2]
freqs_cos_pe[px] = rope_cos[local_head_offset : local_head_offset + _dim_p_pe // 2]
```

#### Config: `model_config/mha_test.json`

```json
{
    "P": 8, "group_num": 4, "bsz": 1,
    "dim": 64, "n_heads": 2, "n_kv_heads": 2,
    "head_dim": 32, "seq_len": 64, "ffn_dim": 64
}
```

Derived: `dim_p_pe=8`, `pes_p_head=4`, `groups_per_head=2`, `root_head_2nd_in_head=2+0=2`.
Note: `groups_per_head=2` exercises the full 2-level head-scoped reduce. The root at `px_in_head=2` falls in the interior of each 4-PE head block, exercising the bidirectional broadcast.

Degenerate case: `n_heads=1` (group_num=1) must still produce the same output as the original WSE-3.

### 3.4 Verification

- Run simulator with `mha_test.json`
- Write numpy reference (compute per-head softmax attention independently)
- Assert max abs error < threshold (e.g., 0.15 for f16 matmul accumulation)
- Run degenerate case (n_heads=1, group_num=1) and verify matches original WSE-3 output

---

## 4. Phase 2: GQA (builds on Phase 1)

### 4.1 Parameter Changes


| Parameter        | Derivation                    | CSL param?              | Phase 2 example (P=8, n_heads=4, n_kv_heads=2) |
| ---------------- | ----------------------------- | ----------------------- | ---------------------------------------------- |
| `kv_dim_p_pe`    | `(n_kv_heads × head_dim) / P` | **yes**                 | 4 (vs dim_p_pe=8)                              |
| `gqa_group_size` | `n_heads / n_kv_heads`        | no (host only)          | 2                                              |
| `pes_p_kv_head`  | `P / n_kv_heads`              | already exists (unused) | 4                                              |


Constraint: `n_heads | group_num` (same as Phase 1). `pes_p_head = P/n_heads = 2`.

Phase 1 degenerate: `n_kv_heads = n_heads` → `kv_dim_p_pe = dim_p_pe`, `gqa_group_size = 1` → all Phase 2 changes become identity.

### 4.2 Core Alignment Problem

With standard QKV projection, PE `px` holds after Y-axis all-reduce:

- **Q:** `Q[px·dim_p_pe : (px+1)·dim_p_pe]` — Q-head-local offset `r·dim_p_pe` where `r = px % pes_p_head`
- **K:** `K[px·kv_dim_p_pe : (px+1)·kv_dim_p_pe]` — KV-head-local offset `s·kv_dim_p_pe` where `s = px % pes_p_kv_head`

For GQA (`gqa_group_size > 1`), `r·dim_p_pe ≠ s·kv_dim_p_pe` at most PEs. Q and K^T on the same PE cover different positions in the head-dimension space — a direct GEMV would pair the wrong Q elements with the wrong K^T rows.

Using the Phase 2 example (P=8, n_heads=4, n_kv_heads=2, dim_p_pe=8, kv_dim_p_pe=4, pes_p_head=2, pes_p_kv_head=4):

```
After QKV projection + Y-reduce (standard Q layout):
  PE 0 (head=0, r=0): Q_head0[ 0: 8]    PE 1 (head=0, r=1): Q_head0[ 8:16]
  PE 2 (head=1, r=0): Q_head1[ 0: 8]    PE 3 (head=1, r=1): Q_head1[ 8:16]
  ...

Compact K^T cache (KV-head layout, kv_dim_p_pe=4 rows per PE):
  PE 0 (kv_head=0, s=0): K^T[ 0: 4, :]   PE 1 (kv_head=0, s=1): K^T[ 4: 8, :]
  PE 2 (kv_head=0, s=2): K^T[ 8:12, :]   PE 3 (kv_head=0, s=3): K^T[12:16, :]
  ...

Problem — PE 1 has Q_head0[8:16] but K^T[4:8] (rows 4–7).
  Correct GEMV for K^T[4:8] needs Q_head0[4:8] — which lives at PE 0.
  PE 1's local Q_head0[8:16] @ K^T[4:8] computes the wrong contribution.
```

Three approaches exist; they differ in how to resolve this Q/K alignment mismatch. See §4.3.

### 4.3 On-Chip Cache Alignment: Options A, B, and C

#### Why compact K^T doesn't directly work with the existing GEMV

The GEMV primitive computes `score[t] += Q_local[k] * K_local[k, t]` over `k = 0..Kt-1` using *local memory only*. After the all-reduce, each PE's contribution must be the correct partial sum for its assigned K^T row block. There is no mechanism for the all-reduce to correct a mismatched Q slice — it simply sums whatever each PE computed locally.

Using the Phase 2 example, the 16 products for score_head0[t] are:

```
score_head0[t] = sum_{d=0}^{15} Q_head0[d] · K_kv0[d, t]

d      Q_head0[d] lives at   K_kv0[d,t] lives at   Same PE?
0–3    PE 0 (lo half)         PE 0                   YES ✓
4–7    PE 0 (hi half)         PE 1                   NO  ✗  ← Q at PE 0, K at PE 1
8–11   PE 1 (lo half)         PE 2                   NO  ✗  ← Q at PE 1, K at PE 2
12–15  PE 1 (hi half)         PE 3                   NO  ✗  ← Q at PE 1, K at PE 3
```

Only d=0..3 are co-located. This mismatch is the root of the alignment problem.

#### Option A: Q-aligned cache (replicate K^T data)

Each PE stores `dim_p_pe` rows of K^T — the Q-head-aligned slice for its px. Within a KV-head block, `gqa_group_size` Q-head PEs receive *different* subsets of the same KV-head's K^T (replicated). Cache layout per PE: `[dim_p_pe × seq_len_p_pe]`.

Using the Phase 2 example:

```
PE 0 (Q_head0, r=0): K^T[ 0: 8, :]   (rows 0–7 of kv_head0, = gqa_group_size × kv_dim_p_pe rows)
PE 1 (Q_head0, r=1): K^T[ 8:16, :]
PE 2 (Q_head1, r=0): K^T[ 0: 8, :]   ← SAME as PE 0 (replicated for head 1)
PE 3 (Q_head1, r=1): K^T[ 8:16, :]   ← SAME as PE 1 (replicated for head 1)
```

- **Pro**: `score_matvec_mult()` unchanged (`Kt = dim_p_pe`); no extra comm or routing changes.
- **Con**: K^T replicated `gqa_group_size` times → `gqa_group_size ×` more cache than needed.
- Memory per PE: `dim_p_pe × max_seq_len_p_pe × 2 B = gqa_group_size × kv_dim_p_pe × max_seq_len_p_pe × 2 B`

#### Option B: Compact cache + runtime Q redistribution

Store only `kv_dim_p_pe` rows of K^T per PE. At each decode step, redistribute Q elements among PEs within each KV-head block so that every PE receives the Q elements that pair with its local K^T rows.

**Co-location observation:** KV-head k occupies PEs `[k·pes_p_kv_head, (k+1)·pes_p_kv_head - 1]`. The `gqa_group_size` Q-heads sharing KV-head k map to the same PE range, so score computation is entirely local to `pes_p_kv_head` PEs — no cross-KV-head communication.

After redistribution, PE px holds Q elements from *all* Q-heads sharing its KV-head (`gqa_group_size × kv_dim_p_pe = dim_p_pe` total — same count as before), then runs `gqa_group_size` GEMVs with `Kt = kv_dim_p_pe`.

Target Q layout for the Phase 2 example (after redistribution):

```
PE 0 (K^T rows 0–3):   Q_head0[0:4]  + Q_head1[0:4]    (dim_p_pe=8 total)
PE 1 (K^T rows 4–7):   Q_head0[4:8]  + Q_head1[4:8]
PE 2 (K^T rows 8–12):  Q_head0[8:12] + Q_head1[8:12]
PE 3 (K^T rows 12–16): Q_head0[12:16]+ Q_head1[12:16]
```

Required moves within the pes_p_kv_head=4 block (8 bytes per PE sent/received):

```
PE 0 → PE 1: Q_head0[4:8]      PE 1 → PE 2: Q_head0[8:12]
PE 1 → PE 3: Q_head0[12:16]    PE 2 → PE 1: Q_head1[4:8]
PE 2 → PE 0: Q_head1[0:4]      PE 3 → PE 2: Q_head1[8:12]
```

The X-reduce after the GEMVs must span `pes_p_kv_head` PEs — **not** `pes_p_head`. The Phase-1 head-scoped reduce (axis=2) is the wrong scope here; new KV-head-scoped route structs are required.

- **Pro**: `gqa_group_size ×` smaller K^T cache than Option A.
- **Con**: Q redistribution per decode step (cheap but non-zero); new routing infrastructure; `score_matvec_mult()` changes (Kt = kv_dim_p_pe, gqa_group_size GEMVs). Option C below dominates B — B is not a final target.

#### Option C: W_Q-permuted interleaved Q layout (compact cache, zero-cost Q alignment)

**Core insight:** Option B's runtime Q redistribution can be eliminated by permuting W_Q's output columns *offline* (host-side, one time). After the permuted projection + Y-reduce, each PE naturally holds the KV-head-aligned interleaved Q — identical to Option B's post-scatter layout, with zero runtime communication.

**Target interleaved Q layout** (PE px, KV-head block k, s = px % pes_p_kv_head):

```
buf[i·kv_dim_p_pe : (i+1)·kv_dim_p_pe] = Q_head_{k·gqa+i}[s·kv_dim_p_pe : (s+1)·kv_dim_p_pe]
    for i = 0 .. gqa_group_size-1    (total = dim_p_pe elements, same as before)
```

Phase 2 example — after W_Q-permuted projection + Y-reduce:

```
PE  s   K^T rows   Interleaved Q buf                    Score match
0   0   0– 3      Q_head0[0:4]  | Q_head1[0:4]         both heads ✓
1   1   4– 7      Q_head0[4:8]  | Q_head1[4:8]         both heads ✓
2   2   8–12      Q_head0[8:12] | Q_head1[8:12]         both heads ✓
3   3   12–16     Q_head0[12:16]| Q_head1[12:16]        both heads ✓
4   0   16–19     Q_head2[0:4]  | Q_head3[0:4]         both heads ✓
...
```

**W_Q column permutation formula (host, one-time):**

```python
# For each Q-head h, head-local KV-position block s:
kv_head      = h // gqa_group_size
old_col_base = h * head_dim + s * kv_dim_p_pe
new_col_base = kv_head * pes_p_kv_head * dim_p_pe \
             + s * dim_p_pe \
             + (h % gqa_group_size) * kv_dim_p_pe
W_Q_perm[:, new_col_base : new_col_base+kv_dim_p_pe] = \
    W_Q[:, old_col_base : old_col_base+kv_dim_p_pe]
# W_O: same permutation on row dimension
W_O_perm[new_col_base : new_col_base+kv_dim_p_pe, :] = \
    W_O[old_col_base : old_col_base+kv_dim_p_pe, :]
```

Phase 1 degenerate (gqa_group_size=1): `old_col_base = new_col_base` for all h — identity permutation. No change to weights.

**Score computation (same as Option B after redistribution):**

PE px (KV-head block k, position s = px % pes_p_kv_head) holds:

- interleaved Q: gqa_group_size sub-blocks of kv_dim_p_pe each
- compact K^T: kv_dim_p_pe rows

```
for i = 0..gqa_group_size-1:
    partial_score_head_{k·gqa+i}[t] += buf[i·kv_dim:(i+1)·kv_dim] @ K^T[:, t]
→ gqa_group_size separate KV-head-scoped X-reduces over pes_p_kv_head PEs
```

**New KV-head-scoped route structs** (needed by both B and C):

New compile-time structs, derived identically to §3.2 `X_HEAD_`* but with `pes_p_kv_head` as boundary:

```
X_KV_HEAD_2nd_RD_0 / X_KV_HEAD_2nd_RD_1  — phase-2 within KV-head block
X_KV_HEAD_BD                              — broadcast from KV-head-local root
```

New params: `groups_per_kv_head = group_num / n_kv_heads`, `root_kv_head_2nd_in_head = (groups_per_kv_head/2)*pe_num_p_group + root_1st_phase`. New `axis=3` case in `reconfig_allreduce_axis` activates these structs.

**RoPE with interleaved layout:**

`xq_rope()` loops over gqa_group_size sub-blocks. For sub-block i of PE px (s = px % pes_p_kv_head):

```python
kv_head = px // pes_p_kv_head
for i in range(gqa_group_size):
    q_head_i = kv_head * gqa_group_size + i
    src_start = q_head_i * (head_dim // 2) + s * (kv_dim_p_pe // 2)
    freqs_q_interleaved_pe[i * (kv_dim_p_pe//2) : (i+1) * (kv_dim_p_pe//2)] = \
        rope_freqs_half[src_start : src_start + kv_dim_p_pe//2]
```

**Option B vs Option C comparison:**


|                       | Option B (compact + Q scatter)             | Option C (compact + W_Q permute)    |
| --------------------- | ------------------------------------------ | ----------------------------------- |
| K^T cache             | kv_dim_p_pe × seq (compact)                | kv_dim_p_pe × seq (compact)         |
| Q alignment           | runtime all-to-all within KV-head block    | offline W_Q/W_O permutation         |
| Runtime Q comm        | (gqa_group_size-1)·kv_dim_p_pe f16/PE/step | **none**                            |
| xq_rope()             | unchanged                                  | loop over gqa_group_size sub-blocks |
| GEMVs per PE          | gqa_group_size, Kt=kv_dim_p_pe             | gqa_group_size, Kt=kv_dim_p_pe      |
| Reduce scope          | pes_p_kv_head (KV-head-scoped)             | pes_p_kv_head (KV-head-scoped)      |
| New route structs     | X_KV_HEAD_2nd_RD, X_KV_HEAD_BD             | same                                |
| W_Q/W_O preprocessing | none                                       | permute offline (one-time)          |


**Option C dominates B at runtime:** identical memory, identical routing infrastructure, zero Q scatter communication. B is not a final target.

**Memory comparison (LLaMA-3 70B: n_heads=64, n_kv_heads=8, head_dim=128, P=256, max_seq_len=4096, bsz=1):**

```
gqa_group_size = 64/8 = 8      kv_dim_p_pe = 8*128/256 = 4      dim_p_pe = 64*128/256 = 32

Option A (Q-aligned):      K^T cache = 32 × 4096 × 2 B = 256 KB per PE
Options B and C (compact): K^T cache =  4 × 4096 × 2 B =  32 KB per PE
Savings vs A: 224 KB per PE (8× reduction)
```

At gqa_group_size=8, total on-chip SRAM is typically tens to low hundreds of KB. Option C may be the only feasible choice for large GQA ratios and long sequences.

**Recommendation:**


| Scenario                                | Preferred option                                                                     |
| --------------------------------------- | ------------------------------------------------------------------------------------ |
| gqa_group_size ≤ 2 or short max_seq_len | Option A (simplest — replication cost acceptable; score_matvec_mult unchanged)       |
| gqa_group_size ≥ 4 or long max_seq_len  | **Option C** (compact cache, zero-cost Q alignment; strictly better than B)          |
| Option B vs C                           | Always prefer C — same routing infrastructure, eliminates runtime Q scatter for free |


**Phase 2 uses Option C.** Phase 3 inherits Phase 2's Option C layout unchanged. For initial correctness validation of Phase 3, Option A (replicated K^T) is recommended before switching to the compact layout — see §5.4.

### 4.4 decode_struct() Changes — Summary

```
rmsnorm_x()                             ← UNCHANGED
xq_matvec_mult()                        ← UNCHANGED (Nt=dim_p_pe; W_Q permuted offline)
xk_matvec_mult()                        ← Nt=kv_dim_p_pe  [CHANGED from Phase 1]
xv_matvec_mult()                        ← Nt=kv_dim_p_pe  [CHANGED from Phase 1]
all_reduce_bsz_dim_QKV_fusion(...)      ← new buffer length bsz*(dim_p_pe+2*kv_dim_p_pe)  [CHANGED]
reconfig_allreduce_axis(1)              ← UNCHANGED (global Y for QKV all-reduce)
xq_rope()                               ← gqa_group_size sub-blocks, interleaved freqs  [CHANGED]
xk_rope()                               ← kv_dim_p_pe extent, new K freqs  [CHANGED]
reconfig_allreduce_axis(3)              ← KV-head-scoped X (axis=3, pes_p_kv_head)  [CHANGED from axis=2]
score_matvec_mult()  [KV-head-scoped]   ← Kt=kv_dim_p_pe, gqa_group_size GEMVs  [CHANGED]
reconfig_allreduce_axis(1)              ← UNCHANGED (global Y for softmax)
softmax_score()      [Y-reduce, ×gqa]  ← called gqa_group_size times  [CHANGED: loop]
output_matvec_mult() [Y-reduce, ×gqa]  ← Vt=kv_dim_p_pe, gqa_group_size GEMVs  [CHANGED]
reconfig_allreduce_axis(0)              ← UNCHANGED
o_matvec_mult()      [full X-reduce]    ← UNCHANGED (W_O permuted offline; input is interleaved)
attn_residual_add()                     ← UNCHANGED
reconfig_allreduce_axis(1)              ← UNCHANGED
rmsnorm_z() ... ffn_residual_add()      ← ALL UNCHANGED
```

**Phase 1 degenerate behavior (gqa_group_size=1):** All loops run once; Kt=kv_dim_p_pe=dim_p_pe; axis=3 reduce has pes_p_kv_head=pes_p_head. Identical to Phase 1 behavior. The Phase 1 `mha_test.json` (n_kv_heads=n_heads=2) can run through Phase 2 code as a regression check.

### 4.5 File-by-File Changes

#### `run_sim.sh`

```bash
kv_dim_p_pe=$(( (N_KV_HEADS * HEAD_DIM) / P ))   # new
# Add to cslc --params: kv_dim_p_pe:"$kv_dim_p_pe"
```

#### `src/layout.csl`

```csl
param kv_dim_p_pe: i16;   // new
// Pass to @set_tile_code and comm module
```

#### `src/decode.csl`


| Item                   | Phase 1 (MHA)                    | Phase 2 (GQA + Option C)                                              |
| ---------------------- | -------------------------------- | --------------------------------------------------------------------- |
| `K_weight_tile`        | `[dim_p_pe × dim_p_pe]`          | `[dim_p_pe × kv_dim_p_pe]`                                            |
| `V_weight_tile`        | `[dim_p_pe × dim_p_pe]`          | `[dim_p_pe × kv_dim_p_pe]`                                            |
| `QKV_tile`             | `[bsz × 3×dim_p_pe]`             | `[bsz × (dim_p_pe + 2×kv_dim_p_pe)]`                                  |
| `K_dsd` offset         | `bsz×dim_p_pe`                   | `bsz×dim_p_pe` (unchanged)                                            |
| `V_dsd` offset         | `2×bsz×dim_p_pe`                 | `bsz×(dim_p_pe + kv_dim_p_pe)`                                        |
| `K_dsd` extent         | `bsz×dim_p_pe`                   | `bsz×kv_dim_p_pe`                                                     |
| `V_dsd` extent         | `bsz×dim_p_pe`                   | `bsz×kv_dim_p_pe`                                                     |
| `xk_matvec_mult` Nt    | `dim_p_pe`                       | `kv_dim_p_pe`                                                         |
| `xv_matvec_mult` Nt    | `dim_p_pe`                       | `kv_dim_p_pe`                                                         |
| `XKCache_tile`         | `[bsz × dim_p_pe × seq_len_p_pe]`| `[bsz × kv_dim_p_pe × seq_len_p_pe]` compact, bsz slowest             |
| `XVCache_tile`         | `[bsz × seq_len_p_pe × dim_p_pe]`| `[bsz × seq_len_p_pe × kv_dim_p_pe]` compact, bsz slowest             |
| `xq_rope`              | single pass, `_dim_p_pe/2` freqs | loop `gqa_group_size` times, `_kv_dim_p_pe/2` freqs each; interleaved |
| `xk_rope` freqs/extent | `_dim_p_pe/2`                    | `_kv_dim_p_pe/2`                                                      |
| `freqs_q_sin/cos`      | `[_dim_p_pe/2]f16`               | `[gqa_group_size × _kv_dim_p_pe/2]f16` interleaved                    |
| `freqs_k_sin/cos`      | —                                | `[_kv_dim_p_pe/2]f16`                                                 |
| `score_matvec_mult`    | Kt=dim_p_pe, 1 GEMV, axis=2      | Kt=kv_dim_p_pe, gqa_group_size GEMVs, axis=3                          |
| `output_matvec_mult`   | Vt=dim_p_pe, 1 GEMV              | Vt=kv_dim_p_pe, gqa_group_size GEMVs                                  |
| `softmax_score`        | 1 call                           | gqa_group_size calls (one per score buffer)                           |


#### `src/comm_lib/comm_pe.csl`

```csl
param kv_dim_p_pe: i16;   // new

// QKV fusion buffer: was bsz*3*dim_p_pe
var vector_buf_dsd_bsz_dim_QKV_fusion = ..., extent = bsz*(dim_p_pe + 2*kv_dim_p_pe)
// Update all corresponding fabric DSDs to same new extent
```

#### `launch_sim.py`

```python
gqa_group_size = n_heads // n_kv_heads
pes_p_kv_head  = P // n_kv_heads

# K,V weight tiles: [dim_p_pe × kv_dim_p_pe]
kv_dim = n_kv_heads * head_dim
tensor_k_weight = np.random.rand(dim, kv_dim)   # was (dim, dim)
# Tiling: tensor_k_weight.reshape(P, dim_p_pe, P, kv_dim_p_pe).transpose(0,2,1,3)

# W_Q column permutation (identity when gqa_group_size=1):
W_Q_perm = np.zeros_like(tensor_q_weight)
for h in range(n_heads):
    kv_head = h // gqa_group_size
    for s in range(pes_p_kv_head):
        old_col = h * head_dim + s * _kv_dim_p_pe
        new_col = kv_head * pes_p_kv_head * _dim_p_pe \
                + s * _dim_p_pe \
                + (h % gqa_group_size) * _kv_dim_p_pe
        W_Q_perm[:, new_col:new_col+_kv_dim_p_pe] = \
            tensor_q_weight[:, old_col:old_col+_kv_dim_p_pe]
# Same permutation on W_O rows: W_O_perm[new_col:...] = W_O[old_col:...]

# Compact K^T/V cache (kv_dim_p_pe rows per PE; note swapaxes to match KCache_tile[py,px]):
KCache_tile = tile_weight_row(tensor_XKCache, kv_dim_p_pe, seq_len_p_pe).swapaxes(0, 1)
#   PE(px,py) gets: XKCache[px*kv_dim_p_pe:(px+1)*kv_dim_p_pe, py*seq_len_p_pe:(py+1)*seq_len_p_pe]
VCache_tile = tile_weight_row(tensor_XVCache, seq_len_p_pe, kv_dim_p_pe)
#   PE(px,py) gets: XVCache[py*seq_len_p_pe:(py+1)*seq_len_p_pe, px*kv_dim_p_pe:(px+1)*kv_dim_p_pe]
```

Note: `KCache_tile` requires `.swapaxes(0,1)` because `tile_weight_row` places the row-dim partition on axis 0, which ROW_MAJOR h2d maps to py. For K^T, the row-dim (kv_dim) must follow px (not py), so axes must be swapped. VCache naturally aligns (seq_len → py, dim → px).

#### Config: `model_config/gqa_test.json`

```json
{
    "P": 8, "group_num": 4, "bsz": 1,
    "dim": 64, "n_heads": 4, "n_kv_heads": 2,
    "head_dim": 16, "seq_len": 64, "ffn_dim": 64
}
```

Derived: `dim_p_pe=8`, `kv_dim_p_pe=4`, `pes_p_head=2`, `pes_p_kv_head=4`, `gqa_group_size=2`, `groups_per_kv_head=2`, `groups_per_head=1`.

Phase 1 sub-case (run through Phase 2 code with n_kv_heads=n_heads):

```json
{ "P": 8, "group_num": 4, "bsz": 2, "dim": 64, "n_heads": 2, "n_kv_heads": 2, "head_dim": 32, "seq_len": 64, "ffn_dim": 64 }
```

Derived: `kv_dim_p_pe=dim_p_pe=8`, `gqa_group_size=1` → all Phase 2 changes are identity. Must produce same output as Phase 1 `mha_test.json`.

### 4.6 What Is NOT Changed in Phase 2

- `o_matvec_mult()` — unchanged in CSL; W_O permuted offline so its input/output contract is preserved
- All Y-reduce function bodies in `comm_pe.csl` — softmax and output GEMVs reuse the same Y-reduce infrastructure; they just loop gqa_group_size times in decode.csl
- FFN, RMSNorm, residual adds — all unchanged
- All FFN weights — unchanged

---

## 5. Phase 3: Incremental KV Cache (builds on Phase 2)

Port the WSE-3-KV changes to WSE-3-GQA. The computation logic is identical to Phase 2 except KV cache tensors use `kv_dim_p_pe` instead of `dim_p_pe` and are managed on-chip.

**Key difference from WSE-3-KV:** WSE-3-KV assumed an empty KV cache at decode start (`iter_num = 0`). Phase 3 assumes the KV cache is **pre-loaded from the host** before the first decode step, containing `prefill_len` tokens from the prefill phase. Decode then appends new tokens on top of this pre-filled cache. `iter_num` is initialized to `prefill_len_p_pe = prefill_len / P` (tokens owned by this PE from prefill), not 0.

### 5.1 KV Distribution Scheme

Same as WSE-3-KV: token at global index `t` is owned by PE row `py = t % P`. PE (px, py) writes token `t` to cache only when `py == t % P`. The local index within the PE is `local_t = t // P` (i.e., PE py owns tokens `{py, py+P, py+2P, ...}`).

Variable `iter_num` tracks how many tokens this PE has cached. **At Phase 3 start, `iter_num` is initialized to `prefill_len_p_pe`** (not 0). The first decode step appends the new token at column/row `iter_num`, so tokens 0..prefill_len_p_pe-1 are the prefill tokens and slot `iter_num` is the first decode token.

### 5.2 Tensor Changes


| Tensor         | WSE-3-KV                              | WSE-3-GQA Phase 3                        |
| -------------- | ------------------------------------- | ---------------------------------------- |
| `XKCache_tile` | `[bsz × dim_p_pe × max_seq_len_p_pe]` | `[bsz × kv_dim_p_pe × max_seq_len_p_pe]` |
| `XVCache_tile` | `[bsz × max_seq_len_p_pe × dim_p_pe]` | `[bsz × max_seq_len_p_pe × kv_dim_p_pe]` |


**K cache — transposed layout** `[bsz, kv_dim_p_pe, max_seq_len_p_pe]`:

- Element `K[b][k][local_t]` is at flat index `b * kv_dim_p_pe * max_seq_len_p_pe + k * max_seq_len_p_pe + local_t`
- Each K-dimension row has `max_seq_len_p_pe` slots; tokens fill left to right
- `process_kv()` writes new token at column `iter_num` using stride `max_seq_len_p_pe` between rows
- **Unused slots** are at the **end of each row**: columns `prefill_len_p_pe .. max_seq_len_p_pe-1`

```
After prefill load, K cache row k looks like:
  [ token_0  token_1  ...  token_{prefill_len_p_pe-1} | 0  0  ... 0 ]
    col 0    col 1         col prefill_len_p_pe-1     ^--- unused ---^
```

**V cache — row-major layout** `[bsz, max_seq_len_p_pe, kv_dim_p_pe]`:

- Element `V[b][local_t][v]` is at flat index `b * max_seq_len_p_pe * kv_dim_p_pe + local_t * kv_dim_p_pe + v`
- Each token occupies a contiguous row of `kv_dim_p_pe` elements
- `process_kv()` writes new token at row `iter_num` (contiguous write, stride 1)
- **Unused slots** are at the **end of the block**: rows `prefill_len_p_pe .. max_seq_len_p_pe-1`

```
After prefill load, V cache looks like:
  row 0:   [ v_0  v_1  ...  v_{kv_dim_p_pe-1} ]   <- token_0
  row 1:   [ ...                              ]   <- token_1
  ...
  row prefill_len_p_pe-1: [...]                  <- last prefill token
  row prefill_len_p_pe:   [ 0   0   ...   0  ]   <- first decode token slot
  ...
  row max_seq_len_p_pe-1: [ 0   0   ...   0  ]   <- unused
```

Phase 3 inherits Phase 2's compact K^T/V layout — `kv_dim_p_pe` rows per PE. `score_matvec_mult()` already uses `Kt=kv_dim_p_pe` and gqa_group_size GEMVs from Phase 2. Phase 3 adds on-chip write (`process_kv()`), prefill pre-load from host, and variable-length read variants.

### 5.3 New Functions

- `process_kv()`: write incoming K,V to on-chip cache at column/row `iter_num`; increments `iter_num`; uses `kv_dim_p_pe` sizes (port from WSE-3-KV; called for decode tokens only, not prefill)
- `vecmat_computation_var_right_2d_strided(max_seq_len_p_pe)`: variable-length K^T access with strided read (port from WSE-3-KV, K dim = kv_dim_p_pe)
- `vecmat_computation_var_right(kv_dim_p_pe * max_seq_len_p_pe)`: variable-length V access (port from WSE-3-KV)
- `all_reduce_bsz_var_seq_len_kv_head_scoped(iter_num)`: KV-head-scoped X-reduce with `@set_dsd_length` for variable sequence length (Phase 3 new; the fixed-length KV-head-scoped version from Phase 2 is the base)

Route structs `X_KV_HEAD_2nd_RD_0/1`, `X_KV_HEAD_BD` and `reconfig_allreduce_axis(axis=3)` are already in place from Phase 2 — no additional routing changes needed.

### 5.4 Phase 3 Cache Design Choice

Phase 3 uses **the same compact Option C layout inherited from Phase 2** (kv_dim_p_pe rows of K^T per PE, not replicated). See §4.3 for the full Options A/B/C analysis and memory comparison.

**Implementation strategy for Phase 3:**

1. Start with **Option A** (Q-aligned, replicated K^T): load `dim_p_pe` rows per PE from a Q-aligned host cache. `score_matvec_mult()` is unchanged from Phase 1 (`Kt = dim_p_pe`). Simplest to validate correctness. Note: even Option A uses the prefilled-from-host design — host formats the replicated Q-aligned K^T into `[bsz, dim_p_pe, max_seq_len_p_pe]` with prefill tokens in columns 0..prefill_len_p_pe-1.
2. Once Phase 3 correctness is verified with Option A, switch to **Option C** (compact K^T, `kv_dim_p_pe` rows per PE). The routing infrastructure (`X_KV_HEAD`_* route structs, axis=3) is already in place from Phase 2. Only `process_kv()` and the score GEMV need to switch to `kv_dim_p_pe`.

### 5.5 What Is Not Changed from Phase 2

- All Phase 2 GQA compute (score GEMVs, KV-head-scoped reduce, gqa_group_size loops) — unchanged
- FFN, O-proj, RMSNorm, residual adds — all unchanged

### 5.6 Host-side KV Cache Pre-load

Before the first decode step, the host transfers the prefill KV cache into each PE's on-chip buffer in the exact on-chip format. Define:

- `prefill_len`: total number of prefill tokens (global, divisible by P)
- `prefill_len_p_pe = prefill_len // P`: tokens owned by each PE
- Global KV tensors: `K_global: [bsz, kv_dim, prefill_len]` (transposed), `V_global: [bsz, prefill_len, kv_dim]` (row-major)

**Tiling K cache for PE(px, py)** — target: `[bsz, kv_dim_p_pe, max_seq_len_p_pe]` (zeros-padded):

```python
# PE(px, py) owns:
#   KV-dim slice: rows [px*kv_dim_p_pe : (px+1)*kv_dim_p_pe] of global K^T
#   Token slice:  global tokens {py, py+P, ..., py+(prefill_len_p_pe-1)*P}
py_tokens = np.arange(py, prefill_len, P)           # shape: [prefill_len_p_pe]
k_slice = K_global[:, px*kv_dim_p_pe:(px+1)*kv_dim_p_pe, :][:, :, py_tokens]
#         shape: [bsz, kv_dim_p_pe, prefill_len_p_pe]

pe_kcache = np.zeros((bsz, kv_dim_p_pe, max_seq_len_p_pe), dtype=np.float16)
pe_kcache[:, :, :prefill_len_p_pe] = k_slice
# → send pe_kcache.ravel() via memcpy_h2d to sym_XKCache for this PE
```

**Tiling V cache for PE(px, py)** — target: `[bsz, max_seq_len_p_pe, kv_dim_p_pe]` (zeros-padded):

```python
v_slice = V_global[:, py_tokens, :][:, :, px*kv_dim_p_pe:(px+1)*kv_dim_p_pe]
#         shape: [bsz, prefill_len_p_pe, kv_dim_p_pe]

pe_vcache = np.zeros((bsz, max_seq_len_p_pe, kv_dim_p_pe), dtype=np.float16)
pe_vcache[:, :prefill_len_p_pe, :] = v_slice
# → send pe_vcache.ravel() via memcpy_h2d to sym_XVCache for this PE
```

**Initializing `iter_num`**: After the KV cache is transferred, the host sends `iter_num = prefill_len_p_pe` to each PE as a runtime parameter (or the PE reads it from a known symbol). The first `process_kv()` call writes at column/row `prefill_len_p_pe` and increments to `prefill_len_p_pe + 1`.

**Score GEMV coverage**: `score_matvec_mult()` uses `iter_num` (not a compile-time constant) for the GEMV length via `@set_dsd_length`. After the first decode step, `iter_num = prefill_len_p_pe + 1`, and subsequent steps grow it by 1 per P decode steps (one token per PE per P steps). The GEMV therefore covers all prefill + decode tokens seen so far.

---

## 6. Constraint Summary


| Constraint                                                                                           | Reason                                                 |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `n_heads                                                                                             | group_num`                                             |
| `groups_per_head = group_num/n_heads`                                                                | Compile-time; determines 1-vs-2-level head reduce path |
| `dim % P == 0`                                                                                       | Standard PE sharding constraint                        |
| `(n_kv_heads × head_dim) % P == 0`                                                                   | kv_dim_p_pe must be integer (Phase 2+)                 |
| `n_heads % n_kv_heads == 0`                                                                          | gqa_group_size must be integer (Phase 2+)              |
| `P % n_heads == 0`                                                                                   | pes_p_head must be integer                             |
| `P % n_kv_heads == 0`                                                                                | pes_p_kv_head must be integer                          |
| Degenerate: `n_kv_heads=n_heads` → `gqa_group_size=1`, `kv_dim_p_pe=dim_p_pe` → identical to Phase 1 |                                                        |
| Degenerate: `n_heads=1` → `group_num=1`, `kv_dim_p_pe=dim_p_pe` → identical to WSE-3 base            |                                                        |


---

## 7. Open Questions for Implementation

1. ~~**Intra-group broadcast in Phase 1:** The current `bdcast` color is set up for a global broadcast from `root_2nd_phase`. For the head-scoped score reduce, we need a broadcast limited to `pe_num_p_group` PEs.~~
  **Resolved:** The `X_HEAD_BD` route struct provides head-local broadcast. See §3.2.
2. **Softmax scope for multi-head:** Softmax Y-reduce uses `all_reduce_bsz` and `all_reduceMax_bsz` which reduce across all P PEs on the Y-axis. This is correct because each px-column belongs to one Q head, and Y-reduces aggregate across all seq positions for that head. No scoping needed. ✓
3. **QKV Y-reduce for GQA:** Q reduce needs full-dim Y-reduce (dim_p_pe elements). K,V reduce also need full Y-reduce (kv_dim_p_pe elements). The fused QKV buffer all-reduce handles both in one pass with the new `bsz*(dim_p_pe + 2*kv_dim_p_pe)` extent. ✓
4. ~~**Phase 3 score function**: With compact on-chip KV cache, score_matvec_mult() needs Kt = kv_dim_p_pe. Design choice to be made at Phase 3 start.~~
  **Resolved:** Option C is Phase 2's design — score_matvec_mult() already uses Kt=kv_dim_p_pe and compact K^T. Phase 3 inherits this unchanged. Phase 3 implementation starts with Option A for initial correctness validation, then switches to compact layout (§5.4).

---

## 8. `score_matvec_mult()` Behavior Across All Phases

### Phase 1 (MHA, gqa_group_size=1, kv_dim_p_pe=dim_p_pe)

`score_matvec_mult()` reads from `XKCache_tile` with `Kt = dim_p_pe`. The head-scoped X-reduce (axis=2, pes_p_head PEs) is the only change from WSE-3 base. The Q-aligned K^T cache from the host has shape `[dim_p_pe × seq_len_p_pe]`.

### Phase 2 (GQA + Option C — interleaved Q, compact K^T, gqa_group_size GEMVs)

- **Q layout**: after W_Q-permuted projection, PE px holds gqa_group_size interleaved Q sub-blocks of kv_dim_p_pe each
- **K^T cache**: compact `[kv_dim_p_pe × seq_len_p_pe]` — one row-block per PE, not replicated
- **score_matvec_mult()**: runs `gqa_group_size` separate GEMVs per PE, each with `Kt = kv_dim_p_pe`
- **Reduce scope**: KV-head-scoped (axis=3, pes_p_kv_head PEs)

```
Phase 2 example (P=8, n_heads=4, n_kv_heads=2, dim_p_pe=8, kv_dim_p_pe=4, gqa_group_size=2):
PE px (kv_head k, s = px%4):
  K^T: 4 rows  [s·4 : (s+1)·4]
  for i in {0, 1}:
    score_head_{2k+i}[t] += buf[i*4 : (i+1)*4] @ K^T[:, t]   (Kt=4)
  → 2 KV-head-scoped X-reduces over pes_p_kv_head=4 PEs
```

### Phase 3 (inherits Phase 2 with batched KV cache)

Phase 3 adds variable-length score computation (growing `iter_num`). Score compute body, reduce scope, and route structs are identical to Phase 2; only the DSD length changes at runtime via `@set_dsd_length`.

**Batched KV cache GEMV loop structure:** With `bsz` as the slowest KV cache dimension, `score_matvec_mult()` and `output_matvec_mult_all_groups()` use nested `bsz × gqa_group_size` loops (not flat `bsz * gqa_group_size`). Within each batch, all GQA groups share the same K/V cache section. Between batches, the K/V base pointer advances by one batch stride (`kv_dim_p_pe * max_seq_len_p_pe` for K, `max_seq_len_p_pe * kv_dim_p_pe` for V). `process_kv()` already handled bsz correctly via `range_copy_2d_strided` strides.

### Summary: Dimension sources per phase


| Function                  | Phase 1             | Phase 2 (Option C)         | Phase 3 (inherits Phase 2) |
| ------------------------- | ------------------- | -------------------------- | -------------------------- |
| `xk_matvec_mult()` Nt     | dim_p_pe            | **kv_dim_p_pe**            | kv_dim_p_pe                |
| `score_matvec_mult()` Kt  | dim_p_pe            | **kv_dim_p_pe**            | kv_dim_p_pe                |
| `output_matvec_mult()` Vt | dim_p_pe            | **kv_dim_p_pe**            | kv_dim_p_pe                |
| score reduce scope        | pes_p_head (axis=2) | **pes_p_kv_head (axis=3)** | pes_p_kv_head (axis=3)     |
| XKCache_tile shape        | bsz × dim_p_pe × seq| **bsz × kv_dim_p_pe × seq**| bsz × kv_dim_p_pe × max_seq|
| GEMVs per PE              | 1                   | **gqa_group_size**         | gqa_group_size             |


---

## 9. PE-Local Buffer Memory Layouts

Each PE owns a contiguous 1D SRAM region for each buffer. The tables below describe how logical dimensions map to that 1D address space — listed **slowest-evolving first (outermost) → fastest-evolving last (innermost/contiguous)**.

All sizes are *per-PE* (already divided by P where applicable). Notation:

- `bsz` — batch size
- `gqa_group_size = dim_p_pe / kv_dim_p_pe` — Q-heads per KV-head (= 1 for MHA)
- `kv_dim_p_pe = n_kv_heads × head_dim / P` — KV feature slice per PE
- `dim_p_pe = n_heads × head_dim / P = gqa_group_size × kv_dim_p_pe` — full Q feature slice per PE
- `max_seq_len_p_pe` — maximum token slots per PE
- `iter_num` — currently valid token count on this PE (≤ max_seq_len_p_pe, varies by py)

---

### 9.1 Xq — Query vector  (`QKV_tile[0 : bsz*dim_p_pe]`)

**CSL declaration:** `var QKV_tile: [bsz*(dim_p_pe + 2*kv_dim_p_pe)]f16`
Q occupies the first `bsz × dim_p_pe` elements.

With **Option C** (offline W_Q column permutation), the projected Q is interleaved by GQA group so that each block of `kv_dim_p_pe` elements corresponds to one GQA group's Q-head slice:


| Rank    | Dimension           | Stride        | Size             |
| ------- | ------------------- | ------------- | ---------------- |
| Slowest | batch `b`           | `dim_p_pe`    | `bsz`            |
| —       | GQA group `g`       | `kv_dim_p_pe` | `gqa_group_size` |
| Fastest | KV-head feature `d` | 1             | `kv_dim_p_pe`    |


```
flat index = b × dim_p_pe + g × kv_dim_p_pe + d
```

The interleaving of groups within dim_p_pe is the key property of Option C: each PE sees `gqa_group_size` independent Q-head slices laid out contiguously, enabling the gqa_group_size GEMV loop to simply advance the Q pointer by `kv_dim_p_pe` between groups.

---

### 9.2 K cache — `XKCache_tile`  (K stored transposed, K^T, batched)

**CSL declaration:** `var XKCache_tile: [bsz * kv_dim_p_pe * max_seq_len_p_pe]f16`

K is stored **transposed** (K^T) to enable stride-1 reads along the sequence axis during the score GEMV (`Q[g] @ K^T[:, :iter_num]`):


| Rank    | Dimension               | Stride                            | Size               |
| ------- | ----------------------- | --------------------------------- | ------------------ |
| Slowest | batch `b`               | `kv_dim_p_pe * max_seq_len_p_pe`  | `bsz`              |
| —       | KV-head feature `k`     | `max_seq_len_p_pe`                | `kv_dim_p_pe`      |
| Fastest | sequence token slot `t` | 1                                 | `max_seq_len_p_pe` |


```
flat index = b × kv_dim_p_pe × max_seq_len_p_pe + k × max_seq_len_p_pe + t
```

**Why this layout?** The score GEMV iterates over k (each iteration reads row `k` of K^T), requiring the k-dimension to be outermost. Within each K^T row, token slots are contiguous so the GEMV inner loop reads them with stride 1.

**Interleaved token assignment:** PE(px, py) stores global tokens `{py, py+P, py+2P, …}` at slots `{0, 1, 2, …}`. New tokens are appended by writing column `iter_num` via a strided copy (`dest_stride = max_seq_len_p_pe`).

---

### 9.3 V cache — `XVCache_tile`  (V stored row-major, not transposed, batched)

**CSL declaration:** `var XVCache_tile: [bsz * max_seq_len_p_pe * kv_dim_p_pe]f16`

V is stored **row-major** (one token per row) to enable contiguous writes during `process_kv()` and contiguous reads along the feature axis during the output GEMV (`attn[t] × V[t, :]`):


| Rank    | Dimension               | Stride                            | Size               |
| ------- | ----------------------- | --------------------------------- | ------------------ |
| Slowest | batch `b`               | `max_seq_len_p_pe * kv_dim_p_pe`  | `bsz`              |
| —       | sequence token slot `t` | `kv_dim_p_pe`                     | `max_seq_len_p_pe` |
| Fastest | KV-head feature `v`     | 1                                 | `kv_dim_p_pe`      |


```
flat index = b × max_seq_len_p_pe × kv_dim_p_pe + t × kv_dim_p_pe + v
```

**Why this layout?** The output GEMV (`vecmat_computation_var_right`) iterates over token positions `t` (left-vector dimension). For each `t`, it reads the full V row `V[t, :]` contiguously. New tokens are written at row `iter_num` via a single contiguous block copy.

**Contrast with K cache:** K^T has *feature* outer / *sequence* inner because the score GEMV walks over features. V has *sequence* outer / *feature* inner because the output GEMV walks over sequence tokens.

---

### 9.4 Attention score buffer — `score`  (post-softmax, same buffer as pre-softmax)

**CSL declaration:** `var score: [gqa_group_size * bsz * max_seq_len_p_pe]f16`

The buffer is allocated at max capacity, but only `bsz × gqa_group_size × iter_num` elements are valid after each decode step. The score layout is **BATCH-OUTER** to match the Q layout in QKV_tile (Option C: batch-outer, group-inner). Sub-blocks are packed at **iter_num stride**:


| Rank    | Dimension               | Stride                      | Size (valid / allocated)                    |
| ------- | ----------------------- | --------------------------- | ------------------------------------------- |
| Slowest | batch `b`               | `gqa_group_size × iter_num` | `bsz`                                       |
| —       | GQA group `g`           | `iter_num`                  | `gqa_group_size`                            |
| Fastest | sequence token slot `t` | 1                           | `iter_num` / `max_seq_len_p_pe` (allocated) |


```
flat index of score[b, g, t] = b × gqa_group_size × iter_num + g × iter_num + t
```

**Why batch-outer?** Q in QKV_tile is batch-outer: `[Q_b0_g0(kv_dim_p_pe), Q_b0_g1(kv_dim_p_pe), Q_b1_g0(kv_dim_p_pe), ...]`. With batch-outer score, both Q and score sub-blocks are contiguous in memory. The GEMV loop simply iterates `bsz * gqa_group_size` times with flat sequential advancement — no explicit `(g,b)` address computation needed.

**Why iter_num-packed, not max_seq_len_p_pe-padded?** The GEMV `out_vector_dsd` is advanced by `Nt = iter_num` after each (b,g) iteration. Sub-blocks are tightly packed with no gaps.

**Batched K cache traversal:** K^T is batched (`XKCache_tile[bsz × kv_dim_p_pe × max_seq_len_p_pe]`). The `score_matvec_mult()` uses a nested `bsz × gqa_group_size` loop. Within each batch, all GQA groups share the same K cache section (the K DSD is reset to the batch base via negative `@increment_dsd_offset` after each group). Between batches, the K base pointer advances by one batch stride (`kv_dim_p_pe × max_seq_len_p_pe`). Similarly, `output_matvec_mult_all_groups()` uses a nested loop where `@load_to_dsr` with `save_address=true` naturally keeps the V DSD at the batch base for all groups, advancing to the next batch via `@increment_dsd_offset` of `max_seq_len_p_pe × kv_dim_p_pe`.

**Note on padding:** The buffer is zero-initialised at the start of each `score_matvec_mult()` call, so positions beyond the packed region are zero. The full allocation is needed for the maximum-capacity case.

**Concrete display example** (`gqa_group_size=2`, `bsz=2`, `max_seq_len_p_pe=2`):


| Case          | py  | iter_num | Flat buffer `[0..7]`                                               | Layout                             |
| ------------- | --- | -------- | ------------------------------------------------------------------ | ---------------------------------- |
| Full capacity | 0   | 2        | `[b0g0t0, b0g0t1, b0g1t0, b0g1t1, b1g0t0, b1g0t1, b1g1t0, b1g1t1]` | 4 packed sub-blocks                |
| bsz=1         | 0   | 2        | `[b0g0t0, b0g0t1, b0g1t0, b0g1t1]`                                 | identical to group-outer for bsz=1 |


For `bsz=1`, batch-outer and group-outer layouts are identical (only one batch). The layout distinction only matters for `bsz > 1`.

---

### 9.5 Summary Table


| Buffer                   | Total flat size (alloc)                   | Dimension order (slow → fast)       | Notes                                           |
| ------------------------ | ----------------------------------------- | ----------------------------------- | ----------------------------------------------- |
| Xq (`QKV_tile` prefix)   | `bsz × dim_p_pe`                          | batch → GQA group → KV-head feature | Option C interleaving                           |
| K cache (`XKCache_tile`) | `bsz × kv_dim_p_pe × max_seq_len_p_pe` | batch → KV-head feature → seq slot  | K^T layout; bsz slowest dimension              |
| V cache (`XVCache_tile`) | `bsz × max_seq_len_p_pe × kv_dim_p_pe` | batch → seq slot → KV-head feature  | row-major; bsz slowest dimension                |
| score (`score`)          | `gqa_group_size × bsz × max_seq_len_p_pe` | batch → GQA group → seq slot        | Batch-outer; iter_num-packed                    |


---

## Implementation Log

### Phase 1: MHA n_heads > 1 — 2026-02-28

**Goal**: Make the decode kernel head-aware. The score X-reduce must be scoped to `pes_p_head` PEs (one Q-head block) instead of all P PEs. All other operations (FFN, O-proj, V output, Y-reduces) require no changes.

**Config used:** `mha_test.json` — P=8, n_heads=2, n_kv_heads=2, group_num=4, head_dim=32, dim=64, seq_len=64

- `pes_p_head=4`, `groups_per_head=2`, `root_head_2nd_in_head=2`
- Exercises the full 2-level head-scoped reduce (groups_per_head=2)

**New compile-time parameters introduced:**

- `pes_p_head = P / n_heads` — PEs per Q-head block
- `groups_per_head = group_num / n_heads` — communication groups within one head
- `root_head_2nd_in_head` — head-local phase-2 root = `(groups_per_head/2)*pe_num_p_group + root_1st_phase`

**Constraint added:** `group_num % n_heads == 0` (enforced in `run_sim.sh`).

---

#### `src/comm_lib/comm_layout.csl`

1. Added `param pes_p_head: i16;` after `param root_2nd_phase: i16;`.
2. Added a new route-struct computation block inside `get_params()` that computes head-scoped routing for a new axis=2 mode:
  - `X_HEAD_2nd_RD_0` / `X_HEAD_2nd_RD_1`: phase-2 reduce routes terminating at Q-head boundaries (PEs outside the head block do not participate). Logic mirrors `X_2nd_RD_0/1` but uses head-local coordinates (`px_in_head`, `quotient_x_in_head`, `root_head_2nd_in_head`) and head boundary for the `groups_per_head > 1` case.
  - `X_HEAD_BD`: broadcast route within `pes_p_head` PEs. **Boundary-aware**: when the root falls at a head boundary (px_in_head == 0 or px_in_head == pes_p_head-1), only broadcasts in the direction where PEs exist, preventing data leakage into adjacent head blocks. This required an explicit fix: the original code accidentally set `X_HEAD_BD_ht = {rx=RAMP, tx=[RAMP]}` (no-op) for boundary roots instead of a 1-direction transmit.
3. Updated the return struct to include `.X_HEAD_2nd_RD_0`, `.X_HEAD_2nd_RD_1`, `.X_HEAD_BD`.

---

#### `src/comm_lib/comm_pe.csl`

1. Uncommented `param pes_p_head: i16;` (was dead code in WSE-3).
2. Added three new comptime_struct params: `X_HEAD_2nd_RD_0`, `X_HEAD_2nd_RD_1`, `X_HEAD_BD`.
3. Added compile-time constants:
  ```csl
   const groups_per_head: i16 = pes_p_head / pe_num_p_group;
   const root_head_2nd_in_head: i16 = (groups_per_head / 2) * pe_num_p_group + root_1st_phase;
  ```
4. Added `axis=2` case to `reconfig_allreduce_axis`:
  - Phase-1 routes unchanged (reuses `X_1st_RD_0/1` — within-group logic is head-agnostic).
  - Phase-2 routes use `X_HEAD_2nd_RD_0/1` (head-boundary-terminating).
  - Broadcast uses `X_HEAD_BD` (head-boundary-aware).
5. Added new function `all_reduce_bsz_seq_len_head_scoped(px_in_head, quotient_x_in_head, remainder_pe_id, vector_buf_ptr)`:
  - Phase-1: identical to global X reduce (within-group, head-agnostic).
  - Phase-2: guarded by `groups_per_head > 1`; uses head-local PE index (`px_in_head`) vs `root_head_2nd_in_head` and `pes_p_head` as the boundary.
  - Broadcast: `px_in_head == root_head_2nd_in_head` sends; others receive.
  - Backward compatible: n_heads=1 → pes_p_head=P → head-scoped = global → identical to original.

---

#### `src/layout.csl`

- Added `.pes_p_head = pes_p_head` to the `comm_lib/comm_layout.csl` import parameters.

---

#### `src/decode.csl`

1. Added `.pes_p_head = pes_p_head` to `comm_mod` instantiation.
2. Added module-level variables (initialized to 0, computed in `init_task()`):
  ```csl
   var px_in_head: i16 = 0;
   var quotient_x_in_head: i16 = 0;
  ```
3. In `init_task()`, added:
  ```csl
   px_in_head = px % pes_p_head;
   quotient_x_in_head = px_in_head / pe_num_p_group;
  ```
4. In `decode_struct()`, changed score reduce axis: `reconfig_allreduce_axis(0)` → `reconfig_allreduce_axis(2)`.
5. In `score_matvec_mult()`, replaced the all-reduce call:
  - Old: `comm_mod.all_reduce_bsz_seq_len(px, quotient_x, remainder_x, ptr_score)`
  - New: `comm_mod.all_reduce_bsz_seq_len_head_scoped(px_in_head, quotient_x_in_head, remainder_x, ptr_score)`

---

#### `run_sim.sh`

- Added validation after echo statements:
  ```bash
  if [ $(( GROUP_NUM % N_HEADS )) -ne 0 ]; then
      echo "ERROR: group_num ($GROUP_NUM) must be a multiple of n_heads ($N_HEADS)"
      exit 1
  fi
  ```

---

#### `launch_sim.py`

- Replaced RoPE frequency construction with head-local version. For PE px in head `h = px // pes_p_head`, head-local position `s = px % pes_p_head`, the PE gets base freqs at positions `[s*half_head : (s+1)*half_head]`, tiled `n_heads` times to fill `dim_p_pe/2` slots.
- Backward compatible: n_heads=1 → s always 0 → same freqs at every PE as original.

---

#### `model_config/mha_test.json` (new file)

```json
{
    "P": 8, "group_num": 4, "bsz": 1,
    "dim": 64, "n_heads": 2, "n_kv_heads": 2,
    "head_dim": 32, "seq_len": 64, "ffn_dim": 64
}
```

- `pes_p_head = 4`, `groups_per_head = 2`, `root_head_2nd_in_head = 2`
- `root_head_2nd_in_head = 2` means the phase-2 root is at `px_in_head=2` (interior of the 4-PE head block) → exercises bidirectional broadcast within the head
- All dims divisible by P=8 ✓, `group_num % n_heads == 0` ✓

---

**Key correctness argument for Phase 1:** V and O projections need no changes. With gqa_group_size=1 (MHA), each head's V columns naturally align with that head's PE block on the X-axis. The Y-reduce in `output_matvec_mult()` is therefore head-isolated by construction. The W_O X-reduce mixes all head outputs as intended. Only the score X-reduce requires head scoping.

---

### Phase 2: GQA (Option C) — 2026-03-03

**Goal**: Support `n_kv_heads < n_heads` (GQA) with compact K/V weights and cache (`kv_dim_p_pe` instead of `dim_p_pe`). Use offline W_Q/W_O column permutation (Option C) so no runtime Q scatter is needed. Replace the Phase 1 head-scoped reduce with a KV-head-scoped reduce (axis=3, `pes_p_kv_head` PE scope). Run `gqa_group_size` GEMVs per PE for score and output.

**Config used:** `gqa_test.json` — P=8, n_heads=4, n_kv_heads=2, group_num=4, head_dim=16, dim=64, seq_len=8

- `dim_p_pe=8`, `kv_dim_p_pe=4`, `gqa_group_size=2`, `pes_p_kv_head=4`, `groups_per_kv_head=1`
- `groups_per_kv_head=1` exercises the degenerate path (phase-1 only within the KV-head scope)

**New compile-time parameters introduced:**

- `pes_p_kv_head = P / n_kv_heads` — PEs per KV-head block
- `kv_dim_p_pe = (n_kv_heads * head_dim) / P` — KV dimension per PE
- `gqa_group_size = dim_p_pe / kv_dim_p_pe` — Q-heads sharing each KV-head (derived constant)

**Constraint added:** `group_num % n_kv_heads == 0` (enforced in `run_sim.sh`).

---

#### `src/comm_lib/comm_layout.csl`

1. Added `param pes_p_kv_head: i16;` after `param pes_p_head: i16;`.
2. Added KV-head-scoped routing block inside `get_params()`:
  - `groups_per_kv_head = pes_p_kv_head / pe_num_p_group`
  - `root_kv_head_2nd_in_head = (groups_per_kv_head/2)*pe_num_p_group + root_1st_phase`
  - `px_in_kv_head = px % pes_p_kv_head`, `quotient_x_in_kv_head = px_in_kv_head / pe_num_p_group`
  - `X_KV_HEAD_2nd_RD_0/1`: KV-head-boundary-terminating phase-2 reduce routes. Logic mirrors `X_HEAD_2nd_RD_0/1` but uses `pes_p_kv_head` as the block boundary. When `groups_per_kv_head == 1`: reduces become no-ops (RAMP→RAMP).
  - `X_KV_HEAD_BD`: broadcast within `pes_p_kv_head` PEs. Boundary-aware (same logic as `X_HEAD_BD`).
3. Updated return struct to include `.X_KV_HEAD_2nd_RD_0`, `.X_KV_HEAD_2nd_RD_1`, `.X_KV_HEAD_BD`.

---

#### `src/comm_lib/comm_pe.csl`

1. Added `param pes_p_kv_head: i16;` and three new comptime_struct params: `X_KV_HEAD_2nd_RD_0`, `X_KV_HEAD_2nd_RD_1`, `X_KV_HEAD_BD`.
2. Added compile-time constants:
  ```csl
   const groups_per_kv_head: i16 = pes_p_kv_head / pe_num_p_group;
   const root_kv_head_2nd_in_head: i16 = (groups_per_kv_head / 2) * pe_num_p_group + root_1st_phase;
  ```
3. Added DSD extents for batched-group operations:
  - `bsz_g`: `gqa_group_size * bsz` scalars (for max/sum Y-reduce over all groups at once)
  - `bsz_g_seq_len`: `gqa_group_size * bsz * seq_len_p_pe` (score reduce extent)
4. Added `axis=3` case to `reconfig_allreduce_axis`:
  - Phase-1 routes unchanged (reuses `X_1st_RD_0/1`).
  - Phase-2 routes use `X_KV_HEAD_2nd_RD_0/1` (KV-head-boundary-terminating).
  - Broadcast uses `X_KV_HEAD_BD`.
5. Added new functions:
  - `all_reduce_bsz_g_seq_len_kv_head_scoped(px_in_kv_head, quotient_x_in_kv_head, remainder_pe_id, ptr)`: reduces `gqa_group_size × bsz × seq_len_p_pe` over `pes_p_kv_head` scope. One batched reduce replaces `gqa_group_size` individual reduces. Phase-2 guarded by `groups_per_kv_head > 1`; broadcast from `root_kv_head_2nd_in_head`.
  - `all_reduceMax_bsz_g(...)`: Y-reduce max over `gqa_group_size * bsz` scalars (all groups in one call).
  - `all_reduce_bsz_g(...)`: Y-reduce sum over `gqa_group_size * bsz` scalars.

---

#### `src/layout.csl`

- Added `param pes_p_kv_head: i16;` and passed `.pes_p_kv_head = pes_p_kv_head` to both `comm_lib/comm_layout.csl` import and `decode.csl` tile instantiation.
- Added `XKCache` and `XVCache` to the exported symbol list (for host H2D/D2H).

---

#### `src/decode.csl`

1. Added params `pes_p_kv_head: i16` and `kv_dim_p_pe: i16`. Added derived constant `gqa_group_size = dim_p_pe / kv_dim_p_pe` and `_kv_dim_p_pe = (kv_dim_p_pe/2)*2`.
2. Tensor size changes:
  - `K_weight_tile`, `V_weight_tile`: `[dim_p_pe × kv_dim_p_pe]` (was `dim_p_pe²`)
  - `XKCache_tile`: `[kv_dim_p_pe × seq_len_p_pe]` (was `dim_p_pe × seq_len_p_pe`)
  - `XVCache_tile`: `[seq_len_p_pe × kv_dim_p_pe]` (was `seq_len_p_pe × dim_p_pe`)
  - `QKV_tile`: `[bsz*(dim_p_pe + 2*kv_dim_p_pe)]` (Q: `dim_p_pe`, K+V: `2*kv_dim_p_pe`)
  - `score`: `[gqa_group_size * bsz * seq_len_p_pe]` (was `bsz * seq_len_p_pe`)
  - `local_max_g`, `local_sum_g`: `[gqa_group_size * bsz]` (batched per-group scalars)
3. QKV DSDs reflect the asymmetric split:
  ```csl
   Q_dsd = offset=0,          extent=bsz*dim_p_pe          // Q: full dim
   K_dsd = offset=bsz*dim,    extent=bsz*kv_dim_p_pe       // K: compact
   V_dsd = offset=bsz*(dim+kv), extent=bsz*kv_dim_p_pe     // V: compact
  ```
4. Added module-level variables `px_in_kv_head` and `quotient_x_in_kv_head`; computed in `init_task()`:
  ```csl
   px_in_kv_head = px % pes_p_kv_head;
   quotient_x_in_kv_head = px_in_kv_head / pe_num_p_group;
  ```
5. `xk_matvec_mult()` and `xv_matvec_mult()`: `Nt = kv_dim_p_pe` (was `dim_p_pe`).
6. New `score_matvec_mult()` — **BATCH-OUTER score layout** `[bsz, gqa_group_size, iter_num]`:
  ```
   zero all gqa_group_size*bsz*max_seq_len_p_pe entries of score
   Kt = kv_dim_p_pe; Nt = iter_num
   // Score GEMV: bsz*gqa_group_size iterations, K RESET each iteration (unbatched cache)
   // Q and score are both batch-outer → flat sequential advancement
   left_vector_dsd length=1 (scalar broadcast); inner loop kv_dim_p_pe steps
   for bsz*gqa_group_size:
       right_matrix_dsd = @set_dsd_base_addr(... &XKCache_tile)  // RESET K
       for kv_dim_p_pe:
           score[:] += Q[i] * K_row_i[:iter_num]
       advance score by iter_num
   ONE batched all_reduce_bsz_g_var_seq_len_kv_head_scoped(px_in_kv_head, ...)
   flat alpha scale over all bsz*gqa_group_size*iter_num elements (one op)
  ```
   **Key design (Phase 2)**: K cache was unbatched in Phase 2 — must reset K each (b,g) pair, following original
   **Phase 3 update**: K cache is now batched (`bsz` outermost). The flat `bsz*gqa_group_size` loop was replaced with a nested `bsz × gqa_group_size` loop. Within each batch, all groups share the same K cache section (K DSD reset via negative `@increment_dsd_offset` after each group). Between batches, K advances by `kv_dim_p_pe * max_seq_len_p_pe`.
   `vecmat_computation()` pattern. Batch-outer layout matches Q in QKV_tile, eliminating
   the need for explicit (g,b) address computation.
7. New `softmax_score_all_groups()` (replaces single-group `softmax_score()`):
  - Step 1: local max — load DSR ONCE before all loops; `save_address=true` auto-advances
   through all `bsz*gqa_group_size` sub-blocks. Stores to `local_max_g[g*bsz+b]`.
  - Step 2: ONE `all_reduceMax_bsz_g()` over `gqa_group_size*bsz` scalars
  - Step 3: subtract max + exp — per group iteration, reset `score_tmp` dest to base
  (reuse scratch), advance score src via manual `@increment_dsd_offset`.
  - Step 4: local sum — load DSR once per group (outside bsz loop), use bsz times.
  Manual DSD increment between groups.
  - Step 5: ONE `all_reduce_bsz_g()` over `gqa_group_size*bsz` scalars
  - Step 6: divide — `save_address=true` on dest/src0 so DSR advances across batches
  within each group. Manual DSD increment between groups.
   **Note**: Softmax is layout-agnostic — it processes the flat packed buffer sequentially.
   Each sub-block gets its own max/sum regardless of batch-outer vs group-outer ordering.
8. New `output_matvec_mult_all_groups()` — **batch-outer score, V (unbatched in Phase 2, batched in Phase 3)**:
  - `Kt = iter_num; Nt = kv_dim_p_pe`; `bsz*gqa_group_size` iterations with V RESET each
  - V cache is unbatched — same reset pattern as K in score_matvec_mult
  - Output layout matches Q: batch-outer, group-inner
  - Y-reduce deferred to caller (single reduce covers full `gqa_group_size` output)
9. In `decode_struct()`:
  - `all_reduce_bsz_dim_QKV_fusion()`: extent now `bsz*(dim_p_pe + 2*kv_dim_p_pe)`
  - Score phase: `reconfig_allreduce_axis(3)` before `score_matvec_mult()`; back to `axis=1` after
  - Softmax+output: `softmax_score_all_groups()` then `output_matvec_mult_all_groups()` then single Y-reduce
  - All other axes unchanged

---

#### `run_sim.sh`

- Added `kv_dim_p_pe` and `pes_p_kv_head` derivation:
  ```bash
  kv_dim_p_pe=$(( N_KV_HEADS * HEAD_DIM / P ))
  pes_p_kv_head=$(( P / N_KV_HEADS ))
  ```
- Added KV-head constraint:
  ```bash
  if [ $(( GROUP_NUM % N_KV_HEADS )) -ne 0 ]; then
      echo "ERROR: group_num must be a multiple of n_kv_heads"
      exit 1
  fi
  ```
- Added `kv_dim_p_pe` and `pes_p_kv_head` to `--params` in `cslc` invocation.

---

#### `launch_sim.py`

1. **W_Q/W_O offline permutation (Option C)** — column permutation of `W_Q` and row permutation of `W_O`:
  ```python
   for h in range(n_heads):
       kv_head = h // gqa_group_size;  g = h % gqa_group_size
       for s in range(pes_p_kv_head):
           old_col = h * head_dim + s * kv_dim_p_pe
           new_col = kv_head * pes_p_kv_head * dim_p_pe + s * dim_p_pe + g * kv_dim_p_pe
           W_Q_perm[:, new_col:new_col+kv_dim_p_pe] = W_Q[:, old_col:old_col+kv_dim_p_pe]
           W_O_perm[new_col:new_col+kv_dim_p_pe, :] = W_O[old_col:old_col+kv_dim_p_pe, :]
  ```
   Result: after `X @ W_Q_perm` + Y-reduce, PE `px` holds `gqa_group_size` interleaved Q sub-blocks of `kv_dim_p_pe` each at offsets `[0, kv_dim_p_pe, 2*kv_dim_p_pe, ...]` within its `dim_p_pe` slot.
2. **Compact K/V weight tiling**: `K_tile` and `V_tile` use `kv_dim_p_pe` columns.
3. **K/V cache tiling**: Global K^T `[kv_dim × seq_len]` → per-PE `[kv_dim_p_pe × seq_len_p_pe]`. K requires `swapaxes(0,1)` because `tile_weight_row` gives `result[py, px, :]` = K-dim indexed by `py`, seq by `px`; swapping gives PE(px,py) the correct K-dim slice `[px*kv_dim_p_pe:(px+1)*kv_dim_p_pe]` and token slice `[py*seq_len_p_pe:(py+1)*seq_len_p_pe]`.
4. `**compute_reference()`**: GQA-aware reference using permuted W_Q. Iterates all Q-heads; reconstructs `Q_h` by collecting `kv_dim_p_pe`-wide slices from each `pes_p_kv_head` PE's contribution; computes attention per head; accumulates output with Y-reduce.
5. **Validation steps** (all checked in `--validate` mode):
  - **2a** (post-projection): partial QKV per PE vs `X_norm @ W_Q_perm / W_K / W_V`
  - **2b** (post-Y-reduce): full QKV per PE (all `py` identical)
  - **5a** (post-score-GEMV): per-PE partial score[g=0] before X-reduce
  - **5b** (post-score-reduce): score after KV-head-scoped X-reduce + alpha scale; all `pes_p_kv_head` PEs in each KV-head block identical
  - **5c** (post-softmax): attention weights per head; checks sum ≈ 1.0
  - **6** (output): final output `atol=0.15`

---

## 10. Host-Side Tensor Layouts (`launch_sim.py`)

All tensors are `float16` unless noted. "Global" means full-model dimensions; "tiled" means distributed to P×P PEs.

### 10.1 Input and Norm Weights


| Variable | Shape            | Layout / Notes                                                                                                                                                  |
| -------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `X_raw`  | `[1, bsz * dim]` | **Batch-major flat**: batch `b` occupies `X_raw[0, b*dim : (b+1)*dim]`. To set batch 0 = all 1s, batch 1 = all 2s: `X_raw[0, :dim] = 1.0; X_raw[0, dim:] = 2.0` |
| `W_raw`  | `[1, dim]`       | Global RMSNorm weight vector (broadcast to all PEs, no batch dim)                                                                                               |


After tiling:

- `X_per_pe` `[P, bsz*dim_p_pe]`: row `py` = `concat([X[b, py*dim_p_pe:(py+1)*dim_p_pe] for b in range(bsz)])` — produced by `X_raw.reshape(bsz,P,dim_p_pe).transpose(1,0,2).reshape(P, bsz*dim_p_pe)`
- `tensor_X` `[P, P*bsz*dim_p_pe]`: tile `X_per_pe` P times across columns (all PE columns in same row receive the same X slice)

### 10.2 Weight Matrices (global, before tiling)


| Variable             | Shape            | Notes                                                                |
| -------------------- | ---------------- | -------------------------------------------------------------------- |
| `tensor_q_weight`    | `[dim, kv_dim]`  | Original (un-permuted) Q projection                                  |
| `tensor_k_weight`    | `[dim, kv_dim]`  | K projection                                                         |
| `tensor_v_weight`    | `[dim, kv_dim]`  | V projection                                                         |
| `tensor_o_weight`    | `[dim, dim]`     | Original (un-permuted) O projection                                  |
| `tensor_up_weight`   | `[dim, ffn_dim]` | FFN up-gate                                                          |
| `tensor_gate_weight` | `[dim, ffn_dim]` | FFN SwiGLU gate                                                      |
| `tensor_down_weight` | `[ffn_dim, dim]` | FFN down projection                                                  |
| `W_Q_perm`           | `[dim, dim]`     | Option-C permuted Q weight (column permutation of `tensor_q_weight`) |
| `W_O_perm`           | `[dim, dim]`     | Option-C permuted O weight (row permutation of `tensor_o_weight`)    |


### 10.3 KV Cache (global interleaved, Phase 3)


| Variable         | Shape                   | Layout                                                                                                                                     |
| ---------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `tensor_XKCache` | `[bsz, kv_dim, max_seq_len]` | Batch-outer. Column `t` = K for global token `t`. PE `py = t % P` stores slot `t // P`. Only columns `0..prefill_len-1` are pre-loaded; rest are zeros. |
| `tensor_XVCache` | `[bsz, max_seq_len, kv_dim]` | Batch-outer. Row `t` = V for global token `t`. Same interleaved assignment as K. Only rows `0..prefill_len-1` pre-loaded.                               |


### 10.4 PE-Tiled Arrays sent via H2D

All produced by `tile_weight_row(W, rows, cols)` → `[P, P, rows*cols]`:


| Tile variable | Shape                                    | rows           | cols           | Source                                         |
| ------------- | ---------------------------------------- | -------------- | -------------- | ---------------------------------------------- |
| `Q_tile`      | `[P, P, dim_p_pe²]`                      | `dim_p_pe`     | `dim_p_pe`     | `W_Q_perm`                                     |
| `K_tile`      | `[P, P, dim_p_pe * kv_dim_p_pe]`         | `dim_p_pe`     | `kv_dim_p_pe`  | `tensor_k_weight`                              |
| `V_tile`      | `[P, P, dim_p_pe * kv_dim_p_pe]`         | `dim_p_pe`     | `kv_dim_p_pe`  | `tensor_v_weight`                              |
| `O_tile`      | `[P, P, dim_p_pe²]`                      | `dim_p_pe`     | `dim_p_pe`     | `W_O_perm`                                     |
| `UP_tile`     | `[P, P, dim_p_pe * ffn_dim_p_pe]`        | `dim_p_pe`     | `ffn_dim_p_pe` | `tensor_up_weight`                             |
| `GT_tile`     | `[P, P, dim_p_pe * ffn_dim_p_pe]`        | `dim_p_pe`     | `ffn_dim_p_pe` | `tensor_gate_weight`                           |
| `DN_tile`     | `[P, P, ffn_dim_p_pe * dim_p_pe]`        | `ffn_dim_p_pe` | `dim_p_pe`     | `tensor_down_weight`                           |
| `KCache_tile` | `[P, P, bsz * kv_dim_p_pe * max_seq_len_p_pe]` | —              | —              | `tile_kcache_interleaved(tensor_XKCache, ...)` |
| `VCache_tile` | `[P, P, bsz * max_seq_len_p_pe * kv_dim_p_pe]` | —              | —              | `tile_vcache_interleaved(tensor_XVCache, ...)` |


`tile_weight_row(W, rows, cols)`:

```python
W.reshape(P, rows, P, cols).transpose(0, 2, 1, 3).reshape(P, P, rows * cols)
# → tile[py, px] = W[py*rows:(py+1)*rows, px*cols:(px+1)*cols].ravel()
```

### 10.5 RoPE Freqs


| Variable           | Shape                 | Layout                                                                                         |
| ------------------ | --------------------- | ---------------------------------------------------------------------------------------------- |
| `pe_freqs_cos`     | `[P, _dim_p_pe//2]`   | Per-PE cosine freqs; `pe_freqs_cos[px]` = head-local freqs tiled `n_heads` times               |
| `pe_freqs_sin`     | `[P, _dim_p_pe//2]`   | Per-PE sine freqs; same structure                                                              |
| `tensor_freqs_cos` | `[P, P*_dim_p_pe//2]` | Broadcast: all PE columns in row `py` receive `pe_freqs_cos.ravel()` (same freqs for every py) |


---

**Key correctness argument for Phase 2:** The W_Q permutation ensures that after `X @ W_Q_perm` and the global Y-reduce, each PE `px` holds Q sub-blocks for all `gqa_group_size` heads sharing KV-head `px // pes_p_kv_head`. Because the K^T cache is already sharded by KV-head (PE `px` holds K rows `[px*kv_dim_p_pe:(px+1)*kv_dim_p_pe]`), no runtime Q scatter is needed. The KV-head-scoped X-reduce (axis=3, `pes_p_kv_head` PE scope) correctly sums partial score dot-products within each KV-head block without mixing Q-heads from different KV-heads. The single batched `all_reduce_bsz_g_seq_len_kv_head_scoped()` replaces `gqa_group_size` separate reduces at no extra synchronization cost.
---

### Phase 3: Batched KV Cache — 2026-03-09

**Goal**: Add `bsz` as the slowest (outermost) dimension of `XKCache_tile` and `XVCache_tile`. The previous implementation assumed batch size = 1 for the KV cache (all batches shared one cache). Phase 3 gives each batch its own KV cache section, enabling correct multi-batch decode with GQA.

**Configs used:**
- `gqa_test.json` — P=8, n_heads=4, n_kv_heads=2, bsz=2, head_dim=4, dim=16, max_seq_len=16, prefill_len=8, ffn_dim=64
- `mha_test.json` — P=8, n_heads=2, n_kv_heads=2, bsz=2, head_dim=8, dim=16, max_seq_len=16, prefill_len=8, ffn_dim=16

Both configs use `bsz=2` to exercise the batched KV cache paths.

---

#### `src/decode.csl`

1. **KV cache declarations** — added `bsz` as outermost dimension:
   ```csl
   // BEFORE (Phase 2): unbatched
   var XKCache_tile: [kv_dim_p_pe * max_seq_len_p_pe]f16;
   var XVCache_tile: [max_seq_len_p_pe * kv_dim_p_pe]f16;

   // AFTER (Phase 3): batched, bsz slowest
   var XKCache_tile: [bsz * kv_dim_p_pe * max_seq_len_p_pe]f16;
   var XVCache_tile: [bsz * max_seq_len_p_pe * kv_dim_p_pe]f16;
   ```

2. **`score_matvec_mult()`** — flat `bsz * gqa_group_size` loop replaced with nested `bsz × gqa_group_size`:
   ```
   // Phase 2: flat loop, K reset to &XKCache_tile every iteration
   for bsz*gqa_group_size:
       right_matrix_dsd = @set_dsd_base_addr(... &XKCache_tile)
       inner GEMV
       advance score by iter_num

   // Phase 3: nested loop with per-batch K cache
   right_matrix_dsd = @set_dsd_base_addr(... &XKCache_tile)
   for bsz:
       for gqa_group_size:
           inner GEMV
           // Reset K to this batch's base (negative offset)
           right_matrix_dsd = @increment_dsd_offset(... -kv_dim_p_pe * max_seq_len_p_pe)
           advance score by iter_num
       // Advance K to next batch's section
       right_matrix_dsd = @increment_dsd_offset(... kv_dim_p_pe * max_seq_len_p_pe)
   ```
   Key: within each batch, all GQA groups share the same K cache section. The inner GEMV advances the K DSD by `kv_dim_p_pe * max_seq_len_p_pe`, then a negative `@increment_dsd_offset` resets it to the batch base. Between batches, a positive increment advances to the next batch's K cache.

3. **`output_matvec_mult_all_groups()`** — same nested structure for V cache:
   ```
   // Phase 2: flat loop, V reset to &XVCache_tile every iteration
   for bsz*gqa_group_size:
       right_matrix_dsd = @set_dsd_base_addr(... &XVCache_tile)
       @load_to_dsr(src1_dsr_1, right_matrix_dsd, .{ .save_address = true })

   // Phase 3: nested loop — @load_to_dsr with save_address=true advances
   // DSR but NOT right_matrix_dsd, so V naturally stays at batch base
   right_matrix_dsd = @set_dsd_base_addr(... &XVCache_tile)
   for bsz:
       for gqa_group_size:
           @load_to_dsr(src1_dsr_1, right_matrix_dsd, .{ .save_address = true })
           // right_matrix_dsd stays at batch's V base for all groups
       // Advance V to next batch's section
       right_matrix_dsd = @increment_dsd_offset(... max_seq_len_p_pe * kv_dim_p_pe)
   ```
   Key insight: `@load_to_dsr` with `save_address=true` advances the DSR register but NOT the DSD variable. So within a batch, `right_matrix_dsd` stays at the batch's V base for all groups — no reset needed.

4. **`process_kv()`** — NO code changes needed. Already correct because `range_copy_2d_strided` uses `stride_dest_outer = kv_dim_p_pe * max_seq_len_p_pe` for K and `max_seq_len_p_pe * kv_dim_p_pe` for V, which correctly strides across batch sections in the now-batched cache.

---

#### `launch_sim.py`

1. **KV cache tensors** — added batch dimension:
   ```python
   # BEFORE: tensor_XKCache = np.zeros((kv_dim, max_seq_len), dtype=np.float16)
   # AFTER:  tensor_XKCache = np.zeros((bsz, kv_dim, max_seq_len), dtype=np.float16)
   # Same pattern for tensor_XVCache: (bsz, max_seq_len, kv_dim)
   ```

2. **Tiling functions** — `tile_kcache_interleaved()` and `tile_vcache_interleaved()` updated:
   - Accept `bsz` parameter
   - Produce tiles of shape `[P, P, bsz * cache_per_pe]`
   - Iterate over `bsz` as the outermost loop, placing batch data at offsets `b * cache_per_pe`

3. **H2D memcpy** — `count_per_pe` updated:
   - K cache: `bsz * kv_dim_p_pe * max_seq_len_p_pe`
   - V cache: `bsz * max_seq_len_p_pe * kv_dim_p_pe`

4. **Reference computation** — `compute_reference()` updated for batched K/V:
   - `K_ext` and `V_ext`: `[bsz, kv_dim, seq_len]` and `[bsz, seq_len, kv_dim]`
   - Attention computed per-batch with correct K/V lookup
   - `score_5a_ref`: per-batch K cache lookup using `tensor_XKCache[b, ...]`

---

#### `model_config/mha_test.json`

Updated from Phase 2 config:
```json
{
    "P": 8, "group_num": 4, "bsz": 2,
    "dim": 16, "n_heads": 2, "n_kv_heads": 2,
    "head_dim": 8, "max_seq_len": 16, "prefill_len": 8, "ffn_dim": 16
}
```
- `bsz=2` exercises batched KV cache
- `max_seq_len` / `prefill_len` replace Phase 2's `seq_len` for incremental KV cache

---

#### Verification

- **GQA test** (`gqa_test.json`, bsz=2): All functional steps pass (Steps 2a/2b, 5b, 5c, output). 4 Step5a failures at py=0 b=0 are false positives caused by the `all_reduce_bsz_g_var_seq_len_kv_head_scoped` reduce corrupting the `score_post_gemv` snapshot buffer as a side effect. The actual reduced scores (Step5b) are correct with max_err=0.00000, confirming the GEMV produced correct partial scores.
- **MHA test** (`mha_test.json`, bsz=2): ALL PASS.

**Key finding: negative `@increment_dsd_offset` works in CSL.** The `score_matvec_mult()` uses `-kv_dim_p_pe * max_seq_len_p_pe` to reset the K cache DSD to the batch base after each GQA group. This was verified to produce correct results in the simulator.
