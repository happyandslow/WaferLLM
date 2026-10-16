# Decode Forward Pass Timing Instrumentation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument the entire `decode_struct()` forward pass with contiguous per-phase timestamps to identify which code segment(s) cause cycle count instability across runs.

**Architecture:** Expand the existing `prof_tsc_N` / `prof_timer_buf` profiling pattern to cover all phases of `decode_struct()`. Each phase boundary gets a `timestamp.get_timestamp()` call. The host reads back the enlarged buffer, computes cycles per phase, and reports statistics (mean/std/min/max) per phase across all PEs. Both WSE-3 and WSE-3-GQA variants get identical treatment.

**Tech Stack:** CSL `<time>` module (`timestamp.get_timestamp`), Python host (`calculate_cycles`), numpy

**Scope:** Only `launch_sim.py` (simulator) scripts are modified. Hardware launch scripts (`launch_wse3.py`) are out of scope — apply the same `NUM_PROF_FNS` / `prof_fn_names` changes there separately if needed.

**Known pre-existing issue:** In `WSE-3/launch_sim.py` line 240, the `decode_host` arguments (repeat, warmup) are in the wrong order vs the CSL signature (warmup, repeat). This works only because `warmup_steps=0`. Not addressed in this plan.

---

## Analysis: Why Overall Timing Is Unstable But Per-Function Is Stable

### Current timing architecture

- **Overall (`tsc_start_buf` → `tsc_end_buf`)**: Spans ALL `repeat_steps` iterations. The host divides by `repeat_steps` to get a per-iteration average. Any variance in ANY iteration accumulates.
- **Per-function (`prof_tsc_N`)**: Overwritten every iteration — only the **last** iteration's timestamps survive to D2H. This is a single-iteration snapshot, so it appears stable.

### Implication

The "instability" in the average isn't necessarily that individual iterations are wildly different — it's that we can't tell. We need to either:
1. Profile the full forward pass in a single iteration (what this plan does), or
2. Accumulate per-phase cycle counts across iterations (more complex, deferred)

This plan takes approach (1): instrument contiguous phases of `decode_struct()` on the **last timed iteration** so we can compare the sum of all phases against the overall total and identify which phase(s) contribute variance.

---

## Phase Boundaries

### WSE-3 (`Decode/WSE-3/src/decode.csl`)

`decode_struct()` has these logical phases:

| Phase | Code                                                                 | Label                    |
|-------|----------------------------------------------------------------------|--------------------------|
| 0     | `rmsnorm_x()`                                                       | `rmsnorm_x`              |
| 1     | `xq/xk/xv_matvec_mult()` + `all_reduce_bsz_dim_QKV_fusion()`      | `qkv_proj_reduce`        |
| 2     | `reconfig(0)` + `xq_rope()` + `xk_rope()`                          | `reconfig0_rope`         |
| 3     | `score_matvec_mult()` (already profiled)                            | `score_gemv`             |
| 4     | `reconfig(1)` + `softmax_score()` (already profiled)               | `reconfig1_softmax`      |
| 5     | `output_matvec_mult()` (already profiled)                           | `output_gemv`            |
| 6     | `reconfig(0)` + `o_matvec_mult()`                                  | `reconfig0_o_proj`       |
| 7     | `attn_residual_add()` + `reconfig(1)` + `rmsnorm_z()`              | `attn_res_rmsnorm_z`     |
| 8     | `up/gate_matvec_mult()` + `all_reduce_bsz_ffn_dim_ZZ_fusion()`     | `ffn_up_gate_reduce`     |
| 9     | `z2_silu()` + `z3_mat()` + `reconfig(0)` + `down_matvec_mult()`    | `silu_z3_down`           |
| 10    | `ffn_residual_add()` + `reconfig(1)`                               | `ffn_res_reconfig`       |

→ 12 timestamp boundaries (prof_tsc_0 through prof_tsc_11), 11 phases, 33 f32 in prof_timer_buf.

### WSE-3-GQA (`Decode/WSE-3-GQA/src/decode.csl`)

Same structure but with GQA-specific functions:

| Phase | Code                                                                 | Label                         |
|-------|----------------------------------------------------------------------|-------------------------------|
| 0     | `rmsnorm_x()`                                                       | `rmsnorm_x`                   |
| 1     | `xq/xk/xv_matvec_mult()` + snapshots + `all_reduce_QKV_fusion()`   | `qkv_proj_reduce`             |
| 2     | `reconfig(3)` + rope (disabled)                                     | `reconfig3_rope`              |
| 3     | `process_kv()`                                                      | `process_kv`                  |
| 4     | `score_matvec_mult_k_outer_v2()`                                    | `score_gemv`                  |
| 5     | `reconfig(1)` + `softmax_score_all_groups_v2()`                     | `reconfig1_softmax`           |
| 6     | `output_matvec_mult_all_groups()`                                   | `output_gemv`                 |
| 7     | `reconfig(0)` + `o_matvec_mult()`                                  | `reconfig0_o_proj`            |
| 8     | `attn_residual_add()` + `reconfig(1)` + `rmsnorm_z()`              | `attn_res_rmsnorm_z`          |
| 9     | `up/gate_matvec_mult()` + `all_reduce_bsz_ffn_dim_ZZ_fusion()`     | `ffn_up_gate_reduce`          |
| 10    | `z2_silu()` + `z3_mat()` + `reconfig(0)` + `down_matvec_mult()`    | `silu_z3_down`                |
| 11    | `ffn_residual_add()` + `reconfig(1)`                               | `ffn_res_reconfig`            |

→ 13 timestamp boundaries (prof_tsc_0 through prof_tsc_12), 12 phases, 36 f32 in prof_timer_buf.

---

## File Structure

### WSE-3

- **Modify:** `Decode/WSE-3/src/decode.csl` — expand prof_tsc vars, expand prof_timer_buf, add timestamps in decode_struct, expand f_pack_prof_timestamps
- **Modify:** `Decode/WSE-3/launch_sim.py` — update NUM_PROF_FNS and prof_fn_names

### WSE-3-GQA

- **Modify:** `Decode/WSE-3-GQA/src/decode.csl` — same changes as above (12 phases instead of 11)
- **Modify:** `Decode/WSE-3-GQA/launch_sim.py` — update NUM_PROF_FNS and prof_fn_names

---

## Task 1: Instrument WSE-3 `decode.csl`

**Files:**
- Modify: `Decode/WSE-3/src/decode.csl:87-95` (timer vars)
- Modify: `Decode/WSE-3/src/decode.csl:775-828` (decode_struct timestamps)
- Modify: `Decode/WSE-3/src/decode.csl:894-921` (f_pack_prof_timestamps)

- [ ] **Step 1: Expand prof_tsc variables and buffer**

Replace the current 4 prof_tsc vars + 9-element buffer with 12 prof_tsc vars + 33-element buffer:

Current (lines 87-95):
```csl
var prof_tsc_0 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_1 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_2 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_3 = @zeros([timestamp.tsc_size_words]u16);

var prof_timer_buf = @zeros([9]f32); // 3 functions × 3 f32 each
var ptr_prof_timer_buf: [*]f32 = &prof_timer_buf;
```

Replace with:
```csl
var prof_tsc_0 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_1 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_2 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_3 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_4 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_5 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_6 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_7 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_8 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_9 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_10 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_11 = @zeros([timestamp.tsc_size_words]u16);

var prof_timer_buf = @zeros([33]f32); // 11 phases × 3 f32 each
var ptr_prof_timer_buf: [*]f32 = &prof_timer_buf;
```

- [ ] **Step 2: Place timestamps in decode_struct**

Replace the current `decode_struct()` (lines 775-828) with full instrumentation:

```csl
fn decode_struct() void {

    timestamp.get_timestamp(&prof_tsc_0);
    rmsnorm_x();

    timestamp.get_timestamp(&prof_tsc_1);
    xq_matvec_mult();
    xk_matvec_mult();
    xv_matvec_mult();
    comm_mod.all_reduce_bsz_dim_QKV_fusion(py, quotient_y, remainder_y, ptr_QKV_tile);

    timestamp.get_timestamp(&prof_tsc_2);
    comm_mod.reconfig_allreduce_axis(0); // 0: px row-wise, 1: py col-wise
    xq_rope();
    xk_rope();

    timestamp.get_timestamp(&prof_tsc_3);
    score_matvec_mult();

    timestamp.get_timestamp(&prof_tsc_4);
    comm_mod.reconfig_allreduce_axis(1);
    softmax_score();

    timestamp.get_timestamp(&prof_tsc_5);
    output_matvec_mult();

    timestamp.get_timestamp(&prof_tsc_6);
    comm_mod.reconfig_allreduce_axis(0);
    o_matvec_mult();

    timestamp.get_timestamp(&prof_tsc_7);
    attn_residual_add();
    comm_mod.reconfig_allreduce_axis(1);
    rmsnorm_z();

    timestamp.get_timestamp(&prof_tsc_8);
    up_matvec_mult();
    gate_matvec_mult();
    comm_mod.all_reduce_bsz_ffn_dim_ZZ_fusion(py, quotient_y, remainder_y, ptr_ZZ_tile);

    timestamp.get_timestamp(&prof_tsc_9);
    z2_silu();
    z3_mat();
    comm_mod.reconfig_allreduce_axis(0);
    down_matvec_mult();

    timestamp.get_timestamp(&prof_tsc_10);
    ffn_residual_add();
    comm_mod.reconfig_allreduce_axis(1);

    timestamp.get_timestamp(&prof_tsc_11);

    decode_entry();
}
```

- [ ] **Step 3: Expand f_pack_prof_timestamps**

Replace the current `f_pack_prof_timestamps()` (lines 894-921) with a version that packs all 11 phases. Each phase packs 3 f32 from (prof_tsc_N, prof_tsc_N+1) pair. The pattern is identical for each — pack the 6 u16 words from the start/end TSC pair into 3 f32 words.

The function must pack prof_tsc_0→prof_tsc_1 into buf[0..2], prof_tsc_1→prof_tsc_2 into buf[3..5], ..., prof_tsc_10→prof_tsc_11 into buf[30..32].

(Full code in implementation — follows exact same bit-packing pattern as existing code, repeated 11 times for phases 0-10.)

- [ ] **Step 4: Verify compilation**

Run: `cd Decode/WSE-3 && python compile.py <test_params> true`
Expected: compiles without errors

---

## Task 2: Update WSE-3 `launch_sim.py`

**Files:**
- Modify: `Decode/WSE-3/launch_sim.py:265-333` (timing readback and reporting)

- [ ] **Step 1: Update NUM_PROF_FNS and prof_fn_names**

Change:
```python
NUM_PROF_FNS = 3  # score_matvec_mult, softmax_score, output_matvec_mult
```
To:
```python
NUM_PROF_FNS = 11  # full forward pass phases
```

Change:
```python
prof_fn_names = ["score_matvec_mult", "softmax_score", "output_matvec_mult"]
```
To:
```python
prof_fn_names = [
    "rmsnorm_x",
    "qkv_proj_reduce",
    "reconfig0_rope",
    "score_gemv",
    "reconfig1_softmax",
    "output_gemv",
    "reconfig0_o_proj",
    "attn_res_rmsnorm_z",
    "ffn_up_gate_reduce",
    "silu_z3_down",
    "ffn_res_reconfig",
]
```

- [ ] **Step 2: Add sum-of-phases vs overall comparison**

After the per-function loop, add:
```python
# Compare sum of phases to overall
phase_sum = np.zeros((P, P))
for fn_idx in range(NUM_PROF_FNS):
    for pe_x in range(P):
        for pe_y in range(P):
            phase_sum[pe_y, pe_x] += calculate_cycles(prof_hwl[pe_y, pe_x, fn_idx*3:(fn_idx+1)*3])
ps = phase_sum.flatten()
print(f"\n--- Sum of all phases (last iteration only) ---")
print(f"  mean   = {ps.mean():.0f}")
print(f"  std    = {ps.std():.0f}")
print(f"\n--- Overall for comparison ({repeat_steps} iters, per-iter) ---")
print(f"  mean   = {c.mean():.0f}")
print(f"  std    = {c.std():.0f}")
print(f"\n  Phase sum / Overall-per-iter = {ps.mean() / c.mean():.4f}")
```

- [ ] **Step 3: Verify by running simulation**

Run: `cd Decode/WSE-3 && python launch_sim.py <test_config>`
Expected: 11 per-phase statistics printed, plus sum-of-phases comparison

---

## Task 3: Instrument WSE-3-GQA `decode.csl`

**Files:**
- Modify: `Decode/WSE-3-GQA/src/decode.csl:122-131` (timer vars)
- Modify: `Decode/WSE-3-GQA/src/decode.csl:1683-1749` (decode_struct timestamps)
- Modify: `Decode/WSE-3-GQA/src/decode.csl:1814-1848` (f_pack_prof_timestamps)

- [ ] **Step 1: Expand prof_tsc variables and buffer**

Replace current 5 prof_tsc vars + 12-element buffer with 13 prof_tsc vars + 36-element buffer:

Current (lines 122-131):
```csl
// Per-function profiling: 4 functions ...
var prof_tsc_0 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_1 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_2 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_3 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_4 = @zeros([timestamp.tsc_size_words]u16);

var prof_timer_buf = @zeros([12]f32); // 4 functions × 3 f32 each
var ptr_prof_timer_buf: [*]f32 = &prof_timer_buf;
```

Replace with:
```csl
// Per-phase profiling: 12 phases of decode_struct → 13 boundary timestamps
var prof_tsc_0 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_1 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_2 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_3 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_4 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_5 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_6 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_7 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_8 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_9 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_10 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_11 = @zeros([timestamp.tsc_size_words]u16);
var prof_tsc_12 = @zeros([timestamp.tsc_size_words]u16);

var prof_timer_buf = @zeros([36]f32); // 12 phases × 3 f32 each
var ptr_prof_timer_buf: [*]f32 = &prof_timer_buf;
```

- [ ] **Step 2: Place timestamps in decode_struct**

Replace the current `decode_struct()` (lines 1683-1749) with full instrumentation:

```csl
fn decode_struct() void {

    timestamp.get_timestamp(&prof_tsc_0);
    rmsnorm_x();

    timestamp.get_timestamp(&prof_tsc_1);
    xq_matvec_mult();
    xk_matvec_mult();
    xv_matvec_mult();
    for (@range(i16, bsz * (dim_p_pe + 2 * kv_dim_p_pe))) |i| { QKV_post_proj[i] = QKV_tile[i]; }
    comm_mod.all_reduce_bsz_dim_QKV_fusion(py, quotient_y, remainder_y, ptr_QKV_tile);
    for (@range(i16, bsz * (dim_p_pe + 2 * kv_dim_p_pe))) |i| { QKV_post_reduce[i] = QKV_tile[i]; }

    timestamp.get_timestamp(&prof_tsc_2);
    comm_mod.reconfig_allreduce_axis(3);
    // xq_rope();  // disabled
    // xk_rope();

    timestamp.get_timestamp(&prof_tsc_3);
    process_kv();

    timestamp.get_timestamp(&prof_tsc_4);
    score_matvec_mult_k_outer_v2();

    timestamp.get_timestamp(&prof_tsc_5);
    comm_mod.reconfig_allreduce_axis(1);
    @fmovh(output_dsd, 0.0);
    softmax_score_all_groups_v2();

    timestamp.get_timestamp(&prof_tsc_6);
    output_matvec_mult_all_groups();

    timestamp.get_timestamp(&prof_tsc_7);
    comm_mod.reconfig_allreduce_axis(0);
    o_matvec_mult();

    timestamp.get_timestamp(&prof_tsc_8);
    attn_residual_add();
    comm_mod.reconfig_allreduce_axis(1);
    rmsnorm_z();

    timestamp.get_timestamp(&prof_tsc_9);
    up_matvec_mult();
    gate_matvec_mult();
    comm_mod.all_reduce_bsz_ffn_dim_ZZ_fusion(py, quotient_y, remainder_y, ptr_ZZ_tile);

    timestamp.get_timestamp(&prof_tsc_10);
    z2_silu();
    z3_mat();
    comm_mod.reconfig_allreduce_axis(0);
    down_matvec_mult();

    timestamp.get_timestamp(&prof_tsc_11);
    ffn_residual_add();
    comm_mod.reconfig_allreduce_axis(1);

    timestamp.get_timestamp(&prof_tsc_12);

    step += 1;

    decode_entry();
}
```

- [ ] **Step 3: Expand f_pack_prof_timestamps**

Replace the current `f_pack_prof_timestamps()` with a version that packs all 12 phases (prof_tsc_0→1 through prof_tsc_11→12 into buf[0..35]).

- [ ] **Step 4: Verify compilation**

Run: `cd Decode/WSE-3-GQA && python compile.py <test_params> true`

---

## Task 4: Update WSE-3-GQA `launch_sim.py`

**Files:**
- Modify: `Decode/WSE-3-GQA/launch_sim.py:617-664` (timing readback and reporting)

- [ ] **Step 1: Update NUM_PROF_FNS and prof_fn_names**

Change:
```python
NUM_PROF_FNS = 4
```
To:
```python
NUM_PROF_FNS = 12  # full forward pass phases
```

Change:
```python
prof_fn_names = ["process_kv", "score_matvec_mult", "softmax_score_all_groups", "output_matvec_mult_all_groups"]
```
To:
```python
prof_fn_names = [
    "rmsnorm_x",
    "qkv_proj_reduce",
    "reconfig3_rope",
    "process_kv",
    "score_gemv",
    "reconfig1_softmax",
    "output_gemv",
    "reconfig0_o_proj",
    "attn_res_rmsnorm_z",
    "ffn_up_gate_reduce",
    "silu_z3_down",
    "ffn_res_reconfig",
]
```

- [ ] **Step 2: Add sum-of-phases vs overall comparison**

Same as Task 2 Step 2.

- [ ] **Step 3: Verify by running simulation**

Run: `cd Decode/WSE-3-GQA && python launch_sim.py <test_config>`
Expected: 12 per-phase statistics printed, plus sum-of-phases comparison

---

## Expected Diagnostic Output

After implementation, running a simulation will show something like:

```
--- Cycle statistics across PxP PEs (per iteration) ---
  mean   = 150000
  std    = 12000    <-- THIS is the instability

--- rmsnorm_x cycle statistics across PxP PEs (last iteration) ---
  mean   = 5000
  std    = 200

--- qkv_proj_reduce cycle statistics across PxP PEs (last iteration) ---
  mean   = 40000
  std    = 500

... (each phase)

--- Sum of all phases (last iteration only) ---
  mean   = 148000
  std    = 800

--- Phase sum / Overall-per-iter = 0.9867
```

**Key diagnostics:**
1. If phase sum std is LOW but overall std is HIGH → instability is between iterations (not within a single forward pass). Next step: accumulate per-iteration totals.
2. If one specific phase has HIGH std → that phase is the source of variance. Focus debugging there.
3. If phase sum is significantly less than overall → the gap represents code not captured by timestamps (e.g., decode_entry overhead, TSC read cost itself).
