# score_tmp scratch-buffer DSD must NOT be incremented in GQA group loop — 2026-03-03

## Finding

In `softmax_score_all_groups()`, `score_tmp` is a **fixed scratch buffer** (size = `bsz * seq_len_p_pe`).
Each GQA group iteration:
1. Resets the **write** destination to `score_tmp[0]` — via `seq_len_p_pe_dsd_1 = @set_dsd_base_addr(..., ptr_score_tmp)` at the top of the loop.
2. Then calls `@map(fast_exp, score_tmp_dsd, score_dsd)` — which **reads** from `score_tmp_dsd`.

**Bug**: `score_tmp_dsd` (the read DSD) was being incremented by `bsz * seq_len_p_pe` each iteration. For `g=1` it read `score_tmp[1]` (out of bounds garbage) instead of `score_tmp[0]`.

**Symptom**: Final attention weights were ~8× too small (0.0157 instead of ~0.125), because the corrupted exp value for group 1 made the global sum ~8× smaller than expected.

**Diagnostic clue**: `local_max_g` and `local_sum_g` looked correct in simprints, but attn didn't sum to 1. The simprint of `score_dsd post exp` was showing garbage because `score_dsd` had been incremented past the end of the array — misleading.

## Fix

Remove the erroneous increment:
```csl
// WRONG (was):
score_tmp_dsd = @increment_dsd_offset(score_tmp_dsd, bsz * seq_len_p_pe, f16);

// CORRECT (removed): score_tmp is a fixed scratch buffer, always rewritten at position 0
// score_tmp_dsd base stays at &score_tmp throughout the loop
```

Also reset `score_dsd` before the post-exp simprint so it shows the actual exp values:
```csl
score_dsd = @set_dsd_base_addr(score_dsd, ptr_score);
simprint.print_string(" score_dsd post exp: ");
simprint_pe_coords_dsd(score_dsd);
```

## Rule

**Scratch buffers that are reset to position 0 each iteration must NOT have their read DSDs incremented.**
Only advance DSDs for buffers that genuinely progress through memory across iterations.
In this softmax pattern:
- `score_dsd` (output): advance — each group writes to a different slice of `score[]`
- `score_tmp_dsd` (scratch input): do NOT advance — always read from position 0

## Source

Trial-and-error debugging in WSE-3-GQA `softmax_score_all_groups()`.
Confirmed by: simprint data showing 0.0157 attn (should be ~0.125), and ratio 0.125/0.0157 ≈ 8 = P = sum factor.
