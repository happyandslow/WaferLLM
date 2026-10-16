# Score GEMV batch/group dimension order bug — 2026-03-04

## Finding
When computing multi-group scores (`Q[g,b] @ K^T`) in GQA decode with `bsz > 1`, an outer loop
over `gqa_group_size` calling `vecmat_computation_var_right_2d_strided` (which internally loops
over `bsz` batches) traverses `QKV_tile` in **(group-outer, batch-inner)** order. But the
QKV_tile Q layout is **(batch-outer, group-inner)**: `[Q_b0(dim_p_pe), Q_b1(dim_p_pe), ...]`
where each `Q_bi = [Q_g0(kv_dim_p_pe), Q_g1(kv_dim_p_pe), ...]`.

For `bsz > 1` and `gqa_group_size > 1`, this causes the device to read `Q[b=0, g=1]` where it
should read `Q[b=1, g=0]`, swapping the wrong (g, b) pairs. For `bsz=1` the layouts are
identical so the bug is invisible.

## Symptom
Step 5a validation (score post-GEMV, before reduce) fails for `bsz > 1` with `gqa_group_size > 1`.
Simprint of `score_dsd` and d2h of `score_post_gemv` agree with each other (both show the device's
wrong values) but do not match the reference. Negative score values are NOT a bug — `Q @ K^T` dot
products are naturally negative.

## Fix
Replace the simple `gqa_group_size` outer loop with an explicit `(g, b)` double loop that
computes Q and score addresses per pair using array indexing with runtime offsets:

```csl
for (@range(i16, gqa_group_size)) |g| {
    for (@range(i16, bsz)) |b| {
        // Q: QKV_tile[b*dim_p_pe + g*kv_dim_p_pe]
        left_vector_dsd = @set_dsd_base_addr(left_vector_dsd,
            &QKV_tile[@as(u16, b) * @as(u16, dim_p_pe) + @as(u16, g) * @as(u16, kv_dim_p_pe)]);
        // Score: score[(g*bsz + b) * iter_num]
        out_vector_dsd = @set_dsd_base_addr(out_vector_dsd,
            &score[(@as(u16, g) * @as(u16, bsz) + @as(u16, b)) * @as(u16, iter_num)]);
        right_matrix_dsd = @set_dsd_base_addr(right_matrix_dsd, ptr_right_matrix);
        // ... Kt iterations of @map(gemv_static_step, left_vector_dsd) with per-step DSD advance
    }
}
```

## Source
Trial-and-error analysis in `Decode/WSE-3-GQA/src/decode.csl` `score_matvec_mult()`.
