# save_address=true advances DSR register, NOT the DSD variable — 2026-03-03

## Finding

`@load_to_dsr(dsr, dsd, .{.save_address = true})` causes the **DSR** (the hardware
register) to auto-advance its internal base address after each operation — it does NOT
update the **DSD** (the SRAM variable) base address. The DSD is left unchanged.

The correct pattern is: load the DSR **once** outside the loop, then use the **same DSR**
N times inside the loop. Each use advances the DSR's position. Re-calling `@load_to_dsr`
inside the loop defeats the advancement because it resets the DSR to the DSD's (still
unchanged) base address each iteration.

**Wrong pattern** (DSD not advanced across iterations — bug):
```csl
seq_len_p_pe_dsd_1 = @set_dsd_base_addr(seq_len_p_pe_dsd_1, ptr_score);
for (@range(i16, gqa_group_size)) |_| {
    // WRONG: re-loading the DSR inside the loop resets it to dsd's original base
    @load_to_dsr(dest_dsr_1, seq_len_p_pe_dsd_1, .{ .save_address = true });
    @load_to_dsr(src0_dsr_1, seq_len_p_pe_dsd_1, .{ .save_address = true });
    @map(fmulh_softmax_func, local_sum_dsd);
    // seq_len_p_pe_dsd_1 still points to score[0] on next iteration!
}
```

**Correct pattern A** — explicit DSD increment:
```csl
seq_len_p_pe_dsd_1 = @set_dsd_base_addr(seq_len_p_pe_dsd_1, ptr_score);
for (@range(i16, gqa_group_size)) |_| {
    @load_to_dsr(dest_dsr_1, seq_len_p_pe_dsd_1, .{ .save_address = false });
    @load_to_dsr(src0_dsr_1, seq_len_p_pe_dsd_1, .{ .save_address = false });
    @map(fmulh_softmax_func, local_sum_dsd);
    local_sum_dsd = @increment_dsd_offset(local_sum_dsd, bsz, f16);
    // Advance dsd explicitly so next @load_to_dsr picks up next group:
    seq_len_p_pe_dsd_1 = @increment_dsd_offset(seq_len_p_pe_dsd_1, bsz * seq_len_p_pe, f16);
}
```

**Correct pattern B** — load DSR once, use N times (save_address meaningful):
```csl
// Load ONCE outside the loop
@load_to_dsr(chunks_out_dsr, chunks_out_dsd, .{ .save_address = true });
@load_to_dsr(chunks_in_dsr, chunks_in_dsd, .{ .save_address = true });
// Use the SAME DSR N times — each use auto-advances the DSR position
for (@range(i16, N_CHUNKS)) |_| {
    @mov16(chunks_out_dsr, chunks_in_dsr);
}
```

## Symptom

In the GQA softmax (`softmax_score_all_groups`), `seq_len_p_pe_dsd_1` was expected to
auto-advance across the `gqa_group_size` loop, but it did not. Score[0] was divided twice
(g=0 and g=1 both operated on score[0]), while score[1] was never divided. This produced
`attn ≈ 0.0157 = 0.125 / 8` (double-divided) instead of `0.125`.

## Source

Official SDK doc: `CSL-document/CSL-1.4/csl_language_dsrs.md:258-313`

> "This causes subsequent DSD operations on the DSR to update the **DSR's** base address
> for the outermost (slowest-varying) dimension after termination to point one position
> past the end of the range covered by the DSD operation. The next operation on the DSR
> will effectively pick up where the previous one ended."

The doc example explicitly loads DSRs in `comptime {}` (once), then loops using the DSR.
