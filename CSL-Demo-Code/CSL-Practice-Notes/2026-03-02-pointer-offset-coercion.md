# CSL array-offset pointer coercion — 2026-03-02

## Finding

`&array[offset]` (where offset is non-zero) gives `*f16`, not `[*]f16`. You cannot
assign it to an existing `[*]f16` variable or declare one with explicit type annotation
(`var p: [*]f16 = &array[offset]` also fails). `@ptrCast` fixes it but is unavailable
in older SDK versions. The safe workaround: pass `&array[offset]` directly to
`@set_dsd_base_addr`, which accepts `*f16` as a builtin — then set the DSD rather than
a raw pointer. Passing `&whole_array_var` (not indexed) gives `*[N]f16` which DOES
coerce to `[*]f16` everywhere.

## Example

```csl
// Fails — *f16 cannot be assigned to [*]f16
ptr_score = &score[g * bsz * seq_len_p_pe];

// Works — @set_dsd_base_addr accepts *f16
left_vector_dsd = @set_dsd_base_addr(left_vector_dsd, &score[g * bsz * seq_len_p_pe]);

// Works — whole-array address is *[N]f16, coerces to [*]f16
comm_mod.reduce(py, ..., &local_max_g);
```

## Source

Trial-and-error + Cerebras Slack (cerebras-epcc-external Feb 18 2026). Adam Procter
confirmed @ptrCast is the intended fix; CSL slices not yet implemented.
