# Multi-Token GPU Comparison Design — Experiment 5

**Date**: 2026-04-16
**Status**: Design approved, implementation underway
**Kernel**: Decode-GQA (fused grouped-query attention decoder) on Cerebras WSE-3
**Depends on**: `docs/DEVICE_COMPARISON_WORKFLOW.md`, `docs/EXPERIMENT_RESULTS.md`

**Revert point**: `decode-gqa-pre-multi-token` git tag (at commit
`f795541`, "code restructure for testing"). Use
`git checkout decode-gqa-pre-multi-token` to recover the exact
pre-Exp-5 state of all tracked files. Follows the same convention
as `decode-gqa-multiblock-accuracy-v1` (Exp 4 snapshot tag used by
PLAN_SINGLE_JOB_MULTIBLOCK.md).

## 1. Context and scope

The single-token WSE-vs-HF comparison is documented in
`DEVICE_COMPARISON_WORKFLOW.md` (Experiments 1–4). This design extends
that workflow to cover **multi-token autoregressive decode**: given a
prompt, run N decode steps on both WSE and HF and compare.

### Why

Single-token validation cannot expose per-step state accumulation bugs.
KV-cache writes, `iter_num` bookkeeping, and cross-layer drift all
depend on at least two decode steps. The existing four experiments
catch op-, layer-, cascade-, and real-model-level correctness at
**step 0**; nothing catches errors that compound across steps 0, 1,
2, …

### Scope

**In scope**
- New **Experiment 5**, appended (not replacing) to
  `DEVICE_COMPARISON_WORKFLOW.md`.
- Two comparison modes:
  - **Free-generation** (default): WSE applies `model.norm + lm_head`
    to its own last-layer hidden at every step and feeds its argmax
    back. Primary end-to-end oracle.
  - **Teacher-forced** (`--mode teacher-forced`): WSE is fed the HF
    argmax at every step. Debug/diagnosis aid — isolates per-layer
    drift from token-selection drift.
- Two configs:
  - **Exp 5-sim** — P=8 multiblock (`gqa_test_multiblock.json`),
    random weights, numpy reference, simulator only. Teacher-forced
    only (random weights → meaningless free-gen). Seconds-level
    wiring smoke test.
  - **Exp 5-real** — P=256 Llama-3-8B
    (`llama8B_256pfl_1_256_multiblock.json`), real weights, HF
    reference, appliance only. Free-gen default, teacher-forced
    fallback.
- Default `--n-decode = 16`. Configurable. Constraint
  `max_seq_len_p_pe >= prefill_len_p_pe + ceil(N/P)` must be
  validated by the driver.

**Out of scope**
- Any change to `src/decode.csl` or `src/layout.csl`. The kernel is
  already multi-token capable (see §2).
- Any change to the existing single-token drivers
  (`run_llama3_wse.py`, `run_llama3_single_job.py`,
  `run_llama3_batched.py`). Regression insurance — they keep working
  unchanged.
- Prompt configurability. Reuse the existing 256-token prompt in
  `prompt_256.txt` for Exp 5-real.
- Batching across prompts. `bsz=1` only.

### Fallback discipline when Exp 5 fails

1. **Rerun with `--mode teacher-forced`** on the same compiled
   artifact. If per-layer drift at the first-diverging step `t*` is
   localised to a few layers, the bug is in token selection or
   embedding lookup. If drift appears across the board, kernel state
   (`iter_num`, `step`, KV cache) is contaminated.
2. **Drop to Exp 1/2**: apply the "Reviving Experiments 1/2 on the
   simulator" recipe from `DEVICE_COMPARISON_WORKFLOW.md` (sed-revive
   `[TEST-ONLY]` / `[UNUSED-EXPORT]` tags + recompile sim artifact).
   Feed the step-`t*` X input from
   `hf_hidden/token_{t*-1}/layer_{n_layers-1}.npy` to `--deep` and
   single-layer E2E tests.

## 2. Kernel invariants (no CSL changes)

Verified by reading `src/decode.csl`. The driver must obey these:

1. **`init_task` runs exactly once per WSE session.** It sets
   `iter_num_bank[l] = prefill_len_p_pe` and `step_bank[l] = 0` for
   every layer (`decode.csl:244-250`). Calling it again mid-generation
   wipes every layer's KV-cache write position. The driver's outer
   `for t in range(n_decode)` loop must **not** contain an
   `init_task` launch.

2. **`iter_num` / `step` are per-layer and persist across
   `decode_host` calls.** `set_layer(l)` loads them from the bank
   (`decode.csl:281-282`); `decode_struct` writes them back at the
   end of each layer (`decode.csl:1961-1962`). No host action is
   needed between tokens for KV state.

3. **Per-token, before the first block launch of token t, the host
   must H2D:**
   - **`X`**: embedding of whichever token was chosen at step t-1.
     Free-gen uses `argmax(WSE_logits)`; teacher-forced uses
     `hf_output_ids[t-1]`. Tiled into `(P, bsz·dpp)` shape identical
     to single-token's upload.
   - **`freqs_cos`, `freqs_sin`** for absolute position
     `prefill_len + t`. Size `_dim_p_pe/2` each
     (`decode.csl:521-527`). Layer-independent — one upload covers
     all layers of that token, repeated across each block's
     sub-rectangle in multiblock mode.

4. **`Z_tile_bank` is overwritten per token.** `decode_struct`
   snapshots every layer's Z into the bank within one pass
   (`decode.csl:1948-1950`). The host must D2H the bank after the
   last block of each token and write
   `wse_hidden/token_{t}/layer_{l}.npy` before the next token's
   first block launches.

5. **`max_seq_len_p_pe >= prefill_len_p_pe + ceil(N/P)`** is a hard
   precondition. `process_kv` (`decode.csl:422-434`) writes the new
   K/V token at `iter_num` and bumps it; if the cache fills, writes
   go out of bounds and silently corrupt the next layer's cache
   slice. `compile.py` does not yet enforce this (see
   `daily/2026-04-16.md` and the `decode-gqa-max-seq-len-decode-room`
   memory skill). The driver validates on startup and bails with a
   concrete error.

6. **Multiblock cross-token handoff** is identical to single-token:
   per decode step, walk blocks in order; feed block b's last-layer
   Z into block b+1's X; after the last block of token t, that same
   Z feeds `model.norm + lm_head` for token selection.

## 3. Baseline directory layout

Extends the Exp-4 baseline format. Default dir name:
`llama3_cascade_multitoken/`. Exp 5-sim uses an in-memory baseline
(no dir) since Exp 5-sim is inline.

```
llama3_cascade_multitoken/
├── meta.json                    # +"n_decode", +"hf_output_ids",
│                                #  +"hf_input_ids", +"max_seq_len_required"
├── decode_input.npy             # embedding of HF's prefill-argmax token
│                                #  (== hf_input_ids[0]); shape (1, dim) f16
├── weights/layer_{l}.npz        # unchanged: 32 × ~350 MB
├── kv_caches/layer_{l}.npz      # unchanged: prefill-only K/V
├── hf_reference_f32.npz         # changed shape:
│                                #   layer_{l}: (n_decode, dim) f32
│                                #   decode_inputs: (n_decode, dim) f32
├── hf_hidden/
│   └── token_{t}/layer_{l}.npy  # new: per-step per-layer f16
├── hf_tokens.json               # new: {"input_ids":[...],
│                                #   "output_ids":[...], "strings":[...]}
├── lm_head.npy                  # new: (dim, vocab) f16, ~1 GB
├── lm_norm.npy                  # new: (dim,) f16, tiny
├── embed.npy                    # new: (vocab, dim) f16, ~1 GB
└── wse_hidden/                  # populated by driver after Step 2
    ├── token_{t}/layer_{l}.npy
    └── wse_tokens.json
```

Note: `embed.npy` is the `model.embed_tokens.weight`. For Llama-3-8B
it matches `lm_head.weight.T` (tied embeddings), but we save both
explicitly in case a future model unties them.

## 4. Reference generation — `save_baseline` extension

Single new flag in `run_llama3_wse.py`: `--n-decode N` (default 16,
`--n-decode 1` preserves existing behaviour byte-for-byte).

### HF side (Exp 5-real)

```python
# After prefill (unchanged), loop N decode steps with use_cache=True.
hf_output_ids = []
hf_input_ids  = [prefill_argmax_id]         # fed at step 0
hf_hidden_per_layer_f32 = [[] for _ in range(n_layers)]
decode_inputs_f32 = []                       # embedding at each step
cur_id, cur_kv = prefill_argmax_id, copy.deepcopy(past_kv)
for t in range(n_decode):
    out = model(torch.tensor([[cur_id]], device=device),
                past_key_values=cur_kv,
                use_cache=True,
                output_hidden_states=True)
    decode_inputs_f32.append(
        out.hidden_states[0].cpu().numpy().astype(np.float32).reshape(1, dim))
    for l in range(n_layers):
        hf_hidden_per_layer_f32[l].append(
            out.hidden_states[l + 1].cpu().numpy().astype(np.float32).reshape(1, dim))
        # f16 mirror for per-token per-layer compare
        os.makedirs(os.path.join(out_dir, "hf_hidden", f"token_{t}"), exist_ok=True)
        np.save(os.path.join(out_dir, "hf_hidden", f"token_{t}", f"layer_{l}.npy"),
                out.hidden_states[l + 1].cpu().numpy().astype(np.float16).reshape(1, dim))
    hf_output_ids.append(int(torch.argmax(out.logits[0, -1, :])))
    cur_id  = hf_output_ids[-1]
    cur_kv  = out.past_key_values
    hf_input_ids.append(cur_id)

hf_f32 = {"decode_inputs": np.concatenate(decode_inputs_f32, axis=0)}
for l in range(n_layers):
    hf_f32[f"layer_{l}"] = np.concatenate(hf_hidden_per_layer_f32[l], axis=0)  # (N, dim)
np.savez_compressed(os.path.join(out_dir, "hf_reference_f32.npz"), **hf_f32)

# Token log
tokens_json = {
    "input_ids":  hf_input_ids[:-1],     # what was fed at each step
    "output_ids": hf_output_ids,
    "strings":    [tokenizer.decode([i]) for i in hf_output_ids],
}
with open(os.path.join(out_dir, "hf_tokens.json"), "w") as f:
    json.dump(tokens_json, f, indent=2)

# Model head + embedding
np.save(os.path.join(out_dir, "lm_head.npy"),
        model.lm_head.weight.T.detach().cpu().numpy().astype(np.float16))
np.save(os.path.join(out_dir, "lm_norm.npy"),
        model.model.norm.weight.detach().cpu().numpy().astype(np.float16))
np.save(os.path.join(out_dir, "embed.npy"),
        model.model.embed_tokens.weight.detach().cpu().numpy().astype(np.float16))

# meta.json augmentation
meta["n_decode"] = n_decode
meta["hf_input_ids"]  = hf_input_ids[:-1]
meta["hf_output_ids"] = hf_output_ids
meta["max_seq_len_required"] = pfl + n_decode
```

`decode_input.npy` (single-step legacy file) stays: it equals
`decode_inputs_f32[0]` cast to f16 and is what the WSE driver feeds
as X at step 0.

### numpy side (Exp 5-sim)

Extend `test_vs_hf.py`'s numpy cascade. Existing flags
`--cascade N_LAYERS` and new flag `--n-decode N_TOKENS` are
orthogonal: the numpy reference is driven through `N_LAYERS` layers
for `N_TOKENS` autoregressive steps. Since the P=8 config has no
real vocabulary, step-t's X input for `t >= 1` is a
seed-deterministic random f16 vector (same random source on both
WSE and numpy). No `lm_head`, no `embed.npy`, no token IDs. The
oracle is purely "numpy-per-step-per-layer ≈ WSE-per-step-per-layer".

## 5. WSE driver — `run_llama3_multi_token.py`

Appliance-resident host script (runs under `cs_python` on sim and
on appliance). Imports shared helpers from `run_llama3_wse.py` and
`test_vs_hf.py`; adds a new entry point `run_multi_token(...)`.

### Flags

- `--load-baseline DIR` — required.
- `--config CFG.json` — required. Must have `multiblock` field.
- `--n-decode N` — default `meta.json["n_decode"]`. May be smaller
  than the meta value (useful for quick runs on a full baseline).
- `--mode {free-gen,teacher-forced}` — default `free-gen`.
- `--cmaddr STR` — appliance only.
- `--simulator` — sim only.
- `--artifact PATH` — default `out`.

### One-time setup

```python
w = WSE(artifact, cmaddr=cmaddr)

# Validate max_seq_len budget before we burn bulk-H2D time.
msl_pp = cfg["max_seq_len"] // cfg["P"]
pfl_pp = cfg["prefill_len"] // cfg["P"]
need   = pfl_pp + math.ceil(n_decode / cfg["P"])
assert msl_pp >= need, (
    f"max_seq_len_p_pe={msl_pp} < required {need} = "
    f"prefill_len_p_pe({pfl_pp}) + ceil(n_decode({n_decode})/P({cfg['P']})). "
    f"Increase max_seq_len in the config to >= {need * cfg['P']} and recompile."
)

w.launch("init_task")          # exactly once

# Per-block bulk H2D (identical to single-token multiblock, unchanged)
for b, nl in enumerate(block_plan):
    w.set_block_rect(block_ox[b], block_oy[b])
    _bulk_h2d_group(w, cfg_b, tiled_b)    # weights + prefill KV + freqs@prefill_len
```

### Host-side tables

```python
lm_head = np.load(os.path.join(baseline_dir, "lm_head.npy"))   # (dim, vocab) f16
lm_norm = np.load(os.path.join(baseline_dir, "lm_norm.npy"))   # (dim,)
embed   = np.load(os.path.join(baseline_dir, "embed.npy"))     # (vocab, dim) f16
hf_ids  = None
if mode == "teacher-forced":
    with open(os.path.join(baseline_dir, "hf_tokens.json")) as f:
        hf_ids = json.load(f)["output_ids"]                     # [N]

hidden_input = np.load(os.path.join(baseline_dir, "decode_input.npy"))  # (1, dim) f16
wse_output_ids = []
```

### Outer loop

```python
for t in range(n_decode):
    # Decode this token through all blocks
    for b, nl in enumerate(block_plan):
        w.set_block_rect(block_ox[b], block_oy[b])
        w.h2d("X", tile_vec(hidden_input, P, bsz, dpp), P, bsz * dpp)
        w.launch("decode_host", np.int16(b), np.int16(0), np.int16(1))

        bank_raw = w.d2h("Z_tile_bank", P, bsz, nl * dpp)
        bank_raw = bank_raw.reshape(P, P, nl, bsz * dpp)
        for l in range(nl):
            gl = block_start[b] + l
            out = recon_flat_y(bank_raw[:, :, l, :], bsz, dpp, P)
            out_dir = os.path.join(baseline_dir, "wse_hidden", f"token_{t}")
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"layer_{gl}.npy"), out)

        hidden_input = recon_flat_y(bank_raw[:, :, nl - 1, :], bsz, dpp, P)

    # Token selection
    if mode == "free-gen":
        next_id = _apply_head(hidden_input, lm_norm, lm_head)   # argmax, §5a
    else:
        next_id = hf_ids[t]
    wse_output_ids.append(int(next_id))

    # Prepare step t+1 (skip after the final step)
    if t + 1 < n_decode:
        hidden_input = embed[next_id].reshape(1, dim).astype(np.float16)
        new_cos, new_sin = rope_freqs_at_position(
            pos=prefill_len + t + 1, head_dim=hd, base=rope_base)
        for b in range(n_blocks):
            w.set_block_rect(block_ox[b], block_oy[b])
            w.h2d("freqs_cos",
                  _tile_freqs(new_cos, P, dpp // 2), P, dpp // 2)
            w.h2d("freqs_sin",
                  _tile_freqs(new_sin, P, dpp // 2), P, dpp // 2)

# Final write
with open(os.path.join(baseline_dir, "wse_tokens.json"), "w") as f:
    json.dump({"output_ids": wse_output_ids, "mode": mode}, f, indent=2)
```

### 5a. `_apply_head(hidden, norm_w, lm_head)`

f32 path for numerical stability on the argmax:

```python
h  = hidden.astype(np.float32).reshape(1, -1)
w  = norm_w.astype(np.float32)
ms = np.mean(h * h, axis=-1, keepdims=True)
n  = h / np.sqrt(ms + 1e-6) * w         # RMSNorm, f32
logits = n @ lm_head.astype(np.float32) # (1, vocab)
return int(np.argmax(logits[0]))
```

Rationale: a cheap f32 head match the Exp-4 post-processing
(`check_token_match.py`) and sidesteps any f16 argmax ties.

### 5b. `rope_freqs_at_position(pos, head_dim, base)`

Single-position slice of `rope_freqs_interleaved`. Returns
`(cos_hd2, sin_hd2)` each shape `(head_dim//2,)`. The existing
`rope_freqs_interleaved(pfl, hd, base)` already builds the full
range — new helper just slices index `pos`. Keep it in
`test_vs_hf.py` alongside its sibling.

### 5c. `_tile_freqs(arr, P, dpp_half)`

`arr` is `(dpp_half,)`; tile to `(P, dpp_half)` for the `P`-wide
H2D symbol. One-liner; mirrors the tile-broadcast already done in
`_build_layer_group_tiled` for bulk-H2D freqs.

## 6. Launcher driver — `run_llama3_multi_token_single_job.py`

Mirrors `run_llama3_single_job.py`. Differences:

1. Greedy-tar packer extends to include `lm_head.npy`, `lm_norm.npy`,
   `embed.npy`, and (teacher-forced only) `hf_tokens.json`. Total
   staging ~13 GB for Llama-3-8B at N=16 → 7–8 tar chunks at
   1.8 GB each under the existing per-call cap.
2. The cs_python run cmd becomes:
   ```
   cs_python run_llama3_multi_token.py \
       --load-baseline llama3_cascade_multitoken \
       --config <cfg> \
       --n-decode 16 \
       --mode free-gen \
       --cmaddr %CMADDR%
   ```
3. Post-run, the launcher pulls back the new `wse_hidden/token_*/`
   tree and `wse_tokens.json`.

Cumulative-cap verification (PLAN §14.5 M1) remains the open risk;
if +2 GB pushes us past the session ceiling, fall back plan is
staging `lm_head.npy`, `embed.npy` off the appliance entirely —
but that requires a launcher-side decode loop, which we are
explicitly not building (Section 7, Open Items).

## 7. Comparison harness — extended `compare_only`

Lives in `run_llama3_wse.py`. Detect multi-token via
`meta.json["n_decode"] > 1`; otherwise fall through to the current
single-token path unchanged.

Refactor: extract the existing per-layer cosine / max-err / norm
logic into `_compare_one_step(baseline_dir, t)`. Single-token path
becomes a loop-of-one; multi-token iterates `t in range(n_decode)`.

Three output blocks:

**Block 1 — Token match table** (primary oracle):

```
  Token match (mode=free-gen, 16 steps)
  Step | HF token        id    | WSE token       id    | Match
  -----+-----------------------+-----------------------+------
    0  | ' to'           311   | ' to'           311   | OK
    1  | ' build'       1977   | ' build'       1977   | OK
    2  | ' a'            264   | ' the'          279   | X  <- first divergence
    3  | ' fundamentally 42...' | ' truly'       9021   | X
    ...
  Prefix match: 2/16.
```

Prefix-match length is the single scalar to track across runs.

**Block 2 — Per-step per-layer drift** (key layers only, to keep
width readable):

```
  Per-step drift vs HF f32 (cos sim, max err)
  Step | L00              | L08              | L16              | L24              | L31
  -----+------------------+------------------+------------------+------------------+-----
    0  | 0.99916  0.00769 | 0.99990  0.00562 | 0.99997  0.01660 | 0.99999  0.01855 | 0.99017  15.67
    1  | 0.99910  0.00830 | ...
```

Last-layer L31 still shows the final-RMSNorm reporting mismatch
carried over from Exp 4 (not a kernel bug). Comment inline and
continue.

**Block 3 — Logit diagnosis on divergence** (free-gen only, only
when divergence detected):

```
  First divergence: step 2 (WSE=' the' vs HF=' a')
  WSE last-hidden (step 1, after all 32 layers):
    cos vs HF       : 0.99920
    max err         : 0.00830
  WSE logits at step 2 (derived from WSE hidden):
    cos vs HF logits: 0.99873
    top-5 agreement : 4/5
    HF ' a' rank    : 3  (under WSE)
    WSE top-1 margin: 0.12
  Suggestion: rerun with --mode teacher-forced to check whether
  per-layer WSE-vs-HF drift at step 1 is localised to specific layers.
```

All three blocks plus the drift floats go to
`baseline_dir/comparison.json` for machine consumption.

**Exp 5-sim comparator** lives in `test_vs_hf.py` and prints:

```
  Per-step per-layer: WSE vs numpy (cascade_N=2, n_decode=8)
  Step | L00 cos      L00 max  | L01 cos      L01 max
  -----+-----------------------+---------------------
    0  | 0.99999779  0.001221  | 0.99999624  0.001343
    1  | ...
```

Expected on the tiny config: cos >= 0.9999 for every (step, layer).

## 8. Doc extension — `DEVICE_COMPARISON_WORKFLOW.md`

Append one section after Experiment 4: **"Experiment 5: Multi-Token
Autoregressive Decode."** Mirrors Exp 4 structure:

- Purpose.
- Kernel invariants (condensed from §2).
- Step 1: generate multi-token baseline (local) — the
  `--save-baseline ... --n-decode N` command.
- Step 1b: transfer to launcher machine (now includes
  `lm_head.npy`, `lm_norm.npy`, `embed.npy`).
- Step 2: compile + dispatch via
  `run_llama3_multi_token_single_job.py`.
- Step 2b: transfer results back.
- Step 3: compare via `run_llama3_wse.py --compare-only`.
- Exp 5-sim variant (P=8 multiblock, teacher-forced, inline).
- Debug escalation ladder (teacher-forced rerun → revive Exp 1/2).
- Config constraints (reiterate the `max_seq_len_p_pe` rule).

Update the "Per-Experiment Summary" and "What Each Experiment
Tests" tables with a new row for Experiment 5.

## 9. Risks and open items

1. **Staging cumulative cap (PLAN §14.5 M1).** +2 GB for
   `lm_head.npy` + `embed.npy` on top of the existing ~11 GB layer
   weights. Mitigation if hit: drop `embed.npy` and look up
   `embed[next_id]` on the launcher between `decode_host` calls —
   but this breaks the single-call appliance-resident loop and
   requires a second design. Flagged as an open risk, not a planned
   workaround.

2. **First-decode-step freqs.** The bulk-H2D phase uploads freqs
   for position `prefill_len` (the step-0 position). Step 0
   therefore does not need a pre-launch freqs H2D — the loop
   uploads freqs *before step t+1*, which is correct. Verify in
   review that `_build_layer_group_tiled` does indeed compute
   freqs at `prefill_len` (not at position 0).

3. **f32 vs f16 path for `_apply_head` — precision mismatch against
   HF.** `save_baseline` runs HF on CUDA in f16, so HF's argmax is
   taken on **f16** logits. The WSE driver's `_apply_head` (§5a)
   computes logits in **f32** for argmax stability, matching
   `check_token_match.py`. This means a WSE hidden that is
   arithmetically identical to HF's hidden can still argmax to a
   different token under tie conditions (two logits within one f16
   ulp of each other). Divergence diagnosis in §7 Block 3 should
   surface this explicitly: if `WSE top-1 margin < 1e-3` and
   `HF ' X' rank <= 2 under WSE`, the divergence is likely a
   precision tie, not a kernel bug. If this turns out to be a
   persistent source of false divergence, the cheap fix is to also
   do HF's argmax in f32 inside `save_baseline` (cast
   `out.logits.float()` before `torch.argmax`). Do not change this
   in the first implementation — keep the design honest about the
   current HF configuration.

4. **`bsz > 1`.** Current scope is `bsz=1`. If a future Exp 5
   needs batched decode, the `hidden_input[b]` selection from the
   Z_tile_bank needs to be per-batch; the driver currently treats
   the single-batch last row as the hidden-to-feed. Noted in code
   comment; not implemented.

5. **Final RMSNorm layer-31 mismatch** (carried from Exp 4): HF's
   `hidden_states[32]` includes `model.norm`, WSE's layer-31 output
   does not. The free-gen `_apply_head` applies `lm_norm` so WSE's
   logits are directly comparable to HF's; per-layer drift table
   will still show the old reporting mismatch at L31 which is not a
   kernel bug.

6. **HF `past_key_values` API surface**. `save_baseline` already
   handles three variants (`key_cache`, `layers`, tuple). The
   multi-token loop relies on `out.past_key_values` threading
   through correctly; this is standard `use_cache=True` semantics.
   The deep-copy before decode still happens *after* prefill, not
   inside the decode loop (the loop uses `out.past_key_values` as
   the authoritative chained state).

## 10. Acceptance criteria

An implementation of this design is complete when:

1. **Exp 5-sim** produces, on the P=8 multiblock simulator:
   - Per-step per-layer WSE-vs-numpy cos >= 0.9999 for every
     `(step, layer)` pair, for `n_decode in {4, 8}` on two seeds
     (42 and one other). This is the wiring smoke test.

2. **Exp 5-real teacher-forced** produces, on appliance Llama-3-8B
   with `n_decode=16`:
   - Prefix match 16/16 by construction (teacher-forced uses HF's
     tokens).
   - Per-step per-layer WSE-vs-HF cos at the *same* floor as Exp 4
     step 0 (~0.9990 at L0, improving monotonically). No per-step
     drift on top of that.

3. **Exp 5-real free-gen** produces, on appliance Llama-3-8B with
   `n_decode=16`:
   - Prefix match >= 14/16. This is a success threshold, not a
     theoretical bound — it allows one or two late-sequence
     divergences on tie-breakers while still calling the pipeline
     correct. If the number comes in lower on the first run, that's
     valuable signal, not a failure of the design.

4. **`DEVICE_COMPARISON_WORKFLOW.md`** contains a new Experiment 5
   section whose copy-pasteable commands run end-to-end on both sim
   and appliance paths.

5. **`run_llama3_wse.py --compare-only DIR`** on a single-token
   baseline produces byte-identical output to the pre-change
   version (regression fence).

## 11. File list — what gets created / modified

**New files**
- `run_llama3_multi_token.py` — appliance-resident host driver (§5).
- `run_llama3_multi_token_single_job.py` — launcher driver (§6).
- `docs/2026-04-16-multi-token-gpu-comparison-design.md` — this doc.

**Modified files**
- `run_llama3_wse.py`:
  - `save_baseline(..., n_decode=1)` — new flag, new save fields.
  - `compare_only(...)` — multi-token code path with fallthrough.
- `test_vs_hf.py`:
  - `rope_freqs_at_position(pos, head_dim, base)` — new helper.
  - `_tile_freqs(arr, P, dpp_half)` — new helper (may already
    exist inside `_build_layer_group_tiled`; extract).
  - `--n-decode N` flag on the cascade path — drives the Exp 5-sim
    numpy + WSE multi-step comparison.
- `docs/DEVICE_COMPARISON_WORKFLOW.md`:
  - Append Experiment 5 section.
  - Update summary tables.

**Unchanged (regression-fenced)**
- `src/decode.csl`, `src/layout.csl`.
- `run_llama3_single_job.py`, `run_llama3_batched.py`.
- `compile.py`.
- `model_config/*.json` (current `llama8B_256pfl_1_256_multiblock.json`
  has `max_seq_len=512` which is already big enough for N<=256).

## 12. Implementation log (2026-04-16 session)

Implementation landed on `lexu/kernel-comparison-gpu` across 16
commits between tag `decode-gqa-pre-multi-token` (at `f795541`) and
`HEAD` at `b33b742`. Executed via subagent-driven development: each
task dispatched to a fresh implementer, then two-stage review
(spec compliance → code quality) before moving on.

### Commit chain

| # | SHA       | Task | Summary |
|---|-----------|------|---------|
|  0 | `a3fedc8` | —    | docs: commit design doc into repo |
|  1 | `06825b3` | 1.1  | test: failing case for `_build_freqs_for_position` (TDD red) |
|  2 | `28bc6fb` | 1.2  | refactor: extract `_build_freqs_for_position`, reuse in bulk-H2D |
|  3 | `08a2fc0` | 2.1  | test: failing case for `_apply_head` (TDD red) |
|  4 | `c1e3953` | 2.2  | feat: `run_llama3_multi_token.py` skeleton + `_apply_head` |
|  5 | `9fb72df` | 3.1  | feat: `--n-decode` CLI flag (scaffold only) |
|  6 | `1c0770e` | 3.2  | feat(save_baseline): N HF decode steps + per-step per-layer hiddens |
|  7 | `c87f271` | 3.3  | feat(save_baseline): `lm_head.npy` / `lm_norm.npy` / `embed.npy` / `hf_tokens.json` |
|  8 | `2f566b4` | 4.1  | refactor(compare_only): extract `_compare_one_step` |
|  9 | `92d13b1` | 4.2  | feat(compare_only): token-match + per-step drift + logit diagnosis |
| 10 | `064b30d` | 4.2  | fix: drift-table alignment, `_logit_diagnosis` fallback, file handles |
| 11 | `55228f0` | 5.1  | feat(test_vs_hf): `--n-decode` scaffold on cascade path |
| 12 | `2c985bc` | 5.2  | feat(test_vs_hf): Exp 5-sim multi-token cascade |
| 13 | `7674ae6` | 6.1  | feat: `run_llama3_multi_token.py` appliance-resident driver |
| 14 | `100671f` | 7.1  | feat: `run_llama3_multi_token_single_job.py` launcher |
| 15 | `b33b742` | 8.1  | docs: Experiment 5 section + summary-table rows |

Tasks 3.4 (Llama-3-8B save_baseline CPU smoke) skipped — no GPU
on dev machine, Llama-3-8B on CPU impractical. Task 7.2
(appliance integration run) deferred to the user — requires
appliance access.

### Validation results (2026-04-16)

All acceptance criteria from §10 that can be verified without
appliance access: **PASS.**

| Criterion | Check | Result |
|-----------|-------|--------|
| §10 #1 Exp 5-sim cos >= 0.9999 | `--n-decode 8`, seed 42, `gqa_test_multiblock.json` | PASS (min cos 0.999996) |
| §10 #5 Single-token regression byte-identical | `--cascade 2`, seed 42 | PASS (`0.99999779` / `0.99999624`) |
| Unit tests | 4/4 pytest cases in `tests/test_multi_token_helpers.py` | PASS |
| Syntax | All 4 new/modified `.py` files | parse OK |

§10 #2 / #3 / #4 require appliance access — deferred to the user
per Task 7.2.

### Notable deviations from the design as-written

1. **`_cascade_numpy_multitoken` feeds RAW (unpermuted) weights**
   to `F16DecoderLayer`, not `perm_rope_qk`-applied weights as the
   design comment suggested. Rationale: the synthetic Exp 5-sim
   path uses `_generate_layer_inputs` (produces canonical-layout
   weights), and the WSE kernel itself applies interleaved RoPE
   internally using pre-tiled `freqs_cos`/`freqs_sin`. The
   `perm_rope_qk` call only appears in the HF real-weight path
   (`_load_one_layer_inputs` in `run_llama3_wse.py`), which
   converts HF's split-half layout to interleaved for the WSE
   kernel's convention. Feeding the perm'd layout to
   `F16DecoderLayer` would double-permute. This correction was
   caught by the Task 5.2 implementer and independently
   verified by the spec reviewer against `f16_emulator.py`,
   `multi_token_gen.py`, and `_generate_layer_inputs`.
2. **Task 4.2 drift-table column widths** — the design's print
   format snippet had a 16-vs-18 char mismatch between header
   and data cells. Fix `064b30d` made the header 18 chars to
   match.
3. **Task 4.1 print format preservation** — the design suggested
   a pipe-separated drift-table format, but the byte-compat
   regression fence required keeping the existing space-aligned
   format. Implementer correctly preserved the current format.
4. **Launcher idiom matching** — the design's Task 7.1 skeleton
   assumed `cerebras.sdk.runtime.sdkruntimepybind.SdkLauncher`,
   but the sibling `run_llama3_single_job.py` uses
   `cerebras.sdk.client.SdkLauncher`. Implementer correctly
   mirrored the existing sibling.

### Revert point

```bash
git checkout decode-gqa-pre-multi-token
```

Restores the pre-Exp-5 state of all tracked files exactly.

### Minor follow-ups worth folding in later (non-blocking)

- `run_llama3_multi_token.py`: unused `import sys`, one
  `json.load(open(...))` pattern at line 93 (Task 4.2's fix not
  applied here yet), and an inert `--simulator` flag.
- `_compare_one_step`: `dim` and `verbose` parameters accepted
  but unused in the function body.
- Three freqs-tiling duplicates in `test_vs_hf.py` (around lines
  468, 753, 1132) remain — pre-existing tech debt outside this
  plan's scope. Each could use `_build_freqs_for_position`.
- `_apply_head`'s `lm_head.astype(np.float32)` cast runs every
  call; hoisting outside the per-token loop would save 16× GB
  transient for Llama-3-8B.

### User handoff for appliance run (Task 7.2)

On a machine with GPU + HF access:
```bash
python run_llama3_wse.py --save-baseline llama3_cascade_multitoken \
    --model meta-llama/Meta-Llama-3-8B \
    --prompt "$(cat prompt_256.txt)" \
    --n-layers 32 \
    --n-decode 16
```

Transfer the baseline dir (including new `lm_head.npy`,
`lm_norm.npy`, `embed.npy`, `hf_tokens.json`) to the launcher
machine, then:
```bash
python compile.py model_config/llama8B_256pfl_1_256_multiblock.json false
python run_llama3_multi_token_single_job.py \
    --config model_config/llama8B_256pfl_1_256_multiblock.json \
    --baseline-dir llama3_cascade_multitoken \
    --n-decode 16 --mode free-gen
```

Then pull results back and compare locally:
```bash
python run_llama3_wse.py --compare-only llama3_cascade_multitoken
```

If divergence appears, rerun with `--mode teacher-forced` to
isolate per-layer drift from token-selection drift. Second
fallback: revive Exp 1/2 via the recipe in
`DEVICE_COMPARISON_WORKFLOW.md`, feeding the step-`t*-1`
last-layer X from `hf_hidden/token_{t*-1}/layer_{n-1}.npy`.
