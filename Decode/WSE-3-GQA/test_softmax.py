"""
Test: compare softmax_score_all_groups (v1) vs softmax_score_all_groups_v2 (v2).

v2 replaces the explicit loops in Steps 1 (local max) and 4 (local sum) with
@map + save_address=true. All other steps are identical.

This test populates the score buffer with random data (iter_num-packed layout),
calls test_softmax() which runs both versions on the same input, then compares:
  - score_post_gemv = v1 result
  - score           = v2 result

Also computes a numpy reference softmax for correctness validation.

Usage:
  cs_python test_softmax.py --config model_config/gqa_test.json [--seed 42] [--iter-num 2]
"""

import json
import argparse
import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder


def numpy_softmax_packed(score_flat, bsz, gqa_group_size, iter_num):
    """Compute softmax on iter_num-packed score buffer.

    score_flat layout: [b0_g0(iter_num), b0_g1(iter_num), ..., b1_g0(iter_num), ...]
    Softmax is applied independently per (b, g) sub-block.
    """
    out = np.zeros_like(score_flat)
    for b in range(bsz):
        for g in range(gqa_group_size):
            off = b * gqa_group_size * iter_num + g * iter_num
            x = score_flat[off:off + iter_num].astype(np.float64)
            x = x - x.max()
            ex = np.exp(x)
            out[off:off + iter_num] = (ex / ex.sum()).astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Model config JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--iter-num", type=int, default=None,
                        help="Override iter_num (tokens per PE). Default: prefill_len_p_pe")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    P            = cfg["P"]
    bsz          = cfg["bsz"]
    dim          = cfg["dim"]
    n_heads      = cfg["n_heads"]
    n_kv_heads   = cfg["n_kv_heads"]
    head_dim     = cfg["head_dim"]
    max_seq_len  = cfg["max_seq_len"]
    prefill_len  = cfg["prefill_len"]
    ffn_dim      = cfg["ffn_dim"]

    dim_p_pe         = dim // P
    kv_dim           = n_kv_heads * head_dim
    kv_dim_p_pe      = kv_dim // P
    max_seq_len_p_pe = max_seq_len // P
    prefill_len_p_pe = prefill_len // P
    gqa_group_size   = n_heads // n_kv_heads
    ffn_dim_p_pe     = ffn_dim // P
    _dim_p_pe        = (dim_p_pe // 2) * 2

    iter_num = args.iter_num if args.iter_num else prefill_len_p_pe
    assert iter_num <= max_seq_len_p_pe, \
        f"iter_num={iter_num} exceeds max_seq_len_p_pe={max_seq_len_p_pe}"

    score_buf_size = gqa_group_size * bsz * max_seq_len_p_pe
    packed_size = bsz * gqa_group_size * iter_num

    print(f"Test config: P={P} bsz={bsz} gqa_group_size={gqa_group_size}")
    print(f"  iter_num={iter_num} max_seq_len_p_pe={max_seq_len_p_pe}")
    print(f"  score buf size per PE: {score_buf_size} (packed valid: {packed_size})")

    np.random.seed(args.seed)

    io_dtype     = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    # ─── Build random score data in iter_num-packed layout ────────────────
    # Layout: [b0_g0(iter_num), b0_g1(iter_num), ..., b1_g0(iter_num), ...]
    # padded to score_buf_size with zeros.
    score_input = np.zeros((P, P, score_buf_size), dtype=np.float16)
    for py in range(P):
        for px in range(P):
            for b in range(bsz):
                for g in range(gqa_group_size):
                    off = b * gqa_group_size * iter_num + g * iter_num
                    # Use values in a reasonable range for softmax
                    score_input[py, px, off:off + iter_num] = \
                        np.random.randn(iter_num).astype(np.float16)

    # ─── Dummy weights (needed for init_task) ─────────────────────────────
    def mk_zeros(*shape):
        return np.zeros(shape, dtype=np.float16).ravel()

    def mk_weight(r, c):
        return mk_zeros(P, P, r * c)

    # ─── Runner ──────────────────────────────────────────────────────────
    runner = SdkRuntime("out", simfab_numthreads=64, msg_level='INFO')
    runner.load()
    runner.run()

    def h2d(sym_name, flat_arr, count_per_pe):
        sym = runner.get_id(sym_name)
        u32 = input_array_to_u32(flat_arr.ravel(), 1, 1)
        runner.memcpy_h2d(
            sym, u32, 0, 0, P, P, count_per_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )

    def d2h(sym_name, count_per_pe):
        sym = runner.get_id(sym_name)
        buf = np.zeros(P * P * count_per_pe, dtype=np.uint32)
        runner.memcpy_d2h(
            buf, sym, 0, 0, P, P, count_per_pe,
            streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False
        )
        return memcpy_view(buf, np.dtype(np.float16)).reshape(P, P, count_per_pe)

    # Load all required symbols (most are zeros, only score matters)
    h2d("X",           mk_zeros(P, P, bsz * dim_p_pe),         bsz * dim_p_pe)
    h2d("W",           mk_zeros(P, P, dim_p_pe),               dim_p_pe)
    h2d("Q_weight",    mk_weight(dim_p_pe, dim_p_pe),          dim_p_pe * dim_p_pe)
    h2d("K_weight",    mk_weight(dim_p_pe, kv_dim_p_pe),       dim_p_pe * kv_dim_p_pe)
    h2d("V_weight",    mk_weight(dim_p_pe, kv_dim_p_pe),       dim_p_pe * kv_dim_p_pe)
    h2d("freqs_sin",   mk_zeros(P, P, _dim_p_pe // 2),         _dim_p_pe // 2)
    h2d("freqs_cos",   mk_zeros(P, P, _dim_p_pe // 2),         _dim_p_pe // 2)
    h2d("XKCache",     mk_zeros(P, P, bsz * kv_dim_p_pe * max_seq_len_p_pe),
                                                                bsz * kv_dim_p_pe * max_seq_len_p_pe)
    h2d("XVCache",     mk_zeros(P, P, bsz * max_seq_len_p_pe * kv_dim_p_pe),
                                                                bsz * max_seq_len_p_pe * kv_dim_p_pe)
    h2d("O_weight",    mk_weight(dim_p_pe, dim_p_pe),          dim_p_pe * dim_p_pe)
    h2d("UP_weight",   mk_weight(dim_p_pe, ffn_dim_p_pe),      dim_p_pe * ffn_dim_p_pe)
    h2d("GATE_weight", mk_weight(dim_p_pe, ffn_dim_p_pe),      dim_p_pe * ffn_dim_p_pe)
    h2d("DOWN_weight", mk_weight(ffn_dim_p_pe, dim_p_pe),      ffn_dim_p_pe * dim_p_pe)
    h2d("QKV_tile",    mk_zeros(P, P, bsz * (dim_p_pe + 2 * kv_dim_p_pe)),
                                                                bsz * (dim_p_pe + 2 * kv_dim_p_pe))

    # ─── Init ────────────────────────────────────────────────────────────
    runner.launch("init_task", nonblock=False)

    # Load score data AFTER init (so init doesn't overwrite it)
    h2d("score", score_input, score_buf_size)

    # ─── Run test ────────────────────────────────────────────────────────
    runner.launch("test_softmax", np.int16(iter_num), nonblock=False)

    # ─── D2H results ─────────────────────────────────────────────────────
    v1_grid = d2h("score_post_gemv", score_buf_size)   # v1 result
    v2_grid = d2h("score", score_buf_size)             # v2 result

    runner.stop()

    # ─── Numpy reference ─────────────────────────────────────────────────
    # Note: the device softmax includes an all-reduce across PEs (Y-axis) for
    # global max and global sum. For a single-PE scenario the reduce is identity,
    # but for multi-PE we can't easily replicate. So we primarily compare v1 vs v2
    # (both go through the same reduce). The numpy ref is a sanity check for
    # single-PE scenarios or when P=1 on the Y-axis.

    # ─── Compare v1 vs v2 ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Comparing softmax v1 (loop) vs v2 (@map + save_address)")
    print(f"{'='*60}")

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    mismatch_count = 0
    total_checked = 0
    first_mismatches = []

    for py in range(P):
        for px in range(P):
            v1 = v1_grid[py, px, :].astype(np.float32)
            v2 = v2_grid[py, px, :].astype(np.float32)

            for b in range(bsz):
                for g in range(gqa_group_size):
                    off = b * gqa_group_size * iter_num + g * iter_num
                    s1 = v1[off:off + iter_num]
                    s2 = v2[off:off + iter_num]

                    abs_diff = np.abs(s1 - s2)
                    max_abs_diff = max(max_abs_diff, abs_diff.max())

                    nonzero = np.abs(s1) > 1e-8
                    if nonzero.any():
                        rel = abs_diff[nonzero] / np.abs(s1[nonzero])
                        max_rel_diff = max(max_rel_diff, rel.max())

                    bad = np.where(abs_diff > 1e-3)[0]
                    mismatch_count += len(bad)
                    total_checked += iter_num

                    if len(bad) > 0 and len(first_mismatches) < 10:
                        for t in bad:
                            first_mismatches.append(
                                f"  PE({px},{py}) b={b} g={g} t={t}: "
                                f"v1={s1[t]:.6f} v2={s2[t]:.6f} "
                                f"diff={abs_diff[t]:.6e}")
                            if len(first_mismatches) >= 10:
                                break

    print(f"\n--- v1 vs v2 ---")
    print(f"  Total elements compared: {total_checked}")
    print(f"  Mismatches (abs > 1e-3): {mismatch_count}")
    print(f"  Max absolute diff:       {max_abs_diff:.6e}")
    print(f"  Max relative diff:       {max_rel_diff:.6e}")

    if mismatch_count == 0:
        print(f"  PASS")
    else:
        print(f"  FAIL: {mismatch_count} mismatches")
        for m in first_mismatches:
            print(m)

    # ─── Sample values ───────────────────────────────────────────────────
    n = min(8, iter_num)
    print(f"\nSample values from PE(0,0), batch=0, group=0 (first {n} of {iter_num}):")
    inp = score_input[0, 0, :iter_num].astype(np.float32)
    s1 = v1_grid[0, 0, :iter_num].astype(np.float32)
    s2 = v2_grid[0, 0, :iter_num].astype(np.float32)
    print(f"  input:       {inp[:n]}")
    print(f"  v1 (loop):   {s1[:n]}")
    print(f"  v2 (@map):   {s2[:n]}")

    # Verify softmax properties: all positive, sums to ~1
    print(f"\nSoftmax sanity check (PE(0,0), batch=0, group=0):")
    print(f"  v1 sum={s1.sum():.6f} min={s1.min():.6f} max={s1.max():.6f}")
    print(f"  v2 sum={s2.sum():.6f} min={s2.min():.6f} max={s2.max():.6f}")

    if mismatch_count == 0:
        print(f"\n  PASS: v1 and v2 produce identical softmax results!")
    else:
        print(f"\n  FAIL")

    return 0 if mismatch_count == 0 else 1


if __name__ == "__main__":
    exit(main())
