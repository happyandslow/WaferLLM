"""
Test: compare score_matvec_mult variants (v0/v1/v2) against numpy reference.

Each variant computes the local (pre-reduce) score GEMV:
  score[b,g,t] = sum_k Q[b,g,k] * K[b,k,t]   for k in 0..kv_dim_p_pe-1

This test fills the K cache with random data up to iter_num slots per PE,
runs all three variants via test_score_gemv(), and compares both cross-variant
agreement and correctness against a numpy reference.

Usage:
  cs_python test_score_gemv.py --config model_config/gqa_test.json [--seed 42] [--iter-num 2]
"""

import json
import argparse
import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder


def tile_kcache_interleaved(K_cache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, fill_slots):
    """Build KCache_tile[P, P, bsz * kv_dim_p_pe * max_seq_len_p_pe] for H2D.

    fill_slots: number of sequence slots per PE to populate (up to max_seq_len_p_pe).
    """
    tile = np.zeros((P, P, bsz * kv_dim_p_pe * max_seq_len_p_pe), dtype=np.float16)
    kv_dim = K_cache.shape[1]
    for py in range(P):
        for px in range(P):
            for b in range(bsz):
                for k in range(kv_dim_p_pe):
                    for s in range(fill_slots):
                        global_k = px * kv_dim_p_pe + k
                        global_t = py + s * P
                        idx = b * kv_dim_p_pe * max_seq_len_p_pe + k * max_seq_len_p_pe + s
                        if global_k < kv_dim and global_t < K_cache.shape[2]:
                            tile[py, px, idx] = K_cache[b, global_k, global_t]
    return tile


def compute_reference_score(QKV_tile_data, KCache_tile, P, bsz, dim_p_pe,
                            kv_dim_p_pe, gqa_group_size, max_seq_len_p_pe, iter_num):
    """Compute numpy reference for the local (pre-reduce) score GEMV.

    Returns ref[P, P, gqa_group_size * bsz * max_seq_len_p_pe] with the
    iter_num-packed layout: score[b,g,t] at offset b*gqa_group_size*iter_num + g*iter_num + t.
    """
    buf_size = gqa_group_size * bsz * max_seq_len_p_pe
    ref = np.zeros((P, P, buf_size), dtype=np.float32)

    for py in range(P):
        for px in range(P):
            q = QKV_tile_data[py, px, :].astype(np.float32)   # [bsz * (dim_p_pe + 2*kv_dim_p_pe)]
            kc = KCache_tile[py, px, :].astype(np.float32)     # [bsz * kv_dim_p_pe * max_seq_len_p_pe]

            for b in range(bsz):
                for g in range(gqa_group_size):
                    # Q[b,g,k] at offset b*dim_p_pe + g*kv_dim_p_pe + k
                    q_offset = b * dim_p_pe + g * kv_dim_p_pe
                    q_vec = q[q_offset:q_offset + kv_dim_p_pe]  # [kv_dim_p_pe]

                    # K[b,k,t] at offset b*kv_dim_p_pe*max_seq_len_p_pe + k*max_seq_len_p_pe + t
                    k_base = b * kv_dim_p_pe * max_seq_len_p_pe
                    # Build K matrix [kv_dim_p_pe, iter_num]
                    k_mat = np.zeros((kv_dim_p_pe, iter_num), dtype=np.float32)
                    for k in range(kv_dim_p_pe):
                        k_mat[k, :] = kc[k_base + k * max_seq_len_p_pe:
                                         k_base + k * max_seq_len_p_pe + iter_num]

                    # score[b,g,:] = Q[b,g,:] @ K[b,:,:]
                    s = q_vec @ k_mat  # [iter_num]

                    # Pack into iter_num-packed layout
                    out_offset = b * gqa_group_size * iter_num + g * iter_num
                    # Store into max_seq_len_p_pe-allocated buffer
                    # The device buffer is [gqa_group_size * bsz * max_seq_len_p_pe],
                    # but iter_num-packed means groups are at stride iter_num, NOT max_seq_len_p_pe.
                    # We need to match whatever the device actually produces.
                    # The device allocates [gqa_group_size * bsz * max_seq_len_p_pe] but only
                    # the first bsz*gqa_group_size*iter_num elements are packed tightly.
                    ref[py, px, out_offset:out_offset + iter_num] = s

    return ref


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
    pes_p_head       = P // n_heads
    pes_p_kv_head    = P // n_kv_heads
    head_dim_p_pe    = head_dim // P
    _dim_p_pe        = (dim_p_pe // 2) * 2

    iter_num = args.iter_num if args.iter_num else prefill_len_p_pe

    print(f"Test config: P={P} bsz={bsz} dim_p_pe={dim_p_pe} kv_dim_p_pe={kv_dim_p_pe}")
    print(f"  gqa_group_size={gqa_group_size} max_seq_len_p_pe={max_seq_len_p_pe}")
    print(f"  iter_num={iter_num} (tokens per PE for score GEMV)")

    np.random.seed(args.seed)

    io_dtype     = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    # ─── Build test data ─────────────────────────────────────────────────────
    # Q lives in QKV_tile[0 .. bsz*dim_p_pe-1] on each PE.
    # We need to load QKV_tile via the existing init path, but for the test
    # we only care about Q and K cache. We'll load Q via the QKV_tile symbol
    # and K via XKCache.

    # Random Q: [P, P, bsz * (dim_p_pe + 2*kv_dim_p_pe)]
    # Only the first bsz*dim_p_pe values are Q; rest are K/V parts (don't matter for score GEMV).
    qkv_size = bsz * (dim_p_pe + 2 * kv_dim_p_pe)
    QKV_tile_data = np.random.rand(P, P, qkv_size).astype(np.float16)

    # Random K cache: [bsz, kv_dim, max_seq_len]
    # Fill up to iter_num*P global columns so the GEMV has non-zero data for all iter_num slots.
    fill_global = min(iter_num * P, max_seq_len)
    tensor_XKCache = np.zeros((bsz, kv_dim, max_seq_len), dtype=np.float16)
    tensor_XKCache[:, :, :fill_global] = np.random.rand(bsz, kv_dim, fill_global).astype(np.float16)

    KCache_tile = tile_kcache_interleaved(tensor_XKCache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, iter_num)

    # We also need to load dummy weights for other buffers that the init_task expects.
    # Minimal approach: just load Q/K/XKCache, other weights can be zeros.
    tensor_X = np.zeros((P, P, bsz * dim_p_pe), dtype=np.float16).ravel()
    tensor_W = np.zeros((P, P, dim_p_pe), dtype=np.float16).ravel()

    def mk_weight(r, c):
        return np.zeros((P, P, r * c), dtype=np.float16).ravel()

    # ─── Runner ──────────────────────────────────────────────────────────────
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

    # Load minimal data
    h2d("X",           tensor_X,                           bsz * dim_p_pe)
    h2d("W",           tensor_W,                           dim_p_pe)
    h2d("Q_weight",    mk_weight(dim_p_pe, dim_p_pe),     dim_p_pe * dim_p_pe)
    h2d("K_weight",    mk_weight(dim_p_pe, kv_dim_p_pe),  dim_p_pe * kv_dim_p_pe)
    h2d("V_weight",    mk_weight(dim_p_pe, kv_dim_p_pe),  dim_p_pe * kv_dim_p_pe)
    h2d("freqs_sin",   np.zeros((P, P, _dim_p_pe // 2), dtype=np.float16).ravel(), _dim_p_pe // 2)
    h2d("freqs_cos",   np.zeros((P, P, _dim_p_pe // 2), dtype=np.float16).ravel(), _dim_p_pe // 2)
    h2d("XKCache",     KCache_tile,                        bsz * kv_dim_p_pe * max_seq_len_p_pe)
    h2d("XVCache",     np.zeros((P, P, bsz * max_seq_len_p_pe * kv_dim_p_pe), dtype=np.float16).ravel(),
                                                           bsz * max_seq_len_p_pe * kv_dim_p_pe)
    h2d("O_weight",    mk_weight(dim_p_pe, dim_p_pe),     dim_p_pe * dim_p_pe)
    h2d("UP_weight",   mk_weight(dim_p_pe, ffn_dim_p_pe), dim_p_pe * ffn_dim_p_pe)
    h2d("GATE_weight", mk_weight(dim_p_pe, ffn_dim_p_pe), dim_p_pe * ffn_dim_p_pe)
    h2d("DOWN_weight", mk_weight(ffn_dim_p_pe, dim_p_pe), ffn_dim_p_pe * dim_p_pe)

    # Load Q data directly into QKV_tile (this is what the score GEMV reads from)
    h2d("QKV_tile", QKV_tile_data, qkv_size)

    # ─── Init + Run test ─────────────────────────────────────────────────────
    runner.launch("init_task", nonblock=False)
    runner.launch("test_score_gemv", np.int16(iter_num), nonblock=False)

    # ─── D2H results ─────────────────────────────────────────────────────────
    score_size = gqa_group_size * bsz * max_seq_len_p_pe
    v0_grid = d2h("score_v0_snapshot", score_size)
    v1_grid = d2h("score_post_gemv", score_size)
    v2_grid = d2h("score_v2_snapshot", score_size)

    runner.stop()

    # ─── Numpy reference ────────────────────────────────────────────────────
    ref_grid = compute_reference_score(
        QKV_tile_data, KCache_tile, P, bsz, dim_p_pe,
        kv_dim_p_pe, gqa_group_size, max_seq_len_p_pe, iter_num
    )

    # ─── Compare ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Comparing v0 (original) vs v1 (k_outer) vs v2 (k_outer_v2) vs numpy ref")
    print(f"Score buffer shape per PE: [{bsz}, {gqa_group_size}, {max_seq_len_p_pe}]")
    print(f"Only first {iter_num} elements per (b,g) group are valid")
    print(f"{'='*60}")

    # The score GEMV produces iter_num-packed output:
    #   score[b,g,t] at flat offset b*gqa_group_size*iter_num + g*iter_num + t
    # But the device buffer is allocated [gqa_group_size * bsz * max_seq_len_p_pe].
    # We extract valid elements using the packed offsets.

    # Build ref into the same max_seq_len_p_pe-padded buffer shape for easy comparison
    ref_padded = np.zeros((P, P, gqa_group_size * bsz * max_seq_len_p_pe), dtype=np.float32)
    for py in range(P):
        for px in range(P):
            for b in range(bsz):
                for g in range(gqa_group_size):
                    packed_off = b * gqa_group_size * iter_num + g * iter_num
                    ref_padded[py, px, packed_off:packed_off + iter_num] = \
                        ref_grid[py, px, packed_off:packed_off + iter_num]

    def compare_pair(name_a, name_b, grid_a, grid_b, use_packed_offsets=True):
        """Compare two grids. Returns (pass, stats_str)."""
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        mismatch_count = 0
        total_checked = 0
        first_mismatches = []

        for py in range(P):
            for px in range(P):
                a = grid_a[py, px, :] if grid_a.dtype == np.float32 \
                    else grid_a[py, px, :].astype(np.float32)
                b_arr = grid_b[py, px, :] if grid_b.dtype == np.float32 \
                    else grid_b[py, px, :].astype(np.float32)

                for bi in range(bsz):
                    for g in range(gqa_group_size):
                        if use_packed_offsets:
                            offset = bi * gqa_group_size * iter_num + g * iter_num
                        else:
                            offset = bi * gqa_group_size * max_seq_len_p_pe + g * max_seq_len_p_pe
                        a_slice = a[offset:offset + iter_num]
                        b_slice = b_arr[offset:offset + iter_num]

                        abs_diff = np.abs(a_slice - b_slice)
                        max_abs_diff = max(max_abs_diff, abs_diff.max())

                        nonzero = np.abs(a_slice) > 1e-8
                        if nonzero.any():
                            rel = abs_diff[nonzero] / np.abs(a_slice[nonzero])
                            max_rel_diff = max(max_rel_diff, rel.max())

                        bad = np.where(abs_diff > 1e-3)[0]
                        mismatch_count += len(bad)
                        total_checked += iter_num

                        if len(bad) > 0 and len(first_mismatches) < 10:
                            for t in bad:
                                first_mismatches.append(
                                    f"    PE({px},{py}) b={bi} g={g} t={t}: "
                                    f"{name_a}={a_slice[t]:.6f} {name_b}={b_slice[t]:.6f} "
                                    f"diff={abs_diff[t]:.6e}")
                                if len(first_mismatches) >= 10:
                                    break

        passed = mismatch_count == 0
        lines = [
            f"\n--- {name_a} vs {name_b} ---",
            f"  Total elements compared: {total_checked}",
            f"  Mismatches (abs > 1e-3): {mismatch_count}",
            f"  Max absolute diff:       {max_abs_diff:.6e}",
            f"  Max relative diff:       {max_rel_diff:.6e}",
            f"  {'PASS' if passed else 'FAIL'}",
        ]
        if not passed:
            lines += first_mismatches
        return passed, "\n".join(lines)

    # Compare each device variant against numpy reference (packed offsets)
    all_pass = True
    for name, grid in [("v0", v0_grid), ("v1", v1_grid), ("v2", v2_grid)]:
        passed, msg = compare_pair(name, "ref", grid, ref_padded, use_packed_offsets=True)
        print(msg)
        if not passed:
            all_pass = False

    # Also cross-compare device variants
    for na, nb, ga, gb in [("v0", "v1", v0_grid, v1_grid),
                           ("v0", "v2", v0_grid, v2_grid)]:
        passed, msg = compare_pair(na, nb, ga, gb, use_packed_offsets=True)
        print(msg)
        if not passed:
            all_pass = False

    # Sample values for sanity check
    n = min(8, iter_num)
    print(f"\nSample values from PE(0,0), batch=0, group=0 (first {n} of {iter_num}):")
    ref_sample = ref_padded[0, 0, :iter_num]
    v0_sample = v0_grid[0, 0, :iter_num].astype(np.float32)
    v1_sample = v1_grid[0, 0, :iter_num].astype(np.float32)
    v2_sample = v2_grid[0, 0, :iter_num].astype(np.float32)
    print(f"  ref (numpy):     {ref_sample[:n]}")
    print(f"  v0 (original):   {v0_sample[:n]}")
    print(f"  v1 (k_outer):    {v1_sample[:n]}")
    print(f"  v2 (k_outer_v2): {v2_sample[:n]}")

    if all_pass:
        print(f"\n  ALL PASS: all variants match numpy reference!")
    else:
        print(f"\n  SOME TESTS FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
