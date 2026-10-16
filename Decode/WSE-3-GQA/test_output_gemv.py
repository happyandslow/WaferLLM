"""
Test: compare output_matvec_mult_all_groups variants (v0/v2/v3/v4) against numpy reference.

Each variant computes the output GEMV (post-softmax attention × V cache):
  output[b,g,k] = sum_t score[b,g,t] * V[b,t,k]   for t in 0..iter_num-1

Score layout (iter_num-packed): [b0_g0(iter_num), b0_g1(iter_num), ...]
V cache layout: XVCache_tile[bsz, max_seq_len_p_pe, kv_dim_p_pe]
Output layout: [out_b0_g0(kv_dim_p_pe), out_b0_g1(kv_dim_p_pe), ...]

Usage:
  cs_python test_output_gemv.py --config model_config/gqa_test.json [--seed 42] [--iter-num 2]
"""

import json
import argparse
import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder


def tile_vcache_interleaved(V_cache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, fill_slots):
    """Build VCache_tile[P, P, bsz * max_seq_len_p_pe * kv_dim_p_pe] for H2D.

    V cache on-device layout: XVCache_tile[bsz, max_seq_len_p_pe, kv_dim_p_pe]
    V_cache host shape: [bsz, max_seq_len, kv_dim]
    Interleaved across PEs: global_t = py + s * P, global_k = px * kv_dim_p_pe + k
    """
    kv_dim = V_cache.shape[2]
    tile = np.zeros((P, P, bsz * max_seq_len_p_pe * kv_dim_p_pe), dtype=np.float16)
    for py in range(P):
        for px in range(P):
            for b in range(bsz):
                for s in range(fill_slots):
                    for k in range(kv_dim_p_pe):
                        global_t = py + s * P
                        global_k = px * kv_dim_p_pe + k
                        idx = b * max_seq_len_p_pe * kv_dim_p_pe + s * kv_dim_p_pe + k
                        if global_t < V_cache.shape[1] and global_k < kv_dim:
                            tile[py, px, idx] = V_cache[b, global_t, global_k]
    return tile


def compute_reference_output(score_tile, VCache_tile, P, bsz, kv_dim_p_pe,
                             gqa_group_size, max_seq_len_p_pe, iter_num):
    """Compute numpy reference for the local (pre-reduce) output GEMV.

    output[b,g,k] = sum_t score[b,g,t] * V[b,t,k]

    Score layout (iter_num-packed): flat offset b*G*iter_num + g*iter_num + t
    V layout: flat offset b*max_seq_len_p_pe*kv_dim_p_pe + t*kv_dim_p_pe + k
    Output layout: flat offset b*G*kv_dim_p_pe + g*kv_dim_p_pe + k

    Returns ref[P, P, bsz * gqa_group_size * kv_dim_p_pe]
    """
    out_size = bsz * gqa_group_size * kv_dim_p_pe
    ref = np.zeros((P, P, out_size), dtype=np.float32)

    for py in range(P):
        for px in range(P):
            sc = score_tile[py, px, :].astype(np.float32)
            vc = VCache_tile[py, px, :].astype(np.float32)

            for b in range(bsz):
                # Build V matrix [iter_num, kv_dim_p_pe]
                v_base = b * max_seq_len_p_pe * kv_dim_p_pe
                V_mat = np.zeros((iter_num, kv_dim_p_pe), dtype=np.float32)
                for t in range(iter_num):
                    V_mat[t, :] = vc[v_base + t * kv_dim_p_pe:
                                     v_base + t * kv_dim_p_pe + kv_dim_p_pe]

                for g in range(gqa_group_size):
                    # Score vector [iter_num]
                    s_off = b * gqa_group_size * iter_num + g * iter_num
                    s_vec = sc[s_off:s_off + iter_num]

                    # output[b,g,:] = score[b,g,:] @ V[b,:,:]
                    out = s_vec @ V_mat  # [kv_dim_p_pe]

                    o_off = b * gqa_group_size * kv_dim_p_pe + g * kv_dim_p_pe
                    ref[py, px, o_off:o_off + kv_dim_p_pe] = out

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
    _dim_p_pe        = (dim_p_pe // 2) * 2

    iter_num = args.iter_num if args.iter_num else prefill_len_p_pe
    assert iter_num <= max_seq_len_p_pe, \
        f"iter_num={iter_num} exceeds max_seq_len_p_pe={max_seq_len_p_pe}"

    output_size = bsz * gqa_group_size * kv_dim_p_pe
    score_buf_size = gqa_group_size * bsz * max_seq_len_p_pe

    print(f"Test config: P={P} bsz={bsz} gqa_group_size={gqa_group_size}")
    print(f"  kv_dim_p_pe={kv_dim_p_pe} iter_num={iter_num} max_seq_len_p_pe={max_seq_len_p_pe}")
    print(f"  output size per PE: {output_size}")

    np.random.seed(args.seed)

    io_dtype     = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    # ─── Build test data ─────────────────────────────────────────────────
    # Random score in iter_num-packed layout (simulates post-softmax attention weights)
    score_input = np.zeros((P, P, score_buf_size), dtype=np.float16)
    for py in range(P):
        for px in range(P):
            for b in range(bsz):
                for g in range(gqa_group_size):
                    off = b * gqa_group_size * iter_num + g * iter_num
                    # Use positive values (softmax output is always positive)
                    vals = np.abs(np.random.randn(iter_num)).astype(np.float16)
                    vals = vals / vals.sum()  # Normalize like softmax
                    score_input[py, px, off:off + iter_num] = vals

    # Random V cache: [bsz, max_seq_len, kv_dim]
    fill_global = min(iter_num * P, max_seq_len)
    tensor_XVCache = np.zeros((bsz, max_seq_len, kv_dim), dtype=np.float16)
    tensor_XVCache[:, :fill_global, :] = np.random.randn(bsz, fill_global, kv_dim).astype(np.float16)

    VCache_tile = tile_vcache_interleaved(
        tensor_XVCache, P, bsz, kv_dim_p_pe, max_seq_len_p_pe, iter_num)

    # ─── Dummy weights ───────────────────────────────────────────────────
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

    # Load all required symbols
    h2d("X",           mk_zeros(P, P, bsz * dim_p_pe),         bsz * dim_p_pe)
    h2d("W",           mk_zeros(P, P, dim_p_pe),               dim_p_pe)
    h2d("Q_weight",    mk_weight(dim_p_pe, dim_p_pe),          dim_p_pe * dim_p_pe)
    h2d("K_weight",    mk_weight(dim_p_pe, kv_dim_p_pe),       dim_p_pe * kv_dim_p_pe)
    h2d("V_weight",    mk_weight(dim_p_pe, kv_dim_p_pe),       dim_p_pe * kv_dim_p_pe)
    h2d("freqs_sin",   mk_zeros(P, P, _dim_p_pe // 2),         _dim_p_pe // 2)
    h2d("freqs_cos",   mk_zeros(P, P, _dim_p_pe // 2),         _dim_p_pe // 2)
    h2d("XKCache",     mk_zeros(P, P, bsz * kv_dim_p_pe * max_seq_len_p_pe),
                                                                bsz * kv_dim_p_pe * max_seq_len_p_pe)
    h2d("XVCache",     VCache_tile,                             bsz * max_seq_len_p_pe * kv_dim_p_pe)
    h2d("O_weight",    mk_weight(dim_p_pe, dim_p_pe),          dim_p_pe * dim_p_pe)
    h2d("UP_weight",   mk_weight(dim_p_pe, ffn_dim_p_pe),      dim_p_pe * ffn_dim_p_pe)
    h2d("GATE_weight", mk_weight(dim_p_pe, ffn_dim_p_pe),      dim_p_pe * ffn_dim_p_pe)
    h2d("DOWN_weight", mk_weight(ffn_dim_p_pe, dim_p_pe),      ffn_dim_p_pe * dim_p_pe)
    h2d("QKV_tile",    mk_zeros(P, P, bsz * (dim_p_pe + 2 * kv_dim_p_pe)),
                                                                bsz * (dim_p_pe + 2 * kv_dim_p_pe))

    # ─── Init ────────────────────────────────────────────────────────────
    runner.launch("init_task", nonblock=False)

    # Load score data AFTER init
    h2d("score", score_input, score_buf_size)

    # ─── Verify inputs on device ─────────────────────────────────────────
    # D2H score and V cache to confirm they match what we loaded
    score_device = d2h("score", score_buf_size)
    vcache_device = d2h("XVCache", bsz * max_seq_len_p_pe * kv_dim_p_pe)

    print(f"\n--- Input verification for PE(0,0) ---")
    sc_host = score_input[0, 0, :].astype(np.float32)
    sc_dev  = score_device[0, 0, :].astype(np.float32)
    vc_host = VCache_tile[0, 0, :].astype(np.float32)
    vc_dev  = vcache_device[0, 0, :].astype(np.float32)
    print(f"  score  (host): {sc_host}")
    print(f"  score  (dev):  {sc_dev}")
    print(f"  score match:   {np.allclose(sc_host, sc_dev)}")
    print(f"  vcache (host): {vc_host}")
    print(f"  vcache (dev):  {vc_dev}")
    print(f"  vcache match:  {np.allclose(vc_host, vc_dev)}")

    # ─── Step-by-step trace for PE(0,0) ──────────────────────────────────
    print(f"\n--- Step-by-step trace PE(0,0) ---")
    sc = sc_dev  # use device values for trace
    vc = vc_dev
    print(f"  Score layout (iter_num-packed, {bsz}x{gqa_group_size}x{iter_num}):")
    for b in range(bsz):
        for g in range(gqa_group_size):
            off = b * gqa_group_size * iter_num + g * iter_num
            print(f"    score[b={b},g={g}] = {sc[off:off+iter_num]}")

    print(f"  V cache layout ({bsz}x{max_seq_len_p_pe}x{kv_dim_p_pe}):")
    for b in range(bsz):
        vb = b * max_seq_len_p_pe * kv_dim_p_pe
        for t in range(max_seq_len_p_pe):
            vt = vb + t * kv_dim_p_pe
            print(f"    V[b={b},t={t}] = {vc[vt:vt+kv_dim_p_pe]}")

    print(f"\n  Numpy ref computation:")
    for b in range(bsz):
        vb = b * max_seq_len_p_pe * kv_dim_p_pe
        V_mat = np.zeros((iter_num, kv_dim_p_pe), dtype=np.float32)
        for t in range(iter_num):
            V_mat[t, :] = vc[vb + t * kv_dim_p_pe:vb + t * kv_dim_p_pe + kv_dim_p_pe]
        for g in range(gqa_group_size):
            s_off = b * gqa_group_size * iter_num + g * iter_num
            s_vec = sc[s_off:s_off + iter_num]
            out = s_vec @ V_mat
            print(f"    ref[b={b},g={g}] = score{s_vec} @ V{V_mat.tolist()} = {out}")

    print(f"\n  Simulated v0 (CSL logic on host-side data):")
    print(f"  left_vector = ptr_score, length=Kt={iter_num}")
    print(f"  right_matrix = &XVCache, length=Nt={kv_dim_p_pe}")
    print(f"  out_vector = ptr_output, length=Nt={kv_dim_p_pe}")
    left_off = 0
    right_base = 0
    out_off = 0
    sim_output = np.zeros(output_size, dtype=np.float32)
    for b in range(bsz):
        right_off = right_base
        for g in range(gqa_group_size):
            right_off = right_base  # reload DSR from DSD (same batch V base)
            print(f"    --- b={b}, g={g} ---")
            print(f"    left_off={left_off} right_off={right_off} out_off={out_off}")
            for t in range(iter_num):
                curX = sc[left_off + t]
                V_row = vc[right_off:right_off + kv_dim_p_pe]
                sim_output[out_off:out_off+kv_dim_p_pe] += V_row * curX
                print(f"      t={t}: curX=score[{left_off+t}]={curX:.6f}, "
                      f"V_row=vc[{right_off}:{right_off+kv_dim_p_pe}]={V_row}, "
                      f"acc={sim_output[out_off:out_off+kv_dim_p_pe]}")
                right_off += kv_dim_p_pe  # save_address=true advances src1
            left_off += iter_num  # advance to next group's score
            out_off += kv_dim_p_pe  # advance to next group's output
        right_base += max_seq_len_p_pe * kv_dim_p_pe  # advance to next batch's V

    ref_pe = compute_reference_output(
        score_input, VCache_tile, P, bsz, kv_dim_p_pe,
        gqa_group_size, max_seq_len_p_pe, iter_num
    )[0, 0, :]
    print(f"\n  Simulated output: {sim_output}")
    print(f"  Numpy ref output: {ref_pe}")

    # ─── Run each variant separately, D2H after each to avoid overflow corruption ─
    runner.launch("test_output_gemv", np.int16(iter_num), nonblock=False)
    v0_grid = d2h("output_tile", output_size)

    # Reload score (in case a buggy variant corrupted it)
    h2d("score", score_input, score_buf_size)
    runner.launch("test_output_gemv_v2", np.int16(iter_num), nonblock=False)
    v2_grid = d2h("output_tile", output_size)

    h2d("score", score_input, score_buf_size)
    runner.launch("test_output_gemv_v3", np.int16(iter_num), nonblock=False)
    v3_grid = d2h("output_tile", output_size)

    h2d("score", score_input, score_buf_size)
    runner.launch("test_output_gemv_v4", np.int16(iter_num), nonblock=False)
    v4_grid = d2h("output_tile", output_size)

    runner.stop()

    # ─── Device v0 trace ─────────────────────────────────────────────────
    print(f"\n  Device v0 output PE(0,0):")
    v0_pe = v0_grid[0, 0, :].astype(np.float32)
    for b in range(bsz):
        for g in range(gqa_group_size):
            off = b * gqa_group_size * kv_dim_p_pe + g * kv_dim_p_pe
            print(f"    v0[b={b},g={g}] = {v0_pe[off:off+kv_dim_p_pe]}")
    print(f"  Sim vs v0 match: {np.allclose(sim_output, v0_pe, atol=2e-3)}")

    # ─── Numpy reference ─────────────────────────────────────────────────
    ref_grid = compute_reference_output(
        score_input, VCache_tile, P, bsz, kv_dim_p_pe,
        gqa_group_size, max_seq_len_p_pe, iter_num
    )

    # ─── Compare ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Comparing output GEMV: v0 vs v2 vs v3 vs v4 vs numpy ref")
    print(f"Output shape per PE: [{bsz}, {gqa_group_size}, {kv_dim_p_pe}]")
    print(f"{'='*60}")

    def compare_pair(name_a, name_b, grid_a, grid_b):
        """Compare two output grids. Returns (pass, msg_str)."""
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
                        off = bi * gqa_group_size * kv_dim_p_pe + g * kv_dim_p_pe
                        a_slice = a[off:off + kv_dim_p_pe]
                        b_slice = b_arr[off:off + kv_dim_p_pe]

                        abs_diff = np.abs(a_slice - b_slice)
                        max_abs_diff = max(max_abs_diff, abs_diff.max())

                        nonzero = np.abs(a_slice) > 1e-8
                        if nonzero.any():
                            rel = abs_diff[nonzero] / np.abs(a_slice[nonzero])
                            max_rel_diff = max(max_rel_diff, rel.max())

                        bad = np.where(abs_diff > 1e-3)[0]
                        mismatch_count += len(bad)
                        total_checked += kv_dim_p_pe

                        if len(bad) > 0 and len(first_mismatches) < 10:
                            for k in bad:
                                first_mismatches.append(
                                    f"    PE({px},{py}) b={bi} g={g} k={k}: "
                                    f"{name_a}={a_slice[k]:.6f} {name_b}={b_slice[k]:.6f} "
                                    f"diff={abs_diff[k]:.6e}")
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

    # Cross-compare device variants against v0 (primary validation)
    # All variants include the same all_reduce_bsz_dim, so post-reduce comparison is valid.
    all_pass = True
    variants = [("v0", v0_grid), ("v2", v2_grid), ("v3", v3_grid), ("v4", v4_grid)]
    for name, grid in variants[1:]:
        passed, msg = compare_pair("v0", name, v0_grid, grid)
        print(msg)
        if not passed:
            all_pass = False

    # Compare v0 against numpy ref (local, no reduce — device also skips reduce in test mode)
    passed_ref, ref_msg = compare_pair("v0", "ref", v0_grid, ref_grid)
    print(ref_msg)
    if not passed_ref:
        all_pass = False

    # Sample values
    n = min(4, kv_dim_p_pe)
    print(f"\nSample values from PE(0,0), batch=0, group=0 (first {n} of {kv_dim_p_pe}):")
    print(f"  ref:  {ref_grid[0, 0, :n]}")
    print(f"  v0:   {v0_grid[0, 0, :n].astype(np.float32)}")
    print(f"  v2:   {v2_grid[0, 0, :n].astype(np.float32)}")
    print(f"  v3:   {v3_grid[0, 0, :n].astype(np.float32)}")
    print(f"  v4:   {v4_grid[0, 0, :n].astype(np.float32)}")

    if all_pass:
        print(f"\n  ALL PASS: all variants match numpy reference!")
    else:
        print(f"\n  SOME TESTS FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
