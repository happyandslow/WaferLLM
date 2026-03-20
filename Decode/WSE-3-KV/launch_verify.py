"""
launch_verify.py  –  Verification test for WSE-3-KV-consecutive Decode module.

This script runs the same simulation as launch_sim.py and then verifies four
intermediate device-side tensors against Python-computed reference values:

  • xkcache_result    – K-cache buffer (XKCache) after repeat_steps decodes
  • xvcache_result    – V-cache buffer (XVCache) after repeat_steps decodes
  • intermediate_score – softmax attention scores from the last decode step
  • intermediate_result – attention output (softmax @ V-cache) from the last step

Data-layout / algorithm notes (derived from decode.csl):

  PE assignment
  ─────────────
  PE(px, py) receives  X[py*d : (py+1)*d]  (all PEs in the same row share the same
  input slice).  After all_reduce_bsz_dim_QKV_fusion (reduce along the y/py axis):
    QKV_tile[0:d]   = Xq[px*d : (px+1)*d]
    QKV_tile[d:2d]  = Xk[px*d : (px+1)*d]
    QKV_tile[2d:3d] = Xv[px*d : (px+1)*d]

  KV-cache storage  (process_kv)
  ──────────────────────────────
  Only PE row  py == (step % P)  stores K and V for that decode step.

  K-cache layout – dim-major, step is finest dimension
  (range_copy_2d_strided: dest stride = max_seq_len_p_pe):
    XKCache_tile[b * d * sl + i * sl + t_local] = Xk[b, px*d + i]
  where i ∈ [0, dim_p_pe), t_local ∈ [0, iter_num_for_this_row).

  V-cache layout – time-major, dim is finest dimension (unchanged):
    XVCache_tile[b * d * sl + t_local * d + j] = Xv[b, px*d + j]
  where t_local ∈ [0, iter_num_for_this_row), j ∈ [0, dim_p_pe).

  Score computation  (score_matvec_mult)
  ──────────────────────────────────────
  Uses vecmat_computation_var_right_2d_strided(right_stride=max_seq_len_p_pe):
  the right matrix advances by sl=max_seq_len_p_pe per Q element (not iter_num),
  so consecutive dim-rows of the K-cache are visited:
    partial_score[b, t] = Σ_i  Q_tile[b, i] * XKCache_flat[b*d*sl + i*sl + t]
  With K-cache layout XKCache_flat[b*d*sl + i*sl + t] = K[b, px*d + i], this is
  a standard dot product repeated for every t: partial_score[b, t] = Q[b]·K[b].
  After all-reduce over px → global_score[t].

  Softmax  (softmax_score)
  ────────────────────────
  • CSL fast_exp:  tmp = 1 + x/256;  tmp *= tmp;  tmp *= tmp  → (1+x/256)^4
  • Initial max seed = 0.0 (not −∞); all-reduceMax over py axis.
  • Sum via all_reduce over py axis; normalize.

  Output computation  (output_matvec_mult)
  ─────────────────────────────────────────
  Standard time-major access (Kt = iter_num, Nt = d):
    partial_output[b, j] = Σ_t  score[b, t] * XVCache_flat[b*d*sl + t*d + j]
  After all-reduce over py → global_output.
"""

import json
import os
import struct
import argparse
import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view, calculate_cycles
from cerebras.sdk.debug.debug_util import debug_util
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder


# ====================================================================== #
# ==================  Python reference implementation  ================= #
# ====================================================================== #

def fast_exp_f16(x: np.float16) -> np.float16:
    """
    Replicates the CSL fast_exp function in float16 arithmetic.

    CSL code:
        tmp = 1.0 + tmp/256.0;
        tmp *= tmp;   // ^2
        tmp *= tmp;   // ^4
        return tmp;   // = (1 + x/256)^4  ≈  e^x
    """
    tmp = np.float16(1.0) + np.float16(x) / np.float16(256.0)
    tmp = tmp * tmp
    tmp = tmp * tmp
    return np.float16(tmp)


def _build_kvcache_flat(Xk, Xv, px: int, iter_num: int,
                        bsz: int, dim_p_pe: int, max_seq_len_p_pe: int):
    """
    Build the expected flat K-cache and V-cache for PE (px, py).

    K-cache layout (range_copy_2d_strided, dest stride = max_seq_len_p_pe):
        k_flat[b * d * sl + i * sl + t_local] = K[b, px*d + i]
    Dim `i` is the outer dimension; step `t_local` is the finest (stride-1)
    dimension.  Elements at t_local ≥ iter_num remain zero.

    V-cache layout unchanged (range_copy_2d_strided, dest stride = 1):
        v_flat[b * d * sl + t_local * d + j] = V[b, px*d + j]
    Time `t_local` is outer; dim `j` is finest.  Elements at t_local ≥ iter_num
    remain zero.
    """
    d  = dim_p_pe
    sl = max_seq_len_p_pe
    k_flat = np.zeros(bsz * d * sl, dtype=np.float16)
    v_flat = np.zeros(bsz * d * sl, dtype=np.float16)
    for b in range(bsz):
        # NEW K-cache: dim-major (i outer), time is finest dimension (stride 1).
        # k_flat[b*d*sl + i*sl + t] = K[b, px*d + i]  for t in [0, iter_num)
        for i in range(d):
            k_flat[b*d*sl + i*sl : b*d*sl + i*sl + iter_num] = Xk[b, px*d + i]
        # V-cache: time-major (t outer), dim is finest (unchanged).
        # v_flat[b*d*sl + t*d + j] = V[b, px*d + j]  for t in [0, iter_num)
        for t in range(iter_num):
            off = b * d * sl + t * d
            v_flat[off : off + d] = Xv[b, px * d : (px + 1) * d]
    return k_flat, v_flat


def _partial_score(Q_tile, k_flat, iter_num: int,
                   bsz: int, dim_p_pe: int, max_seq_len_p_pe: int):
    """
    Compute one PE's partial score contribution.

    Replicates vecmat_computation_var_right_2d_strided(right_stride=max_seq_len_p_pe)
    with Kt=dim_p_pe, Nt=iter_num:
        score[b, t] = Σ_i  Q_tile[b, i] * k_flat[b*d*sl + i*sl + t]

    The right matrix DSD is manually advanced by right_stride=sl per Q element
    (not auto-advanced by Nt), so each dim-row i of the K-cache contributes its
    iter_num stored values.  Since k_flat[b*d*sl + i*sl + t] = K[b, px*d + i]
    for all t in [0, iter_num), this is a standard Q·K dot product repeated for
    every score slot t.
    """
    d  = dim_p_pe
    sl = max_seq_len_p_pe
    # k_3d[b, i, t] = k_flat[b*d*sl + i*sl + t]  (natural reshape)
    k_3d = k_flat.reshape(bsz, d, sl)[:, :, :iter_num]   # [bsz, d, iter_num]
    # Dot product over i dimension  (float32 for accumulation accuracy)
    partial = np.einsum('bi,bit->bt',
                        Q_tile.astype(np.float32),
                        k_3d.astype(np.float32)).astype(np.float16)
    return partial


def _partial_output(score_py, v_flat, iter_num: int,
                    bsz: int, dim_p_pe: int, max_seq_len_p_pe: int):
    """
    Compute one PE's partial output contribution.

    Replicates vecmat_computation_strided with Kt=iter_num, Nt=dim_p_pe:
        output[b, j] = Σ_t  score[b, t] * v_flat[b*d*sl + t*d + j]

    V-cache uses standard time-major (t*d+j) access (width_right = d*sl
    advances the right matrix per batch, not per t step).
    """
    d  = dim_p_pe
    sl = max_seq_len_p_pe
    # v_flat[b*d*sl + t*d + j] = v_flat.reshape(bsz, sl, d)[b, t, j]
    v_3d = v_flat.reshape(bsz, sl, d)[:, :iter_num, :]   # [bsz, iter_num, d]
    output = np.einsum('bt,btj->bj',
                       score_py.astype(np.float32),
                       v_3d.astype(np.float32)).astype(np.float16)
    return output


def compute_expected_values(X_flat, q_weight, k_weight, v_weight,
                            P: int, bsz: int, dim_p_pe: int,
                            max_seq_len_p_pe: int, repeat_steps: int,
                            alpha: np.float16):
    """
    Compute the four reference tensors that should match the device outputs.

    Parameters
    ----------
    X_flat      : float16 array, shape (bsz * dim,)  –  raw input vector
    q_weight    : float16 array, shape (dim, dim)
    k_weight    : float16 array, shape (dim, dim)
    v_weight    : float16 array, shape (dim, dim)
    P           : PE grid side length
    bsz         : batch size
    dim_p_pe    : hidden-dim slice per PE  (= dim / P)
    max_seq_len_p_pe : sequence-length slice per PE  (= max_seq_len / P)
    repeat_steps: number of decode iterations (X fixed across all steps)
    alpha       : attention scale  1 / sqrt(head_dim)

    Returns
    -------
    exp_xkcache  : float16 (P, bsz * dim * max_seq_len_p_pe)
    exp_xvcache  : float16 (P, bsz * dim * max_seq_len_p_pe)
    exp_score    : float16 (P, bsz * max_seq_len)
    exp_output   : float16 (P, bsz * dim)
    """
    dim       = P * dim_p_pe
    max_seq   = P * max_seq_len_p_pe
    d, sl     = dim_p_pe, max_seq_len_p_pe

    # ------------------------------------------------------------------ #
    # 1. Full QKV projections                                             #
    #    Device uses  ptr_X  (not normalised) for Q, K, V projections.   #
    #    After all_reduce_bsz_dim_QKV_fusion (y-axis sum), each PE holds #
    #    the complete Q/K/V slice for its px column.                      #
    #                                                                     #
    #    X_flat is in PE-interleaved layout (NOT batch-major):            #
    #      X_flat[py*bsz*d + j*d : py*bsz*d + (j+1)*d]                  #
    #        = request-j's features at PE-row py's dimension slice        #
    #    We must reshape as (P, bsz, dim_p_pe) then transpose to          #
    #    (bsz, P, dim_p_pe) = (bsz, dim) to get per-request vectors.     #
    # ------------------------------------------------------------------ #
    X_pe_layout = X_flat.reshape(P, bsz, dim_p_pe)          # [P, bsz, d]
    X_per_batch = (X_pe_layout                               # [bsz, P, d]
                   .transpose(1, 0, 2)
                   .reshape(bsz, dim)
                   .astype(np.float16))                      # [bsz, dim]
    Xq = np.matmul(X_per_batch.astype(np.float32),
                   q_weight.astype(np.float32)).astype(np.float16)  # [bsz, dim]
    Xk = np.matmul(X_per_batch.astype(np.float32),
                   k_weight.astype(np.float32)).astype(np.float16)
    Xv = np.matmul(X_per_batch.astype(np.float32),
                   v_weight.astype(np.float32)).astype(np.float16)

    # ------------------------------------------------------------------ #
    # 2. iter_num per PE row                                              #
    #    PE row py stores on steps where  step % P == py.                #
    # ------------------------------------------------------------------ #
    iter_nums = np.zeros(P, dtype=int)
    for s in range(repeat_steps):
        iter_nums[s % P] += 1

    # ------------------------------------------------------------------ #
    # 3. Build KV-cache per PE                                            #
    # ------------------------------------------------------------------ #
    k_cache_pes = np.zeros((P, P, bsz * d * sl), dtype=np.float16)
    v_cache_pes = np.zeros((P, P, bsz * d * sl), dtype=np.float16)
    for py in range(P):
        for px in range(P):
            kf, vf = _build_kvcache_flat(
                Xk, Xv, px, iter_nums[py], bsz, d, sl)
            k_cache_pes[py, px] = kf
            v_cache_pes[py, px] = vf

    # ------------------------------------------------------------------ #
    # 4. Score computation (last decode step, fixed X)                   #
    #    Partial score from each px, then all-reduce (sum) over px.      #
    # ------------------------------------------------------------------ #
    # scores_x[py, b, t]  –  full score after x-axis all-reduce
    scores_x = np.zeros((P, bsz, sl), dtype=np.float16)
    for py in range(P):
        itn = iter_nums[py]
        if itn == 0:
            continue
        full = np.zeros((bsz, itn), dtype=np.float16)
        for px in range(P):
            Q_tile = Xq[:, px * d : (px + 1) * d]  # [bsz, d]
            part = _partial_score(Q_tile, k_cache_pes[py, px], itn, bsz, d, sl)
            full = (full.astype(np.float32) + part.astype(np.float32)).astype(np.float16)
        # Scale by alpha  (device: @fmulh score * alpha)
        full = (full.astype(np.float32) * float(alpha)).astype(np.float16)
        scores_x[py, :, :itn] = full

    # ------------------------------------------------------------------ #
    # 5. Softmax (distributed over py axis)                               #
    #    Mirrors softmax_score() in decode.csl:                           #
    #      • @fmaxh with initial max=0.0  (not −∞)                       #
    #      • all_reduceMax_bsz over py                                   #
    #      • score_tmp = score − max;  score = fast_exp(score_tmp)       #
    #      • all_reduce_bsz (sum) over py;  score /= sum                 #
    # ------------------------------------------------------------------ #
    # 5a. Global max (per batch), starting from 0.0
    global_max = np.zeros(bsz, dtype=np.float16)
    for py in range(P):
        itn = iter_nums[py]
        if itn == 0:
            continue
        for b in range(bsz):
            lmax = float(np.max(scores_x[py, b, :itn]))
            global_max[b] = np.float16(max(float(global_max[b]), lmax))

    # 5b. Subtract max, apply fast_exp
    score_exp = np.zeros((P, bsz, sl), dtype=np.float16)
    for py in range(P):
        itn = iter_nums[py]
        if itn == 0:
            continue
        for b in range(bsz):
            for t in range(itn):
                shifted = np.float16(scores_x[py, b, t] - global_max[b])
                score_exp[py, b, t] = fast_exp_f16(shifted)

    # 5c. Global sum (per batch)
    global_sum = np.zeros(bsz, dtype=np.float16)
    for py in range(P):
        itn = iter_nums[py]
        if itn == 0:
            continue
        for b in range(bsz):
            lsum = np.float16(np.sum(score_exp[py, b, :itn].astype(np.float32)))
            global_sum[b] = np.float16(float(global_sum[b]) + float(lsum))

    # 5d. Normalise
    softmax_s = np.zeros((P, bsz, sl), dtype=np.float16)
    for py in range(P):
        itn = iter_nums[py]
        if itn == 0:
            continue
        for b in range(bsz):
            inv = np.float16(1.0) / global_sum[b]
            softmax_s[py, b, :itn] = (score_exp[py, b, :itn].astype(np.float32) *
                                      float(inv)).astype(np.float16)

    # ------------------------------------------------------------------ #
    # 6. Output computation  (softmax @ V-cache, then y-axis all-reduce) #
    #                                                                     #
    #  decode_struct calls reconfig_allreduce_axis(1) before             #
    #  output_matvec_mult, so the all-reduce inside that function runs   #
    #  along the Y axis (across PE rows / py dimension).                 #
    #                                                                     #
    #  Layout:                                                            #
    #    • Different PE ROWS (py) hold V-cache slices for different       #
    #      sequence-step ranges (step % P == py).                         #
    #    • Different PE COLUMNS (px) hold different feature-dim slices.   #
    #    • Softmax scores also live on PE rows (same py partitioning).    #
    #                                                                     #
    #  Each PE (px, py) computes a partial output:                        #
    #    partial[b, j] = Σ_{t in py's steps}  score[py, b, t] * V[b,j]   #
    #  The Y-axis all-reduce sums these partials across all py in the    #
    #  same column px, yielding the complete attention output for that   #
    #  feature-dim slice.  After the reduce every PE in column px has    #
    #  the same value (the reduce broadcasts back to all rows).          #
    #                                                                     #
    #  Therefore output_per_px[px] is the right data structure:          #
    #    • indexed by px (one entry per feature-dim column)              #
    #    • the SAME value is replicated across all py rows for that px.  #
    # ------------------------------------------------------------------ #
    output_per_px = np.zeros((P, bsz, d), dtype=np.float16)  # [px, bsz, d]
    for px in range(P):
        full_out = np.zeros((bsz, d), dtype=np.float16)
        for py in range(P):
            itn = iter_nums[py]
            if itn == 0:
                continue
            sp = softmax_s[py, :, :itn]  # [bsz, itn]
            part = _partial_output(sp, v_cache_pes[py, px], itn, bsz, d, sl)
            full_out = (full_out.astype(np.float32) +
                        part.astype(np.float32)).astype(np.float16)
        # After Y-axis all-reduce, all PE rows in column px hold this value
        output_per_px[px] = full_out

    # ------------------------------------------------------------------ #
    # 7. Assemble result arrays in the D2H memcpy layout                  #
    #    ROW_MAJOR D2H from a P×P grid:                                   #
    #      result[py, px*count : (px+1)*count]  =  PE(px, py) data       #
    # ------------------------------------------------------------------ #
    exp_xkcache = np.zeros((P, bsz * dim * sl), dtype=np.float16)
    exp_xvcache = np.zeros((P, bsz * dim * sl), dtype=np.float16)
    for py in range(P):
        for px in range(P):
            off = px * bsz * d * sl
            exp_xkcache[py, off : off + bsz * d * sl] = k_cache_pes[py, px]
            exp_xvcache[py, off : off + bsz * d * sl] = v_cache_pes[py, px]

    exp_score = np.zeros((P, bsz * max_seq), dtype=np.float16)
    for py in range(P):
        itn = iter_nums[py]
        for px in range(P):
            off = px * bsz * sl
            # score is the same for all px after the x-axis all-reduce
            exp_score[py, off : off + bsz * itn] = softmax_s[py, :, :itn].ravel()

    # output: PE(px, py) holds output_per_px[px] for every py in that column
    exp_output = np.zeros((P, bsz * dim), dtype=np.float16)
    for py in range(P):
        for px in range(P):
            off = px * bsz * d
            exp_output[py, off : off + bsz * d] = output_per_px[px].ravel()

    return exp_xkcache, exp_xvcache, exp_score, exp_output


# ====================================================================== #
# ========================  Comparison helpers  ======================== #
# ====================================================================== #

def _check(name: str, actual: np.ndarray, expected: np.ndarray,
           rtol: float = 1e-2, atol: float = 1e-2) -> bool:
    """
    Compare actual vs expected, print a summary, return True if they match.

    NaN/Inf values (which arise from float16 overflow in the fill-value test)
    are reported but do not cause a hard failure – structural layout mismatches
    are more informative in that regime.
    """
    has_nan = np.any(np.isnan(actual)) or np.any(np.isnan(expected))
    has_inf = np.any(np.isinf(actual)) or np.any(np.isinf(expected))

    if has_nan or has_inf:
        print(f"  [{name}] WARNING: NaN/Inf present "
              f"(nan_act={np.any(np.isnan(actual))}, "
              f"inf_act={np.any(np.isinf(actual))}, "
              f"nan_exp={np.any(np.isnan(expected))}, "
              f"inf_exp={np.any(np.isinf(expected))})")
        # Fall back to an exact bit-pattern comparison for Inf/NaN cases
        match = np.array_equal(
            actual.view(np.uint16), expected.view(np.uint16))
        status = "PASS (exact bit match)" if match else "FAIL (bit mismatch)"
        print(f"  [{name}] {status}")
        return match

    close = np.allclose(actual.astype(np.float32),
                        expected.astype(np.float32),
                        rtol=rtol, atol=atol)
    if close:
        print(f"  [{name}] PASS  (max_abs_diff={np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))):.4e})")
    else:
        max_diff  = np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
        bad_idx   = np.unravel_index(
            np.argmax(np.abs(actual.astype(np.float32) - expected.astype(np.float32))),
            actual.shape)
        print(f"  [{name}] FAIL  max_abs_diff={max_diff:.4e}  "
              f"at index {bad_idx}  "
              f"actual={actual[bad_idx]}  expected={expected[bad_idx]}")
    return close


# ====================================================================== #
# ===========================  Config / CLI  =========================== #
# ====================================================================== #

class Config:
    def __init__(self):
        self.P            = 8
        self.bsz          = 1
        self.group_num    = 2
        self.dim          = 64
        self.n_heads      = 1
        self.n_kv_heads   = 1
        self.head_dim     = 64
        self.max_seq_len  = 64
        self.ffn_dim      = 64
        self.layer_num    = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Verification test for WSE-3-KV-consecutive Decode")
    parser.add_argument("--config", default="config.json", type=str)
    parser.add_argument("--seed",   default=42,            type=int,
                        help="NumPy random seed for reproducibility")
    parser.add_argument("--repeat",   default=1,            type=int,
                        help="Number of repeated deocoding steps")
    parser.add_argument("--scale",  default=0.02,          type=float,
                        help="Scale factor applied to random weights/input "
                             "to keep values within float16 range. "
                             "Set to 1.0 to use unscaled fill values (may overflow).")
    parser.add_argument("--use-fill-values", action="store_true",
                        help="Use the same fill-value data as launch_sim.py "
                             "(X=1, Qw=1, Kw=2, Vw=3).  May cause float16 overflow.")
    return parser.parse_args()


# ====================================================================== #
# ==============================  Main  ================================ #
# ====================================================================== #

def main():
    args   = parse_args()
    config = Config()

    if os.path.exists(args.config):
        with open(args.config) as f:
            config.__dict__.update(json.load(f))
    else:
        print("Host: config not found – using default parameters.")

    P               = config.P
    bsz             = config.bsz
    group_num       = config.group_num
    dim             = config.dim
    n_heads         = config.n_heads
    n_kv_heads      = config.n_kv_heads
    head_dim        = config.head_dim
    max_seq_len     = config.max_seq_len
    ffn_dim         = config.ffn_dim
    layer_num       = config.layer_num
    repeat_steps    = args.repeat

    dim_p_pe        = dim        // P
    pes_p_head      = P          // n_heads
    pes_p_kv_head   = P          // n_kv_heads
    head_dim_p_pe   = head_dim   // P
    max_seq_len_p_pe = max_seq_len // P
    ffn_dim_p_pe    = ffn_dim    // P

    _dim_p_pe = (dim_p_pe // 2) * 2   # even part (used for RoPE buffers)

    alpha = np.float16(1.0 / np.sqrt(float(head_dim)))

    print(f"Host: P={P}  bsz={bsz}  dim={dim}  head_dim={head_dim}  "
          f"max_seq_len={max_seq_len}  ffn_dim={ffn_dim}  repeat_steps={repeat_steps}")
    print(f"      dim_p_pe={dim_p_pe}  max_seq_len_p_pe={max_seq_len_p_pe}  "
          f"ffn_dim_p_pe={ffn_dim_p_pe}  alpha={alpha}")

    # ------------------------------------------------------------------ #
    # Data generation                                                     #
    # ------------------------------------------------------------------ #
    np.random.seed(args.seed)

    if args.use_fill_values:
        print("Host: Using fill values (X=1, Qw=1, Kw=2, Vw=3).  "
              "Warning: may overflow float16 for score computation.")
        X = np.zeros((1, bsz * dim), dtype=np.float16)
        for i in range(bsz * dim):
            X[0, i] = 0.1 #(i + 1) * 0.1
        # for i in range(P):
        #     for j in range(bsz):
        #         X[0, i*dim_p_pe*bsz + j*dim_p_pe : i*dim_p_pe*bsz + (j+1)*dim_p_pe] = (i + 1) * 0.1 #j + 1
        tensor_q_weight = np.full((dim, dim), 1.0, dtype=np.float16)
        # for i in range(dim):
        #     for j in range(dim):
        #         tensor_q_weight[i, j] = (j + 1) * 0.1 
        tensor_k_weight = np.full((dim, dim), 2.0, dtype=np.float16)
        for i in range(dim):
            for j in range(dim):
                tensor_k_weight[i, j] = (i * dim + j) * 0.1 
        tensor_v_weight = np.full((dim, dim), 3.0, dtype=np.float16)
    else:
        sc = args.scale
        print(f"Host: Using random data (seed={args.seed}, scale={sc}).")
        X = (np.random.rand(1, bsz * dim) * sc).astype(np.float16)
        tensor_q_weight = (np.random.rand(dim, dim) * sc).astype(np.float16)
        tensor_k_weight = (np.random.rand(dim, dim) * sc).astype(np.float16)
        tensor_v_weight = (np.random.rand(dim, dim) * sc).astype(np.float16)

    # Tiled tensors for H2D (same layout as launch_sim.py)
    tensor_X = np.tile(X.reshape(P, bsz * dim_p_pe), reps=(1, P))

    W = np.ones((1, dim), dtype=np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))

    # RoPE buffers (unused in decode_struct because xq_rope/xk_rope are commented out,
    # but still transferred to the device to satisfy the symbol table).
    freqs_sin = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
    tensor_freqs_sin = np.tile(freqs_sin.reshape(P, _dim_p_pe // 2), reps=(1, P))
    freqs_cos = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
    tensor_freqs_cos = np.tile(freqs_cos.reshape(P, _dim_p_pe // 2), reps=(1, P))

    # Remaining weights (not verified here but needed for the full decode pass)
    tensor_o_weight    = (np.random.rand(dim, dim)          * args.scale).astype(np.float16)
    tensor_up_weight   = (np.random.rand(dim, ffn_dim)      * args.scale).astype(np.float16)
    tensor_gate_weight = (np.random.rand(dim, ffn_dim)      * args.scale).astype(np.float16)
    tensor_down_weight = (np.random.rand(ffn_dim, dim)      * args.scale).astype(np.float16)

    # ------------------------------------------------------------------ #
    # Build H2D weight tiles (same reshape/transpose as launch_sim.py)   #
    # ------------------------------------------------------------------ #
    def _tile_weight(W_full, rows_pe, cols_pe):
        r = W_full.reshape(P, rows_pe, P, cols_pe)
        r = r.transpose(0, 2, 1, 3).reshape(P, P, rows_pe * cols_pe)
        return input_array_to_u32(r.ravel(), 1, 1)

    io_dtype      = MemcpyDataType.MEMCPY_16BIT
    memcpy_order  = MemcpyOrder.ROW_MAJOR

    # ------------------------------------------------------------------ #
    # Simulator setup                                                     #
    # ------------------------------------------------------------------ #
    runner = SdkRuntime("out", simfab_numthreads=64, msg_level="INFO")
    runner.load()
    runner.run()

    # ---------- get symbol IDs ----------
    sym_X           = runner.get_id("X")
    sym_W           = runner.get_id("W")
    sym_Q_weight    = runner.get_id("Q_weight")
    sym_K_weight    = runner.get_id("K_weight")
    sym_V_weight    = runner.get_id("V_weight")
    sym_freqs_sin   = runner.get_id("freqs_sin")
    sym_freqs_cos   = runner.get_id("freqs_cos")
    sym_O_weight    = runner.get_id("O_weight")
    sym_UP_weight   = runner.get_id("UP_weight")
    sym_GATE_weight = runner.get_id("GATE_weight")
    sym_DOWN_weight = runner.get_id("DOWN_weight")
    sym_timer_buf   = runner.get_id("timer_buf")
    sym_time_ref    = runner.get_id("time_ref")
    sym_debug       = runner.get_id("debug")
    sym_score       = runner.get_id("score")
    sym_output      = runner.get_id("output")
    sym_QKV_tile    = runner.get_id("QKV_tile")
    sym_XKCache     = runner.get_id("XKCache")
    sym_XVCache     = runner.get_id("XVCache")

    # ---------- H2D ----------
    X_u32 = input_array_to_u32(tensor_X.ravel(), 1, 1)
    runner.memcpy_h2d(sym_X, X_u32, 0, 0, P, P, bsz * dim_p_pe,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    W_u32 = input_array_to_u32(tensor_W.ravel(), 1, 1)
    runner.memcpy_h2d(sym_W, W_u32, 0, 0, P, P, dim_p_pe,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_Q_weight, _tile_weight(tensor_q_weight, dim_p_pe, dim_p_pe),
                      0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_K_weight, _tile_weight(tensor_k_weight, dim_p_pe, dim_p_pe),
                      0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_V_weight, _tile_weight(tensor_v_weight, dim_p_pe, dim_p_pe),
                      0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    sin_u32 = input_array_to_u32(tensor_freqs_sin.ravel(), 1, 1)
    runner.memcpy_h2d(sym_freqs_sin, sin_u32, 0, 0, P, P, _dim_p_pe // 2,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    cos_u32 = input_array_to_u32(tensor_freqs_cos.ravel(), 1, 1)
    runner.memcpy_h2d(sym_freqs_cos, cos_u32, 0, 0, P, P, _dim_p_pe // 2,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_O_weight, _tile_weight(tensor_o_weight, dim_p_pe, dim_p_pe),
                      0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_UP_weight, _tile_weight(tensor_up_weight, dim_p_pe, ffn_dim_p_pe),
                      0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_GATE_weight, _tile_weight(tensor_gate_weight, dim_p_pe, ffn_dim_p_pe),
                      0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    runner.memcpy_h2d(sym_DOWN_weight, _tile_weight(tensor_down_weight, ffn_dim_p_pe, dim_p_pe),
                      0, 0, P, P, ffn_dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype,
                      order=memcpy_order, nonblock=False)

    # ------------------------------------------------------------------ #
    # Run the simulator                                                   #
    # ------------------------------------------------------------------ #
    runner.launch("init_task", nonblock=False)

    warmup_steps  = 0
    runner.launch("decode_host", np.int16(warmup_steps), np.int16(repeat_steps),
                  nonblock=False)

    # ------------------------------------------------------------------ #
    # D2H: read back the four tensors under test                         #
    # ------------------------------------------------------------------ #
    def _d2h_f16(sym, count_per_pe):
        buf = np.zeros(P * P * count_per_pe, dtype=np.uint32)
        runner.memcpy_d2h(buf, sym, 0, 0, P, P, count_per_pe,
                          streaming=False, data_type=io_dtype,
                          order=memcpy_order, nonblock=False)
        return memcpy_view(buf, np.dtype(np.float16))

    xkcache_flat = _d2h_f16(sym_XKCache, bsz * dim_p_pe * max_seq_len_p_pe)
    xkcache_result = xkcache_flat.reshape(P, bsz * dim * max_seq_len_p_pe)

    xvcache_flat = _d2h_f16(sym_XVCache, bsz * dim_p_pe * max_seq_len_p_pe)
    xvcache_result = xvcache_flat.reshape(P, bsz * dim * max_seq_len_p_pe)

    score_flat = _d2h_f16(sym_score, bsz * max_seq_len_p_pe)
    intermediate_score = score_flat.reshape(P, bsz * max_seq_len)

    output_flat = _d2h_f16(sym_output, bsz * dim_p_pe)
    intermediate_result = output_flat.reshape(P, bsz * dim)

    # (also read back other tensors for diagnostics, not verified here)
    debug_flat = _d2h_f16(sym_debug, bsz * dim_p_pe)
    debug_result = debug_flat.reshape(P, bsz * dim)

    xqkv_flat = _d2h_f16(sym_QKV_tile, bsz * dim_p_pe * 3)
    xqkv_result = xqkv_flat.reshape(P, bsz * dim * 3)

    # Timer
    timer_buf_1d = np.zeros((P*P*3), dtype=np.uint32)
    runner.memcpy_d2h(timer_buf_1d, sym_timer_buf, 0, 0, P, P, 3,
                      streaming=False,
                      data_type=MemcpyDataType.MEMCPY_32BIT,
                      order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    timer_buf = timer_buf_1d.view(np.float32).reshape((P, P, 3))

    runner.stop()

    # ------------------------------------------------------------------ #
    # Python-side reference computation                                   #
    # ------------------------------------------------------------------ #
    print("\nHost: Computing Python reference values ...")
    exp_xkcache, exp_xvcache, exp_score, exp_output = compute_expected_values(
        X_flat            = X.ravel(),
        q_weight          = tensor_q_weight,
        k_weight          = tensor_k_weight,
        v_weight          = tensor_v_weight,
        P                 = P,
        bsz               = bsz,
        dim_p_pe          = dim_p_pe,
        max_seq_len_p_pe  = max_seq_len_p_pe,
        repeat_steps      = repeat_steps,
        alpha             = alpha,
    )

    # ------------------------------------------------------------------ #
    # Verification                                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Verification results")
    print("=" * 60)
    results = {}
    results["xkcache_result"]    = _check("xkcache_result",    xkcache_result,    exp_xkcache)
    results["xvcache_result"]    = _check("xvcache_result",    xvcache_result,    exp_xvcache)
    results["intermediate_score"]= _check("intermediate_score",intermediate_score, exp_score)
    results["intermediate_result"]= _check("intermediate_result",intermediate_result,exp_output)

    all_pass = all(results.values())
    print("=" * 60)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Diagnostic prints (mirrors launch_sim.py)                          #
    # ------------------------------------------------------------------ #
    # Reconstruct per-batch X vectors using the same PE-interleaved layout fix
    X_pe_layout_diag = X.ravel().reshape(P, bsz, dim_p_pe)
    X_per_batch_diag = X_pe_layout_diag.transpose(1, 0, 2).reshape(bsz, dim).astype(np.float16)

    print("\nDiagnostics:")
    print(f"  X (raw PE-interleaved): {X}")
    print(f"  X_per_batch (reconstructed): {X_per_batch_diag}")
    Xq_ref = (X_per_batch_diag.astype(np.float32) @
              tensor_q_weight.astype(np.float32)).astype(np.float16)
    Xk_ref = (X_per_batch_diag.astype(np.float32) @
              tensor_k_weight.astype(np.float32)).astype(np.float16)
    Xv_ref = (X_per_batch_diag.astype(np.float32) @
              tensor_v_weight.astype(np.float32)).astype(np.float16)
    print(f"  Xq (expected):        {Xq_ref}")
    print(f"  Xk (expected):        {Xk_ref}")
    print(f"  Xv (expected):        {Xv_ref}")
    print(f"  xqkv (device row 0):  {xqkv_result[0]}")
    # print(f"  xkcache[row 0, :32]:  {xkcache_result[0, :32]}")
    # print(f"  exp_xkcache[row 0, :32]: {exp_xkcache[0, :32]}")
    # print(f"  xvcache[row 0, :32]:  {xvcache_result[0, :32]}")
    # print(f"  exp_xvcache[row 0, :32]: {exp_xvcache[0, :32]}")
    print(f"  xkcache:  {xkcache_result}")
    print(f"  exp_xkcache: {exp_xkcache}")
    print(f"  xvcache:  {xvcache_result}")
    print(f"  exp_xvcache: {exp_xvcache}")
    # print(f"  score[row 0]:         {intermediate_score[0]}")
    # print(f"  exp_score[row 0]:     {exp_score[0]}")
    # print(f"  output[row 0]:        {intermediate_result[0]}")
    # print(f"  exp_output[row 0]:    {exp_output[0]}")
    print(f"  score:         {intermediate_score}")
    print(f"  exp_score:     {exp_score}")
    print(f"  output:        {intermediate_result}")
    print(f"  exp_output:    {exp_output}")

    # Timing
    cycles = np.zeros((P, P))
    for ey in range(P):
        for ex in range(P):
            cycles[ey, ex] = calculate_cycles(timer_buf[ey, ex, :])
    print(f"\n  Mean cycles per decode step: {cycles.mean() / repeat_steps:.1f}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
