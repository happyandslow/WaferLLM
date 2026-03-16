import json
import os
import struct
import argparse
import numpy as np

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view, calculate_cycles
from cerebras.sdk.debug.debug_util import debug_util
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder

def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)


class Config:
    def __init__(self):
        self.P = 8
        self.bsz = 1
        self.group_num = 2
        self.dim = 64
        self.n_heads = 4
        self.n_kv_heads = 2
        self.head_dim = 16
        self.seq_len = 64
        self.ffn_dim = 64

def parse_args():
    parser = argparse.ArgumentParser(description="GQA decode simulator")
    parser.add_argument("--config", default="config.json", type=str, help="Config file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = Config()

    if not os.path.exists(args.config):
        print("Host: Use default test values.")
    else:
        with open(args.config) as f:
            config.__dict__.update(json.load(f))

    P         = config.P
    bsz       = config.bsz
    group_num = config.group_num
    dim       = config.dim
    n_heads   = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim  = config.head_dim
    seq_len   = config.seq_len
    ffn_dim   = config.ffn_dim

    dim_p_pe       = dim // P
    pes_p_head     = P // n_heads
    pes_p_kv_head  = P // n_kv_heads
    head_dim_p_pe  = head_dim // P
    seq_len_p_pe   = seq_len // P
    ffn_dim_p_pe   = ffn_dim // P

    print(f"Host: P={P}, bsz={bsz}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
          f"dim_p_pe={dim_p_pe}, pes_p_head={pes_p_head}, pes_p_kv_head={pes_p_kv_head}, "
          f"head_dim_p_pe={head_dim_p_pe}, seq_len_p_pe={seq_len_p_pe}, ffn_dim_p_pe={ffn_dim_p_pe}")

    io_dtype     = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    # -------------------------------------------------------------------------- #
    # ----------------------- Generate random test tensors -------------------- #
    # -------------------------------------------------------------------------- #

    X = np.random.rand(1, bsz * dim).astype(np.float16)
    tensor_X = np.tile(X.reshape(P, bsz * dim_p_pe), reps=(1, P))

    W = np.random.rand(1, dim).astype(np.float16)
    tensor_W = np.tile(W.reshape(P, dim_p_pe), reps=(1, P))

    # Q weight: full [dim, dim] — same as MHA
    tensor_q_weight = np.random.rand(dim, dim).astype(np.float16)

    # K / V weights: GQA — one weight matrix per KV head: [n_kv_heads, dim, dim]
    tensor_k_weight = np.random.rand(n_kv_heads, dim, dim).astype(np.float16)
    tensor_v_weight = np.random.rand(n_kv_heads, dim, dim).astype(np.float16)

    _dim_p_pe = dim_p_pe
    if (dim_p_pe % 2) == 1:
        _dim_p_pe = dim_p_pe - 1

    freqs_sin = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
    tensor_freqs_sin = np.tile(freqs_sin.reshape(P, _dim_p_pe // 2), reps=(1, P))
    freqs_cos = np.random.rand(1, P * _dim_p_pe // 2).astype(np.float16)
    tensor_freqs_cos = np.tile(freqs_cos.reshape(P, _dim_p_pe // 2), reps=(1, P))

    # KV caches: GQA — one cache shard per KV head
    # XKCache: [n_kv_heads, dim, seq_len]
    # XVCache: [n_kv_heads, seq_len, dim]
    tensor_XKCache = np.random.rand(n_kv_heads, dim, seq_len).astype(np.float16)
    tensor_XVCache = np.random.rand(n_kv_heads, seq_len, dim).astype(np.float16)

    tensor_o_weight   = np.random.rand(dim, dim).astype(np.float16)
    tensor_up_weight  = np.random.rand(dim, ffn_dim).astype(np.float16)
    tensor_gate_weight = np.random.rand(dim, ffn_dim).astype(np.float16)
    tensor_down_weight = np.random.rand(ffn_dim, dim).astype(np.float16)

    # -------------------------------------------------------------------------- #
    # ------------------------------ Runner setup ----------------------------- #
    # -------------------------------------------------------------------------- #

    runner = SdkRuntime("out", simfab_numthreads=64, msg_level='INFO')
    runner.load()
    runner.run()

    # -------------------------------------------------------------------------- #
    # ------------------------------ Get symbols ------------------------------ #
    # -------------------------------------------------------------------------- #

    sym_X          = runner.get_id("X")
    sym_W          = runner.get_id("W")
    sym_Q_weight   = runner.get_id("Q_weight")
    sym_K_weight   = runner.get_id("K_weight")
    sym_V_weight   = runner.get_id("V_weight")
    sym_freqs_sin  = runner.get_id("freqs_sin")
    sym_freqs_cos  = runner.get_id("freqs_cos")
    sym_XKCache    = runner.get_id("XKCache")
    sym_XVCache    = runner.get_id("XVCache")
    sym_O_weight   = runner.get_id("O_weight")
    sym_UP_weight  = runner.get_id("UP_weight")
    sym_GATE_weight = runner.get_id("GATE_weight")
    sym_DOWN_weight = runner.get_id("DOWN_weight")

    symbol_timer_buf = runner.get_id("timer_buf")
    symbol_timer_ref = runner.get_id("time_ref")
    sym_debug        = runner.get_id("debug")

    # -------------------------------------------------------------------------- #
    # ------------------------------ H2D memcpy ------------------------------ #
    # -------------------------------------------------------------------------- #

    # X
    X_u32 = input_array_to_u32(tensor_X.ravel(), 1, 1)
    runner.memcpy_h2d(sym_X, X_u32, 0, 0, P, P, bsz * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # W (RMSNorm)
    W_u32 = input_array_to_u32(tensor_W.ravel(), 1, 1)
    runner.memcpy_h2d(sym_W, W_u32, 0, 0, P, P, dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # Q_weight — standard MHA tiling
    Q_reshape   = tensor_q_weight.reshape(P, dim_p_pe, P, dim_p_pe)
    Q_transpose = Q_reshape.transpose(0, 2, 1, 3)
    Q_reshape   = Q_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
    Q_u32 = input_array_to_u32(Q_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(sym_Q_weight, Q_u32, 0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # K_weight — GQA: pack all KV-head blocks into a single transfer.
    # Each PE receives: [ block_for_kv_head_0 | block_for_kv_head_1 | ... ]
    # size per PE: n_kv_heads * dim_p_pe * dim_p_pe
    k_blocks = []
    for h in range(n_kv_heads):
        K_reshape   = tensor_k_weight[h].reshape(P, dim_p_pe, P, dim_p_pe)
        K_transpose = K_reshape.transpose(0, 2, 1, 3)
        k_blocks.append(K_transpose.reshape(P, P, dim_p_pe * dim_p_pe))
    K_packed = np.concatenate(k_blocks, axis=2)   # [P, P, n_kv_heads * dim_p_pe * dim_p_pe]
    K_u32 = input_array_to_u32(K_packed.ravel(), 1, 1)
    runner.memcpy_h2d(sym_K_weight, K_u32, 0, 0, P, P, n_kv_heads * dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # V_weight — GQA: same packing as K_weight
    v_blocks = []
    for h in range(n_kv_heads):
        V_reshape   = tensor_v_weight[h].reshape(P, dim_p_pe, P, dim_p_pe)
        V_transpose = V_reshape.transpose(0, 2, 1, 3)
        v_blocks.append(V_transpose.reshape(P, P, dim_p_pe * dim_p_pe))
    V_packed = np.concatenate(v_blocks, axis=2)   # [P, P, n_kv_heads * dim_p_pe * dim_p_pe]
    V_u32 = input_array_to_u32(V_packed.ravel(), 1, 1)
    runner.memcpy_h2d(sym_V_weight, V_u32, 0, 0, P, P, n_kv_heads * dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # freqs_sin / freqs_cos
    freqs_sin_u32 = input_array_to_u32(tensor_freqs_sin.ravel(), 1, 1)
    runner.memcpy_h2d(sym_freqs_sin, freqs_sin_u32, 0, 0, P, P, _dim_p_pe // 2,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    freqs_cos_u32 = input_array_to_u32(tensor_freqs_cos.ravel(), 1, 1)
    runner.memcpy_h2d(sym_freqs_cos, freqs_cos_u32, 0, 0, P, P, _dim_p_pe // 2,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # XKCache — GQA: pack all KV-head K-cache blocks
    # tensor_XKCache: [n_kv_heads, dim, seq_len]
    xk_blocks = []
    for h in range(n_kv_heads):
        XKCache_reshape   = tensor_XKCache[h].reshape(P, dim_p_pe, P, seq_len_p_pe)
        XKCache_transpose = XKCache_reshape.transpose(0, 2, 1, 3)
        xk_blocks.append(XKCache_transpose.reshape(P, P, dim_p_pe * seq_len_p_pe))
    XKCache_packed = np.concatenate(xk_blocks, axis=2)  # [P, P, n_kv_heads * dim_p_pe * seq_len_p_pe]
    XKCache_u32 = input_array_to_u32(XKCache_packed.ravel(), 1, 1)
    runner.memcpy_h2d(sym_XKCache, XKCache_u32, 0, 0, P, P, n_kv_heads * dim_p_pe * seq_len_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # XVCache — GQA: pack all KV-head V-cache blocks
    # tensor_XVCache: [n_kv_heads, seq_len, dim]
    xv_blocks = []
    for h in range(n_kv_heads):
        XVCache_reshape   = tensor_XVCache[h].reshape(P, seq_len_p_pe, P, dim_p_pe)
        XVCache_transpose = XVCache_reshape.transpose(0, 2, 1, 3)
        xv_blocks.append(XVCache_transpose.reshape(P, P, seq_len_p_pe * dim_p_pe))
    XVCache_packed = np.concatenate(xv_blocks, axis=2)  # [P, P, n_kv_heads * seq_len_p_pe * dim_p_pe]
    XVCache_u32 = input_array_to_u32(XVCache_packed.ravel(), 1, 1)
    runner.memcpy_h2d(sym_XVCache, XVCache_u32, 0, 0, P, P, n_kv_heads * seq_len_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # O_weight
    O_reshape   = tensor_o_weight.reshape(P, dim_p_pe, P, dim_p_pe)
    O_transpose = O_reshape.transpose(0, 2, 1, 3)
    O_reshape   = O_transpose.reshape(P, P, dim_p_pe * dim_p_pe)
    O_u32 = input_array_to_u32(O_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(sym_O_weight, O_u32, 0, 0, P, P, dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # UP_weight
    UP_reshape   = tensor_up_weight.reshape(P, dim_p_pe, P, ffn_dim_p_pe)
    UP_transpose = UP_reshape.transpose(0, 2, 1, 3)
    UP_reshape   = UP_transpose.reshape(P, P, dim_p_pe * ffn_dim_p_pe)
    UP_u32 = input_array_to_u32(UP_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(sym_UP_weight, UP_u32, 0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # GATE_weight
    GATE_reshape   = tensor_gate_weight.reshape(P, dim_p_pe, P, ffn_dim_p_pe)
    GATE_transpose = GATE_reshape.transpose(0, 2, 1, 3)
    GATE_reshape   = GATE_transpose.reshape(P, P, dim_p_pe * ffn_dim_p_pe)
    GATE_u32 = input_array_to_u32(GATE_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(sym_GATE_weight, GATE_u32, 0, 0, P, P, dim_p_pe * ffn_dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # DOWN_weight  (ffn_dim -> dim, so tile is [ffn_dim_p_pe x dim_p_pe])
    DOWN_reshape   = tensor_down_weight.reshape(P, ffn_dim_p_pe, P, dim_p_pe)
    DOWN_transpose = DOWN_reshape.transpose(0, 2, 1, 3)
    DOWN_reshape   = DOWN_transpose.reshape(P, P, ffn_dim_p_pe * dim_p_pe)
    DOWN_u32 = input_array_to_u32(DOWN_reshape.ravel(), 1, 1)
    runner.memcpy_h2d(sym_DOWN_weight, DOWN_u32, 0, 0, P, P, ffn_dim_p_pe * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    # -------------------------------------------------------------------------- #
    # ------------------------------ Run simulator ---------------------------- #
    # -------------------------------------------------------------------------- #

    runner.launch("init_task", nonblock=False)

    repeat_steps = 1
    warmup_steps = 0
    runner.launch("decode_host", np.int16(repeat_steps), np.int16(warmup_steps), nonblock=False)

    # -------------------------------------------------------------------------- #
    # ------------------------------ D2H memcpy ------------------------------ #
    # -------------------------------------------------------------------------- #

    debug_1d_u32 = np.zeros(P * bsz * dim, dtype=np.uint32)
    runner.memcpy_d2h(debug_1d_u32, sym_debug, 0, 0, P, P, bsz * dim_p_pe,
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
    debug = memcpy_view(debug_1d_u32, np.dtype(np.float16))
    debug = debug.reshape(P, bsz * dim)

    # Timer
    timer_buf_1d_u32 = np.zeros((P * P * 3), dtype=np.uint32)
    runner.memcpy_d2h(timer_buf_1d_u32, symbol_timer_buf, 0, 0, P, P, 3, streaming=False,
                      data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    timer_buf_time_hwl = timer_buf_1d_u32.view(np.float32).reshape((P, P, 3))

    runner.stop()

    # -------------------------------------------------------------------------- #
    # ------------------------------ Debug Check ------------------------------ #
    # -------------------------------------------------------------------------- #

    print("Expected Result (input X):")
    print(X)
    print("Simulated Result (debug output):")
    print(debug)

    debug_mod = debug_util("out")
    core_offset_x = 4
    core_offset_y = 1

    # Uncomment to enable per-PE trace dumps:
    # for px in range(P):
    #     for py in range(P):
    #         trace_output = debug_mod.read_trace(core_offset_x+px, core_offset_y+py, 'debug_main')
    #         print(f"PE: {px}, {py}  {trace_output}")

    # -------------------------------------------------------------------------- #
    # ------------------------------ Compute time ------------------------------ #
    # -------------------------------------------------------------------------- #

    cycles_count = np.zeros((P, P))
    for pe_x in range(P):
        for pe_y in range(P):
            cycles_count[pe_y, pe_x] = calculate_cycles(timer_buf_time_hwl[pe_y, pe_x, :])

    cycles_count_mean = cycles_count.mean()
    print(f"Host: mean cycles count: {cycles_count_mean / repeat_steps}")

if __name__ == "__main__":
    main()
