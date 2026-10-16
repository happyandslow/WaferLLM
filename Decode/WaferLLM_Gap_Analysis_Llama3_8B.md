# WaferLLM Gap Analysis: Llama3-8B on WSE-3

**Date:** April 2026
**Scope:** Prefill (`WaferServe/WaferLLM/Prefill/WSE-3`) | Decode (`WaferLLM/Decode/WSE-3-GQA`)

---

## 1. Current Implementation: Single Transformer Layer

| Operation | Prefill | Decode (GQA) | Notes |
|---|---|---|---|
| RMSNorm (pre-attention) | Yes | Yes | |
| Q / K / V Projections | Yes | Yes | GEMM (prefill) / GEMV (decode) |
| RoPE | Yes | **Disabled** | Strided DSD `@fmulh` bug in decode |
| KV Cache Write | No (allocated) | Yes | Incremental write in decode |
| Causal Masking | **No** | N/A | Autoregressive by nature in decode |
| Score: Q @ K^T | Yes | Yes | Variable-length in decode |
| Softmax | Yes | Yes | `fast_exp()` approx in decode |
| Output: attn @ V | Yes | Yes | 4 variants in decode |
| O-Projection | Yes | Yes | |
| Residual Add | Yes | Yes | |
| RMSNorm (pre-FFN) | Yes | Yes | |
| UP Projection (SwiGLU) | Yes | Yes | |
| GATE Projection (SwiGLU) | Yes | Yes | |
| SiLU Activation | Yes | Yes | |
| DOWN Projection | Yes | Yes | |
| FFN Residual Add | Yes | Yes | |

---

## 2. Missing Operators & Features for Llama3-8B

| Priority | Operator / Feature | Complexity | Notes |
|---|---|---|---|
| **P0** | Fix RoPE in Decode | Medium | Debug strided DSD bug or restructure memory layout |
| **P0** | Causal Mask in Prefill | Low | Add `-inf` masking to upper triangle before softmax |
| **P0** | Multi-Layer Loop + Weight Mgmt | High | 32 layers; weight streaming or full on-chip storage |
| **P0** | LM Head (Vocab Projection) | High | 4096 → 128,256 GEMV; largest single operation (~1 GB) |
| **P0** | Sampling (argmax minimum) | Medium | Distributed argmax reduction over 128K sharded logits |
| **P1** | Token Embedding Lookup | Medium | Host-side compute + transfer, or on-chip scatter |
| **P1** | Final RMSNorm | Low | Reuse existing RMSNorm kernel with separate weight vector |
| **P1** | Per-Layer KV Cache Management | High | 32 × KV caches; memory allocation strategy needed |
| **P1** | Host ↔ WSE Orchestration | High | Weight loading, token I/O loop, pipeline control |
| **P2** | Top-k / Top-p Sampling | Medium | More complex distributed reduction |
| **P2** | Tokenizer / Detokenizer | Low | Host-side; use existing libraries (tiktoken) |

---

## 3. Llama3-8B Inference Pipeline

```
Tokenizer ──► Embedding ──► Transformer x32 ──► Final RMSNorm ──► LM Head ──► Sampling ──► Detokenizer
(Host CPU)     Lookup        Layers              (post-loop)       (Vocab Proj)  (argmax)    (Host CPU)
  [MISSING]    [MISSING]     [PARTIAL]           [MISSING]         [MISSING]     [MISSING]   [MISSING]
```

**Inside each Transformer Layer (implemented):**
```
Attention: RMSNorm → Q/K/V Proj → RoPE* → Score (Q@K^T) → Causal Mask** → Softmax → Attn@V → O-Proj → Residual Add
FFN:       RMSNorm → UP Proj → GATE Proj → SiLU → DOWN Proj → Residual Add
```
> \* RoPE disabled in Decode (strided DSD bug)
> \*\* Causal mask missing in Prefill

---

## 4. P0 Critical Items — Detail

### 4.1 Fix RoPE in Decode
**Complexity:** Medium

RoPE is disabled due to a strided DSD `@fmulh` bug. Without positional encoding, the model produces garbage. Options:
- (a) Fix the strided DSD issue
- (b) Restructure memory layout to avoid strided access
- (c) Pre-apply RoPE on host

### 4.2 Causal Mask in Prefill
**Complexity:** Low

Current prefill computes full `seq_len × seq_len` attention with no mask. Need to set upper-triangle scores to `-inf` before softmax. Simple element-wise operation on the score matrix.

### 4.3 Multi-Layer Loop + Weight Management
**Complexity:** High

Llama3-8B has 32 layers. Options:
- (a) **Weight streaming** from host per layer (~14 GB total) — simple but slow
- (b) **All weights on-chip** (feasible on WSE-3 ~40 GB SRAM) — fast but requires careful memory management
- (c) **Hybrid** — keep hot layers on-chip, stream cold layers

Each layer needs separate weights and KV cache.

### 4.4 LM Head (Vocab Projection)
**Complexity:** High

Final `[1, 4096] × [4096, 128256]` projection. The weight matrix alone is ~1 GB in f16. Llama3 ties embedding and LM head weights. Requires new sharding strategy for the extremely wide output dimension.

### 4.5 Sampling (argmax)
**Complexity:** Medium

Distributed argmax over 128K logits sharded across PEs. Need a new reduction primitive that returns both max value and its index. Temperature scaling is a simple element-wise multiply before argmax.

---

## 5. P1 / P2 Items — Detail

### P1: Token Embedding Lookup
**Complexity:** Medium

Convert token IDs to 4096-dim vectors. 128256 × 4096 table (~1 GB). Can be done on host and transferred, or implement on-chip indexed scatter.

### P1: Final RMSNorm
**Complexity:** Low

Same kernel as existing RMSNorm, just needs to be invoked post-loop with a separate weight vector. Minimal new code.

### P1: Per-Layer KV Cache Management
**Complexity:** High

32 layers × KV cache. At seq_len=8192: ~512 MB total. Need memory allocation strategy across PEs and lifetime management.

### P1: Host ↔ WSE Orchestration
**Complexity:** High

Full pipeline control: weight loading, token I/O per decode step, KV cache init from prefill. Extend launch scripts to support the full autoregressive loop.

### P2: Top-k / Top-p Sampling
**Complexity:** Medium

More sophisticated sampling beyond argmax. Requires distributed partial sort or threshold-based filtering across PEs.

### P2: Tokenizer / Detokenizer
**Complexity:** Low

Host-side only. Use tiktoken or sentencepiece. Not a WSE operator.

---

## 6. Llama3-8B Memory Budget on WSE-3

| Component | Size (f16) | Per Layer | Total (32 layers) |
|---|---|---|---|
| Q / O Weights | 2 × 4096 × 4096 × 2B = 64 MB | 64 MB | ~2 GB |
| K / V Weights (GQA: 4096×1024) | 2 × 4096 × 1024 × 2B = 16 MB | 16 MB | ~0.5 GB |
| FFN Weights (UP + GATE + DOWN) | 3 × 4096 × 14336 × 2B = 336 MB | 336 MB | ~10.5 GB |
| RMSNorm Weights | 2 × 4096 × 2B = 16 KB | 16 KB | ~0.5 MB |
| KV Cache (seq=8192, bsz=1) | 2 × 8 × 128 × 8192 × 2B = 32 MB | 32 MB | ~1 GB |
| Embedding / LM Head | 128256 × 4096 × 2B = 1 GB | -- | 1 GB (shared) |
| **Total** | | | **~15.5 GB** |

> WSE-3 has ~40 GB distributed SRAM across all PEs.
> Llama3-8B at ~15.5 GB fits comfortably with room for activations and intermediates.
> **All-on-chip strategy is feasible — no weight streaming required.**
