# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WaferLLM is a wafer-scale LLM inference system designed for Cerebras WSE-2/3 accelerators. It implements optimized parallel algorithms for transformer model inference on wafer-scale architectures with hundreds of thousands of cores in a mesh topology. Published at OSDI 2025. [Paper: arXiv:2502.04563](https://arxiv.org/abs/2502.04563)

**Key Technologies:**
- **CSL (Cerebras Software Language)**: Domain-specific language for wafer-scale programming
- **Cerebras SDK**: Required for compilation and execution (v1.2.0 or v1.4.0)
- **PE Mesh Architecture**: Processing Elements arranged in P×P grid with distributed on-chip memory

## Common Development Commands

### Compilation and Execution

Each module (MeshGEMV, MeshGEMM, Decode, Prefill, etc.) follows a consistent pattern:

```bash
# Simulator execution (development/testing)
bash run_sim.sh <params_or_config.json>

# Hardware execution (WSE-2)
bash run_device.sh <params_or_config.json>

# Hardware execution (WSE-3)
bash run_wse3.sh <params_or_config.json> true   # Appliance simulator
bash run_wse3.sh <params_or_config.json> false  # Real device
```

**Examples:**

```bash
# MeshGEMV: bash run_sim.sh P M N group_num
cd MeshGEMV/WSE-3
bash run_sim.sh 64 1024 1024 8

# Decode/Prefill: bash run_sim.sh model_config/test.json
cd Decode/WSE-3-KV
bash run_sim.sh model_config/test.json
```

### Compilation Details

The `run_sim.sh` scripts invoke `cslc` (CSL compiler) with standard patterns:

```bash
cslc --arch=wse3 ./src/layout.csl \
    --fabric-dims="$FABRIC_W","$FABRIC_H" \
    --fabric-offsets=4,1 \
    --params=P:"$P",dim_p_pe:"$dim_p_pe",... \
    -o out --memcpy --channels 1
```

Then execute with: `cs_python launch_sim.py --config <config.json>`

### Cleanup

Simulator runs generate temporary files that are typically cleaned up:
```bash
rm -rf simfab_traces wio_flows_tmpdir.* wsjob-*.json run_meta.json
```

## Architecture and Code Structure

### Module Organization

Each module has a consistent directory structure:

```
<module_name>/WSE-<2|3>/
├── src/
│   ├── layout.csl              # PE grid configuration
│   ├── <module>.csl            # Module implementation (decode.csl, gemv.csl, etc.)
│   └── comm_lib/               # Shared communication library
│       ├── comm_layout.csl     # Communication topology/routing
│       └── comm_pe.csl         # PE-level communication primitives
├── model_config/               # JSON config files (Decode/Prefill only)
├── compile.py                  # Python-based compilation script
├── launch_sim.py               # Simulator launcher
├── launch_wse<2|3>.py         # Hardware launcher
├── run_sim.sh                  # Simulator execution script
├── run_wse<2|3>.sh            # Hardware execution script
└── compile_out/                # Compiled artifacts (generated)
```

### Module Relationships

**Important**: MeshGEMV and MeshGEMM are **standalone reference implementations** that are **NOT directly imported** by other modules. They serve as:
- Algorithmic building blocks demonstrating core wafer-scale matrix operations
- Benchmarking tools for performance evaluation
- Conceptual foundation for operations in Decode/Prefill

**Decode and Prefill** implement similar GEMV/GEMM operations inline, optimized for transformer-specific workloads:
- Decode: Uses `vecmat_computation()` for QKV projections, output projection, FFN layers
- Prefill: Uses `matmul_compute()` and `matmul_T_compute()` for matrix operations
- Both share the same `comm_lib` for hierarchical reduction and broadcast

### Communication Library Architecture

All modules share a common `comm_lib` implementing **hierarchical two-phase communication**:

1. **Phase 1**: Intra-group reduction (within PE groups)
2. **Phase 2**: Inter-group reduction (across group roots)
3. **Broadcast**: Distribute results back to all PEs

This pattern optimizes for wafer-scale mesh where:
- On-chip memory bandwidth is extremely high (tens of PB/s)
- Routing resources are limited
- Latency scales with mesh distance

**Key parameters:**
- `P`: Number of PEs per dimension (creates P×P grid)
- `group_num`: Number of PE groups (trades routing resources vs latency)
- `pe_num_p_group`: PEs per group = P / group_num
- `root_1st_phase`, `root_2nd_phase`: Root PEs for each reduction phase

### PE Grid and Data Distribution

**PE Grid:**
- P×P Processing Elements arranged in 2D mesh
- Each PE has coordinates (px, py)
- Fabric dimensions: `(P+7, P+2)` with offsets `(4, 1)`

**Data Sharding:**
- Data is distributed across PEs for parallel processing
- Per-PE dimensions: `dim_p_pe = dim / P`, `seq_len_p_pe = seq_len / P`, etc.
- **Critical**: All dimensions must be divisible by P

**Communication Primitives:**
- **Reduce**: Sum values across PEs using hierarchical pattern
- **Broadcast**: Distribute data from root to all PEs
- **Colors**: Enable parallel routing (multiple colors = parallel communication paths)
- **SIMD**: Uses `simd_64` for vectorized operations

### Configuration Files

Decode and Prefill modules use JSON configuration files:

```json
{
  "P": 64,              // PE grid dimension (creates 64×64 grid)
  "bsz": 1,             // Batch size
  "group_num": 8,       // Number of PE groups for communication
  "dim": 4096,          // Model hidden dimension
  "n_heads": 32,        // Number of attention heads
  "n_kv_heads": 8,      // Number of key-value heads (GQA)
  "head_dim": 128,      // Dimension per head
  "seq_len": 4096,      // Sequence length (Prefill) or max_seq_len (Decode)
  "ffn_dim": 11008      // FFN hidden dimension
}
```

MeshGEMV and MeshGEMM use command-line parameters instead.

## File Types and Tools

- **`.csl`**: Cerebras Software Language source files (similar to C syntax)
- **`.py`**: Python scripts using Cerebras SDK (`SdkCompiler`, `SdkRuntime`)
- **`.sh`**: Shell scripts for workflow automation
- **`.json`**: Configuration files for model parameters
- **`compile_out/`**: Generated directory containing compiled artifacts
- **`simfab_traces/`**: Generated simulator trace files for debugging

**Required tools:**
- `cslc`: CSL compiler (from Cerebras SDK)
- `cs_python`: Python wrapper for Cerebras SDK
- `jq`: JSON parser for shell scripts

## Development Workflow

1. **Modify CSL Code**: Edit `.csl` files in `src/` directory
2. **Update Configuration**: Adjust JSON config or script parameters
3. **Test on Simulator**: Run `bash run_sim.sh <config>` (fast iteration, no hardware required)
4. **Check Output**: Examine `sim_stats.json`, logs, and traces
5. **Deploy to Hardware**: Run `bash run_wse3.sh <config> false` (when ready)

**Best Practices:**
- Always test on simulator before hardware deployment
- Ensure dimensions are divisible by P (validation in launch scripts)
- Monitor memory and routing resource usage
- Use smaller P values (e.g., P=8) for quick testing
- Start with MeshGEMV if learning the system (simplest module)

## Key Concepts for Wafer-Scale Programming

**Parallelism Strategy:**
- WaferLLM implements novel wafer-scale LLM parallelism guided by the PLMR model
- Optimizes utilization of hundreds of thousands of on-chip cores
- Achieves 200× higher accelerator utilization vs. state-of-the-art methods

**Performance Characteristics:**
- MeshGEMV operations: 606× faster and 16× more energy-efficient vs. A100 GPU
- Full LLM inference: 10-20× speedups over A100 GPU clusters (SGLang/vLLM)
- Ultra-high on-chip memory bandwidth (tens of PB/s)

**Memory and Communication:**
- Each PE has local on-chip SRAM (tens of GB total distributed memory)
- Data must be explicitly distributed and collected via communication library
- Routing fabric enables direct PE-to-PE communication
- Hierarchical reduction minimizes routing congestion

**Dimensions and Constraints:**
- All tensor dimensions must be evenly divisible by P
- `group_num` must evenly divide P
- Fabric dimensions: `(P+7, P+2)` for WSE-3, vary by architecture
- Memory per PE is limited; check resource usage in compilation logs

## Debugging and Troubleshooting

**Simulator Output:**
- `sim_stats.json`: Performance statistics and cycle counts
- `sim.log`: Execution logs
- `simfab_traces/`: Detailed trace files for debugging communication patterns

**Common Issues:**
- **Dimension not divisible by P**: Validation error in launch scripts
- **Routing resource exhaustion**: Reduce `group_num` or adjust communication pattern
- **Memory overflow**: Reduce dimensions or increase P
- **Compilation errors**: Check CSL syntax, parameter types, fabric configuration

**Debugging Tools:**
- Set `SINGULARITYENV_SIMFABRIC_DEBUG=router` for router debugging
- Use `debug_util` from `cerebras.sdk.debug.debug_util`
- Check `simfab_traces/` for communication visualization

## Module-Specific Details

**MeshGEMV** (`[1,N] = [1,M] @ [M,N]`):
- Simplest module, good starting point
- Parameters: `P M N group_num`
- Use case: Understand basic wafer-scale GEMV pattern

**MeshGEMM** (`[M,N] = [M,K] @ [K,N]`):
- Matrix-matrix multiplication
- Parameters: `P M K N group_num`
- Use case: Benchmark GEMM performance

**Prefill**:
- Processes initial input sequence in parallel
- Implements transformer layer with attention and FFN
- Configuration via JSON (see model_config/)
- Performs `[seq_len, dim] @ [dim, dim]` operations

**Decode**:
- Token generation phase (autoregressive)
- Most complex module with KV cache management
- Variants: WSE-3-KV (with KV buffer), WSE-3-configurable, WSE-3-demo-*
- Performs `[1, dim] @ [dim, dim]` operations
- Configuration via JSON with additional `bsz` parameter

**Shift/Resize**:
- Supporting operations for data manipulation on mesh
- Simpler modules focused on specific data movement patterns

## Dependencies and SDK Versions

- **Cerebras SDK**: v1.2.0 or v1.4.0 required
  - Download: https://www.cerebras.ai/developers/sdk-request
  - Docs v1.2.0: https://cerebras-sdk-docs-120.netlify.app/
  - Docs v1.4.0: https://sdk.cerebras.net/
- **Cerebras ML Software**: v2.3 (WSE-2), v2.5 (WSE-3)
- **Python**: 3.8 or higher
- **System**: 32GB+ RAM recommended for simulator
- **Hardware Access**: Cerebras WSE-2/3 system for actual deployment

## Additional Resources

- **Paper**: [WaferLLM: Large Language Model Inference at Wafer Scale](https://arxiv.org/abs/2502.04563)
- **REPOSITORY_OVERVIEW.md**: Detailed technical documentation
- **Module READMEs**: Each WSE-*/README.md has module-specific details
- **GEMV_COMPARISON.md**: Performance comparison documentation
- **CSL-document/**: Local CSL language documentation


## SDK docs lookup (must-follow)

The SDK documentation is vendored in this repo under:

- `CSL-document/`  

When answering any question that could be clarified by SDK docs:

1) **Search first (do not guess).**
   - Use ripgrep to locate the most relevant references:
     - `rg -n --hidden --no-ignore-vcs "<query>" PATH/TO/YOUR/SDK/DOCS`
     - If too many hits, narrow with additional terms or anchors (function/type names, error strings).
   - Prefer searching for exact identifiers, error messages, and headings.

2) **Open only the minimal necessary context.**
   - Read the smallest relevant portion of the best-matching file(s) (typically **~50–200 lines** around the hit).
   - Do **not** paste entire files or large sections into the conversation unless explicitly requested.

3) **Cite precisely.**
   - In the final answer, cite the source as:
     - `path:line-line` (and include the nearest heading/title if present).
   - If multiple sources are used, list each citation.

4) **If docs are missing/insufficient, say so explicitly.**
   - State what you searched (keywords) and what you found.
   - Then provide best-effort reasoning and clearly label it as inference.

5) **Preference order**
   - SDK docs > code comments in this repo > general knowledge.

### Notes on doc formats
- Prefer Markdown/text sources when available.
- For HTML, search with `rg` on `.html` and cite file+line.
- For PDF, prefer any accompanying `.md`/`.txt` export; only use PDF excerpts if necessary.

## CSL Practice Notes (auto-capture)

Whenever a session produces a non-obvious finding about CSL — a gotcha, a workaround, a pattern that wasn't obvious from docs, or a confirmed behavior — **write it immediately** to:

```
/home/lexu/WaferLLM/CSL-Demo-Code/CSL-Practice-Notes/YYYY-MM-DD-<topic>.md
```

**Format** (keep it brief):
```markdown
# <topic> — <date>

## Finding
One paragraph: what was discovered and why it matters.

## Example
Minimal code snippet if applicable.

## Source
Where this was confirmed (tutorial, doc, trial-and-error).
```

**Do not wait to be asked.** Write the note as soon as the finding is clear.
At the end of any productive CSL session, also ask: *"Are there findings worth consolidating into AI-Knowledge-Base?"*