# WaferLLM Repository Overview

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dependencies](#dependencies)
4. [Module Organization](#module-organization)
5. [Communication Library](#communication-library)
6. [Usage Workflow](#usage-workflow)
7. [Key Concepts](#key-concepts)

---

## Project Overview

**WaferLLM** is a wafer-scale LLM inference system designed for Cerebras WSE-2/3 accelerators. It implements optimized parallel algorithms for transformer model inference, including:

- **MeshGEMV**: Matrix-vector multiplication `[1,N] = [1,M] @ [M,N]`
- **MeshGEMM**: Matrix-matrix multiplication
- **Prefill**: Initial sequence processing phase
- **Decode**: Token generation phase
- **Shift/Resize**: Supporting operations for data manipulation

The system leverages wafer-scale architectures with hundreds of thousands of cores in a mesh topology, achieving significant performance improvements over GPU-based systems.

---

## Repository Structure

```
WaferLLM/
├── README.md                    # Main project documentation
├── LICENSE                      # Apache License 2.0
├── .gitignore                   # Git ignore patterns
│
├── MeshGEMV/                    # Matrix-vector multiplication module
│   ├── WSE-2/                   # WSE-2 implementation
│   └── WSE-3/                   # WSE-3 implementation
│
├── MeshGEMM/                    # Matrix-matrix multiplication module
│   ├── WSE-2/
│   └── WSE-3/
│
├── Prefill/                     # Prefill phase implementation
│   └── WSE-2/
│
├── Decode/                      # Decode phase implementation
│   ├── WSE-2/
│   └── WSE-3/
│
├── Shift/                       # Data shifting operations
│   └── WSE-3/
│
└── Resize/                      # Data resizing operations
    └── WSE-3/
```

### Standard Module Structure

Each module follows a consistent structure:

```
<module_name>/
└── WSE-<2|3>/
    ├── compile.py               # Compilation script (uses Cerebras SDK)
    ├── launch_sim.py            # Simulator launcher
    ├── launch_wse<2|3>.py       # Hardware launcher
    ├── run_sim.sh               # Simulator execution script
    ├── run_wse<2|3>.sh          # Hardware execution script
    ├── benchmark.sh             # Benchmarking script (if available)
    ├── README.md                # Module-specific documentation
    ├── model_config/            # JSON configuration files (Decode/Prefill)
    ├── compile_out/             # Compiled output directory
    └── src/
        ├── layout.csl           # Main layout file (PE grid configuration)
        ├── <module>.csl         # Module implementation
        └── comm_lib/            # Communication library
            ├── comm_layout.csl  # Communication topology/routing
            └── comm_pe.csl     # PE-level communication primitives
```

---

## Dependencies

### External Dependencies

1. **Cerebras SDK**
   - **Required**: Cerebras SDK v1.2.0 or v1.4.0
   - **Download**: https://www.cerebras.ai/developers/sdk-request
   - **Documentation**:
     - SDK v1.2.0: https://cerebras-sdk-docs-120.netlify.app/
     - SDK v1.4.0: https://sdk.cerebras.net/

2. **Cerebras ML Software**
   - WSE-2 modules: Version 2.3
   - WSE-3 modules: Version 2.5

3. **Python**
   - Python 3.8 or higher
   - Required packages (installed via Cerebras SDK):
     - `cerebras.sdk.client` (SdkCompiler)
     - `cerebras.sdk.sdk_utils` (input_array_to_u32, memcpy_view, calculate_cycles)
     - `cerebras.sdk.runtime.sdkruntimepybind` (SdkRuntime)
     - `numpy`

4. **System Requirements**
   - Access to Cerebras WSE-2/3 system OR Cerebras SDK simulator
   - 32GB+ RAM recommended for simulator
   - `jq` command-line tool (for parsing JSON configs in shell scripts)

### Language & Tools

- **CSL (Cerebras Software Language)**: Domain-specific language for wafer-scale programming
- **cslc**: CSL compiler (provided by Cerebras SDK)
- **cs_python**: Python wrapper for Cerebras SDK tools

---

## Module Organization

### 1. MeshGEMV

**Purpose**: General matrix-vector multiplication `[1,N] = [1,M] @ [M,N]`

**Parameters**:
- `P`: Number of PEs per dimension (creates P×P grid)
- `M`: Input vector length (shared dimension)
- `N`: Output vector length (matrix columns)
- `group_num`: Number of PE groups for parallel execution

**Usage**:
```bash
# Simulator
bash ./run_sim.sh 64 1024 1024 8

# Hardware
bash ./run_wse3.sh 64 1024 1024 8 true   # Simulator
bash ./run_wse3.sh 64 1024 1024 8 false  # Real device
```

### 2. MeshGEMM

**Purpose**: Matrix-matrix multiplication `[M,N] = [M,K] @ [K,N]`

**Parameters**:
- `P`: Number of PEs per dimension (creates P×P grid)
- `M`: First matrix rows
- `K`: Shared dimension
- `N`: Second matrix columns
- `group_num`: Number of PE groups for parallel execution

**Structure**: Similar to MeshGEMV but for matrix-matrix operations

### 3. Prefill

**Purpose**: Processes initial input sequence in parallel

**Configuration** (via JSON):
- `P`: PE grid dimension
- `dim`: Model hidden dimension
- `n_heads`: Attention heads
- `n_kv_heads`: Key-value heads (GQA)
- `head_dim`: Dimension per head
- `seq_len`: Sequence length
- `ffn_dim`: FFN hidden dimension

**Usage**:
```bash
bash ./run_sim.sh model_config/test.json
bash ./run_device.sh model_config/test.json
```

### 4. Decode

**Purpose**: Token generation phase (autoregressive)

**Configuration** (via JSON):
- All Prefill parameters plus:
- `bsz`: Batch size
- `group_num`: PE groups for parallel execution

**Usage**:
```bash
bash ./run_sim.sh model_config/test.json
bash ./run_wse3.sh model_config/test.json true   # Simulator
bash ./run_wse3.sh model_config/test.json false  # Real device
```

### 5. Shift & Resize

**Purpose**: Supporting operations for data manipulation on wafer-scale mesh

**Structure**: Simpler modules focused on specific data movement patterns

---

## Module Relationships

### Relationship Between MeshGEMV/MeshGEMM and Other Modules

**Important**: MeshGEMV and MeshGEMM are **standalone reference implementations** that are **NOT directly imported** by Decode, Prefill, or other modules. Instead, they serve as:

1. **Algorithmic Building Blocks**: They demonstrate the core wafer-scale algorithms for matrix operations
2. **Benchmarking Tools**: Standalone modules for performance evaluation and comparison
3. **Conceptual Foundation**: Decode and Prefill implement similar operations inline, optimized for their specific transformer workloads

**How They're Related**:

- **Decode Module**: 
  - Implements `vecmat_computation()` which performs GEMV operations (similar to MeshGEMV)
  - Uses GEMV for: QKV projections, output projection, FFN layers
  - Pattern: `[1, dim] @ [dim, dim]` → `[1, dim]` (similar to MeshGEMV's `[1,M] @ [M,N]`)

- **Prefill Module**:
  - Implements `matmul_compute()` and `matmul_T_compute()` for GEMM operations (similar to MeshGEMM)
  - Uses GEMM for: QKV projections, attention score computation, output projection, FFN layers
  - Pattern: `[seq_len, dim] @ [dim, dim]` → `[seq_len, dim]` (similar to MeshGEMM's `[M,K] @ [K,N]`)

- **Shared Communication Library**: All modules use the same `comm_lib` for hierarchical reduction and broadcast operations

**Why Not Direct Dependencies?**:
- Decode/Prefill need operations integrated with transformer-specific logic (attention, normalization, etc.)
- Inline implementations allow for better optimization and fusion opportunities
- Different data layouts and memory access patterns for transformer workloads

**Development Workflow**:
1. Develop and optimize MeshGEMV/MeshGEMM as standalone benchmarks
2. Validate algorithms and communication patterns
3. Adapt patterns into Decode/Prefill with transformer-specific optimizations

## Communication Library

The `comm_lib` directory contains a **shared communication library** used across all modules.

### Components

1. **`comm_layout.csl`**
   - Defines communication topology and routing
   - Implements two-phase reduction pattern:
     - **Phase 1**: Intra-group reduction (within PE groups)
     - **Phase 2**: Inter-group reduction (across groups)
   - Configures broadcast and reduce operations
   - Uses color-coded routing for parallel communication

2. **`comm_pe.csl`**
   - Implements PE-level communication primitives
   - Provides functions for:
     - **Reduce operations**: Two-phase allreduce pattern
     - **Broadcast operations**: Data distribution
   - Handles different data sizes (bsz, bsz×dim_p_pe, etc.)
   - Uses SIMD operations (simd_64) for efficiency

### Communication Pattern

The library implements a **hierarchical reduction**:
- **Groups**: PEs are organized into `group_num` groups
- **Phase 1**: Reduce within each group to a root PE
- **Phase 2**: Reduce across group roots to final root
- **Broadcast**: Distribute results back to all PEs

This pattern optimizes for wafer-scale mesh architectures where:
- On-chip memory bandwidth is extremely high
- Routing resources are limited
- Latency scales with mesh distance

---

## Usage Workflow

### Development Workflow

1. **Write/Modify CSL Code**
   - Edit `.csl` files in `src/` directory
   - Modify `layout.csl` for PE grid configuration
   - Update module-specific `.csl` for algorithm logic

2. **Compile**
   ```bash
   # Using compile.py (Python SDK)
   python compile.py <params...>
   
   # Or using run_sim.sh (calls cslc directly)
   bash run_sim.sh <params...>
   ```

3. **Test on Simulator**
   ```bash
   bash run_sim.sh <config_or_params>
   ```
   - Generates cycle-accurate performance estimates
   - Allows debugging without hardware access
   - Outputs: `sim_stats.json`, `sim.log`, `simfab_traces/`

4. **Deploy to Hardware**
   ```bash
   bash run_wse<2|3>.sh <config_or_params> false
   ```
   - Requires hardware access
   - Uses compiled artifacts from `compile_out/`

### Compilation Process

1. **CSL Compilation** (`cslc`)
   - Compiles CSL code to execution code
   - Parameters:
     - `--arch=wse3` or `wse2`: Target architecture
     - `--fabric-dims`: Fabric dimensions (P+7, P+2)
     - `--fabric-offsets`: Starting offsets (4, 1)
     - `--params`: Runtime parameters
     - `--memcpy`: Enable memory copy operations
     - `--channels`: Number of communication channels

2. **Python SDK Compilation** (`compile.py`)
   - Uses `SdkCompiler` from Cerebras SDK
   - Manages compilation resources (CPU, memory)
   - Generates artifact IDs for hardware deployment

### Execution Process

1. **Simulator Execution** (`launch_sim.py`)
   - Creates `SdkRuntime` instance
   - Loads compiled code
   - Performs host-to-device (H2D) memory copies
   - Launches computation tasks
   - Performs device-to-host (D2H) memory copies
   - Collects performance metrics (cycles, timers)

2. **Hardware Execution** (`launch_wse<2|3>.py`)
   - Similar to simulator but targets real hardware
   - Requires proper authentication and access
   - Uses artifact IDs from compilation

---

## Key Concepts

### PE Grid

- **P×P Grid**: Processing Elements arranged in a 2D mesh
- **Coordinates**: Each PE has (px, py) coordinates
- **Fabric**: Physical chip layout with offsets and dimensions
- **Groups**: PEs organized into groups for hierarchical communication

### Data Distribution

- **Sharding**: Data distributed across PEs
- **Per-PE Dimensions**: `dim_p_pe = dim / P`, `seq_len_p_pe = seq_len / P`, etc.
- **Memory Layout**: Data reshaped for efficient mesh communication

### Communication Primitives

- **Reduce**: Sum values across PEs (two-phase pattern)
- **Broadcast**: Distribute data from root to all PEs
- **Colors**: Used for parallel routing (multiple colors = parallel paths)

### Performance Optimization

- **SIMD**: Uses simd_64 for vectorized operations
- **Grouping**: `group_num` parameter trades routing resources vs latency
- **Memory Bandwidth**: Optimized for high on-chip bandwidth
- **Load Balancing**: Dimensions should be divisible by P

### Configuration Files

JSON configs (Decode/Prefill) specify:
- Model architecture parameters
- PE grid configuration
- Batch and sequence dimensions

Example structure:
```json
{
  "P": 64,
  "bsz": 1,
  "group_num": 8,
  "dim": 4096,
  "n_heads": 32,
  "n_kv_heads": 8,
  "head_dim": 128,
  "seq_len": 4096,
  "ffn_dim": 11008
}
```

---

## File Types

- **`.csl`**: Cerebras Software Language source files
- **`.py`**: Python scripts for compilation and execution
- **`.sh`**: Shell scripts for workflow automation
- **`.json`**: Configuration files for model parameters
- **`compile_out/`**: Compiled artifacts (binary, metadata)
- **`simfab_traces/`**: Simulator trace files for debugging

---

## Best Practices

1. **Start with Simulator**: Always test on simulator before hardware
2. **Check Dimensions**: Ensure dimensions are divisible by P
3. **Monitor Resources**: Watch memory and routing resource usage
4. **Use Config Files**: Use JSON configs for complex parameter sets
5. **Benchmark Properly**: Use warmup runs and multiple repeats for accurate timing
6. **Debug Traces**: Use `simfab_traces/` and debug utilities for troubleshooting

---

## Getting Started

1. **Install Cerebras SDK** (request access from Cerebras)
2. **Choose a Module**: Start with MeshGEMV (simplest) or Decode (most complex)
3. **Run Simulator**: `bash run_sim.sh` with test parameters
4. **Examine Output**: Check `sim_stats.json` and logs
5. **Modify Code**: Experiment with parameters and algorithms
6. **Deploy**: When ready, deploy to hardware

---

## Additional Resources

- **Paper**: [arXiv:2502.04563](https://arxiv.org/abs/2502.04563)
- **Cerebras SDK Docs**: See README.md for links
- **Module READMEs**: Each module has detailed documentation
- **Citation**: See main README.md for BibTeX citation

