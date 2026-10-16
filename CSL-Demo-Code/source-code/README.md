# Cerebras SDK Source Code Extraction

This directory contains source code extracted from the Cerebras SDK container image `sdk-cbcore-202505230211-4-9382352f.sif`.

## Directory Structure

### 1. **python_sdk/** (13M)
- **Source**: `/cbcore/py_root/cerebras/`
- **Description**: Complete Python SDK source code for Cerebras systems
- **Key Modules**:
  - `common/` - Common utilities and shared code
  - `das/` - Data Access Service
  - `elf/` - ELF file handling
  - `framework/` - Framework components
  - `sdk/` - Main SDK components
  - `hal/` - Hardware Abstraction Layer
  - `hwdebug/` - Hardware debugging utilities
  - `plangen/` - Plan generation
  - `stack/` - Stack management
  - `util/` - Utility functions
  - `ws/` - Workspace management

### 2. **csl_libraries/** (952K)
- **Source**: `/cb/toolchains/cslang/rel-sdk-1.4.0/202505212226-1442-a23e251d/csl-libs/`
- **Description**: CSL (Cerebras Software Language) libraries and runtime
- **Key Components**:
  - `memcpy/` - Memory copy operations
  - `kernels/` - Kernel implementations
  - `math/` - Mathematical functions
  - `runtime/` - Runtime system
  - `collectives_2d/` - 2D collective operations
  - `debug/` - Debug utilities
  - `message_passing/` - Message passing system
  - `tile_config/` - Tile configuration
  - Various `.csl` files (math.csl, control.csl, etc.)

### 3. **csl_runtime/** (12K)
- **Source**: `/cb/toolchains/cslang/rel-sdk-1.4.0/202505212226-1442-a23e251d/csl-rt/`
- **Description**: CSL runtime components

### 4. **llvm_toolchain/** (228M)
- **Source**: `/cb/toolchains/llvm/cerebras/202504211254-5201-6f1a5455/`
- **Description**: LLVM-based compiler toolchain for Cerebras
- **Contains**: Compiled binaries and libraries for the LLVM compiler

### 5. **cbcore_bin/** (404K)
- **Source**: `/cbcore/bin/`
- **Description**: Core binaries and executables

### 6. **cbcore_lib/** (41M)
- **Source**: `/cbcore/lib/`
- **Description**: Core libraries and runtime components

### 7. **numpy_f2py_src/** (48K)
- **Source**: `/cb/toolchains/buildroot/.../numpy/f2py/src/`
- **Description**: NumPy F2PY source code for Fortran interoperability

### 8. **scipy_linalg_src/** (60K)
- **Source**: `/cb/toolchains/buildroot/.../scipy/linalg/src/`
- **Description**: SciPy linear algebra source code

### 9. **osqp_sources/** (268K)
- **Source**: `/cb/toolchains/buildroot/.../osqp/codegen/sources/`
- **Description**: OSQP optimization library source code

## Total Size
Approximately **283M** of source code extracted

## Extraction Date
$(date)

## Original Container
`sdk-cbcore-202505230211-4-9382352f.sif`

## Notes
- All source code maintains original directory structure and permissions
- Python SDK contains the main high-level APIs
- CSL libraries contain low-level hardware-specific code
- LLVM toolchain provides compilation infrastructure
- Additional libraries (NumPy, SciPy, OSQP) are included for reference 