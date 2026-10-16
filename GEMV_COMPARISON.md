# GEMV Implementation Comparison: MeshGEMV vs Decode Module

## Executive Summary

Both modules implement GEMV (General Matrix-Vector Multiplication) operations, but serve different purposes:
- **MeshGEMV**: Standalone benchmark/reference implementation
- **Decode**: Integrated GEMV operations within transformer inference pipeline

---

## 1. Core Algorithm Comparison

### MeshGEMV Implementation

**Operation**: `[1, Nt] = [1, Mt] @ [Mt, Nt]`

**Key Function**: `mv_compute()`
```csl
fn mv_compute() void {
    @load_to_dsr(res_dest_dsr, res_dsd, .{ .save_address = false });
    @load_to_dsr(res_src0_dsr, res_dsd, .{ .save_address = false });
    @load_to_dsr(W_src1_dsr, W_dsd, .{ .save_address = true });
    
    @map(gemv_static_step, X_dsd);  // Single vector-matrix multiply
    
    comm_mod.two_tree_allreduce_y(ptr_res);  // Reduction along Y-axis
    meshgemv_entry();
}
```

**Characteristics**:
- Single vector input (`X_tile: [Mt]f16`)
- Single matrix weight (`W_tile: [Mt*Nt]f16`)
- Single output vector (`res_tile: [Nt]f16`)
- One GEMV operation per call

### Decode Implementation

**Operation**: `[bsz, Nt] = [bsz, Kt] @ [Kt, Nt]` (batch processing)

**Key Function**: `vecmat_computation()`
```csl
fn vecmat_computation() void {
    left_vector_dsd = @set_dsd_length(left_vector_dsd, @bitcast(u16, Kt));
    right_matrix_dsd = @set_dsd_length(right_matrix_dsd, @bitcast(u16, Nt));
    out_vector_dsd = @set_dsd_length(out_vector_dsd, @bitcast(u16, Nt));
    
    for (@range(i16, bsz)) |b| {  // Batch loop
        right_matrix_dsd = @set_dsd_base_addr(right_matrix_dsd, ptr_right_matrix);
        
        @load_to_dsr(dest_dsr_1, out_vector_dsd, .{ .save_address = false });
        @load_to_dsr(src0_dsr_1, out_vector_dsd, .{ .save_address = false });
        @load_to_dsr(src1_dsr_1, right_matrix_dsd, .{ .save_address = true });
        
        @map(gemv_static_step, left_vector_dsd);  // MFU 2x4Kx4K-flops
        
        left_vector_dsd = @increment_dsd_offset(left_vector_dsd, Kt, f16);
        out_vector_dsd = @increment_dsd_offset(out_vector_dsd, Nt, f16);
    }
}
```

**Characteristics**:
- Batch processing: loops over `bsz` (batch size)
- Dynamic dimensions: `Kt` and `Nt` set per operation
- Flexible pointers: `ptr_left_vector`, `ptr_right_matrix`, `ptr_out_vector` can point to different buffers
- No reduction inside function (reduction called separately after)

---

## 2. Data Structures

### MeshGEMV

```csl
// Fixed-size buffers
var X_tile: [Mt]f16 = @zeros([Mt]f16);           // Input vector
var W_tile: [Mt*Nt]f16 = @zeros([Mt*Nt]f16);     // Weight matrix
var res_tile: [Nt]f16 = @zeros([Nt]f16);         // Output vector

// Fixed DSDs
var X_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Mt} -> X_tile[i] });
var W_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> W_tile[i] });
var res_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{Nt} -> res_tile[i] });
```

**Design**: Simple, fixed-size buffers for benchmarking

### Decode

```csl
// Dynamic pointers (can point to different buffers)
var ptr_left_vector: [*]f16 = &dummy;
var ptr_right_matrix: [*]f16 = &dummy;
var ptr_out_vector: [*]f16 = &dummy;

// Dynamic DSDs (lengths set at runtime)
var left_vector_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &dummy, .extent = 1 });
var right_matrix_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &dummy, .extent = 1 });
var out_vector_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &dummy, .extent = 1 });

// Runtime dimensions
var Kt: i16 = 0;  // Set before each call
var Nt: i16 = 0;  // Set before each call
```

**Design**: Flexible, reusable function that works with different buffers and dimensions

---

## 3. Communication Patterns

### MeshGEMV

**Reduction**: `two_tree_allreduce_y(ptr_res)`
- Single reduction call
- Always reduces along Y-axis
- Two-phase hierarchical reduction:
  1. Phase 1: Reduce within PE groups
  2. Phase 2: Reduce across groups
- Fixed to `Nt` elements

**Implementation**:
```csl
fn two_tree_allreduce_y(vector_buf_ptr: [*]f16) void {
    // Phase 1: Intra-group reduction
    // Phase 2: Inter-group reduction
    // Broadcast back to all PEs
}
```

### Decode

**Multiple Reduction Functions** (context-dependent):
- `all_reduce_bsz()`: For `[bsz]` vectors (e.g., normalization sums)
- `all_reduce_bsz_dim()`: For `[bsz * dim_p_pe]` vectors (QKV projections, output)
- `all_reduce_bsz_seq_len()`: For `[bsz * seq_len_p_pe]` vectors (attention scores)
- `all_reduce_bsz_ffn_dim()`: For `[bsz * ffn_dim_p_pe]` vectors (FFN layers)
- `all_reduce_bsz_dim_QKV_fusion()`: Fused reduction for QKV
- `all_reduce_bsz_ffn_dim_ZZ_fusion()`: Fused reduction for FFN

**Reduction Called Separately**:
```csl
// Example: Q projection
vecmat_computation();  // Compute locally
// comm_mod.all_reduce_bsz_dim(py, quotient_y, remainder_y, ptr_Q);  // Commented out

// Example: Attention scores
vecmat_computation();  // Compute locally
comm_mod.all_reduce_bsz_seq_len(px, quotient_x, remainder_x, ptr_score);  // Reduce along X-axis
```

**Key Differences**:
- Reduction axis varies (X or Y) depending on operation
- Different reduction sizes for different tensor shapes
- Some reductions are fused for multiple operations
- Reduction can be skipped (commented out) for certain operations

---

## 4. Usage Context

### MeshGEMV

**Purpose**: Benchmarking and algorithm validation
- Standalone execution
- Repeatable timing measurements
- Clean separation of computation and communication
- Simple input/output interface

**Execution Flow**:
```csl
meshgemv_entry() → mv_compute() → two_tree_allreduce_y() → repeat
```

### Decode

**Purpose**: Transformer inference operations
- Integrated into larger computation graph
- Used for multiple operations:
  1. **QKV Projections**: `xq_matvec_mult()`, `xk_matvec_mult()`, `xv_matvec_mult()`
  2. **Attention Scores**: `score_matvec_mult()`
  3. **Output Projection**: `output_matvec_mult()`
  4. **FFN Layers**: `h1_matvec_mult()`, `z1_matvec_mult()`, `z2_matvec_mult()`, `h2_matvec_mult()`

**Execution Flow** (simplified):
```csl
decode_host() 
  → RMSNorm
  → xq_matvec_mult() → vecmat_computation() → RoPE
  → xk_matvec_mult() → vecmat_computation() → RoPE
  → xv_matvec_mult() → vecmat_computation()
  → score_matvec_mult() → vecmat_computation() → all_reduce_bsz_seq_len() → softmax
  → output_matvec_mult() → vecmat_computation() → all_reduce_bsz_dim()
  → FFN layers (multiple vecmat_computation calls)
```

---

## 5. Performance Optimizations

### MeshGEMV

**Optimizations**:
- Simple, focused implementation
- Single reduction path
- Fixed-size buffers (compile-time optimization)
- Clean timing separation (warmup + repeat)

### Decode

**Optimizations**:
- **Batch Processing**: Processes multiple vectors in one call
- **Fused Operations**: Some reductions are fused (QKV fusion, ZZ fusion)
- **Axis Reconfiguration**: Can switch reduction axis (X/Y) dynamically
- **Selective Reduction**: Some operations skip reduction (commented out) when not needed
- **Memory Reuse**: Same function works with different buffers

**Example - QKV Fusion**:
```csl
// Compute Q, K, V separately but reduce together
xq_matvec_mult();  // vecmat_computation() - no reduction
xk_matvec_mult();  // vecmat_computation() - no reduction  
xv_matvec_mult();  // vecmat_computation() - no reduction
comm_mod.all_reduce_bsz_dim_QKV_fusion(...);  // Fused reduction
```

---

## 6. Code Complexity

### MeshGEMV

**Lines of Code**: ~165 lines (meshgemv.csl)
- Simple, focused
- Single responsibility
- Easy to understand and modify

### Decode

**Lines of Code**: ~921 lines (decode.csl)
- Complex transformer logic
- Multiple GEMV operations
- Integrated with attention, normalization, etc.
- Context-dependent behavior

---

## 7. Key Differences Summary

| Aspect | MeshGEMV | Decode |
|--------|----------|--------|
| **Purpose** | Benchmarking | Transformer inference |
| **Batch Support** | No (single vector) | Yes (bsz loop) |
| **Dimensions** | Fixed (Mt, Nt) | Dynamic (Kt, Nt) |
| **Reduction** | Single function (`two_tree_allreduce_y`) | Multiple functions (context-dependent) |
| **Reduction Axis** | Always Y-axis | X or Y (configurable) |
| **Reduction Size** | Fixed (Nt) | Variable (bsz, dim, seq_len, ffn_dim) |
| **Integration** | Standalone | Integrated with transformer ops |
| **Complexity** | Simple (~165 lines) | Complex (~921 lines) |
| **Flexibility** | Low (fixed buffers) | High (dynamic pointers) |
| **Optimization** | Clean timing | Fused ops, selective reduction |

---

## 8. When to Use Which?

### Use MeshGEMV When:
- Benchmarking GEMV performance
- Understanding wafer-scale GEMV algorithms
- Testing communication patterns
- Validating reduction implementations
- Simple, isolated GEMV operations

### Use Decode's `vecmat_computation()` When:
- Implementing transformer inference
- Need batch processing
- Working with variable dimensions
- Requiring flexible buffer management
- Integrating with other transformer operations

---

## 9. Algorithmic Similarities

Despite differences, both share core algorithmic patterns:

1. **Same Core Operation**: Both use `@map(gemv_static_step, ...)` with `@fmach()` instruction
2. **Same Communication Library**: Both use `comm_lib` (though different functions)
3. **Same Reduction Pattern**: Both use two-phase hierarchical reduction
4. **Same PE Grid**: Both operate on P×P PE grid

The main difference is **integration level**: MeshGEMV is a clean, isolated implementation, while Decode's GEMV is optimized for transformer workloads.

---

## 10. Code Examples

### MeshGEMV: Simple GEMV
```csl
// Single operation
@map(gemv_static_step, X_dsd);  // [1, Mt] @ [Mt, Nt]
comm_mod.two_tree_allreduce_y(ptr_res);  // Reduce [Nt]
```

### Decode: Batch GEMV with Context
```csl
// Batch operation with dynamic setup
Kt = dim_p_pe;
Nt = dim_p_pe;
ptr_left_vector = ptr_X_norm;
ptr_right_matrix = &Q_weight_tile;
ptr_out_vector = ptr_QKV_tile;
vecmat_computation();  // Processes bsz vectors
// Reduction called separately if needed
comm_mod.all_reduce_bsz_dim(...);
```

---

## Conclusion

**MeshGEMV** provides a clean, benchmarkable reference implementation of wafer-scale GEMV operations. **Decode's `vecmat_computation()`** adapts this pattern for transformer inference with batch processing, flexible dimensions, and context-aware reductions. Both are valuable: MeshGEMV for understanding and benchmarking, Decode's implementation for production transformer workloads.




