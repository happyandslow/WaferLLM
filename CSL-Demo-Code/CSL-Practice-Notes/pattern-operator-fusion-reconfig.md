# Operator Fusion via Runtime Reconfiguration — 2026-03-17

**Date**: 2026-03-17
**Context**: Studying the gemm_gemv fused operator in WaferServe/fused_ops/ to establish patterns for future fusion work
**Category**: pattern

## Problem / Observation

Running GEMM then GEMV as separate compiled artifacts requires two `SdkRuntime` instances and a host↔device round-trip between stages (device → host → device). This wastes bandwidth and adds latency.

## Solution / Insight

Fuse both operators into a single compiled artifact. The three core techniques:

### 1. Shared On-PE Buffer (`shared_tile`)

Declare a buffer in the tile code that connects stages. Pass its address as a comptime param to both operator modules:

```csl
// tile code (gemm_gemv.csl)
var shared_tile: [Mt*Nt]f16 = @zeros([Mt*Nt]f16);

const gemm_mod = @import_module("meshgemm.csl", .{
    ..., .ptr_shared = &shared_tile, ...
});
const gemv_mod = @import_module("meshgemv.csl", .{
    ..., .ptr_shared = &shared_tile, ...
});
```

Op1 writes to `shared_tile` via DSDs; op2 reads from it. No host involvement.

### 2. Color Reuse + `reconfig()`

Colors are scarce. Instead of allocating unique colors for each operator, time-multiplex them. GEMM uses colors 5-9, 12-16 for X/Y communication; GEMV reuses those same IDs for up/down reduce and broadcast.

After GEMM completes, `reconfig()` reprograms the shared colors:

```csl
const tile_config = @import_module("<tile_config>");
const color_config = tile_config.color_config;
const input_queue_config = tile_config.input_queue_config;
const output_queue_config = tile_config.output_queue_config;

fn reconfig() void {
    // Reset color routes for GEMV patterns
    color_config.reset_routes(C_U_B_color, C_U_B_routes);
    color_config.reset_routes(C_D_B_color, C_D_B_routes);
    color_config.reset_routes(comm_params_gemv.c_down_send,
        .{ .rx = RAMP, .tx = [1]direction{SOUTH} });
    color_config.reset_routes(comm_params_gemv.c_up_recv,
        .{ .rx = SOUTH, .tx = [1]direction{RAMP} });
    // ... more route resets for reduce colors ...

    // Rebind queues to GEMV colors
    input_queue_config.encode_input_queue(
        @get_int(queue_id), @get_int(gemv_color), true);
    output_queue_config.encode_output_queue(
        @get_int(queue_id), @get_int(gemv_color));
    // ... 12 queue rebindings total ...

    sys_mod.unblock_cmd_stream();
}
```

**Critical**: GEMV colors must NOT be configured via `@set_color_config` in layout.csl — that would conflict with GEMM's compile-time routing. Only GEMM routes are set in layout; GEMV routes are set at runtime.

### 3. Shared Module Instantiation

`<memcpy/memcpy>` and `<layout>` must be instantiated exactly once. The tile code owns them and passes as `imported_module` params:

```csl
// Tile code instantiates once
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
const layout_module = @import_module("<layout>");

// Each operator module receives them
// meshgemm.csl / meshgemv.csl:
param sys_mod: imported_module;
param layout_module: imported_module;
```

## Queue Architecture Detail

GEMM queues (2-5) are compile-time bound via `@initialize_queue` in `gemm_comm_pe.csl`. GEMV queues (2-7) are runtime-bound via `encode_*_queue` in `reconfig()`, because they share queue slots 2-5 with GEMM and add slots 6-7 for broadcast.

The GEMV comm module exposes configurable queue IDs:

```csl
fn get_configurable_queues() comptime_struct {
    return .{
        .reduce_1st_up_recv_queue_id = reduce_1st_up_recv_queue_id,
        .reduce_1st_up_send_queue_id = reduce_1st_up_send_queue_id,
        // ... 12 total queue IDs ...
    };
}
```

## Host-Side Execution Sequence

```python
runner = SdkRuntime("out")
runner.load(); runner.run()

# Transfer ALL inputs at once (GEMM X, W + GEMV V)
runner.memcpy_h2d(sym_X, X_u32, ...)
runner.memcpy_h2d(sym_W, W_u32, ...)
runner.memcpy_h2d(sym_V, V_u32, ...)

# Run pipeline
runner.launch('init_task', nonblock=False)
runner.launch('mesh_gemm_once', nonblock=False)
# Optional: readback shared_tile to verify GEMM
runner.launch('reconfig', nonblock=False)
runner.launch('mesh_gemv_once', nonblock=False)
runner.memcpy_d2h(res_u32, sym_res, ...)
runner.stop()

# Or use the fused host function for production:
# runner.launch('mesh_gemm_gemv_host', warmup, repeat, nonblock=False)
```

## Related Resources

- Full guide: `WaferServe/fused_ops/OPERATOR_FUSION_GUIDE.md`
- Reference impl: `WaferServe/fused_ops/gemm_gemv/`
- `<tile_config>` docs for `color_config`, `input_queue_config`, `output_queue_config`
