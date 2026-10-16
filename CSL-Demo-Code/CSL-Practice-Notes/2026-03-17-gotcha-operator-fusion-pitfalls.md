# Operator Fusion Gotchas — 2026-03-17

**Date**: 2026-03-17
**Context**: Lessons from gemm_gemv fusion in WaferServe/fused_ops/
**Category**: gotcha

## Problem / Observation

Several non-obvious failure modes when fusing multiple CSL operators into a single artifact with runtime reconfiguration.

## Gotcha 1: Missing `sys_mod.unblock_cmd_stream()`

Every host-launchable function (`init_task`, `mesh_gemm_once`, `reconfig`, `mesh_gemv_once`) must end with `sys_mod.unblock_cmd_stream()`. Missing it causes the host command stream to hang indefinitely — no error, just a silent deadlock.

```csl
fn reconfig() void {
    // ... reconfigure colors and queues ...
    sys_mod.unblock_cmd_stream();  // MUST be last
}
```

## Gotcha 2: Double-instantiating `<memcpy/memcpy>` or `<layout>`

If both `meshgemm.csl` and `meshgemv.csl` independently call `@import_module("<memcpy/memcpy>", ...)`, you get linker errors (duplicate symbol definitions). Instantiate once in the tile code, pass as `imported_module` param.

```csl
// WRONG — each module instantiates its own
// meshgemm.csl:
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);  // ❌

// RIGHT — tile code owns it, modules receive it
// meshgemm.csl:
param sys_mod: imported_module;  // ✅
```

## Gotcha 3: `@initialize_queue` for shared queues

If op1 uses `@initialize_queue(queue_2, .{.color = gemm_x_recv})` at compile time, op2 cannot also `@initialize_queue(queue_2, .{.color = gemv_up_recv})` — it's a compile-time conflict. Op2 must use runtime binding:

```csl
// In reconfig():
input_queue_config.encode_input_queue(
    @get_int(queue_id), @get_int(new_color), true);
```

## Gotcha 4: Unblocked recv queues consume stale data

If a queue that will be runtime-bound for op2 is left unblocked during op1, it may consume stray fabric data. Always `@block(queue_id)` in comptime for queues that will be runtime-bound later:

```csl
comptime {
    @block(reduce_1st_down_recv_queue_id);  // Blocked until reconfig()
    @block(reduce_1st_up_recv_queue_id);
}
```

## Gotcha 5: GEMV color routes in layout vs reconfig

Setting GEMV routes via `@set_color_config(px, py, gemv_color, ...)` in `layout.csl` conflicts with GEMM routes on the same color IDs. GEMV routes must be set at runtime in `reconfig()` using `color_config.reset_routes()`. The commented-out `@set_color_config` blocks in the gemm_gemv layout.csl are there as documentation of what the routes should be — they must remain commented out.

## Related Resources

- `WaferServe/fused_ops/gemm_gemv/src/gemm_gemv.csl` — reconfig() implementation
- `WaferServe/fused_ops/gemm_gemv/src/comm_lib/gemv_comm_pe.csl` — blocked queues in comptime
- `WaferServe/fused_ops/gemm_gemv/src/layout.csl` — commented-out GEMV color configs
