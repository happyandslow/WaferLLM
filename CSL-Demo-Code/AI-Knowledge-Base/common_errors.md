# CSL Common Errors Reference

> **Purpose**: Quick lookup for CSL compiler errors and runtime issues. Format: Error → Cause → Fix → Example.
> **SDK Version**: 1.4.0

---

## Compilation Errors

### 1. Queue not initialized (WSE-3)

**Error**: `error: queue N has not been initialized`

**Cause**: On WSE-3, every input/output queue must be explicitly initialized via `@initialize_queue` in a `comptime` block.

**Fix**:
```csl
comptime {
    @initialize_queue(my_iq, .{ .color = my_color });
    @initialize_queue(my_oq, .{ .color = my_color });
}
```

---

### 2. Two fabric inputs in DSD operation (WSE-3)

**Error**: `error: DSD operation cannot have more than one fabric input`

**Cause**: WSE-3 hardware does not support two `fabin_dsd` sources in a single DSD operation.

**Fix**: Buffer one input through a FIFO:
```csl
var buf: [size]u16;
const fifo = @allocate_fifo(buf);
@mov16(fifo, fabin_dsd_2, .{ .async = true });       // Buffer second input
@add16(output, fabin_dsd_1, fifo, .{ .async = true }); // Use FIFO + fabric
```

---

### 3. DSD op setting must be comptime

**Error**: `error: DSD op setting must be comptime`

**Cause**: The `.ut_id` or other DSD operation settings require comptime-known values.

**Fix**: Use `const` for ut_id values, not `var`:
```csl
const out_ut: ut_id = @get_ut_id(3);  // Must be const (comptime)
```

---

### 4. Type mismatch in DSD operation

**Error**: `error: expected type 'f32', found 'i16'`

**Cause**: DSD operation type (e.g., `@fmovs` for f32) doesn't match the DSD's element type.

**Fix**: Match the DSD operation suffix to the data type:
- `@fmovs` / `@fadds` / `@fmacs` → f32
- `@fmovh` / `@faddh` / `@fmach` → f16
- `@mov16` / `@add16` → i16/u16
- `@mov32` → i32/u32

---

### 5. Task ID collision

**Error**: `error: task ID N is already bound`

**Cause**: Two tasks bound to the same task ID.

**Fix**: Use different IDs. On WSE-3, consider `@set_control_task_table()` to separate control/local namespaces:
```csl
comptime {
    @set_control_task_table(.{ .instructions = 8, .stride = 4 });
}
```

---

### 6. Cannot use reserved task/queue/color

**Error**: `error: resource N is reserved by memcpy`

**Cause**: Using a color, queue, or task ID reserved by the memcpy library.

**Fix**: Avoid these resources:
- Colors: 21 (D2H), 22 (cmd), 23 (H2D)
- Local task IDs: 27, 28, 30
- Control task IDs: 33-37
- IQ 0 (and IQ 1 on WSE-3), OQ 0
- Start user queues at IQ 2 / OQ 2

---

### 7. @export_symbol for non-pointer array

**Error**: `error: cannot export array type directly`

**Cause**: Trying to export an array directly instead of a pointer to it.

**Fix**:
```csl
var data = @zeros([10]f32);
var ptr_data: [*]f32 = &data;       // Create pointer
comptime {
    @export_symbol(ptr_data, "data"); // Export pointer, not array
}
```

---

### 8. Comptime assert failure

**Error**: `error: comptime assertion failed`

**Cause**: A `@comptime_assert` condition evaluated to false during compilation.

**Fix**: Check the assertion's condition. Common causes:
- Wrong type passed to generic function
- Invalid parameter value
- Architecture mismatch

---

## Runtime Errors / Hangs

### 9. Program hangs after launch

**Cause**: Missing `sys_mod.unblock_cmd_stream()` in the exit path.

**Fix**: Ensure every execution path ends with:
```csl
sys_mod.unblock_cmd_stream();
```

For state machines, ensure the EXIT state calls `unblock_cmd_stream()`.

---

### 10. Host hangs on memcpy_d2h

**Cause**: D2H transfer size mismatch between host and device, or forgetting `nonblock=False` on a prior launch.

**Fix**:
```python
# Ensure launch completes before memcpy
runner.launch("compute", nonblock=False)
# Ensure size matches device data
runner.memcpy_d2h(result, data_symbol, 0, 0, width, height, size_per_pe)
```

---

### 11. Data corruption / wrong results

**Cause**: Multiple possibilities:
1. DSD extent mismatch (reading/writing more/fewer elements than expected)
2. Missing `.async = true` when operations should overlap
3. Race condition between async operations

**Fix**: Verify DSD extents match data sizes. Use FIFO or block/unblock for ordering.

---

### 12. Stall in neighbor exchange

**Cause**: Boundary PEs not counting skipped send/receive operations.

**Fix**: Edge PEs must still increment the completion counter for non-existent neighbors:
```csl
if (last_px) {
    count_send_recv += 1;  // Skip send east, but still count
} else {
    @fmovs(fabout_east, data, .{ .async = true, .activate = f_send });
}
```

---

### 13. Incorrect wavelet data (all zeros)

**Cause**: Sending data before it's ready, or wrong routing configuration.

**Fix**:
1. Check `@set_color_config` routes: sender tx → receiver rx must match
2. Verify color is the same on both sides
3. Ensure data is populated before the send DSD operation starts

---

### 14. State machine stuck in loop

**Cause**: `next_state` not set before calling an async function. The callback fires and reads stale `next_state`.

**Fix**: **Always set `next_state` BEFORE calling the async function:**
```csl
// CORRECT:
next_state = STATE_REDUCE;    // Set first
f_compute();                   // Then call async

// WRONG:
f_compute();                   // Callback may fire here!
next_state = STATE_REDUCE;     // Too late — callback already read old value
```

---

## Simulator-Specific Issues

### 15. simprint produces no output

**Cause**: Missing `\n` newline character. `<simprint>` buffers output until newline.

**Fix**:
```csl
prt.print_string("value = ");
prt.print_f32(x);
prt.print_string("\n");  // Required to flush
```

---

### 16. sim.log not generated

**Cause**: Missing environment variable.

**Fix**:
```bash
SINGULARITYENV_SIMFABRIC_DEBUG=landing   # For wavelet landing logs
SINGULARITYENV_SIMFABRIC_DEBUG=inst_trace # For instruction traces
SINGULARITYENV_SIMFABRIC_DEBUG=router     # For router state
```

---

## Related Documents

- **Full debugging guide**: `CSL_Debugging_Guide.md`
- **WSE-3 migration issues**: `CSL_WSE2_vs_WSE3_Migration_Guide.md`
- **Queue/FIFO details**: `CSL_Queue_FIFO_Guide.md`
- **All builtins**: `CSL_Builtins_Reference_Guide.md`
