# CSL Builtins Reference Guide
## SDK Version: 1.4.0

Comprehensive reference for all CSL builtin functions: DSD operations, task management, queue/color configuration, type system, layout, export, and utility builtins.

---

## Table of Contents

1. [f32 Floating Point Operations](#1-f32-floating-point-operations)
2. [f16 Floating Point Operations](#2-f16-floating-point-operations)
3. [Integer Operations (i16/u16)](#3-integer-operations-i16u16)
4. [32-bit Integer & Move Operations](#4-32-bit-integer--move-operations)
5. [@map (Element-wise User Function)](#5-map-element-wise-user-function)
6. [Task Binding & Control](#6-task-binding--control)
7. [Queue & Color Configuration](#7-queue--color-configuration)
8. [DSD & DSR Builtins](#8-dsd--dsr-builtins)
9. [Export & RPC](#9-export--rpc)
10. [Layout & Rectangle](#10-layout--rectangle)
11. [Type System & Conversion](#11-type-system--conversion)
12. [Array & Tensor](#12-array--tensor)
13. [String & Struct](#13-string--struct)
14. [Range & Iteration](#14-range--iteration)
15. [Assertions & Debug](#15-assertions--debug)
16. [Module & Import](#16-module--import)
17. [PRNG & Utility](#17-prng--utility)
18. [WSE-3 Specific Builtins](#18-wse-3-specific-builtins)
19. [WSE-2 vs WSE-3 Differences](#19-wse-2-vs-wse-3-differences)

---

## 1. f32 Floating Point Operations

All DSD operations accept an optional config struct as the last argument (e.g., `.{ .async = true }`).

### Arithmetic

| Builtin | Signature | Description |
|---|---|---|
| `@fadds` | `(dest, src1, src2)` | f32 addition: dest = src1 + src2 |
| `@fsubs` | `(dest, src1, src2)` | f32 subtraction: dest = src1 - src2 |
| `@fmuls` | `(dest, src1, src2)` | f32 multiplication: dest = src1 * src2 |
| `@fmacs` | `(dest, src1, src2, scalar)` | f32 multiply-accumulate: dest = src1 + src2 * scalar |
| `@fmaxs` | `(dest, src1, src2)` | f32 maximum: dest = max(src1, src2) |

Each supports multiple operand combinations:
```csl
@fadds(dest_dsd, src_dsd1, src_dsd2);      // DSD + DSD
@fadds(dest_dsd, f32_value, src_dsd1);      // scalar + DSD
@fadds(dest_dsd, src_dsd1, f32_value);      // DSD + scalar
@fadds(f32_pointer, f32_value, src_dsd1);   // pointer dest
```

### Movement & Negation

| Builtin | Signature | Description |
|---|---|---|
| `@fmovs` | `(dest, src)` | f32 move |
| `@fnegs` | `(dest, src)` | f32 negate: dest = -src |
| `@fabss` | `(dest, src)` | f32 absolute value |

### Conversion & Normalization

| Builtin | Signature | Description |
|---|---|---|
| `@fs2h` | `(dest, src)` | f32 → f16 conversion |
| `@fs2xp16` | `(dest, src)` | f32 → i16 conversion (round toward zero) |
| `@fnorms` | `(f32_ptr, f32_val)` | f32 normalization |
| `@fscales` | `(f32_ptr, f32_val, i16_val)` | f32 × constant |

### Canonical Example

```csl
// From examples/tutorials/gemv-02-memory-dsds/pe_program.csl
var y_dsd = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = M });
var A_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> A[i * N] });
var b_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{M} -> b[i] });

fn gemv() void {
    for (@range(u16, N)) |i| {
        @fmacs(y_dsd, y_dsd, A_dsd, x[i]);  // y += A[:,i] * x[i]
        A_dsd = @increment_dsd_offset(A_dsd, 1, f32);
    }
    @fadds(y_dsd, y_dsd, b_dsd);  // y += b
}
```

---

## 2. f16 Floating Point Operations

### Arithmetic

| Builtin | Signature | Description |
|---|---|---|
| `@faddh` | `(dest, src1, src2)` | f16 addition |
| `@fsubh` | `(dest, src1, src2)` | f16 subtraction |
| `@fmulh` | `(dest, src1, src2)` | f16 multiplication |
| `@fmach` | `(dest, src1, src2, scalar)` | f16 multiply-accumulate |
| `@fmaxh` | `(dest, src1, src2)` | f16 maximum |

### Mixed Precision (f16 × f32)

| Builtin | Signature | Description |
|---|---|---|
| `@faddhs` | `(dest, src1, src2)` | f16 + f32 mixed addition |
| `@fmachs` | `(dest, src1, src2, scalar)` | f16 multiply-accumulate with f32 |

### Movement & Conversion

| Builtin | Signature | Description |
|---|---|---|
| `@fmovh` | `(dest, src)` | f16 move |
| `@fnegh` | `(dest, src)` | f16 negate |
| `@fabsh` | `(dest, src)` | f16 absolute value |
| `@fh2s` | `(dest, src)` | f16 → f32 conversion |
| `@fh2xp16` | `(dest, src)` | f16 → i16 conversion |
| `@xp162fh` | `(dest, src)` | i16 → f16 conversion |
| `@fnormh` | `(fp16_ptr, fp16_val)` | f16 normalization |
| `@fscaleh` | `(fp16_ptr, fp16_val, i16_val)` | f16 × constant |

### Example

```csl
// From examples/benchmarks/fft-1d-2d/fft.csl
@fmulh(dest_dsd, src_dsd1, src_dsd2);
@faddh(dest_dsd, src_dsd1, src_dsd2);
@fnegh(dest_dsd, src_dsd);
```

---

## 3. Integer Operations (i16/u16)

### Arithmetic

| Builtin | Signature | Description |
|---|---|---|
| `@add16` | `(dest, src1, src2)` | 16-bit add |
| `@addc16` | `(dest, src1, src2)` | 16-bit add with carry |
| `@sub16` | `(dest, src1, src2)` | 16-bit subtract |

### Bitwise

| Builtin | Signature | Description |
|---|---|---|
| `@and16` | `(dest, src1, src2)` | Bitwise AND |
| `@or16` | `(dest, src1, src2)` | Bitwise OR |
| `@xor16` | `(dest, src1, src2)` | Bitwise XOR |

### Shifts

| Builtin | Signature | Description |
|---|---|---|
| `@sll16` | `(dest, src, amount)` | Logical shift left |
| `@slr16` | `(dest, src, amount)` | Logical shift right |
| `@sar16` | `(dest, src, amount)` | Arithmetic shift right |

### Bit Operations

| Builtin | Signature | Description |
|---|---|---|
| `@clz` | `(dest, src)` | Count leading zeros |
| `@ctz` | `(dest, src)` | Count trailing zeros |
| `@popcnt` | `(dest, src)` | Population count |

### Movement

| Builtin | Signature | Description |
|---|---|---|
| `@mov16` | `(dest, src)` | 16-bit move |

### Type Conversion

| Builtin | Signature | Description |
|---|---|---|
| `@xp162fs` | `(dest, src)` | i16 → f32 |
| `@xp162fh` | `(dest, src)` | i16 → f16 |

### Canonical Example

```csl
// From examples/tutorials/pipeline-01-basic/pe_program.csl
const in_dsd = @get_dsd(fabin_dsd, .{
    .extent = size,
    .fabric_color = sys_mod.MEMCPYH2D_1,
    .input_queue = h2d_1_iq
});

const out_dsd = @get_dsd(fabout_dsd, .{
    .extent = size,
    .fabric_color = sys_mod.MEMCPYD2H_1,
    .output_queue = d2h_1_oq
});

task main_task() void {
    @add16(out_dsd, in_dsd, one_dsd, .{ .async = true });
}
```

---

## 4. 32-bit Integer & Move Operations

| Builtin | Signature | Description |
|---|---|---|
| `@mov32` | `(dest, src)` | 32-bit move (i32/u32) |

---

## 5. @map (Element-wise User Function)

### Syntax

```csl
@map(callback, Input...);
@map(callback, Input..., Output);
```

### Rules

- Applies `callback` element-wise across DSD/DSR operands
- Callback parameter types must be "DSD compatible": `i16`, `i32`, `u16`, `u32`, `@fp16()`, `f32`
- `fabout_dsd` not allowed as Input; `fabin_dsd` not allowed as Output
- Terminates when any DSD/DSR is exhausted
- At most one FIFO allowed; FIFO cannot be used with @map

### Canonical Example

```csl
// From examples/tutorials/topic-10-map-builtin/pe_program.csl
const math_lib = @import_module("<math>");

// Single input, single output
@map(math_lib.sqrt_f32, dsdA, dsd_sqrt_diag_A);

// Multi-input with scalars, single output
@map(transformation, dsdA, 2.0, 6.0, dsd_weight, dsdA);

// Reduction with pointer accumulator
@map(reduction, dsdB, &sum[0], &sum[0]);
```

### Callback Signature

```csl
fn transformation(a: f32, low: f32, high: f32, weight: *f32, result: *f32) void {
    if (a >= low and a <= high) {
        weight.* += 1.0;
    }
    result.* = a;
}
```

---

## 6. Task Binding & Control

### Task ID Creation

| Builtin | Returns | Description |
|---|---|---|
| `@get_local_task_id(id)` | `local_task_id` | WSE-2: 8-24; WSE-3: 8-30 |
| `@get_data_task_id(id)` | `data_task_id` | WSE-2: `id` is `color`; WSE-3: `id` is `input_queue` |
| `@get_control_task_id(id)` | `control_task_id` | Any unsigned int |
| `@get_ut_id(id)` | `ut_id` | WSE-3 only, 0-7 |

### Task Binding (comptime only, top-level comptime block)

| Builtin | Description |
|---|---|
| `@bind_local_task(fn, id)` | Bind function (no params) to local task |
| `@bind_data_task(fn, id)` | Bind function (>=1 param) to data task |
| `@bind_control_task(fn, id)` | Bind function to control task |

### Task State Control

| Builtin | Description | Call Site |
|---|---|---|
| `@activate(local_task_id)` | Add task to scheduling queue | comptime/runtime |
| `@block(id)` | Block task/microthread | comptime/runtime |
| `@unblock(id)` | Unblock task/microthread | runtime |

`@block`/`@unblock` accept: `local_task_id`, `data_task_id`, `color` (WSE-2), `input_queue` (WSE-3), `ut_id` (WSE-3).

---

## 7. Queue & Color Configuration

### Queue/Color ID Creation

| Builtin | Returns | Description |
|---|---|---|
| `@get_input_queue(id)` | `input_queue` | id: comptime non-negative int |
| `@get_output_queue(id)` | `output_queue` | id: comptime non-negative int |
| `@get_color(id)` | `color` | id: comptime integer |

### Queue Initialization

```csl
@initialize_queue(queue_id);                           // Basic
@initialize_queue(queue_id, .{ .color = my_color });   // With color binding
```

**WSE-3**: Required before using queues with async operations. Must appear in top-level comptime block. Called once per queue.

Config options:
- `color`: routable color (required on WSE-3)
- `priority`: `.{.high=true}`, `.{.medium=true}`, `.{.low=true}` (WSE-2 input only)
- `ctrl_table_id`: control task table index (WSE-3 input only, default=0)

### Queue Flushing (WSE-3 only)

```csl
@queue_flush(queue_id);                          // Activate teardown when queue empty
@set_empty_queue_handler(func, queue_id);        // Custom handler (comptime)
```

### Color Configuration (layout block only)

```csl
@set_color_config(x, y, color, .{
    .routes = .{ .rx = .{WEST}, .tx = .{EAST} },
    .switches = .{ .pos1 = .{.WEST}, .pos2 = .{.EAST} },
    .filter = filter_config,
    .teardown = true,
    .color_swap_x = false,
    .color_swap_y = false
});
```

### Local Color Configuration (comptime block)

```csl
@set_local_color_config(color, config);  // Applies to all PEs with same code
```

### Other Color Builtins

| Builtin | Description |
|---|---|
| `@set_teardown_handler(fn, color)` | Handler called on teardown (comptime) |
| `@get_filter_id(color)` | Get filter integer ID (comptime) |

---

## 8. DSD & DSR Builtins

### DSD Creation & Manipulation

| Builtin | Description |
|---|---|
| `@get_dsd(type, config)` | Create DSD (mem1d, mem4d, fabin, fabout) |
| `@increment_dsd_offset(dsd, offset, type)` | Advance base address |
| `@set_dsd_base_addr(dsd, addr)` | Clone with new base |
| `@set_dsd_length(dsd, length)` | Clone with new extent |
| `@set_dsd_stride(dsd, stride)` | Clone with new stride (mem1d only) |
| `@allocate_fifo(buffer)` | Allocate FIFO |
| `@set_fifo_read_length(fifo, len)` | Set FIFO read length |
| `@set_fifo_write_length(fifo, len)` | Set FIFO write length |

### DSR Management

| Builtin | Description |
|---|---|
| `@get_dsr(type, id)` | Allocate DSR register |
| `@get_xdsr(id)` | Allocate extended DSR |
| `@load_to_dsr(dsr, dsd)` | Load DSD into DSR |
| `@load_to_dsr(dsr, dsd, config)` | Load with config (.async, .activate, .save_address, etc.) |

### Filter Operation

```csl
@dfilt(fabin_dsd, .{ .async = true });
// Drop data wavelets until N control wavelets arrive (N = extent of DSD)
```

---

## 9. Export & RPC

### Symbol Export

```csl
// In layout block: declare exported name
@export_name("symbol_name", type);
@export_name("symbol_name", type, true);  // writable (for H2D memcpy)

// In comptime block: bind symbol
@export_symbol(variable);                 // Use variable name as export name
@export_symbol(variable, "custom_name");  // Custom export name
```

### Symbol Retrieval

| Builtin | Description |
|---|---|
| `@get_symbol_id(symbol)` | Unique u16 ID for exported symbol |
| `@get_symbol_value(type, id)` | Get copy of exported symbol (runtime) |
| `@get_tensor_ptr(id)` | Get tensor pointer as `[*]u16` (runtime) |
| `@has_exported_tensors()` | True if >=1 tensor pointer exported (comptime) |

### RPC Server

```csl
@rpc(data_task_id);  // Comptime only, max 1 per PE
```

### Canonical Example

```csl
// From examples/tutorials/gemv-01-complete-program/pe_program.csl
const y_ptr: [*]f32 = &y;

fn init_and_compute() void {
    initialize();
    compute();
    sys_mod.unblock_cmd_stream();
}

comptime {
    @export_symbol(y_ptr, "y");
    @export_symbol(init_and_compute);
}
```

---

## 10. Layout & Rectangle

### Layout Block Builtins

| Builtin | Description |
|---|---|
| `@set_rectangle(width, height)` | Define PE grid size (exactly once per layout) |
| `@get_rectangle()` | Get rectangle info: `.width`, `.height`, `.fabric`, `.offsets` |
| `@set_tile_code(x, y)` | Set PE code at (x,y) |
| `@set_tile_code(x, y, filename)` | Set PE code with file |
| `@set_tile_code(x, y, filename, params)` | Set PE code with params |
| `@set_color_config(x, y, color, config)` | Configure color at (x,y) |

### Example

```csl
// From examples/tutorials/gemv-01-complete-program/layout.csl
layout {
    @set_rectangle(1, 1);
    @set_tile_code(0, 0, "pe_program.csl", .{
        .memcpy_params = memcpy.get_params(0),
    });
}
```

---

## 11. Type System & Conversion

### Type Introspection

| Builtin | Returns | Description |
|---|---|---|
| `@type_of(expr)` | type | Get type of expression (comptime) |
| `@is_same_type(t1, t2)` | bool | Compare types |
| `@is_comptime()` | bool | True if in comptime context |

### Type Conversion

| Builtin | Description |
|---|---|
| `@as(result_type, value)` | Numeric coercion (float→int: round toward zero) |
| `@bitcast(result_type, value)` | Reinterpret bits (same bitwidth required) |
| `@ptrcast(ptr_type, ptr)` | Pointer type cast (runtime only) |
| `@get_int(value)` | Extract underlying integer from color, task_id, enum, etc. |
| `@fp16()` | Returns selected f16 format type (f16, cb16, or bf16) |

---

## 12. Array & Tensor

### Creation

| Builtin | Description |
|---|---|
| `@constants(tensor_type, value)` | Initialize tensor with single value |
| `@zeros(tensor_type)` | Initialize tensor with zeros |

### Introspection

| Builtin | Returns | Description |
|---|---|---|
| `@dimensions(array_type)` | `[n]u32` | Size of each dimension |
| `@element_type(array_type)` | type | Element type |
| `@element_count(array_type)` | `u32` | Total element count |
| `@rank(array_type)` | `u16` | Number of dimensions |

---

## 13. String & Struct

### String Operations (comptime)

| Builtin | Description |
|---|---|
| `@strcat(s1, s2, ..., sN)` | Concatenate strings |
| `@strlen(s)` | String length |
| `@get_array(string)` | Convert to byte array (not null-terminated) |
| `@get_string_from_byte(byte)` | Single-byte string from u8 |

### Struct Operations

| Builtin | Description |
|---|---|
| `@concat_structs(s1, s2)` | Merge two comptime structs |
| `@field(struct, name)` | Access field by string name (lvalue) |
| `@has_field(struct, name)` | Check if field exists |

### @concat_structs Example

```csl
// From examples/tutorials/gemv-08-routes-3/layout.csl
const params = @concat_structs(base_params, .{
    .pe_id = pe_id,
    .is_border = is_border,
});
@set_tile_code(x, y, "pe_program.csl", params);
```

---

## 14. Range & Iteration

```csl
@range(elem_type, stop)                    // [0, stop)
@range(elem_type, start, stop, step)       // [start, stop) with step

// Introspection
@range_start(range)
@range_stop(range)
@range_step(range)
```

### Example

```csl
for (@range(u16, N)) |i| {
    @fmacs(y_dsd, y_dsd, A_dsd, x[i]);
}
```

---

## 15. Assertions & Debug

| Builtin | Description |
|---|---|
| `@assert(cond)` | Runtime assertion (simulation only); at comptime, equivalent to `@comptime_assert` |
| `@comptime_assert(cond)` | Compile-time assertion |
| `@comptime_assert(cond, msg)` | Compile-time assertion with message |
| `@comptime_print(val...)` | Print at compile time |

---

## 16. Module & Import

| Builtin | Description |
|---|---|
| `@import_module(path)` | Load CSL module |
| `@import_module(path, params)` | Load module with parameter binding |
| `@import_comptime_value(type, filename)` | Import JSON file as comptime value |

### Example

```csl
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
const math_lib = @import_module("<math>");
const my_mod = @import_module("my_module.csl", .{ .N = 10 });
```

---

## 17. PRNG & Utility

| Builtin | Description |
|---|---|
| `@random16()` | Random u16 (LFSR: x^23 + x^18 + 1, initial: 0xdeadbeef) |
| `@set_active_prng(id)` | Select PRNG (runtime only) |

### Hardware Configuration (advanced)

| Builtin | Description |
|---|---|
| `@get_config(addr)` | Read configuration register |
| `@set_config(addr, value)` | Write configuration register |
| `@get_config_unchecked(addr)` | Read without validation |
| `@set_config_unchecked(addr, value)` | Write without validation |

---

## 18. WSE-3 Specific Builtins

### @set_control_task_table

```csl
@set_control_task_table();  // Default: 4 instructions, stride 1
@set_control_task_table(.{
    .instructions = 8,  // 2, 4, or 8
    .stride = 2         // [1, 7]
});
```

Separates control tasks from data/local tasks. At most once per top-level comptime block.

### @is_arch

```csl
if (@is_arch("wse2")) { ... }
if (@is_arch("wse3")) { ... }
```

Comptime evaluation. Used for architecture-conditional compilation.

---

## 19. WSE-2 vs WSE-3 Differences

| Feature | WSE-2 | WSE-3 |
|---|---|---|
| Queue initialization | Implicit | **@initialize_queue() required** |
| `@get_data_task_id` | Takes `color` | Takes `input_queue` |
| Local task IDs | 8-24 | 8-30 |
| `@get_ut_id` | N/A | Required (0-7) |
| Queue↔Microthread | Coupled (same ID) | **Decoupled** |
| Multiple fabric inputs | Allowed | **Not allowed** (use FIFO) |
| `@initialize_queue` params | color, priority | color, ctrl_table_id |
| `@queue_flush` | N/A | Available |
| `@set_empty_queue_handler` | N/A | Available |
| `@set_control_task_table` | N/A | Available |

### Architecture-Conditional Pattern

```csl
comptime {
    if (@is_arch("wse2")) {
        const task_id = @get_data_task_id(my_color);
    } else if (@is_arch("wse3")) {
        const task_id = @get_data_task_id(my_input_queue);
        @initialize_queue(my_input_queue, .{ .color = my_color });
        @initialize_queue(my_output_queue, .{ .color = my_color });
    }
}
```

---

## Related Topics

- **DSD Details**: See `CSL_DSD_Complete_Guide.md` for DSD types, DSR management, FIFO usage
- **Async Programming**: See `CSL_Async_Programming_Guide.md` for task system, block/unblock, microthreads
- **Official builtins docs**: `Cerebras_Docs/csl/language/builtins.md`
- **WSE-3 builtins docs**: `Cerebras_Docs/csl/language/builtins_wse3.md`
- **Tutorials**: `examples/tutorials/gemv-02-memory-dsds/`, `topic-10-map-builtin/`, `pipeline-01-basic/`

---

**Document Version**: 1.0
**Last Updated**: 2026-02-18
**Applicable SDK**: Cerebras CSL 1.4.0
