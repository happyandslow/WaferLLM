# CSL v1.4.0 — Unified Resource Index

> **Purpose**: This project is a Knowledge Base for the Cerebras CSL language, intended for manual lookup or building MCP services for AI Agent queries. Future updates to CLAUDE.md should follow this purpose.

> **Language rule**: All content in CLAUDE.md must be written in English, except for file names and paths. Follow this rule when updating.

> Maps CSL features → **AI knowledge summaries** → **docs** + **example code** + **library source**.
> When asked about any CSL feature, use this index to find all relevant resources, read them, then answer.
> SDK v1.4.0 | WSE-2, WSE-3

---

## Search Strategy

When querying any CSL-related topic, search the following layers in priority order, balancing **accuracy** and **token efficiency**:

> **Recall-first principle**: When uncertain about relevance, include the resource. It is better to read an extra file than to miss critical information. Always search at least 2 layers before concluding a topic is not covered.

1. **`AI-Knowledge-Base/`** (query first) — AI-curated intermediate knowledge docs. If a relevant summary exists, use its conclusions and code patterns directly, avoiding large reads of raw docs/code.
2. **`Cerebras_Docs/`** — Official documentation, for supplementing details, verifying AI summaries, and covering topics not yet summarized.
3. **`examples/`** — Official tutorials and benchmark code, for complete runnable examples.
4. **`source-code/`** — CSL standard library source, for deep understanding of library internals.
5. **`CSL-Practice-Notes/`** (write target) — Staging area for practical findings. When writing CSL code and discovering gotchas, tips, or optimizations, record them here. Periodically consolidate into `AI-Knowledge-Base/`.

> The content in `AI-Knowledge-Base/` is an **intermediate knowledge layer** between "final answers to questions/tasks" and "raw official docs/code". This progressive, layered search approach achieves comprehensive knowledge coverage (accuracy) while reducing token usage.

---

## Directory & Path Keys

```
CSL-Demo-Code/
├── AI-Knowledge-Base/      → kb/    # AI-curated knowledge docs (query first)
├── CSL-Practice-Notes/     → notes/ # Practical findings staging area (→ merge into kb/)
├── Cerebras_Docs/          → doc/    # Official docs (89 pages, markdown)
│   ├── csl/language/                  #   Language ref: syntax, types, builtins, DSDs, tasks...
│   ├── csl/tutorials/                 #   Tutorial concept guides (GEMV series)
│   ├── csl/code-examples/             #   Per-example documentation
│   ├── api-docs/                      #   API ref: SdkRuntime, SdkLayout, Appliance
│   └── debug/                         #   Debugging guide, SDK GUI
├── examples/
│   ├── tutorials/          → tut/    # 33 tutorials (gemv, pipeline, topic, sdklayout)
│   ├── benchmarks/         → bench/  # 22 benchmarks + benchmark-libs/
│   └── updates/            → upd/    # Rolling additions from SDK updates
└── source-code/
    └── csl_libraries/      → lib/    # CSL stdlib source (.csl)
```

**File convention**: `layout.csl` (PE grid/routing) → `pe_program.csl` (kernel) → `run.py` (host). Read in this order.
**Code example docs**: `doc/csl/code-examples/tutorial-{name}.md` or `benchmark-{name}.md` mirrors each example.
**General docs**: `doc/computing-with-cerebras.md` (concepts), `doc/installation-guide.md` (setup), `doc/csl/csl-compiler.md` (compiler), `doc/csl/working-with-code-samples.md` (howto).

---

## AI Knowledge Base

`AI-Knowledge-Base/` stores AI-curated topic-specific knowledge documents. It is the **first-priority query source** in this project's layered search system.

**Role**: An intermediate knowledge layer between final answers and raw official docs/code. Each document covers one topic, integrating key points from official docs, code patterns, best practices, and common pitfalls.

**Usage**:
- When facing a relevant question, **check this directory first**
- If an existing summary answers the question, use it directly to avoid large reads of raw docs
- If more detail or verification is needed, proceed layer by layer through official docs and code per the Search Strategy

### Available Documents

| File | Topic | Coverage |
|---|---|---|
| `CSL_Async_Programming_Guide.md` | Async programming guide | Microthreads, async DSD ops, task system, block/unblock sync, FIFO, queue init, WSE-2/3 differences |
| `CSL_DSD_Complete_Guide.md` | DSD & DSR complete guide | mem1d/mem4d/fabin/fabout DSDs, FIFO DSDs, DSR management, @load_to_dsr, save_address, DSD manipulation, async operations, INFINITE_DSD_LEN, queue depths |
| `CSL_Builtins_Reference_Guide.md` | Builtins reference | f32/f16/i16 DSD ops, @map, task binding/control, queue/color config, export/RPC, layout, type system, assertions, WSE-3 builtins |
| `CSL_Host_Device_Communication_Guide.md` | Host-device communication | Memcpy infrastructure, copy/streaming modes, @export_name/@export_symbol, SdkRuntime API, memcpy_h2d/d2h, launch, unblock_cmd_stream, 16-bit transfer, buffers |
| `CSL_Routing_Colors_Switches_Guide.md` | Routing, colors & switches | Colors, @set_color_config, routes, checkerboard alternation, switches (.pos1-.pos3, ring_mode), control wavelets (SWITCH_ADV), sentinels, filters (.kind/.range), color swapping, torus routing |
| `CSL_Language_Fundamentals_Guide.md` | Language fundamentals | Syntax, variables (const/var/param), type system, @as casting, comptime, @comptime_assert, generics, anytype, modules (@import_module), storage classes (export/extern), pointers, arrays, structs, enums |
| `CSL_Standard_Libraries_Guide.md` | Standard libraries guide | math, random, time/tsc, debug, simprint, collectives_2d, control, complex, directions, malloc, layout, message_passing — purpose, API, usage |
| `CSL_SdkLayout_Guide.md` | SdkLayout Python API | SdkLayout overview, create_code_region, ports, streams, paint_all, set_param, set_symbol, H2D/D2H patterns, GEMV example |
| `CSL_SdkLayout_DirectLink_Pattern_Guide.md` | SdkLayout direct-link patterns | Direct-link vs memcpy paradigms, demux/mux fan-out/fan-in, router state machine (SWITCH_ADV), multi-host SdkRuntime, loopback kernel, deadlock prevention, single-pass constraint |
| `CSL_Queue_FIFO_Guide.md` | Queues & FIFOs | Hardware queues, @get_input/output_queue, @initialize_queue, @allocate_fifo, FIFO+loopback, WSE-3 multi-input workaround, INFINITE_DSD_LEN, queue depths |
| `CSL_Algorithm_Patterns_Guide.md` | Algorithm patterns | State machines, BLAS with @map (dot/nrm2/axpy), stencil/neighbor exchange, counter sync, block/unblock, allreduce, iterative solver (CG) |
| `CSL_Debugging_Guide.md` | Debugging guide | debug library (trace buffers), simprint, CSDB, sim.log (landing/instruction/router logs), SDK GUI (fabric/timeline/wavelet) |
| `CSL_FFT_Complex_Arithmetic_Guide.md` | FFT & complex arithmetic | complex library, DSD-based complex ops, interleaved layout, Cooley-Tukey FFT, IFFT, f16/f32 precision, twiddle factors |
| `CSL_WSE2_vs_WSE3_Migration_Guide.md` | WSE-2 → WSE-3 migration | Quick reference, queue init, data_task_id, microthreads, multi-input restriction, queue depths, task ID ranges, portable patterns, migration checklist |

> Update this table when adding new knowledge summary documents.

---

## Keyword → Resource Map

### Language Basics

| Feature | Doc | Example |
|---|---|---|
| Syntax, loops, @range, functions | `doc/csl/language/syntax.md` | `tut/gemv-00-basic-syntax/code.csl` |
| Type system, @as casting | `doc/csl/language/types.md` | `tut/gemv-00-basic-syntax/code.csl` |
| Comptime, @comptime_assert | `doc/csl/language/comptime.md` | `tut/gemv-08-routes-3/layout.csl` |
| Generics | `doc/csl/language/generics.md` | — |
| Modules, @import_module | `doc/csl/language/modules.md` | `tut/topic-02-libraries/pe_program.csl` |
| Storage classes | `doc/csl/language/storage-classes.md` | — |
| param (compile-time params) | `doc/csl/language/comptime.md` | `tut/gemv-04-params/pe_program.csl` |
| Pointers, arrays, indexing | `doc/csl/language/syntax.md` | `tut/topic-01-arrays-and-pointers/pe_program.csl` |
| @export_name, @export_symbol | `doc/csl/language/builtins.md` | `tut/gemv-01-complete-program/pe_program.csl` |
| @is_arch("wse2"/"wse3") | `doc/csl/language/builtins.md` | `tut/gemv-06-routes-1/pe_program.csl` |
| @concat_structs | `doc/csl/language/builtins.md` | `tut/gemv-08-routes-3/layout.csl` |

### DSDs & DSRs
| Feature | Doc | Example |
|---|---|---|
| mem1d_dsd, @get_dsd, tensor_access | `doc/csl/language/dsds.md` | `tut/gemv-02-memory-dsds/pe_program.csl` |
| fabout_dsd, fabin_dsd (fabric I/O) | `doc/csl/language/dsds.md` | `tut/gemv-06-routes-1/pe_program.csl` |
| @increment_dsd_offset | `doc/csl/language/dsds.md` | `tut/gemv-02-memory-dsds/pe_program.csl` |
| INFINITE_DSD_LEN (0x7fff) | `doc/csl/language/dsds.md` | `tut/pipeline-02-fifo/pe_program.csl` |
| .fabric_color, .input_queue, .output_queue | `doc/csl/language/dsds.md` | `tut/gemv-06-routes-1/pe_program.csl` |
| DSR, @load_to_dsr | `doc/csl/language/dsrs.md` | `bench/benchmark-libs/stencil_3d_7pts/wse3/pe.csl` |

### Builtins (DSD-level)

| Feature | Doc | Example |
|---|---|---|
| @fmacs, @fadds, @fmovs, @fsubs, @fmuls, @fnegs | `doc/csl/language/builtins.md` | `tut/gemv-02-memory-dsds/pe_program.csl` |
| @mov16, @add16 (i16 ops) | `doc/csl/language/builtins.md` | `tut/pipeline-01-basic/pe_program.csl` |
| @map (element-wise user fn) | `doc/csl/language/builtins.md` | `tut/topic-10-map-builtin/pe_program.csl` |
| f16: @fmach, @faddh, @fmovh, @fnegh | `doc/csl/language/builtins.md` | `bench/fft-1d-2d/fft.csl` |
| WSE-3 specific builtins | `doc/csl/language/builtins_wse3.md` | — |

### Task System
| Feature | Doc | Example |
|---|---|---|
| local_task, @get_local_task_id (8-30 WSE-3) | `doc/csl/language/task-ids.md` | `tut/gemv-06-routes-1/pe_program.csl` |
| data_task, @get_data_task_id, @bind_data_task | `doc/csl/language/task-ids.md` | `tut/gemv-07-routes-2/pe_program.csl` |
| control_task, @get_control_task_id | `doc/csl/language/task-ids.md` | `tut/topic-05-sentinels/sentinel.csl` |
| @activate, @block, @unblock | `doc/csl/language/task-ids.md` | `tut/topic-09-fifos/buffer.csl` |
| .async + .activate callback | `doc/csl/language/task-ids.md` | `tut/topic-15-wse3-microthreads/left_pe.csl` |
| .async + .unblock callback | `doc/csl/language/task-ids.md` | `tut/topic-09-fifos/buffer.csl` |
| State machine (task chain) | — | `bench/conjugate-gradient/src/kernel_cg.csl` |
| Multi-task sync (counter) | — | `bench/benchmark-libs/stencil_3d_7pts/wse3/pe.csl` |
| Multi-task sync (block+unblock+activate) | — | `lib/memcpy/wse3/memcpy.csl` |

### Microthreads (WSE-3 only)
| Feature | Doc | Example |
|---|---|---|
| @get_ut_id(0-7), .ut_id | `doc/csl/language/microthreads_wse3.md` | `tut/topic-15-wse3-microthreads/` |
| Queue↔microthread decoupling | `doc/csl/language/microthreads_wse3.md` | `bench/benchmark-libs/stencil_3d_7pts/wse3/pe.csl` |

### Queues & FIFOs

| Feature | Doc | Example |
|---|---|---|
| @get_input/output_queue, @initialize_queue | `doc/csl/language/builtins.md` | `tut/gemv-06-routes-1/pe_program.csl` |
| @allocate_fifo | `doc/csl/language/builtins.md` | `tut/pipeline-02-fifo/pe_program.csl` |
| FIFO + loopback color | — | `tut/topic-09-fifos/buffer.csl` |
| FIFO as WSE-3 multi-input workaround | — | `tut/sdklayout-04-h2d-d2h/add2vec.csl` |
| @queue_flush, @set_empty_queue_handler (WSE-3) | `doc/csl/language/builtins_wse3.md` | `upd/queue_flush/queue_flush_sender.csl` |
| Output/input queue reuse via flush (WSE-3) | — | `upd/queue_flush/` |

### Routing, Colors & Advanced Hardware
| Feature | Doc | Example |
|---|---|---|
| @get_color, @set_color_config, routes | `doc/csl/language/advanced-features.md` | `tut/gemv-06-routes-1/layout.csl` |
| Checkerboard color alternation | — | `tut/gemv-07-routes-2/layout.csl` |
| color_swap_x (ping-pong) | — | `tut/topic-14-color-swap/pe_program.csl` |
| Switches (.pos1/.pos2/.pos3, .ring_mode) | `doc/csl/language/advanced-features.md` | `tut/topic-06-switches/layout.csl` |
| Filters (.kind .range, .min/max_idx) | `doc/csl/language/advanced-features.md` | `tut/topic-08-filters/layout.csl` |
| Control wavelets, SWITCH_ADV, TEARDOWN | `doc/csl/language/advanced-features.md` | `tut/topic-06-switches/send.csl` |
| Sentinels (end-of-stream) | — | `tut/topic-05-sentinels/sentinel.csl` |
| Torus routing | — | `bench/histogram-torus/routes.csl` |
| 4-dir neighbor exchange | — | `bench/benchmark-libs/stencil_3d_7pts/wse3/pe.csl` |
| Row/col broadcast & reduction | — | `bench/gemm-collectives_2d/pe.csl` |

### Host-Device Communication
| Feature | Doc | Example |
|---|---|---|
| Memcpy pattern, @import_module("<memcpy/memcpy>") | `doc/tensor-streaming.md` | `tut/gemv-01-complete-program/layout.csl` |
| @export_name (H2D symbol, writable=true) | `doc/csl/language/builtins.md` | `tut/gemv-03-memcpy/pe_program.csl` |
| @export_symbol (host RPC) | `doc/csl/language/builtins.md` | `tut/gemv-01-complete-program/pe_program.csl` |
| MEMCPYH2D_1..4, MEMCPYD2H_1..4 streaming | — | `tut/gemv-09-streaming/pe_program.csl` |
| sys_mod.unblock_cmd_stream() | — | `tut/topic-15-wse3-microthreads/left_pe.csl` |
| SdkRuntime: load→run→launch→memcpy→stop | `doc/api-docs/sdkruntime-api.md` | `tut/gemv-01-complete-program/run.py` |
| runner.memcpy_h2d/d2h, get_id, launch | `doc/api-docs/sdkruntime-api.md` | `tut/gemv-03-memcpy/run.py` |
| SdkLayout: create_code_region, ports, streams | `doc/api-docs/sdklayout-api.md` | `tut/sdklayout-04-h2d-d2h/run.py` |
| SdkLayout: paint_all, set_param, set_symbol | `doc/api-docs/sdklayout-api.md` | `tut/sdklayout-05-gemv/run.py` |
| Appliance mode | `doc/api-docs/appliance-api.md`, `doc/appliance-mode.md` | — |

### Layout Configuration

| Feature | Doc | Example |
|---|---|---|
| @set_rectangle, @set_tile_code | `doc/csl/language/builtins.md` | `tut/gemv-01-complete-program/layout.csl` |
| Multi-PE for loop layout | — | `tut/gemv-05-multiple-pes/layout.csl` |
| 2D grid with per-PE params | — | `tut/gemv-07-routes-2/layout.csl` |
| Parametric NxM layout | — | `tut/gemv-08-routes-3/layout.csl` |

### Standard Libraries

| Library | Import | Example | Source |
|---|---|---|---|
| memcpy | `<memcpy/memcpy>` | `tut/gemv-01*/` | `lib/memcpy/` |
| math | `<math>` | `tut/topic-02*/` | `lib/math.csl` |
| random | `<random>` | `tut/topic-02*/` | `lib/random.csl` |
| time/tsc | `<time>` | `tut/topic-02*/` | `lib/time.csl` |
| debug | `<debug>` | `tut/topic-12*/` | `lib/debug.csl` |
| simprint | `<simprint>` | `tut/topic-13*/` | `lib/simprint.csl` |
| layout | `<layout>` | `tut/gemv-07*/` | `lib/layout.csl` |
| collectives_2d | `<collectives_2d/pe>` | `tut/topic-11*/` | `lib/collectives_2d/` |
| control | `<control>` | `tut/topic-06*/` | `lib/control.csl` |
| complex | `<complex>` | `bench/fft-1d-2d/` | `lib/complex.csl` |
| directions | `<directions>` | `bench/game-of-life/` | `lib/directions.csl` |
| malloc | `<malloc>` | `bench/spmv-hypersparse/` | `lib/malloc.csl` |
| message_passing | `<message_passing>` | — | `lib/message_passing.csl` |
| **Lib docs** | `doc/csl/language/libraries.md` | WSE-3: `doc/csl/language/libraries_wse3.md` | |

---

## Tutorials Quick Reference

| ID | Focus |
|---|---|
| gemv-00 | Pure syntax (no run.py) |
| gemv-01 | First runnable: @export_name/symbol, memcpy |
| gemv-02 | Memory DSDs: mem1d_dsd, @fmacs, @fadds |
| gemv-03 | H2D memcpy: writable exports, get_id() |
| gemv-04 | Compile-time params |
| gemv-05 | Multi-PE 1D array |
| gemv-06 | Fabric routing: colors, queues, fabout/fabin DSDs |
| gemv-07 | 2D PE grid, data_tasks, checkerboard color |
| gemv-08 | Scalable NxM, @concat_structs |
| gemv-09 | Streaming memcpy, FIFO, multi-channel |
| pipeline-01 | H2D→compute→D2H pipeline (@add16, @mov16) |
| pipeline-02 | FIFO decoupling, INFINITE_DSD_LEN |
| pipeline-03 | Multi-PE pipeline, @block/@unblock |
| topic-01 | Arrays & pointers |
| topic-02 | Standard libraries (math, random, time) |
| topic-03 | Streaming wavelet data (data_task + MEMCPYH2D) |
| topic-04 | Sparse tensors (wavelet_data + index) |
| topic-05 | Sentinels & control tasks |
| topic-06 | Switches (.pos1/.pos2/.pos3, SWITCH_ADV) |
| topic-07 | Switch entrypoints (control payload forwarding) |
| topic-08 | Wavelet filters (.kind .range) |
| topic-09 | FIFOs + block/unblock sync |
| topic-10 | @map builtin |
| topic-11 | 2D collectives (broadcast, scatter, gather) |
| topic-12 | Debug library (trace buffers) |
| topic-13 | Simprint (formatted output) |
| topic-14 | Color swap (.color_swap_x) |
| topic-15 | WSE-3 microthreads (@get_ut_id, .ut_id) |
| sdklayout-01~05 | Python SdkLayout API (no layout.csl) |

## Benchmarks Quick Reference

| Name | Algorithm | Key Files |
|---|---|---|
| gemm-collectives_2d | GEMM (SUMMA) | pe.csl, run.py |
| gemv-collectives_2d | GEMV collectives | pe.csl, run.py |
| gemv-checkerboard-pattern | GEMV checkerboard | pe.csl, layout.csl, run.py |
| single-tile-matvec | MATVEC (@map, DSR) | src/pe_matvec.csl, run.py |
| cholesky | Cholesky factorization | pe.csl, run.py |
| residual | Residual norm | residual.csl, gemv.csl, axpy.csl, nrminf.csl |
| conjugate-gradient | CG solver (state machine) | src/kernel_cg.csl, src/blas.csl |
| preconditioned-conjugate-gradient | PCG solver | src/kernel_pcg.csl |
| bicgstab | BiCGSTAB (complex state machine) | src/kernel_bicgstab.csl |
| power-method | Power iteration | src/kernel_power.csl |
| 7pt-stencil-spmv | 7-pt stencil SpMV | src/kernel.csl |
| 25-pt-stencil | Wave equation (25pt) | task.csl, routes.csl, switches.csl |
| game-of-life | Cellular automaton | pe_program.csl |
| fft-1d-2d | 1D/2D FFT | fft.csl, reshape.csl |
| fft-3d | 3D FFT | layout.csl |
| spmv-hypersparse | Hypersparse SpMV | src/kernel.csl, src/hypersparse_spmv/pe.csl |
| bandwidth-test | H2D/D2H benchmark | src/bw_sync_kernel.csl |
| row-col-broadcast | Broadcast benchmark | src/kernel.csl |
| histogram-torus | Torus histogram | histogram.csl, routes.csl |
| mandelbrot | Mandelbrot set | left.csl, middle.csl |
| wide-multiplication | 256-bit multiply | pe.csl |

**Shared libs**: `bench/benchmark-libs/stencil_3d_7pts/` (reusable 7-pt stencil, WSE-2+3), `bench/benchmark-libs/allreduce/` (row→col→broadcast, ADD/MAX)

## Updates Quick Reference

Examples added incrementally via `examples/updates/` (SDK rolling additions):

| Name | Feature | Key Files |
| --- | --- | --- |
| queue_flush | WSE-3 queue reuse via `@queue_flush` + handler | `queue_flush_sender.csl`, `queue_flush_receiver.csl` |

---

## How-to → Files

| Question | Read (docs → code) |
|---|---|
| PE-to-PE communication | `doc/csl/language/advanced-features.md` → `tut/gemv-06-routes-1/` → `tut/gemv-07-routes-2/` |
| Async operations | `doc/csl/language/task-ids.md` → `tut/pipeline-02-fifo/` → `tut/topic-09-fifos/` |
| Task scheduling | `doc/csl/language/task-ids.md` → `tut/gemv-06-routes-1/` → `bench/conjugate-gradient/src/kernel_cg.csl` |
| H2D/D2H transfer | `doc/tensor-streaming.md` + `doc/api-docs/sdkruntime-api.md` → `tut/gemv-03-memcpy/` |
| SdkLayout | `doc/api-docs/sdklayout-api.md` → `tut/sdklayout-03*/` → `tut/sdklayout-05*/run.py` |
| WSE-3 microthreads | `doc/csl/language/microthreads_wse3.md` → `tut/topic-15*/` → `bench/benchmark-libs/stencil_3d_7pts/wse3/pe.csl` |
| Neighbor exchange | `tut/gemv-06-routes-1/` → `bench/game-of-life/pe_program.csl` → `bench/benchmark-libs/stencil_3d_7pts/wse3/pe.csl` |
| Collectives | `doc/csl/language/libraries.md` → `tut/topic-11*/` → `bench/gemm-collectives_2d/pe.csl` |
| FIFOs | `doc/csl/language/builtins.md` → `tut/pipeline-02-fifo/` → `tut/topic-09-fifos/` |
| Control wavelets & switches | `doc/csl/language/advanced-features.md` → `tut/topic-06-switches/` → `tut/topic-07-switches-entrypt/` |
| Iterative solvers | `bench/conjugate-gradient/src/kernel_cg.csl` → `src/blas.csl` → `run.py` |
| Debugging | `doc/debug/debugging.md` + `doc/debug/sdk-gui.md` → `tut/topic-12*/` → `tut/topic-13*/` |
| Memcpy internals | `lib/memcpy/wse3/memcpy.csl` → `memcpyh2d.csl` → `memcpyd2h.csl` |
| Complex arithmetic | `lib/complex.csl` → `bench/fft-1d-2d/fft.csl` |
| Compiler usage | `doc/csl/csl-compiler.md` |
| Conceptual overview | `doc/computing-with-cerebras.md` |

---

## Common Patterns (with canonical examples)

1. **Minimal memcpy program**: layout + pe_program + run.py → `tut/gemv-03-memcpy/`
2. **Fabric routing**: @get_color + @set_color_config + fabin/fabout DSDs → `tut/gemv-06-routes-1/`
3. **Async FIFO pipeline**: @allocate_fifo + async DSD ops → `tut/pipeline-02-fifo/`
4. **Block/unblock sync**: @block(task_id) + .async .unblock callback → `tut/topic-09-fifos/buffer.csl`
5. **State machine**: task chain via @activate(next_id) → `bench/conjugate-gradient/src/kernel_cg.csl`
6. **Dual-state sync**: comptime @block + .unblock + .activate → `lib/memcpy/wse3/memcpy.csl`
7. **Counter sync**: counter += 1; if (N) @activate(next) → `bench/benchmark-libs/stencil_3d_7pts/wse3/pe.csl`
8. **SdkLayout**: SdkLayout→region→ports→streams→compile→runtime → `tut/sdklayout-04-h2d-d2h/run.py`
9. **WSE-3 microthread**: @get_ut_id + .ut_id + @initialize_queue → `tut/topic-15-wse3-microthreads/left_pe.csl`
10. **Queue reuse (WSE-3)**: @queue_flush + @set_empty_queue_handler + queue_flush.exit → `upd/queue_flush/`

---

## WSE-2 vs WSE-3

| Feature | WSE-2 | WSE-3 |
|---|---|---|
| Queue init | Implicit | **@initialize_queue() required** |
| Queue↔Microthread | Coupled (same ID) | **Decoupled**: any ut_id(0-7) + any queue(0-7) |
| data_task_id from | `@get_data_task_id(color)` | `@get_data_task_id(input_queue)` |
| @get_ut_id | N/A | `@get_ut_id(0-7)` |
| Local task IDs | 8-24 | 8-30 |
| Arch check | `@is_arch("wse2")` | `@is_arch("wse3")` |
| Input queue depths | — | IQ0,1: 6; IQ2,3: 4; IQ4-7: 2 |
| Output queue depths | — | OQ0,1: 2; OQ2,3: 6; OQ4,5: 2 |
| Multi fabric inputs/op | Allowed | **Not allowed** (use FIFO workaround) |
