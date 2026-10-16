# AI-Knowledge-Base Document Index

> **Purpose**: Manifest of all AI-curated knowledge documents. Read this file first to quickly locate relevant docs by topic or keyword.
>
> **SDK Version**: 1.4.0 | **Last updated**: 2026-02-26

---

## Document Catalog

| # | File | Summary | Keywords |
|---|---|---|---|
| 1 | `CSL_Async_Programming_Guide.md` | Async programming, task management, microthreads, and inter-PE synchronization | `async`, `task`, `microthread`, `ut_id`, `@activate`, `@block`, `@unblock`, `FIFO`, `@allocate_fifo`, `@initialize_queue`, `data_task`, `local_task`, `control_task`, `.async`, `.activate`, `.unblock`, `WSE-3` |
| 2 | `CSL_DSD_Complete_Guide.md` | All DSD types, DSR management, runtime DSD manipulation, and async DSD operations | `DSD`, `DSR`, `mem1d_dsd`, `mem4d_dsd`, `fabin_dsd`, `fabout_dsd`, `@get_dsd`, `tensor_access`, `@increment_dsd_offset`, `@load_to_dsr`, `@set_dsd_base_addr`, `INFINITE_DSD_LEN`, `FIFO`, `queue_depth` |
| 3 | `CSL_Builtins_Reference_Guide.md` | Comprehensive reference for all CSL builtin functions | `@fmacs`, `@fadds`, `@fmovs`, `@fsubs`, `@fmuls`, `@fnegs`, `@fmach`, `@faddh`, `@fmovh`, `@mov16`, `@add16`, `@mov32`, `@map`, `@export_name`, `@export_symbol`, `@is_arch`, `@set_rectangle`, `@set_tile_code`, `@get_color`, `@set_color_config`, `@comptime_assert`, `WSE-3` |
| 4 | `CSL_Host_Device_Communication_Guide.md` | Host-device communication: memcpy infrastructure, copy/streaming modes, SdkRuntime API, kernel launching | `memcpy`, `SdkRuntime`, `memcpy_h2d`, `memcpy_d2h`, `launch`, `@export_name`, `@export_symbol`, `unblock_cmd_stream`, `streaming`, `copy`, `MEMCPYH2D`, `MEMCPYD2H`, `get_id`, `load`, `run`, `stop`, `run.py` |
| 5 | `CSL_Routing_Colors_Switches_Guide.md` | Fabric routing: colors, routes, switches, filters, control wavelets, sentinels, color swapping, torus | `color`, `route`, `@get_color`, `@set_color_config`, `NORTH`, `SOUTH`, `EAST`, `WEST`, `RAMP`, `switch`, `SWITCH_ADV`, `filter`, `sentinel`, `control_wavelet`, `color_swap`, `checkerboard`, `torus`, `fabout_dsd`, `fabin_dsd` |
| 6 | `CSL_Language_Fundamentals_Guide.md` | Language syntax, type system, comptime, generics, modules, storage classes, pointers & arrays | `const`, `var`, `param`, `type`, `@as`, `comptime`, `@comptime_assert`, `anytype`, `@type_of`, `@is_same_type`, `@import_module`, `export`, `extern`, `struct`, `enum`, `@range`, `for`, `while`, `switch`, `inline`, `fn`, `task`, `[*]T`, `*T` |
| 7 | `CSL_Standard_Libraries_Guide.md` | All 15+ standard libraries: math, random, time, debug, simprint, layout, control, complex, directions, malloc, message_passing, collectives_2d | `math`, `random`, `time`, `debug`, `simprint`, `layout`, `control`, `complex`, `directions`, `malloc`, `message_passing`, `collectives_2d`, `dsd_ops`, `@import_module`, `library`, `<math>`, `<random>`, `<time>`, `<debug>`, `<simprint>` |
| 8 | `CSL_SdkLayout_Guide.md` | SdkLayout Python API: CodeRegion, colors, ports, connections, H2D/D2H streams, parameters, placement, compilation | `SdkLayout`, `CodeRegion`, `create_code_region`, `set_color`, `set_route`, `add_port`, `connect`, `H2D`, `D2H`, `create_input_stream`, `create_output_stream`, `set_param`, `set_symbol`, `paint_all`, `compile`, `run.py` |
| 9 | `CSL_Queue_FIFO_Guide.md` | Hardware queues, FIFOs, queue initialization, FIFO pipelines, multi-input workaround, queue depths | `input_queue`, `output_queue`, `@get_input_queue`, `@get_output_queue`, `@initialize_queue`, `@allocate_fifo`, `FIFO`, `INFINITE_DSD_LEN`, `queue_depth`, `loopback`, `WSE-3` |
| 10 | `CSL_Algorithm_Patterns_Guide.md` | State machines, BLAS with @map, stencil/neighbor exchange, counter sync, block/unblock, allreduce, iterative solvers | `state_machine`, `@activate`, `@map`, `dot`, `nrm2`, `stencil`, `neighbor_exchange`, `counter`, `allreduce`, `conjugate_gradient`, `BLAS`, `@fmacs`, `solver` |
| 11 | `CSL_Debugging_Guide.md` | Debug library, simprint, CSDB debugger, simulator logs (landing/instruction/router), SDK GUI visualizer | `debug`, `simprint`, `csdb`, `sim.log`, `landing`, `inst_trace`, `router`, `SDK_GUI`, `trace_string`, `trace_timestamp`, `coredump`, `wavelet_trace`, `timeline` |
| 12 | `CSL_FFT_Complex_Arithmetic_Guide.md` | Complex number types, arithmetic, FFT/IFFT implementation, f16/f32 precision, twiddle factors | `FFT`, `IFFT`, `complex`, `complex_32`, `complex_64`, `twiddle`, `butterfly`, `Cooley-Tukey`, `f16`, `f32`, `@fmulh`, `@fmach`, `interleaved`, `stride` |
| 13 | `CSL_WSE2_vs_WSE3_Migration_Guide.md` | All WSE-2→WSE-3 differences: queue init, data_task_id, microthreads, multi-input restriction, portable patterns | `WSE-2`, `WSE-3`, `migration`, `@is_arch`, `@initialize_queue`, `@get_data_task_id`, `@get_ut_id`, `microthread`, `FIFO_workaround`, `queue_depth`, `portable` |
| 14 | `CSL_SdkLayout_DirectLink_Pattern_Guide.md` | Advanced SdkLayout: direct-link vs memcpy paradigms, demux/mux fan-out/fan-in, router state machine, multi-host runtime, loopback kernel design, deadlock prevention | `direct_link`, `demux`, `mux`, `SWITCH_ADV`, `fan-out`, `fan-in`, `loopback`, `nonblock`, `multi-host`, `SdkRuntime`, `paint_all`, `get_b_demux`, `get_mux`, `get_demux_adaptor`, `bandwidth`, `router_state`, `single_pass`, `region.color` |

---

## Quick Lookup by Topic

| Topic | Primary Doc | Also See |
|---|---|---|
| How to transfer data host ↔ device | `CSL_Host_Device_Communication_Guide.md` | `CSL_Routing_Colors_Switches_Guide.md`, `CSL_SdkLayout_Guide.md` |
| DSD types and usage | `CSL_DSD_Complete_Guide.md` | `CSL_Builtins_Reference_Guide.md` (DSD operations) |
| Async operations and sync patterns | `CSL_Async_Programming_Guide.md` | `CSL_DSD_Complete_Guide.md`, `CSL_Algorithm_Patterns_Guide.md` |
| Routing data between PEs | `CSL_Routing_Colors_Switches_Guide.md` | `CSL_DSD_Complete_Guide.md` (fabric DSDs) |
| Builtin function syntax | `CSL_Builtins_Reference_Guide.md` | — |
| Task system (data/local/control tasks) | `CSL_Async_Programming_Guide.md` | `CSL_Builtins_Reference_Guide.md` (task binding) |
| WSE-3 microthreads | `CSL_Async_Programming_Guide.md` | `CSL_WSE2_vs_WSE3_Migration_Guide.md` |
| Queue and FIFO setup | `CSL_Queue_FIFO_Guide.md` | `CSL_Async_Programming_Guide.md`, `CSL_DSD_Complete_Guide.md` |
| Switches and control wavelets | `CSL_Routing_Colors_Switches_Guide.md` | — |
| Sentinels (end-of-stream) | `CSL_Routing_Colors_Switches_Guide.md` | — |
| Filters (selective wavelet reception) | `CSL_Routing_Colors_Switches_Guide.md` | — |
| Color swapping (ping-pong) | `CSL_Routing_Colors_Switches_Guide.md` | — |
| SdkRuntime Python API | `CSL_Host_Device_Communication_Guide.md` | — |
| SdkLayout Python API | `CSL_SdkLayout_Guide.md` | `CSL_Host_Device_Communication_Guide.md` |
| Direct-link vs memcpy paradigm | `CSL_SdkLayout_DirectLink_Pattern_Guide.md` | `CSL_SdkLayout_Guide.md` |
| Demux/mux fan-out/fan-in pattern | `CSL_SdkLayout_DirectLink_Pattern_Guide.md` | `CSL_Routing_Colors_Switches_Guide.md` |
| Multi-host SdkRuntime (separate FPGA links) | `CSL_SdkLayout_DirectLink_Pattern_Guide.md` | — |
| Loopback kernel (recv→buf→send) | `CSL_SdkLayout_DirectLink_Pattern_Guide.md` | `CSL_Async_Programming_Guide.md` |
| Deadlock prevention (nonblock=True) | `CSL_SdkLayout_DirectLink_Pattern_Guide.md` | `CSL_SdkLayout_Guide.md` |
| Router state reset via runtime.run() | `CSL_SdkLayout_DirectLink_Pattern_Guide.md` | `CSL_Routing_Colors_Switches_Guide.md` |
| @export_name / @export_symbol | `CSL_Host_Device_Communication_Guide.md` | `CSL_Builtins_Reference_Guide.md` |
| sys_mod.unblock_cmd_stream() | `CSL_Host_Device_Communication_Guide.md` | — |
| Checkerboard color alternation | `CSL_Routing_Colors_Switches_Guide.md` | — |
| Torus routing | `CSL_Routing_Colors_Switches_Guide.md` | — |
| Language syntax (variables, loops, functions) | `CSL_Language_Fundamentals_Guide.md` | — |
| Type system (@as, comptime_int, FP16) | `CSL_Language_Fundamentals_Guide.md` | `CSL_Builtins_Reference_Guide.md` |
| Compile-time computation (comptime) | `CSL_Language_Fundamentals_Guide.md` | — |
| Generics and anytype | `CSL_Language_Fundamentals_Guide.md` | — |
| Modules (@import_module) | `CSL_Language_Fundamentals_Guide.md` | `CSL_Host_Device_Communication_Guide.md` |
| Storage classes (export/extern) | `CSL_Language_Fundamentals_Guide.md` | `CSL_Host_Device_Communication_Guide.md` |
| Pointers and arrays | `CSL_Language_Fundamentals_Guide.md` | `CSL_DSD_Complete_Guide.md` |
| Structs and enumerations | `CSL_Language_Fundamentals_Guide.md` | — |
| Standard libraries (math, random, etc.) | `CSL_Standard_Libraries_Guide.md` | `CSL_Builtins_Reference_Guide.md` |
| Complex arithmetic | `CSL_FFT_Complex_Arithmetic_Guide.md` | `CSL_Standard_Libraries_Guide.md` |
| FFT / IFFT | `CSL_FFT_Complex_Arithmetic_Guide.md` | — |
| Debugging and tracing | `CSL_Debugging_Guide.md` | `CSL_Standard_Libraries_Guide.md` (debug lib) |
| State machines and iterative solvers | `CSL_Algorithm_Patterns_Guide.md` | `CSL_Async_Programming_Guide.md` |
| Stencil and neighbor exchange | `CSL_Algorithm_Patterns_Guide.md` | `CSL_Routing_Colors_Switches_Guide.md` |
| BLAS operations (@map, dot, nrm2) | `CSL_Algorithm_Patterns_Guide.md` | `CSL_Builtins_Reference_Guide.md` |
| Allreduce / collectives | `CSL_Algorithm_Patterns_Guide.md` | `CSL_Standard_Libraries_Guide.md` |
| WSE-2 vs WSE-3 differences | `CSL_WSE2_vs_WSE3_Migration_Guide.md` | `CSL_Queue_FIFO_Guide.md` |
| Queue initialization (WSE-3) | `CSL_WSE2_vs_WSE3_Migration_Guide.md` | `CSL_Queue_FIFO_Guide.md` |
| FIFO workaround for multi-input | `CSL_Queue_FIFO_Guide.md` | `CSL_WSE2_vs_WSE3_Migration_Guide.md` |

---

## Last updated

All 14 documents cover the complete CSL SDK v1.4.0 knowledge base. See individual documents for details.
