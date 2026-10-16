# CSL Knowledge Base Improvement Todo List

> **Purpose**: Track all tasks to improve this knowledge base, making the MCP server return more accurate and complete results. Execute items as token budget allows.
>
> **Core principle**: **Recall-first — better to return extra content than to miss anything.**

---

## Priority Legend

- **P0**: Critical — almost every query benefits from this
- **P1**: Important — frequently used scenarios
- **P2**: Nice-to-have — advanced or specific scenarios
- **Status**: `[ ]` = todo, `[x]` = done

---

## A. AI-Knowledge-Base Documents (Multi-layer Summaries)

Each document follows the format of `CSL_Async_Programming_Guide.md`: TOC → Core Concepts → Code Patterns → API Quick-ref → Best Practices → Common Pitfalls → Related Topics. Integrates official docs + tutorial code + benchmark code + library source.

### P0 — Nearly All Programs Depend On These

- [x] **A1. `CSL_DSD_Complete_Guide.md`**
  - Topics: mem1d_dsd, fabout/fabin_dsd, @get_dsd, tensor_access, @increment_dsd_offset, INFINITE_DSD_LEN, .fabric_color/.input_queue/.output_queue, DSR/@load_to_dsr
  - Sources: `Cerebras_Docs/csl/language/dsds.md`, `dsrs.md`, `examples/tutorials/gemv-02-memory-dsds/`, `gemv-06-routes-1/`, `pipeline-02-fifo/`, `benchmarks/benchmark-libs/stencil_3d_7pts/`

- [x] **A2. `CSL_Builtins_Reference_Guide.md`**
  - Topics: f32 ops (@fmacs/@fadds/@fmovs/@fsubs/@fmuls/@fnegs), f16 ops (@fmach/@faddh/@fmovh/@fnegh), i16 ops (@mov16/@add16), @map, @export_name/@export_symbol, @is_arch, @concat_structs, @set_rectangle/@set_tile_code, @get_color/@set_color_config
  - Sources: `Cerebras_Docs/csl/language/builtins.md`, `builtins_wse3.md`, all tutorial PE programs

- [x] **A3. `CSL_Host_Device_Communication_Guide.md`**
  - Topics: Memcpy H2D/D2H pattern, @import_module("<memcpy/memcpy>"), @export_name(writable), MEMCPYH2D_1..4/MEMCPYD2H_1..4 streaming, SdkRuntime API (load→run→launch→memcpy→stop), runner.memcpy_h2d/d2h, sys_mod.unblock_cmd_stream()
  - Sources: `Cerebras_Docs/tensor-streaming.md`, `api-docs/sdkruntime-api.md`, `examples/tutorials/gemv-01~03/`, `gemv-09-streaming/`, `topic-15-wse3-microthreads/`

- [x] **A4. `CSL_Routing_Colors_Switches_Guide.md`**
  - Topics: Colors, @get_color, @set_color_config, routes, checkerboard alternation, color_swap_x, Switches (.pos1/.pos2/.pos3, .ring_mode), Filters (.kind/.range), control wavelets (SWITCH_ADV, TEARDOWN), Sentinels, Torus routing
  - Sources: `Cerebras_Docs/csl/language/advanced-features.md`, `examples/tutorials/gemv-06~08/`, `topic-05~08/`, `topic-14-color-swap/`, `benchmarks/histogram-torus/`

### P1 — Frequently Used in Specific Scenarios

- [x] **A5. `CSL_Language_Fundamentals_Guide.md`**
  - Topics: Syntax (loops, @range, functions), type system (@as), comptime (@comptime_assert, param), generics, modules (@import_module), storage classes, pointers & arrays
  - Sources: `Cerebras_Docs/csl/language/syntax.md`, `types.md`, `comptime.md`, `generics.md`, `modules.md`, `storage-classes.md`, `examples/tutorials/gemv-00/`, `gemv-04/`, `topic-01~02/`

- [x] **A6. `CSL_Standard_Libraries_Guide.md`**
  - Topics: math, random, time/tsc, debug, simprint, collectives_2d, control, complex, directions, malloc, layout, message_passing — each lib's purpose, API, usage example
  - Sources: `Cerebras_Docs/csl/language/libraries.md`, `libraries_wse3.md`, `source-code/csl_libraries/*.csl`, `examples/tutorials/topic-02/`, `topic-11~13/`

- [x] **A7. `CSL_SdkLayout_Guide.md`**
  - Topics: SdkLayout Python API: create_code_region, ports, streams, paint_all, set_param, set_symbol, programming without layout.csl
  - Sources: `Cerebras_Docs/api-docs/sdklayout-api.md`, `examples/tutorials/sdklayout-01~05/`

- [x] **A8. `CSL_Queue_FIFO_Guide.md`**
  - Topics: @get_input/output_queue, @initialize_queue (WSE-3 required), @allocate_fifo, FIFO+loopback color, FIFO as WSE-3 multi-input workaround, queue depth limits
  - Sources: `Cerebras_Docs/csl/language/builtins.md`, `examples/tutorials/gemv-06/`, `pipeline-02-fifo/`, `topic-09-fifos/`, `sdklayout-04-h2d-d2h/`

### P2 — Advanced / Specific Algorithms

- [x] **A9. `CSL_Algorithm_Patterns_Guide.md`**
  - Topics: State machines (task chain @activate), iterative solvers (CG/PCG/BiCGSTAB), 2D Collectives (SUMMA GEMM), Stencil + neighbor exchange, counter sync
  - Sources: `examples/benchmarks/conjugate-gradient/`, `bicgstab/`, `gemm-collectives_2d/`, `benchmark-libs/stencil_3d_7pts/`, `25-pt-stencil/`

- [x] **A10. `CSL_Debugging_Guide.md`**
  - Topics: debug library (trace buffers), simprint, SDK GUI, @assert, @comptime_assert, common debugging techniques
  - Sources: `Cerebras_Docs/debug/debugging.md`, `debug/sdk-gui.md`, `examples/tutorials/topic-12/`, `topic-13/`, `source-code/csl_libraries/debug.csl`

- [x] **A11. `CSL_FFT_Complex_Arithmetic_Guide.md`**
  - Topics: complex library, FFT 1D/2D/3D implementation, f16 complex ops
  - Sources: `source-code/csl_libraries/complex.csl`, `examples/benchmarks/fft-1d-2d/`, `fft-3d/`

- [x] **A12. `CSL_WSE2_vs_WSE3_Migration_Guide.md`**
  - Topics: All WSE-2 → WSE-3 differences in one place: queue init, microthreads, decoupled queues, data_task_id changes, multi-input restrictions, @is_arch conditional compilation patterns
  - Sources: CLAUDE.md WSE-2 vs WSE-3 table, `Cerebras_Docs/csl/language/microthreads_wse3.md`, `builtins_wse3.md`, `libraries_wse3.md`

---

## B. Knowledge Base Structure Optimization

- [x] **B1. Create `AI-Knowledge-Base/_index.md`** (P0)
  - A manifest listing every document with: filename, one-line summary, keyword tags
  - Agent reads this first to quickly locate relevant docs without grep
  - Update this file every time a new document is added

- [x] **B2. Add "Related Topics" cross-references to each knowledge doc** (P1)
  - At the end of each document, add a "See Also" section linking to other relevant knowledge docs
  - Creates a knowledge network instead of isolated files

- [x] **B3. Create `AI-Knowledge-Base/code-patterns/` directory** (P1)
  - Extract 10-15 most reused code patterns as small standalone files
  - Examples: `minimal_memcpy_program.csl`, `fabric_routing_setup.csl`, `async_fifo_pipeline.csl`, `block_unblock_sync.csl`, `state_machine_task_chain.csl`, `wse3_microthread_setup.csl`
  - Each file: 20-50 lines, heavily commented, self-contained
  - More token-efficient than reading an entire tutorial

- [x] **B4. Create Tutorial learning path diagram** (P2)
  - `AI-Knowledge-Base/tutorial_learning_path.md`
  - Shows dependency: gemv-00 → gemv-01 → gemv-02 → ... and which topics branch off where
  - Helps Agent recommend the right reference example based on user's current knowledge level

- [x] **B5. Create `AI-Knowledge-Base/common_errors.md`** (P2)
  - Compile common CSL compiler errors and their solutions
  - Format: Error message → Cause → Fix → Example

---

## C. Existing Content Fixes

- [x] **C1. Translate `CSL_Async_Programming_Guide.md` to English** (P0)
  - Current content is in Chinese, violating CLAUDE.md's language rule
  - All AI-Knowledge-Base content must be in English

---

## D. MCP Server Optimization

### Search Rule Improvements

- [x] **D1. Add recall-first search principle to all tool descriptions** (P0)
  - Add to server-level instructions and each search tool's description:
  - > "Search rule: Prefer recall over precision — it is better to return extra results than to miss relevant content. When uncertain, broaden the search scope and search multiple layers."

- [x] **D2. Add multi-layer search guidance to `lookup_feature` results** (P0)
  - When returning CLAUDE.md index matches, append:
  - > "For comprehensive coverage, continue searching: search_knowledge → search_docs → search_code. Do not stop at one layer if the answer may be incomplete."

- [x] **D3. Enhance `search_knowledge` with document directory listing** (P1)
  - Return each document's title + one-line summary (from `_index.md` or first line) alongside grep results
  - Even when grep misses, Agent can decide to read_file a potentially relevant doc

### New Tools

- [x] **D4. Add `get_topic_guide` tool** (P0)
  - Input: CSL topic keyword
  - Output: Complete layered search path for that topic:
    1. AI-Knowledge-Base doc (if exists)
    2. Official doc paths
    3. Tutorial paths
    4. Benchmark paths
    5. Library source paths
  - Sources: CLAUDE.md "How-to → Files" table + "Keyword → Resource Map"
  - Agent follows this path layer by layer, ensuring nothing is missed

- [x] **D5. Add `search_all` tool** (P1)
  - Searches across ALL layers simultaneously: AI-Knowledge-Base + Cerebras_Docs + examples + source-code
  - Returns results grouped by layer (knowledge → docs → examples → source)
  - Includes de-duplication logic
  - Use when Agent doesn't know which layer contains the answer

- [x] **D6. Add `get_wse_differences` tool** (P2)
  - Input: optional feature keyword
  - Output: WSE-2 vs WSE-3 differences, filtered by keyword if provided
  - Sources: CLAUDE.md WSE-2/WSE-3 table + `CSL_WSE2_vs_WSE3_Migration_Guide.md` (once created)

### Tool Description Improvements

- [x] **D7. Add "when to use" guidance to each tool description** (P1)
  - Current descriptions only say "what it does"; add "when to use it"
  - `search_docs`: "Use when you need official language specification details, API documentation, or tutorial explanations that go beyond AI-curated summaries."
  - `search_code`: "Use when you need to find concrete implementation patterns, usage examples of specific APIs, or algorithm implementations."
  - `search_knowledge`: "Use FIRST for any CSL topic. These summaries cover the essentials. Only search deeper layers if the summary is insufficient."
  - `read_file`: "Use to read a specific file in full after search results indicate it's relevant. Also use to list directory contents."
  - `list_examples`: "Use to discover available tutorials and benchmarks by keyword. Good for finding reference implementations."
  - `lookup_feature`: "Use as the starting point for any CSL feature lookup. Returns an index of all relevant docs, examples, and source paths."

- [x] **D8. Add .py file search hint to `search_code`** (P1)
  - Current `file_pattern` defaults to `*.csl`, but `run.py` files are critical
  - Add to description: 'Default searches .csl files. Use file_pattern "*.py" for host-side Python code (run.py). Use file_pattern "*" for all file types.'

### Server-Level Optimization

- [x] **D9. Add MCP server-level instructions** (P0)
  - Use MCP SDK's `instructions` capability to provide global search strategy:
    1. Layered search order: search_knowledge → search_docs → search_code
    2. Recall-first principle
    3. WSE-2/WSE-3 awareness reminder
    4. File reading convention: layout.csl → pe_program.csl → run.py

- [x] **D10. Expose CLAUDE.md and `_index.md` as MCP Resources** (P1)
  - Use MCP SDK's `resources` capability
  - Agent automatically gets the full knowledge base overview on connect
  - Reduces the need for initial lookup_feature calls

- [x] **D11. Group search results by knowledge layer** (P2)
  - Currently grep results are sorted by file path
  - Modify output to group by: AI-Knowledge-Base → Cerebras_Docs → examples → source-code
  - Makes it easier for Agent to identify which layer answered the question

---

## E. CLAUDE.md Index Maintenance

- [x] **E1. Add "Recall-first" rule to Search Strategy section** (P0)
  - Add: "**Recall-first principle**: When uncertain about relevance, include the resource. It is better to read an extra file than to miss critical information. Always search at least 2 layers before concluding a topic is not covered."

- [x] **E2. Update "Available Documents" table after each new doc** (ongoing)
  - Keep CLAUDE.md's AI Knowledge Base table in sync with actual files

- [x] **E3. Add path validation script** (P2)
  - `scripts/validate_claude_md_paths.py`
  - Checks all file paths referenced in CLAUDE.md actually exist
  - Run periodically to prevent documentation rot

---

## F. Quality Assurance

- [x] **F1. MCP Server test suite** (P2)
  - Basic smoke tests for each tool: does it return results for known queries?
  - Path validation: do tool results reference existing files?
  - Edge cases: empty query, non-existent feature, very long query

---

## Execution Guide

| Budget | Recommended Tasks |
|---|---|
| Small (1 conversation) | C1 (translate to English), or E1 (add recall-first rule), or D1+D2 (search rule descriptions) |
| Medium (2-3 conversations) | One P0 document (A1-A4), then B1 (_index.md) |
| Large (half day) | D4+D5 (new MCP tools) + D9 (server instructions) + one P0 document |
| Systematic | A1 → A2 → A3 → A4 → B1 → A5 → ... (one doc at a time, update E2 after each) |

> **Rule**: After completing any AI-Knowledge-Base document, always update `_index.md` (B1) and CLAUDE.md's "Available Documents" table (E2).
