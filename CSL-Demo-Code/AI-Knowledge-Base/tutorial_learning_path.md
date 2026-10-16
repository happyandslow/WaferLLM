# CSL Tutorial Learning Path

> **Purpose**: Shows the recommended learning order for CSL tutorials and which topics branch where.
> **SDK Version**: 1.4.0

---

## Prerequisites

Before starting, read:
- `CSL_Language_Fundamentals_Guide.md` — CSL syntax, types, comptime
- Cerebras overview: `Cerebras_Docs/computing-with-cerebras.md`

---

## Core Path (Start Here)

```
gemv-00-basic-syntax          CSL syntax basics: variables, functions, tasks
       │
       ▼
gemv-01-complete-program      First complete program: layout.csl + pe_program.csl + run.py
       │
       ▼
gemv-02-memory-dsds           Memory DSDs: mem1d_dsd, tensor_access
       │
       ▼
gemv-03-memcpy                Host-device communication: memcpy H2D/D2H
       │
       ▼
gemv-04-params                Parameters: param, set_tile_code, @export_symbol
       │
       ▼
gemv-05-multiple-pes          Multiple PEs: @set_rectangle, different code per PE
       │
       ├──────────────────────────────┐
       ▼                              ▼
gemv-06-routes-1              topic-01-arrays-and-pointers
  Fabric routing:                 Arrays, pointers, runtime DSD
  colors, fabin/fabout            manipulation
       │
       ▼
gemv-07-routes-2
  Multi-hop routing,
  checkerboard patterns
       │
       ▼
gemv-08-routes-3
  Switches, control wavelets,
  multi-position routing
       │
       ├──────────────────────────────┐
       ▼                              ▼
gemv-09-streaming             topic-02-libraries
  Streaming mode:                 Using standard libraries
  MEMCPYH2D/D2H colors           (@import_module, <math>)
```

---

## Branching Topics (After Core Path)

### Pipeline & FIFO Branch

```
pipeline-01-basic             Basic pipeline: wavelet forwarding
       │
       ▼
pipeline-02-fifo              FIFO pipeline: @allocate_fifo, INFINITE_DSD_LEN
       │
       ▼
pipeline-03-multiple          Multiple pipeline stages
       │
       ▼
topic-09-fifos                Advanced: FIFO + loopback + block/unblock
```

### SdkLayout Branch (New Layout API)

```
sdklayout-01-introduction     SdkLayout basics: replace layout.csl with Python
       │
       ▼
sdklayout-02-routing          Routing with SdkLayout: set_color, set_route
       │
       ▼
sdklayout-03-ports-and-connections    Ports, connections, multi-region
       │
       ▼
sdklayout-04-h2d-d2h          H2D/D2H streams with SdkLayout
       │
       ▼
sdklayout-05-gemv             Full GEMV with SdkLayout (multi-PE)
```

### Advanced Color & Switch Topics

```
topic-03-streaming-wavelet-data    Wavelet data in data tasks
       │
       ▼
topic-04-sparse-tensors            Sparse tensor via streaming
       │
       ▼
topic-05-sentinels                 End-of-stream sentinels
       │
       ▼
topic-06-switches                  Switch positions and advance
       │
       ▼
topic-07-switches-entrypt          Switches with entry point tasks
       │
       ▼
topic-08-teardown                  Teardown mechanism
```

### Communication Patterns

```
topic-10-collectives               Row/column collectives
       │
       ▼
topic-11-collectives-advanced      Allreduce, broadcast patterns
       │
       ▼
topic-14-color-swap                Color swapping (ping-pong)
```

### Debugging Branch

```
topic-12-debug-library             <debug> library: trace buffers
       │
       ▼
topic-13-simprint                  <simprint>: printf-style output
```

### WSE-3 Specific

```
topic-15-wse3-microthreads         Microthread IDs, @get_ut_id, concurrent async
```

---

## Benchmarks (After Tutorials)

| Benchmark | Prerequisites | Topics |
|---|---|---|
| `single-tile-matvec/` | gemv-02, gemv-04 | Single-PE matrix-vector |
| `gemv-checkerboard-pattern/` | gemv-07, gemv-08 | Multi-PE with checkerboard routing |
| `gemv-collectives_2d/` | topic-10, topic-11 | GEMV with 2D collectives |
| `gemm-collectives_2d/` | topic-10, topic-11 | GEMM (SUMMA algorithm) |
| `bandwidth-test/` | gemv-06 | Fabric throughput measurement |
| `7pt-stencil-spmv/` | gemv-08, pipeline-01 | 3D stencil with neighbor exchange |
| `conjugate-gradient/` | 7pt-stencil, topic-10 | Full CG solver (state machine) |
| `bicgstab/` | conjugate-gradient | BiCGSTAB solver |
| `fft-1d-2d/` | gemv-09, topic-02 | FFT/IFFT implementation |
| `game-of-life/` | gemv-07 | 2D neighbor exchange |
| `histogram-torus/` | topic-14 | Torus routing |
| `mandelbrot/` | gemv-05 | Embarrassingly parallel |

---

## Knowledge Base Document Mapping

| Tutorial Range | Primary Knowledge Doc |
|---|---|
| gemv-00 through gemv-04 | `CSL_Language_Fundamentals_Guide.md`, `CSL_DSD_Complete_Guide.md` |
| gemv-05 through gemv-09 | `CSL_Routing_Colors_Switches_Guide.md`, `CSL_Host_Device_Communication_Guide.md` |
| pipeline-01 through pipeline-03 | `CSL_Queue_FIFO_Guide.md` |
| sdklayout-01 through sdklayout-05 | `CSL_SdkLayout_Guide.md` |
| topic-01 through topic-02 | `CSL_Language_Fundamentals_Guide.md`, `CSL_Standard_Libraries_Guide.md` |
| topic-03 through topic-08 | `CSL_Routing_Colors_Switches_Guide.md`, `CSL_Async_Programming_Guide.md` |
| topic-09 | `CSL_Queue_FIFO_Guide.md` |
| topic-10 through topic-11 | `CSL_Algorithm_Patterns_Guide.md` |
| topic-12 through topic-13 | `CSL_Debugging_Guide.md` |
| topic-14 | `CSL_Routing_Colors_Switches_Guide.md` |
| topic-15 | `CSL_WSE2_vs_WSE3_Migration_Guide.md`, `CSL_Async_Programming_Guide.md` |
