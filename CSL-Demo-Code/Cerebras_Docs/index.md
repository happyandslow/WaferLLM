# Cerebras SDK Documentation

> Crawled from: <https://sdk.cerebras.net/>
>
> Generated: 2026-02-14 23:55:48
>
> Total pages: 89

---

## Getting Started

- [Running SDK on a Wafer-Scale Cluster](appliance-mode.md)
- [A Conceptual View](computing-with-cerebras.md)
- [Documentation for Developing with CSL](index.md)
- [Installation and Setup](installation-guide.md)
- [Host Runtime and Tensor Streaming](tensor-streaming.md)

## CSL (Cerebras Streaming Language)

### Language Guide - Overview

- [CSL Language Guide](csl/language_index.md)

### Language Reference

- [Advanced Hardware Features](csl/language/advanced-features.md)
- [Appendix](csl/language/appendix.md)
- [Builtins](csl/language/builtins.md)
- [Builtins for WSE-3](csl/language/builtins_wse3.md)
- [Comptime](csl/language/comptime.md)
- [Data Structure Descriptors](csl/language/dsds.md)
- [Data Structure Registers](csl/language/dsrs.md)
- [Generics](csl/language/generics.md)
- [Libraries](csl/language/libraries.md)
- [Libraries for WSE-3](csl/language/libraries_wse3.md)
- [Microthread IDs](csl/language/microthreads_wse3.md)
- [Modules](csl/language/modules.md)
- [Storage Classes](csl/language/storage-classes.md)
- [Syntax of CSL](csl/language/syntax.md)
- [Task Identifiers and Task Execution](csl/language/task-ids.md)
- [Type System in CSL](csl/language/types.md)

### Tutorials

- [GEMV Tutorial 0: Basic CSL Syntax](csl/tutorials/gemv-00-basic-syntax.md)
- [GEMV Tutorial 1: A Complete Program](csl/tutorials/gemv-01-complete-program.md)
- [GEMV Tutorial 2: Memory DSDs](csl/tutorials/gemv-02-memory-dsds.md)
- [GEMV Tutorial 3: Memcpy](csl/tutorials/gemv-03-memcpy.md)
- [GEMV Tutorial 4: Parameters](csl/tutorials/gemv-04-params.md)
- [GEMV Tutorial 5: Multiple PEs](csl/tutorials/gemv-05-multiple-pes.md)
- [GEMV Tutorial 6: Routes and Fabric DSDs](csl/tutorials/gemv-06-routes-1.md)

### Code Examples

- [25-Point Stencil](csl/code-examples/benchmark-25-pt-stencil.md)
- [3D 7-Point Stencil SpMV](csl/code-examples/benchmark-7pt-stencil-spmv.md)
- [Bandwidth Test](csl/code-examples/benchmark-bandwidth-test.md)
- [BiCGSTAB](csl/code-examples/benchmark-bicgstab.md)
- [Cholesky](csl/code-examples/benchmark-cholesky.md)
- [Conjugate Gradient](csl/code-examples/benchmark-conjugate-gradient.md)
- [3D FFT](csl/code-examples/benchmark-fft-3d.md)
- [Conway’s Game of Life](csl/code-examples/benchmark-game-of-life.md)
- [GEMM with Collective Operations](csl/code-examples/benchmark-gemm-collectives.md)
- [GEMV with Checkerboard Pattern](csl/code-examples/benchmark-gemv-checkerboard.md)
- [GEMV with Collective Communications](csl/code-examples/benchmark-gemv-collectives.md)
- [Power Method](csl/code-examples/benchmark-power-method.md)
- [Preconditioned Conjugate Gradient](csl/code-examples/benchmark-preconditioned-conjugate-gradient.md)
- [Residual](csl/code-examples/benchmark-residual.md)
- [Host-to-Device Broadcast Test](csl/code-examples/benchmark-row-col-broadcast.md)
- [Single Tile Matvec](csl/code-examples/benchmark-single-tile-matvec.md)
- [GEMV 0: Basic CSL Syntax](csl/code-examples/tutorial-gemv-00-basic-syntax.md)
- [GEMV 1: A Complete Program](csl/code-examples/tutorial-gemv-01-complete-program.md)
- [GEMV 2: Memory DSDs](csl/code-examples/tutorial-gemv-02-memory-dsds.md)
- [GEMV 3: H2D and D2H Memcpy](csl/code-examples/tutorial-gemv-03-memcpy.md)
- [GEMV 4: Parameters](csl/code-examples/tutorial-gemv-04-params.md)
- [GEMV 5: Multiple PEs](csl/code-examples/tutorial-gemv-05-multiple-pes.md)
- [GEMV 6: Routes and Fabric DSDs, Part I](csl/code-examples/tutorial-gemv-06-routes-1.md)
- [GEMV 7: Routes and Fabric DSDs, Part II](csl/code-examples/tutorial-gemv-07-routes-2.md)
- [GEMV 8: Routes and Fabric DSDs, Part III](csl/code-examples/tutorial-gemv-08-routes-3.md)
- [GEMV 9: Memcpy Streaming Mode](csl/code-examples/tutorial-gemv-09-streaming.md)
- [Pipeline 1: Redirect fabin to fabout](csl/code-examples/tutorial-pipeline-01-basic.md)
- [Pipeline 2: Attach a FIFO to H2D](csl/code-examples/tutorial-pipeline-02-fifo.md)
- [Pipeline 3: Add an artificial halo](csl/code-examples/tutorial-pipeline-03-multiple.md)
- [SdkLayout 1: Introduction](csl/code-examples/tutorial-sdklayout-01-introduction.md)
- [SdkLayout 2: Basic routing](csl/code-examples/tutorial-sdklayout-02-routing.md)
- [SdkLayout 3: Ports and connections](csl/code-examples/tutorial-sdklayout-03-ports-and-connections.md)
- [SdkLayout 4: Host-to-device and device-to-host data streaming](csl/code-examples/tutorial-sdklayout-04-h2d-d2h.md)
- [SdkLayout 5: Generalized matrix-vector multiplication (GEMV)](csl/code-examples/tutorial-sdklayout-05-gemv.md)
- [Topic 1: Arrays and Pointers](csl/code-examples/tutorial-topic-01-arrays-and-pointers.md)
- [Topic 2: Libraries](csl/code-examples/tutorial-topic-02-libraries.md)
- [Topic 3: Streaming Wavelet Data](csl/code-examples/tutorial-topic-03-streaming-wavelet-data.md)
- [Topic 4: Wavelets for Sparse Tensors](csl/code-examples/tutorial-topic-04-sparse-tensors.md)
- [Topic 5: Sentinels](csl/code-examples/tutorial-topic-05-sentinels.md)
- [Topic 6: Switches](csl/code-examples/tutorial-topic-06-switches.md)
- [Topic 7: Switches and Control Entrypoints](csl/code-examples/tutorial-topic-07-switches-entrypt.md)
- [Topic 8: Filters](csl/code-examples/tutorial-topic-08-filters.md)
- [Topic 9: FIFOs](csl/code-examples/tutorial-topic-09-fifos.md)
- [Topic 10: @map Builtin](csl/code-examples/tutorial-topic-10-map-builtin.md)
- [Topic 11: Collective Communications](csl/code-examples/tutorial-topic-11-collectives.md)
- [Topic 12: Debug Library](csl/code-examples/tutorial-topic-12-debug-library.md)
- [Topic 13: Simprint Library](csl/code-examples/tutorial-topic-13-simprint.md)
- [Topic 14: Color Swap](csl/code-examples/tutorial-topic-14-color-swap.md)
- [Topic 13: WSE-3 Microthreads](csl/code-examples/tutorial-topic-15-wse3-microthreads.md)

### Compiler

- [CSL Compiler](csl/csl-compiler.md)

### Working with Code Samples

- [Working With Code Samples](csl/working-with-code-samples.md)


## Debugging

- [Debugging Guide](debug/debugging.md)
- [SDK GUI](debug/sdk-gui.md)

## API Reference

- [SDK Appliance API Reference](api-docs/appliance-api.md)
- [SdkLayout API Reference](api-docs/sdklayout-api.md)
- [SdkRuntime API Reference](api-docs/sdkruntime-api.md)

## Release Notes

- [Documentation Updates](sdk-release-notes/sdk-doc-updates.md)
- [SDK Release Notes](sdk-release-notes/sdk-rel-notes-cumulative.md)

