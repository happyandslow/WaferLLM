---
title: "Appendix"
url: "https://sdk.cerebras.net/csl/language/appendix"
crawled_at: "2026-02-14 23:55:31"
---

# Appendix

> Source: [https://sdk.cerebras.net/csl/language/appendix](https://sdk.cerebras.net/csl/language/appendix)

## SIMD Mode

Many of the
[builtins for DSD operations](csl/language/builtins.md#language-builtins-for-dsd-operations)
have a SIMD (single instruction, multiple data) mode, in which multiple
operations can be performed in a single cycle.
Under appropriate conditions, these builtins will automatically
execute in SIMD mode when operating on DSDs, if possible.

In particular, builtins can only operate at their full SIMD width
if no bank conflicts occur when fetching the operands from memory.
The 48 KB of memory in a PE are laid out into 8 banks of 6 KB each.
Each successive 16 bits are located in successive banks.
In a single cycle, the PE can perform two 32-bit reads and one 32-bit write.
However, the reads must occur from separate banks. More specifically,
if the 8 banks are numbered 0 to 7, then the bank IDs `bank_1` and
`bank_2` of the two reads must be such that `bank_1 % 4 != bank_2 % 4`.

For best results in avoiding bank conflicts with SIMD operations, the operand
addresses should be 32-bit aligned, and
`(src0_addr % 8) == ((src1_addr + 4) % 8)`, where `src0_addr` and
`src1_addr` are the addresses of the operands.
Dumping the ELF file’s symbol table with the `--sym` option of
`cs_readelf` provides addresses and banking information for all symbols
in a compiled CSL program, and can be useful for determining if bank conflicts
may occur.

Additionally, if the DSD operands have non-contiguous strided accesses,
the SIMD width may be limited:

- Strides of 0 and 1 can operate at full SIMD width.
- Strides such that `stride % 8` is 2, 3, 5, or 6 can operate at full
  SIMD width.
- Unless stride is 1, strides such that `stride % 8` is 1 or 7 is limited
  to two operations per cycle, or a SIMD width of 2.
- Unless stride is 0, strides such that `stride % 8` is 0 is limited to
  one operation per cycle, or a SIMD width of 1.
- Strides such that `stride % 8` is 4 are limited to two operations per
  cycle, or a SIMD width of 4.

The maximum width of builtins which can operate in SIMD mode are given
in the table below.

| Builtin | | WSE-2 SIMD Width | | WSE-3 SIMD Width | |
| --- | --- | --- | --- | --- | --- |
| [@add16](csl/language/builtins.md#add16) | | 4 | | 8 | |
| [@addc16](csl/language/builtins.md#addc16) | | 1 | | 8 | |
| [@and16](csl/language/builtins.md#and16) | | 4 | | 8 | |
| [@fabsh](csl/language/builtins.md#fabsh) | | 4 | | 8 | |
| [@fabss](csl/language/builtins.md#fabss) | | 2 | | 4 | |
| [@faddh](csl/language/builtins.md#faddh) | | 4 | | 8 | |
| [@faddhs](csl/language/builtins.md#faddhs) | | 2 | | 4 | |
| [@fadds](csl/language/builtins.md#fadds) | | 2 | | 4 | |
| [@fnormh](csl/language/builtins.md#fnormh) | | 4 | | 8 | |
| [@fnorms](csl/language/builtins.md#fnorms) | | 2 | | 4 | |
| [@fh2s](csl/language/builtins.md#fh2s) | | 1 | | 4 | |
| [@fh2xp16](csl/language/builtins.md#fh2xp16) | | 1 | | 8 | |
| [@fmach](csl/language/builtins.md#fmach) | | 4 | | 8 | |
| [@fmachs](csl/language/builtins.md#fmachs) | | 2 | | 4 | |
| [@fmaxh](csl/language/builtins.md#fmaxh) | | 1 | | 8 | |
| [@fmaxs](csl/language/builtins.md#fmaxs) | | 1 | | 4 | |
| [@fmovh](csl/language/builtins.md#fmovh) | | 4 | | 8 | |
| [@fmovs](csl/language/builtins.md#fmovs) | | 2 | | 4 | |
| [@fmulh](csl/language/builtins.md#fmulh) | | 4 | | 8 | |
| [@fnegh](csl/language/builtins.md#fnegh) | | 4 | | 8 | |
| [@fnegs](csl/language/builtins.md#fnegs) | | 2 | | 4 | |
| [@fs2h](csl/language/builtins.md#fs2h) | | 1 | | 4 | |
| [@fs2xp16](csl/language/builtins.md#fs2xp16) | | 1 | | 4 | |
| [@fscaleh](csl/language/builtins.md#fscaleh) | | 4 | | 8 | |
| [@fscales](csl/language/builtins.md#fscales) | | 2 | | 4 | |
| [@fsubh](csl/language/builtins.md#fsubh) | | 4 | | 8 | |
| [@fsubs](csl/language/builtins.md#fsubs) | | 2 | | 4 | |
| [@mov16](csl/language/builtins.md#mov16) | | 4 | | 8 | |
| [@mov32](csl/language/builtins.md#mov32) | | 2 | | 4 | |
| [@or16](csl/language/builtins.md#or16) | | 4 | | 8 | |
| [@sar16](csl/language/builtins.md#sar16) | | 1 | | 4 | |
| [@sll16](csl/language/builtins.md#sll16) | | 1 | | 4 | |
| [@slr16](csl/language/builtins.md#slr16) | | 1 | | 4 | |
| [@sub16](csl/language/builtins.md#sub16) | | 4 | | 8 | |
| [@xor16](csl/language/builtins.md#xor16) | | 4 | | 8 | |
| [@xp162fh](csl/language/builtins.md#xp162fh) | | 1 | | 8 | |
| [@xp162fs](csl/language/builtins.md#xp162fs) | | 1 | | 4 | |
[previous

Advanced Hardware Features](csl/language/advanced-features.md "previous page")
[next

Running SDK on a Wafer-Scale Cluster](appliance-mode.md "next page")
