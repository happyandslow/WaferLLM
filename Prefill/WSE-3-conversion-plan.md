# Prefill WSE-2 to WSE-3 Conversion Plan

## Key Findings

The Prefill WSE-2 code is in a **partially-migrated hybrid state**:
- The **two-hop GEMM communication** DSDs (comm_pe.csl lines 304-538) already use WSE-3 patterns: `@get_input_queue()`, `@get_output_queue()`, `@initialize_queue()`, and `.simd_64`
- The **allreduce communication** DSDs (comm_pe.csl lines 92-158) are still WSE-2: no queue assignments, `.simd_32`, and `@block(color)` instead of `@block(queue)`
- The **layout.csl** uses WSE-2 patterns: `LAUNCH` color, even/odd PE iteration (`px += 2`), split `comm_even`/`comm_odd` params
- The **prefill.csl** uses `@rpc(@get_data_task_id(sys_mod.LAUNCH))` (WSE-2 only)
- The **run_sim.sh** lacks `--arch=wse3`

## Changes Per File

### Step 1: `run_sim.sh`
- Add `--arch=wse3` to `cslc` command
- Add cleanup commands for simulator artifacts

### Step 2: `src/layout.csl` — Major restructure
1. Remove `LAUNCH` color and `.LAUNCH = LAUNCH` from memcpy import
2. Merge two separate even/odd loops (`px += 2`) into single loop (`px += 1`)
3. Compute per-PE comm params directly (no more `comm_even`/`comm_odd` split)
4. Renumber colors (LAUNCH freed up color 0)

### Step 3: `src/comm_lib/comm_layout.csl` — Restructure
1. Replace `get_step_params(is_px_even)` with unified per-PE approach
2. `get_two_hop_comm_params()` stays largely the same

### Step 4: `src/comm_lib/comm_pe.csl` — WSE-3 queue/SIMD migration (most complex)
1. Add queue IDs to ALL allreduce fabric DSDs
2. Change `.simd_32` → `.simd_64` on all allreduce DSDs
3. Add `@initialize_queue()` calls for all new queues
4. Change `@block(color)` → `@block(queue_id)` in comptime blocks
5. Fix IQ 1 usage (reserved by memcpy on WSE-3) → move to IQ 2/3

### Step 5: `src/prefill.csl` — Minor changes
1. Remove `@rpc(@get_data_task_id(sys_mod.LAUNCH))`
2. Everything else is architecture-independent

### Step 6: `launch_sim.py` — Minor update
1. Optional: Add `simfab_numthreads=64` to `SdkRuntime()` for faster sim

## Queue Budget Solution

WSE-3 has 8 input queues (IQ 0-7) and 8 output queues (OQ 0-7).
Memcpy reserves IQ 0 and IQ 1 on WSE-3 (vs only IQ 0 on WSE-2).

### Key Design Decision: Eliminate Broadcast Colors

The WSE-2 version used 6 allreduce colors (4 reduce + 2 broadcast) plus 12 two-hop colors = 18 total.
This requires 8 IQ + 8 OQ which exceeds the 6 available IQ slots (IQ 2-7).

**Solution**: Replace dedicated broadcast colors (C_L_B, C_R_B) with a **software relay broadcast**
that reuses the reduce colors in the reverse direction. After the reduce converges to root:
- Root sends left via `c_left_send`, right via `c_right_send`
- PEs left of root receive via `c_left_recv`, relay via `c_left_send`
- PEs right of root receive via `c_right_recv`, relay via `c_right_send`

This saves 2 colors and 2 IQ + 2 OQ, at the cost of P/2 sequential relay hops
(vs hardware multicast). Also eliminated the even/odd color alternation (4 allreduce
colors → 4 directional colors).

### Final Queue Allocation (as implemented)

| Queue | Usage |
|-------|-------|
| IQ 0  | memcpy (reserved) |
| IQ 1  | memcpy (reserved on WSE-3) |
| IQ 2  | c_right_recv (allreduce) |
| IQ 3  | c_left_recv (allreduce) |
| IQ 4  | y_recv (two-hop) |
| IQ 5  | x_recv (two-hop, moved from IQ 1) |
| IQ 6  | x_shift_recv (two-hop shift) |
| IQ 7  | x_shift_re_recv (two-hop shift reverse) |
| OQ 0  | memcpy (reserved) |
| OQ 1  | c_right_send (allreduce) |
| OQ 2  | c_left_send (allreduce) |
| OQ 3  | x_send (two-hop) |
| OQ 4  | y_send (two-hop) |
| OQ 5  | x_shift_send (two-hop shift) |
| OQ 6  | x_shift_re_send (two-hop shift reverse) |

Total: 6 IQ + 6 OQ used out of 6 IQ + 7 OQ available.
