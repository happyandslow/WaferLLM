---
title: "BiCGSTAB"
url: "https://sdk.cerebras.net/csl/code-examples/benchmark-bicgstab"
crawled_at: "2026-02-14 23:55:08"
---

# BiCGSTAB

> Source: [https://sdk.cerebras.net/csl/code-examples/benchmark-bicgstab](https://sdk.cerebras.net/csl/code-examples/benchmark-bicgstab)

This example evaluates the performance of BiCGSTAB with a sparse matrix `A`
built by 7-point stencil. The kernel records the `start` and `end` of
BiCGSTAB by tsc counter. In addition the tsc counters of all PEs are not
synchronized in the beginning. To avoid the timing variation among those PEs,
`sync()` synchronizes all PEs and samples the reference clock.

There are two implementations, `kernel.csl` and `kernel_bicgstab.csl`
compiled by `run.py` and `device_run.py` respectively. Both kernels
define host-callable functions `f_sync()`, `f_tic()` and `f_toc()` in
order to synchronize the PEs and record the timing.

The kernel `kernel.csl` also defines a couple of host-callable functions to
implement BiCGSTAB algorithm, including

- `spmv()`: compute `A*x`, `A*p` and `A*s`
- `dot()`: compute `dot(r,r)`, `dot(r,v)`, `dot(t,s)`, `dot(t,t)`
  and `dot(r0,r)`
- others: update `x`, `p`, `r`, `v`, `s` and `t`

The kernel `kernel_bicgstab.csl` defines a host-callable function
`f_bicgstab` which implements the BiCGSTAB on the WSE. The `f_bicgstab`
introduces a state machine to call a sequence of `spmv()`, `dot()` and
others. Such state machine simply realizes the algorithm in `run.py`.

The kernel `allreduce/pe.csl` performs a reduction over the whole rectangle
to synchronize the PEs, then the bottom-right PE sends a signal to other PEs
to sample the reference clock.

The kernel `stencil_3d_7pts/pe.csl` performs a matrix-vector product (spmv)
where the matrix has 7 diagonals corresponding to 7 point stencil. The stencil
coefficients can vary per PE, but must be the same for the local vector. The
user can change the coefficients based on the boundary condition or curvilinear
coordinate transformation.

The script `run.py` or `device_run.py` has the following parameters:

- `-k=<int>` specifies the maximum size of local vector.
- `--zDim=<int>` specifies how many elements per PE are computed.
- `--max-ite=<int>` specifies number of iterations in power method.
- `--channels=<int>` specifies the number of I/O channels, no bigger than 16.

The `tic()` samples “time\_start” and `toc()` samples “time\_end”. The
`sync()` samples “time\_ref” which is used to adjust “time\_start” and
“time\_end”. The elapsed time (unit: cycles) is measured by
`cycles_send = max(time_end) - min(time_start)`

The overall runtime (us) is computed via the following formula
`time_send = (cycles_send / 0.85) * 1.e-3 us`

Note that the `allreduce` and `stencil_3d_7pts` modules used
in this code are identical to those used in [3D 7-Point Stencil SpMV](csl/code-examples/benchmark-7pt-stencil-spmv.md#sdkruntime-7pt-stencil-spmv).

## layout.csl

```csl
// color map: memcpy + allreduce + stencil
//
// color  var   color  var        color  var              color  var
//   0   C0       9                18    EN_REDUCE_2       27   reserved (memcpy)
//   1   C1      10                19    EN_REDUCE_3       28   reserved (memcpy)
//   2   C2      11                20    EN_REDUCE_4       29   reserved (memcpy)
//   3   C3      12                21    reserved (memcpy) 30   reserved (memcpy)
//   4   C4      13                22    reserved (memcpy) 31   reserved
//   5   C5      14  EN_STENCIL_1  23    reserved (memcpy) 32
//   6   C6      15  EN_STENCIL_2  24                      33
//   7   C7      16  EN_STENCIL_3  25                      34
//   8   C8      17  EN_REDUCE_1   26                      35
//

// c0,c1,c2,c3,c4,c5,c6,c7 are internal colors of 7-point stencil
param C0_ID: i16;
param C1_ID: i16;
param C2_ID: i16;
param C3_ID: i16;
param C4_ID: i16;
param C5_ID: i16;
param C6_ID: i16;
param C7_ID: i16;
// c8 is an internal color of allreduce
param C8_ID: i16;

param MAX_ZDIM: i16; // maximum size of local vector x and y
param width: i16 ; // width of the core
param height: i16 ; // height of the core

param BLOCK_SIZE: i16; // size of temporary buffers for communication

const C0: color = @get_color(C0_ID);
const C1: color = @get_color(C1_ID);
const C2: color = @get_color(C2_ID);
const C3: color = @get_color(C3_ID);
const C4: color = @get_color(C4_ID);
const C5: color = @get_color(C5_ID);
const C6: color = @get_color(C6_ID);
const C7: color = @get_color(C7_ID);
const C8: color = @get_color(C8_ID);

// entrypoints of 7-point stenil
const EN_STENCIL_1: local_task_id = @get_local_task_id(14);
const EN_STENCIL_2: local_task_id = @get_local_task_id(15);
const EN_STENCIL_3: local_task_id = @get_local_task_id(16);

// entrypoints of allreduce
const EN_REDUCE_1: local_task_id = @get_local_task_id(17);
const EN_REDUCE_2: local_task_id = @get_local_task_id(18);
const EN_REDUCE_3: local_task_id = @get_local_task_id(19);
const EN_REDUCE_4: local_task_id = @get_local_task_id(20);

const stencil = @import_module( "../../benchmark-libs/stencil_3d_7pts/layout.csl", .{
    .colors = [8]color{C0, C1, C2, C3, C4, C5, C6, C7},
    .entrypoints = [3]local_task_id{EN_STENCIL_1, EN_STENCIL_2, EN_STENCIL_3},
    .width = width,
    .height = height
    });

const reduce = @import_module( "../../benchmark-libs/allreduce/layout.csl", .{
    .colors = [1]color{C8},
    .entrypoints = [4]local_task_id{EN_REDUCE_1, EN_REDUCE_2, EN_REDUCE_3, EN_REDUCE_4},
    .width = width,
    .height = height
    });

const memcpy = @import_module( "<memcpy/get_params>", .{
    .width = width,
    .height = height,
    });

layout{

    @comptime_assert(C0_ID < C1_ID);
    @comptime_assert(C1_ID < C2_ID);
    @comptime_assert(C2_ID < C3_ID);
    @comptime_assert(C3_ID < C4_ID);
    @comptime_assert(C4_ID < C5_ID);
    @comptime_assert(C5_ID < C6_ID);
    @comptime_assert(C6_ID < C7_ID);
    @comptime_assert(C7_ID < C8_ID);

    // step 1: configure the rectangle which does not include halo
    @set_rectangle( width, height );

    // step 2: compile csl code for a set of PEx.y and generate out_x_y.elf
    //   format: @set_tile_code(x, y, code.csl, param_binding);

    var py: i16 = 0;
    while(py < height) : (py +=1) {
        var px: i16 = 0;
        while(px < width) : (px +=1) {

            const memcpyParams = memcpy.get_params(px);
            const stencilParams = stencil.get_params(px, py);
            const reduceParams = reduce.get_params(px, py);
            var params: comptime_struct = .{
                .memcpyParams = memcpyParams,
                .reduceParams = reduceParams,
                .MAX_ZDIM = MAX_ZDIM,
                .BLOCK_SIZE = BLOCK_SIZE,
                .stencilParams = stencilParams
            };

            @set_tile_code(px, py, "kernel.csl", params);
        }
    }

    @export_name("b", [*]f32, true);
    @export_name("x", [*]f32, true);
    @export_name("stencil_coeff", [*]f32, true);
    @export_name("time_buf_u16", [*]u16, true);
    @export_name("time_ref", [*]u16, true);
    @export_name("rho", [*]f32, true);
    @export_name("xi", [*]f32, true);

    @export_name("f_enable_timer", fn()void);
    @export_name("f_tic", fn()void);
    @export_name("f_toc", fn()void);
    @export_name("f_memcpy_timestamps", fn()void);

    @export_name("f_bicgstab_init", fn(i16)void);
    @export_name("f_spmv_Ax", fn()void);
    @export_name("f_residual", fn()void);
    @export_name("f_init_rho", fn()void);
    @export_name("f_spmv_Ap", fn()void);
    @export_name("f_r0_dot_v", fn()void);
    @export_name("f_update_alpha_s_t", fn()void);
    @export_name("f_t_dot_s", fn()void);
    @export_name("f_t_dot_t", fn()void);
    @export_name("f_update_w_x_r_rho", fn()void);
    @export_name("f_update_beta_p_xi", fn()void);

    @export_name("f_sync", fn()void);
    @export_name("f_reference_timestamps", fn()void);
} // end of layout

```

## kernel.csl

```csl
param memcpyParams: comptime_struct;

param reduceParams: comptime_struct;

param stencilParams: comptime_struct;

param MAX_ZDIM: i16; // size of vector x

param BLOCK_SIZE: i16; // size of temporary buffers for communication

const timestamp = @import_module("<time>");

const math_lib = @import_module("<math>");

const blas_lib = @import_module("blas.csl");

// memcpy module reserves
// - input/output queue 0 and 1
const sys_mod = @import_module( "<memcpy/memcpy>", memcpyParams);

// allreduce uses input queue/output queue 1
const reduce_mod = @import_module( "../../benchmark-libs/allreduce/pe.csl", @concat_structs(reduceParams, .{
     .f_callback = sys_mod.unblock_cmd_stream,
     .queues = [1]u16{2},
     .dest_dsr_ids = [1]u16{1},
     .src0_dsr_ids = [1]u16{1},
     .src1_dsr_ids = [1]u16{1}
     }));

// output queue cannot overlap input queues
const stencil_mod = @import_module( "../../benchmark-libs/stencil_3d_7pts/pe.csl", @concat_structs(stencilParams, .{
     .f_callback = sys_mod.unblock_cmd_stream,
     .input_queues = [4]u16{4, 5, 6, 7},
     .output_queues = if (@is_arch("wse3")) [4]u16{4, 5, 6, 7} else [1]u16{3},
     .output_ut_id = 3,
     .BLOCK_SIZE = BLOCK_SIZE,
     .dest_dsr_ids = [2]u16{2,3},
     .src0_dsr_ids = [1]u16{2},
     .src1_dsr_ids = [2]u16{2,3}
     }));

// tsc_size_words = 3
// starting time of H2D/D2H
var tscStartBuffer = @zeros([timestamp.tsc_size_words]u16);
// ending time of H2D/D2H
var tscEndBuffer = @zeros([timestamp.tsc_size_words]u16);

var b = @zeros([MAX_ZDIM]f32); // right-hand-side
var x = @zeros([MAX_ZDIM]f32); // approximated solution

var r0 = @zeros([MAX_ZDIM]f32); // initial residual
var r = @zeros([MAX_ZDIM]f32); // residual
var p = @zeros([MAX_ZDIM]f32); // Krylov space
var v = @zeros([MAX_ZDIM]f32); // Krylov space
var s = @zeros([MAX_ZDIM]f32); // can align with r
var t = @zeros([MAX_ZDIM]f32); // t = A * s

var dot = @zeros([1]f32); // dummy variable for f_sync
var xi = @zeros([1]f32);
var rho = @zeros([1]f32);
var rho_old = @zeros([1]f32);
var r0_dot_v = @zeros([1]f32);
var t_dot_s = @zeros([1]f32);
var t_dot_t = @zeros([1]f32);
var w: f32 = @as(f32,0); // w = (t,s)/(t,t)
var alpha: f32 = @as(f32,0); // alpha = rho/(r0, v)
var beta: f32 = @as(f32,0); // beta = (rho/rho_old)*(alpha/w)

// stencil coefficients are organized as
// {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
//
// The formula is
//    c_west * x[i-1][j][k] + c_east * x[i+1][j][k] +
//    c_south * x[i][j-1][k] + c_north * x[i][j+1][k] +
//    c_bottom * x[i][j][k-1] + c_top * x[i][j][k+1] +
//    c_center * x[i][j][k]
var stencil_coeff = @zeros([7]f32);

// time_buf_u16[0:5] = {tscStartBuffer, tscEndBuffer}
var time_buf_u16 = @zeros([timestamp.tsc_size_words*2]u16);

// reference clock inside allreduce module
var time_ref_u16 = @zeros([timestamp.tsc_size_words]u16);

var ptr_b: [*]f32 = &b;
var ptr_x: [*]f32 = &x;
var ptr_stencil_coeff: [*]f32 = &stencil_coeff;
var ptr_time_buf_u16: [*]u16 = &time_buf_u16;
var ptr_time_ref: [*]u16 = &time_ref_u16;
var ptr_rho: [*]f32 = &rho;
var ptr_xi: [*]f32 = &xi;

// size of local tensor during the BiCGSTAB
var n: i16 = 0;

var mem_b_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> b[i] });
var mem_x_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> x[i] });
var mem_r0_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> r0[i] });
var mem_r_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> r[i] });
var mem_p_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> p[i] });
var mem_v_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> v[i] });
var mem_s_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> s[i] });
var mem_t_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> t[i] });

fn f_enable_timer() void {
    timestamp.enable_tsc();
    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

fn f_tic() void {
    timestamp.get_timestamp(&tscStartBuffer);

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

fn f_toc() void {
    timestamp.get_timestamp(&tscEndBuffer);

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

fn f_memcpy_timestamps() void {

    time_buf_u16[0] = tscStartBuffer[0];
    time_buf_u16[1] = tscStartBuffer[1];
    time_buf_u16[2] = tscStartBuffer[2];

    time_buf_u16[3] = tscEndBuffer[0];
    time_buf_u16[4] = tscEndBuffer[1];
    time_buf_u16[5] = tscEndBuffer[2];

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

// initialization of BiCGSTAB
// - setup the size of local tensor
// - setup the length of all DSDs
//
fn f_bicgstab_init(size:i16) void {

    // setup the size of local tensor
    n = size;

    // set the length of all DSDs
    mem_b_dsd = @set_dsd_length(mem_b_dsd, @bitcast(u16,n));
    mem_x_dsd = @set_dsd_length(mem_x_dsd, @bitcast(u16,n));
    mem_r0_dsd = @set_dsd_length(mem_r0_dsd, @bitcast(u16,n));
    mem_r_dsd = @set_dsd_length(mem_r_dsd, @bitcast(u16,n));
    mem_p_dsd = @set_dsd_length(mem_p_dsd, @bitcast(u16,n));
    mem_v_dsd = @set_dsd_length(mem_v_dsd, @bitcast(u16,n));
    mem_s_dsd = @set_dsd_length(mem_s_dsd, @bitcast(u16,n));
    mem_t_dsd = @set_dsd_length(mem_t_dsd, @bitcast(u16,n));

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

// v = A*x = A*x0
fn f_spmv_Ax() void {
    stencil_mod.spmv(n, &stencil_coeff, &x, &v);
}

// - r0 = b - v = b - A*x0
//   where v = A*x is computed by f_spmv_Ax
// - r = r0
// - p = r0
// - xi = |r0|^2
fn f_residual() void {

    // r0 = b - v
    @fsubs(mem_r0_dsd, mem_b_dsd, mem_v_dsd);

    // r = r0
    @fmovs(mem_r_dsd, mem_r0_dsd);

    // p = r0
    @fmovs(mem_p_dsd, mem_r0_dsd);

    // xi = |r0|^2
    xi[0] = blas_lib.dot(n, &r, &r);
    reduce_mod.allreduce(1, &xi, reduce_mod.TYPE_BINARY_OP.ADD);
}

// rho = xi
fn f_init_rho() void {
    rho[0] = xi[0];

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

// v = A*p
fn f_spmv_Ap() void {
    stencil_mod.spmv(n, &stencil_coeff, &p, &v);
}

// r0_dot_v = np.dot(r0,v)
fn f_r0_dot_v() void {

    // r0_dot_v = dot(r0, v)
    r0_dot_v[0] = blas_lib.dot(n, &r0, &v);
    reduce_mod.allreduce(1, &r0_dot_v, reduce_mod.TYPE_BINARY_OP.ADD);
}

// alpha = rho / (r0, v)
// s = r - alpha*v
// t = A*s
fn f_update_alpha_s_t() void {

    alpha = rho[0] / r0_dot_v[0];

    var alpha_minus: f32 = -alpha;

    // s = r - alpha*v
    @fmacs(mem_s_dsd, mem_r_dsd, mem_v_dsd, alpha_minus);

    // t = A*s
    stencil_mod.spmv(n, &stencil_coeff, &s, &t);
}

// t_dot_s = np.dot(t,s)
fn f_t_dot_s() void {

    t_dot_s[0] = blas_lib.dot(n, &t, &s);
    reduce_mod.allreduce(1, &t_dot_s, reduce_mod.TYPE_BINARY_OP.ADD);
}

// t_dot_t = np.dot(t,t)
fn f_t_dot_t() void {

    t_dot_t[0] = blas_lib.dot(n, &t, &t);
    reduce_mod.allreduce(1, &t_dot_t, reduce_mod.TYPE_BINARY_OP.ADD);
}

// w = (t,s)/(t,t)
// x = x + alpha*p + w*s
// r = s - w*t
// rho_old = rho
// rho = (r0, r)
//
fn f_update_w_x_r_rho() void {

    w = t_dot_s[0]/t_dot_t[0];

    // update x = x + alpha*p + w*s
    // i) x = x + alpha*p
    @fmacs(mem_x_dsd, mem_x_dsd, mem_p_dsd, alpha);
    // ii) x = x + w*s
    @fmacs(mem_x_dsd, mem_x_dsd, mem_s_dsd, w);

    // r = s - w * t
    var w_minus: f32 = -w;
    @fmacs(mem_r_dsd, mem_s_dsd, mem_t_dsd, w_minus);

    // update rho
    rho_old[0] = rho[0];

    // rho = np.dot(r0,r)
    rho[0] = blas_lib.dot(n, &r0, &r);
    reduce_mod.allreduce(1, &rho, reduce_mod.TYPE_BINARY_OP.ADD);
}

// beta = (rho/rho_old)*(alpha/w)
// p = r + beta*(p - w*v)
// xi = np.dot(r,r)
fn f_update_beta_p_xi() void {

    beta = (rho[0]/rho_old[0])*(alpha/w);

    // compute p = r + beta*(p - w*v)
    // i) p = p - w*v
    var w_minus: f32 = -w;
    @fmacs(mem_p_dsd, mem_p_dsd, mem_v_dsd, w_minus);
    // ii) p = r + beta*p
    @fmacs(mem_p_dsd, mem_r_dsd, mem_p_dsd, beta);

    // xi = np.dot(r,r)
    xi[0] = blas_lib.dot(n, &r, &r);
    reduce_mod.allreduce(1, &xi, reduce_mod.TYPE_BINARY_OP.ADD);
}

fn f_sync() void {
   reduce_mod.allreduce(1, &dot, reduce_mod.TYPE_BINARY_OP.ADD);
}

fn f_reference_timestamps() void {

    time_ref_u16[0] = reduce_mod.tscRefBuffer[0];
    time_ref_u16[1] = reduce_mod.tscRefBuffer[1];
    time_ref_u16[2] = reduce_mod.tscRefBuffer[2];

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

comptime {
    @export_symbol(ptr_b, "b");
    @export_symbol(ptr_x, "x");
    @export_symbol(ptr_stencil_coeff, "stencil_coeff");
    @export_symbol(ptr_time_buf_u16, "time_buf_u16");
    @export_symbol(ptr_time_ref, "time_ref");
    @export_symbol(ptr_rho, "rho");
    @export_symbol(ptr_xi, "xi");
}

comptime{
    @export_symbol(f_enable_timer);
    @export_symbol(f_tic);
    @export_symbol(f_toc);
    @export_symbol(f_memcpy_timestamps);

    @export_symbol(f_bicgstab_init);
    @export_symbol(f_spmv_Ax);
    @export_symbol(f_residual);
    @export_symbol(f_init_rho);
    @export_symbol(f_spmv_Ap);
    @export_symbol(f_r0_dot_v);
    @export_symbol(f_update_alpha_s_t);
    @export_symbol(f_t_dot_s);
    @export_symbol(f_t_dot_t);
    @export_symbol(f_update_w_x_r_rho);
    @export_symbol(f_update_beta_p_xi);

    @export_symbol(f_sync);
    @export_symbol(f_reference_timestamps);
}

```

## blas.csl

```csl
const math_lib = @import_module("<math>");

const dummy = @zeros([1]i16);

var mem_x_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> dummy[i] });
var mem_y_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> dummy[i] });

// (alpha, inv_alpha) = approx(x) approximates x by positive alpha such that
//     x = alpha * (x/alpha)
// where alpha = 2^(exp) and (x/alpha) has no precision loss.
//
// If x is a normal number, |x| = 2^(exp) * r, then alpha = 2^(exp)
//
// The purpose of this approximation is for nrm2(x).
// nrm2(x) can hit overflow if we just do square-sum.
// The simple workaround is to square-sum of x/max(x).
// However the division is very expensive, about 50 cycles.
// We just need a number alpha close to max(x) such that x/alpha = O(1).
// The cost of approx is about 11 instructions, much cheaper than div.
//
// Assume x = sign * 2^(E-127) * mantissa, "approx" handles the following
// four cases:
//
// case 1: x is a normal number
//    0 < E < 255
//    x is normal
//    x = sign * 2^(E-127) * 1.b22b21... b1b0
//    min(x) = 0x0080 0000
//           = 2^(-126) = 1.1754943508 x 10^(-38)
//    max(x) = 0x7f7f ffff
//           = 2^127 x (2 - 2^(-23)) = 3.4028234664 * 10^38
//
// case 2: x is a subnormal number
//    E = 0 and mantissa > 0
//    x = sign * 2^(-127) * b22.b21... b1b0
//      = sign * 2^(-126) * 0.b22b21... b1b0
//    min(x) = 0x000 0001
//           = 2^(-126) x 2^(-23) = 2^(-149) = 1.4*10^(-45)
//    max(x) = 007f ffff
//           = 2^(-126) x (1 - 2^(-23)) = 1.17 x 10^(-38)
//
// case 3: x = 0
//    E = 0 and mantissa = 0
//
// case 4: x = inf or nan
//    inf: E = 255 and mantissa = 0
//    nan: E = 255 and mantissa > 0
//
// Example 1: x = 14.0
//    alpha_u32 = 0x41000000
//    inv_alpha_u32 = 0x3e000000
//    alpha = 8.
//    inv_alpha = 0.125
// Example 2: x = 0.15625
//    alpha_u32 = 0x3e000000
//    inv_alpha_u32 = 0x41000000
//    alpha = 0.125
//    inv_alpha = 8.0
// Example 3: x = 1.e-43
//    alpha_u32 = 0x800000
//    inv_alpha_u32 = 0x7e800000
//    alpha = 1.1754943508222875e-38
//    inv_alpha = 8.507059173023462e+37
// Example 4: x = 1.0/0.0 (np.float32(np.inf))
//    alpha_u32 = 0x3f800000
//    inv_alpha_u32 = 0x3f800000
//    alpha = 1.0
//    inv_alpha = 1.0
// Example 5: x = 0.0/0.0 (np.float32(np.nan))
//    alpha_u32 = 0x3f800000
//    inv_alpha_u32 = 0x3f800000
//    alpha = 1.0
//    inv_alpha = 1.0
//
fn approx(x: f32, alpha: *f32, inv_alpha: *f32) void {
   const MASK_EXPONENT: u32 = 0x7F800000;
   const MASK_MANTISSA: u32 = 0x007FFFFF;
   const x_u32: u32 = @bitcast(u32, x);
   // x is presented by (sign | E | mantissa)
   // sign has 1 bit, E has 8 bits and mantissa has 23 bits
   // E = (x & MASK_EXPONEN) >> 23
   const exp: u32 = (x_u32 & MASK_EXPONENT);
   // mantissa = b22b21...b1b0 has 23-bit, need u32
   const mantissa: u32 = (x_u32) & MASK_MANTISSA;
   // E has 8-bit, use u16
   var E: u16 = @as(u16, (exp >> 23));

   // case 1: 0 < E < 255, x is normal
   // the following if-clause handles case 2, 3 and 4
   if (0 == E){
        if (0 == mantissa){
            // case 3: x = 0
            // reset alpha = 1
            E = 127;
        }else{
            // case 2: x is subnormal
            // reset alpha= 2^(-126)
            E = 1;
        }
    }
    if (255 == E){
        // case 4: x = inf or NAN
        // reset alpha = 1
        E = 127;
    }
    // alpha and inv_alpha are u32
    // alpha = 2^(E - 127)
    // inv_alpha = 1/alpha = 2^(127 - E)
    var alpha_u32: u32 = (@as(u32, E) << 23);
    var inv_alpha_u32: u32 = @as(u32, (254 - E)) << 23;

    alpha.* = @bitcast(f32, alpha_u32);
    inv_alpha.* = @bitcast(f32, inv_alpha_u32);
}

// kernel of ymax = max(|y|)
// return max(ymax, |yval|)
fn reduce_fabs(yval : f32, ymax : *f32) f32 {
    var yreg: f32 = math_lib.abs(yval);
    if (yreg > ymax.*){
        return yreg;
    }else{
        return ymax.*;
    }
}

// kernel of sum = reduce( (y/alpha)^2, +)
// return sum + (yval/alpha)**2
fn reduce_scale_square(yval: f32, inv_alpha: f32, sum: *f32) f32 {
    var yreg: f32 = yval * inv_alpha;
    return sum.* + yreg * yreg;
}

// return |y[0:n]|_2
fn nrm2(n:i16, y: [*]f32) f32 {
    var alpha: f32;
    var inv_alpha: f32;

    // step 1: ymax = max(|y|)
    var ymax: f32 = @as(f32,0);
    mem_y_dsd = @set_dsd_base_addr(mem_y_dsd, y);
    mem_y_dsd = @set_dsd_length(mem_y_dsd, @bitcast(u16,n));
    @map(reduce_fabs, mem_y_dsd, &ymax, &ymax);

    // step 2: ymax = alpha * (ymax/alpha)
    approx(ymax, &alpha, &inv_alpha);

    // step 3: sum = reduce( (y/alpha)^2, +)
    var sum: f32 = @as(f32, 0);
    @map(reduce_scale_square, mem_y_dsd, inv_alpha, &sum, &sum);

    // step 4: nrm2 = |y|_2 locally
    sum = math_lib.sqrt(sum);
    return (sum * alpha);
}

// kernel of sum = reduce( (y/alpha)^2, +)
// return sum + (yval/alpha)**2
fn reduce_dot(xval: f32, yval: f32, sum: *f32) f32 {
    return sum.* + xval * yval;
}

// return dot(x,y)
fn dot(n:i16, x: [*]f32, y: [*]f32) f32 {
    mem_x_dsd = @set_dsd_base_addr(mem_x_dsd, x);
    mem_x_dsd = @set_dsd_length(mem_x_dsd, @bitcast(u16,n));
    mem_y_dsd = @set_dsd_base_addr(mem_y_dsd, y);
    mem_y_dsd = @set_dsd_length(mem_y_dsd, @bitcast(u16,n));
    var sum: f32 = @as(f32, 0);
    @map(reduce_dot, mem_x_dsd, mem_y_dsd, &sum, &sum);

    return sum;
}

```

## run.py

```python
#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" test BiCGSTAB of a sparse matrix A built by 7-point stencil

  The following BiCGSTAB algorithm is modified from [1].
  ---
  The algorithm of unpreconditioned BiCGSTAB is
    Given b, x0 and tol = eps*|b|
    k = 0
    x = x0
    r0 = b - A*x
    r = r0
    p = r0
    xi = |r|^2 = (r0, r0)
    rho = xi
    while xi > tol*tol and k < max_ite
        k = k + 1
        v = A*p
        alpha = rho/(r0, v)
        s = r - alpha*v
        t = A*s
        w = (t,s)/(t,t)
        x = x + alpha*p + w*s
        r = s - w*t
        rho_old = rho
        rho = (r0, r)
        beta = (rho/rho_old)*(alpha/w)
        p = r + beta*(p - w*v)
        xi = |r|^2
    end
    x approximates the solution of a linear system Ax = b

  The sparse matrix A is built by a 7-point stenil.
  The 7-point stencil is defined by the following:
  ---
    The Laplacian operator L on 3-dimensional domain can be represented by 7-point
  stencil based on the standard 2nd order Finite Difference Method. The operator form
  with Dirichlet boundary conditions can be written by
         L[u](i,j,k) = u(i+1, j,  k  ) + u(i-1, j,   k  ) +
                       u(i,   j+1,k  ) + u(i,   j-1, k  ) +
                       u(i,   j,  k+1) + u(i,   j,   k-1) +
                      -6*u(i, j, k)
  In general the coefficients of those 7 points can vary. To minimize the memory
  consumption, this example assumes the coefficients are independent of index k and
  whole vector u(i,j,:) is placed in one PE (px=j, py=i).
  The above formula can be re-written by
     c_west   * x[i-1][j  ][k  ] + c_east  * x[i+1][j  ][k  ] +
     c_south  * x[i  ][j-1][k  ] + c_north * x[i  ][j+1][k  ] +
     c_bot    * x[i  ][j  ][k-1] + c_top   * x[i  ][j  ][k+1] +
     c_center * x[i][j][k]
  Each PE only holds 7 coefficients organized by c_west, c_east, c_south, c_north,
  c_bot, c_top and c_center.

  This example provides two modules, one is allreduce and the other is stencil_3d_7pts.
  "allreduce" module can synchronize all PEs to form a reference clock.
  "allreduce" module also computes dot(x,y) over a core rectangle.
  "stencil_3d_7pts" module can compute y = A*x where A is the matrix from 7-point stencil.

  The framework is
  ---
       sync()      // synchronize all PEs to sample the reference clock
       tic()       // record start time
       r = b - A*x
       for k = ...
         v = A*p
         update s
         t = A*s
         update x
         update r
         update xi=(r,r)
         update p
         D2H(xi) to check convergence
       end
       toc()       // record end time
  ---
  This framework does transfer the nrm(r) back to host for each iteration of BiCGSTAB.
  So the I/O pressure is high, not good for performance. device_run.py removes
  this IO pressure.

  The tic() samples "time_start" and toc() samples "time_end". The sync() samples
  "time_ref" which is used to shift "time_start" and "time_end".
  The elapsed time is measured by
       cycles_send = max(time_end) - min(time_start)

  The overall runtime is computed via the following formula
       time_send = (cycles_send / 0.85) *1.e-3 us
  where a PE runs with clock speed 850MHz

  Here is the list of parameters:
    -m=<int> is the height of the core
    -n=<int> is the width of the core
    -k=<int> is size of x and y allocated in the core
    --zDim=<int> is the number of f32 per PE, computed by y = A*x
                 zDim must be not greater than k
    --max-ite=<int> number of iterations
    --channels=<int> specifies the number of I/O channels, no bigger than 16

  WARNING: BiCGSTAB may not converge if rho=(rj, r0) is close to zero.
  If it does happen, we need to restart the BiCGSTAB.
  The algorithm here does not provide the restart.

  Reference:
  [1] H. A. VAN DER VORST, BI-CGSTAB: A FAST AND SMOOTHLY CONVERGING VARIANT
      OF BI-CG FOR THE SOLUTION OF NONSYMMETRIC LINEAR SYSTEMS,
      SIAM J. ScI. STAT. COMPUT. Vol. 13, No. 2, pp. 631-644, March 1992
"""

import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from bicgstab import bicgstab
from cmd_parser import parse_args
from scipy.sparse import linalg as sparse_LA
from util import csr_7_pt_stencil, hwl_2_oned_colmajor, oned_to_hwl_colmajor

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def csl_compile_core(
    cslc: str,
    width: int,  # width of the core
    height: int,  # height of the core
    pe_length: int,
    blockSize: int,
    file_config: str,
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,  # fabric-offsets of the core
    core_fabric_offset_y: int,
    use_precompile: bool,
    arch: Optional[str],
    C0: int,
    C1: int,
    C2: int,
    C3: int,
    C4: int,
    C5: int,
    C6: int,
    C7: int,
    C8: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
):
  if not use_precompile:
    args = []
    args.append(cslc)  # command
    args.append(file_config)
    args.append(f"--fabric-dims={fabric_width},{fabric_height}")
    args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
    args.append(f"--params=width:{width},height:{height},MAX_ZDIM:{pe_length}")
    args.append(f"--params=BLOCK_SIZE:{blockSize}")
    args.append(f"--params=C0_ID:{C0}")
    args.append(f"--params=C1_ID:{C1}")
    args.append(f"--params=C2_ID:{C2}")
    args.append(f"--params=C3_ID:{C3}")
    args.append(f"--params=C4_ID:{C4}")
    args.append(f"--params=C5_ID:{C5}")
    args.append(f"--params=C6_ID:{C6}")
    args.append(f"--params=C7_ID:{C7}")
    args.append(f"--params=C8_ID:{C8}")

    args.append(f"-o={elf_dir}")
    if arch is not None:
      args.append(f"--arch={arch}")
    args.append("--memcpy")
    args.append(f"--channels={channels}")
    args.append(f"--width-west-buf={width_west_buf}")
    args.append(f"--width-east-buf={width_east_buf}")

    print(f"subprocess.check_call(args = {args}")
    subprocess.check_call(args)
  else:
    print("\tuse pre-compile ELFs")

def timing_analysis(height, width, time_memcpy_hwl, time_ref_hwl):
  # time_start = start time of spmv
  time_start = np.zeros((height, width)).astype(int)
  # time_end = end time of spmv
  time_end = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_memcpy_hwl[(h, w, 0)]
      word[1] = time_memcpy_hwl[(h, w, 1)]
      word[2] = time_memcpy_hwl[(h, w, 2)]
      time_start[(h, w)] = make_u48(word)
      word[0] = time_memcpy_hwl[(h, w, 3)]
      word[1] = time_memcpy_hwl[(h, w, 4)]
      word[2] = time_memcpy_hwl[(h, w, 5)]
      time_end[(h, w)] = make_u48(word)

  # time_ref = reference clock
  time_ref = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_ref_hwl[(h, w, 0)]
      word[1] = time_ref_hwl[(h, w, 1)]
      word[2] = time_ref_hwl[(h, w, 2)]
      time_ref[(h, w)] = make_u48(word)

  # adjust the reference clock by the propagation delay
  # the right-bottom PE signals other PEs, the propagation delay is
  #     (h-1) - py + (w-1) - px
  for py in range(height):
    for px in range(width):
      time_ref[(py, px)] = time_ref[(py, px)] - ((width + height - 2) - (px + py))

  # shift time_start and time_end by time_ref
  time_start = time_start - time_ref
  time_end = time_end - time_ref

  # cycles_send = time_end[(h,w)] - time_start[(h,w)]
  # 850MHz --> 1 cycle = (1/0.85) ns = (1/0.85)*1.e-3 us
  # time_send = (cycles_send / 0.85) *1.e-3 us
  #
  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) * 1.0e-3
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")

# How to compile
#   python run.py -m=5 -n=5 -k=5 --latestlink latest --channels=1 \
#   --width-west-buf=0 --width-east-buf=0 --compile-only
# How to run
#   python run.py -m=5 -n=5 -k=5 --latestlink latest --channels=1 \
#   --width-west-buf=0 --width-east-buf=0 --run-only --zDim=5 --max-ite=1
def main():
  """Main method to run the example code."""

  random.seed(127)

  args, dirname = parse_args()

  cslc = "cslc"
  if args.driver is not None:
    cslc = args.driver

  print(f"cslc = {cslc}")

  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  channels = args.channels
  assert channels <= 16, "only support up to 16 I/O channels"
  assert channels >= 1, "number of I/O channels must be at least 1"

  print(f"width_west_buf = {width_west_buf}")
  print(f"width_east_buf = {width_east_buf}")
  print(f"channels = {channels}")

  height = args.m
  width = args.n
  pe_length = args.k
  zDim = args.zDim
  blockSize = args.blockSize
  max_ite = args.max_ite

  print(
      f"width={width}, height={height}, pe_length={pe_length}, zDim={zDim}, blockSize={blockSize}"
  )
  print(f"max_ite = {max_ite}")
  assert pe_length >= 2, "the maximum size of z must be greater than 1"
  assert zDim <= pe_length, "[0, zDim) cannot exceed the storage"

  np.random.seed(2)
  x = np.zeros((height, width, pe_length), dtype=np.float32)
  #x = (np.arange(height * width * zDim).reshape(height, width, zDim).astype(np.float32) + 100)

  x_1d = hwl_2_oned_colmajor(height, width, zDim, x, np.float32)
  nrm2_x = np.linalg.norm(x_1d.ravel(), 2)
  # |x0|_2 = 1
  if nrm2_x > 0:
    x_1d = x_1d / nrm2_x
    x = x / nrm2_x

  b = (np.arange(height * width * pe_length).reshape(height, width, pe_length).astype(np.float32) +
       1)
  b_1d = hwl_2_oned_colmajor(height, width, pe_length, b, np.float32)

  # stencil coefficients has the following order
  # {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
  # BiCGSTAB does not converge, it hits serious breakdown when
  # rho = (r0, rj) is close to zero.
  # Such behavior is confirmed by scipy.sparse.linalg.bicgstab
  stencil_coeff = np.zeros((height, width, 7), dtype=np.float32)
  for i in range(height):
    for j in range(width):
      stencil_coeff[(i, j, 0)] = -1  # west
      stencil_coeff[(i, j, 1)] = -1  # east
      stencil_coeff[(i, j, 2)] = -1  # south
      stencil_coeff[(i, j, 3)] = -1  # north
      stencil_coeff[(i, j, 4)] = -1  # bottom
      stencil_coeff[(i, j, 5)] = -1  # top
      stencil_coeff[(i, j, 6)] = 6 + j  # center

  # fabric-offsets = 1,1
  fabric_offset_x = 1
  fabric_offset_y = 1
  # starting point of the core rectangle = (core_fabric_offset_x, core_fabric_offset_y)
  # memcpy framework requires 3 columns at the west of the core rectangle
  # memcpy framework requires 2 columns at the east of the core rectangle
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y
  # (min_fabric_width, min_fabric_height) is the minimal dimension to run the app
  min_fabric_width = core_fabric_offset_x + width + 2 + 1 + width_east_buf
  min_fabric_height = core_fabric_offset_y + height + 1

  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)

  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height

  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  # prepare the simulation
  print("store ELFs and log files in the folder ", dirname)

  # layout of a rectangle
  code_csl = "layout.csl"

  C0 = 0
  C1 = 1
  C2 = 2
  C3 = 3
  C4 = 4
  C5 = 5
  C6 = 6
  C7 = 7
  C8 = 8

  csl_compile_core(
      cslc,
      width,
      height,
      pe_length,
      blockSize,
      code_csl,
      dirname,
      fabric_width,
      fabric_height,
      core_fabric_offset_x,
      core_fabric_offset_y,
      args.run_only,
      args.arch,
      C0,
      C1,
      C2,
      C3,
      C4,
      C5,
      C6,
      C7,
      C8,
      channels,
      width_west_buf,
      width_east_buf,
  )
  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  A_csr = csr_7_pt_stencil(stencil_coeff, height, width, zDim)
  [nrows, ncols] = A_csr.shape
  nnz = A_csr.nnz
  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros")

  if nrows < 1000:
    A = A_csr.todense()
    # s is singular value in descending order
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    min_svd = min(s)
    max_svd = max(s)
    print(f"min(svd) = {min_svd}")
    print(f"max(svd) = {max_svd}")
    print(f"cond(A) = {max_svd/min_svd}")

  nrm_b = np.linalg.norm(b_1d.ravel(), 2)
  eps = 1.0e-3
  tol = eps * nrm_b
  print(f"|b| = {nrm_b}")
  print(f"max_ite = {max_ite}")
  print(f"eps = {eps}")
  print(f"tol = {tol}")

  xf_1d, xi, k = bicgstab(A_csr, x_1d, b_1d, max_ite, tol)
  print(f"[host] after BiCGSTAB, xi = {xi}, k = {k}")

  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  simulator = SdkRuntime(dirname, cmaddr=args.cmaddr)

  symbol_b = simulator.get_id("b")
  symbol_x = simulator.get_id("x")
  symbol_xi = simulator.get_id("xi")
  symbol_stencil_coeff = simulator.get_id("stencil_coeff")
  symbol_time_buf_u16 = simulator.get_id("time_buf_u16")
  symbol_time_ref = simulator.get_id("time_ref")

  simulator.load()
  simulator.run()

  print("copy vector b and x0")
  simulator.memcpy_h2d(
      symbol_b,
      b_1d,
      0,
      0,
      width,
      height,
      zDim,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  simulator.memcpy_h2d(
      symbol_x,
      x_1d,
      0,
      0,
      width,
      height,
      zDim,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  print("copy 7 stencil coefficients")
  stencil_coeff_1d = hwl_2_oned_colmajor(height, width, 7, stencil_coeff, np.float32)
  simulator.memcpy_h2d(
      symbol_stencil_coeff,
      stencil_coeff_1d,
      0,
      0,
      width,
      height,
      7,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  print("step 0: enable timer")
  simulator.launch("f_enable_timer", nonblock=False)

  print("step 1: sync all PEs")
  simulator.launch("f_sync", nonblock=False)

  print("step 2: copy reference clock from reduce module")
  simulator.launch("f_reference_timestamps", nonblock=False)

  print("step 3: tic() records time_start")
  simulator.launch("f_tic", nonblock=True)

  print(f"step 4: BiCGSTAB with max_ite = {max_ite}, zDim = {zDim}")

  print("step 4.1: initialization")
  # - setup the length of all DSDs
  # - setup the size of local tensor
  simulator.launch("f_bicgstab_init", np.int16(zDim), nonblock=False)

  k = 0
  print("step 4.2: compute r0=b-A*x0, r=r0, p=r0, and rho=xi=|r0|^2")
  # v = A*x0
  simulator.launch("f_spmv_Ax", nonblock=False)
  # r0 = b - v = b - A*x0
  # r = r0
  # p = r0
  # xi = |r0|^2
  simulator.launch("f_residual", nonblock=False)

  # rho = xi = (r0, r0)
  simulator.launch("f_init_rho", nonblock=False)

  # [optional] D2H(xi)
  xi_wse = np.zeros(1, np.float32)
  simulator.memcpy_d2h(
      xi_wse,
      symbol_xi,
      0,
      0,
      1,
      1,
      1,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  xi = xi_wse[0]
  print(f"[BiCGSTAB] iter {k}: xi = {xi}")
  # if |r| < tol, then exit
  while (xi > tol * tol) and (k < max_ite):
    k = k + 1
    print("step 4.3: v = A*p")
    simulator.launch("f_spmv_Ap", nonblock=False)

    print("step 4.4: compute (r0, v)")
    # r0_dot_v = dot(r0, v)
    simulator.launch("f_r0_dot_v", nonblock=False)

    print("step 4.4: update alpha, s and t")
    # alpha = rho / (r0, v)
    # s = r - alpha*v
    # t = A*s
    simulator.launch("f_update_alpha_s_t", nonblock=False)

    print("step 4.5: compute (t, s)")
    # t_dot_s = np.dot(t,s)
    simulator.launch("f_t_dot_s", nonblock=False)

    print("step 4.6: compute (t, t)")
    # t_dot_t = np.dot(t,t)
    simulator.launch("f_t_dot_t", nonblock=False)

    print("step 4.7: update w, x, r and rho ")
    # w = (t,s)/(t,t)
    # x = x + alpha*p + w*s
    # r = s - w*t
    # rho_old = rho
    # rho = (r0, r)
    simulator.launch("f_update_w_x_r_rho", nonblock=False)

    # beta = (rho/rho_old)*(alpha/w)
    # p = r + beta*(p - w*v)
    # xi = np.dot(r,r)
    simulator.launch("f_update_beta_p_xi", nonblock=False)

    # [optional] D2H(xi)
    simulator.memcpy_d2h(
        xi_wse,
        symbol_xi,
        0,
        0,
        1,
        1,
        1,
        streaming=False,
        data_type=memcpy_dtype,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=False,
    )
    xi = xi_wse[0]
    print(f"[BiCGSTAB] iter {k}: xi = {xi}")

  print("step 5: toc() records time_end")
  simulator.launch("f_toc", nonblock=False)

  print("step 6: prepare (time_start, time_end)")
  simulator.launch("f_memcpy_timestamps", nonblock=False)

  print("step 7: D2H (time_start, time_end)")
  time_memcpy_hwl_1d = np.zeros(height * width * 6, np.uint32)
  simulator.memcpy_d2h(
      time_memcpy_hwl_1d,
      symbol_time_buf_u16,
      0,
      0,
      width,
      height,
      6,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_16BIT,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  time_memcpy_hwl = oned_to_hwl_colmajor(height, width, 6, time_memcpy_hwl_1d, np.uint16)

  print("step 8: D2H reference clock")
  time_ref_1d = np.zeros(height * width * 3, np.uint32)
  simulator.memcpy_d2h(
      time_ref_1d,
      symbol_time_ref,
      0,
      0,
      width,
      height,
      3,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_16BIT,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  time_ref_hwl = oned_to_hwl_colmajor(height, width, 3, time_ref_1d, np.uint16)

  print("step 9: D2H x[zDim]")
  xf_wse_1d = np.zeros(height * width * zDim, np.float32)
  simulator.memcpy_d2h(
      xf_wse_1d,
      symbol_x,
      0,
      0,
      width,
      height,
      zDim,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )

  simulator.stop()

  if args.cmaddr is None:
    # move simulation log and core dump to the given folder
    dst_log = Path(f"{dirname}/sim.log")
    src_log = Path("sim.log")
    if src_log.exists():
      shutil.move(src_log, dst_log)

    dst_trace = Path(f"{dirname}/simfab_traces")
    src_trace = Path("simfab_traces")
    if dst_trace.exists():
      shutil.rmtree(dst_trace)
    if src_trace.exists():
      shutil.move(src_trace, dst_trace)

  timing_analysis(height, width, time_memcpy_hwl, time_ref_hwl)

  nrm2_xf = np.linalg.norm(xf_wse_1d.ravel(), 2)
  print(f"|xf|_2 = {nrm2_xf}")
  print(f"|b - A*xf|_2 = {math.sqrt(xi)}")
  print(f"number of iterations = {k}")

  z = xf_1d.ravel() - xf_wse_1d.ravel()
  nrm_z = np.linalg.norm(z, np.inf)
  print(f"|xf_ref - xf_wse| = {nrm_z}")
  np.testing.assert_allclose(xf_1d.ravel(), xf_wse_1d.ravel(), 1.0e-5)
  print("\nSUCCESS!")

  print("==== measure the residual b - A*x by |.|_inf")
  # r = b - A*xf
  r = b_1d - A_csr.dot(xf_wse_1d)
  nrm_r = np.linalg.norm(r, np.inf)
  nrm_b = np.linalg.norm(b_1d, np.inf)
  nrm_x = np.linalg.norm(xf_wse_1d, np.inf)
  nrm_A = sparse_LA.norm(A_csr, np.inf)
  print(f"|xf| = {nrm_x}")
  print(f"|b| = {nrm_b}")
  print(f"|A| = {nrm_A}")
  print(f"|b - A*xf| = {nrm_r}")
  rel_err = nrm_r / (nrm_A * nrm_x + nrm_b)
  print(f"|b-A*x|/(|A|*|x|+|b|) = {rel_err}")
  # only check the relative error if max_ite is big enough
  # i.e. BiCGSTAB converges before reaching max_ite
  if k < max_ite:
    assert eps > rel_err, "relative error is too large"

if __name__ == "__main__":
  main()

```

## cmd\_parser.py

```python
"""command parser for bandwidthTest

   -m <int>     number of rows of the core rectangle
   -n <int>     number of columns of the core rectangle
   -k <int>     number of elements of local tensor
   --zDim <int>   number of elements to compute y=A*x
   --blockSize <int>  the size of temporary buffers for communication
   --latestlink   working directory
   --driver     path to CSL compiler
   --fabric-dims  fabric dimension of a WSE
   --cmaddr       IP address of a WSE
   --channels        number of I/O channels, 1 <= channels <= 16
   --width-west-buf  number of columns of the buffer in the west of the core rectangle
   --width-east-buf  number of columns of the buffer in the east of the core rectangle
   --compile-only    compile ELFs
   --run-only        run the test with precompiled binary
"""

import os
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", default=1, type=int, help="number of rows")
  parser.add_argument("-n", default=1, type=int, help="number of columns")
  parser.add_argument("-k", default=1, type=int, help="size of local tensor, no less than 2")
  parser.add_argument("--zDim", default=2, type=int, help="[0 zDim-1) is the domain of Laplacian")
  parser.add_argument(
      "--max-ite",
      default=1,
      type=int,
      help="maximum number of iterations of power method",
  )
  parser.add_argument("--latestlink", help="folder to contain the log files (default: latest)")
  parser.add_argument("-d", "--driver", help="The path to the CSL compiler")
  parser.add_argument("--compile-only", help="Compile only", action="store_true")
  parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")
  parser.add_argument("--cmaddr", help="CM address and port, i.e. <IP>:<port>")
  parser.add_argument("--run-only", help="Run only", action="store_true")
  parser.add_argument("--arch", help="wse2 or wse3. Default is wse2 when not supplied.")
  parser.add_argument("--width-west-buf", default=0, type=int, help="width of west buffer")
  parser.add_argument("--width-east-buf", default=0, type=int, help="width of east buffer")
  parser.add_argument(
      "--channels",
      default=1,
      type=int,
      help="number of I/O channels, between 1 and 16",
  )
  parser.add_argument(
      "--blockSize",
      default=2,
      type=int,
      help="the size of temporary buffers for communication",
  )

  args = parser.parse_args()

  logs_dir = "latest"
  if args.latestlink:
    logs_dir = args.latestlink

  dir_exist = os.path.isdir(logs_dir)
  if dir_exist:
    print(f"{logs_dir} already exists")
  else:
    print(f"create {logs_dir} to store log files")
    os.mkdir(logs_dir)

  return args, logs_dir

```

## util.py

```python
import numpy as np
from scipy.sparse import coo_matrix

def COL_MAJOR(h, w, l, height, width, pe_length):
  assert 0 <= h < height
  assert 0 <= w < width
  assert 0 <= l < pe_length

  return h + w * height + l * height * width

def hwl_2_oned_colmajor(height: int, width: int, pe_length: int, A_hwl: np.ndarray, dtype):
  """
    Given a 3-D tensor A[height][width][pe_length], transform it to
    1D array by column-major
    """
  A_1d = np.zeros(height * width * pe_length, dtype)
  idx = 0
  for l in range(pe_length):
    for w in range(width):
      for h in range(height):
        A_1d[idx] = A_hwl[(h, w, l)]
        idx = idx + 1
  return A_1d

def oned_to_hwl_colmajor(height: int, width: int, pe_length: int, A_1d: np.ndarray, dtype):
  """
    Given a 1-D tensor A_1d[height*width*pe_length], transform it to
    3-D tensor A[height][width][pe_length] by column-major
    """
  if dtype == np.float32:
    # only support f32 to f32
    assert A_1d.dtype == np.float32, "only support f32 to f32"
    A_hwl = np.reshape(A_1d, (height, width, pe_length), order="F")

  elif dtype == np.uint16:
    # only support u32 to u16 by dropping upper 16-bit
    assert A_1d.dtype == np.uint32, "only support u32 to u16"
    A_hwl = np.zeros((height, width, pe_length), dtype)
    idx = 0
    for l in range(pe_length):
      for w in range(width):
        for h in range(height):
          x = A_1d[idx]
          x = x & 0x0000FFFF  # drop upper 16-bit
          A_hwl[(h, w, l)] = np.uint16(x)
          idx = idx + 1
  else:
    raise RuntimeError(f"{dtype} is not supported")

  return A_hwl

#  y = Laplacian(x) for z=0,1,..,zDim-1
#
# The capacity of x and y can be bigger than zDim, but the physical domain is [0,zDim)
#
# The coordinates of physical domain are x,y,z.
# The physical layout of WSE is width, height.
# To avoid confusion, the kernel is written based on the layout of
# WSE, not physical domain of the application.
# For example, the user can match x-coordinate to x direction of
# WSE and y-coordinate to y-direction of WSE.
#              x-coord
#            +--------+
#    y-coord |        |
#            +--------+
#
# The stencil coefficients "stencil_coeff" can vary along x-y direction,
# but universal along z-direction. Each PE can have seven coefficents,
# west, east, south, north, bottom, top and center.
#
# Input:
#   stencil_coeff: size is (h,w,7)
#   x: size is (h,w,l)
# Output:
#   y: size is (h,w,l)
#
def laplacian(stencil_coeff, zDim, x, y):
  (height, width, pe_length) = x.shape
  assert zDim <= pe_length
  # y and x must have the same dimensions
  (m, n, k) = y.shape
  assert m == height
  assert n == width
  assert pe_length == k
  # stencil_coeff must be (h,w,7)
  (m, n, k) = stencil_coeff.shape
  assert m == height
  assert n == width
  assert k == 7

  #          North
  #           j
  #        +------+
  # West i |      | East
  #        +------+
  #          south
  for i in range(height):
    for j in range(width):
      for k in range(zDim):
        c_west = stencil_coeff[(i, j, 0)]
        c_east = stencil_coeff[(i, j, 1)]
        c_south = stencil_coeff[(i, j, 2)]
        c_north = stencil_coeff[(i, j, 3)]
        c_bottom = stencil_coeff[(i, j, 4)]
        c_top = stencil_coeff[(i, j, 5)]
        c_center = stencil_coeff[(i, j, 6)]

        west_buf = 0  # x[(i,-1,k)]
        if j > 0:
          west_buf = x[(i, j - 1, k)]
        east_buf = 0  # x[(i,w,k)]
        if j < width - 1:
          east_buf = x[(i, j + 1, k)]
        north_buf = 0
        # x[(-1,j,k)]
        if i > 0:
          north_buf = x[(i - 1, j, k)]
        south_buf = 0  # x[(h,j,k)]
        if i < height - 1:
          south_buf = x[(i + 1, j, k)]
        bottom_buf = 0  # x[(i,j,-1)]
        if k > 0:
          bottom_buf = x[(i, j, k - 1)]
        top_buf = 0  # x[(i,j,l)]
        if k < zDim - 1:
          top_buf = x[(i, j, k + 1)]
        center_buf = x[(i, j, k)]
        y[(i, j, k)] = (c_west * west_buf + c_east * east_buf + c_south * south_buf +
                        c_north * north_buf + c_bottom * bottom_buf + c_top * top_buf +
                        c_center * center_buf)

# Given a 7-point stencil, generate sparse matrix A.
# A is represented by CSR.
# The order of grids is column-major
def csr_7_pt_stencil(stencil_coeff, height, width, pe_length):
  # stencil_coeff must be (h,w,7)
  (m, n, k) = stencil_coeff.shape
  assert m == height
  assert n == width
  assert k == 7

  N = height * width * pe_length

  # each point has 7 coefficents at most
  cooRows = np.zeros(7 * N, np.int32)
  cooCols = np.zeros(7 * N, np.int32)
  cooVals = np.zeros(7 * N, np.float32)

  #          North
  #           j
  #        +------+
  # West i |      | East
  #        +------+
  #          south
  nnz = 0
  for i in range(height):
    for j in range(width):
      for k in range(pe_length):
        c_west = stencil_coeff[(i, j, 0)]
        c_east = stencil_coeff[(i, j, 1)]
        c_south = stencil_coeff[(i, j, 2)]
        c_north = stencil_coeff[(i, j, 3)]
        c_bottom = stencil_coeff[(i, j, 4)]
        c_top = stencil_coeff[(i, j, 5)]
        c_center = stencil_coeff[(i, j, 6)]

        center_idx = COL_MAJOR(i, j, k, height, width, pe_length)
        cooRows[nnz] = center_idx
        cooCols[nnz] = center_idx
        cooVals[nnz] = c_center
        nnz += 1
        # west_buf = 0 # x[(i,-1,k)]
        if j > 0:
          west_idx = COL_MAJOR(i, j - 1, k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = west_idx
          cooVals[nnz] = c_west
          nnz += 1
        # east_buf = 0  # x[(i,w,k)]
        if j < width - 1:
          east_idx = COL_MAJOR(i, j + 1, k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = east_idx
          cooVals[nnz] = c_east
          nnz += 1
        # north_buf = 0; # x[(-1,j,k)]
        if i > 0:
          north_idx = COL_MAJOR(i - 1, j, k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = north_idx
          cooVals[nnz] = c_north
          nnz += 1
        # south_buf = 0  # x[(h,j,k)]
        if i < height - 1:
          south_idx = COL_MAJOR(i + 1, j, k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = south_idx
          cooVals[nnz] = c_south
          nnz += 1
        # bottom_buf = 0 # x[(i,j,-1)]
        if k > 0:
          bottom_idx = COL_MAJOR(i, j, k - 1, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = bottom_idx
          cooVals[nnz] = c_bottom
          nnz += 1
        # top_buf = 0    # x[(i,j,l)]
        if k < pe_length - 1:
          top_idx = COL_MAJOR(i, j, k + 1, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = top_idx
          cooVals[nnz] = c_top
          nnz += 1

  A_coo = coo_matrix((cooVals, (cooRows, cooCols)), shape=(N, N))

  A_csr = A_coo.tocsr(copy=True)
  # sort column indices
  A_csr = A_csr.sorted_indices().astype(np.float32)
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  return A_csr

```

## bicgstab.py

```python
import numpy as np

# solve a linear system A * x = b
# where A is a unsymmetric matrix
#
# The algorithm is modified from
#  H. A. VAN DER VORST, BI-CGSTAB: A FAST AND SMOOTHLY CONVERGING VARIANT
#  OF BI-CG FOR THE SOLUTION OF NONSYMMETRIC LINEAR SYSTEMS,
#  SIAM J. ScI. STAT. COMPUT. Vol. 13, No. 2, pp. 631-644, March 1992
#
# Input
#  A_csr     sparse matrix of type scipy.sparse.csr_matrix
#  x0        initial guess, could be a random vector or the approximated solution
#            of some other iterative solver
#  b         right-hand-side vector
#  max_ite   maximum number of iterations
#  tol       tolerance to stop the algorithm
#            the bigger, the more iterations
#            usually tol = eps * |b| where eps > 1.e-6 for f32
# Output
#  x         approximated solution of A*x=b
#  xi        |b - A*x|^2
#  k         the number of iterations
#
# BiCGSTAB does not converge, it hits serious breakdown when
# rho = (r0, rj) is close to zero.
# Such behavior is confirmed by scipy.sparse.linalg.bicgstab
def bicgstab(A_csr, x0, b, max_ite, tol):
  # storage
  #  b: right-hand-side, read-nly
  #  r0: read-only
  #  x: initial value x0
  #  r: r = b-A*x
  #  p
  #  v
  #  s: can align with r
  #  t
  # To save more space, the input b can be a temporary buffer r
  # Exit: x0 contains the final result, b contains the residual
  k = 0
  # x = x0
  x = np.copy(x0)
  # r0 = b - A*x0
  y = A_csr.dot(x)
  r0 = b - y
  # r = r0
  r = np.copy(r0)
  # p = r0
  p = np.copy(r)
  # xi = |r0|^2
  xi = np.dot(r, r)
  # rho = (r0, r0)
  rho = xi
  print(f"[bicgstab] iter {k}: xi = {xi}")
  # if |r_k|_2 < tol, then exit
  while (xi > tol * tol) and (k < max_ite):
    k = k + 1
    # v = A*p
    v = A_csr.dot(p)
    # alpha = rho / (r0, v)
    r0_dot_v = np.dot(r0, v)
    alpha = rho / r0_dot_v
    # s = r - alpha*v
    s = r - alpha * v
    # t = A*s
    t = A_csr.dot(s)
    # w = (t,s)/(t,t)
    t_dot_s = np.dot(t, s)
    t_dot_t = np.dot(t, t)
    w = t_dot_s / t_dot_t
    # x = x + alpha*p + w*s
    x = x + alpha * p + w * s
    # r = s - w*t
    r = s - w * t
    # update rho
    rho_old = rho
    # rho = (r0, r)
    rho = np.dot(r0, r)
    # beta = (rho/rho_old)*(alpha/w)
    beta = (rho / rho_old) * (alpha / w)
    # p = r + beta*(p - w*v)
    p = r + beta * (p - w * v)
    # update the residual
    xi = np.dot(r, r)
    # serious breakdown when rho = (rj, r0) is close to zero
    # if such scenario happens, we need to restart
    print(f"[bicgstab] iter {k}: xi = {xi}, rho = {rho}")
    if abs(rho) < 1.0e-4:
      print(f"WARNING: breakdown due to tiny rho ({rho}), restart")
      break
  return x, xi, k

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc ./src/layout.csl --arch wse3 --fabric-dims=12,7 --fabric-offsets=4,1 \
--params=width:5,height:5,MAX_ZDIM:5 --params=BLOCK_SIZE:2 --params=C0_ID:0 \
--params=C1_ID:1 --params=C2_ID:2 --params=C3_ID:3 --params=C4_ID:4 --params=C5_ID:5 \
--params=C6_ID:6 --params=C7_ID:7 --params=C8_ID:8 -o=out \
--memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
cs_python ./run.py -m=5 -n=5 -k=5 --latestlink out --channels=1 \
--width-west-buf=0 --width-east-buf=0 --zDim=5 --run-only --max-ite=2

```

---

## layout\_bicgstab.csl

```csl
// color map: memcpy + allreduce + stencil
//
// color  var   color  var        color  var              color  var
//   0   C0       9                18    EN_REDUCE_2       27   reserved (memcpy)
//   1   C1      10                19    EN_REDUCE_3       28   reserved (memcpy)
//   2   C2      11                20    EN_REDUCE_4       29   reserved (memcpy)
//   3   C3      12  STATE         21    reserved (memcpy) 30   reserved (memcpy)
//   4   C4      13                22    reserved (memcpy) 31   reserved
//   5   C5      14  EN_STENCIL_1  23    reserved (memcpy) 32
//   6   C6      15  EN_STENCIL_2  24                      33
//   7   C7      16  EN_STENCIL_3  25                      34
//   8   C8      17  EN_REDUCE_1   26                      35
//

// c0,c1,c2,c3,c4,c5,c6,c7 are internal colors of 7-point stencil
param C0_ID: i16;
param C1_ID: i16;
param C2_ID: i16;
param C3_ID: i16;
param C4_ID: i16;
param C5_ID: i16;
param C6_ID: i16;
param C7_ID: i16;
// c8 is an internal color of allreduce
param C8_ID: i16;

param MAX_ZDIM: i16; // maximum size of local vector x and y
param width: i16 ; // width of the core
param height: i16 ; // height of the core

param BLOCK_SIZE: i16; // size of temporary buffers for communication

const C0: color = @get_color(C0_ID);
const C1: color = @get_color(C1_ID);
const C2: color = @get_color(C2_ID);
const C3: color = @get_color(C3_ID);
const C4: color = @get_color(C4_ID);
const C5: color = @get_color(C5_ID);
const C6: color = @get_color(C6_ID);
const C7: color = @get_color(C7_ID);
const C8: color = @get_color(C8_ID);

// entrypoint of state machine in BiCGSTAB
const STATE: local_task_id = @get_local_task_id(12);

// entrypoints of 7-point stenil
const EN_STENCIL_1: local_task_id = @get_local_task_id(14);
const EN_STENCIL_2: local_task_id = @get_local_task_id(15);
const EN_STENCIL_3: local_task_id = @get_local_task_id(16);

// entrypoints of allreduce
const EN_REDUCE_1: local_task_id = @get_local_task_id(17);
const EN_REDUCE_2: local_task_id = @get_local_task_id(18);
const EN_REDUCE_3: local_task_id = @get_local_task_id(19);
const EN_REDUCE_4: local_task_id = @get_local_task_id(20);

const stencil = @import_module( "../../benchmark-libs/stencil_3d_7pts/layout.csl", .{
    .colors = [8]color{C0, C1, C2, C3, C4, C5, C6, C7},
    .entrypoints = [3]local_task_id{EN_STENCIL_1, EN_STENCIL_2, EN_STENCIL_3},
    .width = width,
    .height = height
    });

const reduce = @import_module( "../../benchmark-libs/allreduce/layout.csl", .{
    .colors = [1]color{C8},
    .entrypoints = [4]local_task_id{EN_REDUCE_1, EN_REDUCE_2, EN_REDUCE_3, EN_REDUCE_4},
    .width = width,
    .height = height
    });

const memcpy = @import_module( "<memcpy/get_params>", .{
    .width = width,
    .height = height,
    });

layout{

    @comptime_assert(C0_ID < C1_ID);
    @comptime_assert(C1_ID < C2_ID);
    @comptime_assert(C2_ID < C3_ID);
    @comptime_assert(C3_ID < C4_ID);
    @comptime_assert(C4_ID < C5_ID);
    @comptime_assert(C5_ID < C6_ID);
    @comptime_assert(C6_ID < C7_ID);
    @comptime_assert(C7_ID < C8_ID);

    // step 1: configure the rectangle which does not include halo
    @set_rectangle( width, height );

    // step 2: compile csl code for a set of PEx.y and generate out_x_y.elf
    //   format: @set_tile_code(x, y, code.csl, param_binding);

    var py: i16 = 0;
    while(py < height) : (py +=1) {
        var px: i16 = 0;
        while(px < width) : (px +=1) {

            const memcpyParams = memcpy.get_params(px);
            const stencilParams = stencil.get_params(px, py);
            const reduceParams = reduce.get_params(px, py);
            var params: comptime_struct = .{
                .memcpyParams = memcpyParams,
                .reduceParams = reduceParams,
                .MAX_ZDIM = MAX_ZDIM,
                .BLOCK_SIZE = BLOCK_SIZE,
                .STATE = STATE,
                .stencilParams = stencilParams
            };

            @set_tile_code(px, py, "kernel_bicgstab.csl", params);
        }
    }

    @export_name("b", [*]f32, true);
    @export_name("x", [*]f32, true);
    @export_name("stencil_coeff", [*]f32, true);
    @export_name("time_buf_u16", [*]u16, true);
    @export_name("time_ref", [*]u16, true);
    @export_name("rho", [*]f32, true);
    @export_name("xi", [*]f32, true);

    @export_name("f_enable_timer", fn()void);
    @export_name("f_tic", fn()void);
    @export_name("f_toc", fn()void);
    @export_name("f_memcpy_timestamps", fn()void);

    @export_name("f_bicgstab", fn(i16,f32,i16)void);

    @export_name("f_sync", fn()void);
    @export_name("f_reference_timestamps", fn()void);
} // end of layout

```

## kernel\_bicgstab.csl

```csl
param memcpyParams: comptime_struct;

param reduceParams: comptime_struct;

param stencilParams: comptime_struct;

param MAX_ZDIM: i16; // size of vector x

param BLOCK_SIZE: i16; // size of temporary buffers for communication

param STATE: local_task_id;

const timestamp = @import_module("<time>");

const math_lib = @import_module("<math>");

const blas_lib = @import_module("blas.csl");

// memcpy module reserves
// - input/output queue 0 and 1
const sys_mod = @import_module( "<memcpy/memcpy>", memcpyParams);

// allreduce uses input queue/output queue 1
const reduce_mod = @import_module( "../../benchmark-libs/allreduce/pe.csl", @concat_structs(reduceParams, .{
     .f_callback = f_trigger_state_machine,
     .queues = [1]u16{2},
     .dest_dsr_ids = [1]u16{1},
     .src0_dsr_ids = [1]u16{1},
     .src1_dsr_ids = [1]u16{1}
     }));

// output queue cannot overlap input queues
const stencil_mod = @import_module( "../../benchmark-libs/stencil_3d_7pts/pe.csl", @concat_structs(stencilParams, .{
     .f_callback = f_trigger_state_machine,
     .input_queues = [4]u16{4, 5, 6, 7},
     .output_queues = if (@is_arch("wse3")) [4]u16{4, 5, 6, 7} else [1]u16{3},
     .output_ut_id = 3,
     .BLOCK_SIZE = BLOCK_SIZE,
     .dest_dsr_ids = [2]u16{2,3},
     .src0_dsr_ids = [1]u16{2},
     .src1_dsr_ids = [2]u16{2,3}
     }));

// tsc_size_words = 3
// starting time of H2D/D2H
var tscStartBuffer = @zeros([timestamp.tsc_size_words]u16);
// ending time of H2D/D2H
var tscEndBuffer = @zeros([timestamp.tsc_size_words]u16);

var b = @zeros([MAX_ZDIM]f32); // right-hand-side
var x = @zeros([MAX_ZDIM]f32); // approximated solution

var r0 = @zeros([MAX_ZDIM]f32); // initial residual
var r = @zeros([MAX_ZDIM]f32); // residual
var p = @zeros([MAX_ZDIM]f32); // Krylov space
var v = @zeros([MAX_ZDIM]f32); // Krylov space
var s = @zeros([MAX_ZDIM]f32); // can align with r
var t = @zeros([MAX_ZDIM]f32); // t = A * s

var dot = @zeros([1]f32); // dummy variable for f_sync
var xi = @zeros([1]f32);
var rho = @zeros([1]f32);
var rho_old = @zeros([1]f32);
var r0_dot_v = @zeros([1]f32);
var t_dot_s = @zeros([1]f32);
var t_dot_t = @zeros([1]f32);
var w: f32 = @as(f32,0); // w = (t,s)/(t,t)
var alpha: f32 = @as(f32,0); // alpha = rho/(r0, v)
var beta: f32 = @as(f32,0); // beta = (rho/rho_old)*(alpha/w)

// stencil coefficients are organized as
// {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
//
// The formula is
//    c_west * x[i-1][j][k] + c_east * x[i+1][j][k] +
//    c_south * x[i][j-1][k] + c_north * x[i][j+1][k] +
//    c_bottom * x[i][j][k-1] + c_top * x[i][j][k+1] +
//    c_center * x[i][j][k]
var stencil_coeff = @zeros([7]f32);

// time_buf_u16[0:5] = {tscStartBuffer, tscEndBuffer}
var time_buf_u16 = @zeros([timestamp.tsc_size_words*2]u16);

// reference clock inside allreduce module
var time_ref_u16 = @zeros([timestamp.tsc_size_words]u16);

var ptr_b: [*]f32 = &b;
var ptr_x: [*]f32 = &x;
var ptr_stencil_coeff: [*]f32 = &stencil_coeff;
var ptr_time_buf_u16: [*]u16 = &time_buf_u16;
var ptr_time_ref: [*]u16 = &time_ref_u16;
var ptr_rho: [*]f32 = &rho;
var ptr_xi: [*]f32 = &xi;

// size of local tensor during the BiCGSTAB
var n: i16 = 0;
var tol: f32 = @as(f32, 0);
var max_ite: i16 = 0;

var mem_b_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> b[i] });
var mem_x_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> x[i] });
var mem_r0_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> r0[i] });
var mem_r_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> r[i] });
var mem_p_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> p[i] });
var mem_v_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> v[i] });
var mem_s_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> s[i] });
var mem_t_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> t[i] });

const STATE_SYNC: i16 = 0;
const STATE_INIT: i16 = 1;
const STATE_SPMV_AX: i16 = 2;
const STATE_RESIDUAL: i16 = 3;
const STATE_INIT_RHO: i16 = 4;
const STATE_CONV_CHECK: i16 = 5;
const STATE_SPMV_AP: i16 = 6;
const STATE_R0_DOT_V: i16 = 7;
const STATE_UPDATE_ALPHA_S_T: i16 = 8;
const STATE_T_DOT_S: i16 = 9;
const STATE_T_DOT_T: i16 = 10;
const STATE_UPDATE_W_X_R_RHO: i16 = 11;
const STATE_UPDATE_BETA_P_XI: i16 = 12;
const STATE_EXIT: i16 = 13;

var k: i16 = 0;
var cur_state: i16 = 0;
var next_state: i16 = 0;

fn f_enable_timer() void {
    timestamp.enable_tsc();
    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

fn f_tic() void {
    timestamp.get_timestamp(&tscStartBuffer);

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

fn f_toc() void {
    timestamp.get_timestamp(&tscEndBuffer);

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

fn f_memcpy_timestamps() void {

    time_buf_u16[0] = tscStartBuffer[0];
    time_buf_u16[1] = tscStartBuffer[1];
    time_buf_u16[2] = tscStartBuffer[2];

    time_buf_u16[3] = tscEndBuffer[0];
    time_buf_u16[4] = tscEndBuffer[1];
    time_buf_u16[5] = tscEndBuffer[2];

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

fn f_sync() void {
    cur_state = STATE_SYNC;
    @activate(STATE);
}

fn f_bicgstab(size:i16, tol_val:f32, max_ite_val: i16) void {
    n = size;
    tol = tol_val;
    max_ite = max_ite_val;

    cur_state = STATE_INIT;
    @activate(STATE);
}

// initialization of BiCGSTAB
// - setup the length of all DSDs
//
fn f_bicgstab_init() void {

    // set the length of all DSDs
    mem_b_dsd = @set_dsd_length(mem_b_dsd, @bitcast(u16,n));
    mem_x_dsd = @set_dsd_length(mem_x_dsd, @bitcast(u16,n));
    mem_r0_dsd = @set_dsd_length(mem_r0_dsd, @bitcast(u16,n));
    mem_r_dsd = @set_dsd_length(mem_r_dsd, @bitcast(u16,n));
    mem_p_dsd = @set_dsd_length(mem_p_dsd, @bitcast(u16,n));
    mem_v_dsd = @set_dsd_length(mem_v_dsd, @bitcast(u16,n));
    mem_s_dsd = @set_dsd_length(mem_s_dsd, @bitcast(u16,n));
    mem_t_dsd = @set_dsd_length(mem_t_dsd, @bitcast(u16,n));

    // must go back to state machine
    f_trigger_state_machine();
}

// v = A*x = A*x0
fn f_spmv_Ax() void {
    stencil_mod.spmv(n, &stencil_coeff, &x, &v);
}

// - r0 = b - v = b - A*x0
//   where v = A*x is computed by f_spmv_Ax
// - r = r0
// - p = r0
// - xi = |r0|^2
fn f_residual() void {

    // r0 = b - v
    @fsubs(mem_r0_dsd, mem_b_dsd, mem_v_dsd);

    // r = r0
    @fmovs(mem_r_dsd, mem_r0_dsd);

    // p = r0
    @fmovs(mem_p_dsd, mem_r0_dsd);

    // xi = |r0|^2
    xi[0] = blas_lib.dot(n, &r, &r);
    reduce_mod.allreduce(1, &xi, reduce_mod.TYPE_BINARY_OP.ADD);
}

// rho = xi
fn f_init_rho() void {
    rho[0] = xi[0];

    // must go back to state machine
    f_trigger_state_machine();
}

// v = A*p
fn f_spmv_Ap() void {
    stencil_mod.spmv(n, &stencil_coeff, &p, &v);
}

// r0_dot_v = np.dot(r0,v)
fn f_r0_dot_v() void {

    // r0_dot_v = dot(r0, v)
    r0_dot_v[0] = blas_lib.dot(n, &r0, &v);
    reduce_mod.allreduce(1, &r0_dot_v, reduce_mod.TYPE_BINARY_OP.ADD);
}

// alpha = rho / (r0, v)
// s = r - alpha*v
// t = A*s
fn f_update_alpha_s_t() void {

    alpha = rho[0] / r0_dot_v[0];

    var alpha_minus: f32 = -alpha;

    // s = r - alpha*v
    @fmacs(mem_s_dsd, mem_r_dsd, mem_v_dsd, alpha_minus);

    // t = A*s
    stencil_mod.spmv(n, &stencil_coeff, &s, &t);
}

// t_dot_s = np.dot(t,s)
fn f_t_dot_s() void {

    t_dot_s[0] = blas_lib.dot(n, &t, &s);
    reduce_mod.allreduce(1, &t_dot_s, reduce_mod.TYPE_BINARY_OP.ADD);
}

// t_dot_t = np.dot(t,t)
fn f_t_dot_t() void {

    t_dot_t[0] = blas_lib.dot(n, &t, &t);
    reduce_mod.allreduce(1, &t_dot_t, reduce_mod.TYPE_BINARY_OP.ADD);
}

// w = (t,s)/(t,t)
// x = x + alpha*p + w*s
// r = s - w*t
// rho_old = rho
// rho = (r0, r)
//
fn f_update_w_x_r_rho() void {

    w = t_dot_s[0]/t_dot_t[0];

    // update x = x + alpha*p + w*s
    // i) x = x + alpha*p
    @fmacs(mem_x_dsd, mem_x_dsd, mem_p_dsd, alpha);
    // ii) x = x + w*s
    @fmacs(mem_x_dsd, mem_x_dsd, mem_s_dsd, w);

    // r = s - w * t
    var w_minus: f32 = -w;
    @fmacs(mem_r_dsd, mem_s_dsd, mem_t_dsd, w_minus);

    // update rho
    rho_old[0] = rho[0];

    // rho = np.dot(r0,r)
    rho[0] = blas_lib.dot(n, &r0, &r);
    reduce_mod.allreduce(1, &rho, reduce_mod.TYPE_BINARY_OP.ADD);
}

// beta = (rho/rho_old)*(alpha/w)
// p = r + beta*(p - w*v)
// xi = np.dot(r,r)
fn f_update_beta_p_xi() void {

    beta = (rho[0]/rho_old[0])*(alpha/w);

    // compute p = r + beta*(p - w*v)
    // i) p = p - w*v
    var w_minus: f32 = -w;
    @fmacs(mem_p_dsd, mem_p_dsd, mem_v_dsd, w_minus);
    // ii) p = r + beta*p
    @fmacs(mem_p_dsd, mem_r_dsd, mem_p_dsd, beta);

    // xi = np.dot(r,r)
    xi[0] = blas_lib.dot(n, &r, &r);
    reduce_mod.allreduce(1, &xi, reduce_mod.TYPE_BINARY_OP.ADD);
}

fn f_trigger_state_machine() void {
    cur_state = next_state; // go to next state
    @activate(STATE);
}

// state machine of BiCGSTAB module
// it contains two operations
// - sync operation of allreduce
// - BiCGSTAB algorithm
//
// The callback f_trigger_state_machine is provided for the
// allreduce module and stencil module.
//
// The state transition of sync is
// SYNC --> EXIT
//
// The state transition of BiCGSTAB algorithm is
// INIT --> SPMV_AX --> RESIDUAL --> INIT_RHO --> CONV_CHECK --> EXIT OR SPMV_AP
// --> R0_DOT_V --> UPDATE_ALPHA_S_T --> T_DOT_S --> T_DOT_T
// --> UPDATE_W_X_R_RHO --> UPDATE_BETA_P_XI --> CONV_CHECK
//
task f_state() void {

    if (STATE_SYNC == cur_state){
        // sync all PEs by internal allreduce module
        next_state = STATE_EXIT;
        reduce_mod.allreduce(1, &dot, reduce_mod.TYPE_BINARY_OP.ADD);

    }else if (STATE_INIT == cur_state){
        next_state = STATE_SPMV_AX;
        f_bicgstab_init();

    }else if (STATE_SPMV_AX == cur_state){
        next_state = STATE_RESIDUAL;
        k = 0;
        // v = A*x0
        f_spmv_Ax();

    }else if (STATE_RESIDUAL == cur_state){
        next_state = STATE_INIT_RHO;
        // r0 = b - v = b - A*x0
        // r = r0
        // p = r0
        // xi = |r0|^2
        f_residual();

    }else if (STATE_INIT_RHO == cur_state){
        next_state = STATE_CONV_CHECK;
        // rho = xi = (r0, r0)
        f_init_rho();

    }else if (STATE_CONV_CHECK == cur_state){
        // if |r|_2 < tol, then exit
        if ((xi[0] > tol*tol) and (k < max_ite)){
            next_state = STATE_SPMV_AP;
        }else{
            next_state = STATE_EXIT;
        }
        f_trigger_state_machine();

    }else if (STATE_SPMV_AP == cur_state){
        next_state = STATE_R0_DOT_V;
        k = k + 1;
        // v = A*p
        f_spmv_Ap();

    }else if (STATE_R0_DOT_V == cur_state){
        next_state = STATE_UPDATE_ALPHA_S_T;
        // r0_dot_v = dot(r0, v)
        f_r0_dot_v();

    }else if (STATE_UPDATE_ALPHA_S_T == cur_state){
        next_state = STATE_T_DOT_S;
        // alpha = rho / (r0, v)
        // s = r - alpha*v
        // t = A*s
        f_update_alpha_s_t();

    }else if (STATE_T_DOT_S == cur_state){
        next_state = STATE_T_DOT_T;
        // t_dot_s = np.dot(t,s)
        f_t_dot_s();

    }else if (STATE_T_DOT_T == cur_state){
        next_state = STATE_UPDATE_W_X_R_RHO;
        // t_dot_t = np.dot(t,t)
        f_t_dot_t();

    }else if (STATE_UPDATE_W_X_R_RHO == cur_state){
        next_state = STATE_UPDATE_BETA_P_XI;
        // w = (t,s)/(t,t)
        // x = x + alpha*p + w*s
        // r = s - w*t
        // rho_old = rho
        // rho = (r0, r)
        f_update_w_x_r_rho();

    }else if (STATE_UPDATE_BETA_P_XI == cur_state){
        next_state = STATE_CONV_CHECK;
        // beta = (rho/rho_old)*(alpha/w)
        // p = r + beta*(p - w*v)
        // xi = np.dot(r,r)
        f_update_beta_p_xi();

    }else if (STATE_EXIT == cur_state){
        sys_mod.unblock_cmd_stream();
    }else{
        @assert(false); // Error: unknown state
        // assert() is ignored by HW, it could hang here
        // To avoid a stall, trigger callback (the caveat is the wrong result)
        sys_mod.unblock_cmd_stream();
    }
}

fn f_reference_timestamps() void {

    time_ref_u16[0] = reduce_mod.tscRefBuffer[0];
    time_ref_u16[1] = reduce_mod.tscRefBuffer[1];
    time_ref_u16[2] = reduce_mod.tscRefBuffer[2];

    // the user must unblock cmd color for every PE
    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task( f_state, STATE);
}

comptime {
    @export_symbol(ptr_b, "b");
    @export_symbol(ptr_x, "x");
    @export_symbol(ptr_stencil_coeff, "stencil_coeff");
    @export_symbol(ptr_time_buf_u16, "time_buf_u16");
    @export_symbol(ptr_time_ref, "time_ref");
    @export_symbol(ptr_rho, "rho");
    @export_symbol(ptr_xi, "xi");
}

comptime{
    @export_symbol(f_enable_timer);
    @export_symbol(f_tic);
    @export_symbol(f_toc);
    @export_symbol(f_memcpy_timestamps);

    @export_symbol(f_bicgstab);

    @export_symbol(f_sync);
    @export_symbol(f_reference_timestamps);
}

```

## device\_run.py

```python
#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args
""" test BiCGSTAB of a sparse matrix A built by 7-point stencil

  The following BiCGSTAB algorithm is modified from [1].
  ---
  The algorithm of unpreconditioned BiCGSTAB is
    Given b, x0 and tol = eps*|b|
    k = 0
    x = x0
    r0 = b - A*x
    r = r0
    p = r0
    xi = |r|^2 = (r0, r0)
    rho = xi
    while xi > tol*tol and k < max_ite
        k = k + 1
        v = A*p
        alpha = rho/(r0, v)
        s = r - alpha*v
        t = A*s
        w = (t,s)/(t,t)
        x = x + alpha*p + w*s
        r = s - w*t
        rho_old = rho
        rho = (r0, r)
        beta = (rho/rho_old)*(alpha/w)
        p = r + beta*(p - w*v)
        xi = |r|^2
    end
    x approximates the solution of a linear system Ax = b

  The sparse matrix A is built by a 7-point stenil.
  The 7-point stencil is defined by the following:
  ---
    The Laplacian operator L on 3-dimensional domain can be represented by 7-point
  stencil based on the standard 2nd order Finite Difference Method. The operator form
  with Dirichlet boundary conditions can be written by
         L[u](i,j,k) = u(i+1, j,  k  ) + u(i-1, j,   k  ) +
                       u(i,   j+1,k  ) + u(i,   j-1, k  ) +
                       u(i,   j,  k+1) + u(i,   j,   k-1) +
                      -6*u(i, j, k)
  In general the coefficients of those 7 points can vary. To minimize the memory
  consumption, this example assumes the coefficients are independent of index k and
  whole vector u(i,j,:) is placed in one PE (px=j, py=i).
  The above formula can be re-written by
     c_west   * x[i-1][j  ][k  ] + c_east  * x[i+1][j  ][k  ] +
     c_south  * x[i  ][j-1][k  ] + c_north * x[i  ][j+1][k  ] +
     c_bot    * x[i  ][j  ][k-1] + c_top   * x[i  ][j  ][k+1] +
     c_center * x[i][j][k]
  Each PE only holds 7 coefficients organized by c_west, c_east, c_south, c_north,
  c_bot, c_top and c_center.

  This example provides two modules, one is allreduce and the other is stencil_3d_7pts.
  "allreduce" module can synchronize all PEs to form a reference clock.
  "allreduce" module also computes dot(x,y) over a core rectangle.
  "stencil_3d_7pts" module can compute y = A*x where A is the matrix from 7-point stencil.

  The framework is
  ---
       sync()      // synchronize all PEs to sample the reference clock
       tic()       // record start time
       BiCGSTAB(n, tol. max_ite) // BiCGSTAB on WSE
       toc()       // record end time
  ---
  The device_run.py performs CG on the WSE, not calls a sequence of spmv and dot.
  It is faster than run.py because the nrm(r) is not transferred back to the host.
  WSE can check the convergence without the host.

  The tic() samples "time_start" and toc() samples "time_end". The sync() samples
  "time_ref" which is used to shift "time_start" and "time_end".
  The elapsed time is measured by
       cycles_send = max(time_end) - min(time_start)

  The overall runtime is computed via the following formula
       time_send = (cycles_send / 0.85) *1.e-3 us
  where a PE runs with clock speed 850MHz

  Here is the list of parameters:
    -m=<int> is the height of the core
    -n=<int> is the width of the core
    -k=<int> is size of x and y allocated in the core
    --zDim=<int> is the number of f32 per PE, computed by y = A*x
                 zDim must be not greater than k
    --max-ite=<int> number of iterations
    --channels=<int> specifies the number of I/O channels, no bigger than 16

  WARNING: BiCGSTAB may not converge if rho=(rj, r0) is close to zero.
  If it does happen, we need to restart the BiCGSTAB.
  The algorithm here does not provide the restart.

  Reference:
  [1] H. A. VAN DER VORST, BI-CGSTAB: A FAST AND SMOOTHLY CONVERGING VARIANT
      OF BI-CG FOR THE SOLUTION OF NONSYMMETRIC LINEAR SYSTEMS,
      SIAM J. ScI. STAT. COMPUT. Vol. 13, No. 2, pp. 631-644, March 1992
"""

import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from bicgstab import bicgstab
from cmd_parser import parse_args
from scipy.sparse import linalg as sparse_LA
from util import csr_7_pt_stencil, hwl_2_oned_colmajor, oned_to_hwl_colmajor

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def csl_compile_core(
    cslc: str,
    width: int,  # width of the core
    height: int,  # height of the core
    pe_length: int,
    blockSize: int,
    file_config: str,
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,  # fabric-offsets of the core
    core_fabric_offset_y: int,
    use_precompile: bool,
    arch: Optional[str],
    C0: int,
    C1: int,
    C2: int,
    C3: int,
    C4: int,
    C5: int,
    C6: int,
    C7: int,
    C8: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
):
  if not use_precompile:
    args = []
    args.append(cslc)  # command
    args.append(file_config)
    args.append(f"--fabric-dims={fabric_width},{fabric_height}")
    args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
    args.append(f"--params=width:{width},height:{height},MAX_ZDIM:{pe_length}")
    args.append(f"--params=BLOCK_SIZE:{blockSize}")
    args.append(f"--params=C0_ID:{C0}")
    args.append(f"--params=C1_ID:{C1}")
    args.append(f"--params=C2_ID:{C2}")
    args.append(f"--params=C3_ID:{C3}")
    args.append(f"--params=C4_ID:{C4}")
    args.append(f"--params=C5_ID:{C5}")
    args.append(f"--params=C6_ID:{C6}")
    args.append(f"--params=C7_ID:{C7}")
    args.append(f"--params=C8_ID:{C8}")

    args.append(f"-o={elf_dir}")
    if arch is not None:
      args.append(f"--arch={arch}")
    args.append("--memcpy")
    args.append(f"--channels={channels}")
    args.append(f"--width-west-buf={width_west_buf}")
    args.append(f"--width-east-buf={width_east_buf}")

    print(f"subprocess.check_call(args = {args}")
    subprocess.check_call(args)
  else:
    print("\tuse pre-compile ELFs")

def timing_analysis(height, width, time_memcpy_hwl, time_ref_hwl):
  # time_start = start time of spmv
  time_start = np.zeros((height, width)).astype(int)
  # time_end = end time of spmv
  time_end = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_memcpy_hwl[(h, w, 0)]
      word[1] = time_memcpy_hwl[(h, w, 1)]
      word[2] = time_memcpy_hwl[(h, w, 2)]
      time_start[(h, w)] = make_u48(word)
      word[0] = time_memcpy_hwl[(h, w, 3)]
      word[1] = time_memcpy_hwl[(h, w, 4)]
      word[2] = time_memcpy_hwl[(h, w, 5)]
      time_end[(h, w)] = make_u48(word)

  # time_ref = reference clock
  time_ref = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_ref_hwl[(h, w, 0)]
      word[1] = time_ref_hwl[(h, w, 1)]
      word[2] = time_ref_hwl[(h, w, 2)]
      time_ref[(h, w)] = make_u48(word)

  # adjust the reference clock by the propagation delay
  # the right-bottom PE signals other PEs, the propagation delay is
  #     (h-1) - py + (w-1) - px
  for py in range(height):
    for px in range(width):
      time_ref[(py, px)] = time_ref[(py, px)] - ((width + height - 2) - (px + py))

  # shift time_start and time_end by time_ref
  time_start = time_start - time_ref
  time_end = time_end - time_ref

  # cycles_send = time_end[(h,w)] - time_start[(h,w)]
  # 850MHz --> 1 cycle = (1/0.85) ns = (1/0.85)*1.e-3 us
  # time_send = (cycles_send / 0.85) *1.e-3 us
  #
  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) * 1.0e-3
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")

# How to compile
#   python device_run.py -m=5 -n=5 -k=5 --latestlink latest --channels=1 \
#   --width-west-buf=0 --width-east-buf=0 --compile-only
# How to run
#   python device_run.py -m=5 -n=5 -k=5 --latestlink latest --channels=1 \
#   --width-west-buf=0 --width-east-buf=0 --run-only --zDim=5 --max-ite=1
def main():
  """Main method to run the example code."""

  random.seed(127)

  args, dirname = parse_args()

  cslc = "cslc"
  if args.driver is not None:
    cslc = args.driver

  print(f"cslc = {cslc}")

  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  channels = args.channels
  assert channels <= 16, "only support up to 16 I/O channels"
  assert channels >= 1, "number of I/O channels must be at least 1"

  print(f"width_west_buf = {width_west_buf}")
  print(f"width_east_buf = {width_east_buf}")
  print(f"channels = {channels}")

  height = args.m
  width = args.n
  pe_length = args.k
  zDim = args.zDim
  blockSize = args.blockSize
  max_ite = args.max_ite

  print(
      f"width={width}, height={height}, pe_length={pe_length}, zDim={zDim}, blockSize={blockSize}"
  )
  print(f"max_ite = {max_ite}")
  assert pe_length >= 2, "the maximum size of z must be greater than 1"
  assert zDim <= pe_length, "[0, zDim) cannot exceed the storage"

  np.random.seed(2)
  x = np.zeros((height, width, pe_length), dtype=np.float32)
  #x = (np.arange(height * width * zDim).reshape(height, width, zDim).astype(np.float32) + 100)

  x_1d = hwl_2_oned_colmajor(height, width, zDim, x, np.float32)
  nrm2_x = np.linalg.norm(x_1d.ravel(), 2)
  # |x0|_2 = 1
  if nrm2_x > 0:
    x_1d = x_1d / nrm2_x
    x = x / nrm2_x

  b = (np.arange(height * width * pe_length).reshape(height, width, pe_length).astype(np.float32) +
       1)
  b_1d = hwl_2_oned_colmajor(height, width, pe_length, b, np.float32)

  # stencil coefficients has the following order
  # {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
  # BiCGSTAB does not converge, it hits serious breakdown when
  # rho = (r0, rj) is close to zero.
  # Such behavior is confirmed by scipy.sparse.linalg.bicgstab
  stencil_coeff = np.zeros((height, width, 7), dtype=np.float32)
  for i in range(height):
    for j in range(width):
      stencil_coeff[(i, j, 0)] = -1  # west
      stencil_coeff[(i, j, 1)] = -1  # east
      stencil_coeff[(i, j, 2)] = -1  # south
      stencil_coeff[(i, j, 3)] = -1  # north
      stencil_coeff[(i, j, 4)] = -1  # bottom
      stencil_coeff[(i, j, 5)] = -1  # top
      stencil_coeff[(i, j, 6)] = 6 + j  # center

  # fabric-offsets = 1,1
  fabric_offset_x = 1
  fabric_offset_y = 1
  # starting point of the core rectangle = (core_fabric_offset_x, core_fabric_offset_y)
  # memcpy framework requires 3 columns at the west of the core rectangle
  # memcpy framework requires 2 columns at the east of the core rectangle
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y
  # (min_fabric_width, min_fabric_height) is the minimal dimension to run the app
  min_fabric_width = core_fabric_offset_x + width + 2 + 1 + width_east_buf
  min_fabric_height = core_fabric_offset_y + height + 1

  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)

  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height

  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  # prepare the simulation
  print("store ELFs and log files in the folder ", dirname)

  # layout of a rectangle
  code_csl = "layout_bicgstab.csl"

  C0 = 0
  C1 = 1
  C2 = 2
  C3 = 3
  C4 = 4
  C5 = 5
  C6 = 6
  C7 = 7
  C8 = 8

  csl_compile_core(
      cslc,
      width,
      height,
      pe_length,
      blockSize,
      code_csl,
      dirname,
      fabric_width,
      fabric_height,
      core_fabric_offset_x,
      core_fabric_offset_y,
      args.run_only,
      args.arch,
      C0,
      C1,
      C2,
      C3,
      C4,
      C5,
      C6,
      C7,
      C8,
      channels,
      width_west_buf,
      width_east_buf,
  )
  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  A_csr = csr_7_pt_stencil(stencil_coeff, height, width, zDim)
  [nrows, ncols] = A_csr.shape
  nnz = A_csr.nnz
  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros")

  if nrows < 1000:
    A = A_csr.todense()
    # s is singular value in descending order
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    min_svd = min(s)
    max_svd = max(s)
    print(f"min(svd) = {min_svd}")
    print(f"max(svd) = {max_svd}")
    print(f"cond(A) = {max_svd/min_svd}")

  nrm_b = np.linalg.norm(b_1d.ravel(), 2)
  eps = 1.0e-3
  tol = eps * nrm_b
  print(f"|b| = {nrm_b}")
  print(f"max_ite = {max_ite}")
  print(f"eps = {eps}")
  print(f"tol = {tol}")

  xf_1d, xi, k = bicgstab(A_csr, x_1d, b_1d, max_ite, tol)
  print(f"[host] after BiCGSTAB, xi = {xi}, k = {k}")

  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  simulator = SdkRuntime(dirname, cmaddr=args.cmaddr)

  symbol_b = simulator.get_id("b")
  symbol_x = simulator.get_id("x")
  symbol_xi = simulator.get_id("xi")
  symbol_stencil_coeff = simulator.get_id("stencil_coeff")
  symbol_time_buf_u16 = simulator.get_id("time_buf_u16")
  symbol_time_ref = simulator.get_id("time_ref")

  simulator.load()
  simulator.run()

  print("copy vector b and x0")
  simulator.memcpy_h2d(
      symbol_b,
      b_1d,
      0,
      0,
      width,
      height,
      zDim,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  simulator.memcpy_h2d(
      symbol_x,
      x_1d,
      0,
      0,
      width,
      height,
      zDim,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  print("copy 7 stencil coefficients")
  stencil_coeff_1d = hwl_2_oned_colmajor(height, width, 7, stencil_coeff, np.float32)
  simulator.memcpy_h2d(
      symbol_stencil_coeff,
      stencil_coeff_1d,
      0,
      0,
      width,
      height,
      7,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=True,
  )

  print("step 0: enable timer")
  simulator.launch("f_enable_timer", nonblock=False)

  print("step 1: sync all PEs")
  simulator.launch("f_sync", nonblock=False)

  print("step 2: copy reference clock from reduce module")
  simulator.launch("f_reference_timestamps", nonblock=False)

  print("step 3: tic() records time_start")
  simulator.launch("f_tic", nonblock=True)

  print(f"step 4: BiCGSTAB with max_ite={max_ite}, zDim={zDim}, tol={tol}")
  simulator.launch(
      "f_bicgstab",
      np.int16(zDim),
      np.float32(tol),
      np.int16(max_ite),
      nonblock=False,
  )

  print("step 5: toc() records time_end")
  simulator.launch("f_toc", nonblock=False)

  xi_wse = np.zeros(1, np.float32)
  simulator.memcpy_d2h(
      xi_wse,
      symbol_xi,
      0,
      0,
      1,
      1,
      1,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  xi = xi_wse[0]
  print(f"[BiCGSTAB] xi = |b-A*x|^2 = {xi}")

  print("step 6: prepare (time_start, time_end)")
  simulator.launch("f_memcpy_timestamps", nonblock=False)

  print("step 7: D2H (time_start, time_end)")
  time_memcpy_hwl_1d = np.zeros(height * width * 6, np.uint32)
  simulator.memcpy_d2h(
      time_memcpy_hwl_1d,
      symbol_time_buf_u16,
      0,
      0,
      width,
      height,
      6,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_16BIT,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  time_memcpy_hwl = oned_to_hwl_colmajor(height, width, 6, time_memcpy_hwl_1d, np.uint16)

  print("step 8: D2H reference clock")
  time_ref_1d = np.zeros(height * width * 3, np.uint32)
  simulator.memcpy_d2h(
      time_ref_1d,
      symbol_time_ref,
      0,
      0,
      width,
      height,
      3,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_16BIT,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )
  time_ref_hwl = oned_to_hwl_colmajor(height, width, 3, time_ref_1d, np.uint16)

  print("step 9: D2H x[zDim]")
  xf_wse_1d = np.zeros(height * width * zDim, np.float32)
  simulator.memcpy_d2h(
      xf_wse_1d,
      symbol_x,
      0,
      0,
      width,
      height,
      zDim,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )

  simulator.stop()

  if args.cmaddr is None:
    # move simulation log and core dump to the given folder
    dst_log = Path(f"{dirname}/sim.log")
    src_log = Path("sim.log")
    if src_log.exists():
      shutil.move(src_log, dst_log)

    dst_trace = Path(f"{dirname}/simfab_traces")
    src_trace = Path("simfab_traces")
    if dst_trace.exists():
      shutil.rmtree(dst_trace)
    if src_trace.exists():
      shutil.move(src_trace, dst_trace)

  timing_analysis(height, width, time_memcpy_hwl, time_ref_hwl)

  nrm2_xf = np.linalg.norm(xf_wse_1d.ravel(), 2)
  print(f"|xf|_2 = {nrm2_xf}")
  print(f"|b - A*xf|_2 = {math.sqrt(xi)}")
  print(f"number of iterations = {k}")

  z = xf_1d.ravel() - xf_wse_1d.ravel()
  nrm_z = np.linalg.norm(z, np.inf)
  print(f"|xf_ref - xf_wse| = {nrm_z}")
  np.testing.assert_allclose(xf_1d.ravel(), xf_wse_1d.ravel(), 1.0e-5)
  print("\nSUCCESS!")

  print("==== measure the residual b - A*x by |.|_inf")
  # r = b - A*xf
  r = b_1d - A_csr.dot(xf_wse_1d)
  nrm_r = np.linalg.norm(r, np.inf)
  nrm_b = np.linalg.norm(b_1d, np.inf)
  nrm_x = np.linalg.norm(xf_wse_1d, np.inf)
  nrm_A = sparse_LA.norm(A_csr, np.inf)
  print(f"|xf| = {nrm_x}")
  print(f"|b| = {nrm_b}")
  print(f"|A| = {nrm_A}")
  print(f"|b - A*xf| = {nrm_r}")
  rel_err = nrm_r / (nrm_A * nrm_x + nrm_b)
  print(f"|b-A*x|/(|A|*|x|+|b|) = {rel_err}")
  # only check the relative error if max_ite is big enough
  # i.e. BiCGSTAB converges before reaching max_ite
  if k < max_ite:
    assert eps > rel_err, "relative error is too large"

if __name__ == "__main__":
  main()

```
[previous

Preconditioned Conjugate Gradient](csl/code-examples/benchmark-preconditioned-conjugate-gradient.md "previous page")
[next

Single Tile Matvec](csl/code-examples/benchmark-single-tile-matvec.md "next page")
