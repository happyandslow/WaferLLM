---
title: "25-Point Stencil"
url: "https://sdk.cerebras.net/csl/code-examples/benchmark-25-pt-stencil"
crawled_at: "2026-02-14 23:55:41"
---

# 25-Point Stencil

> Source: [https://sdk.cerebras.net/csl/code-examples/benchmark-25-pt-stencil](https://sdk.cerebras.net/csl/code-examples/benchmark-25-pt-stencil)

The stencil code is a time-marching app, requiring the following three inputs:

- scalar `iterations`: number of time steps
- tensor `vp`: velocity field
- tensor `source`: source term

and producing the following three outputs:

- maximum and minimum value of vector field of last time step, two f32 per PE
- timestamps of the time-marching per PE, three uint32 per PE
- vector field `z` of last time step, `zdim` f32 per PE

The stencil code uses 21 colors and task IDs for communication patterns,
and `SdkRuntime` reserves 6 colors,
so only 4 colors are left for `streaming` H2D/D2H transfers
and some entrypoints for control flow.
We use one color (color 0) to launch kernel functions
and one entrypoint (color 2) to trigger the time marching.
The `copy` mode of memcpy is used for two inputs and two outputs.

After the simulator (or WSE) has been launched,
we send input tensors `vp` and `source` to the device via `copy` mode.

Second, we launch time marching with the argument `iterations`.

In this example, we have two kernel launches.
One performs time marching after `vp` and `source` are received,
and the other prepares the output data `zValues`.
The former has the function symbol `f_activate_comp`
and the latter has the function symbol `f_prepare_zout`.
Here `SdkRuntime.launch()` triggers a host-callable function, in which
the first argument is the function symbol `f_activate_comp`,
and the second argument is `iterations`,
which is received as an argument by `f_activate_comp`.

The end of time marching (`f_checkpoint()` in `task.csl`)
will record the maximum and minimum value
of the vector field and timing info into an array `d2h_buf_f32`.
The host calls `memcpy_d2h()` to receive the data in `d2h_buf_f32`.

To receive the vector field of the last time step,
the function `f_prepare_zout()` is called by `SdkRuntime.launch()`
to prepare this data into a temporary array `zout`,
because the result is in either `zValues[0, :]` or `zValues[1, :]`.

The last operation, `memcpy_d2h()`, sends the array `zout` back to the host.

When `f_activate_comp` is launched, it triggers the entrypoint `f_comp()`
to start the time-marching and to record the starting time.

At the end of time marching, the function `epilog()` checks
`iterationCount`.
If it reaches the given `iterations`, `epilog()` triggers the entrypoint
`CHECKPOINT` to prepare the data for the first `memcpy_d2h()`.

The function `f_checkpoint()` calls `unblock_cmd_stream()` to process the
next operation which is the first `memcpy_d2h()`.
Without `unblock_cmd_stream()`, the program stalls because the
`memcpy_d2h()` is never scheduled.

The function `f_prepare_zout()` prepares the vector field into `zout`.
It also calls `unblock_cmd_stream()` to process the next operation, which is
the second `memcpy_d2h()`.

## layout.csl

```csl
////////////////////////////////////////////////////////////////////////////////
// The code for this 3D 25-point stencil was inspired by the proprietary code //
// of TotalEnergies EP Research & Technology US.                              //
////////////////////////////////////////////////////////////////////////////////

// The core kernel must start at P4.1 so the memcpy infrastructure has enough
// resources to route the data between the host and the device.
//

// color map of FD + memcpy:
//
// color  var             color  var          color  var              color  var
//   0                      9    westDataFin   18    northCtrlFin2     27   reserved (memcpy)
//   1                     10    northDataFin  19    southCtrlFin2     28   reserved (memcpy)
//   2   f_comp            11    southDataFin  20    eastFin           29   reserved (memcpy)
//   3   send              12    eastCtrlFin   21    reserved (memcpy) 30   reserved (memcpy)
//   4   eastChannel       13    westCtrlFin   22    reserved (memcpy) 31   reserved
//   5   westChannel       14    northCtrlFin  23    reserved (memcpy) 32
//   6   northChannel      15    southCtrlFin  24    westFin           33
//   7   southChannel      16    eastCtrlFin2  25    northFin          34
//   8   eastDataFin       17    westCtrlFin2  26    southFin          35
//

// Colors
param eastChannel:  color = @get_color(4);
param westChannel:  color = @get_color(5);
param northChannel: color = @get_color(6);
param southChannel: color = @get_color(7);

// Task IDs
param send: local_task_id = @get_local_task_id(3);
param COMP: local_task_id = @get_local_task_id(2);

param eastDataFin:  local_task_id = @get_local_task_id(8);
param westDataFin:  local_task_id = @get_local_task_id(9);
param northDataFin: local_task_id = @get_local_task_id(10);
param southDataFin: local_task_id = @get_local_task_id(11);

param eastCtrlFin:  local_task_id = @get_local_task_id(12);
param westCtrlFin:  local_task_id = @get_local_task_id(13);
param northCtrlFin: local_task_id = @get_local_task_id(14);
param southCtrlFin: local_task_id = @get_local_task_id(15);

// the following four are entrypoints (send control wavelets for switch)
// we don't need to bind it to 0~23
param eastCtrlFin2:  local_task_id = @get_local_task_id(16);
param westCtrlFin2:  local_task_id = @get_local_task_id(17);
param northCtrlFin2: local_task_id = @get_local_task_id(18);
param southCtrlFin2: local_task_id = @get_local_task_id(19);

param eastFin:  local_task_id = @get_local_task_id(20);
// WARNING: ID 21: reserved (memcpy)
// WARNING: ID 22: reserved (memcpy)
//          ID 23: reserved (memcpy)
param westFin:  local_task_id = @get_local_task_id(24);
param northFin: local_task_id = @get_local_task_id(25);
param southFin: local_task_id = @get_local_task_id(26);

param width: u16;
param height: u16;
param zDim: u16;
param sourceLength: u16;
param dx: u16;

// Number of neighbors (plus self) that each PE will communicate with in all
// directions.  The (three-dimensional) stencil size is `6 * (pattern - 1) + 1`.
const pattern: u16 = 5;

//// The coordinates of the "source" PE, which adds a small value to the wavefield
//// in each iteration.
param srcX: u16;
param srcY: u16;
param srcZ: u16;

// The following parameters are the same for every PE, so we extract them out of
// the loop that sets PE-specific parameters below.
const invariants = .{
  .send = send,
  .zDim = zDim,
  .pattern = pattern,
  .sourceLength = sourceLength,
  .dx = dx,
  .width = width,
  .height = height,
  .srcZ = srcZ,

  .eastFin = eastFin,
  .westFin = westFin,
  .northFin = northFin,
  .southFin = southFin,

  .eastDataFin = eastDataFin,
  .westDataFin = westDataFin,
  .northDataFin = northDataFin,
  .southDataFin = southDataFin,

  .eastCtrlFin = eastCtrlFin,
  .westCtrlFin = westCtrlFin,
  .northCtrlFin = northCtrlFin,
  .southCtrlFin = southCtrlFin,

  .eastCtrlFin2 = eastCtrlFin2,
  .westCtrlFin2 = westCtrlFin2,
  .northCtrlFin2 = northCtrlFin2,
  .southCtrlFin2 = southCtrlFin2,

  .eastChannel = eastChannel,
  .westChannel = westChannel,
  .northChannel = northChannel,
  .southChannel = southChannel,
};

const util = @import_module("util.csl");

const memcpy = @import_module( "<memcpy/get_params>", .{
    .width = width,
    .height = height,
    });

layout {
  @comptime_assert(pattern <= width);
  @comptime_assert(pattern <= height);
  @comptime_assert(pattern > 1 and pattern < 8);

  // step 1: configure the rectangle which does not include halo
  @set_rectangle(width, height);

  // step 2: compile csl code for a set of PEx.y and generate out_x_y.elf
  //   format: @set_tile_code(x, y, code.csl, param_binding);

  var xId = 0;
  while (xId < width) : (xId += 1) {

    // We specify the communication pattern is just one
    // (eastward) direction out of the four cardinal directions (east, west,
    // north, and south).  We then mirror the communication pattern in all other
    // directions using relative PE IDs.  For instance, westward communication
    // is identical to eastward communication with decreasing X coordinates.
    // Similarly, southward communication is the same as eastward communication,
    // except using the Y coordinate instead of the X coordinate.

    // Here we compute the relative coordinates for westward and eastward
    // communication.
    const westPeId = util.computeRelativePeId(xId, width, WEST);
    const eastPeId = util.computeRelativePeId(xId, width, EAST);

    const westParams = .{
      .westFirst = westPeId == 0,
      .westLast = westPeId == width - 1,
      .westPatternId = westPeId % pattern,
      .westNotNeedsPos3 = westPeId < pattern - 1,
      .westPatternFirst = westPeId % pattern == 0,
      .westPatternLast = westPeId % pattern == pattern - 1,
      .westSenderCount = util.min(pattern, westPeId + 1),
    };

    const eastParams = .{
      .eastFirst = eastPeId == 0,
      .eastLast = eastPeId == width - 1,
      .eastPatternId = eastPeId % pattern,
      .eastNotNeedsPos3 = eastPeId < pattern - 1,
      .eastPatternFirst = eastPeId % pattern == 0,
      .eastPatternLast = eastPeId % pattern == pattern - 1,
      .eastSenderCount = util.min(pattern, eastPeId + 1),
    };

    const hParams = @concat_structs(westParams, eastParams);

    var yId = 0;
    while (yId < height) : (yId += 1) {

      // Here we compute the relative coordinates for northward and southward
      // communication.
      const northPeId = util.computeRelativePeId(yId, height, NORTH);
      const southPeId = util.computeRelativePeId(yId, height, SOUTH);

      const northParams = .{
        .northFirst = northPeId == 0,
        .northLast = northPeId == height - 1,
        .northPatternId = northPeId % pattern,
        .northNotNeedsPos3 = northPeId < pattern - 1,
        .northPatternFirst = northPeId % pattern == 0,
        .northPatternLast = northPeId % pattern == pattern - 1,
        .northSenderCount = util.min(pattern, northPeId + 1),
      };

      const southParams = .{
        .southFirst = southPeId == 0,
        .southLast = southPeId == height - 1,
        .southPatternId = southPeId % pattern,
        .southNotNeedsPos3 = southPeId < pattern - 1,
        .southPatternFirst = southPeId % pattern == 0,
        .southPatternLast = southPeId % pattern == pattern - 1,
        .southSenderCount = util.min(pattern, southPeId + 1),
      };

      const vParams = @concat_structs(northParams, southParams);
      const dirParams = @concat_structs(hParams, vParams);
      const baseParams = @concat_structs(invariants, dirParams);

      const params = @concat_structs(.{
        .isSourcePe = xId == srcX and yId == srcY,
        .isTscOutPe = xId == width - 1 and yId == 0,
      }, baseParams);

      // additional colors for memcpy
      const params_task = @concat_structs( .{
            .COMP = COMP,
            ._px=xId,
      }, params);

      const memcpyParams = memcpy.get_params(xId);

      @set_tile_code(xId, yId, "task.csl", @concat_structs( .{
            .memcpyParams = memcpyParams,
      }, params_task));

    }
  }

  // step 3: global and internal routing
  //  format: @set_color_config(x, y, color, route);

  // export symbol name
  @export_name("vp", [*]f32, true);
  @export_name("source", [*]f32, true);
  @export_name("maxmin_time", [*]f32, true);
  @export_name("zout", [*]f32, true);

  @export_name("f_activate_comp", fn(u32)void);
  @export_name("f_prepare_zout", fn()void);
}

```

## task.csl

```csl
//
// FD kernel with memcpy
//
// The sequence of execution is
// - H2D(vp) : prepare vp
// - H2D(source): prepare source
// - launch(0): trigger time marching
// - D2H(maxmin_time): record max/min of zValues and time stamps
// - launch(1): prepare zout which is either zValues[0, zOffset] or zValues[1, zOffset]
// - D2H(zout)
//
param memcpyParams: comptime_struct;

// Colors
param eastChannel:  color;
param westChannel:  color;
param northChannel: color;
param southChannel: color;

// Task IDs
param COMP: local_task_id; // start time marching
param send: local_task_id;

param eastFin:  local_task_id;
param westFin:  local_task_id;
param northFin: local_task_id;
param southFin: local_task_id;

param eastDataFin:  local_task_id;
param westDataFin:  local_task_id;
param northDataFin: local_task_id;
param southDataFin: local_task_id;

param eastCtrlFin:  local_task_id;
param westCtrlFin:  local_task_id;
param northCtrlFin: local_task_id;
param southCtrlFin: local_task_id;

param eastCtrlFin2:  local_task_id;
param westCtrlFin2:  local_task_id;
param northCtrlFin2: local_task_id;
param southCtrlFin2: local_task_id;

param _px: i16;

param isTscOutPe: bool;

param zDim: i16;
param pattern: u16;
param isSourcePe: bool;
param sourceLength: u32;
param dx: u16;
param width: u16;
param height: u16;
param srcZ: u16;

// Code allows do receive along 4 cardinal directions only
// Anisotropy will require "diagonal" broadcasts
const directionCount: u16 = 4;

const timestamp = @import_module("<time>");
var tscEndBuffer = @zeros([timestamp.tsc_size_words]u16);
var tscStartBuffer = @zeros([timestamp.tsc_size_words]u16);

var iterations: u32 = 0;

const sys_mod = @import_module( "<memcpy/memcpy>", memcpyParams);

//
// FD uses input_queue = 4,5,6,7
// oned_exch.csl:    .input_queue = 4 + queueId,
//                   .output_queue = queueId
// task.csl:         .queueId = 0,
// task.csl:         .queueId = 1,
// task.csl:         .queueId = 2,
// task.csl:         .queueId = 3,
//
// so memcpyH2D with input_queue = 0 does not collide others
// The D2H uses output_queue = 0
// There should not have any problem with output_queue = 0
// because multiple colors can share the same output_queue
//

const zOffset: i16 = pattern - 1;
const math = @import_module("<math>");

var recvChunkCounter: i16 = 0;
var sendChunkCounter: i16 = 0;

const util = @import_module("util.csl");

const numChunks = util.computeChunks(zDim);
const chunkSize = util.computeChunkSize(zDim, @as(u16, numChunks));
const paddedZDim = chunkSize * @as(u16, numChunks);

const routes = @import_module("routes.csl", .{
  .pattern = pattern,
  .peWidth = width,
  .peHeight = height,
});

const consts = @import_module("consts.csl", .{
  .pattern = pattern,
  .paddedZDim = paddedZDim,
});

const xConsts = consts.computeMinimigConsts(dx);
const yConsts = consts.computeMinimigConsts(dx);
const zConsts = consts.computeMinimigConsts(dx);

// The `zValues` array determines the seed value of the program.  For now, we
// use all zeros to match the reference code.
var zValues = consts.initBuffer();
var vp = @zeros([zDim]f32);

//var source = @zeros([sourceLength]f32);
var source = @zeros([zDim]f32);

//--- MEMCPY
const dummy = @zeros([1]f32);

// d2h_buf_f32[0] = max(zValues)
// d2h_buf_f32[1] = min(zValues)
// d2h_buf_f32[2:4] = timestamps
var d2h_buf_f32 = @zeros([5]f32);

// temporary array to hold ether zValues[0, zOffset] or zValues[1, zOffset]
var zout = @zeros([zDim]f32);

// WARNING: export pointers, not arrays
var ptr_vp : [*]f32 = &vp;
var ptr_source : [*]f32 = &source;
var ptr_d2h_buf_f32 : [*]f32 = &d2h_buf_f32;
var ptr_zout : [*]f32 = &zout;

var mem_z_buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{zDim} -> dummy[i] });
var mem_zout_buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{zDim} -> zout[i] });

//--- END MEMCPY

//These are broadcasts stuff
param westFirst: bool;
param westLast: bool;
param westPatternId: u16;
param westNotNeedsPos3: bool;
param westPatternFirst: bool;
param westPatternLast: bool;
param westSenderCount: u16;

param eastFirst: bool;
param eastLast: bool;
param eastPatternId: u16;
param eastNotNeedsPos3: bool;
param eastPatternFirst: bool;
param eastPatternLast: bool;
param eastSenderCount: u16;

param northFirst: bool;
param northLast: bool;
param northPatternId: u16;
param northNotNeedsPos3: bool;
param northPatternFirst: bool;
param northPatternLast: bool;
param northSenderCount: u16;

param southFirst: bool;
param southLast: bool;
param southPatternId: u16;
param southNotNeedsPos3: bool;
param southPatternFirst: bool;
param southPatternLast: bool;
param southSenderCount: u16;

// Since our code essentially uses the same communication code with parameters
// for the direction of the communication, we compute the subset of constants
// that will be used by each instance of the communication code.  The boolean
// value to `fetch*Consts()` function specifies whether the constant for the
// element at the center should be included or not.  Since we want to include
// the center element only once, we pass `true` only for the _first_ invocation
// of this function, while all other values are false.
const eastConsts = consts.fetchFirstHalfConsts(xConsts, true);
const permutedEastConsts = consts.permuteConsts(eastPatternId, eastConsts);

const westConsts = consts.fetchSecondHalfConsts(xConsts, false);
const permutedWestConsts = consts.permuteConsts(westPatternId, westConsts);

const southConsts = consts.fetchFirstHalfConsts(yConsts, false);
const permutedSouthConsts = consts.permuteConsts(southPatternId, southConsts);

const northConsts = consts.fetchSecondHalfConsts(yConsts, false);
const permutedNorthConsts = consts.permuteConsts(northPatternId, northConsts);

var accumulator = @zeros([paddedZDim]f32);
var buffer = @zeros([directionCount, pattern, chunkSize]f32);

// We import a module that is parameterized on the direction of the
// communication.  The following module handles eastward communication.
const eastBus = @import_module("oned_exch.csl", .{
  .zValues = &zValues,
  .buffer = &buffer,
  .pattern = pattern,
  .chunkSize = chunkSize,
  .paddedZDim = paddedZDim,

  .pos = 0,
  .dir = EAST,
  .queueId = 0,
  .dataFin_task_id = eastDataFin,
  .ctrlFin_task_id = eastCtrlFin,
  .channel = eastChannel,
  .callback = eastFinTask,
  .senderCount = eastSenderCount,
  .ctrlCallback = eastCtrlFinTask,
  .constants = &permutedEastConsts,
});

const westBus = @import_module("oned_exch.csl", .{
  .zValues = &zValues,
  .buffer = &buffer,
  .pattern = pattern,
  .chunkSize = chunkSize,
  .paddedZDim = paddedZDim,

  .pos = 1,
  .dir = WEST,
  .queueId = 1,
  .dataFin_task_id = westDataFin,
  .ctrlFin_task_id = westCtrlFin,
  .channel = westChannel,
  .callback = westFinTask,
  .senderCount = westSenderCount,
  .ctrlCallback = westCtrlFinTask,
  .constants = &permutedWestConsts,
});

const southBus = @import_module("oned_exch.csl", .{
  .zValues = &zValues,
  .buffer = &buffer,
  .pattern = pattern,
  .chunkSize = chunkSize,
  .paddedZDim = paddedZDim,

  .pos = 2,
  .dir = SOUTH,
  .queueId = 2,
  .dataFin_task_id = southDataFin,
  .ctrlFin_task_id = southCtrlFin,
  .channel = southChannel,
  .callback = southFinTask,
  .senderCount = southSenderCount,
  .ctrlCallback = southCtrlFinTask,
  .constants = &permutedSouthConsts,
});

const northBus = @import_module("oned_exch.csl", .{
  .zValues = &zValues,
  .buffer = &buffer,
  .pattern = pattern,
  .chunkSize = chunkSize,
  .paddedZDim = paddedZDim,

  .pos = 3,
  .dir = NORTH,
  .queueId = 3,
  .dataFin_task_id = northDataFin,
  .ctrlFin_task_id = northCtrlFin,
  .channel = northChannel,
  .callback = northFinTask,
  .senderCount = northSenderCount,
  .ctrlCallback = northCtrlFinTask,
  .constants = &permutedNorthConsts,
});

var sendCount: u16 = 0;
var recvCount: u16 = 0;
var iterationCount: u32 = 0;

var maxValue: f32 = 0.0;
var minValue: f32 = 0.0;

const accDsd = @get_dsd(mem1d_dsd, .{
  .tensor_access = |i|{zDim} -> accumulator[i]
});

const vpDsd = @get_dsd(mem1d_dsd, .{
  .tensor_access = |k|{zDim} -> vp[k]
});

const zValuesDsd0 = @get_dsd(mem1d_dsd, .{
  .tensor_access = |i|{zDim} -> zValues[0, zOffset + i]
});

const zValuesDsd1 = @get_dsd(mem1d_dsd, .{
  .tensor_access = |i|{zDim} -> zValues[1, zOffset + i]
});

// This function is called when the program completes communication in any one
// of the east, west, north, and south directions.
fn recvFin() void {
  recvCount += 1;

  // Don't proceed until we've finished communicating in _all_ four directions.
  if (recvCount != directionCount) {
    return;
  }

  recvCount = 0;

  // Each direction's communication module writes to a separate chunk of the
  // buffer, so the following function call performs a sum reduction across all
  // of these chunks.  This enables us to reuse this buffer for the next round
  // of `chunkSize` communication without forcing us to allocate one large
  // buffer for all chunks and for all four directions, which may require more
  // memory than is available at any given PE.
  reduceBuffer(recvChunkCounter * @as(i16, chunkSize));

  // The above code multiplies the source data with constants for neighbors in
  // the X and Y dimension, but we still need to multiply with the right
  // constants in the Z dimension.  Here, we keep track of the number of chunks
  // we've received so that we know when to start computing over the Z dim.
  recvChunkCounter += 1;

  // Note the difference in branch predicates below.  We want to continue
  // receiving until we've received `chunkSize` values `numChunks` number of
  // times.  However, the condition for calling `epilog()`, which processes
  // values in the Z dimension, checks whether we've finished _sending_.  This
  // way, we ensure that the _both_ sending and receiving code is fully complete
  // before we begin further processing.  This also ensures that only _one_ of
  // the `recvFin()` or `sendFin()` functions calls the `epilog()` code.
  if (recvChunkCounter != numChunks) {
    // Set the PE to again receive `chunkSize` values from all four directions.
    startReceiving();
  } else if (sendChunkCounter == numChunks) {
    // Remainder tasks after exchanging data in all four direction.
    epilog();
  }
}

// Just like the code to receive `chunkSize` elements need to be called for the
// total number of chunks, the sending code is also called multiple times so
// that each call sends `chunkSize` elements to its neighbors.
fn sendFin() void {
  sendCount += 1;

  // Don't proceed until we've finished sending to all four neighbors.
  if (sendCount != directionCount) {
    return;
  }

  sendCount = 0;
  sendChunkCounter += 1;

  // Note the difference in branch predicates below.  We want to continue
  // sending until we've sent `chunkSize` values `numChunks` number of times.
  // However, the condition for calling `epilog()`, which processes values in
  // the Z dimension, checks whether we've finished _receiving_.  This way, we
  // ensure that the _both_ sending and receiving code is fully complete before
  // we begin further processing.  This also ensures that only _one_ of the
  // `recvFin()` or `sendFin()` functions calls the `epilog()` code.
  if (sendChunkCounter != numChunks) {
    startSending(sendChunkCounter * @as(i16, chunkSize));
  } else if (recvChunkCounter == numChunks) {
    // Remainder tasks after exchanging data in all four direction.
    epilog();
  }
}

fn epilog() void {
  // Multiply shifted versions of zValues with various constants, before
  // accumulating them into `accumulator`.
  scaleWithZConsts();

  // Multiply by the velocity field vp
  //
  // Minimig - target_3d.c:30
  // vp[IDX3(i,j,k)]*lap
  //
  @fmuls(accDsd, accDsd, vpDsd);

  // Add 2x the value of the previous iteration (referred to as `u`) then
  // subtract the value from two iterations ago (referred to as `v`).
  // Since we want to keep track of values for _two_ iterations and not
  // just the previous iterations, we toggle between `zValues[0, :]`
  // and `zValues[1, :]`.
  //
  // Minimig - target_3d.c:30
  // If iterationCount is even, `zValues[0, :]` contains `v[IDX3_l(i,j,k)]`
  // and ``zValues[1, :]` contains `2.f*u[IDX3_l(i,j,k)]+vp[IDX3(i,j,k)]*lap`
  // (and vice-versa if iterationCount is odd).
  // This operation orresponds to `-v[IDX3_l(i,j,k)]` in:
  // ```
  // v[IDX3_l(i,j,k)] = 2.f*u[IDX3_l(i,j,k)]-v[IDX3_l(i,j,k)]+vp[IDX3(i,j,k)]*lap;
  // ```
  if (iterationCount & 1 == 0) {
    //add 2u
    @fmacs(accDsd, accDsd, zValuesDsd1, 2.0);
    @fsubs(zValuesDsd0, accDsd, zValuesDsd0);
  } else {
    //add 2u
    @fmacs(accDsd, accDsd, zValuesDsd0, 2.0);
    @fsubs(zValuesDsd1, accDsd, zValuesDsd1);
  }

  // At this point, we've finished a single iteration's computation.  We now add
  // the gaussian value to the wavefield, assuming this is the appropriate PE.
  //
  // Minimig - main.c:203 and data_setup.c:21-31
  // ```
  // kernel_add_source(grid, v, source, istep, sx, sy, sz);
  // ```
  //
  if (iterationCount < sourceLength) {
    if (isSourcePe) {
      const thisIterationIdx = iterationCount & 1;
      const offset = @as(u16, zOffset) + srcZ;
      zValues[thisIterationIdx, offset] += source[iterationCount];
    }
  }

  iterationCount += 1;

  // Are we done yet?  If not, start the next iteration by triggering the send
  // operation.
  if (iterationCount < iterations) {
    @activate(send);
  } else {
    // Now that we've finished executing the program, we have to perform four
    // things:
    // ref: hpc_apps/src/cslang/fd/task.csl
    // 1. Record the value of the timestamp counter, so that the host can
    // compute the difference and determine the number of cycles per element.

    // 2. Compute the minimum and maximum value of the wavefield for each PE's
    // local data, so that the host can simply compute the min and max of these
    // (reduced) values instead of computing the min and max over the entire
    // wavefield.

    // 3. Assuming this is the top-right PE, send the timestamp values
    f_checkpoint();
  }
}

// This function computes the maximum of the computed result.  It switches
// between the two `zValues` buffers depending on the executed iteration count.
//
// Minimig - data_setup.cc:49
fn computeMaxValue() f32 {
  var maxValue:f32 = math.NEGATIVE_INF_f32;
  const lastIterationIdx = 1 - (iterationCount & 1);

  if (lastIterationIdx == 0) {
    @fmaxs(&maxValue, maxValue, zValuesDsd0);
  } else {
    @fmaxs(&maxValue, maxValue, zValuesDsd1);
  }

  return maxValue;
}

// This function computes the _minimum_ of the computed result.  Since there is
// no instruction for computing the minimum and because we want to use DSDs
// (instead of a software loop), we first negate the result, compute the
// maximum, and negate the computed maximum (before negating the source values
// again so as to make this operation idempotent).
//
// Minimig - data_setup.cc:48
fn computeMinValue() f32 {
  var minValue:f32 = math.NEGATIVE_INF_f32;
  const lastIterationIdx = 1 - (iterationCount & 1);

  if (lastIterationIdx == 0) {
    @fnegs(zValuesDsd0, zValuesDsd0);
    @fmaxs(&minValue, minValue, zValuesDsd0);
    @fnegs(zValuesDsd0, zValuesDsd0);
  } else {
    @fnegs(zValuesDsd1, zValuesDsd1);
    @fmaxs(&minValue, minValue, zValuesDsd1);
    @fnegs(zValuesDsd1, zValuesDsd1);
  }

  return -minValue;
}

// The following are tasks that are activated when (asynchronous) send and
// reveive operations in various directions complete.  Each task funnels to
// either the `recvFin()` or the `sendFin()` function.  While it may _seem_
// better to activate just one task instead of four, we cannot do so since the
// hardware does not queue activations (instead, the hardware uses a single bit
// to track task activations).  Thus, depending on the sequence of task
// activations and executions, activating a task multiple times does not
// guarantee that the said will execute multiple times.
task eastFinTask() void {
  recvFin();
}

task westFinTask() void {
  recvFin();
}

task southFinTask() void {
  recvFin();
}

task northFinTask() void {
  recvFin();
}

task eastCtrlFinTask() void {
  sendFin();
}

task westCtrlFinTask() void {
  sendFin();
}

task southCtrlFinTask() void {
  sendFin();
}

task northCtrlFinTask() void {
  sendFin();
}

fn scaleWithZConsts() void {
  @comptime_assert(pattern == 5);

  // Ideally, we would express the following statements in a loop.  Since the
  // loop bound is comptime-known, the compiler would then unroll the loop for
  // us.  However, the current version of the compiler lacks the ability to
  // unroll loops if the bounds are comptime-known, so the following code is the
  // manually-unrolled version of the loop over `2 * pattern - 1`.
  //
  // Minimig - target_3d.c:3,6,9,12,15 and target_3d.c:30
  // `vp` and `2u` are folded into `zConsts` so this corresponds to:
  // ```
  //  2.f*u[IDX3_l(i,j,k)] + vp * (coef0*u[IDX3_l(i,j,k)] \
  //    +coefz[1]*(u[IDX3_l(i,j,k+1)]+u[IDX3_l(i,j,k-1)]) \
  //    +coefz[2]*(u[IDX3_l(i,j,k+2)]+u[IDX3_l(i,j,k-2)]) \
  //    +coefz[3]*(u[IDX3_l(i,j,k+3)]+u[IDX3_l(i,j,k-3)]) \
  //    +coefz[4]*(u[IDX3_l(i,j,k+4)]+u[IDX3_l(i,j,k-4)]))
  if (iterationCount & 1 != 0) {
    const srcZ = @get_dsd(mem1d_dsd, .{
      .tensor_access = |i|{zDim} -> zValues[0, i]
    });
    @fmacs(accDsd, accDsd, srcZ, zConsts[0]);

    const srcZ1 = @increment_dsd_offset(srcZ, 1, f32);
    @fmacs(accDsd, accDsd, srcZ1, zConsts[1]);

    const srcZ2 = @increment_dsd_offset(srcZ, 2, f32);
    @fmacs(accDsd, accDsd, srcZ2, zConsts[2]);

    const srcZ3 = @increment_dsd_offset(srcZ, 3, f32);
    @fmacs(accDsd, accDsd, srcZ3, zConsts[3]);

    const srcZ5 = @increment_dsd_offset(srcZ, 5, f32);
    @fmacs(accDsd, accDsd, srcZ5, zConsts[5]);

    const srcZ6 = @increment_dsd_offset(srcZ, 6, f32);
    @fmacs(accDsd, accDsd, srcZ6, zConsts[6]);

    const srcZ7 = @increment_dsd_offset(srcZ, 7, f32);
    @fmacs(accDsd, accDsd, srcZ7, zConsts[7]);

    const srcZ8 = @increment_dsd_offset(srcZ, 8, f32);
    @fmacs(accDsd, accDsd, srcZ8, zConsts[8]);
  } else {
    const srcZ = @get_dsd(mem1d_dsd, .{
      .tensor_access = |i|{zDim} -> zValues[1, i]
    });
    @fmacs(accDsd, accDsd, srcZ, zConsts[0]);

    const srcZ1 = @increment_dsd_offset(srcZ, 1, f32);
    @fmacs(accDsd, accDsd, srcZ1, zConsts[1]);

    const srcZ2 = @increment_dsd_offset(srcZ, 2, f32);
    @fmacs(accDsd, accDsd, srcZ2, zConsts[2]);

    const srcZ3 = @increment_dsd_offset(srcZ, 3, f32);
    @fmacs(accDsd, accDsd, srcZ3, zConsts[3]);

    const srcZ5 = @increment_dsd_offset(srcZ, 5, f32);
    @fmacs(accDsd, accDsd, srcZ5, zConsts[5]);

    const srcZ6 = @increment_dsd_offset(srcZ, 6, f32);
    @fmacs(accDsd, accDsd, srcZ6, zConsts[6]);

    const srcZ7 = @increment_dsd_offset(srcZ, 7, f32);
    @fmacs(accDsd, accDsd, srcZ7, zConsts[7]);

    const srcZ8 = @increment_dsd_offset(srcZ, 8, f32);
    @fmacs(accDsd, accDsd, srcZ8, zConsts[8]);
  }
}

fn reduceBuffer(offset: i16) void {
  const bufferDsd = @get_dsd(mem4d_dsd, .{
    .tensor_access = |i,j,k|{directionCount, pattern, chunkSize} -> buffer[i, j, k]
  });

  const accumulatorDsd = @get_dsd(mem4d_dsd, .{
    .tensor_access = |i,j,k|{directionCount, pattern, chunkSize} -> accumulator[k]
  });

  // Minimig - target_3d.c:4-14
  // This corresponds to the sum between each component of the laplacian
  // over x and y (buffer contains data received from each neighbor in all
  // 4 cardinal directions)
  const dstDsd = @increment_dsd_offset(accumulatorDsd, offset, f32);
  @fadds(dstDsd, dstDsd, bufferDsd);
}

fn startReceiving() void {
  // Put the PE in the receive mode for all four directions.
  eastBus.recvMode();
  westBus.recvMode();
  southBus.recvMode();
  northBus.recvMode();
}

fn startSending(offset: i16) void {
  // Asynchronously send data to neighbors in all four directions.
  eastBus.send(iterationCount, offset);
  westBus.send(iterationCount, offset);
  southBus.send(iterationCount, offset);
  northBus.send(iterationCount, offset);
}

fn startExchange() void {
  // Reset the chunk counters since we will be exchanging all chunks now.
  sendChunkCounter = 0;
  recvChunkCounter = 0;

  // We first need to put the PEs in receive mode before sending local data.
  // Starts Laplacian receive and multiplies on the fly for all 4 directions
  startReceiving();
  // Sends data from the previous iterations along all 4 directions
  startSending(0);
}

task sendTask() void {
  // zero out the accumulation buffer
  @fmovs(accDsd, 0.0);
  startExchange();
}

//----[MEMCPY]

// iteration count, we start the timer and trigger the broadcast of the source
// data to all the PE's neighbors.  A side effect of this design is that running
// the code with a different iteration count simply requires sending a new
// wavelet (with the new iteration count) from the host.
//
task f_comp() void {

  // WARNING: iterations is received by fn f_activate_comp called by
  // RPC mechanism

  timestamp.enable_tsc();
  timestamp.get_timestamp(&tscStartBuffer);
  @activate(send);
}
//--- END MEMCPY

comptime {
  @bind_local_task(sendTask, send);

  @bind_local_task(eastFinTask, eastFin);

  const eastRoute = routes.computeRoute(EAST, eastFirst, eastLast,
      eastNotNeedsPos3, eastPatternFirst, eastPatternLast);
  @set_local_color_config(eastChannel, eastRoute);

  @bind_local_task(westFinTask, westFin);

  const westRoute = routes.computeRoute(WEST, westFirst, westLast,
      westNotNeedsPos3, westPatternFirst, westPatternLast);
  @set_local_color_config(westChannel, westRoute);

  @bind_local_task(southFinTask, southFin);

  const southRoute = routes.computeRoute(SOUTH, southFirst, southLast,
      southNotNeedsPos3, southPatternFirst, southPatternLast);
  @set_local_color_config(southChannel, southRoute);

  @bind_local_task(northFinTask, northFin);

  const northRoute = routes.computeRoute(NORTH, northFirst, northLast,
      northNotNeedsPos3, northPatternFirst, northPatternLast);
  @set_local_color_config(northChannel, northRoute);

  @bind_local_task(eastCtrlFinTask, eastCtrlFin2);
  @bind_local_task(westCtrlFinTask, westCtrlFin2);
  @bind_local_task(northCtrlFinTask, northCtrlFin2);
  @bind_local_task(southCtrlFinTask, southCtrlFin2);
}

//----[MEMCPY]

// time marching is done, epilog calls f_checkpoint
// 1. recrod time stamps
// 2. compute max and min of zValues
// 3. prepare max, min and time stamps
fn f_checkpoint() void {

  // 1. Record the value of the timestamp counter, so that the host can
  // compute the difference and determine the number of cycles per element.
  timestamp.get_timestamp(&tscEndBuffer);
  timestamp.disable_tsc();

  // 2. Compute the minimum and maximum value of the wavefield for each PE's
  // local data, so that the host can simply compute the min and max of these
  // (reduced) values instead of computing the min and max over the entire
  // wavefield.
  maxValue = computeMaxValue();
  minValue = computeMinValue();

  // 3. prepares d2h_buf_f32[0:4]
  // D2H max/min
  d2h_buf_f32[0] = maxValue;
  d2h_buf_f32[1] = minValue;

  // D2H (timestamps)
  // d2h_buf_f32[2] = {tscStartBuffer[1], tscStartBuffer[0]}
  // d2h_buf_f32[3] = {tscEndBuffer[0], tscStartBuffer[2]}
  // d2h_buf_f32[4] = {tscEndBuffer[2], tscEndBuffer[1]}
  var lo_ : u16 = 0;
  var hi_ : u16 = 0;
  var word : u32 = 0;

  lo_ = tscStartBuffer[0];
  hi_ = tscStartBuffer[1];
  d2h_buf_f32[2] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

  lo_ = tscStartBuffer[2];
  hi_ = tscEndBuffer[0];
  d2h_buf_f32[3] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

  lo_ = tscEndBuffer[1];
  hi_ = tscEndBuffer[2];
  d2h_buf_f32[4] = @bitcast(f32, (@as(u32,hi_) << @as(u16,16)) | @as(u32, lo_) );

  // WARNING: the user must unblock cmd color for every PE
  sys_mod.unblock_cmd_stream();
}

// set number of iterations and activate f_comp task
fn f_activate_comp(iter_cnt: u32) void {
  iterations = iter_cnt;
  @activate(COMP);
}

// copy zValues to zout such that D2H can output zout
fn f_prepare_zout() void {
  // toggle = 1 - (iterations % 2)
  var toggle: i32 = 1 - (@as(i32,iterations) % 2);
  if (0 == toggle){
    mem_z_buf_dsd = @set_dsd_base_addr(mem_z_buf_dsd, @ptrcast([*]f32, &(zValues[0, zOffset])));
  }else{
    mem_z_buf_dsd = @set_dsd_base_addr(mem_z_buf_dsd, @ptrcast([*]f32, &(zValues[1, zOffset])));
  }
  @mov32(mem_zout_buf_dsd, mem_z_buf_dsd);

  // WARNING: the user must unblock cmd color for every PE
  sys_mod.unblock_cmd_stream();
}

comptime {
    @comptime_assert( sourceLength <= @as(u32,zDim));

    @bind_local_task(f_comp, COMP);

    @export_symbol(ptr_vp, "vp");
    @export_symbol(ptr_source, "source");
    @export_symbol(ptr_d2h_buf_f32, "maxmin_time");
    @export_symbol(ptr_zout, "zout");

    @export_symbol(f_activate_comp);
    @export_symbol(f_prepare_zout);
}

```

## run.py

```python
#!/usr/bin/env cs_python
# pylint: disable=too-many-function-args

import struct
import json
import os
import shutil
import subprocess
import time
from glob import glob
from pathlib import Path
from typing import List
from ic import computeGaussianSource
import numpy as np
from cmd_parser import parse_args

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

SIZE = 10
ZDIM = 10
PATTERN = 5
ITERATIONS = 10
DX = 20
arch_default = "wse2"
# "+5" for infrastructure of memcpy
# "+2" for a halo of size 1
FABRIC_WIDTH = SIZE + 2 + 5
FABRIC_HEIGHT = SIZE + 2

FILE_PATH = os.path.realpath(__file__)
MEMCPY_DIR = os.path.dirname(FILE_PATH)
DEPIPELINE_DIR = os.path.dirname(MEMCPY_DIR)
TEST_DIR = os.path.dirname(DEPIPELINE_DIR)
HPC_DIR = os.path.dirname(TEST_DIR)
ROOT_DIR = os.path.dirname(HPC_DIR)
CSL_DIR = os.path.join(ROOT_DIR, "cslang")
DRIVER = os.path.join(CSL_DIR, "build") + "/bin/cslc"

def float_to_hex(f):
  return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def sub_ts(words):
  return make_u48(words[3:]) - make_u48(words[0:3])

def cast_uint32(x):
  if isinstance(x, (np.float16, np.int16, np.uint16)):
    z = x.view(np.uint16)
    return np.uint32(z)
  if isinstance(x, (np.float32, np.int32, np.uint32)):
    return x.view(np.uint32)
  if isinstance(x, int):
    return np.uint32(x)
  if isinstance(x, float):
    z = np.float32(x)
    return z.view(np.uint32)

  raise RuntimeError(f"type of x {type(x)} is not supported")

def csl_compile(
    cslc: str,
    arch: str,
    width: int,
    height: int,
    core_fabric_offset_x: int, # fabric-offsets of the core
    core_fabric_offset_y: int,
    zDim: int,
    sourceLength: int,
    dx: int,
    srcX: int,
    srcY: int,
    srcZ: int,
    fabric_width: int,
    fabric_height: int,
    name: str,
    n_channels: int,
    width_west_buf: int,
    width_east_buf: int
)  -> List[str]:
  """Generate ELFs for the layout."""

  start = time.time()
  # CSL Compilation Step
  args = []
  args.append(cslc)
  args.append("layout.csl")
  args.append(f"--fabric-dims={fabric_width},{fabric_height}")
  args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
  args.append(f"--params=width:{width},height:{height},zDim:{zDim},sourceLength:{sourceLength}")
  args.append(f"--params=dx:{dx}")
  args.append(f"--params=srcX:{srcX},srcY:{srcY},srcZ:{srcZ}")
  args.append("--verbose")
  args.append(f"-o={name}_code")
  if arch is not None:
    args.append(f"--arch={arch}")
  args.append("--memcpy")
  args.append(f"--channels={n_channels}")
  args.append(f"--width-west-buf={width_west_buf}")
  args.append(f"--width-east-buf={width_east_buf}")
  print(f"subprocess.check_call(args = {args}")
  subprocess.check_call(args)

  end = time.time()
  print(f"Code compiled in {end-start}s")

  elf_paths = glob(f"{name}_code/bin/out_[0-9]*.elf")

  return elf_paths

def main():
  """Main method to run the example code."""

  args = parse_args()

  # Path to the CSLC driver
  cslc = DRIVER
  print(f"cslc = {cslc}")

  name = args.name
  dx = args.dx
  iterations = args.iterations

  n_channels = args.n_channels
  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  print(f"n_channels = {n_channels}")
  print(f"width_west_buf = {width_west_buf}, width_east_buf = {width_east_buf}")

  source, sourceLength = computeGaussianSource(iterations)
  print("Gaussian source computed")
  print(f"sourceLength = {sourceLength}")
  print(f"source = {source}")

  if args.skip_compile:
    # Parse the compile metadata
    with open(f"{name}_code/out.json", encoding="utf-8") as json_file:
      compile_data = json.load(json_file)

    size = int(compile_data["params"]["width"])
    zDim = int(compile_data["params"]["zDim"])
  else:
    size = args.size
    zDim = args.zDim

  width = size
  height = size

  fabric_offset_x = 1
  fabric_offset_y = 1

  # if WSE is the target, fabric_[width|height] must be the size of WSE
  if args.fabric_width is not None:
    fabric_width = args.fabric_width
  else:
    fabric_width = fabric_offset_x + 3 + width + 2 + 1 + width_west_buf + width_east_buf

  if args.fabric_height is not None:
    fabric_height = args.fabric_height
  else:
    fabric_height = fabric_offset_y + height + 1

  print(f"width = {width}, height={height}")
  print(f"fabric_offset_x = {fabric_offset_x}, fabric_offset_y={fabric_offset_y}")
  print(f"fabric_width = {fabric_width}, fabric_height={fabric_height}")

  assert fabric_width >= (fabric_offset_x + width + 5 + 1 + width_west_buf + width_east_buf)
  assert fabric_height >= (fabric_offset_y + height + 1)

  srcX = width // 2 - 5
  srcY = height // 2 - 5
  srcZ = zDim // 2 - 5
  assert srcX >= 0
  assert srcY >= 0
  assert srcZ >= 0
  print(f"srcX (x-coordinate of the source) = width/2 - 5  = {srcX}")
  print(f"srcY (y-coordinate of the source) = height/2 - 5 = {srcY}")
  print(f"srcZ (z-coordinate of the source) = zdim/2 - 5   = {srcZ}")

  if not args.skip_compile:
    print("Cleaned up existing elf files before compilation")
    elf_paths = glob(f"{name}_code_*.elf")
    for felf in elf_paths:
      os.remove(felf)

    core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
    core_fabric_offset_y = fabric_offset_y

    start = time.time()
    csl_compile(
        cslc, arch_default, width, height, core_fabric_offset_x, core_fabric_offset_y,
        zDim, sourceLength, dx, srcX, srcY, srcZ,
        fabric_width, fabric_height, name,
        n_channels,
        width_west_buf,
        width_east_buf)
    end = time.time()
    print(f"compilation of kernel in {end-start}s")
  else:
    print("skip-compile: No compilation, read existing ELFs")

  if args.skip_run:
    print("skip-run: early return")
    return

#----------- run the test --------

  # vp[h][w][l] = 10.3703699112
  vp_all = 10.3703699112
  vp = np.full(width*height*zDim, vp_all, dtype=np.float32)
  vp = vp.reshape(height, width, zDim)

  # source_all[h][w][l]
  source_all = np.zeros(width*height*zDim).reshape(width*height*zDim, 1).astype(np.float32)
  for tidx in range(sourceLength):
    #source_all[(srcY, srcX, tidx, 1)] = source[tidx]
    offset = srcY * width*zDim + srcX * zDim + tidx
    source_all[offset] = source[tidx]
  source_all = source_all.reshape(height, width, zDim)

#
# Step 2: the user creates CSRunner
#
  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  memcpy_order = MemcpyOrder.ROW_MAJOR
  dirname = f"{name}_code"
  runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

  sym_vp = runner.get_id("vp")
  sym_source = runner.get_id("source")
  sym_maxmin_time = runner.get_id("maxmin_time")
  sym_zout = runner.get_id("zout")

  runner.load()
  runner.run()

  start = time.time()
#
# Step 3: The user has to prepare the sequence of H2D/D2H/RPC
#
  # H2D vp[h][w][zDim]
  # vp is h-by-w-by-zDim in row-major
  runner.memcpy_h2d(sym_vp, vp.ravel(), 0, 0, width, height, zDim,
                    streaming=False, data_type=memcpy_dtype,
                    order=memcpy_order, nonblock=False)

  # H2D source[h][w][zDim]
  runner.memcpy_h2d(sym_source, source_all.ravel(), 0, 0, width, height, zDim,
                    streaming=False, data_type=memcpy_dtype,
                    order=memcpy_order, nonblock=False)

  # time marching: call f_activate_comp() to set num iters and start computation
  runner.launch("f_activate_comp", cast_uint32(iterations), nonblock=False)

  # D2H [h][w][6]
  maxmin_time_1d = np.zeros(height*width*6, np.float32)
  runner.memcpy_d2h(maxmin_time_1d, sym_maxmin_time, 0, 0, width, height, 6,
                    streaming=False, data_type=memcpy_dtype,
                    order=memcpy_order, nonblock=False)
  maxmin_time_hwl = maxmin_time_1d.reshape(height, width, 6)

  # prepare zout: call f_prepare_zout()
  runner.launch("f_prepare_zout", nonblock=False)

  # D2H [h][w][zDim]
  z_1d = np.zeros(height*width*zDim, np.float32)
  runner.memcpy_d2h(z_1d, sym_zout, 0, 0, width, height, zDim,
                    streaming=False, data_type=memcpy_dtype,
                    order=memcpy_order, nonblock=False)
  z_hwl = z_1d.reshape(height, width, zDim)

  runner.stop()
  end = time.time()

  print(f"Run done in {end-start}s")

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

#
# step 4: verification
#
  # D2H(max/min)
  # d2h_buf_f32[0] = maxValue
  # d2h_buf_f32[1] = minValue
  # D2H (timestamps)
  # d2h_buf_f32[2] = {tscStartBuffer[1], tscStartBuffer[0]}
  # d2h_buf_f32[3] = {tscEndBuffer[0], tscStartBuffer[2]}
  # d2h_buf_f32[4] = {tscEndBuffer[2], tscEndBuffer[1]}
  maxValues_d2h = np.zeros(width*height).reshape(height, width).astype(np.float32)
  for h in range(height):
    for w in range(width):
      maxValues_d2h[(h, w)] = maxmin_time_hwl[(h, w, 0)]

  minValues_d2h = np.zeros(width*height).reshape(height, width).astype(np.float32)
  for h in range(height):
    for w in range(width):
      minValues_d2h[(h, w)] = maxmin_time_hwl[(h, w, 1)]

  computedMax = maxValues_d2h.max()
  computedMin = minValues_d2h.min()
  print(f"[computed] min_d2h: {computedMin}, max_d2h: {computedMax}")

  timestamp_d2h = np.zeros(width*height*6).reshape(width, height, 6).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      hex_t0 = int(float_to_hex(maxmin_time_hwl[(h, w, 2)]), base=16)
      hex_t1 = int(float_to_hex(maxmin_time_hwl[(h, w, 3)]), base=16)
      hex_t2 = int(float_to_hex(maxmin_time_hwl[(h, w, 4)]), base=16)
      timestamp_d2h[(w, h, 0)] = hex_t0 & 0x0000ffff
      timestamp_d2h[(w, h, 1)] = (hex_t0 >> 16) & 0x0000ffff
      timestamp_d2h[(w, h, 2)] = hex_t1 & 0x0000ffff
      timestamp_d2h[(w, h, 3)] = (hex_t1 >> 16) & 0x0000ffff
      timestamp_d2h[(w, h, 4)] = hex_t2 & 0x0000ffff
      timestamp_d2h[(w, h, 5)] = (hex_t2 >> 16) & 0x0000ffff
  tsc_tensor_d2h = np.zeros(6).astype(np.uint16)
  tsc_tensor_d2h[0] = timestamp_d2h[(width-1, 0, 0)]
  tsc_tensor_d2h[1] = timestamp_d2h[(width-1, 0, 1)]
  tsc_tensor_d2h[2] = timestamp_d2h[(width-1, 0, 2)]
  tsc_tensor_d2h[3] = timestamp_d2h[(width-1, 0, 3)]
  tsc_tensor_d2h[4] = timestamp_d2h[(width-1, 0, 4)]
  tsc_tensor_d2h[5] = timestamp_d2h[(width-1, 0, 5)]

  print(f"tsc_tensor_d2h = {tsc_tensor_d2h}")
  cycles = sub_ts(tsc_tensor_d2h)
  cycles_per_element = cycles / (iterations * zDim)
  print(f"cycles per element = {cycles_per_element}")

  zMax_d2h = z_hwl.max()
  zMin_d2h = z_hwl.min()
  print(f"[computed] zMin_d2h: {zMin_d2h}, zMax_d2h: {zMax_d2h}")

  if zDim == 10 and size == 10 and iterations == 10:
    print("[verification] w=h=zdim=10, iters = 10, check golden vector")
    np.testing.assert_allclose(computedMin, -1.3100899, atol=0.01, rtol=0)
    np.testing.assert_allclose(computedMax, 1200.9414062, atol=0.01, rtol=0)
    print("\nSUCCESS!")
  elif zDim == 10 and size == 10 and iterations == 2:
    print("[verification] w=h=zdim=10, iters = 2, check golden vector")
    np.testing.assert_allclose(computedMin, -0.0939295, atol=0.01, rtol=0)
    np.testing.assert_allclose(computedMax, 57.403816, atol=0.01, rtol=0)
    print("\nSUCCESS!")
  else:
    print("Results are not checked for those parameters")
    assert False

if __name__ == "__main__":
  main()

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc ./layout.csl --arch=wse2 --fabric-dims=17,12 --fabric-offsets=4,1 \
-o=out_code --params=width:10,height:10,zDim:10,sourceLength:10,dx:20 \
--params=srcX:0,srcY:0,srcZ:0 --verbose --memcpy --channels=1 \
--width-west-buf=0 --width-east-buf=0
cs_python run.py --name out \
--iterations=10 --dx=20 --skip-compile

```
[previous

Cholesky](csl/code-examples/benchmark-cholesky.md "previous page")
[next

Bandwidth Test](csl/code-examples/benchmark-bandwidth-test.md "next page")
