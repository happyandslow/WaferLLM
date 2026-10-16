---
title: "Conway’s Game of Life"
url: "https://sdk.cerebras.net/csl/code-examples/benchmark-game-of-life"
crawled_at: "2026-02-14 23:55:25"
---

# Conway’s Game of Life

> Source: [https://sdk.cerebras.net/csl/code-examples/benchmark-game-of-life](https://sdk.cerebras.net/csl/code-examples/benchmark-game-of-life)

This program implements
[Conway’s Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
on the WSE.

Conway’s Game of Life is a cellular automaton which evolves on a 2D grid of
square cells. Each cell is in one of two possible states, LIVE or DEAD.
Every cell interacts with its neighbors, which are the cells horziontally,
vertically, or diagonally adjacent. At each step in time, the following
transitions occur:

- Any LIVE cell with fewer than two LIVE neighbours becomes a DEAD cell.
- Any LIVE cell with two or three LIVE neighbours stays a LIVE cell.
- Any LIVE cell with more than three LIVE neighbours becomes a DEAD cell.
- Any DEAD cell with exactly three LIVE neighbours becomes a LIVE cell.

This program implements the Game of Life be assigning one cell to each PE.
Zero boundary conditions are used, and thus the neighbors of a border PE that
fall outside of the program rectangle are treaded as always DEAD.

In each generation, each PE sends its state to its four N, S, E, and W
neighbors. Each PE receives the state of its four N, S, E, and W neighbors, and
also forwards the received state from its N and S neighbors to its E and W
neighbors. Thus, each PE receives from its E and W links both the state of its
E and W adjacent neighbors, as well as its four diagonal neighbors.

The program implements two initial conditions, `random` and `glider`.
`random` randomly initializes the state of all cells. `glider` generates
several glider objects across the grid. The initial condition can be set with
the `--initial-state` flag.

The `--show-ascii-animation` flag will generate an ASCII animation of the
cellular automoton’s evolution when the program is complete.
`--save-animation` will save a GIF of the automoton’s evolution.

## layout.csl

```csl
// kernel dimensions
param x_dim: i16;
param y_dim: i16;

// Colors
const east_color_0:  color = @get_color(0);
const east_color_1:  color = @get_color(1);
const west_color_0:  color = @get_color(2);
const west_color_1:  color = @get_color(3);
const south_color_0: color = @get_color(4);
const south_color_1: color = @get_color(5);
const north_color_0: color = @get_color(6);
const north_color_1: color = @get_color(7);

// This example uses x_dim x y_dim PEs
const memcpy = @import_module("<memcpy/get_params>", .{
  .width = x_dim,
  .height = y_dim
});

layout {
  // PE coordinates are (column, row)
  @set_rectangle(x_dim, y_dim);

  const x_even_params = .{
    .send_east_color = east_color_0, .send_west_color = west_color_1,
    .recv_east_color = west_color_0, .recv_west_color = east_color_1,
  };

  const x_odd_params = .{
    .send_east_color = east_color_1, .send_west_color = west_color_0,
    .recv_east_color = west_color_1, .recv_west_color = east_color_0,
  };

  const y_even_params = .{
    .send_south_color = south_color_0, .send_north_color = north_color_1,
    .recv_south_color = north_color_0, .recv_north_color = south_color_1,
  };

  const y_odd_params = .{
    .send_south_color = south_color_1, .send_north_color = north_color_0,
    .recv_south_color = north_color_1, .recv_north_color = south_color_0,
  };

  for (@range(i16, x_dim)) |pe_x| {
    const west_edge = (pe_x == 0);
    const east_edge = (pe_x == x_dim-1);

    const x_color_params = if (pe_x % 2 == 0) x_even_params else x_odd_params;

    const x_params = @concat_structs(
                       .{ .is_west_edge = west_edge, .is_east_edge = east_edge,
                          .memcpy_params = memcpy.get_params(pe_x) },
                       x_color_params
                     );

    for (@range(i16, y_dim)) |pe_y| {
      const north_edge = (pe_y == 0);
      const south_edge = (pe_y == y_dim-1);

      const y_color_params = if (pe_y % 2 == 0) y_even_params else y_odd_params;

      const y_params = @concat_structs(
                         .{ .is_north_edge = north_edge, .is_south_edge = south_edge },
                         y_color_params
                       );

      @set_tile_code(pe_x, pe_y, "pe_program.csl", @concat_structs(x_params, y_params));
    }
  }

  // Create route values
  const RX_R_TX_E = .{ .rx = .{ RAMP  }, .tx = .{ EAST  }};
  const RX_W_TX_R = .{ .rx = .{ WEST  }, .tx = .{ RAMP  }};
  const RX_R_TX_W = .{ .rx = .{ RAMP  }, .tx = .{ WEST  }};
  const RX_E_TX_R = .{ .rx = .{ EAST  }, .tx = .{ RAMP  }};

  const RX_R_TX_S = .{ .rx = .{ RAMP  }, .tx = .{ SOUTH }};
  const RX_N_TX_R = .{ .rx = .{ NORTH }, .tx = .{ RAMP  }};
  const RX_R_TX_N = .{ .rx = .{ RAMP  }, .tx = .{ NORTH }};
  const RX_S_TX_R = .{ .rx = .{ SOUTH }, .tx = .{ RAMP  }};

  for (@range(i16, x_dim)) |pe_x| {
    for (@range(i16, y_dim)) |pe_y| {
      if (pe_x % 2 == 0) {
        @set_color_config(pe_x, pe_y, east_color_0, .{ .routes = RX_R_TX_E });
        @set_color_config(pe_x, pe_y, east_color_1, .{ .routes = RX_W_TX_R });
        @set_color_config(pe_x, pe_y, west_color_0, .{ .routes = RX_E_TX_R });
        @set_color_config(pe_x, pe_y, west_color_1, .{ .routes = RX_R_TX_W });
      } else {
        @set_color_config(pe_x, pe_y, east_color_0, .{ .routes = RX_W_TX_R });
        @set_color_config(pe_x, pe_y, east_color_1, .{ .routes = RX_R_TX_E });
        @set_color_config(pe_x, pe_y, west_color_0, .{ .routes = RX_R_TX_W });
        @set_color_config(pe_x, pe_y, west_color_1, .{ .routes = RX_E_TX_R });
      }

      if (pe_y % 2 == 0) {
        @set_color_config(pe_x, pe_y, south_color_0, .{ .routes = RX_R_TX_S });
        @set_color_config(pe_x, pe_y, south_color_1, .{ .routes = RX_N_TX_R });
        @set_color_config(pe_x, pe_y, north_color_0, .{ .routes = RX_S_TX_R });
        @set_color_config(pe_x, pe_y, north_color_1, .{ .routes = RX_R_TX_N });
      } else {
        @set_color_config(pe_x, pe_y, south_color_0, .{ .routes = RX_N_TX_R });
        @set_color_config(pe_x, pe_y, south_color_1, .{ .routes = RX_R_TX_S });
        @set_color_config(pe_x, pe_y, north_color_0, .{ .routes = RX_R_TX_N });
        @set_color_config(pe_x, pe_y, north_color_1, .{ .routes = RX_S_TX_R });
      }
    }
  }

  // export symbol names
  @export_name("states", [*]u32, true);
  @export_name("generate", fn(u16)void);
}

```

## pe\_program.csl

```csl
param memcpy_params: comptime_struct;

param is_east_edge:  bool;
param is_west_edge:  bool;
param is_south_edge: bool;
param is_north_edge: bool;

// Colors
param send_east_color:  color;
param send_west_color:  color;
param send_south_color: color;
param send_north_color: color;

param recv_east_color:  color;
param recv_west_color:  color;
param recv_south_color: color;
param recv_north_color: color;

// Queue IDs
const send_east_oq:  output_queue = @get_output_queue(2);
const send_west_oq:  output_queue = @get_output_queue(3);
const send_south_oq: output_queue = @get_output_queue(4);
const send_north_oq: output_queue = @get_output_queue(5);

const recv_east_iq:  input_queue  = @get_input_queue(2);
const recv_west_iq:  input_queue  = @get_input_queue(3);
const recv_south_iq: input_queue  = @get_input_queue(4);
const recv_north_iq: input_queue  = @get_input_queue(5);

// Task IDs
const send_task_id:           local_task_id = @get_local_task_id(8);
const sync_send_task_id:      local_task_id = @get_local_task_id(9);
const sync_fwd_task_id:       local_task_id = @get_local_task_id(10);
const start_next_gen_task_id: local_task_id = @get_local_task_id(11);
const fwd_east_west_task_id:  local_task_id = @get_local_task_id(12);
const exit_task_id:           local_task_id = @get_local_task_id(13);

// On WSE-2, data task IDs are created from colors; on WSE-3, from input queues
const recv_east_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(recv_east_color)
  else if (@is_arch("wse3")) @get_data_task_id(recv_east_iq);
const recv_west_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(recv_west_color)
  else if (@is_arch("wse3")) @get_data_task_id(recv_west_iq);
const recv_south_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(recv_south_color)
  else if (@is_arch("wse3")) @get_data_task_id(recv_south_iq);
const recv_north_task_id: data_task_id =
  if      (@is_arch("wse2")) @get_data_task_id(recv_north_color)
  else if (@is_arch("wse3")) @get_data_task_id(recv_north_iq);

// memcpy module provides infrastructure for copying data
// and launching functions from the host
const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
const layout_mod = @import_module("<layout>");

const MAX_GENERATIONS = 1000; // Max num total generations that can be stored

// Number of neighboring PEs for this cell
const num_neighbors: u16 = (if (is_west_edge)  0 else 1) + (if (is_east_edge)  0 else 1) // W, E
                         + (if (is_north_edge) 0 else 1) + (if (is_south_edge) 0 else 1) // N, S
                         + (if (is_north_edge or is_west_edge) 0 else 1)  // NW
                         + (if (is_north_edge or is_east_edge) 0 else 1)  // NE
                         + (if (is_south_edge or is_west_edge) 0 else 1)  // SW
                         + (if (is_south_edge or is_east_edge) 0 else 1); // SE

const num_west_nbrs: u16 = if (is_west_edge) 0
                           else (1 + (if (is_north_edge) 0 else 1) + (if (is_south_edge) 0 else 1));
const num_east_nbrs: u16 = if (is_east_edge) 0
                           else (1 + (if (is_north_edge) 0 else 1) + (if (is_south_edge) 0 else 1));

const num_ns_nbrs: u16 = (if (is_north_edge) 0 else 1) + (if (is_south_edge) 0 else 1);

var iters: u16 = 0; // Number of generations for current run
var current_iter: u16 = 0; // Track num generations completed so far

// Store states of all cells for each generation
var states: [MAX_GENERATIONS]u32;
var states_ptr: [*]u32 = &states;
var state_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> states[i] });

// For current generation, track received states from neighbors
var num_recv: u16 = 0;
var current_sum: u32 = 0;

var num_west_recv: u16 = 0;
var num_east_recv: u16 = 0;
var num_ns_recv: u16 = 0;

// Store values received from N and S to forward E and W
var fwd_vals: [2]u32;

// DSDs for sending values to N, S, E, W neighbors
const send_west_dsd = @get_dsd(fabout_dsd, .{
  .fabric_color = send_west_color, .extent = 1, .output_queue = send_west_oq });
const send_east_dsd = @get_dsd(fabout_dsd, .{
  .fabric_color = send_east_color, .extent = 1, .output_queue = send_east_oq });
const send_north_dsd = @get_dsd(fabout_dsd, .{
  .fabric_color = send_north_color, .extent = 1, .output_queue = send_north_oq });
const send_south_dsd = @get_dsd(fabout_dsd, .{
  .fabric_color = send_south_color, .extent = 1, .output_queue = send_south_oq });

// Send current state to all four neighbors
task send() void {
  if (!is_north_edge) @fmovs(send_north_dsd, state_dsd, .{ .async = true });
  if (!is_south_edge) @fmovs(send_south_dsd, state_dsd, .{ .async = true });

  // When sending to E and W finishes, allow sync_fwd task to proceed
  // sync_fwd allows us to begin forwarding states received from N/ S to E/ W
  if (!is_west_edge) @fmovs(send_west_dsd, state_dsd,
                            .{ .async = true, .unblock = sync_fwd_task_id });
  if (!is_east_edge) @fmovs(send_east_dsd, state_dsd,
                            .{ .async = true, .activate = sync_fwd_task_id });

  if (is_west_edge) @unblock(sync_fwd_task_id);
  if (is_east_edge) @activate(sync_fwd_task_id);

  // Do no send again until we forward N/ S recvs to E/ W neighbors
  @block(send_task_id);
}

// Guarantee that we do not begin forwarding N/ S recvs to E/ W neighbors
// until E/ W sends from our cell complete
task sync_fwd() void {
  @block(sync_fwd_task_id);
  @unblock(fwd_east_west_task_id);
}

// Forward states received from N/ S neighbors to E/ W neighbors
task fwd_east_west() void {
  // fwd_vals[0] is N neighbor forwarded to E and W
  // fwd_vals[1] is S neighbor forwarded to E and W
  // if we are N edge, there is no N neighbor to forward, so we access only fwd_vals[1]
  const offset = if (is_north_edge) 1 else 0;
  const fwd_dsd = @get_dsd(mem1d_dsd,
                           .{ .tensor_access = |i|{num_ns_nbrs} -> fwd_vals[i + offset] });

  // When forwarding to E and W finishes, allow sync_send task to proceed
  // sync_send allows us to begin sending next generation
  if (!is_west_edge) @fmovs(send_west_dsd, fwd_dsd,
                            .{ .async = true, .unblock = sync_send_task_id });
  if (!is_east_edge) @fmovs(send_east_dsd, fwd_dsd,
                            .{ .async = true, .activate = sync_send_task_id });

  if (is_west_edge) @unblock(sync_send_task_id);
  if (is_east_edge) @activate(sync_send_task_id);

  // Do not forward again until we complete next generation E/ W sends
  // from our cell to neighbors
  @block(fwd_east_west_task_id);
}

// Guarantee that we do not begin sending next generation until we have forwarded
// all neighbors from current generation
task sync_send() void {
  @block(sync_send_task_id);
  @unblock(send_task_id);
}

// In each generation, PE will receive from W up to three times:
// W neighbor, NW neighbor, and SW neighbor
task recv_west(val: u32) void {
  num_west_recv += 1;
  num_recv += 1;
  current_sum += val;

  // If we have received from all W neighbors, block to prevent
  // any activations until we begin next generation
  if (num_west_recv == num_west_nbrs) @block(recv_west_task_id);
  // If we have received from all neighbors, begin next generation
  if (num_recv == num_neighbors) @activate(start_next_gen_task_id);
}

// In each generation, PE will receive from E up to three times
// E neighbor, NE neighbor, and SE neighbor
task recv_east(val: u32) void {
  num_east_recv += 1;
  num_recv += 1;
  current_sum += val;

  // If we have received from all E neighbors, block to prevent
  // any activations until we begin next generation
  if (num_east_recv == num_east_nbrs) @block(recv_east_task_id);
  // If we have received from all neighbors, begin next generation
  if (num_recv == num_neighbors) @activate(start_next_gen_task_id);
}

// In each generation, PE will receive from N if there is N neighbor
task recv_north(val: u32) void {
  num_ns_recv += 1;
  num_recv += 1;
  current_sum += val;

  // Per generation, we only receive from N once. Block to prevent any
  // activations until we begin next generation.
  @block(recv_north_task_id);

  // Store value received from N to forward to E and W neighbors
  fwd_vals[0] = val;

  // If we have received from N and S, fwd to E and W neighbors
  if (num_ns_recv == num_ns_nbrs) @activate(fwd_east_west_task_id);
  // If we have received from all neighbors, begin next generation
  if (num_recv == num_neighbors) @activate(start_next_gen_task_id);
}

// In each generation, PE will receive from S if there is S neighbor
task recv_south(val: u32) void {
  num_ns_recv += 1;
  num_recv += 1;
  current_sum += val;

  // Per generation, we only receive from S once. Block to prevent any
  // activations until we begin next generation.
  @block(recv_south_task_id);

  // Store value received from S to forward to E and W neighbors
  fwd_vals[1] = val;

  // If we have received from N and S, fwd to E and W neighbors
  if (num_ns_recv == num_ns_nbrs) @activate(fwd_east_west_task_id);
  // If we have received from all neighbors, begin next generation
  if (num_recv == num_neighbors) @activate(start_next_gen_task_id);
}

// Update current state and begin sending next generation to neighbors
task start_next_gen() void {

  current_iter += 1;
  state_dsd = @increment_dsd_offset(state_dsd, 1, u32);

  // Previous generation of cell is alive
  if (states[current_iter-1] == 1) {
    states[current_iter] = if (current_sum == 2 or current_sum == 3) 1 else 0;
  // Previous generation of cell is dead
  } else {
    states[current_iter] = if (current_sum == 3) 1 else 0;
  }

  if (current_iter == iters - 1) {
    @activate(exit_task_id);
  } else {
    current_sum = 0;
    num_recv = 0;
    num_west_recv = 0;
    num_east_recv = 0;
    num_ns_recv = 0;
    @unblock(recv_west_task_id);
    @unblock(recv_east_task_id);
    @unblock(recv_north_task_id);
    @unblock(recv_south_task_id);
    @activate(send_task_id);
  }
}

task exit() void {
  sys_mod.unblock_cmd_stream();
}

fn generate(num_gen: u16) void {
  // Set number of generations for current run
  iters = num_gen;
  @assert(iters <= MAX_GENERATIONS);

  // Begin sending to neighbors
  @activate(send_task_id);
}

comptime {
  @bind_local_task(send, send_task_id);
  @bind_local_task(sync_send, sync_send_task_id);
  @bind_local_task(sync_fwd, sync_fwd_task_id);
  @bind_local_task(start_next_gen, start_next_gen_task_id);
  @bind_local_task(fwd_east_west, fwd_east_west_task_id);
  @bind_local_task(exit, exit_task_id);

  @bind_data_task(recv_west,  recv_west_task_id);
  @bind_data_task(recv_east,  recv_east_task_id);
  @bind_data_task(recv_north, recv_north_task_id);
  @bind_data_task(recv_south, recv_south_task_id);

  @block(sync_send_task_id);
  @block(sync_fwd_task_id);

  // Will only become unbocked after first executoin of sync_fwd
  @block(fwd_east_west_task_id);

  // On WSE-3, we must explicitly initialize input and output queues
  if (@is_arch("wse3")) {
    @initialize_queue(send_west_oq,  .{ .color = send_west_color });
    @initialize_queue(send_east_oq,  .{ .color = send_east_color });
    @initialize_queue(send_north_oq, .{ .color = send_north_color });
    @initialize_queue(send_south_oq, .{ .color = send_south_color });

    @initialize_queue(recv_west_iq,  .{ .color = recv_west_color });
    @initialize_queue(recv_east_iq,  .{ .color = recv_east_color });
    @initialize_queue(recv_north_iq, .{ .color = recv_north_color });
    @initialize_queue(recv_south_iq, .{ .color = recv_south_color });
  }

  @export_symbol(states_ptr, "states");
  @export_symbol(generate);
}

```

## run.py

```python
#!/usr/bin/env cs_python

import argparse
import json
import subprocess
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

matplotlib.use('Agg')

def game_of_life_ref(initial_state, num_generations):
  """Compute reference to check WSE result for game of life generation"""

  x_dim = initial_state.shape[0]
  y_dim = initial_state.shape[1]

  ref_states = np.zeros((x_dim, y_dim, num_generations))
  ref_states[:,:,0] = initial_state

  for gen in range(1, num_generations):
    for i in range(x_dim):
      for j in range(y_dim):
        total = (0 if (i == 0)                           else ref_states[i-1,j,  gen-1]) \
              + (0 if (i == x_dim-1)                     else ref_states[i+1,j,  gen-1]) \
              + (0 if (j == 0)                           else ref_states[i,  j-1,gen-1]) \
              + (0 if (j == y_dim-1)                     else ref_states[i,  j+1,gen-1]) \
              + (0 if ((i == 0)     or (j == 0))         else ref_states[i-1,j-1,gen-1]) \
              + (0 if ((i == 0)     or (j == y_dim-1))   else ref_states[i-1,j+1,gen-1]) \
              + (0 if ((i == x_dim-1) or (j == 0))       else ref_states[i+1,j-1,gen-1]) \
              + (0 if ((i == x_dim-1) or (j == y_dim-1)) else ref_states[i+1,j+1,gen-1])

        if (ref_states[i, j, gen-1] == 1):
          ref_states[i, j, gen] = 1 if (total in (2, 3)) else 0
        else:
          ref_states[i, j, gen] = 1 if (total == 3) else 0

  return ref_states

def show_ascii_animation(states):
  """Generate a command-line ASCII animation"""

  num_generations = states.shape[2]
  try:
    for i in range(num_generations):
      subprocess.run(['clear'], shell=True, check=True)
      print(f'Generation {i}:\n')
      for row in states[:, :, i]:
        print(' '.join(['#' if cell else '.' for cell in row]))
      print('\nPress Ctrl+C to exit.')
      time.sleep(0.1)  # Wait for 0.1 seconds before displaying the next frame
  except KeyboardInterrupt:
    print('\nAnimation stopped.')

def save_animation(states, fname):
  """Save an animation as a GIF"""

  fig, ax = plt.subplots()
  ax.set_xticks([])
  ax.set_yticks([])
  ax.axis('off')

  frame_image = ax.imshow(states[:, :, 0], cmap='Greys', vmin=0, vmax=1)

  def update_plot(frame_index):
    frame_image.set_data(states[:, :, frame_index])
    return [frame_image]

  anim = FuncAnimation(
    fig,
    update_plot,
    frames=states.shape[2],
    interval=100,  # 0.1 seconds per frame
    blit=True
  )

  output_file = fname + '.gif'
  anim.save(output_file, writer=PillowWriter(fps=10))

def create_initial_state(state_type, x_dim, y_dim):
  """Generate intitial state for Game of Life"""

  initial_state = np.zeros((x_dim, y_dim), dtype=np.uint32)

  if state_type == 'glider':
    assert x_dim >= 4 and y_dim >=4, \
           'For glider initial state, x_dim and y_dim must be at least 4'

    glider = np.array([[0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1]])

    for i in range(x_dim//4):
      for j in range(y_dim//4):
        if i%2 == 0 and j%2 == 0:
          initial_state[4*i:4*i+3, 4*j:4*j+3] = glider
        elif i%2 == 0 and j%2 == 1:
          initial_state[4*i:4*i+3, 4*j:4*j+3] = glider[:,::-1]
        elif i%2 == 1 and j%2 == 0:
          initial_state[4*i:4*i+3, 4*j:4*j+3] = glider[::-1,:]
        elif i%2 == 1 and j%2 == 1:
          initial_state[4*i:4*i+3, 4*j:4*j+3] = glider[::-1,:]

  else: # state_type == 'random'
    np.random.seed(seed=7)
    initial_state = np.random.binomial(1, 0.5, (x_dim, y_dim)).astype(np.uint32)

  return initial_state

def main():
  """Main method to run the example code."""

  # Read arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', help='the test compile output dir', required=True)
  parser.add_argument('--cmaddr', help='IP:port for CS system')
  parser.add_argument('--iters', type=int, default=10, help='Number of generations (default: 10)')
  parser.add_argument('--initial-state', choices=['glider', 'random'], default='glider',
    help='Specify the initial state of the system (default: glider)'
  )
  parser.add_argument('--save-animation', action='store_true',
    help="Save animated GIF of states"
  )
  parser.add_argument('--show-ascii-animation', action='store_true',
    help="Show ascii animation of states"
  )
  args = parser.parse_args()

  # Get matrix dimensions from compile metadata
  with open(f'{args.name}/out.json', encoding='utf-8') as json_file:
    compile_data = json.load(json_file)

  # PE grid dimensions
  x_dim = int(compile_data['params']['x_dim'])
  y_dim = int(compile_data['params']['y_dim'])

  # Number of generations
  iters = args.iters

  initial_state = create_initial_state(args.initial_state, x_dim, y_dim)

  # Construct a runner using SdkRuntime
  runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

  states_symbol = runner.get_id('states')

  # Load and run the program
  runner.load()
  runner.run()

  print('Copy initial state to device...')
  # Copy initial state into all PEs
  runner.memcpy_h2d(states_symbol, initial_state.flatten(), 0, 0, x_dim, y_dim, 1,
    streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
    nonblock=False)

  print(f'Run for {iters} generations...')
  # Launch the generate function on device
  runner.launch('generate', np.uint16(iters), nonblock=False)

  # Copy states back
  states_result = np.zeros([x_dim * y_dim * iters], dtype=np.uint32)
  runner.memcpy_d2h(states_result, states_symbol, 0, 0, x_dim, y_dim, iters, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

  # Stop the program
  runner.stop()

  print('Create output...')

  # Reshape states results to x_dim x y_dim frames
  all_states = states_result.reshape((x_dim, y_dim, iters))

  # Loop through the frames and display them
  if args.show_ascii_animation:
    show_ascii_animation(all_states)

  # Generate animated GIF of generations
  if args.save_animation:
    save_animation(all_states, 'game_of_life')

  print('Create reference solution...')
  ref_states = game_of_life_ref(initial_state, iters)

  # Test that wafer output is equal to the reference
  np.testing.assert_equal(ref_states, all_states)
  print('SUCCESS!')

if __name__ == '__main__':
  main()

```

## commands.sh

```shell
#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=19,14 --fabric-offsets=4,1 \
--params=x_dim:12,y_dim:12 --memcpy --channels=1 -o out
cs_python run.py --name out --initial-state glider --iters 20

```
[previous

3D FFT](csl/code-examples/benchmark-fft-3d.md "previous page")
[next

CSL Language Guide](csl/language_index.md "next page")
