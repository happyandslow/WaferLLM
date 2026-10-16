#!/usr/bin/env python
"""
SDK debug shell.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

import cerebras.sdk.sdk_debug_instr_trace as sdk_debug_instr_trace
import cerebras.sdk.sdk_debug_wavelet_trace as sdk_debug_wavelet_trace
import cerebras.sdk.sdk_smoke as sdk_smoke
from cerebras.gen.build_paths import get_cslang_dir

# from click_shell import shell

SIMFAB_JSON_LOG = "simfab_traces.json"


# TODO: Enable shell once click_shell module is in toolchain.
# @shell(prompt='sdk_debug> ', intro='')
@click.group()
def sdk_debug_app():
    """
    Root group for the app.
    """

@sdk_debug_app.command(
    context_settings={"ignore_unknown_options":True,"allow_extra_args":True},
    add_help_option=False, hidden=True)
@click.pass_context
def compile(ctx): # pylint: disable=redefined-builtin
    """
    Run the CSL compiler.
    """
    # locate the CSL compiler
    cslc_frontend = Path(get_cslang_dir()) / "bin/cslc"
    assert cslc_frontend.exists(), f"CSL compiler frontend ({cslc_frontend}) not found"

    # make the compiler think it is called `cslc` to avoid user confusion
    # become the compiler
    os.execv(cslc_frontend, ["cslc"] + ctx.args)


@sdk_debug_app.command()
@click.argument('csl_examples_dir', nargs=-1)
@click.pass_context
def smoke(ctx, csl_examples_dir): # pylint: disable=redefined-builtin
    """
    Run the smoke tests.
    """
    if len(csl_examples_dir) > 0:
        csl_examples_dir = Path(csl_examples_dir[0])
    else:
        print("Smoke test requires path to csl examples directory as argument.")
        sys.exit(1)

    if not csl_examples_dir.is_dir():
        raise click.BadArgumentUsage(f"{csl_examples_dir} is not a directory.")

    sdk_smoke.main(csl_examples_dir)


X_COORD = "X coordinate of PE"
Y_COORD = "Y coordinate of PE"

@sdk_debug_app.group()
@click.option(
    '--artifact_dir',
    help="Directory for elf files and simfab json logs",
    required=True,
    type=click.Path(exists=True)
)
@click.option('--x', help=X_COORD, type=int, required=True)
@click.option('--y', help=Y_COORD, type=int, required=True)
@click.pass_context
def instr_trace(ctx, artifact_dir, x, y):
    """
    Group for instr-trace commands for a PE.
    """
    ctx.ensure_object(dict)
    ctx.obj['elf_dir'] = artifact_dir
    if 'CSL_SIMFAB_JSON_TRACE' in os.environ:
        simfab_log = os.path.join(artifact_dir, SIMFAB_JSON_LOG)
    else:
        print("sdk_debug_shell does not support ctf trace debug, please use csdb")
        sys.exit(1)
    ctx.obj['simfab_log'] = simfab_log
    ctx.obj['x'] = x
    ctx.obj['y'] = y

    if not os.path.exists(simfab_log):
        print(f"Simfab log {simfab_log} does not exist")
        sys.exit(1)


@instr_trace.command(help="Show instruction trace for a PE")
@click.option('--uthread', help="Microthread to filter by", type=int)
@click.option('--task', help="Task to filter by", type=int)
@click.pass_context
def trace(ctx, uthread, task):
    """
    Show instruction trace for a PE.
    """
    x = ctx.obj['x']
    y = ctx.obj['y']
    elf_dir = ctx.obj['elf_dir']
    simfab_log = ctx.obj['simfab_log']

    header = "=" * 50 + f"\nShowing instruction trace on P{x}.{y}"
    if uthread is not None:
        header += f" for uthread {uthread}\n"
    elif task is not None:
        header += f" for task {task}\n"
    else:
        header += "\n"
    header += "=" * 50
    print(header)
    instr_trace_obj = sdk_debug_instr_trace.InstrTrace(elf_dir, simfab_log)
    if uthread is not None:
        instr_trace_obj.print_trace_per_uthread(x, y, uthread)
    elif task is not None:
        instr_trace_obj.print_trace_per_task(x, y, task)
    else:
        instr_trace_obj.print_trace(x, y)


@instr_trace.command(help="Show a list of uthreads for a PE")
@click.pass_context
def uthreads(ctx):
    """
    Show a list of uthreads for a PE.
    """
    x = ctx.obj['x']
    y = ctx.obj['y']
    elf_dir = ctx.obj['elf_dir']
    simfab_log = ctx.obj['simfab_log']

    header = "=" * 50 + f"\nShowing list of uthreads on P{x}.{y}\n" + "=" * 50
    print(header)
    instr_trace_obj = sdk_debug_instr_trace.InstrTrace(elf_dir, simfab_log)
    instr_trace_obj.print_uthreads(x, y)


@instr_trace.command(help="Show a list of tasks for a PE")
@click.pass_context
def tasks(ctx):
    """
    Show a list of tasks for a PE.
    """
    x = ctx.obj['x']
    y = ctx.obj['y']
    elf_dir = ctx.obj['elf_dir']
    simfab_log = ctx.obj['simfab_log']

    header = "=" * 50 + f"\nShowing list of tasks on P{x}.{y}\n" + "=" * 50
    print(header)
    instr_trace_obj = sdk_debug_instr_trace.InstrTrace(elf_dir, simfab_log)
    instr_trace_obj.print_tasks(x, y)


@sdk_debug_app.command()
@click.option("--artifact_dir", default=os.getcwd(), type=click.Path())
def visualize(artifact_dir):
    """
    Visualize routing between PEs, post-simulation run results. For example, wavelet trace
    and instruction trace in a web browser.
    """
    csviz_exe = shutil.which("csviz")
    if csviz_exe is None:
        print("Error. No csviz found.")
        return
    subprocess.check_call([csviz_exe, artifact_dir])


WAVELET_FORMATS = ['i16', 'f16', 'f32', 'i16,f16', 'hex']
FAB_NUM_DATA_COLORS = 24


@sdk_debug_app.group()
@click.option(
    '--artifact_dir',
    help="Directory for simfab json logs",
    required=True,
    type=click.Path(exists=True)
)
@click.option('--x', help=X_COORD, type=int, required=True)
@click.option('--y', help=Y_COORD, type=int, required=True)
@click.option(
    '--color',
    '-c',
    help="Color to filter by",
    type=click.IntRange(0, FAB_NUM_DATA_COLORS - 1),
    default=[],
    multiple=True,
)
@click.option(
    '--fmt',
    help="Display fmt",
    type=click.Choice(WAVELET_FORMATS),
    default='i16',
    show_default=True,
)
@click.option(
    '--backpressure',
    help="Backpressure trace for a given tile",
    is_flag=True,
    default=False
)
@click.pass_context
def wavelet_trace(ctx, artifact_dir, x, y, color, fmt, backpressure):
    """
    Group for wavelet-trace command.
    """
    ctx.ensure_object(dict)
    if 'CSL_SIMFAB_JSON_TRACE' in os.environ:
        simfab_log = os.path.join(artifact_dir, SIMFAB_JSON_LOG)
    else:
        print("sdk_debug_shell does not support ctf trace debug, please use csdb")
        sys.exit(1)
    ctx.obj['simfab_log'] = simfab_log
    ctx.obj['x'] = x
    ctx.obj['y'] = y
    ctx.obj['color'] = color
    ctx.obj['fmt'] = fmt
    ctx.obj['backpressure'] = backpressure

    if not os.path.exists(simfab_log):
        print(f"Simfab log {simfab_log} does not exist")
        sys.exit(1)


@wavelet_trace.command(name="trace", help="Show wavelet trace for a PE")
@click.pass_context
def wavelet_trace_show(ctx):
    """
    Show wavelet trace for a PE.
    Examples:
    # get all wavelets for tile P0.0
    python sdk_debug_shell.py wavelet-trace --simfab_log token.js trace  --x 0 --y 0
    # get wavelets for tile P0.0 for colors 1,23
    python sdk_debug_shell.py wavelet-trace --simfab_log token.js trace  --x 0 --y 0 --color 1 -c 23
    # get backpressure trace for tile P0.0
    python sdk_debug_shell.py wavelet-trace --simfab_log token.js trace  --x 0 --y 0 --backpressure
    # get backpressure trace for tile P0.0 for colors 1,23
    python sdk_debug_shell.py wavelet-trace --simfab_log token.js trace  --x 0 --y 0 --backpressure
    --color 1 -c 23
    """
    wavelet_trace_obj = sdk_debug_wavelet_trace.WaveletTrace(
        ctx.obj['simfab_log']
    )
    if ctx.obj['backpressure'] and ctx.obj['color']:
        wavelet_trace_obj.print_backpressure_trace_per_colors(
            ctx.obj['x'], ctx.obj['y'], ctx.obj['color']
        )
    elif ctx.obj['backpressure']:
        wavelet_trace_obj.print_backpressure_trace(
            ctx.obj['x'], ctx.obj['y']
        )
    elif ctx.obj['color']:
        wavelet_trace_obj.print_trace_per_colors(
            ctx.obj['x'], ctx.obj['y'], ctx.obj['color'], ctx.obj['fmt']
        )
    else:
        wavelet_trace_obj.print_trace(
            ctx.obj['x'], ctx.obj['y'], ctx.obj['fmt']
        )


@sdk_debug_app.group()
def symbol_info():
    """
    Command deprecated.
    """
    print("symbol-info command is deprecated.")


if __name__ == '__main__':
    sdk_debug_app()
