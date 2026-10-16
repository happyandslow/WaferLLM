import os
import re
from pathlib import Path
from typing import Optional

import __main__
import click
from click_shell import shell

import cerebras.message.messagepybind as Message
from cerebras.das_common.das_commonpybind import Dimensions
from cerebras.hal.halpybind import Rectangle

# returns the name of the main module without the extension
# and without the directory
def getMainName():
    if '__file__' in dir(__main__):
        return os.path.splitext(os.path.basename(__main__.__file__))[0]

    # This should never be executed in practice only here to circumvent jenkins test please
    return "debug"


class DbgCommonShell:
    # context string
    compile_context_name = ""

    # Common members between execution model specific
    # Settings command library
    workdir = None
    target = None
    context = None
    # Execution command library
    image = None
    register = None
    memory = None
    rectangle_cmd = None
    # Execution model specific initialization callback
    rectangle_cb = None
    logger = None

    Message.enable_cout_intercept(True)

    def init_workdir(self, workdir: str):
        # Do not initialize workdir if path is not a directory
        workdir_path = Path(workdir)
        if not workdir_path.is_dir():
            self.logger.warning(" %s is not valid directory.", workdir)
            return

        self.workdir = workdir_path

    def show_workdir(self):
        workdir_str = str(self.workdir) if self.workdir else "n/a"
        self.logger.info(" Workdir: %s", workdir_str)

    def init(self):
        pass

    def print_no_workdir_selected(self):
        self.logger.warning(" No workdir selected.")
        self.logger.info(" Use `workdir select` to select a workdir first")

    def print_no_context_selected(self):
        self.logger.warning(" No %s loaded.", self.compile_context_name)
        self.logger.info(
            " Use `context select` to select a %s first.",
            self.compile_context_name,
        )

    def print_no_target_created(self):
        self.logger.warning(" No target was created.")
        self.logger.info(" Use `target create` to create a target first")

    # shell commands
    def list_context(self):
        """
        List all the compile context found in the subdirectory g_workdir.
        For pipeline execution, the compile context is referring to plan.meta.
        """
        if self.workdir is None:
            self.print_no_workdir_selected()
            return

        self.context.list_context(self.workdir, True)

    def select_context(self, compile_dir):
        """
        Allows the user to select a specific compile context within `workdir`.
        """
        if self.workdir is None:
            self.print_no_workdir_selected()
            return

        ret = self.context.select_context(compile_dir, self.workdir)
        if ret:
            self.logger.info(
                " Use \"context list\" to show list of compile directories."
            )

    def show_context(self):
        """
        Prints the location of the selected compile context.
        """
        if self.workdir is None:
            self.print_no_workdir_selected()
            return

        self.context.show_context()

    def list_target(self):
        """
        List all the checkpoint files  (ML) or core files (SDK)  found in the 
        subdirectory `workdir`. If the workdir is not a valid workdir, i.e. 
        workdir does not contain a `model_dir` subdirectory, list all the 
        checkpoint files found in the current working directory instead.
        """
        if self.workdir is None:
            self.print_no_workdir_selected()
            return

        self.target.list_target(self.workdir, True, self.target.CHECKPOINT_ONLY)

    def create_target(self, name: str):
        """
        Connects the user to a CM or a checkpoint/core file.

        @param name: either the checkpoint path or the CM address

        Checkpoint path: the full path to the checkpoint file with only the prefix
        For example:
            if there exist: /cb/home/.../model_dir/.../debug_session/PRE_LOAD_PLAN-ckpt_0.gz 
            pass in:        /cb/home/.../model_dir/.../debug_session/PRE_LOAD_PLAN
        
        CM address: CM endpoint IP address
        """
        if self.workdir is None:
            self.print_no_workdir_selected()
            return

        # Regular expression to deterine cmaddr
        # Example: `10.254.7.224:9000`
        # Compile context does not need to be set to
        # connect to a CM address.
        if re.match("^(?:\d+\\.){3}\d+:\d+$", name):
            self.target.create_cmaddr(name)
            self.target.unset_checkpoint()
            return

        # Read from a checkpoint or core file
        if self.target.create_checkpoint(name, self.workdir):
            self.target.unset_cmdevice()

    def show_target(self):
        """
        Prints the location of the selected checkpoint/core file.
        """
        if self.workdir is None:
            self.print_no_workdir_selected()
            return

        self.target.show_target()

    def save_core(self, name):
        if self.workdir is None:
            self.print_no_workdir_selected()
            return

        self.target.save_core(name)

    def error_check_exe_common(self, context_required: bool = True) -> bool:
        if self.workdir is None:
            self.print_no_workdir_selected()
            return False

        if context_required and self.context.compile_dir() is None:
            self.print_no_context_selected()
            return False

        return True

    def error_check_register_command(self, context_required: bool = True) -> bool:
        """
        @brief error_check_register_command checks if workdir, target,
        context, rectangle callback are all initialized correctly before
        proceeding.

        @returns True when there's no error found. False otherwise.
        """
        # TODO: Want to read register without any context initialized. This is useful
        # when target is for CM, we can the dimension from the CM.
        if not self.error_check_exe_common(context_required):
            return False

        if (
            self.target.hal_device() is None
            and self.target.checkpoint_file() is None
        ):
            self.print_no_target_created()
            return False

        # Error: both target CM and checkpoint are set.
        assert not (
            self.target.hal_device() is not None
            and self.target.checkpoint_file() != None
        )

        # Make sure the callback is initialized
        assert self.rectangle_cb is not None
        assert self.rectangle_cmd is not None

        return True

    def read_register(self, stdoutput):
        """
        Read all the registers from a specified rectangle
        TODO: allow register name to be specified
        """
        if not self.error_check_register_command():
            return

        rect_list = self.rectangle_cb()
        for name, rect in rect_list:
            output_filename = ""
            if not stdoutput:
                output_filename = self.register.create_filename(
                    "register", self.workdir, rect, name,
                    symbol_name=None, address=None, length=None
                )
                # Message manager appends to existing file.
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            # Redirect output to <output_filename>
            Message.set_logfile(output_filename)
            self.register.read(
                self.target.hal_device(), 
                self.target.checkpoint_file(), 
                rect,
            )

        # Redirect output to <stdout>
        Message.set_logfile("")

    def write_register(self, value):
        """
        write a register is the specified kernel and rectangle
        """
        self.register.write()

    def _convert_string_to_uint(self, data_str: str, error_msg: str) -> int:
        """
        Convert string to int.
        Returns None and prints error message upon if fails.
        """
        data = None

        # Hexidecimal support
        # https://stackoverflow.com/questions/209513/convert-hex-string-to-int-in-python
        try:
            data = int(data_str, 0)
            if data < 0:
                self.logger.warning(" Invalid %s: %s", error_msg, data_str)
                return None
        except ValueError:
            self.logger.warning(" Invalid %s format: %s", error_msg, data_str)
            return None

        return data

    def read_memory(self, 
        address: int = -1, 
        length: int = 0, 
        stdoutput: bool = False,
        symbol_name: str = "",
        symbol_all: bool = False):
        """
        Read a memory with specific address and length or symbol name or all symbols
        """

        if symbol_all and (symbol_name or address >= 0 or length > 0):
            if not self.error_check_register_command():
                return
            self.logger.error(" Invalid option, --all cannot be use with other options")
            return
        if symbol_name and (symbol_all or address >= 0 or length > 0):
            if not self.error_check_register_command():
                return
            self.logger.error(" Invalid option, --name cannot be use with other options")
            return
        if (address >= 0 or length > 0) and (symbol_name or symbol_all):
            if not self.error_check_register_command(False):
                return
            self.logger.error(" Invalid option, --address and --length cannot be use with other options")
            return

        rect_list = self.rectangle_cb()
        if not rect_list:
            self.logger.error("No rectangles selected.")
            return

        for name, rect in rect_list:
            output_filename = ""
            if not stdoutput:
                output_filename = self.memory.create_filename(
                    "memory", self.workdir, rect, name,
                    symbol_name=symbol_name, address=address,
                    length=length
                )
                if os.path.exists(output_filename):
                    os.remove(output_filename)
            # Redirect output to <output_filename>
            Message.set_logfile(output_filename)
            symbol_table = None
            if self.context and self.context.das_ctx():
                symbol_table = self.context.symbol_table

            self.memory.read(
                self.target.hal_device(),
                self.target.checkpoint_file(),
                rect,
                address,
                length,
                symbol_name,
                symbol_all,
                symbol_table
            )
        # Redirect output to <stdout>
        Message.set_logfile("")

    def write_memory(self, address: int, values_str: str):
        """
        Read a memory with specific address and length
        """
        if not self.error_check_register_command():
            return

        if address < 0:
            self.logger.error("Invalid address %i", address)
            return

        value_str_list = values_str.split(",")
        value_list = []
        for v_str in value_str_list:
            v = self._convert_string_to_uint(v_str, "value")
            if v is None:
                return
            if v > 0xFFFF:
                self.logger.warning(
                    " Invalid value: %s. Expects between 0x0 and 0xFFFF", v_str
                )
                return
            value_list.append(v)

        rect_list = self.rectangle_cb()
        for name, rect in rect_list:
            self.memory.write(
                self.target.hal_device(), 
                rect,
                address,
                len(value_list),
                value_list,
            )

    def select_rectangle(self, rect: str, dimension: str = "1,1",):
        self.rectangle_cmd.select(
            rect, dimension, 
            self.target.fabric_dimensions(),
        )

    def reset_rectangle(self):
        # Resets the rectangle to fabric dimension if possible
        self.rectangle_cmd.clear()
        # When context is initialized and das context initialized
        # properly.
        if (
            self.target is not None
            and self.target.hal_device() is not None
        ):
            self.rectangle_cmd.set_rect_to_device(
                self.target.fabric_dimensions()
            )
        elif (
            self.context is not None
            and self.context.das_ctx() is not None
            and self.context.das_ctx().dimensions()
        ):
            self.rectangle_cmd.set_rect_to_device(
                self.context.das_ctx().dimensions()
            )

    def show_rectangle(self):
        self.rectangle_cmd.show()
    
    def deselect_rectangle(self, rect, dimension, all):
        self.rectangle_cmd.deselect(rect, dimension, all)

    def list_image(self, all: bool):
        assert self.rectangle_cb is not None
        rect_list = self.rectangle_cb()
        for name, rect in rect_list:
            self.image.list(self.context.symbol_table, all, rect)

    def lookup_image(self, all: bool, symbol_name: Optional[str]):
        if self.context is None or self.context.das_ctx() is None:
            self.print_no_context_selected()
            return

        assert self.rectangle_cb is not None
        rect_list = self.rectangle_cb()
        for name, rect in rect_list:
            self.image.lookup(
                self.context.symbol_table, all, rect, symbol_name
            )

    def image_info(self):
        if self.context is None or self.context.das_ctx() is None:
            self.print_no_context_selected()
            return

        self.image.info(self.context.symbol_table)

class BasedIntParamType(click.ParamType):
    name = 'integer'

    def convert(self, value, param, ctx):
        if isinstance(value, int):
            return value
        
        try:
            if value[:2].lower() == '0x':
                return int(value[2:], 16)
            elif value[:1] == '0':
                return int(value, 8)
            return int(value, 10)
        except ValueError:
            self.fail('%s is not a valid integer' % value, param, ctx)

BASED_INT = BasedIntParamType()

global_shell = None

def set_shell_instance(shell):
    global global_shell
    global_shell = shell

# TODO get main file name for prompt
@shell(prompt='({}) '.format(getMainName()), intro="Starting debug shell...")
@click.pass_context
@click.argument("workdir", required=False)
def debug_commands(ctx, workdir: str):
    """
    Entry point for the debug shell
    """
    global global_shell
    ctx.obj = global_shell

    # Just start the shell without anything initialize
    if workdir is not None:
        ctx.obj.init_workdir(workdir)

    # Initialize context, targets and other commands
    ctx.obj.init()


@debug_commands.group(help="workdir is the directory for debug session.")
def workdir():
    pass


@workdir.command()
@click.pass_context
@click.argument("workdir")
def select(ctx, workdir):
    """Select a workdir."""
    ctx.obj.init_workdir(workdir)
    # Initialize context, targets and other commands
    ctx.obj.init()


#
# context commands
#
@debug_commands.group(help="Set compile context.")
def context():
    pass


@context.command()
@click.pass_context
def list(ctx):
    """List all the compile context in workdir."""
    ctx.obj.list_context()


@context.command()
@click.pass_context
def show(ctx):
    """Show the selected compile context."""
    ctx.obj.show_context()


#
# target commands
#
@debug_commands.group(
    help="Connect to CM for interactive debugging or examine a core file.",
)
def target():
    pass


@target.command()
@click.pass_context
def list(ctx):
    """List all the core files."""
    ctx.obj.list_target()


@target.command()
@click.pass_context
def show(ctx):
    """Show selected core file."""
    ctx.obj.show_target()


@target.command()
@click.pass_context
@click.argument("name_prefix", required=False)
def save_core(ctx, name_prefix):
    """Save a core file after connecting to a CM target"""
    ctx.obj.save_core(name_prefix)


@target.command()
@click.pass_context
@click.option(
    '--core-file', 
    'name', 
    help='Specify the core file.',
    metavar="<path>"
    )
@click.option(
    '--cmaddr', 
    'name', 
    help='Specify the CM address.',
    metavar="<#.#.#.#:port #>"
    )
def create(ctx, name):
    """Connect to a CM or read core file as target."""
    if name:
        ctx.obj.create_target(name)
    else:
        ctx.obj.logger.info(" Missing argument: `--core-file` or `--cmaddr`.")


#
# image commands
#
@debug_commands.group(help="Show ELF files and symbol table for different PE.",)
def image():
    pass


@image.command()
@click.pass_context
@click.option("--all", "-a", is_flag=True)
def list(ctx, all):
    """List the ELF binary for each PE coordinate."""
    ctx.obj.list_image(all)


@image.command()
@click.pass_context
@click.option("--all", "-a", is_flag=True)
@click.option("--name", required=False)
def lookup(ctx, all, name):
    """Look up symbol name in the symbol table for entire fabric."""
    ctx.obj.lookup_image(all, name)


@image.command()
@click.pass_context
def info(ctx):
    """Shows the fabric dimension."""
    ctx.obj.image_info()


#
# register commands
#
@debug_commands.group(help="Show configuration registers for PE.")
def register():
    pass


#
# memory commands
#
@debug_commands.group(help="Read and write to memory locations in PEs.")
def memory():
    pass


@memory.command()
@click.pass_context
@click.option(
    "--address",
    "-a",
    "address",
    metavar="<int|hex>",
    type=BasedIntParamType(),
    required=False,
    default=-1,
    help="Address of in PE memory in units of 2 bytes.",
)
@click.option(
    "--length",
    "-l",
    "length",
    metavar="<int|hex>",
    type=BasedIntParamType(),
    required=False,
    default=0,
    help="Length to read in units of 2 bytes.",
)
@click.option(
    "--stdoutput",
    "-o",
    "stdoutput",
    is_flag=True,
    default=False,
    help="redirect output to stdout instead of a file",
)
@click.option(
    "--name",
    "-n",
    "symbol_name",
    default="",
    help="Get the value of a specific symbol."
)
@click.option(
    "--all",
    "-j",
    "symbol_all",
    is_flag=True,
    default=False,
    help="Set this flag to list all the symbols and their values."
)
def read(ctx, address, length, stdoutput, symbol_name, symbol_all):
    """Read memory from a core file or a device."""
    ctx.obj.read_memory(address, length, stdoutput, symbol_name, symbol_all)


@memory.command()
@click.pass_context
@click.argument("values")
@click.option(
    "--address",
    "-a",
    "address",
    metavar="<int|hex>",
    type=BasedIntParamType(),
    required=True,
    help="Address of in PE memory in units of 2 bytes.",
)
def write(ctx, address, values):
    """Write memory to a device in units of 2-bytes. Separate consecutive values by comma."""
    ctx.obj.write_memory(address, values)

#
# rectangle commands
#
@debug_commands.group(help="Select rectangle")
def rectangle():
    pass


@rectangle.command()
@click.pass_context
@click.argument("rect")
@click.option(
    "--dimension",
    "-d",
    "dimension",
    default="1,1",
    help="The width and height of the selected rectangle."
)
def select(ctx, rect: str, dimension: str):
    """Selects a rectangle."""
    ctx.obj.select_rectangle(rect, dimension)


@rectangle.command()
@click.pass_context
def reset(ctx):
    """Resets the current rectangle to fabric dimension."""
    ctx.obj.reset_rectangle()


@rectangle.command()
@click.pass_context
def show(ctx):
    """Shows the current rectangle."""
    ctx.obj.show_rectangle()

@rectangle.command()
@click.pass_context
@click.argument("rect")
@click.option(
    "--dimension",
    "-d",
    "dimension",
    default="",
    help="""The width and height of the selected rectangle
    (separated by a comma, e.g. -d size_x,size_y)."""
)
@click.option(
    "--all",
    "-a",
    "all",
    is_flag=True,
    default=False,
    help="This flag will remove all the rectangles from the list."
)
def deselect(ctx, rect, dimension, all):
    """remove a rectangle from the list"""
    ctx.obj.deselect_rectangle(rect, dimension, all)
