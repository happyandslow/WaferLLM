import logging
from pathlib import Path
from typing import Optional

import cerebras.sdk.debug.lib.symbol.csldebugpybind as CSLDebug
from cerebras.hwdebug_common.shell_lib.context_base import ContextBase
from cerebras.hwdebug_common.shell_lib.utils import (
    glob_num_file,
    parse_index_input,
    rglob_file,
)


class CslDebugContext(ContextBase):
    """
    Handles where the artifacts for passing into debugger shell or UI
    SDKCache.
    """

    # Constants for SDK context
    OUT_JSON = "out.json"

    def __init__(self, logger: Optional[logging.Logger]):
        super().__init__(logger)

    def init_das_context(self):
        """
        Initialized data access context which include symbol table.
        """
        assert self._absolute_compile_dir is not None
        bin_dir = str(self._absolute_compile_dir / "bin")
        self._das_ctx = CSLDebug.CSLELFSymbol(bin_dir)
        # Initialize the symbol table for image command
        if self._das_ctx.dimensions().area():
            self._das_ctx.init_symbol()
            self.symbol_table = self._das_ctx.symbol_table

    def auto_detect_context(self, workdir: Path) -> bool:
        """
        Auto detect context sets _absolute_compile_dir when the workdir contains only 1 out.json in it's child directories.
        Returns True only when it finds 1 out.json in workdir. False otherwise.
        """
        # Automatically finds out.json directory inside any child directories.
        # The search here stops as soon as it finds more than 1 out.json in order to save
        # time during initializing context at startup.
        outjson_rel_path_list = glob_num_file(workdir, self.OUT_JSON, 2)

        # Found only 1 out.json file inside model_dir
        num_outjson_found = len(outjson_rel_path_list)
        if num_outjson_found == 1:
            self._absolute_compile_dir = (
                workdir / outjson_rel_path_list[0].parent
            ).resolve()
            self._ctx_list.append(outjson_rel_path_list[0].parent)
            return True
        # Handle none found
        elif num_outjson_found == 0:
            self._logger.info(
                " %s does not contain any CSL compile directory.", workdir
            )
        # Handle more than 1.
        else:
            self._logger.info(
                " %s contains more than one CSL compile directory.", workdir
            )
        return False

    def list_context(self, workdir: Path, display_to_stdout: bool) -> int:
        """
        Prints all the directory that contains out.json in given workdir
        and cache a list of directory that has out.json to save time for user to use
        index selection without doing search again.

        @return the number of out.json found.
        """
        # Cache a list of out.json relative path
        self._ctx_list = rglob_file(workdir, self.OUT_JSON)

        # Sort by path depth so that user program compile output is served
        # before memcpy infrastructure compile output
        self._ctx_list.sort(key = lambda path: len(path.parents))

        # Remove file name and just keep parent directory
        for idx in range(len(self._ctx_list)):
            self._ctx_list[idx] = self._ctx_list[idx].parent

        if display_to_stdout:
            for idx, comp_dir in enumerate(self._ctx_list):
                self._logger.info(" [%d] %s", idx, comp_dir)

        return len(self._ctx_list)

    def select_context(self, outdir_idx: str, workdir: Path) -> int:
        """
        Sets the compile_dir and context if out.json exist. When out.json
        is a relative path, it will concatentate it with workdir path.

        @param outdir_idx  The user input. Either parent path of the out.json
        or [INDEX] shown by context list.
        @param workdir. The parent workdir is used for case when user
        pass in a relative path for outdir_idx.

        @return 0 when success and negative numbers when fail for printing proper
        error message.
        """
        outdir = outdir_idx
        # User can pass in an index after context list was invoked to type less.
        if outdir_idx.endswith("]"):
            # Ask the user to cache the list first
            if len(self._ctx_list) == 0:
                return -2
            # Parse and error check user inputs
            index = parse_index_input(outdir_idx)
            if index < 0:
                self._logger.warn(" Invalid input %s.", outdir_idx)
                return -3
            elif index >= len(self._ctx_list):
                self._logger.warn(
                    " Index %s exceeds the context list.", outdir_idx
                )
                return -4

            outdir = self._ctx_list[index]

        outdir_path = Path(outdir)
        # Prepend workdir if path given is a relative path.
        if not outdir_path.is_absolute():
            outdir_path = Path(workdir) / outdir_path

        if Path(outdir_path).is_dir():
            self._absolute_compile_dir = outdir_path.resolve()
            self.init_das_context()
            return 0

        self._logger.warn(" %s does not exist.", outdir)
        return -1

    def show_context(self):
        self.show_context_msg("")
