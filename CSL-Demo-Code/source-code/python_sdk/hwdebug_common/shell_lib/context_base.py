import logging
from abc import ABC, abstractmethod
from typing import List, Optional


class ContextBase(ABC):
    """
    Abstract base class for all Context instances whether it
    is Pipeline or SDK.

    All Context instances should implement

    init_das_context      -- Initalize DasContext object
    auto_detect_context   -- Initialize compile directory if found exactly 1

    list_context          -- List all the possible context
    select_context        -- Selects the context
    show_context          -- Show the context 
    """

    def __init__(self, logger: Optional[logging.Logger]):
        """
        _absolute_compile_dir is the absolute path to compile directory.
        Compile directory is generic for both pipeline and SDK. For example, in pipeline the value would
        be the parent directory to file plan.meta. Absolute path is used here since within the usage of
        debugger, the compile dir is not necessary a child of workdir. And so, the context holds absolute path
        and allow caller to make determination if they want relative or absolute path.
        """
        self._logger = logger

        self._absolute_compile_dir = None  # Type: Path
        self._das_ctx = None  # Type: daspybind.DasContext
        self.symbol_table = None  # Type: SymbolTableBase
        # Saves a list after context list
        self._ctx_list = []
        # TODO: ElfSymbolTable and info [--dimension] [--arch] that shows
        # list of ELF binaries, fabric dimension, arch, simulator or HW.

    @abstractmethod
    def init_das_context(self):
        pass

    @abstractmethod
    def auto_detect_context(self, workdir):
        pass

    @abstractmethod
    def list_context(self, workdir, display_to_stdout: bool) -> int:
        """
        Populates the compile directories found to a list.

        @param workdir is directory to search
        @display_to_stdout prints the result to stdout when set to True.
        """

    @abstractmethod
    def select_context(self, compile_dir):
        pass

    def das_ctx(self):
        return self._das_ctx

    def compile_dir(self):
        return self._absolute_compile_dir
    
    def context(self) -> List:
        """
        returns the list of cached context list after list_context is called.
        """
        return self._ctx_list

    @abstractmethod
    def show_context(self):
        pass

    def show_context_msg(self, epilog: str):
        """
        Prints the absolute path of selected compile context. 
        Prints <not selected> if not initialized.   
        """
        if self._absolute_compile_dir is None:
            self._logger.info(" Compile context: not selected.")
            return

        self._logger.info(
            " Compile context: %s",
            f"{str(self._absolute_compile_dir)}/{epilog}",
        )
