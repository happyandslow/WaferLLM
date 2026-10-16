import logging
import re
from pathlib import Path
from typing import List, Optional

from cerebras.hal.halpybind import instantiateDevice
from cerebras.das_common.das_commonpybind import Dimensions
from cerebras.hwdebug_common.shell_lib.utils import (
    parse_index_input,
    rglob_file,
)


class Target:
    """
    Selects either checkpoint (aka core file) or CM address.
    """

    # Constants
    ALL_COREDUMPS = 0
    CHECKPOINT_ONLY = 1
    COREFILE_ONLY = 2

    def __init__(
        self, logger: Optional[logging.Logger], display_ckpt_name: str
    ):
        """
        A checkpoint file is stored in multiple files, typically 4. The same checkpoint 
        will have the same prefix with a suffix of "-ckpt_[<number>].gz"
        For example with "PRE_LOAD_PLAN-ckpt_0.gz":

        Prefix: PRE_LOAD_PLAN
        Suffix: "-ckpt_0.gz", 
        """
        self._logger = logger
        self._display_ckpt_name = display_ckpt_name

        # Pipeline execution local simfab
        self._include_simfab_checkpoint = False

        # Constants
        self._CKPT_SUFFIX = "ckpt"
        # CSL SDK
        self._CKPT_DEFAULT_NAME = f"core{self._CKPT_SUFFIX}"
        # ML application
        if "checkpoint" in self._display_ckpt_name:
            self._CKPT_DEFAULT_NAME = f"dump{self._CKPT_SUFFIX}"
            self._include_simfab_checkpoint = True

        self._cmaddr = None
        self._hal_device = None

        # Type: String, The prefix of the selected checkpoint file
        self._checkpoint_file = None

        # Type: List, saves all checkpoint prefixes in the workdir
        self._checkpoint_list = []

    def fabric_dimensions(self) -> Dimensions:
        if self._hal_device is not None:
            return self._hal_device.dimensions()
        
        return None

    def checkpoint_file(self):
        return self._checkpoint_file

    def checkpoint_list(self):
        return self._checkpoint_list

    def cmaddr(self):
        return self._cmaddr

    def hal_device(self):
        return self._hal_device

    def _get_checkpoint_prefixes(self, dir: Path, prefix: str) -> List[str]:
        """
        Given a path to a directory search the directory for checkpoint files and store
        their unique prefixes in a list

        Using PRE_LOAD_PLAN-ckpt_0.gz as an example, 
        Prefix = PRE_LOAD_PLAN

        @param dir    The directory to search for the checkpoint files
        @param prefix The prefix to search for, pass in "*" to match any prefix

        @return the list unique checkpoint prefixes
        """
        if prefix == "*":
            prefix = self._CKPT_SUFFIX

        unique_ckpt_prefix = set()
        rel_ckpt_list = rglob_file(dir, f"*{self._CKPT_SUFFIX}_[0-3].*gz")
        for rel_ckpt_name in rel_ckpt_list:
            rel_ckpt_name = str(rel_ckpt_name)

            # Filter the prefix of checkpoint file and store the unique ones
            m = re.search(f"(.*{prefix})_[0-3].*gz", rel_ckpt_name)
            if m:
                ckpt_name = m.group(1)
                unique_ckpt_prefix.add(ckpt_name)
        return list(unique_ckpt_prefix)

    def _get_core_file(self, dir: Path) -> List[str]:
        rel_corefile_list = rglob_file(dir, "*core*.out")
        return rel_corefile_list

    def _get_local_simfab_checkpoint_file(self, dir: Path) -> List[str]:
        """
        Given a path to a directory search the directory for local simfab checkpoint
        files and returns a list of unique local simfab checkpoint prefixes

        Local simfab checkpoint file format: 'corefile.cs1_[0-3]'

        @param dir    The directory to search for the local simfab checkpoint files

        @return the list unique local simfab checkpoint prefixes
        """
        LOCALSIMFAB_PREFIX = "corefile.cs1"
        unique_simfab_ckpt_prefix = set()
        rel_simfab_ckpt_list = rglob_file(dir, f"*{LOCALSIMFAB_PREFIX}_[0-3]*")
        for rel_ckpt_name in rel_simfab_ckpt_list:
            rel_ckpt_name = str(rel_ckpt_name)

            m = re.search(f"(.*{LOCALSIMFAB_PREFIX})_[0-3]*", rel_ckpt_name)
            if m:
                ckpt_name = m.group(1)
                unique_simfab_ckpt_prefix.add(ckpt_name)
        return list(unique_simfab_ckpt_prefix)

    def list_target(
        self, dir: Path, display_to_stdout: bool, file_type: int
    ) -> int:
        """
        Create a unique set since one single checkpoint is stored in multiple files, typically 4.
        The same checkpoint will have the same prefix with a suffix of "-ckpt_[<number>].gz"
        For example:
            PRE_LOAD_PLAN-ckpt_0.gz, PRE_LOAD_PLAN-ckpt_1.gz, ... --> show: PRE_LOAD_PLAN
        
        @param dir is directory to search.
        @display_to_stdout means display the result to stdout.
        @file_type 0 means both checkpoint and core file, 1 means checkpoint only, 2 means
        corefile only.
        @return the number of checkpoint files found.
        """

        checkpoint_list = []
        core_file_list = []

        # Get the list of checkpoint
        if file_type == self.ALL_COREDUMPS or file_type == self.CHECKPOINT_ONLY:
            checkpoint_list = self._get_checkpoint_prefixes(dir, "*")

            # Look for additional local simfab checkpoints
            if self._include_simfab_checkpoint:
                checkpoint_list += self._get_local_simfab_checkpoint_file(dir)

        if file_type == self.ALL_COREDUMPS or file_type == self.COREFILE_ONLY:
            core_file_list = self._get_core_file(dir)

        self._checkpoint_list = checkpoint_list + core_file_list

        if display_to_stdout:
            for idx, ckpt_name in enumerate(self._checkpoint_list):
                self._logger.info(" [%d] %s", idx, ckpt_name)

        return len(self._checkpoint_list)

    def create_checkpoint(self, checkpoint_prefix: str, workdir: Path) -> bool:
        """
        A checkpoint_prefix is passed in and can be 1 of 3 different types:
            1. A previously listed checkpoint option with the form [<VALUE>], e.g. [1]
            2. A relative path to a checkpoint prefix, e.g. model_dir/cerebras_logs/.../POST_LOAD_PLAN
            3. An absolute path to a checkpoint prefix, e.g. /cb/home/.../model_dir/.../POST_LOAD_PLAN

        If the checkpoint_prefix passed in is valid, i.e. there exists files with the suffix
        "-ckpt_[<number>].gz" attached to the end of the checkpoint_prefix in the specified directory
        or is one of the previously listed checkpoints, set the _checkpoint_file attribute to the 
        checkpoint_prefix passed in.
        """

        # Check if input is a previously listed checkpoint option, i.e. passing in [<VALUE>]
        if checkpoint_prefix.endswith("]"):

            # Check if "list target" has previously been called
            if len(self._checkpoint_list) == 0:
                self._logger.warn(" No checkpoints saved invalid option")
                return False

            # If value is not out of bound set self._checkpoint_file to previously listed checkpoint
            idx = int(parse_index_input(checkpoint_prefix))
            if idx < 0:
                self._logger.warn(" Invalid option index")
                return False
            elif idx >= len(self._checkpoint_list):
                self._logger.warn(
                    " Index %s exceeds the context list.", checkpoint_prefix
                )
                return False
            else:
                # recreate path before assigning
                full_path = workdir / Path(self._checkpoint_list[idx])
                self._checkpoint_file = str(full_path)
                return True

        # Check if user passed in a relative path by appending it to workdir and checking if the
        # parent directory exists
        checkpoint_prefix_path = Path(checkpoint_prefix)
        absolute_path = checkpoint_prefix_path
        if not checkpoint_prefix_path.is_absolute():

            # Check if user has previously called "target list"
            reconstructed_path = workdir / checkpoint_prefix_path
            if len(self._checkpoint_list) > 0:
                # If user has previously invoked "target list" and checkpoint_prefix is one of the
                # listed paths save it
                if checkpoint_prefix in self._checkpoint_list:
                    self._checkpoint_file = str(reconstructed_path)
                    return True
                else:
                    self._logger.warn(
                        "Relative path not found in previously listed checkpoints"
                    )
                    return False

            # User entered an invalid relative path or has not invoked "target list" yet therefore
            # reconstruct full path and set checkpoint_prefix_path to full path
            else:
                absolute_path = reconstructed_path

        # User entered an absolute path or has not invoked "target list" use _get_checkpoint_prefixes
        # to check if checkpoint files exist in the directory of the given checkpoint_prefix
        prefix = absolute_path.name
        checkpoint_files_found = self._get_checkpoint_prefixes(
            absolute_path.parents[0], prefix
        )
        checkpoint_files_found_2 = self._get_local_simfab_checkpoint_file(
            absolute_path.parents[0]
        )
        if len(checkpoint_files_found) > 0:
            self._checkpoint_file = str(absolute_path)
        elif len(checkpoint_files_found_2) > 0:
            self._checkpoint_file = str(absolute_path)
        else:
            self._logger.warn(
                " %s is an invalid checkpoint file", checkpoint_prefix
            )
            return False

        return True

    def unset_checkpoint(self):
        self._checkpoint_file = None

    def save_core(self, prefix):
        if self._hal_device is None:
            self._logger.warning(
                "not connected to a cm device, cannot save core file"
            )
            return

        # Get the default name. The name is different between execution models.
        file_prefix = prefix
        if file_prefix is None:
            file_prefix = self._CKPT_DEFAULT_NAME
        else:
            file_prefix = f"{file_prefix}{self._CKPT_SUFFIX}"

        self._hal_device.core_dump(file_prefix)

    def create_cmaddr(self, cmaddr: str):
        """
        Sets the self._cmaddr parameter to CM address passed in

        cmaddr : CM address
        """

        self._cmaddr = cmaddr
        self._hal_device = instantiateDevice(self._cmaddr)

    def unset_cmdevice(self):
        self._cmaddr = None
        self._hal_device = None

    def show_target(self):
        """
        Print the selected target
        """

        if self._checkpoint_file is not None:
            self._logger.info(
                " Target (%s): %s",
                self._display_ckpt_name,
                self._checkpoint_file,
            )

        elif self._cmaddr:
            self._logger.info(" Target (CM address): %s", self._cmaddr)

        else:
            self._logger.info(
                " Target: no %s/CM address created.", self._display_ckpt_name
            )
