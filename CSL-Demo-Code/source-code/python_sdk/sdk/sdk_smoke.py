'''
Smoke test by running all the CSL examples
'''

import argparse
import fnmatch
import os
import shutil
import tempfile
from pathlib import Path
from subprocess import run


def get_examples(csl_examples_dir: Path, dest: Path = Path(os.getcwd())):
    '''Copy examples to a `dest` directory.
    '''
    assert dest.is_dir(), f'directory {dest} does not exist'
    dest_dir = dest / csl_examples_dir.name
    shutil.copytree(csl_examples_dir, dest_dir)


def main(csl_examples_dir: Path):
    '''Entrypoint.'''

    # Create a temporary directory in the CWD and copy over `csl-examples` to
    # there. Now, traverse child directories, and for every directory with a
    # `commands*.sh`, run to the list of examples. Then run all examples.

    # TODO: why is CWD needed? The default location `/tmp/` does not work since
    #       it cannot be bind-mounted in the container (might be an overlayFS
    #       limitation in kernels < 3.18). Workaround can be to use `/mnt`, but
    #       there may be other issues with that.

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as d:
        examples_to_run = []
        tmpdir = Path(d)
        get_examples(csl_examples_dir, tmpdir)
        examples_dir = tmpdir / csl_examples_dir.name
        assert examples_dir.is_dir(), f'{examples_dir} directory not found'

        # Prevent simfab_traces from being produced
        os.environ["CSL_SUPPRESS_SIMFAB_TRACE"] = "1"

        # walk through every directory inside of examples_dir,
        # and append every one with a `commands*.sh` file
        # to the `examples_to_run` list
        for root, _, _ in os.walk(examples_dir):

            if any(fnmatch.fnmatch(file_name, 'commands*.sh') for file_name in os.listdir(root)):
                examples_to_run.append(Path(root))

        # The commands.sh scripts use 'cslc' and 'cs_python', which are wrapper
        # scripts that invoke the singularity container. However, this code is
        # already inside the container. Rather than try to rewrite the
        # commands.sh on the fly, we can alias those two commands locally at the
        # start of the script.
        prelude = (
            "function cslc() {\n"
            "  sdk_debug_shell compile \"$@\"\n"
            "}\n"
            "function cs_python() {\n"
            "  python \"$@\"\n"
            "}\n"
            "set -x\n"
        )

        # Run all the examples
        for example in examples_to_run:
            ex_name = f'{example.parent.name}/{example.name}'

            for cmd_script in \
                [f for f in os.listdir(example) if fnmatch.fnmatch(f, 'commands*.sh')]:

                print(f'Running example: \033[34m{ex_name}\033[0m with commands: '
                      f'\033[34m{cmd_script}\033[0m')
                cmds = (example / cmd_script).read_text()
                cmd = prelude + cmds
                run(
                    cmd,
                    executable=shutil.which('bash'),
                    shell=True,
                    cwd=example,
                    check=True,
                )

                print(f'\033[92mPASSED\033[0m\n')

        print('\033[92mSMOKE CHECK COMPLETED SUCCESSFULLY\033[0m')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples_dir", help="CSL examples location")

    options = parser.parse_args()

    main(Path(options.examples_dir))
