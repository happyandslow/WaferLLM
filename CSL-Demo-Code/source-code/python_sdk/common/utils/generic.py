""" Generic module to provide helper methods.
"""
import os
import re
import shlex
import signal
import logging
import subprocess
import resource
import time
import tempfile

import collections
from collections import OrderedDict

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    LOG.addHandler(logging.NullHandler())


def compute_cycle_quantities(cycle_data, coef1, coef2):
    '''
        Eq(1) : C_set + COEF1*C_exe = C_t1
        Eq(2) : C_set + COEF2*C_exe = C_t2
        Solving for C_exe,
         C_exe = (C_t2 - C_t1)/(COEF2-COEF1)

        And using Eq(1), we solve for C_set
         C_set = C_t1 - COEF1*C_exe
    '''
    #Perform safety checks
    if coef2 <= coef1:
        raise Exception("COEF2 must be greater than COEF1.")

    if ("total_cycles_test2" not in cycle_data) or ("total_cycles_test1" not in cycle_data):
        raise Exception("Cannot solve the equation, missing total_cycles_test{1,2}.")
    tot_cycle_t1 = cycle_data["total_cycles_test1"]
    tot_cycle_t2 = cycle_data["total_cycles_test2"]
    exe_cycles = int((tot_cycle_t2-tot_cycle_t1)/(coef2-coef1))
    set_cycles = tot_cycle_t1-(coef1*exe_cycles)
    cycle_data["execution_cycles"]=exe_cycles
    cycle_data["setup_cycles"]=set_cycles
    return cycle_data


## Pythonic Impl of tail.
def tail(f, lines=1, _buffer=4098):
    """Tail a file and get X lines from the end"""
    # place holder for the lines found
    lines_found = []

    # block counter will be multiplied by buffer
    # to get the block size from the end
    block_counter = -1

    # loop until we find X lines
    while len(lines_found) < lines:
        try:
            f.seek(block_counter * _buffer, os.SEEK_END)
        except IOError:  # either file is too small, or too many lines requested
            f.seek(0)
            lines_found = f.readlines()
            break
        lines_found = f.readlines()
        # decrement the block counter to get the
        # next X bytes
        block_counter -= 1

    return lines_found[-lines:]

def fetch_cycles_from_simoutfile(simout_filename):
    """Fetches cycles from the simout file."""
    cycles = 0
    with open(simout_filename, "r") as f:
        last100 = tail(f, 100)
        for line in reversed(last100):
            match = re.search(r'@(\d+)', line)
            if match:
                cycles=int(match.group(1))
                break
    if cycles == 0:
        raise Exception("The file did not contain any cycle information.")
    return cycles

def generate_unique_outfile(outfile_base, output_dir, ext=None):
    """
    Creates a unique outfile in the output dir by just appending an
    integer to the end if the given file is not an existing file
    already, e.g.:
        generate_unique_outfile("plan") ==> 'out/plan'
        generate_unique_outfile("plan") ==> 'out/plan.0'
        generate_unique_outfile("plan") ==> 'out/plan.1'
    """

    if os.path.sep in outfile_base:
        outfile = outfile_base
    else:
        outfile = os.path.join(output_dir, outfile_base)
    if ext and ext.startswith("."):
        ext = ext[1:]  # e.g. '.tar' -> 'tar'

    open_flags = (os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    open_mode = 0o664

    filenum = None
    tryfile = None
    while True:
        tryfile = outfile
        if filenum is not None:
            # Then this is the first try. Let's actually try and use
            # the given name
            tryfile += "." + str(filenum)
            filenum += 1
        else:
            # Else, time to start numbering
            filenum = 0
        if ext:
            tryfile += "." + ext
        try:
            tryfile_fd = os.open(tryfile, open_flags, open_mode)
            os.close(tryfile_fd)
            return tryfile
        except FileExistsError:
            pass
    return None


def subprocess_call(cmd, shell=False, timing=False, **kwargs):
    """ Logs then runs the command, returns the exitcode int """
    if 'close_fds' not in kwargs:
        kwargs['close_fds'] = True
    if isinstance(cmd, str):
        cmdstr = cmd
    else:
        if shell is True:
            cmdstr = ' '.join(cmd)
        else:
            cmdstr = ' '.join([shlex.quote(arg) for arg in cmd])
    LOG.info("Run command: %s", cmdstr)
    start = time.time()
    ret = subprocess.call(cmd, shell=shell, **kwargs)
    end = time.time()
    if ret < 0:
        LOG.error("Aborted on %s", _signum_to_signame(-ret))
    elif timing is True:
        LOG.info("Runtime for above command: %.2f seconds", end - start)
    return ret


def count_file_lines(filepath):
    """ Equivalent to 'wc -l', returns an int """
    count = 0
    with open(filepath, 'r') as filehandle:
        for _ in filehandle:
            count += 1
    return count


def extract_file_lines_from_tail(infile, output_dir, outfile=None, count=None):
    """
    Equivalent to: tail -n $count $infile

    :param str infile: File to extract lines from
    :param str output_dir: Output directory
    :param str outfile: File to write to (if None, auto-generated)
    :param int count: How many lines to extract
    :returns: outfile
    """
    if count is None:
        raise ValueError("count parameter is required")
    infile_total_lines = count_file_lines(infile)
    if count > infile_total_lines:
        raise ValueError("Cannot extract %d lines, file has only %d" %
                         (count, infile_total_lines))
    skip_initial = infile_total_lines - count
    return extract_file_lines(infile, output_dir, outfile=outfile,
                              skip_initial=skip_initial)


def extract_file_lines_from_head(infile, output_dir, outfile=None, count=None):
    """
    Equivalent to: head -n $count $infile
    """
    if count is None:
        raise ValueError("count parameter is required")
    return extract_file_lines(infile, output_dir, outfile=outfile, count=count)


def extract_file_lines(infile, output_dir, outfile=None,
                       skip_initial=None, skip_final=None, count=None):
    """
    Similar to the head/tail commands
    This only works on text files

    :param str infile: File to extract lines from
    :param str output_dir: Output directory.
    :param str outfile: File to write to (if None, auto-generated)
    :param int skip_initial: lines to skip from beginning of infile
    :param int skip_final: lines to ignore from the end of infile
    :param int count: total lines to write
    :returns: outfile

    Example: tail -n +3 $infile:
      extract_file_lines(infile, output_dir, skip_initial=2)
    Example: head -n 3 $infile:
      extract_file_lines(infile, output_dir, count=3)
    """
    infile_total_lines = count_file_lines(infile)
    if outfile is None:
        outfile_base = os.path.basename(infile)
        if outfile_base.endswith(".txt"):
            outfile_base = outfile_base[:-4]
        outfile_base += "_relevant"
        outfile = generate_unique_outfile(outfile_base, output_dir, ext="txt")
    if skip_initial is not None:
        skip_initial = int(skip_initial)
    else:
        skip_initial = 0
    if skip_final is not None:
        skip_final = int(skip_final)
    else:
        skip_final = 0
    if count is not None:
        count = int(count)
    else:
        count = infile_total_lines - skip_initial - skip_final
    if skip_initial + count + skip_final > infile_total_lines:
        raise ValueError("Cannot satisfy skip_initial=%s skip_final=%d"
                         " count=%d for file with %d lines" %
                         (skip_initial, skip_final, count,
                          infile_total_lines))
    with open(infile, 'r') as infp:
        # skip_initial:
        if skip_initial:
            for _ in range(skip_initial):
                infp.readline()
        with open(outfile, 'w') as outfp:
            cur_count = 0
            for line in infp:
                outfp.write(line)
                cur_count += 1
                if cur_count >= count:
                    break
    return outfile


def run_numdiff(file1, file2,
                output=None,
                separators=None,
                delimiters=None,
                absolute_tolerance=None,
                relative_tolerance=None,
                strict=False,
                formula=0,
                digits=None,
                positive_differences=False,
                negative_differences=False,
                ignore_case=False,
                essential=False,
                brief=False,
                verbose=False):
    """
    Note: consider using this instead:
      cerebras.regress.datafile.get_numeric_files_correlation()

    Runs the 'numdiff' command, returns its exit code

    :param str file1: file path
    :param str file2: file path

    :param str output: file path (optional)

    :param str separators: set of characters to use as delimiters
    :param str delimiters: set of strings to use as delimiters
    :param absolute_tolerance: float or str (default 0)
                               maximum absolute difference permitted
    :param relative_tolerance: float or str (default 0)
                               maximum relative difference permitted
    :param bool strict: Consider two numerical values as equal only if
                        both absolute and relative difference do not exceed
                        the corresponding tolerance threshold
    :param int formula: 0 for classic formula, 1 for relative errors
                        considering file1 as sample values, 2 for
                        considering file2 as sample values.
    :param int digits: number of digits in the significands used in
                       multiple precision arithmetic
    :param bool positive_differences: If True, gnore all differences due
                                      to numeric fields of the second file
                                      that are less than the corresponding
                                      numeric fields in the first file
    :param bool negative_differences: If True, gnore all differences due
                                      to numeric fields of the second file
                                      that are greater than the
                                      corresponding numeric fields in the
                                      first file
    :param bool ignore_case: If True, ignore changes in case while doing
                             literal comparisons
    :param bool essential: if True, show only the numerical differences
    :param bool brief: Suppress all messages concerning the differences
                       discovered in the structures of the two files
    :param bool verbose: For every couple of lines which differ in at
                         least one field print a header to show how these
                         lines appear in the two compared files
    :returns: exit code of numdiff command
    :rtype: int
    """
    cmd = ["numdiff"]
    if separators is not None:
        cmd.append("--separators=" + separators)
    if delimiters is not None:
        cmd.append("--delimiters=" + delimiters)
    if absolute_tolerance is not None:
        cmd.append("--absolute-tolerance=" + str(absolute_tolerance))
    if relative_tolerance is not None:
        cmd.append("--relative-tolerance=" + str(relative_tolerance))
    if strict:
        cmd.append("--strict")
    cmd.append("--formula=" + str(formula))
    if digits is not None:
        cmd.append("--digits=" + str(digits))
    if positive_differences:
        cmd.append("--positive-differences")
    if negative_differences:
        cmd.append("--negative-differences")
    if ignore_case:
        cmd.append("--ignore-case")
    if essential:
        cmd.append("--essential")
    if brief:
        cmd.append("--brief")
    if verbose:
        cmd.append("--verbose")
    if output:
        cmd.append("--output=" + output)
    cmd.append(file1)
    cmd.append(file2)
    cmdstr = ("module load numdiff ; " +
              ' '.join([shlex.quote(arg) for arg in cmd]))
    ret = subprocess_call(cmdstr, shell=True)
    return ret


def dict_to_cmdline(arg_dict, option_dashes=False, option_equals=False, shell=False):
    """
    Helper function used to term a dict of command-line arguments into
    a list of command-line strs.

    Returns a list
    :param dict arg_dict: arguments to convert to command-line
    :param bool option_dashes: if True make "--" options
    :param bool option_equals: if True args have "="
    :param bool shell: if True, args are shell-quoted

    Example: dict_to_cmdline({'foo': 'bar'}, option_dashes=False)
             ==> ['foo=bar']
    Example: dict_to_cmdline({'foo': 'bar'},
                              option_dashes=True, option_equals=True)
             ==> ['--foo=bar']
    Example: dict_to_cmdline({'foo': 'bar'}, option_dashes=True)
             ==> ['--foo', 'bar']
    Example: dict_to_cmdline({'foo': True}, option_dashes=True)
             ==> ['--foo']
    Example: dict_to_cmdline({'foo': 'fu bar'},
                              option_dashes=True, shell=True)
             ==> ['--foo', '"fu bar"']
    Example: dict_to_cmdline({'foo': ['one', 'two']},
                              option_dashes=True, shell=False)
             ==> ['--foo', 'one', '--foo', 'two']
    """
    cmd_args = []
    for key in arg_dict:
        val = arg_dict[key]
        if shell:
            key = shlex.quote(key)
        if isinstance(val, type(None)):
            raise ValueError("Invalid value None for %r" % key)
        # Handle bool types specially:
        if isinstance(val, bool):
            if val is True:
                if option_dashes:
                    cmd_args.append("--" + key)
                else:
                    cmd_args.append(key)
            continue
        # Handle list types specially:
        elif isinstance(val, (list, tuple)):
            val_list = val
            for val in val_list:
                if shell:
                    val = shlex.quote(val)
                if option_dashes:
                    if option_equals:
                        cmd_args.append("--%s=%s" % (key, val))
                    else:
                        cmd_args.append("--" + key)
                        cmd_args.append(val)
                else:
                    cmd_args.append("%s=%s" % (key, val))
        # Regular types (usually str or int)
        else:
            val = str(val)
            if shell:
                val = shlex.quote(val)
            if option_dashes:
                if option_equals:
                    cmd_args.append("--%s=%s" % (key, val))
                else:
                    cmd_args.append("--" + key)
                    cmd_args.append(val)
            else:
                cmd_args.append("%s=%s" % (key, val))

    return cmd_args


def dict_overlay(base_dict, overlay_dict):
    """
    :param dict base_dict:
    :param dict overlay_dict:

    Overlays a new dict on top of an existing dict.  (Note that this is
    not the same as dict.update().)  Returns a new dict; the original
    base_dict is not modified).

    Example:
      >>> dict_overlay({'params': {'C': 16, 'H': 16}}, {'params': {'C': 9999}})
      {'params': {'C': 9999, 'H': 16}}

    Note: this function uses recursion.
    """
    base_dict = base_dict.copy()  # mutable copy
    for key, val in overlay_dict.items():
        if isinstance(val, dict):
            base_dict[key] = dict_overlay(base_dict.get(key, {}),
                                          overlay_dict[key])
        else:
            base_dict[key] = val
    return base_dict


def get_value_from_dict(obj, key):
    '''
    Find an item in dictionary.
    :param obj: Dictionary object
    :param key: Key can be of two formats:
    1. Simple key: Example 'mykey'
    2. In cases of nested dictionaries, there can be duplicate keys. To avoid
    ambiguity, key can represented as 'field1,filed2,key'. This function will
    search for 'key' only in nested dict obj['field1']['field2'].

    :return: Value of key in obj or None if key doesn't exist.
    '''
    # Get dict obj and key to find.
    new_obj = obj.copy()
    new_key = key
    keys = key.split(",")
    if len(keys) > 1:
        for k in keys[0:-1]:
            if k not in new_obj:
                return None
            if not isinstance(new_obj[k], dict):
                return None
            new_obj = new_obj[k]
        new_key = keys[len(keys) - 1]

    # Search for new_key in new_obj and return value corresponding to new_key.
    if new_key in new_obj:
        return new_obj[new_key]
    for _, v in new_obj.items():
        if isinstance(v,dict):
            item = get_value_from_dict(v, new_key)
            if item is not None:
                return item
    return None


def get_flatten_dict(d, parent_key='', sep=','):
    '''
    Flatten a nested dictionary with sep as delimiter for keys.
    Input: {'foo': {'foo1': 'bar1'}, 'foo2': 'bar2'} with sep="," and
    parent_key=''
    Return: {'foo,foo1': 'bar1', 'foo2': 'bar2'}

    :param d: input dict
    :param parent_key: parent_key, empty to operate on given dictionary.
    :param sep: delimiter for keys
    :return: flatten dictionary
    '''
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(get_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def time_linux_command(cmd, redirect_stdout_to=None):
    '''
    Run linux command with time prefix.

    :param cmd: Command to run in format ["cmd", "arg1", "arg2"...]
    :return: cmd_result dictionary for following format.
             {"stdout": <stdout>,
              "stderr": <stderr>,
              "real": <real time in seconds>,
              "user": <user cpu time in seconds>,
              "sys": <kernel cpu time in seconds>,
              "cpu_time_from_resource": <user+ kernel cpu time computed from resource package>}

    '''
    cmd_result = {}
    time_fields = ["real", "user", "sys"]
    time_cmd = ["time"] + ["-p"] + cmd
    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    if redirect_stdout_to:
        p = subprocess.Popen(time_cmd, stdout=redirect_stdout_to,
                             stderr=subprocess.PIPE)
    else:
        p = subprocess.Popen(time_cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

    cmd_result["stdout"], cmd_result["stderr"] = p.communicate()
    usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_user_time = usage_end.ru_utime - usage_start.ru_utime
    cpu_sys_time = usage_end.ru_stime - usage_start.ru_stime
    cmd_result["cpu_time_from_resource"] = cpu_user_time + cpu_sys_time
    cmd_result["retcode"] = p.returncode

    if cmd_result["stdout"]:
        cmd_result["stdout"] = cmd_result["stdout"].decode('utf-8')

    if cmd_result["stderr"]:
        cmd_result["stderr"] = cmd_result["stderr"].decode('utf-8')

    # Time information is stderr, parse stderr.
    std_err_lines = cmd_result["stderr"].strip().split('\n')
    time_info = std_err_lines[-3:]

    for line in time_info:
        fields = line.split(" ")
        assert len(fields) == 2, "Unexpected time format %s." %(line)
        assert fields[0] in time_fields, "Field %s not expected in time_info"\
                                         %(fields[0])
        cmd_result[fields[0]] = float(fields[1])
    return cmd_result

def get_cpuinfo():
    '''
    Get local cpu info.
    :return: cpu_info ordered dict.
    '''
    cpuinfo = OrderedDict()
    procinfo = OrderedDict()
    nprocs = 0

    with open('/proc/cpuinfo') as f:
        for line in f:
            if not line.strip():
                cpuinfo['proc%s' % nprocs] = procinfo
                nprocs = nprocs+1
                procinfo = OrderedDict()
            else:
                if len(line.split(':')) == 2:
                    procinfo[line.split(':')[0].strip()] = line.split(':')[1].strip()
                else:
                    procinfo[line.split(':')[0].strip()] = ''
    return cpuinfo


########################################


def _signum_to_signame(signum):
    """
    Examine the SIG* constants in the signal module looking for a match
    e.g. 8 -> 'SIGFPE'
    """
    if signum < 0:
        signum = -signum
    for signame in dir(signal):
        if signame.startswith("SIG") and not signame.startswith("SIG_"):
            if getattr(signal, signame) == signum:
                return signame
    return str(signum)


def symlink(target, link_name, overwrite=False):
    """
    https://stackoverflow.com/questions/8299386/modifying-a-symlink-in-python/55742015#55742015
    Provides an (almost) atomic symlink overwrite.
    Use with caution: This has a race condition / security hole
    if another processs modifies the tmp symlink, after the symlink is
    created but before it is replaced.
    """
    if not overwrite:
        os.symlink(target, link_name)
        return

    # os.replace() may fail if files are on different filesystems
    link_dir = os.path.dirname(link_name)

    # Create link to target with temporary filename
    while True:
        temp_link_name = tempfile.mktemp(dir=link_dir)

        # os.* functions mimic as closely as possible system functions
        # The POSIX symlink() returns EEXIST if link_name already exists
        # https://pubs.opengroup.org/onlinepubs/9699919799/functions/symlink.html
        try:
            os.symlink(target, temp_link_name)
            break
        except FileExistsError:
            pass

    # Replace link_name with temp_link_name
    try:
        # Pre-empt os.replace on a directory with a nicer message
        if not os.path.islink(link_name) and os.path.isdir(link_name):
            raise IsADirectoryError(f"Cannot symlink over existing directory: '{link_name}'")
        os.replace(temp_link_name, link_name)
    except:
        if os.path.islink(temp_link_name):
            os.remove(temp_link_name)
        raise

