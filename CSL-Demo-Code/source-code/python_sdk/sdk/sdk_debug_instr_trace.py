#!/usr/bin/env python3

'''
sdk debug instruction trace library
'''

from enum import Enum
import numpy
import tabulate

from cerebras.sdk.debug.lib.instruction_trace import sdkinstrtracepybind

from cerebras.sdk.debug.lib.instruction_trace.sdkinstrtracepybind import (
    get_opname_size,
    get_operand_format,
    data_format_t,
    operand_type_t,
    arch_e
)

import cerebras.numeric.numeric as cnum

class InstrTraceField(Enum):
    """
    Enum for possible field names
    """
    cycle = 0
    opaddr = 1
    opname = 2
    dest = 3
    src0 = 4
    src1 = 5
    src2 = 6
    source = 7
    source_line = 8
    task = 9
    is_uthread = 10
    opxaddr = 11
    opvalue = 12
    instr_index = 13
    opcode = 14
    instruction = 15
    id = 16


def get_instrs_array(num_instrs):
    """
    Return an array of instructions.
    """

    instr_dtype = numpy.dtype(
        [(InstrTraceField.id.name, 'u4'),
         (InstrTraceField.opcode.name, 'u4'),
         (InstrTraceField.task.name, 'u1'),
         (InstrTraceField.is_uthread.name, '?'),
         (InstrTraceField.opaddr.name, 'u2'),
         (InstrTraceField.opname.name, (f"S{get_opname_size()}"))
        ])
    instrs = numpy.zeros(shape=num_instrs, dtype=instr_dtype)
    return instrs


def get_sequences_array(num_sequences):
    """
    Return an array of sequences.
    """
    sequence_dtype = numpy.dtype(
        [(InstrTraceField.cycle.name, 'i4'),
         (InstrTraceField.opxaddr.name, ('u2', 4) ),
         (InstrTraceField.opvalue.name, ('u4', 4) ),
         (InstrTraceField.instr_index.name, 'u4')
        ])
    sequences = numpy.zeros(shape=num_sequences, dtype=sequence_dtype)
    return sequences

def get_trace(instrs, sequences):
    """
    Return a trace object with the given instrs and sequences.
    """
    trace = sdkinstrtracepybind.Trace(instrs, sequences) \
      # pylint: disable=c-extension-no-member
    return trace

def print_operand_value(opvalue, operand_format):
    """
    Print the operand value based on the given operand format.
    """
    if operand_format == data_format_t.DATA_FORMAT_XP16:
        return f'{opvalue}'
    elif operand_format == data_format_t.DATA_FORMAT_FP16 or \
        operand_format == data_format_t.DATA_FORMAT_CB16:
        return f'{numpy.uint16(opvalue).view(numpy.float16).astype(numpy.float64)}'
    elif operand_format == data_format_t.DATA_FORMAT_FP32:
        return f'{cnum.fp32_to_string(opvalue)}'
    else:
        return f'{opvalue:#06x}'

def iterate_iterator(instr_trace, itr, x, y, num_sequences):
    """
    Iterate through the given iterator, and return sequence as an iterator.
    (iterated type: a dictionary for all InstrTraceFields and their values)
    """
    instrs = get_instrs_array(num_sequences)
    sequences = get_sequences_array(num_sequences)

    while itr.has_next():
        trace = get_trace(instrs, sequences)
        itr.get_next(trace)
        curr_instrs = trace.instrs()
        assert curr_instrs.ctypes.data == \
            instrs.ctypes.data, "instrs array does not match"
        curr_sequences = trace.sequences()
        assert curr_sequences.ctypes.data == \
            sequences.ctypes.data, "sequences array does not match"
        for i in range(trace.num_sequences):
            sequence = curr_sequences[i]
            instr_index = sequence[InstrTraceField.instr_index.name]
            assert instr_index < trace.num_instrs, \
                f"instr_index:{instr_index} is not less than {trace.num_instrs}"
            instr = curr_instrs[instr_index]
            opname = instr[InstrTraceField.opname.name]
            task = instr[InstrTraceField.task.name]

            row = {}
            row[InstrTraceField.cycle.value] = sequence[InstrTraceField.cycle.name]
            row[InstrTraceField.opaddr.value] = hex(instr[InstrTraceField.opaddr.name])
            row[InstrTraceField.opname.value] = opname
            row[InstrTraceField.task.value] = task

            opvalue_index = [
                InstrTraceField.src0.value,
                InstrTraceField.src1.value,
                InstrTraceField.src2.value,
                InstrTraceField.dest.value
            ]
            for operand_type in [ operand_type_t.OPERAND_TYPE_DEST,
                operand_type_t.OPERAND_TYPE_SRC0,
                operand_type_t.OPERAND_TYPE_SRC1,
                operand_type_t.OPERAND_TYPE_SRC2 ]:
                no = int(operand_type) - 1
                assert(no < 4 and no >= 0)
                opvalue = sequence[InstrTraceField.opvalue.name][no]
                if opvalue != 0:
                    # only support EINSTEIN and FEYNMAN for now.
                    operand_format = get_operand_format(
                        arch_e.FEYNMAN, instr[InstrTraceField.opcode.name], operand_type)
                    row[opvalue_index[no]] = "{} ({})".format(
                        hex(sequence[InstrTraceField.opxaddr.name][no]),
                        print_operand_value(opvalue, operand_format))
                else:
                    row[opvalue_index[no]] = ""

            source_line = instr_trace.get_source_line(x, y, instr[InstrTraceField.opaddr.name])
            row[InstrTraceField.source_line.value] = source_line
            yield row


def print_iterator(instr_trace, x, y, itr, num_sequences):
    """
    Print all traces using a given iterator.
    """
    total_cycles = 0
    cycles_per_instr = {}

    headers = [
        InstrTraceField.cycle.name,
        InstrTraceField.opaddr.name,
        InstrTraceField.opname.name,
        InstrTraceField.dest.name,
        InstrTraceField.src0.name,
        InstrTraceField.src1.name,
        InstrTraceField.src2.name,
        InstrTraceField.source.name,
        InstrTraceField.task.name
    ]
    rows = []
    for values in iterate_iterator(instr_trace, itr, x, y, num_sequences):
        source_line = values[InstrTraceField.source_line.value]
        if source_line.source_file == "?":
            source_line_str = ""
        else:
            source_line_str = f"{source_line.source_file}:{source_line.line}"

        row = [
            values[InstrTraceField.cycle.value],
            values[InstrTraceField.opaddr.value],
            values[InstrTraceField.opname.value],
            values[InstrTraceField.dest.value],
            values[InstrTraceField.src0.value],
            values[InstrTraceField.src1.value],
            values[InstrTraceField.src2.value],
            source_line_str,
            values[InstrTraceField.task.value]
        ]
        rows.append(row)

        total_cycles += 1
        opaddr = row[InstrTraceField.opaddr.value]
        if opaddr in cycles_per_instr:
            (cycles, name) = cycles_per_instr[opaddr]
            cycles += 1
            cycles_per_instr[opaddr] = (cycles, name)
        else:
            cycles_per_instr[opaddr] = (1, row[InstrTraceField.opname.value])

    print(tabulate.tabulate(rows, headers=headers))

    print("Summary:")
    headers = [
        'cycles (%)',
        InstrTraceField.opaddr.name,
        InstrTraceField.instruction.name
    ]
    rows = []
    row = [f"{total_cycles} (100%)", 'ALL']
    rows.append(row)
    sorted_cycles_per_instr = \
        dict(sorted(cycles_per_instr.items(), reverse=True, key=lambda item: item[1]))
    for opaddr, value in sorted_cycles_per_instr.items():
        (cycles, opname) = value
        row = ["{} ({:0.2f}%)".format(cycles, cycles/total_cycles * 100), opaddr, opname]
        rows.append(row)
    print(tabulate.tabulate(rows, headers=headers))


class InstrTrace:
    """
    InstrTrace object
    """
    def __init__(self, elf_dir, simfab_log):
        self.instr_trace = sdkinstrtracepybind.SimfabInstructionTrace( \
            # pylint: disable=c-extension-no-member
            elf_dir,
            simfab_log
        )

    def print_trace(self, x, y, num_sequences=1000):
        """
        Print traces for a given (x, y).
        """
        itr = self.instr_trace.get_instructions(x, y)
        print_iterator(self.instr_trace, x, y, itr, num_sequences)

    def get_tasks(self, x, y):
        """
        Return a list of tasks for a given (x, y).
        """
        return self.instr_trace.get_tasks(x, y)

    def print_tasks(self, x, y):
        """
        Print a list of tasks for a given (x, y).
        """
        print(self.instr_trace.get_tasks(x, y))

    def get_uthreads(self, x, y):
        """
        Return a list of uthreads for a given (x, y).
        """
        return self.instr_trace.get_uthreads(x, y)

    def print_uthreads(self, x, y):
        """
        Print a list of uthreads for a given (x, y).
        """
        print(self.instr_trace.get_uthreads(x, y))

    def print_trace_per_task(self, x, y, task, num_sequences=1000):
        """
        Print traces for a given (x, y) and a given task.
        """
        itr = self.instr_trace.get_instructions_per_task(x, y, task)
        print_iterator(self.instr_trace, x, y, itr, num_sequences)

    def print_trace_per_uthread(self, x, y, uthread, num_sequences=1000):
        """
        Print traces for a given (x, y) and a given uthread.
        """
        itr = self.instr_trace.get_instructions_per_uthread(x, y, uthread)
        print_iterator(self.instr_trace, x, y, itr, num_sequences)

    def get_trace_iterator(self, x, y, num_sequences=1000):
        """
        Return trace iterator for all traces in a given (x, y).
        """
        itr = self.instr_trace.get_instructions(x, y)
        return iterate_iterator(self.instr_trace, itr, x, y, num_sequences)

    def get_trace_per_task_iterator(self, x, y, task, num_sequences=1000):
        """
        Return trace iterator for all traces in a given (x, y) and a given task.
        """
        itr = self.instr_trace.get_instructions_per_task(x, y, task)
        return iterate_iterator(self.instr_trace, itr, x, y, num_sequences)

    def get_trace_per_uthread_iterator(self, x, y, uthread, num_sequences=1000):
        """
        Return trace iterator for all traces in a given (x, y) and a given uthread.
        """
        itr = self.instr_trace.get_instructions_per_uthread(x, y, uthread)
        return iterate_iterator(self.instr_trace, itr, x, y, num_sequences)
