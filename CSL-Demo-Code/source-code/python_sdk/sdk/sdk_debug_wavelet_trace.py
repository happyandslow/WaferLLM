#!/usr/bin/env python3

'''
sdk debug wavelet trace library
'''
import struct

import numpy

import cerebras.sdk.debug.lib.wavelet_trace.wavelettracepybind as sdkwavelettracepybind

def fp16_to_double(v):
    """
    v: 16-bit int encoding a float16
    returns: equivalent float (64-bit)
    """
    return numpy.uint16(v).view(numpy.float16).astype(numpy.float64)

def get_wavelet_array(num_wavelets):
    """
    Return an array of wavelets.
    """
    wavelets_dtype = numpy.dtype(
        [
            ('color', 'u1'),
            ('link', 'u1'),
            ('ctrl', 'u1'),
            ('wvlt_idx', 'u2'),
            ('wlvt_data', 'u2'),
            ('cycle', 'u8'),
            ('ident', 'u8'),
        ]
    )
    wavelets = numpy.zeros(shape=num_wavelets, dtype=wavelets_dtype)
    return wavelets


def get_trace(wavelets):
    """
    Return a trace object with the given instrs and sequences.
    """
    trace = sdkwavelettracepybind.Wavlets(  # pylint: disable=c-extension-no-member
        wavelets
    )
    return trace

# use same mapping as simfabric:
# https://github.com/Cerebras/monolith/blob/master/src/fabricmodel/frame/fnet.h#L33
LINK_INT_TO_CHAR = {
    0 : 'E',
    1 : 'N',
    2 : 'W',
    3 : 'S',
    4 : 'R',
}
def get_link_char(link):
    """
    convert wavelet link int to char
    """
    link_char = LINK_INT_TO_CHAR.get(link, None)
    assert link != None, f'Invalid link {link}'
    return link_char


def format_wavelet_bits(fmt, wvlt_idx, wvlt_data):
    """
    Format the wavelet index and data bits in requested format
    Returns a string of formated data
    """
    if fmt == 'i16':
        w_fmt = f'{wvlt_idx},{wvlt_data}'
    elif fmt == 'f16':
        w_fmt = f'{fp16_to_double(wvlt_idx)},{fp16_to_double(wvlt_data)}'
    elif fmt == 'i16,f16':
        w_fmt = f'{wvlt_idx},{fp16_to_double(wvlt_data)}'
    elif fmt == 'f32':
        combined = (wvlt_idx << 16) | wvlt_data
        e = struct.unpack('<f', struct.pack('<I', combined))[0]
        w_fmt = f'{e}'
    elif fmt == 'hex':
        # for debug comparison with landing logs
        w_fmt = f'{wvlt_idx:#06x},{wvlt_data:#06x}'
    return w_fmt


def iterate_iterator(itr, fmt):
    """
    Take a wavelet iterator from pybind, decode data with given format, return
    wavelets as a Python iterator (iterated type:
     tuple of (cycle, color, ident, link, ctrl, formatted data))
    """
    while itr.has_next():
        wavelets = get_trace(get_wavelet_array(10))
        itr.get_next(wavelets)
        for i in range(wavelets.num_wavelets()):
            w = wavelets.get_wavelet(i)
            lnk = get_link_char(w.link)
            w_fmt = format_wavelet_bits(fmt, w.wvlt_idx, w.wvlt_data)
            yield (w.cycle, w.color, w.ident, lnk, w.ctrl, w_fmt)

def iterate_vector(trace):
    """
    Take a trace from pybind, return trace as a Python iterator
    (iterated type: tuple of (cycle, color, link))
    """
    for trace_element in trace:
        lnk = get_link_char(trace_element.link)
        yield (trace_element.cycle, trace_element.color, lnk)

def print_iterator(itr, fmt):
    """
    Print all traces using a given iterator.
    """
    print(f"cycle\t color\t ident\t\t link\t ctrl\t wvlt({fmt})")
    for cycle, color, ident, lnk, ctrl, data in iterate_iterator(itr, fmt):
        print(f"{cycle}\t {color}\t {ident}\t {lnk}\t {ctrl}\t {data}")

def print_vector(trace):
    """
    Print trace vector.
    """
    print(f"cycle\t color\t link")
    for cycle, color, lnk in iterate_vector(trace):
        print(f"{cycle}".ljust(8), end = " ")
        print(f"{color}".ljust(7), end = " ")
        print(f"{lnk}")

class WaveletTrace:
    """
    WaveletTrace object
    """

    def __init__(self, simfab_log):
        self.wavelet_trace = sdkwavelettracepybind.SimfabWaveletTrace(  # pylint: disable=c-extension-no-member
            simfab_log
        )

    @staticmethod
    def print_header_msg(msg):
        """
        Print header details
        """
        print('-'*len(msg))
        print(f'{msg}')
        print('-'*len(msg))

    def print_trace(self, x, y, fmt):
        """
        Print traces for a given (x, y, fmt).
        """
        msg = f'Showing wavelets received at router on P{x}.{y} for all colors'
        self.print_header_msg(msg)
        itr = self.wavelet_trace.get_wavelets(x, y)
        print_iterator(itr, fmt)

    def get_trace_iterator(self, x, y, fmt):
        """
        Called by sdk-viz
        """
        itr = self.wavelet_trace.get_wavelets(x, y)
        return iterate_iterator(itr, fmt)

    def print_trace_per_colors(self, x, y, colors, fmt):
        """
        Print traces for a given (x, y, color, fmt) and a given color.
        """
        msg = f'Showing wavelets received at router on P{x}.{y} for color(s) {colors}'
        self.print_header_msg(msg)
        itr = self.wavelet_trace.get_wavelets_per_colors(x, y, colors)
        print_iterator(itr, fmt)

    def get_trace_iterator_per_colors(self, x, y, colors, fmt):
        """
        Called by sdk-viz
        """
        itr = self.wavelet_trace.get_wavelets_per_colors(x, y, colors)
        return iterate_iterator(itr, fmt)

    def print_backpressure_trace(self, x, y):
        """
        Print backpressure stall traces for a given (x, y, fmt).
        """
        msg = f'Showing backpressure stalls at all routers on P{x}.{y} for all colors'
        self.print_header_msg(msg)
        bckpress_trace = self.wavelet_trace.get_backpressure_trace(x, y)
        print_vector(bckpress_trace)

    def get_backpressure_trace(self, x, y):
        """
        Can be called by sdk-viz
        """
        bckpress_trace = self.wavelet_trace.get_backpressure_trace(x, y)
        return iterate_vector(bckpress_trace)

    def print_backpressure_trace_per_colors(self, x, y, colors):
        """
        Print backpressure traces for a given (x, y) and a given color.
        """
        msg = f'Showing backpressure stalls at all routers on P{x}.{y} for color(s) {colors}'
        self.print_header_msg(msg)
        bckpress_trace = self.wavelet_trace.get_backpressure_trace_per_colors(x, y, colors)
        print_vector(bckpress_trace)

    def get_backpressure_trace_per_colors(self, x, y, colors):
        """
        Can be called by sdk-viz
        """
        bckpress_trace = self.wavelet_trace.get_backpressure_trace_per_colors(x, y, colors)
        return iterate_vector(bckpress_trace)
