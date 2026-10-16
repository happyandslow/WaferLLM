# previously the distutils library was used to compile C code into Python.
# distutils is deprecated so we have manually ported the few src/numeric
# functions needed by Python.

import numpy as np

def mask(b):
    return (1 << b) - 1

FP16_BIAS = 15
FP16_MANT_LEN = 10
FP16_SIGN_BIT = 15
FP16_EXP_LEN = 5

def string_to_fp16(s):
    s = s.strip() # allow newline/whitespace
    # allow "0" or "-0"
    if s == "0" or s == "+0":
        return 0x0
    elif s == "-0":
        return 0x8000
    # otherwise expect format like "-1.450(+05)"
    assert len(s) == 11
    sign = 0x0000 if s[0] == "+" else 0x8000
    if "(000)" in s:
        # plus or minus 0
        return sign

    digits = s[3:6]
    # 12 bit value shifted right 2
    mant_bits = int(digits, 16) >> 2
    exp_s = s[7]
    exp_digits = s[8:10]
    # exponent is in decimal
    exponent = int(exp_digits) * (1 if exp_s == "+" else -1)

    # subnormal cases. exponent will become 0
    if (exponent <= -FP16_BIAS):
        mant_with_1 = mant_bits | (1<< FP16_MANT_LEN)
        mant_shifted = mant_with_1 >> ((-FP16_BIAS) - exponent + 1)
        return sign | mant_shifted
    
    # normal cases
    return sign | ((exponent + FP16_BIAS) << FP16_MANT_LEN) | mant_bits

# mimics the behavior of (now deprecated) fp16_to_string python method.
# similar to fptos but no special casing for nan/inf, just direct byte translation.
# Subnormals, however, are handled.
def fp16_to_string(v):
    assert v <= mask(16) and v >= 0
    # sign and mantissa

    vs = (v & (1 << FP16_SIGN_BIT)) >> (FP16_MANT_LEN + FP16_EXP_LEN)
    ve = (v & (mask(FP16_EXP_LEN) << FP16_MANT_LEN)) >> FP16_MANT_LEN
    vm = v & mask(FP16_MANT_LEN)
    sign_str = ("-" if vs else "+")

    # zero
    if ve == 0 and vm == 0:
        return sign_str + "0.000(000)"
    
    if ve > 0:
        exponent = ve - FP16_BIAS
    else:
        # turn subnormal case into regular case
        exponent = -FP16_BIAS + 1
        # left shift mantissa until there is a 1 in the implicit position
        while vm & (1 << FP16_MANT_LEN) == 0:
            vm = vm << 1
            exponent -= 1
        # preserve only the implicit bits
        vm = vm & mask(FP16_MANT_LEN)

    # now can handle subnormal and subnormal identically

    # e.g. "(-24)"
    exponent_string = "(" + ("+" if exponent >= 0 else "-") + ("%02d)" % abs(exponent))

    # left shift 2 bits so msb 1 is at msb position of 12 bit encoding
    mant_string = "%03x" % (vm << 2)

    return sign_str + "1." + mant_string + exponent_string

# basic float print, up to 8 characters
def fp32_to_string(v):
    return ("%8f" % np.uint32(v).view(np.float32)).strip()

# ported from src/numeric/lfsr.c, used in tests/ws/kernel/common/kernel/test_helpers
# state: u32
def lfsr_expand_to_64(state):
    result = state & mask(20)
    count = 44
    eights = count >> 3
    rest = count % 8

    eight_mask = mask(8)
    tap1, tap2, tap3, tap4 = 20, 19, 16, 14

    # python supports long ints so we need to mask to keep u64
    u64_mask = mask(64)
    for i in range(eights):
        chunk1 = result >> (tap1 - 8)
        chunk2 = result >> (tap2 - 8)
        chunk3 = result >> (tap3 - 8)
        chunk4 = result >> (tap4 - 8)
        new_eight = (chunk1 ^ chunk2 ^ chunk3 ^ chunk4) & eight_mask
        result = ((result << 8) | new_eight) & u64_mask
    
    for i in range(rest):
        bit1 = result >> (tap1 - 1)
        bit2 = result >> (tap2 - 1)
        bit3 = result >> (tap3 - 1)
        bit4 = result >> (tap4 - 1)
        new_bit = (bit1 ^ bit2 ^ bit3 ^ bit4) & 1
        result = ((result << 1) | new_bit) & u64_mask
    
    return result

# ported from src/numeric/lfsr.c, used in tests/ws/kernel/common/kernel/test_helpers
def lfsr_advance(rng_select, state, count):
    if rng_select == 0:
        # lfsr_23_advance
        total = 23
        tap1 = 23
        tap2 = 18
    elif rng_select == 1:
        # lfsr_22 advance
        total = 22
        tap1 = 22
        tap2 = 21
    elif rng_select == 2:
        # lfsr_21_advance
        total = 21
        tap1 = 19
        tap2 = 21
    else:
        # lfsr_20_advance
        total = 20
        tap1 = 17
        tap2 = 20

    # function body of lfsr_*_advance is the same, up to total, tap1 and tap2 values.

    eights = count >> 3
    rest = count % 8

    total_mask = mask(total)
    eight_mask = mask(8)

    for i in range(eights):
        chunk1 = state >> (tap1 - 8)
        chunk2 = state >> (tap2 - 8)
        new_eight = (chunk1 ^ chunk2) & eight_mask
        state = ((state << 8) | new_eight) & total_mask
    
    for i in range(rest):
        bit1 = state >> (tap1 - 1)
        bit2 = state >> (tap2 - 1)
        new_bit = (bit1 ^ bit2) & 1
        state = ((state << 1) | new_bit) & total_mask
    
    return state