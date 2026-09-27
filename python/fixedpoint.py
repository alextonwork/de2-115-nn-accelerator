"""Bit-exact fixed-point helpers shared by training export and test-vector generation.

Everything here mirrors rtl/mac.v exactly, so Python acts as the golden model.

Format: Q8.8 signed, 16-bit two's complement
    value = raw / 2**FRAC_W,  range [-128.0, +127.99609375],  LSB = 1/256
Product of two Q8.8 numbers is Q16.16 (32 bits). The accumulator is 40 bits
(8 guard bits), so at least 256 worst-case products can be summed without wrap.
"""

DATA_W = 16
FRAC_W = 8
ACC_W = 40

ONE = 1 << FRAC_W                    # 1.0 in Q8.8 (= 256)
DATA_MAX = (1 << (DATA_W - 1)) - 1   # +32767
DATA_MIN = -(1 << (DATA_W - 1))      # -32768


def to_fixed(x):
    """Float -> Q8.8 raw integer, round-to-nearest, saturating."""
    r = int(round(x * ONE))
    return max(DATA_MIN, min(DATA_MAX, r))


def to_float(raw):
    return raw / ONE


def wrap(value, width):
    """Two's complement wrap to `width` bits (what a hardware register does)."""
    mask = (1 << width) - 1
    value &= mask
    if value >> (width - 1):
        value -= 1 << width
    return value


def to_unsigned(value, width):
    return value & ((1 << width) - 1)


def mac_result(acc):
    """Accumulator (Q16.16 in ACC_W bits) -> Q8.8, matching mac.v:
    round half up (add 0.5 LSB, arithmetic shift right), then saturate."""
    rounded = (acc + (1 << (FRAC_W - 1))) >> FRAC_W   # Python >> is arithmetic
    return max(DATA_MIN, min(DATA_MAX, rounded))


def dot(a_list, b_list):
    """Golden MAC: returns (acc, result) exactly as the hardware produces them."""
    acc = 0
    for a, b in zip(a_list, b_list):
        acc = wrap(acc + a * b, ACC_W)
    return acc, mac_result(acc)
