"""Step 3: generate stimulus + golden expected results for tb/tb_mac.v.

Python (fixedpoint.dot) is the reference model; the testbench replays the
stimulus cycle by cycle and checks every finished dot product bit-for-bit.

Outputs (tb/vectors/):
  mac_in.hex   one line per clock: {valid, start, last, 1'b0, a[15:0], b[15:0]}  (9 hex digits)
  mac_exp.hex  one line per dot product: {acc[39:0], result[15:0]}             (14 hex digits)
  mac_vectors.vh  `define N_IN / N_EXP counts for the testbench

Run from the repo root:  python3 python/gen_mac_vectors.py   (after train_xor.py)
"""
import os
import random
import numpy as np

from fixedpoint import (ONE, DATA_MAX, DATA_MIN, DATA_W, ACC_W, to_fixed,
                        to_unsigned, dot)
import train_xor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_DIR = os.path.join(ROOT, "tb", "vectors")

rng = random.Random(1234)
cycles = []      # (valid, start, last, a, b)
expected = []    # (acc, result, label)


def add_dot(a_list, b_list, label, gap_prob=0.0):
    """Queue one dot product; optionally insert idle cycles between terms."""
    n = len(a_list)
    for k, (a, b) in enumerate(zip(a_list, b_list)):
        while rng.random() < gap_prob:
            # idle cycle with garbage on the data bus: the MAC must ignore it
            cycles.append((0, rng.randint(0, 1), rng.randint(0, 1),
                           rng.randint(DATA_MIN, DATA_MAX), rng.randint(DATA_MIN, DATA_MAX)))
        cycles.append((1, int(k == 0), int(k == n - 1), a, b))
    acc, res = dot(a_list, b_list)
    expected.append((acc, res, label))


def rand_q(lo=-8.0, hi=8.0):
    return to_fixed(rng.uniform(lo, hi))


def main():
    f = to_fixed
    # ---- directed: basic arithmetic ----
    add_dot([f(1.0)], [f(1.0)], "1.0 * 1.0 = 1.0")
    add_dot([f(-1.5)], [f(2.25)], "-1.5 * 2.25 = -3.375")
    add_dot([f(0.5), f(0.25)], [f(4.0), f(-8.0)], "0.5*4 + 0.25*-8 = 0")
    add_dot([0], [DATA_MIN], "0 * min = 0")
    # ---- directed: rounding (half up) ----
    add_dot([1], [128], "1 LSB * 0.5 -> 0.5 LSB rounds up to 1 LSB")
    add_dot([-1], [128], "-1 LSB * 0.5 -> -0.5 LSB rounds up to 0")
    add_dot([-1], [129], "-1 LSB * 0.504 -> rounds to -1 LSB")
    add_dot([3], [85], "3 * 85 / 2^16 -> truncation vs rounding check")
    # ---- directed: saturation ----
    add_dot([DATA_MAX], [DATA_MAX], "max*max saturates to +max")
    add_dot([DATA_MIN], [DATA_MIN], "min*min (+128^2) saturates to +max")
    add_dot([DATA_MIN], [DATA_MAX], "min*max saturates to min")
    add_dot([f(100.0), f(-100.0)], [f(1.0), f(1.0)], "100 - 100 = 0, no sat")
    add_dot([f(100.0), f(100.0)], [f(1.0), f(1.0)], "100 + 100 saturates")
    # ---- directed: guard bits, 256 worst-case terms must not wrap ----
    add_dot([DATA_MIN] * 256, [DATA_MIN] * 256, "256 x min*min uses all 8 guard bits")
    add_dot([DATA_MIN] * 200 + [DATA_MAX] * 200, [DATA_MIN] * 200 + [DATA_MIN] * 200,
            "large intermediate sum returning near 0")
    # ---- XOR network neurons, using the trained weights (bias = extra term * 1.0) ----
    for seed in range(100):
        params, p, loss = train_xor.train(seed)
        if np.all((p > 0.5) == (train_xor.Y > 0.5)) and loss < 0.05:
            break
    W1, b1, W2, b2 = params
    qW1 = [[f(v) for v in row] for row in W1]; qb1 = [f(v) for v in b1]
    qW2 = [[f(v) for v in row] for row in W2]; qb2 = [f(v) for v in b2]
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        xq = [x1 * ONE, x2 * ONE]
        hidden = []
        for j in range(len(qb1)):
            add_dot(qW1[j] + [qb1[j]], xq + [ONE], "XOR x=(%d,%d) hidden[%d]" % (x1, x2, j))
            hidden.append(max(expected[-1][1], 0))
        add_dot(qW2[0] + [qb2[0]], hidden + [ONE], "XOR x=(%d,%d) output logit" % (x1, x2))
    # ---- back-to-back single-term products (start and last on every cycle) ----
    for _ in range(20):
        add_dot([rand_q()], [rand_q()], "back-to-back single term")
    # ---- random dot products, random length, random idle gaps ----
    for i in range(300):
        n = rng.randint(1, 32)
        gap = rng.choice([0.0, 0.0, 0.3])
        add_dot([rand_q() for _ in range(n)], [rand_q() for _ in range(n)],
                "random #%d len %d" % (i, n), gap_prob=gap)
    # ---- random full-range values (exercises saturation a lot) ----
    for i in range(100):
        n = rng.randint(1, 16)
        add_dot([rng.randint(DATA_MIN, DATA_MAX) for _ in range(n)],
                [rng.randint(DATA_MIN, DATA_MAX) for _ in range(n)],
                "full-range random #%d len %d" % (i, n))

    os.makedirs(VEC_DIR, exist_ok=True)
    with open(os.path.join(VEC_DIR, "mac_in.hex"), "w") as fh:
        for v, s, l, a, b in cycles:
            word = (v << 35) | (s << 34) | (l << 33) | \
                   (to_unsigned(a, DATA_W) << 16) | to_unsigned(b, DATA_W)
            fh.write("%09X\n" % word)
    with open(os.path.join(VEC_DIR, "mac_exp.hex"), "w") as fh:
        for acc, res, label in expected:
            word = (to_unsigned(acc, ACC_W) << DATA_W) | to_unsigned(res, DATA_W)
            fh.write("%014X  // %s\n" % (word, label))
    with open(os.path.join(VEC_DIR, "mac_vectors.vh"), "w") as fh:
        fh.write("// generated by python/gen_mac_vectors.py\n")
        fh.write("`define N_IN  %d\n`define N_EXP %d\n" % (len(cycles), len(expected)))
    print("wrote %d input cycles, %d expected dot products to tb/vectors/"
          % (len(cycles), len(expected)))


if __name__ == "__main__":
    main()
