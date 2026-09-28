"""Parallel-MAC stage: re-lay out the MNIST weights for an N-lane core.

rtl/nn_core_par.v runs N_MAC dot products at once: lane p computes neuron
g*N_MAC + p of the current layer, for group g = 0, 1, ... Every cycle all lanes
see the same activation (pixel k, or hidden[k]) and each lane needs its own
weight, so the weight ROM is N_MAC*16 bits wide and one address feeds all
lanes. The address simply counts up once per issued term, across both layers:

    layer 1: for g in groups(32):  k = 0..195 -> W1[g*N+p][k],  k = 196 -> b1[g*N+p]
    layer 2: for g in groups(10):  k = 0..31  -> W2[g*N+p][k],  k = 32  -> b2[g*N+p]

Lanes whose neuron index is past the end of the layer (e.g. N=4 has 3 output
groups for 10 outputs, the last with only 2 real neurons) hold zeros; the core
ignores their results.

Each neuron still accumulates its terms in the same order as the 1-MAC core,
so every logit is bit-identical to python/train_mnist.py's golden model, for
any N_MAC. The testbench reuses tb/vectors/mnist_exp.hex.

Input: weights/mnist_weights.hex (the exact words the 1-MAC ROM holds).
Output: weights/mnist_par{01,02,04,08,16,32}.hex, one N_MAC*16-bit word per
line, lane N-1 leftmost (lane p at bits [p*16 +: 16]).

Run from the repo root:  python3 python/gen_par_weights.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_DIR = os.path.join(ROOT, "weights")

N_IN, N_HID, N_OUT = 196, 32, 10
LANES = [1, 2, 4, 8, 16, 32]


def ceil_div(a, b):
    return -(-a // b)


def load_rom(path):
    with open(path) as f:
        return [int(t.split("//")[0], 16) for t in f if t.split("//")[0].strip()]


def layout(rom, n):
    b1_base = N_HID * N_IN
    w2_base = b1_base + N_HID
    b2_base = w2_base + N_OUT * N_HID

    def w1(j, k): return rom[j * N_IN + k] if k < N_IN else rom[b1_base + j]
    def w2(o, k): return rom[w2_base + o * N_HID + k] if k < N_HID else rom[b2_base + o]

    rows = []
    for size, n_prev, get in ((N_HID, N_IN, w1), (N_OUT, N_HID, w2)):
        for g in range(ceil_div(size, n)):
            for k in range(n_prev + 1):
                rows.append([get(g * n + p, k) if g * n + p < size else 0 for p in range(n)])
    return rows


def cycles(n):
    """Issue cycles only; the core adds pipeline fill/drain and the argmax."""
    return ceil_div(N_HID, n) * (N_IN + 1) + ceil_div(N_OUT, n) * (N_HID + 1)


def main():
    rom = load_rom(os.path.join(W_DIR, "mnist_weights.hex"))
    assert len(rom) == N_HID * N_IN + N_HID + N_OUT * N_HID + N_OUT
    print(" N_MAC  ROM depth x width   issue cycles")
    for n in LANES:
        rows = layout(rom, n)
        path = os.path.join(W_DIR, "mnist_par%02d.hex" % n)
        with open(path, "w") as f:
            for r in rows:
                f.write("".join("%04X" % w for w in reversed(r)) + "\n")
        print("  %2d    %4d x %3d bits     %5d" % (n, len(rows), 16 * n, cycles(n)))
    print("wrote weights/mnist_parNN.hex for N_MAC in %s" % LANES)


if __name__ == "__main__":
    main()
