"""Re-pack the MNIST weights for the N-lane core (rtl/nn_core_par.v).

Reads weights/mnist_weights.hex (the single-MAC memory map written by
train_mnist.py: W1 row-major, b1, W2 row-major, b2) and writes one wide ROM
per lane count:

  weights/mnist_weights_parNN.hex    one row per clock, N words per row,
                                     lane 0 in the lowest 16 bits

Row order, which the core reads with a plain counter:

  for each group g of the hidden layer (32/N groups):
      N_IN rows of W1[g*N + lane][i], then one row of b1[g*N + lane]
  for each group g of the output layer (ceil(10/N) groups):
      N_HID rows of W2[g*N + lane][j], then one row of b2[g*N + lane]

Lanes past the last output neuron hold zeros. The values are the same Q8.8
words, only reordered, so every N gives bit-identical logits.
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
W_DIR = os.path.join(ROOT, "weights")

N_IN, N_HID, N_OUT = 196, 32, 10
N_MACS = (1, 2, 4, 8, 16, 32)


def load_words(path):
    with open(path) as f:
        return [int(line, 16) for line in f if line.strip()]


def split(words):
    b1_base = N_HID * N_IN
    w2_base = b1_base + N_HID
    b2_base = w2_base + N_OUT * N_HID
    assert len(words) == b2_base + N_OUT, len(words)
    w1 = [words[j * N_IN:(j + 1) * N_IN] for j in range(N_HID)]
    b1 = words[b1_base:w2_base]
    w2 = [words[w2_base + o * N_HID:w2_base + (o + 1) * N_HID] for o in range(N_OUT)]
    b2 = words[b2_base:]
    return w1, b1, w2, b2


def pack(w1, b1, w2, b2, n):
    """Rows of n lane words each, in the order nn_core_par reads them."""
    assert N_HID % n == 0
    rows = []

    def layer(w, b, n_neurons, n_prev):
        for g in range((n_neurons + n - 1) // n):
            neurons = [g * n + lane for lane in range(n)]
            for k in range(n_prev + 1):
                row = []
                for j in neurons:
                    if j >= n_neurons:
                        row.append(0)
                    else:
                        row.append(b[j] if k == n_prev else w[j][k])
                rows.append(row)

    layer(w1, b1, N_HID, N_IN)
    layer(w2, b2, N_OUT, N_HID)
    return rows


def write_hex(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write("".join("%04x" % v for v in reversed(row)) + "\n")


def main():
    w1, b1, w2, b2 = split(load_words(os.path.join(W_DIR, "mnist_weights.hex")))
    for n in N_MACS:
        rows = pack(w1, b1, w2, b2, n)
        path = os.path.join(W_DIR, "mnist_weights_par%02d.hex" % n)
        write_hex(path, rows)
        print("wrote %s: %d rows x %d bits (%d bits total)"
              % (os.path.relpath(path, ROOT), len(rows), 16 * n, 16 * n * len(rows)))


if __name__ == "__main__":
    main()
