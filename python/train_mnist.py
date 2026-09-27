"""MNIST stage: train a 196-32-10 MLP on 14x14 digits, quantize to Q8.8, export ROMs.

Network:  x(196) -> Dense(32) -> ReLU -> Dense(10) -> argmax
          (softmax only during training; argmax of the logits is the same class)

Inputs are the 28x28 MNIST digits average-pooled 2x2 down to 14x14, scaled to
[0, 1] and quantized to Q8.8 like everything else (1.0 = 0x0100).

Everything is plain numpy so the only dependency stays numpy. MNIST is
downloaded once into data/mnist/ (git-ignored).

Outputs:
  weights/mnist_weights.mif / .hex   6634 words, same memory map as XOR:
                                     W1 (32x196) | b1 (32) | W2 (10x32) | b2 (10)
  weights/mnist_images.mif / .hex    16 demo test digits, 256 words per image
                                     (196 pixels + zero padding, so the board
                                     address is just {image, pixel})
  weights/mnist_labels.hex           the 16 true labels
  weights/mnist_weights_float.npz    float weights, for reference
  tb/vectors/mnist_*                 golden vectors for tb/tb_mnist.v and
                                     tb/tb_mnist_top.v

Run from the repo root:  python3 python/train_mnist.py
"""
import gzip
import os
import urllib.request

import numpy as np

from fixedpoint import ONE, DATA_W, FRAC_W, ACC_W, DATA_MIN, DATA_MAX, to_unsigned

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "mnist")
W_DIR = os.path.join(ROOT, "weights")
VEC_DIR = os.path.join(ROOT, "tb", "vectors")

N_IN, N_HID, N_OUT = 196, 32, 10
N_DEMO = 16          # digits stored on the board, selected with SW[3:0]
IMG_STRIDE = 256     # words per image in the image ROM (power of two)
N_SIM = 1000         # test images run through the RTL by tb/tb_mnist.v

MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
           "https://storage.googleapis.com/cvdf-datasets/mnist/"]
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}


# ---------------------------------------------------------------- data
def fetch(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        for m in MIRRORS:
            try:
                print("downloading", m + name)
                urllib.request.urlretrieve(m + name, path)
                break
            except OSError as e:
                print("  failed:", e)
        else:
            raise RuntimeError("could not download " + name)
    with gzip.open(path, "rb") as f:
        return f.read()


def load_images(name):
    raw = fetch(name)
    n = int.from_bytes(raw[4:8], "big")
    return np.frombuffer(raw, np.uint8, offset=16).reshape(n, 28, 28)


def load_labels(name):
    return np.frombuffer(fetch(name), np.uint8, offset=8).astype(np.int64)


def downsample(imgs):
    """28x28 uint8 -> 14x14 float in [0, 1] by 2x2 average pooling."""
    x = imgs.astype(np.float64).reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)) / 255.0
    return x.reshape(len(imgs), N_IN)


# ---------------------------------------------------------------- training
def train(xtr, ytr, xva, yva, epochs=20, batch=128, lr=1e-3, wd=1e-4, seed=0):
    """Mini-batch Adam on softmax cross-entropy. Small L2 keeps weights well
    inside the Q8.8 range and makes the net less sensitive to rounding."""
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(0, np.sqrt(2 / N_IN), (N_HID, N_IN)),
        "b1": np.zeros(N_HID),
        "W2": rng.normal(0, np.sqrt(2 / N_HID), (N_OUT, N_HID)),
        "b2": np.zeros(N_OUT),
    }
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    b1m, b2m, t = 0.9, 0.999, 0
    n = len(xtr)
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            x, y = xtr[idx], ytr[idx]
            z1 = x @ params["W1"].T + params["b1"]
            h = np.maximum(z1, 0)
            z2 = h @ params["W2"].T + params["b2"]
            z2 -= z2.max(1, keepdims=True)
            p = np.exp(z2)
            p /= p.sum(1, keepdims=True)
            dz2 = p
            dz2[np.arange(len(y)), y] -= 1
            dz2 /= len(y)
            dh = dz2 @ params["W2"]
            dz1 = dh * (z1 > 0)
            grads = {"W2": dz2.T @ h + wd * params["W2"], "b2": dz2.sum(0),
                     "W1": dz1.T @ x + wd * params["W1"], "b1": dz1.sum(0)}
            t += 1
            for k in params:
                m[k] = b1m * m[k] + (1 - b1m) * grads[k]
                v[k] = b2m * v[k] + (1 - b2m) * grads[k] ** 2
                mh = m[k] / (1 - b1m ** t)
                vh = v[k] / (1 - b2m ** t)
                params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if ep % 5 == 4 or ep == epochs - 1:
            print("  epoch %2d  val acc %.2f%%" % (ep + 1, 100 * float_accuracy(params, xva, yva)))
    return params


def float_forward(params, x):
    h = np.maximum(x @ params["W1"].T + params["b1"], 0)
    return h @ params["W2"].T + params["b2"]


def float_accuracy(params, x, y):
    return np.mean(float_forward(params, x).argmax(1) == y)


# ---------------------------------------------------------------- fixed point
def quantize(a):
    """Float array -> Q8.8 raw ints, round to nearest, saturating (= to_fixed)."""
    return np.clip(np.round(a * ONE), DATA_MIN, DATA_MAX).astype(np.int64)


def mac_layer(xq, Wq, bq):
    """Bit-exact nn_core/mac.v arithmetic for one layer, vectorized.

    Each neuron is sum_i W[j][i] * x[i] + b[j] * 1.0 accumulated in Q16.16,
    then rounded half up and saturated to Q8.8 (fixedpoint.mac_result).
    """
    acc = xq @ Wq.T + bq * ONE
    # the 40-bit accumulator never wraps for these sizes; make sure of it
    assert np.abs(acc).max() < (1 << (ACC_W - 1)), "accumulator would wrap"
    return np.clip((acc + (1 << (FRAC_W - 1))) >> FRAC_W, DATA_MIN, DATA_MAX)


def fixed_forward(q, xq):
    hidden = np.maximum(mac_layer(xq, q["W1"], q["b1"]), 0)      # ReLU
    logits = mac_layer(hidden, q["W2"], q["b2"])
    pred = logits.argmax(1)     # first maximum wins, same as the RTL's strict '>'
    return hidden, logits, pred


# ---------------------------------------------------------------- export
def write_mif(path, words, comments, title):
    with open(path, "w") as f:
        f.write("-- %s, Q%d.%d signed, generated by python/train_mnist.py\n"
                % (title, DATA_W - FRAC_W, FRAC_W))
        f.write("WIDTH=%d;\nDEPTH=%d;\n\nADDRESS_RADIX=UNS;\nDATA_RADIX=HEX;\n\n"
                % (DATA_W, len(words)))
        f.write("CONTENT BEGIN\n")
        for addr, (w, c) in enumerate(zip(words, comments)):
            f.write("    %d : %04X;%s\n" % (addr, to_unsigned(int(w), DATA_W),
                                           ("  -- " + c) if c else ""))
        f.write("END;\n")


def write_hex(path, words):
    """$readmemh file, one word per line. Quartus loads these too, so they
    carry no comments; the .mif twin has the annotated copy."""
    with open(path, "w") as f:
        for w in words:
            f.write("%04X\n" % to_unsigned(int(w), DATA_W))


def pick_demo(y):
    """First test image of each digit 0-9, then the next lowest indices not yet
    used, so all ten classes are on the board. Deterministic, not cherry-picked
    for correctness."""
    idx = [int(np.argmax(y == d)) for d in range(10)]
    for i in range(len(y)):
        if len(idx) == N_DEMO:
            break
        if i not in idx:
            idx.append(i)
    return idx


def ascii_digit(img14):
    shades = " .:-=+*#%@"
    return ["".join(shades[min(9, int(p * 10))] * 2 for p in row) for row in img14]


def main():
    xtr = downsample(load_images(FILES["train_x"]))
    ytr = load_labels(FILES["train_y"])
    xte = downsample(load_images(FILES["test_x"]))
    yte = load_labels(FILES["test_y"])
    # last 5k training images held out for validation; test set untouched
    xva, yva = xtr[55000:], ytr[55000:]
    xtr, ytr = xtr[:55000], ytr[:55000]
    print("train %d, val %d, test %d, inputs %d (14x14)" % (len(xtr), len(xva), len(xte), N_IN))

    params = train(xtr, ytr, xva, yva)
    acc_f = float_accuracy(params, xte, yte)

    q = {k: quantize(v) for k, v in params.items()}
    xq_te = quantize(xte)
    _, logits_te, pred_te = fixed_forward(q, xq_te)
    acc_q = np.mean(pred_te == yte)
    agree = np.mean(pred_te == float_forward(params, xte).argmax(1))

    all_f = np.concatenate([p.ravel() for p in params.values()])
    all_q = np.concatenate([q[k].ravel() for k in params]) / ONE
    print("\nweight range [%.3f, %.3f], max quantization error %.5f (LSB %.5f)"
          % (all_f.min(), all_f.max(), np.abs(all_f - all_q).max(), 1 / ONE))
    print("test accuracy, float32 model : %.2f%%" % (100 * acc_f))
    print("test accuracy, Q8.8 bit-exact: %.2f%%  (%d / %d)"
          % (100 * acc_q, int(np.sum(pred_te == yte)), len(yte)))
    print("Q8.8 and float agree on %.2f%% of test images" % (100 * agree))
    print("logit range over test set [%.2f, %.2f]"
          % (logits_te.min() / ONE, logits_te.max() / ONE))

    # ---- weight ROM: W1 | b1 | W2 | b2, identical map to the XOR network
    words, comments = [], []
    for j in range(N_HID):
        for i in range(N_IN):
            words.append(q["W1"][j, i]); comments.append("W1[%d][%d]" % (j, i) if i == 0 else "")
    for j in range(N_HID):
        words.append(q["b1"][j]); comments.append("b1[%d]" % j)
    for o in range(N_OUT):
        for j in range(N_HID):
            words.append(q["W2"][o, j]); comments.append("W2[%d][%d]" % (o, j) if j == 0 else "")
    for o in range(N_OUT):
        words.append(q["b2"][o]); comments.append("b2[%d]" % o)
    assert len(words) == N_HID * N_IN + N_HID + N_OUT * N_HID + N_OUT
    os.makedirs(W_DIR, exist_ok=True)
    write_mif(os.path.join(W_DIR, "mnist_weights.mif"), words, comments, "MNIST 196-32-10 weights")
    write_hex(os.path.join(W_DIR, "mnist_weights.hex"), words)
    np.savez(os.path.join(W_DIR, "mnist_weights_float.npz"), **params)

    # ---- demo image ROM for the board
    demo = pick_demo(yte)
    img_words, img_comments = [], []
    for n, i in enumerate(demo):
        img = list(xq_te[i]) + [0] * (IMG_STRIDE - N_IN)
        img_words += img
        img_comments += ["image %d = test[%d], label %d" % (n, i, yte[i])] + [""] * (IMG_STRIDE - 1)
    write_mif(os.path.join(W_DIR, "mnist_images.mif"), img_words, img_comments,
              "%d MNIST 14x14 test digits, %d words each" % (N_DEMO, IMG_STRIDE))
    write_hex(os.path.join(W_DIR, "mnist_images.hex"), img_words)
    with open(os.path.join(W_DIR, "mnist_labels.hex"), "w") as f:
        for i in demo:
            f.write("%X\n" % yte[i])

    print("\ndemo digits on the board (SW[3:0] -> image):")
    print("  SW  test#  label  Q8.8 pred")
    for n, i in enumerate(demo):
        print("  %2d  %5d    %d       %d%s" % (n, i, yte[i], pred_te[i],
                                              "" if pred_te[i] == yte[i] else "   <- wrong"))
    os.makedirs(VEC_DIR, exist_ok=True)
    with open(os.path.join(VEC_DIR, "mnist_demo_pred.hex"), "w") as f:
        for i in demo:
            f.write("%X\n" % pred_te[i])
    print("  %d / %d demo digits correct" % (int(np.sum(pred_te[demo] == yte[demo])), N_DEMO))

    # ---- golden vectors for the RTL testbench: first N_SIM test images,
    #      pixels (196) then logits (10) then pred, one line per image
    with open(os.path.join(VEC_DIR, "mnist_exp.hex"), "w") as f:
        for i in range(N_SIM):
            w = list(xq_te[i]) + list(logits_te[i]) + [pred_te[i]]
            f.write(" ".join("%04X" % to_unsigned(int(v), DATA_W) for v in w))
            f.write("  // test[%d] label %d\n" % (i, yte[i]))
    with open(os.path.join(VEC_DIR, "mnist_labels.hex"), "w") as f:
        for i in range(N_SIM):
            f.write("%X\n" % yte[i])
    n_ok = int(np.sum(pred_te[:N_SIM] == yte[:N_SIM]))
    with open(os.path.join(VEC_DIR, "mnist_vectors.vh"), "w") as f:
        f.write("// generated by python/train_mnist.py\n")
        f.write("`define MNIST_CASES   %d\n" % N_SIM)
        f.write("`define MNIST_STRIDE  %d\n" % (N_IN + N_OUT + 1))
        f.write("`define MNIST_PY_OK   %d  // Python Q8.8 correct on these images\n" % n_ok)
    print("\nwrote weights/mnist_{weights,images}.{mif,hex}, weights/mnist_labels.hex")
    print("wrote %d golden cases to tb/vectors/mnist_exp.hex (Python Q8.8: %d correct)"
          % (N_SIM, n_ok))


if __name__ == "__main__":
    main()
