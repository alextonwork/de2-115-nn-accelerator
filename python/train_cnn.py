"""CNN stage: train a small conv + max-pool network on 14x14 MNIST, Q8.8 golden model.

Network (defaults):
    x 14x14x1 -> Conv 3x3, C=16, valid -> 12x12x16 -> ReLU -> MaxPool 2x2 -> 6x6x16
              -> flatten (576) -> Dense(10) -> argmax
    --hidden H inserts Dense(H) + ReLU before the output layer.

Same input as the MLP stage (python/train_mnist.py): 28x28 digits average-pooled
to 14x14, [0, 1], Q8.8. So the board's image ROM and demo digits carry over.

Fixed-point model (bit-exact target for the RTL):
  * every conv output and dense neuron is one MAC dot product: the taps, then
    the bias as a last term with x = 1.0, accumulated in Q16.16 and rounded
    half up + saturated to Q8.8 (fixedpoint.mac_result / train_mnist.mac_layer)
  * ReLU then 2x2 max on Q8.8 integers (max and ReLU commute, so the RTL may
    pool first and clamp once)
  * flatten order is k = pos*C + ch, pos = pr*6 + pc: channel-minor, because
    the C conv lanes produce all channels of one position together

Run from the repo root:  python3 python/train_cnn.py [--channels 16] [--hidden 0] [--export]
It trains, reports float and Q8.8 accuracy and the op/parameter counts used for
the hardware plan, and with --export (no hidden layer) writes the ROMs and
golden vectors for rtl/cnn_core.v:
  weights/cnn_c{C}_conv.hex       K*K+1 rows of C words (taps row-major, then
                                  bias), lane 0 in the low 16 bits: the
                                  wide ROM rtl/conv_pool.v reads with a counter
  weights/cnn_c{C}_dense.hex      36*C+1 rows of 10 words: row k holds
                                  W[o][k] for output o in lane o, last row
                                  the biases (read with a counter, like above)
  weights/cnn_c{C}_float.npz      float weights, for reference
  tb/vectors/cnn_c{C}_pool_exp.hex  pooled+ReLU feature map of the 16 board
                                  demo digits, 36 rows of C words per image
  tb/vectors/cnn_c{C}_demo_pred.hex  its predictions for those 16 digits
  tb/vectors/cnn_c{C}_exp.hex     10 logits + pred for the first 1000 test
                                  images (pixels come from mnist_exp.hex)
  tb/vectors/cnn_vectors.vh       case count and Python's correct count
"""
import argparse
import os

import numpy as np

from fixedpoint import ONE
from train_mnist import (FILES, ROOT, W_DIR, VEC_DIR, load_images, load_labels,
                         downsample, quantize, mac_layer, pick_demo, N_SIM)

H = W = 14
K = 3
P = 2


# ---------------------------------------------------------------- shapes
def conv_out():
    return H - K + 1                          # 12


def pool_out():
    return conv_out() // P                    # 6


def patches(x):
    """(B, 196) -> (B, 144, 9) im2col, positions row-major, taps (ky, kx) row-major."""
    x = x.reshape(-1, H, W)
    n = conv_out()
    cols = [x[:, ky:ky + n, kx:kx + n] for ky in range(K) for kx in range(K)]
    return np.stack(cols, axis=-1).reshape(len(x), n * n, K * K)


def maxpool(a):
    """(B, 12, 12, C) -> (B, 6, 6, C) and the one-hot argmax mask for backprop."""
    b, n, _, c = a.shape
    m = n // P
    win = a.reshape(b, m, P, m, P, c).transpose(0, 1, 3, 5, 2, 4).reshape(b, m, m, c, P * P)
    idx = win.argmax(-1)                      # first max wins, like a strict '>' scan
    out = np.take_along_axis(win, idx[..., None], -1)[..., 0]
    return out, idx


def unpool(g, idx):
    b, m, _, c = g.shape
    win = np.zeros((b, m, m, c, P * P))
    np.put_along_axis(win, idx[..., None], g[..., None], -1)
    return win.reshape(b, m, m, c, P, P).transpose(0, 1, 4, 2, 5, 3).reshape(b, m * P, m * P, c)


# ---------------------------------------------------------------- float model
def init(c, hidden, rng):
    n_flat = pool_out() ** 2 * c
    p = {"Wc": rng.normal(0, np.sqrt(2 / (K * K)), (c, K * K)), "bc": np.zeros(c)}
    if hidden:
        p["W1"] = rng.normal(0, np.sqrt(2 / n_flat), (hidden, n_flat)); p["b1"] = np.zeros(hidden)
        p["W2"] = rng.normal(0, np.sqrt(2 / hidden), (10, hidden)); p["b2"] = np.zeros(10)
    else:
        p["W2"] = rng.normal(0, np.sqrt(2 / n_flat), (10, n_flat)); p["b2"] = np.zeros(10)
    return p


def forward(p, x, cache=False):
    c = len(p["bc"])
    pt = patches(x)
    z = pt @ p["Wc"].T + p["bc"]                                  # (B, 144, C)
    a = np.maximum(z, 0).reshape(-1, conv_out(), conv_out(), c)
    pooled, idx = maxpool(a)
    f = pooled.reshape(len(x), -1)                                # k = pos*C + ch
    if "W1" in p:
        z1 = f @ p["W1"].T + p["b1"]
        h = np.maximum(z1, 0)
    else:
        z1 = h = f
    out = h @ p["W2"].T + p["b2"]
    if cache:
        return out, (pt, z, idx, f, z1, h)
    return out


def train(p, xtr, ytr, xva, yva, epochs, batch=128, lr=2e-3, wd=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(a) for k, a in p.items()}
    t = 0
    c = len(p["bc"])
    for ep in range(epochs):
        order = rng.permutation(len(xtr))
        for s in range(0, len(xtr), batch):
            ix = order[s:s + batch]
            x, y = xtr[ix], ytr[ix]
            out, (pt, z, idx, f, z1, h) = forward(p, x, cache=True)
            out -= out.max(1, keepdims=True)
            pr = np.exp(out); pr /= pr.sum(1, keepdims=True)
            pr[np.arange(len(y)), y] -= 1
            d = pr / len(y)
            g = {"W2": d.T @ h, "b2": d.sum(0)}
            dh = d @ p["W2"]
            if "W1" in p:
                dz1 = dh * (z1 > 0)
                g["W1"] = dz1.T @ f; g["b1"] = dz1.sum(0)
                df = dz1 @ p["W1"]
            else:
                df = dh
            da = unpool(df.reshape(len(x), pool_out(), pool_out(), c), idx)
            dz = da.reshape(len(x), -1, c) * (z > 0)
            g["Wc"] = np.einsum("bpc,bpk->ck", dz, pt); g["bc"] = dz.sum((0, 1))
            t += 1
            for k in p:
                gk = g[k] + (wd * p[k] if k.startswith("W") else 0)
                m[k] = 0.9 * m[k] + 0.1 * gk
                v[k] = 0.999 * v[k] + 0.001 * gk ** 2
                p[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        print("  epoch %2d  val acc %.2f%%"
              % (ep + 1, 100 * np.mean(forward(p, xva).argmax(1) == yva)))
    return p


# ---------------------------------------------------------------- Q8.8 golden model
def fixed_forward(q, xq):
    """Bit-exact integer forward pass. Returns (conv, pooled, flat, logits, pred)."""
    c = len(q["bc"])
    pt = patches(xq)                                              # (B, 144, 9)
    conv = mac_layer(pt.reshape(-1, K * K), q["Wc"], q["bc"]).reshape(len(xq), conv_out(), conv_out(), c)
    pooled, _ = maxpool(np.maximum(conv, 0))
    flat = pooled.reshape(len(xq), -1)
    h = np.maximum(mac_layer(flat, q["W1"], q["b1"]), 0) if "W1" in q else flat
    logits = mac_layer(h, q["W2"], q["b2"])
    return conv, pooled, flat, logits, logits.argmax(1)


def report_cost(p):
    c = len(p["bc"])
    n_flat = pool_out() ** 2 * c
    convs = conv_out() ** 2 * c * (K * K + 1)
    dense = sum(p[k].size + p["b" + k[1:]].size for k in ("W1", "W2") if k in p)
    params = sum(a.size for a in p.values())
    print("\nparams %d (%d words of ROM), MAC terms per image: conv %d + dense %d = %d"
          % (params, params, convs, dense, convs + dense))
    print("feature map after pool: %dx%dx%d = %d values" % (pool_out(), pool_out(), c, n_flat))
    # cycle estimate: conv with C lanes (one per channel, pixel broadcast) does
    # one position (K*K taps + bias) per K*K+1 clocks; dense as nn_core_par
    conv_cyc = conv_out() ** 2 * (K * K + 1)
    dense_cyc = 0
    fan_in = n_flat
    for k in ("W1", "W2"):
        if k in p:
            dense_cyc += -(-p[k].shape[0] // c) * (fan_in + 1)
            fan_in = p[k].shape[0]
    print("estimated cycles with N=C=%d lanes: conv %d + dense %d = %d (MLP stage: 6644 @ N=1, 865 @ N=8)"
          % (c, conv_cyc, dense_cyc, conv_cyc + dense_cyc))


def write_rows(path, rows):
    """One row per line, lane 0 in the lowest 16 bits (same as gen_par_weights.py)."""
    with open(path, "w") as f:
        for row in rows:
            f.write("".join("%04x" % (int(v) & 0xFFFF) for v in reversed(list(row))) + "\n")


def export(p, q, xq_te, yte):
    c = len(q["bc"])
    rows = [q["Wc"][:, t] for t in range(K * K)] + [q["bc"]]
    write_rows(os.path.join(W_DIR, "cnn_c%d_conv.hex" % c), rows)
    np.savez(os.path.join(W_DIR, "cnn_c%d_float.npz" % c), **p)
    demo = pick_demo(yte)
    _, pooled, _, _, demo_pred = fixed_forward(q, xq_te[demo])
    write_rows(os.path.join(VEC_DIR, "cnn_c%d_pool_exp.hex" % c), pooled.reshape(-1, c))
    with open(os.path.join(VEC_DIR, "cnn_c%d_demo_pred.hex" % c), "w") as f:
        for v in demo_pred:
            f.write("%X\n" % v)
    print("demo digits: CNN predicts %d / %d correctly" % (int(np.sum(demo_pred == yte[demo])), len(demo)))
    assert "W1" not in q, "export supports conv -> pool -> dense(10) only"
    rows = [q["W2"][:, k] for k in range(q["W2"].shape[1])] + [q["b2"]]
    write_rows(os.path.join(W_DIR, "cnn_c%d_dense.hex" % c), rows)
    _, _, _, logits, pred = fixed_forward(q, xq_te[:N_SIM])
    with open(os.path.join(VEC_DIR, "cnn_c%d_exp.hex" % c), "w") as f:
        for i in range(N_SIM):
            f.write(" ".join("%04X" % (int(v) & 0xFFFF) for v in list(logits[i]) + [pred[i]]))
            f.write("  // test[%d] label %d\n" % (i, yte[i]))
    n_ok = int(np.sum(pred == yte[:N_SIM]))
    with open(os.path.join(VEC_DIR, "cnn_vectors.vh"), "w") as f:
        f.write("// generated by python/train_cnn.py\n")
        f.write("`define CNN_CASES   %d\n" % N_SIM)
        f.write("`define CNN_STRIDE  11  // 10 logits, pred\n")
        f.write("`define CNN_PY_OK   %d  // Python Q8.8 correct on these images\n" % n_ok)
    print("wrote weights/cnn_c%d_{conv,dense}.hex, tb/vectors/cnn_c%d_{pool_exp,exp}.hex"
          " (%d demo digits, %d test images, Python correct %d)" % (c, c, len(demo), N_SIM, n_ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    xtr = downsample(load_images(FILES["train_x"])); ytr = load_labels(FILES["train_y"])
    xte = downsample(load_images(FILES["test_x"])); yte = load_labels(FILES["test_y"])
    xva, yva, xtr, ytr = xtr[55000:], ytr[55000:], xtr[:55000], ytr[:55000]
    print("conv 3x3x%d -> maxpool 2 -> %s10" % (args.channels, "%d -> " % args.hidden if args.hidden else ""))

    p = train(init(args.channels, args.hidden, np.random.default_rng(0)), xtr, ytr, xva, yva, args.epochs)
    acc_f = np.mean(forward(p, xte).argmax(1) == yte)
    q = {k: quantize(v) for k, v in p.items()}
    xq_te = quantize(xte)
    conv, _, _, logits, pred = fixed_forward(q, xq_te)
    acc_q = np.mean(pred == yte)
    print("test accuracy, float : %.2f%%" % (100 * acc_f))
    print("test accuracy, Q8.8  : %.2f%%  (%d / %d)" % (100 * acc_q, int(np.sum(pred == yte)), len(yte)))
    print("conv output range [%.2f, %.2f], logit range [%.2f, %.2f]"
          % (conv.min() / ONE, conv.max() / ONE, logits.min() / ONE, logits.max() / ONE))
    report_cost(p)
    if args.export:
        export(p, q, xq_te, yte)


if __name__ == "__main__":
    main()
