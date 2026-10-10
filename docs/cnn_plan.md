# Step 8: conv + pooling (toward a small CNN)

Goal: replace the 196-32-10 MLP with a small CNN on the same 14x14 Q8.8 input,
reusing `mac.v`, `weight_rom.v` and the broadcast-activation idea from
`nn_core_par.v`, then compare MLP and CNN on accuracy, weights, latency and area.

## Network candidates (python/train_cnn.py, 8 epochs unless noted)

| Network | Weights | Float | Q8.8 bit-exact | Est. cycles |
|---|---:|---:|---:|---:|
| MLP 196-32-10 (step 6/7, current) | 6634 | 96.17% | 96.16% | 6644 @ 1 MAC, 865 @ 8, 249 @ 32 |
| conv 3x3x4 -> pool -> 10 | 1490 | 95.71% | 95.72% | ~1880 @ 4 lanes |
| **conv 3x3x8 -> pool -> 10** | 2970 | 97.20% | **97.18%** | ~1750 @ 8 lanes |
| conv 3x3x8 -> pool -> 32 -> 10 | 9658 | 97.77% | 97.77% | ~2660 @ 8 lanes |
| conv 3x3x16 -> pool -> 10 (10 epochs) | 5930 | 98.04% | **98.02%** | ~1750 @ 16 lanes |

Q8.8 costs at most 0.02 points for every candidate. The 16-channel net beats the
MLP by ~1.9 points with fewer weights, at the same latency as the 8-channel one
(the conv layer, not the channel count, sets the latency; see below).

Decision: C = 16 is the shipped network (brought up at C = 8 first; C stays a parameter).
In RTL it gets 973/1000 on the first 1000 test images, bit-exact with Python, in
**2037 clocks (40.7 us at 50 MHz)**: 3.3x faster than the single-MAC MLP and 1.9 points
more accurate, with 26 MACs instead of 1.

## Network

```
x 14x14x1 -> conv 3x3, C filters, valid -> 12x12xC -> ReLU -> maxpool 2x2 -> 6x6xC
          -> flatten (36*C, index k = pos*C + ch) -> Dense(10) -> argmax
```

Every conv output and dense neuron is one MAC dot product (taps, then bias with x = 1.0),
rounded half up and saturated to Q8.8, exactly like the MLP. ReLU and max commute, so the
RTL pools first and clamps once.

## Hardware

```
image ROM --x_addr/x_data--> conv_pool (C lanes) --fm write--> feature RAM (36 x C words)
                                                                     |
                                     dense (10 lanes) <--------------+  -> logits -> argmax
                      \____________________ rtl/cnn_core.v ____________________/
```

- `rtl/conv_pool.v` (done): lane l = output channel l. Each clock one pixel is broadcast to
  all lanes and each lane reads its tap weight from a 10-row wide ROM (9 taps + bias), so
  the ROM address is just the tap counter. Loop order is pool window -> 4 conv positions ->
  10 terms, so the 4 results of a window leave the MACs back to back and are max-reduced on
  the way out. One feature-map row (all C channels) is written per window.
  36 x 4 x 10 = **1440 clocks + 5 of pipeline = 1445**, no bubbles.
- Feature RAM (in `rtl/cnn_core.v`): 36 rows x C words (9216 bits at C = 16; the 256-bit
  row needs 8 M9Ks side by side). Written a row at a time by `conv_pool`, read one activation
  per clock by the dense layer (row k / C, channel k mod C).
- Dense layer (in `rtl/cnn_core.v`): 10 lanes, one group, 36*C + 1 terms (577 clocks at
  C = 16) from a 577-row wide ROM read with the term counter, then the one-logit-per-clock
  argmax scan from `nn_core_par`. Total **2037 clocks** at C = 16, measured by `tb_cnn`.
- Multipliers: C (conv) + 10 (dense) MACs, 26 MACs = 52 9-bit multipliers at C = 16,
  well inside the EP4CE115's 532. Weight ROM: 10 rows of C words (conv) plus
  10 x (36C + 1) words (dense).

### Why the conv layer is the bottleneck

The image is read one pixel per clock and broadcast, so conv costs 144 positions x 10 terms
whatever C is. More channels are free in time and cost only multipliers. The next speedup is
spatial: P conv positions in parallel (for example the 4 positions of a pool window) with a
3-row line buffer or a banked image RAM feeding P pixels per clock. That gives
1440 / P conv clocks and a second speed-vs-area sweep (P = 1, 2, 4) to put next to the
N_MAC sweep from step 7.

## Status and next steps

1. Done: `python/train_cnn.py` (numpy training + bit-exact Q8.8 golden model),
   `rtl/conv_pool.v`, `tb/tb_conv_pool.v` (16 board demo digits x 36 windows bit-exact,
   1445 clocks; fails on mutations of the max compare, the ReLU and the row addressing).
2. Done: C = 16 export, `rtl/cnn_core.v` (feature RAM + dense + argmax), `tb/tb_cnn.v`:
   1000/1000 test images bit-exact, 973 correct, 2037 clocks. CI runs 200 images to stay
   quick. Mutations of the channel select, the bias input and the feature-RAM row address
   all fail it.
3. Done in sim: `de2_115_mnist_top` has `USE_CNN = 1`. Switches, HEX and LEDs work the same,
   and HEX7-4 reads `2037`. The CNN gets all 16 stored digits right, including the messy 5
   (test #8) that the MLP calls a 6. `tb_mnist_top -P USE_CNN=1` checks it.
   Quartus project: `quartus/de2_115_cnn.qpf`. **Needs a board run** to measure LEs, M9Ks,
   multipliers and Fmax.
4. README: MLP vs CNN table (accuracy, weights, cycles, resources).
5. Stretch: spatial parallelism (P positions per clock) and its sweep.

