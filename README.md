# DE2-115 Neural Network Inference Accelerator

A small neural network inference accelerator written in Verilog for the Terasic DE2-115
(Cyclone IV E, EP4CE115F29C7). The flow goes from training in Python to bit-exact fixed-point
verification in simulation, and then to hardware.

| Step | What | Status |
|---|---|---|
| 1 | Train a 2-4-1 XOR MLP in Python, quantize to Q8.8 | done |
| 2 | Export weights (.mif for Quartus, .hex for simulation), build a pipelined MAC unit | done |
| 3 | Self-checking testbench, verified against a bit-exact Python model | done, 455/455 pass |
| 4 | Full forward pass: weight ROM, neuron sequencer, ReLU | next |
| 5 | Run on DE2-115: switches in, LEDs / 7-segment out | |

## Layout

```
python/fixedpoint.py       bit-exact fixed-point model (the golden reference)
python/train_xor.py        train, quantize, check quantized accuracy, export weights
python/gen_mac_vectors.py  generate stimulus + expected results for the testbench
weights/xor_weights.mif    Quartus memory init file (17 x 16-bit words)
weights/xor_weights.hex    same contents for $readmemh
rtl/mac.v                  parameterized 2-stage pipelined multiply-accumulate unit
tb/tb_mac.v                self-checking testbench
tb/vectors/                generated stimulus / expected results
sim/run_modelsim.do        ModelSim script
sim/run_iverilog.sh        Icarus Verilog script (open-source alternative)
quartus/                   timing constraint + virtual-pin script for a standalone fit
```

## Fixed-point format: Q8.8

All weights, biases and activations are **16-bit signed Q8.8**: 8 integer bits (including the sign)
and 8 fractional bits. The range is [-128, +127.996] and the resolution is 1/256 ≈ 0.0039.

Why this format:

- **It fits the hardware.** The Cyclone IV E has 266 embedded 18x18 multipliers. A 16x16 signed
  multiply uses exactly one of them, with no LUT logic.
- **It has range headroom.** The trained XOR weights span [-7.27, +4.35], and the output
  pre-activation reaches -8.48. Q4.4 (8-bit, range ±8) still classifies XOR correctly, but I checked
  and the output logits pin at the rails (+7.94 / -8.00). That leaves no margin. For MNIST, where
  each neuron sums hundreds of products, Q4.4 would overflow in the hidden layers.
- **It has enough resolution.** The largest quantization error on the XOR weights is 0.0018, which is
  under half an LSB. Q4.4 has a 1/16 LSB, which rounds small weights such as 0.05 to zero.
- **It is a common choice** in real edge inference hardware, and it makes the math easy to follow in
  waveforms: 1.0 = 0x0100.

Each arithmetic step keeps full precision until the end:

| Signal | Width | Format | Notes |
|---|---|---|---|
| a, b | 16 | Q8.8 | weight × activation |
| product | 32 | Q16.16 | exact, no rounding |
| accumulator | 40 | Q24.16 | 8 guard bits, so ≥256 worst-case products sum without overflow |
| result | 16 | Q8.8 | round half up, then **saturate** (clamp) instead of wrapping |

Saturation matters because a wrapped overflow turns a large positive activation into a large
negative one, which silently flips classifications.

**Bias** is fed to the MAC as one extra term with b = 1.0 (0x0100), so the MAC needs no bias port.
**Output activation:** training uses a sigmoid, but sigmoid(z) > 0.5 exactly when z > 0, so the
hardware just checks the sign bit of the output logit. ReLU on the hidden layer is a single mux.

## Step 1-2: train and export

Needs Python 3 and numpy (`pip install numpy`). Run from the repo root:

```
python3 python/train_xor.py
```

```
seed 0, final BCE loss 0.00073
weight range [-7.267, 4.346], max quantization error 0.00182 (LSB = 0.00391)

 x1 x2 | float p  | fixed z2 (Q8.8)      | class | expect
  0  0 | 0.0021   |  -1579 ( -6.1680)    |   0   |   0
  0  1 | 0.9997   |   2077 ( +8.1133)    |   1   |   1
  1  0 | 0.9997   |   2077 ( +8.1133)    |   1   |   1
  1  1 | 0.0002   |  -2171 ( -8.4805)    |   0   |   0
quantized network classifies all 4 XOR inputs correctly
```

The "fixed" column comes from the integer model, which is bit-exact with the hardware. It is not a
float approximation.

Weight memory map (one 16-bit word per address), shared by the .mif and the .hex:

| Address | Contents |
|---|---|
| 0-7 | W1[j][i], row-major (hidden neuron j, input i) |
| 8-11 | b1[0..3] |
| 12-15 | W2[0][0..3] |
| 16 | b2[0] |

## The MAC unit (`rtl/mac.v`)

Parameters: `DATA_W` (16), `FRAC_W` (8), `ACC_W` (40).

Pipeline: **stage 1** registers `a*b`, which maps onto the DSP block's output register. **Stage 2**
accumulates. It accepts one term per clock with no bubbles, including between dot products.

| Port | Meaning |
|---|---|
| `in_valid` | a/b carry a real term this cycle (idle cycles are ignored) |
| `in_start` | first term of a new dot product; reloads the accumulator (no clear cycle needed) |
| `in_last` | last term of the dot product |
| `out_done` | one-cycle pulse, 2 clocks after `in_last`, when `acc`/`result` are final |
| `acc` | full-precision 40-bit sum |
| `result` | rounded and saturated Q8.8 |

## Step 3: simulate

```
python3 python/gen_mac_vectors.py      # regenerate vectors (already checked in)
```

This produces 455 dot products (7,331 clock cycles) covering:
- directed arithmetic, including negative values and zero
- rounding edge cases, including exact half-LSB ties for both signs
- saturation at +max and -min
- 256 worst-case terms, which uses all 8 guard bits
- all 20 neuron computations of the trained XOR network
- back-to-back single-term dot products, where start and last are both set every cycle
- 400 random dot products of random length, with random idle cycles and garbage on the bus while
  idle

Python computes the expected `acc` and `result` for each one. The testbench compares them bit for bit
on every `out_done` and also checks that the number of results is correct.

### ModelSim (Intel FPGA Starter Edition)

In the ModelSim transcript window:

```
cd C:/path/to/de2-115-accel/sim
do run_modelsim.do
```

It compiles, loads the waveform, runs, and should print:

```
# TEST PASSED: 455 dot products (7331 cycles) matched the Python model
```

Set `VERBOSE` to 1 (`vsim -gVERBOSE=1 ...`) to print every check. Useful signals are already in the
wave window. Try zooming in on an XOR neuron and reading `result` in decimal: 256 = 1.0.

### Icarus Verilog (optional, no license)

```
sh sim/run_iverilog.sh
```

### Is the testbench actually checking anything?

It was mutation-tested. Each of these deliberate bugs was injected into `mac.v`, and the testbench
failed every time:

| Injected bug | Failures |
|---|---|
| rounding adds 1 LSB instead of ½ LSB | 170 / 455 |
| accumulator subtracts instead of adds | 411 / 455 |
| accumulator ignores `in_valid` | 90 / 455 |
| saturation disabled (wraps on overflow) | 73 / 455 |

## Bringing it into Quartus (DE2-115)

1. **File > New Project Wizard.** Put the project directory in `quartus/`, name it `mac`, and set the
   top-level entity to `mac`.
2. Add `../rtl/mac.v` and `mac.sdc`.
3. Family: **Cyclone IV E**. Device: **EP4CE115F29C7**.
4. **Assignments > Settings > EDA Tool Settings > Simulation:** ModelSim-Altera, Verilog HDL. This is
   optional, but it lets you use Tools > Run Simulation later.
5. The standalone MAC has more than 90 data ports, and none of them are wired to board pins yet.
   Before a full compile, open the Tcl console and run `source mac_virtual_pins.tcl` so they
   become virtual pins.
6. **Processing > Start Compilation.** Then check:
   - **Fitter > Resource Usage Summary:** "Embedded Multiplier 9-bit elements" should be 2, which
     means one 18x18 multiplier with no LUT-based multiply.
   - **TimeQuest > Slow 1200mV 85C Model > Fmax:** this should be comfortably above 50 MHz.
     Record the number, because Fmax and resource usage are good figures for a resume.

In step 5, a real top level (`de2_115_top.v`) will replace the virtual pins. It will map SW[1:0] to
the XOR inputs, the result to LEDG[0], and the logit to HEX0-3, using the pin assignments from the
Terasic DE2-115 System CD.

## Next: step 4

- Instantiate a ROM (`altsyncram`, or an inferred `reg` array with `(* ram_init_file = "xor_weights.mif" *)`)
  initialized from `weights/xor_weights.mif`.
- Add a small FSM that walks the memory map, streams (weight, activation) pairs into the MAC,
  applies ReLU, stores hidden activations, and then runs the output neuron. One MAC takes
  4×3 + 5 = 17 cycles per inference.
- Reuse the same verification approach: Python computes the logits and the testbench checks them.
