# DE2-115 Neural Network Inference Accelerator

A small neural network inference accelerator written in Verilog for the Terasic DE2-115
(Cyclone IV E, EP4CE115F29C7). The flow goes from training in Python to bit-exact fixed-point
verification in simulation, and then to hardware.

| Step | What | Status |
|---|---|---|
| 1 | Train a 2-4-1 XOR MLP in Python, quantize to Q8.8 | done |
| 2 | Export weights (.mif for Quartus, .hex for simulation), build a pipelined MAC unit | done |
| 3 | Self-checking testbench, verified against a bit-exact Python model | done, 455/455 pass |
| 4 | Full forward pass: .mif-initialized weight ROM, FSM sequencer, ReLU, board top level | done, 32/32 + board test pass |
| 5 | Program the DE2-115 and verify XOR on LEDs / 7-segment | ready to try |

## Layout

```
python/fixedpoint.py       bit-exact fixed-point model (the golden reference)
python/train_xor.py        train, quantize, check quantized accuracy, export weights
python/gen_mac_vectors.py  generate stimulus + expected results for the MAC testbench
python/gen_nn_vectors.py   expected hidden/logit/class for the forward-pass testbench
weights/xor_weights.mif    Quartus memory init file (17 x 16-bit words)
weights/xor_weights.hex    same contents for $readmemh
rtl/mac.v                  parameterized 2-stage pipelined multiply-accumulate unit
rtl/weight_rom.v           M9K ROM, initialized from the .mif in Quartus (.hex in simulation)
rtl/nn_core.v              2-layer MLP forward pass: FSM + ROM + one MAC
rtl/hex7seg.v              7-segment decoder
rtl/de2_115_top.v          board top level: switches in, LEDs + 7-segment out
tb/tb_mac.v                self-checking MAC testbench
tb/tb_nn_core.v            self-checking forward-pass testbench
tb/tb_top.v                board-level test (switches -> LEDs / HEX digits)
tb/vectors/                generated stimulus / expected results
sim/run_modelsim.do        ModelSim script (takes the testbench name)
sim/run_iverilog.sh        runs all testbenches with Icarus Verilog
quartus/de2_115_top.*      ready-made Quartus project for the board (device, files, pins, timing)
quartus/mac.sdc, mac_virtual_pins.tcl   for fitting the MAC on its own
.github/workflows/sim.yml  CI: regenerates vectors and runs every testbench on each push
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
cd C:/path/to/de2-115-nn-accelerator/sim
do run_modelsim.do                 # MAC unit
do run_modelsim.do tb_nn_core      # full forward pass (FSM, ROM, MAC in the wave window)
do run_modelsim.do tb_top          # board top level
```

It compiles, loads the waveform, runs, and should print:

```
# TEST PASSED: 455 dot products (7331 cycles) matched the Python model
```

Set `VERBOSE` to 1 (`vsim -gVERBOSE=1 ...`) to print every check. Useful signals are already in the
wave window. Try zooming in on an XOR neuron and reading `result` in decimal: 256 = 1.0.

### Icarus Verilog (optional, no license)

```
sh sim/run_iverilog.sh      # runs all three testbenches
```

GitHub Actions runs the same script on every push. It also regenerates the test vectors from the
committed weights and fails if they drift.

### Is the testbench actually checking anything?

It was mutation-tested. Each of these deliberate bugs was injected into `mac.v`, and the testbench
failed every time:

| Injected bug | Failures |
|---|---|
| rounding adds 1 LSB instead of ½ LSB | 170 / 455 |
| accumulator subtracts instead of adds | 411 / 455 |
| accumulator ignores `in_valid` | 90 / 455 |
| saturation disabled (wraps on overflow) | 73 / 455 |

## Step 4: full forward pass (`rtl/nn_core.v`)

```
          +-----------+  weight   +-------+  result  +---------------------+
 addr --> | weight_rom|---------->|       |--------->| ReLU -> hidden[0:3] |---+
  ^       | (M9K,.mif)|           |  MAC  |          | logit[0]            |   |
  |       +-----------+  act      |       |          +---------------------+   |
 FSM ------------------------->   +-------+                                    |
 (layer, neuron, k)   x[i] / hidden[j] / 1.0 (bias)  <-------------------------+
```

A single MAC is shared by every neuron. The FSM walks the weight memory in the same order as the
memory map above. It streams one (weight, activation) pair per clock, puts neurons back to back,
and makes the bias the last term of each neuron. The ROM has one cycle of read latency, so the
activation and control bits are registered once to stay aligned with the weight. Results come
out of the MAC in the order they were issued. ReLU is applied as each hidden result is captured.
The output layer starts once all four hidden activations are back.

- **Latency:** 17 MAC terms, 27 clocks from `start` to `done`, which is 0.54 µs at 50 MHz. The
  pipeline drains once between layers.
- **Resources:** one 18x18 multiplier, one M9K block, and a few hundred LEs. Quartus reports the
  exact numbers.
- **Scales to MNIST:** `N_IN`, `N_HID` and `N_OUT` are parameters. With `N_OUT > 1`, `pred`
  becomes the argmax of the logits.

**Weight loading.** `weight_rom.v` has the attribute
`(* ram_init_file = "../weights/xor_weights.mif" *)`, so Quartus bakes the .mif into the M9K
block in the .sof. Simulation loads the matching .hex with `$readmemh`, and that part is inside
`translate_off` so it never reaches synthesis. You can change the weights without recompiling:
edit the .mif, then run **Processing > Update Memory Initialization File** and
**Assembler**.

**Verification.** `python/gen_nn_vectors.py` reads the weights back from `xor_weights.hex`,
which is exactly what the ROM holds. It computes the expected hidden activations, logit and class
for the 4 XOR inputs plus 28 random real-valued inputs, which exercise ReLU clamping. The
testbench checks all of them bit for bit:

```
x=(0,0)  logit=-6.167969  pred=0  (27 cycles)
x=(0,1)  logit=8.113281  pred=1  (27 cycles)
x=(1,0)  logit=8.113281  pred=1  (27 cycles)
x=(1,1)  logit=-8.480469  pred=0  (27 cycles)
TEST PASSED: 32 inferences matched the Python model, 27 cycles each
```

Mutation check: removing ReLU, reading the wrong bias address, or feeding 0 instead of 1.0 for the
bias each causes 80+ failures.

## Step 5: on the board

`rtl/de2_115_top.v` wires the core to the board. The core runs inference continuously, so the
display follows the switches right away.

| Board | Meaning |
|---|---|
| KEY0 | reset |
| SW1, SW0 | inputs x[1], x[0] (up = 1.0) |
| LEDR1, LEDR0 | echo the switches |
| **LEDG0** | predicted class: should light for 01 and 10 only |
| HEX0 | predicted class as a digit |
| HEX6 | `-` when the logit is negative |
| HEX5-HEX2 | the logit's magnitude in Q8.8 hex: `06.2B` means 0x062B / 256 = 6.168 |

What you should see, and what `tb/tb_top.v` checks in simulation:

| SW1 SW0 | LEDG0 | HEX6..HEX2 | logit |
|---|---|---|---|
| 0 0 | off | `- 06.2B` | -6.168 |
| 0 1 | on | `  08.1D` | +8.113 |
| 1 0 | on | `  08.1D` | +8.113 |
| 1 1 | off | `- 08.7B` | -8.480 |

### Build and program

1. In Quartus, **File > Open Project** and open `quartus/de2_115_top.qpf`. The device, source files,
   .mif, timing constraints and pins are already set.
2. **Processing > Start Compilation.**
3. Connect the board's USB-Blaster port, the one labeled BLASTER, and power it on. Then open
   **Tools > Programmer**. Select USB-Blaster under Hardware Setup, add
   `output_files/de2_115_top.sof`, check Program/Configure, and click Start.
4. Flip SW0 and SW1 and compare against the table above.

**Pins:** `de2_115_top.qsf` contains pin locations taken from the DE2-115 User Manual tables. The
port names match Terasic's golden top. If an LED or digit behaves oddly, import the official
`DE2_115.qsf` from the Terasic DE2-115 System CD (**Assignments > Import Assignments**). That file
is authoritative and also sets the I/O standards.

**Numbers for your resume**, from the compilation report:
- **Fitter > Resource Usage Summary:** total logic elements, "Embedded Multiplier 9-bit elements"
  (expect 2), and memory bits (expect one M9K).
- **TimeQuest > Slow 1200mV 85C Model > Fmax Summary:** the Fmax of CLOCK_50.
- **Throughput:** Fmax / 27 inferences per second.

## Fitting the MAC on its own (optional)

This is useful for measuring the Fmax of the MAC alone.

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

## Ideas after XOR

- **MNIST:** for example 784-32-10 with inputs downsampled to 14x14. `nn_core` already
  parameterizes the layer sizes and argmax. The weights (about 25k words) fit in M9K blocks. The
  image could come from a ROM of test digits selected by the switches.
- **Parallelism:** use N MACs, one per hidden neuron, reading N weights per clock from a wider
  ROM. This is the classic latency and area trade-off to measure and write up.
- **8-bit weights with a per-layer scale factor:** this is closer to how real INT8 accelerators work.
