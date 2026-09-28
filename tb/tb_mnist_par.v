// ---------------------------------------------------------------------------
// tb_mnist_par.v  -  self-checking testbench for nn_core_par (N_MAC lanes)
//
// Same checks as tb_mnist.v, against the same golden vectors, but on the
// parallel core. Run it once per lane count: -P tb_mnist_par.N_MAC=8 (the
// weight file follows N_MAC). Every N must give the same logits bit for bit,
// since the wide ROMs hold the same Q8.8 words, only reordered. The testbench plays the external input
// memory: one-clock synchronous read of the current image's pixels.
//
// Checks all 10 logits and the argmax bit-for-bit against Python, counts how
// many predictions match the true labels (hardware accuracy), and measures
// latency in clock cycles.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps
`include "mnist_vectors.vh"

module tb_mnist_par;

    parameter DATA_W   = 16;
    parameter N_IN     = 196;
    parameter N_HID    = 32;
    parameter N_OUT    = 10;
    parameter N_CASES  = `MNIST_CASES;     // override (e.g. -P tb_mnist.N_CASES=50) for a quick run
    parameter VEC_DIR  = "../tb/vectors/";
    parameter N_MAC    = 8;
    parameter HEX_FILE = "";              // default: ../weights/mnist_weights_par<NN>.hex
    parameter TIMEOUT  = 20000;

    localparam STRIDE = `MNIST_STRIDE;     // pixels, logits, pred

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #10 clk = ~clk;          // 50 MHz

    reg                        start = 1'b0;
    wire                       busy, done;
    wire [7:0]                 x_addr;
    reg  [DATA_W-1:0]          x_data;
    wire [N_HID*DATA_W-1:0]    hidden_flat;
    wire [N_OUT*DATA_W-1:0]    logits_flat;
    wire [3:0]                 pred;

    localparam WFILE = (HEX_FILE != "") ? HEX_FILE :
                       (N_MAC ==  1) ? "../weights/mnist_weights_par01.hex" :
                       (N_MAC ==  2) ? "../weights/mnist_weights_par02.hex" :
                       (N_MAC ==  4) ? "../weights/mnist_weights_par04.hex" :
                       (N_MAC ==  8) ? "../weights/mnist_weights_par08.hex" :
                       (N_MAC == 16) ? "../weights/mnist_weights_par16.hex" :
                                       "../weights/mnist_weights_par32.hex";

    nn_core_par #(.N_IN(N_IN), .N_HID(N_HID), .N_OUT(N_OUT), .N_MAC(N_MAC), .HEX_FILE(WFILE)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .x_addr(x_addr), .x_data(x_data),
        .busy(busy), .done(done),
        .hidden_flat(hidden_flat), .logits_flat(logits_flat), .pred(pred)
    );

    reg [DATA_W-1:0] vec   [0:`MNIST_CASES*STRIDE-1];
    reg [3:0]        label [0:`MNIST_CASES-1];

    integer c, j, base, cycles, min_cyc, max_cyc, n_errors, n_correct;

    // external input memory model: synchronous read, like an M9K
    always @(posedge clk)
        x_data <= vec[base + x_addr];

    initial begin
        $readmemh({VEC_DIR, "mnist_exp.hex"}, vec);
        $readmemh({VEC_DIR, "mnist_labels.hex"}, label);
        n_errors = 0; n_correct = 0; base = 0;
        min_cyc = TIMEOUT; max_cyc = 0;
        repeat (3) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);

        for (c = 0; c < N_CASES; c = c + 1) begin
            base = c * STRIDE;
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
            cycles = 1;
            while (!done && cycles < TIMEOUT) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
            if (!done) begin
                $display("FAIL image %0d: no done after %0d cycles", c, TIMEOUT);
                n_errors = n_errors + 1;
            end
            #1;
            if (cycles < min_cyc) min_cyc = cycles;
            if (cycles > max_cyc) max_cyc = cycles;

            for (j = 0; j < N_OUT; j = j + 1)
                if (logits_flat[j*DATA_W +: DATA_W] !== vec[base + N_IN + j]) begin
                    $display("FAIL image %0d logit[%0d]: got %h expected %h",
                             c, j, logits_flat[j*DATA_W +: DATA_W], vec[base + N_IN + j]);
                    n_errors = n_errors + 1;
                end
            if ({12'b0, pred} !== vec[base + N_IN + N_OUT]) begin
                $display("FAIL image %0d pred: got %0d expected %0d",
                         c, pred, vec[base + N_IN + N_OUT]);
                n_errors = n_errors + 1;
            end
            if (pred == label[c]) n_correct = n_correct + 1;
            if (c < 8)
                $display("image %0d: label %0d  pred %0d  (%0d cycles)", c, label[c], pred, cycles);
            @(posedge clk);
        end

        $display("--------------------------------------------------");
        $display("RTL accuracy: %0d / %0d correct (%.1f%%)",
                 n_correct, N_CASES, 100.0 * n_correct / N_CASES);
        $display("N_MAC = %0d  latency: %0d cycles = %.2f us at 50 MHz", N_MAC, max_cyc, max_cyc * 0.02);
        if (N_CASES == `MNIST_CASES && n_correct != `MNIST_PY_OK) begin
            $display("FAIL: Python counted %0d correct", `MNIST_PY_OK);
            n_errors = n_errors + 1;
        end
        if (min_cyc != max_cyc) begin
            $display("FAIL: latency varies (%0d..%0d)", min_cyc, max_cyc);
            n_errors = n_errors + 1;
        end
        if (n_errors == 0)
            $display("TEST PASSED: N_MAC = %0d, %0d MNIST inferences matched the Python model bit-for-bit",
                     N_MAC, N_CASES);
        else
            $display("TEST FAILED: %0d errors", n_errors);
        $display("--------------------------------------------------");
        $finish;
    end

endmodule
