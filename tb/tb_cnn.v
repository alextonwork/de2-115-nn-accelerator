// ---------------------------------------------------------------------------
// tb_cnn.v  -  self-checking testbench for cnn_core (conv + pool + dense)
//
// Runs the first N_CASES MNIST test images (pixels taken from mnist_exp.hex,
// the same Q8.8 words the MLP testbenches use) and checks all 10 logits and
// the argmax bit-for-bit against python/train_cnn.py (cnn_c16_exp.hex).
// Also counts correct predictions against the true labels and measures
// latency. The testbench plays the external image memory: one-clock
// synchronous read, like an M9K.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps
`include "mnist_vectors.vh"
`include "cnn_vectors.vh"

module tb_cnn;

    parameter DATA_W  = 16;
    parameter N_CASES = `CNN_CASES;       // override (-P tb_cnn.N_CASES=50) for a quick run
    parameter VEC_DIR = "../tb/vectors/";
    parameter TIMEOUT = 20000;

    localparam N_IN   = 196;
    localparam N_OUT  = 10;
    localparam STRIDE = `MNIST_STRIDE;    // pixels, MLP logits, MLP pred
    localparam ESTR   = `CNN_STRIDE;      // CNN logits, CNN pred

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #10 clk = ~clk;                // 50 MHz

    reg                     start = 1'b0;
    wire                    busy, done;
    wire [7:0]              x_addr;
    reg  [DATA_W-1:0]       x_data;
    wire [N_OUT*DATA_W-1:0] logits_flat;
    wire [3:0]              pred;

    cnn_core dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .x_addr(x_addr), .x_data(x_data),
        .busy(busy), .done(done),
        .logits_flat(logits_flat), .pred(pred)
    );

    reg [DATA_W-1:0] vec   [0:`MNIST_CASES*STRIDE-1];
    reg [DATA_W-1:0] exp_v [0:`CNN_CASES*ESTR-1];
    reg [3:0]        label [0:`CNN_CASES-1];

    integer c, j, base, cycles, min_cyc, max_cyc, n_errors, n_correct;

    always @(posedge clk)
        x_data <= vec[base + x_addr];

    initial begin
        $readmemh({VEC_DIR, "mnist_exp.hex"}, vec);
        $readmemh({VEC_DIR, "cnn_c16_exp.hex"}, exp_v);
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
                if (logits_flat[j*DATA_W +: DATA_W] !== exp_v[c*ESTR + j]) begin
                    if (n_errors < 20)
                        $display("FAIL image %0d logit[%0d]: got %h expected %h",
                                 c, j, logits_flat[j*DATA_W +: DATA_W], exp_v[c*ESTR + j]);
                    n_errors = n_errors + 1;
                end
            if ({12'b0, pred} !== exp_v[c*ESTR + N_OUT]) begin
                if (n_errors < 20)
                    $display("FAIL image %0d pred: got %0d expected %0d", c, pred, exp_v[c*ESTR + N_OUT]);
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
        $display("CNN latency: %0d cycles = %.2f us at 50 MHz", max_cyc, max_cyc * 0.02);
        if (N_CASES == `CNN_CASES && n_correct != `CNN_PY_OK) begin
            $display("FAIL: Python counted %0d correct", `CNN_PY_OK);
            n_errors = n_errors + 1;
        end
        if (min_cyc != max_cyc) begin
            $display("FAIL: latency varies (%0d..%0d)", min_cyc, max_cyc);
            n_errors = n_errors + 1;
        end
        if (n_errors == 0)
            $display("TEST PASSED: %0d CNN inferences matched the Python model bit-for-bit", N_CASES);
        else
            $display("TEST FAILED: %0d errors", n_errors);
        $display("--------------------------------------------------");
        $finish;
    end

endmodule
