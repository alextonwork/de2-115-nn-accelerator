// ---------------------------------------------------------------------------
// tb_mac.v  -  self-checking testbench for rtl/mac.v
//
// Replays tb/vectors/mac_in.hex (one line per clock) into the MAC and compares
// every out_done result against tb/vectors/mac_exp.hex, which was produced by
// the bit-exact Python model (python/fixedpoint.py). Prints PASS/FAIL and a
// summary, then $finish. Works in ModelSim and Icarus Verilog.
//
// VEC_DIR is relative to the simulator's working directory (sim/ by default).
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps
`include "mac_vectors.vh"

module tb_mac;

    parameter DATA_W  = 16;
    parameter FRAC_W  = 8;
    parameter ACC_W   = 40;
    parameter VEC_DIR = "../tb/vectors/";
    parameter VERBOSE = 0;          // 1 = print every check, not just failures

    localparam IN_W  = 36;               // {valid, start, last, pad, a, b}
    localparam EXP_W = ACC_W + DATA_W;   // {acc, result}

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #10 clk = ~clk;               // 50 MHz, same as the DE2-115 CLOCK_50

    reg                      in_valid, in_start, in_last;
    reg  signed [DATA_W-1:0] a, b;
    wire                     out_done;
    wire signed [ACC_W-1:0]  acc;
    wire signed [DATA_W-1:0] result;

    mac #(.DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W)) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_start(in_start), .in_last(in_last),
        .a(a), .b(b),
        .out_done(out_done), .acc(acc), .result(result)
    );

    reg [IN_W-1:0]  vec_in  [0:`N_IN-1];
    reg [EXP_W-1:0] vec_exp [0:`N_EXP-1];

    integer i, n_checked, n_errors;
    reg signed [ACC_W-1:0]  exp_acc;
    reg signed [DATA_W-1:0] exp_res;

    // ---------------- checker: runs on every out_done pulse ----------------
    always @(posedge clk) begin
        if (rst_n && out_done) begin
            if (n_checked >= `N_EXP) begin
                $display("FAIL: unexpected extra out_done at %0t", $time);
                n_errors = n_errors + 1;
            end else begin
                exp_acc = vec_exp[n_checked][EXP_W-1:DATA_W];
                exp_res = vec_exp[n_checked][DATA_W-1:0];
                if (acc !== exp_acc || result !== exp_res) begin
                    $display("FAIL #%0d: acc=%h (exp %h)  result=%h %f (exp %h %f)",
                             n_checked, acc, exp_acc, result, $itor(result) / (1 << FRAC_W),
                             exp_res, $itor(exp_res) / (1 << FRAC_W));
                    n_errors = n_errors + 1;
                end else if (VERBOSE) begin
                    $display("pass #%0d: result=%h (%f)", n_checked, result,
                             $itor(result) / (1 << FRAC_W));
                end
            end
            n_checked = n_checked + 1;
        end
    end

    // ---------------- stimulus ----------------
    initial begin
        $readmemh({VEC_DIR, "mac_in.hex"},  vec_in);
        $readmemh({VEC_DIR, "mac_exp.hex"}, vec_exp);
        n_checked = 0;
        n_errors  = 0;
        in_valid = 0; in_start = 0; in_last = 0; a = 0; b = 0;

        repeat (3) @(posedge clk);
        rst_n <= 1'b1;

        for (i = 0; i < `N_IN; i = i + 1) begin
            @(posedge clk);
            // drive with nonblocking assignments: no race with the DUT sampling
            in_valid <= vec_in[i][35];
            in_start <= vec_in[i][34];
            in_last  <= vec_in[i][33];
            a        <= vec_in[i][31:16];
            b        <= vec_in[i][15:0];
        end
        @(posedge clk);
        in_valid <= 1'b0; in_start <= 1'b0; in_last <= 1'b0;
        repeat (5) @(posedge clk);     // drain the 2-stage pipeline

        if (n_checked != `N_EXP) begin
            $display("FAIL: saw %0d results, expected %0d", n_checked, `N_EXP);
            n_errors = n_errors + 1;
        end
        $display("--------------------------------------------------");
        if (n_errors == 0)
            $display("TEST PASSED: %0d dot products (%0d cycles) matched the Python model",
                     n_checked, `N_IN);
        else
            $display("TEST FAILED: %0d errors in %0d dot products", n_errors, n_checked);
        $display("--------------------------------------------------");
        $finish;
    end

endmodule
