// ---------------------------------------------------------------------------
// tb_mnist_top.v  -  board-level test for de2_115_mnist_top
//
// Lets the core sweep all 16 stored digits, then flips SW[3:0] through every
// image and checks what a person would see: HEX0 = the Python Q8.8 prediction,
// HEX2 = the true label, LEDG[0]/LEDR[17] = right/wrong, LEDR[15:0] = the
// per-image hit lamps, and HEX7-4 = latency (SW[17] down) / correct count (up).
// Run with -P tb_mnist_top.N_MAC=<n> for each lane count.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_mnist_top;

    parameter N_MAC = 32;

    // latency formula from rtl/nn_core_par.v, for MNIST 196-32-10
    localparam G1      = 32 / N_MAC;
    localparam G2      = (10 + N_MAC - 1) / N_MAC;
    localparam EXP_LAT = G1*197 + G2*33 + 9 + (10 - (G2-1)*N_MAC);

    reg CLOCK_50 = 1'b0;
    always #10 CLOCK_50 = ~CLOCK_50;

    reg  [3:0]  KEY = 4'b1110;      // KEY[0] pressed = reset
    reg  [17:0] SW  = 18'b0;
    wire [8:0]  LEDG;
    wire [17:0] LEDR;
    wire [6:0]  HEX0, HEX1, HEX2, HEX3, HEX4, HEX5, HEX6, HEX7;

    de2_115_mnist_top #(.N_MAC(N_MAC)) dut (.CLOCK_50(CLOCK_50), .KEY(KEY), .SW(SW), .LEDG(LEDG), .LEDR(LEDR),
                           .HEX0(HEX0), .HEX1(HEX1), .HEX2(HEX2), .HEX3(HEX3),
                           .HEX4(HEX4), .HEX5(HEX5), .HEX6(HEX6), .HEX7(HEX7));

    // Python's Q8.8 predictions for the 16 stored images (python/train_mnist.py)
    reg [3:0] exp_pred [0:15];
    reg [3:0] label    [0:15];

    function [6:0] seg;             // same table as hex7seg.v
        input [3:0] v;
        case (v)
            4'h0: seg = 7'b1000000; 4'h1: seg = 7'b1111001; 4'h2: seg = 7'b0100100;
            4'h3: seg = 7'b0110000; 4'h4: seg = 7'b0011001; 4'h5: seg = 7'b0010010;
            4'h6: seg = 7'b0000010; 4'h7: seg = 7'b1111000; 4'h8: seg = 7'b0000000;
            4'h9: seg = 7'b0010000; default: seg = 7'b1111111;
        endcase
    endfunction

    function [3:0] digit;           // 7-seg pattern back to a number, 15 = blank/other
        input [6:0] s;
        integer d;
        begin
            digit = 4'hF;
            for (d = 0; d < 10; d = d + 1) if (s == seg(d)) digit = d;
        end
    endfunction

    function integer dval;          // a leading-blank digit counts as 0
        input [6:0] s;
        dval = (digit(s) == 4'hF) ? 0 : digit(s);
    endfunction

    integer s, n_errors = 0, n_hit = 0;
    reg ok;

    initial begin
        $readmemh("../tb/vectors/mnist_demo_pred.hex", exp_pred);
        $readmemh("../weights/mnist_labels.hex", label);
        repeat (5) @(posedge CLOCK_50);
        KEY[0] = 1'b1;                                   // release reset
        repeat (17 * (EXP_LAT + 50)) @(posedge CLOCK_50); // one full sweep + margin

        for (s = 0; s < 16; s = s + 1) begin
            SW[3:0] = s;
            repeat (4) @(posedge CLOCK_50);
            ok = (exp_pred[s] == label[s]);
            if (ok) n_hit = n_hit + 1;
            if (HEX0 !== seg(exp_pred[s]) || HEX2 !== seg(label[s]) ||
                LEDG[0] !== ok || LEDR[17] !== !ok || LEDR[s] !== ok) begin
                $display("FAIL SW=%0d: HEX0=%0d HEX2=%0d LEDG0=%b LEDR17=%b (expected pred %0d label %0d)",
                         s, digit(HEX0), digit(HEX2), LEDG[0], LEDR[17], exp_pred[s], label[s]);
                n_errors = n_errors + 1;
            end else
                $display("SW=%2d  HEX2 label %0d  HEX0 pred %0d  %s", s, digit(HEX2), digit(HEX0),
                         ok ? "LEDG0 (right)" : "LEDR17 (wrong)");
        end

        // HEX7-4: latency, then the correct count
        SW[17] = 1'b0;
        repeat (4) @(posedge CLOCK_50);
        $display("SW17 down: HEX7-4 = %0d (latency, clock cycles)",
                 dval(HEX7) * 1000 + dval(HEX6) * 100 + dval(HEX5) * 10 + dval(HEX4));
        if (dval(HEX7) * 1000 + dval(HEX6) * 100 + dval(HEX5) * 10 + dval(HEX4) != EXP_LAT ||
            (EXP_LAT < 1000 && HEX7 !== 7'h7F)) begin
            $display("FAIL latency display, expected %0d", EXP_LAT);
            n_errors = n_errors + 1;
        end
        SW[17] = 1'b1;
        repeat (4) @(posedge CLOCK_50);
        $display("SW17 up:   HEX5-4 = %0d%0d (digits correct of 16)", digit(HEX5), digit(HEX4));
        if (digit(HEX5) * 10 + digit(HEX4) != n_hit || HEX7 !== 7'h7F || HEX6 !== 7'h7F) begin
            $display("FAIL correct-count display, expected %0d", n_hit);
            n_errors = n_errors + 1;
        end

        $display("--------------------------------------------------");
        if (n_errors == 0) $display("TEST PASSED: board outputs correct for all 16 stored digits");
        else               $display("TEST FAILED: %0d errors", n_errors);
        $display("--------------------------------------------------");
        $finish;
    end

endmodule
