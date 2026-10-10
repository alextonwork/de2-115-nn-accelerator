// ---------------------------------------------------------------------------
// tb_top.v  -  board-level smoke test for de2_115_top
//
// Flips SW[1:0] through all four XOR inputs and checks what a person would see:
// LEDG[0], the class digit on HEX0, and the minus sign on HEX6.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_top;

    parameter HEX_FILE = "../weights/xor_weights.hex";

    reg CLOCK_50 = 1'b0;
    always #10 CLOCK_50 = ~CLOCK_50;

    reg  [3:0]  KEY = 4'b1110;      // KEY[0] pressed = reset
    reg  [17:0] SW  = 18'b0;
    wire [8:0]  LEDG;
    wire [17:0] LEDR;
    wire [6:0]  HEX0, HEX1, HEX2, HEX3, HEX4, HEX5, HEX6, HEX7;

    de2_115_top dut (.CLOCK_50(CLOCK_50), .KEY(KEY), .SW(SW), .LEDG(LEDG), .LEDR(LEDR),
                     .HEX0(HEX0), .HEX1(HEX1), .HEX2(HEX2), .HEX3(HEX3),
                     .HEX4(HEX4), .HEX5(HEX5), .HEX6(HEX6), .HEX7(HEX7));
    defparam dut.u_core.HEX_FILE = HEX_FILE;

    localparam [6:0] SEG_0 = 7'b1000000, SEG_1 = 7'b1111001,
                     SEG_MINUS = 7'b0111111, SEG_OFF = 7'b1111111;

    integer s, n_errors = 0;
    reg exp_class;

    initial begin
        repeat (5) @(posedge CLOCK_50);
        KEY[0] = 1'b1;                          // release reset
        for (s = 0; s < 4; s = s + 1) begin
            SW[1:0] = s[1:0];
            repeat (100) @(posedge CLOCK_50);   // sync + a few inferences
            exp_class = s[1] ^ s[0];
            if (LEDG[0] !== exp_class || HEX0 !== (exp_class ? SEG_1 : SEG_0) ||
                HEX6 !== (exp_class ? SEG_OFF : SEG_MINUS) || LEDR[1:0] !== s[1:0]) begin
                $display("FAIL SW=%b: LEDG0=%b HEX0=%b HEX6=%b", SW[1:0], LEDG[0], HEX0, HEX6);
                n_errors = n_errors + 1;
            end else
                $display("SW1 SW0 = %b %b -> LEDG0=%b, HEX6..HEX2 = %s%h.%h",
                         SW[1], SW[0], LEDG[0], HEX6 == SEG_MINUS ? "-" : " ",
                         dut.mag[15:8], dut.mag[7:0]);
        end
        $display("--------------------------------------------------");
        if (n_errors == 0) $display("TEST PASSED: board outputs correct for all 4 XOR inputs");
        else               $display("TEST FAILED: %0d errors", n_errors);
        $display("--------------------------------------------------");
        $finish;
    end

endmodule
