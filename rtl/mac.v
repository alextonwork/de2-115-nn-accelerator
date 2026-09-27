// ---------------------------------------------------------------------------
// mac.v  -  pipelined signed fixed-point multiply-accumulate unit
//
// Computes a dot product one term per clock:   acc = sum_k a[k] * b[k]
//
//   a, b    : signed Q(DATA_W-FRAC_W).FRAC_W   (default Q8.8, 16-bit)
//   product : signed, 2*DATA_W bits, 2*FRAC_W fractional bits (Q16.16)
//   acc     : signed, ACC_W bits (default 40 = 32 + 8 guard bits, so at least
//             2^(ACC_W-2*DATA_W) = 256 worst-case products sum without overflow)
//   result  : acc rounded (half up) back to DATA_W bits with FRAC_W fraction
//             bits, saturated to [min, max] instead of wrapping.
//
// Pipeline (2 stages, full throughput, no bubbles needed between dot products):
//   stage 1: prod_r <= a * b        (maps onto a Cyclone IV 18x18 multiplier
//                                   with its output register)
//   stage 2: acc    <= start ? prod : acc + prod
//
// Handshake:
//   in_valid : a/b hold a valid term this cycle (may be deasserted freely)
//   in_start : this term is the FIRST of a new dot product (reloads acc)
//   in_last  : this term is the LAST of the dot product
//   out_done : pulses for one cycle, 2 cycles after in_last, when acc/result
//              hold the finished dot product. Both stay stable until the
//              next valid term reaches stage 2.
//
// Bias: feed the bias as one extra term with b = 1.0 (1 << FRAC_W), so no
// separate bias port is needed.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module mac #(
    parameter DATA_W = 16,
    parameter FRAC_W = 8,
    parameter ACC_W  = 40
) (
    input  wire                     clk,
    input  wire                     rst_n,      // active-low synchronous reset

    input  wire                     in_valid,
    input  wire                     in_start,
    input  wire                     in_last,
    input  wire signed [DATA_W-1:0] a,
    input  wire signed [DATA_W-1:0] b,

    output reg                      out_done,
    output reg  signed [ACC_W-1:0]  acc,
    output wire signed [DATA_W-1:0] result
);

    localparam PROD_W = 2 * DATA_W;

    // ---------------- stage 1: multiply ----------------
    reg  signed [PROD_W-1:0] prod_r;
    reg                      v1, start1, last1;

    always @(posedge clk) begin
        if (!rst_n) begin
            v1     <= 1'b0;
            start1 <= 1'b0;
            last1  <= 1'b0;
        end else begin
            v1     <= in_valid;
            start1 <= in_start;
            last1  <= in_last;
        end
    end

    // Data register has no reset (v1 qualifies it), so Quartus can pack it
    // into the embedded multiplier's own output register.
    always @(posedge clk) begin
        if (in_valid)
            prod_r <= a * b;
    end

    // ---------------- stage 2: accumulate ----------------
    // Sign-extend the product to the accumulator width.
    wire signed [ACC_W-1:0] prod_ext = {{(ACC_W-PROD_W){prod_r[PROD_W-1]}}, prod_r};

    always @(posedge clk) begin
        if (!rst_n) begin
            acc      <= {ACC_W{1'b0}};
            out_done <= 1'b0;
        end else begin
            out_done <= v1 & last1;
            if (v1)
                acc <= start1 ? prod_ext : acc + prod_ext;
        end
    end

    // ---------------- output: round + saturate to DATA_W ----------------
    // acc has 2*FRAC_W fraction bits; result needs FRAC_W. Add half an output
    // LSB, then arithmetic shift right by FRAC_W (= round half toward +inf).
    // One extra bit on the adder so the rounding add itself cannot overflow.
    localparam RND_W = ACC_W + 1;

    wire signed [RND_W-1:0] acc_rnd     = {acc[ACC_W-1], acc} + ({{(RND_W-1){1'b0}}, 1'b1} <<< (FRAC_W-1));
    wire signed [RND_W-1:0] acc_shifted = acc_rnd >>> FRAC_W;

    localparam signed [RND_W-1:0] SAT_MAX = (1 <<< (DATA_W-1)) - 1;
    localparam signed [RND_W-1:0] SAT_MIN = -(1 <<< (DATA_W-1));

    assign result = (acc_shifted > SAT_MAX) ? SAT_MAX[DATA_W-1:0] :
                    (acc_shifted < SAT_MIN) ? SAT_MIN[DATA_W-1:0] :
                                              acc_shifted[DATA_W-1:0];

endmodule
