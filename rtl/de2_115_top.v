`timescale 1ns / 1ps

// ---------------------------------------------------------------------------
// de2_115_top.v  -  XOR MLP accelerator on the Terasic DE2-115
//
// Port names match Terasic's DE2_115 pin assignment file, so the official
// assignments can be imported as-is.
//
//   KEY[0]      reset (press to reset)
//   SW[1:0]     network inputs: SW[i] = x[i] (down = 0.0, up = 1.0)
//   LEDR[1:0]   echo the switches
//   LEDG[0]     predicted class (XOR of SW1, SW0 if the network is right)
//   HEX0        predicted class as a digit
//   HEX6        '-' when the logit is negative
//   HEX5..HEX2  |logit| in Q8.8 hex: HEX5-4 = integer part, HEX3-2 = fraction/256
//               e.g. x=(1,1): "-  06.2B" style -> -(0x062B / 256) = -6.168
//
// The core re-runs inference continuously (27 clocks each), so the display
// follows the switches immediately.
// ---------------------------------------------------------------------------
module de2_115_top (
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire [17:0] SW,
    output wire [8:0]  LEDG,
    output wire [17:0] LEDR,
    output wire [6:0]  HEX0, HEX1, HEX2, HEX3, HEX4, HEX5, HEX6, HEX7
);

    localparam DATA_W = 16;
    localparam FRAC_W = 8;
    localparam [DATA_W-1:0] ONE = 1 << FRAC_W;

    wire clk = CLOCK_50;

    // ---------------- reset + switch synchronizers ----------------
    reg [1:0] rst_sync = 2'b00;
    always @(posedge clk) rst_sync <= {rst_sync[0], KEY[0]};
    wire rst_n = rst_sync[1];

    reg [1:0] sw_meta, sw_sync;
    always @(posedge clk) begin
        sw_meta <= SW[1:0];
        sw_sync <= sw_meta;
    end

    // ---------------- accelerator ----------------
    wire [2*DATA_W-1:0] x_flat = { sw_sync[1] ? ONE : {DATA_W{1'b0}},
                                   sw_sync[0] ? ONE : {DATA_W{1'b0}} };
    wire                busy, done, pred;
    wire [4*DATA_W-1:0] hidden_flat;
    wire [DATA_W-1:0]   logit;

    nn_core #(.N_IN(2), .N_HID(4), .N_OUT(1)) u_core (
        .clk(clk), .rst_n(rst_n),
        .start(!busy && !done), .x_flat(x_flat),
        .busy(busy), .done(done),
        .hidden_flat(hidden_flat), .logits_flat(logit), .pred(pred)
    );

    // ---------------- display ----------------
    wire              neg = logit[DATA_W-1];
    wire [DATA_W-1:0] mag = neg ? (~logit + 1'b1) : logit;

    assign LEDG = {8'b0, pred};
    assign LEDR = {16'b0, sw_sync};

    hex7seg h0 (.value({3'b0, pred}), .blank(1'b0), .minus(1'b0), .seg(HEX0));
    hex7seg h1 (.value(4'h0),         .blank(1'b1), .minus(1'b0), .seg(HEX1));
    hex7seg h2 (.value(mag[3:0]),     .blank(1'b0), .minus(1'b0), .seg(HEX2));
    hex7seg h3 (.value(mag[7:4]),     .blank(1'b0), .minus(1'b0), .seg(HEX3));
    hex7seg h4 (.value(mag[11:8]),    .blank(1'b0), .minus(1'b0), .seg(HEX4));
    hex7seg h5 (.value(mag[15:12]),   .blank(1'b0), .minus(1'b0), .seg(HEX5));
    hex7seg h6 (.value(4'h0),         .blank(!neg), .minus(neg),  .seg(HEX6));
    hex7seg h7 (.value(4'h0),         .blank(1'b1), .minus(1'b0), .seg(HEX7));

endmodule
