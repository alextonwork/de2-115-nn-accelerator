`timescale 1ns / 1ps

// ---------------------------------------------------------------------------
// conv_pool.v  -  3x3 convolution + ReLU + 2x2 max-pool on N_CH parallel MACs
//
// First layer of the CNN stage (python/train_cnn.py is the golden model):
//   14x14x1 image -> conv 3x3, N_CH filters, valid -> 12x12xN_CH
//                 -> ReLU -> max-pool 2x2 -> 6x6xN_CH feature map
//
// Same idea as nn_core_par: lane l is output channel l. Every clock one pixel
// is read from the external image memory and broadcast to all lanes, and each
// lane takes its own tap weight from one wide ROM row (N_CH words per row).
// The ROM has only K*K+1 rows (taps row-major, then the bias), so its address
// is just the tap counter.
//
// Loop order (outer to inner):
//   pool window (pr, pc) in 0..5 x 0..5      -> one feature-map row write
//     conv position q in 0..3 = (dy, dx)     -> one dot product per lane
//       tap t in 0..9 = 9 pixels, then bias  -> one MAC term per clock
// Each conv output is a full dot product with in_start/in_last, so the MACs
// run back to back with no bubbles: 36*4*10 = 1440 clocks per image.
// The four results of a window are max-reduced as they leave the MACs and
// written once, ReLU applied after the max (max and ReLU commute).
//
// Interfaces
//   x_addr/x_data : synchronous image memory, data valid 1 clock after addr
//                   (the X_EXT convention of nn_core), pixel = row*14 + col
//   fm_*          : one write per pool window, all N_CH channels at once,
//                   fm_addr = pr*6 + pc, lane 0 in the low DATA_W bits
// ---------------------------------------------------------------------------
module conv_pool #(
    parameter N_CH     = 16,
    parameter DATA_W   = 16,
    parameter FRAC_W   = 8,
    parameter ACC_W    = 40,
    parameter HEX_FILE = "../weights/cnn_c16_conv.hex"
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    output reg                      busy,
    output reg                      done,

    output wire [7:0]               x_addr,
    input  wire signed [DATA_W-1:0] x_data,

    output reg                      fm_we,
    output reg  [5:0]               fm_addr,
    output reg  [N_CH*DATA_W-1:0]   fm_wdata
);

    localparam IMG   = 14;
    localparam K     = 3;
    localparam POOL  = 6;                 // (IMG - K + 1) / 2
    localparam TAPS  = K * K;             // bias is term number TAPS
    localparam ROW_W = N_CH * DATA_W;
    localparam [DATA_W-1:0] ONE = 1 << FRAC_W;

    // ---------------- loop counters ----------------
    reg       running;
    reg [2:0] pr, pc;                     // pool window
    reg [1:0] q;                          // conv position inside it: {dy, dx}
    reg [1:0] ky, kx;                     // tap
    wire      is_bias = (ky == K);        // ky runs 0..2, then 3 = bias term

    wire [4:0] row = {pr, 1'b0} + q[1] + ky;
    wire [4:0] col = {pc, 1'b0} + q[0] + kx;
    assign x_addr = row * IMG + col;      // ignored on the bias term

    wire [3:0] tap = is_bias ? TAPS : ky * K + kx;

    // ---------------- weight ROM (K*K+1 rows of N_CH words) ----------------
    wire [ROW_W-1:0] weights;
    weight_rom #(.DATA_W(ROW_W), .DEPTH(TAPS + 1), .ADDR_W(4), .HEX_FILE(HEX_FILE))
    u_rom (.clk(clk), .addr(tap), .q(weights));

    // stage 0: control aligned with the ROM / image read latency
    reg s_valid, s_start, s_last;
    wire signed [DATA_W-1:0] mac_b = s_last ? ONE : x_data;

    // ---------------- N_CH lanes ----------------
    wire [N_CH-1:0]        lane_done;
    wire [ROW_W-1:0]       lane_result;

    genvar l;
    generate
        for (l = 0; l < N_CH; l = l + 1) begin : g_lane
            wire signed [ACC_W-1:0] acc_unused;
            mac #(.DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W)) u_mac (
                .clk(clk), .rst_n(rst_n),
                .in_valid(s_valid), .in_start(s_start), .in_last(s_last),
                .a(weights[l*DATA_W +: DATA_W]), .b(mac_b),
                .out_done(lane_done[l]), .acc(acc_unused),
                .result(lane_result[l*DATA_W +: DATA_W])
            );
        end
    endgenerate

    wire mac_done = lane_done[0];         // lanes run in lockstep

    // ---------------- max-pool + ReLU on the way out ----------------
    reg  [1:0]      q_cap;                // conv position of the result now leaving
    reg  [5:0]      w_cap;                // pool window it belongs to
    reg  [ROW_W-1:0] mx;                  // running max per lane
    wire [ROW_W-1:0] mx_next;

    generate
        for (l = 0; l < N_CH; l = l + 1) begin : g_pool
            wire signed [DATA_W-1:0] r = lane_result[l*DATA_W +: DATA_W];
            wire signed [DATA_W-1:0] m = mx[l*DATA_W +: DATA_W];
            wire signed [DATA_W-1:0] n = (q_cap == 0 || r > m) ? r : m;
            assign mx_next[l*DATA_W +: DATA_W] = n;
            always @(posedge clk)
                if (mac_done && q_cap == 3)
                    fm_wdata[l*DATA_W +: DATA_W] <= n[DATA_W-1] ? {DATA_W{1'b0}} : n;   // ReLU
        end
    endgenerate

    // ---------------- control ----------------
    always @(posedge clk) begin
        if (!rst_n) begin
            busy    <= 1'b0;
            done    <= 1'b0;
            running <= 1'b0;
            {pr, pc, q, ky, kx} <= 0;
            {s_valid, s_start, s_last} <= 3'b000;
            q_cap   <= 0;
            w_cap   <= 0;
            mx      <= 0;
            fm_we   <= 1'b0;
            fm_addr <= 0;
        end else begin
            done  <= 1'b0;
            fm_we <= 1'b0;

            if (start && !busy) begin
                busy    <= 1'b1;
                running <= 1'b1;
                {pr, pc, q, ky, kx} <= 0;
                q_cap   <= 0;
                w_cap   <= 0;
            end

            // issue one term per clock
            s_valid <= running;
            s_start <= running && ky == 0 && kx == 0;
            s_last  <= running && is_bias;
            if (running) begin
                if (is_bias) begin
                    ky <= 0;
                    kx <= 0;
                    q  <= q + 1'b1;
                    if (q == 3) begin
                        if (pc == POOL - 1) begin
                            pc <= 0;
                            if (pr == POOL - 1) running <= 1'b0;
                            else                pr <= pr + 1'b1;
                        end else begin
                            pc <= pc + 1'b1;
                        end
                    end
                end else if (kx == K - 1) begin
                    kx <= 0;
                    ky <= ky + 1'b1;
                end else begin
                    kx <= kx + 1'b1;
                end
            end

            // collect results
            if (mac_done) begin
                mx    <= mx_next;
                q_cap <= q_cap + 1'b1;
                if (q_cap == 3) begin
                    fm_we   <= 1'b1;
                    fm_addr <= w_cap;
                    w_cap   <= w_cap + 1'b1;
                    if (w_cap == POOL * POOL - 1) begin
                        busy <= 1'b0;
                        done <= 1'b1;    // same edge as the last fm write
                    end
                end
            end
        end
    end

endmodule
