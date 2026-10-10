`timescale 1ns / 1ps

// ---------------------------------------------------------------------------
// cnn_core.v  -  small CNN forward pass: conv 3x3 + ReLU + max-pool + dense
//
//   x 14x14 -> conv_pool (N_CH lanes) -> feature RAM 6x6xN_CH
//           -> Dense(10) on 10 lanes -> logits -> argmax (1 logit / clock)
//
// python/train_cnn.py is the bit-exact golden model.
//
// Feature RAM: 36 rows of N_CH words, written one row per pool window by
// conv_pool. The dense layer reads activation k = pos*N_CH + ch as row
// k / N_CH, word k % N_CH (N_CH is a power of two, so that is a bit split).
//
// Dense layer: like the output layer of nn_core_par with N_MAC = 10. All ten
// lanes see the same activation each clock and read their own weight from
// one wide ROM (python/train_cnn.py writes it): row k holds W[o][k] in lane
// o, row 36*N_CH holds the biases, so the address is the term counter k.
// The ROM and the feature RAM are both read synchronously in the issue
// clock, so their outputs line up with the stage-0 control one clock later.
//
// Latency (clocks from start to done, measured by tb/tb_cnn.v):
//   conv_pool 1445 + dense 36*N_CH + 1 terms + MAC/ROM pipeline + 10-logit
//   argmax scan. N_CH = 16: 2037.
//
// Inputs come from an external synchronous memory (x_addr = row*14 + col,
// x_data one clock later), the same convention as nn_core_par, so the MNIST
// board top's image ROM works unchanged.
// ---------------------------------------------------------------------------
module cnn_core #(
    parameter N_CH     = 16,      // conv filters, power of two
    parameter DATA_W   = 16,
    parameter FRAC_W   = 8,
    parameter ACC_W    = 40,
    parameter CONV_HEX  = "../weights/cnn_c16_conv.hex",
    parameter DENSE_HEX = "../weights/cnn_c16_dense.hex"
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,      // pulse while !busy
    output wire [7:0]               x_addr,
    input  wire [DATA_W-1:0]        x_data,
    output reg                      busy,
    output reg                      done,       // 1-cycle pulse, outputs valid from here
    output wire [10*DATA_W-1:0]     logits_flat,
    output wire [3:0]               pred
);

    localparam N_OUT  = 10;
    localparam N_POS  = 36;
    localparam NK     = N_POS * N_CH;          // dense fan-in; term NK is the bias
    localparam CS     = $clog2(N_CH);
    localparam K_W    = $clog2(NK + 1);
    localparam ROW_C  = N_CH * DATA_W;
    localparam ROW_D  = N_OUT * DATA_W;
    localparam [DATA_W-1:0] ONE = 1 << FRAC_W;

    generate if ((1 << CS) != N_CH) begin : g_bad_n_ch
        N_CH_must_be_a_power_of_two bad();
    end endgenerate

    // ---------------- conv + pool ----------------
    reg         conv_start;
    wire        conv_busy, conv_done;
    wire        fm_we;
    wire [5:0]  fm_waddr;
    wire [ROW_C-1:0] fm_wdata;

    conv_pool #(.N_CH(N_CH), .DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W),
                .HEX_FILE(CONV_HEX)) u_conv (
        .clk(clk), .rst_n(rst_n), .start(conv_start),
        .busy(conv_busy), .done(conv_done),
        .x_addr(x_addr), .x_data(x_data),
        .fm_we(fm_we), .fm_addr(fm_waddr), .fm_wdata(fm_wdata));

    // ---------------- feature RAM (simple dual port, registered read) ----------------
    reg  [ROW_C-1:0] fm [0:N_POS-1];
    reg  [ROW_C-1:0] fm_q;
    reg  [K_W-1:0]   k;                         // dense term being issued

    always @(posedge clk) begin
        if (fm_we)
            fm[fm_waddr] <= fm_wdata;
        fm_q <= fm[k >> CS];
    end

    // ---------------- dense weight ROM ----------------
    wire [ROW_D-1:0] weights;
    weight_rom #(.DATA_W(ROW_D), .DEPTH(NK + 1), .ADDR_W(K_W), .HEX_FILE(DENSE_HEX))
    u_rom (.clk(clk), .addr(k), .q(weights));

    // stage 0: aligned with fm_q / weights
    reg            s_valid, s_start, s_last;
    reg [CS-1:0]   s_ch;
    wire signed [DATA_W-1:0] act = s_last ? ONE : fm_q[s_ch*DATA_W +: DATA_W];

    // ---------------- 10 dense lanes ----------------
    wire [N_OUT-1:0] lane_done;
    wire [ROW_D-1:0] lane_result;

    genvar l;
    generate
        for (l = 0; l < N_OUT; l = l + 1) begin : g_lane
            wire signed [ACC_W-1:0] acc_unused;
            mac #(.DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W)) u_mac (
                .clk(clk), .rst_n(rst_n),
                .in_valid(s_valid), .in_start(s_start), .in_last(s_last),
                .a(weights[l*DATA_W +: DATA_W]), .b(act),
                .out_done(lane_done[l]), .acc(acc_unused),
                .result(lane_result[l*DATA_W +: DATA_W])
            );
        end
    endgenerate

    // ---------------- logits + argmax scan ----------------
    reg  [ROW_D-1:0]         logits;
    reg                      scanning;
    reg  [3:0]               scan;
    reg  [3:0]               best_idx;
    reg  signed [DATA_W-1:0] best_val;
    wire signed [DATA_W-1:0] scan_val = logits[scan*DATA_W +: DATA_W];

    // ---------------- control ----------------
    localparam S_IDLE = 2'd0, S_CONV = 2'd1, S_DENSE = 2'd2, S_WAIT = 2'd3;
    reg [1:0] state;

    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            busy       <= 1'b0;
            done       <= 1'b0;
            conv_start <= 1'b0;
            k          <= 0;
            {s_valid, s_start, s_last} <= 3'b000;
            s_ch       <= 0;
            logits     <= 0;
            scanning   <= 1'b0;
            scan       <= 0;
            best_idx   <= 0;
            best_val   <= 0;
        end else begin
            done       <= 1'b0;
            conv_start <= 1'b0;
            s_valid    <= 1'b0;
            s_start    <= 1'b0;
            s_last     <= 1'b0;

            if (lane_done[0]) begin                 // lanes run in lockstep
                logits   <= lane_result;
                scanning <= 1'b1;
                scan     <= 0;
            end

            // ties keep the lower index, same as numpy argmax
            if (scanning) begin
                if (scan == 0 || scan_val > best_val) begin
                    best_val <= scan_val;
                    best_idx <= scan;
                end
                scan <= scan + 1'b1;
                if (scan == N_OUT - 1) begin
                    scanning <= 1'b0;
                    busy     <= 1'b0;
                    done     <= 1'b1;               // same edge as the last update
                    state    <= S_IDLE;
                end
            end

            case (state)
            S_IDLE:
                if (start) begin
                    busy       <= 1'b1;
                    conv_start <= 1'b1;
                    state      <= S_CONV;
                end
            S_CONV:
                if (conv_done) begin
                    k     <= 0;
                    state <= S_DENSE;
                end
            S_DENSE: begin
                s_valid <= 1'b1;
                s_start <= (k == 0);
                s_last  <= (k == NK);
                s_ch    <= k[CS-1:0];
                if (k == NK) state <= S_WAIT;
                else         k     <= k + 1'b1;
            end
            default: ;                              // S_WAIT: the scan ends it
            endcase
        end
    end

    assign logits_flat = logits;
    assign pred        = best_idx;

endmodule
