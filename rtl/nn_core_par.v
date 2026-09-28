`timescale 1ns / 1ps

// ---------------------------------------------------------------------------
// nn_core_par.v  -  2-layer MLP forward pass on N_MAC parallel MACs
//
//   x (N_IN) -> Dense(N_HID) -> ReLU -> Dense(N_OUT) -> logits, pred
//
// Same math, same Q8.8 format and same results as nn_core.v, but the neurons
// of a layer are computed N_MAC at a time: lane l of group g computes neuron
// g*N_MAC + l. Every lane sees the same activation each clock (x[k], or
// hidden[k] in layer 2, broadcast to all MACs) and multiplies it by its own
// weight. So input bandwidth stays at 1 word per clock while weight bandwidth
// grows to N_MAC words per clock, which is why the weights live in one wide
// ROM: word a holds N_MAC weights, lane l in bits [l*16 +: 16].
//
// Wide ROM layout (written by python/gen_par_weights.py), one row per clock,
// read strictly in order, so the address is just a counter:
//
//   group g of layer 1 (G1 = N_HID/N_MAC groups):  N_IN weights, then bias
//   group g of layer 2 (G2 = ceil(N_OUT/N_MAC)):   N_HID weights, then bias
//
// Lanes past N_OUT in the last layer-2 group hold zeros; their results are
// dropped. DEPTH = G1*(N_IN+1) + G2*(N_HID+1) rows of N_MAC*16 bits.
//
// Argmax: with N_MAC >= N_OUT all logits arrive in the same clock, and a
// 10-deep compare chain in one cycle limited Fmax to 28.7 MHz in nn_core. So
// the argmax scans one logit per clock, starting as soon as each logit is
// written (for small N_MAC it keeps pace with the MACs and costs ~1 clock).
//
// Latency (clocks from start to done, measured by tb/tb_mnist_par.v):
//   G1*(N_IN+1) + G2*(N_HID+1) + 9 + L
// where L = N_OUT - (G2-1)*N_MAC is the number of logits in the last group,
// which the argmax scans after the last MAC result (the 9 are ROM, MAC
// pipeline, capture and handshake over both layers). N_MAC = 1 takes exactly
// as long as nn_core.
//
//   MNIST 196-32-10:  N_MAC =  1: 6644    2: 3328    4: 1686
//                     N_MAC =  8:  865   16:  446   32:  249
//
// Inputs are read from an external synchronous memory exactly like
// nn_core's X_EXT = 1 mode: the core drives x_addr = k and expects x_data one
// clock later, held for the whole inference.
// ---------------------------------------------------------------------------
module nn_core_par #(
    parameter DATA_W   = 16,
    parameter FRAC_W   = 8,
    parameter ACC_W    = 40,
    parameter N_IN     = 196,
    parameter N_HID    = 32,
    parameter N_OUT    = 10,
    parameter N_MAC    = 8,       // must divide N_HID
    parameter HEX_FILE = "../weights/mnist_weights_par08.hex"
) (
    input  wire                      clk,
    input  wire                      rst_n,

    input  wire                      start,     // pulse while !busy to run one inference
    output wire [((N_IN > 1) ? $clog2(N_IN) : 1)-1:0] x_addr,  // input index to read
    input  wire [DATA_W-1:0]         x_data,    // mem[x_addr] one clock later, Q8.8
    output reg                       busy,
    output reg                       done,      // 1-cycle pulse, outputs valid from here
    output wire [N_HID*DATA_W-1:0]   hidden_flat,
    output wire [N_OUT*DATA_W-1:0]   logits_flat,
    output wire [((N_OUT > 1) ? $clog2(N_OUT) : 1)-1:0] pred
);

    // ---------------- sizes ----------------
    localparam G1      = N_HID / N_MAC;
    localparam G2      = (N_OUT + N_MAC - 1) / N_MAC;
    localparam DEPTH   = G1*(N_IN + 1) + G2*(N_HID + 1);
    localparam ADDR_W  = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam ROW_W   = N_MAC * DATA_W;

    localparam MAXN    = (N_IN > N_HID) ? N_IN : N_HID;
    localparam K_W     = $clog2(MAXN + 2);
    localparam G_W     = $clog2(((G1 > G2) ? G1 : G2) + 1);
    localparam P_W     = $clog2(N_OUT + 1);
    localparam CLS_W   = (N_OUT > 1) ? $clog2(N_OUT) : 1;

    localparam [DATA_W-1:0] ONE = 1 << FRAC_W;

    // synthesis-time check (iverilog and Quartus both stop on the bad index)
    generate if (N_HID % N_MAC != 0) begin : g_bad_n_mac
        N_MAC_must_divide_N_HID bad();
    end endgenerate

    // ---------------- storage ----------------
    // flat vectors rather than arrays: each word is written by its own
    // always block below, which Quartus maps to plain registers
    reg [N_HID*DATA_W-1:0]  hidden;
    reg [N_OUT*DATA_W-1:0]  logits;
    reg [CLS_W-1:0]         best_idx;
    reg signed [DATA_W-1:0] best_val;

    // ---------------- FSM ----------------
    localparam S_IDLE  = 2'd0,
               S_ISSUE = 2'd1,   // stream one term per clock into all lanes
               S_WAIT  = 2'd2;   // drain the pipeline (and the argmax scan)

    reg [1:0]        state;
    reg              layer;      // 0 = hidden layer, 1 = output layer
    reg [G_W-1:0]    grp;        // neuron group within the current layer
    reg [K_W-1:0]    k;          // term within the current neuron (last = bias)
    reg [ADDR_W-1:0] addr;       // ROM row, simply counts up through the inference
    reg [G_W-1:0]    cap_g;      // next group whose results come out of the MACs
    reg [P_W-1:0]    n_ready;    // logits written so far
    reg [P_W-1:0]    scan;       // next logit the argmax looks at

    wire [K_W-1:0] n_prev   = layer ? N_HID : N_IN;
    wire [G_W-1:0] n_groups = layer ? G2 : G1;
    wire           is_bias  = (k == n_prev);
    wire           last_grp = (grp == n_groups - 1'b1);

    reg signed [DATA_W-1:0] act;
    always @(*) begin
        if (is_bias)    act = ONE;
        else if (layer) act = hidden[k*DATA_W +: DATA_W];
        else            act = {DATA_W{1'b0}};   // layer 1 uses x_data instead
    end

    // ---------------- wide weight ROM + stage-0 control registers ----------------
    wire [ROW_W-1:0] weights;

    weight_rom #(.DATA_W(ROW_W), .DEPTH(DEPTH), .ADDR_W(ADDR_W), .HEX_FILE(HEX_FILE))
    u_rom (.clk(clk), .addr(addr), .q(weights));

    reg                     s_valid, s_start, s_last;
    reg signed [DATA_W-1:0] s_act;
    reg                     s_use_x;

    assign x_addr = k;
    wire signed [DATA_W-1:0] mac_b = s_use_x ? x_data : s_act;

    // ---------------- N_MAC lanes ----------------
    wire [N_MAC-1:0]         lane_done;
    wire [N_MAC*DATA_W-1:0]  lane_result;

    genvar l;
    generate
        for (l = 0; l < N_MAC; l = l + 1) begin : g_lane
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

    wire mac_done = lane_done[0];   // all lanes run in lockstep

    // ---------------- result capture ----------------
    // Each hidden/logit register has exactly one source lane and one group,
    // so the write is a constant-index enable, not a mux tree.
    genvar j;
    generate
        for (j = 0; j < N_HID; j = j + 1) begin : g_cap_hid
            wire signed [DATA_W-1:0] r = lane_result[(j % N_MAC)*DATA_W +: DATA_W];
            always @(posedge clk)
                if (!rst_n)
                    hidden[j*DATA_W +: DATA_W] <= 0;
                else if (mac_done && !layer && cap_g == j / N_MAC)
                    hidden[j*DATA_W +: DATA_W] <= r[DATA_W-1] ? {DATA_W{1'b0}} : r;   // ReLU
        end
        for (j = 0; j < N_OUT; j = j + 1) begin : g_cap_out
            always @(posedge clk)
                if (!rst_n)
                    logits[j*DATA_W +: DATA_W] <= 0;
                else if (mac_done && layer && cap_g == j / N_MAC)
                    logits[j*DATA_W +: DATA_W] <= lane_result[(j % N_MAC)*DATA_W +: DATA_W];
        end
    endgenerate

    wire signed [DATA_W-1:0] scan_val  = logits[scan*DATA_W +: DATA_W];
    wire                     scan_step = (scan < n_ready);

    // ---------------- control ----------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            busy     <= 1'b0;
            done     <= 1'b0;
            layer    <= 1'b0;
            grp      <= 0;
            k        <= 0;
            addr     <= 0;
            cap_g    <= 0;
            n_ready  <= 0;
            scan     <= 0;
            s_valid  <= 1'b0;
            s_start  <= 1'b0;
            s_last   <= 1'b0;
            s_act    <= 0;
            s_use_x  <= 1'b0;
            best_idx <= 0;
            best_val <= 0;
        end else begin
            done    <= 1'b0;
            s_valid <= 1'b0;
            s_start <= 1'b0;
            s_last  <= 1'b0;

            if (mac_done) begin
                cap_g <= cap_g + 1'b1;
                if (layer)
                    n_ready <= (n_ready + N_MAC > N_OUT) ? N_OUT : n_ready + N_MAC;
            end

            // argmax: one logit per clock, as soon as it has been written;
            // ties keep the lower index, same as numpy argmax
            if (scan_step) begin
                if (scan == 0 || scan_val > best_val) begin
                    best_val <= scan_val;
                    best_idx <= scan[CLS_W-1:0];
                end
                scan <= scan + 1'b1;
            end

            case (state)
            S_IDLE: begin
                if (start) begin
                    busy    <= 1'b1;
                    layer   <= 1'b0;
                    grp     <= 0;
                    k       <= 0;
                    addr    <= 0;
                    cap_g   <= 0;
                    n_ready <= 0;
                    scan    <= 0;
                    state   <= S_ISSUE;
                end
            end

            S_ISSUE: begin
                s_valid <= 1'b1;
                s_start <= (k == 0);
                s_last  <= is_bias;
                s_act   <= act;
                s_use_x <= !layer && !is_bias;
                addr    <= addr + 1'b1;
                if (is_bias) begin
                    k <= 0;
                    if (last_grp) state <= S_WAIT;
                    else          grp   <= grp + 1'b1;
                end else begin
                    k <= k + 1'b1;
                end
            end

            S_WAIT: begin
                if (!layer) begin
                    // cap_g reaches G1 the cycle after the last capture
                    if (cap_g == G1) begin
                        cap_g <= 0;
                        grp   <= 0;
                        layer <= 1'b1;
                        state <= S_ISSUE;
                    end
                end else if (cap_g == G2 && scan + scan_step == N_OUT) begin
                    // done goes high on the same edge as the last argmax
                    // update, so pred is already final while done is high
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end

    // ---------------- outputs ----------------
    assign hidden_flat = hidden;
    assign logits_flat = logits;

    assign pred = best_idx;

endmodule
