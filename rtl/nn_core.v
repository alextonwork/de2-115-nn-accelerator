`timescale 1ns / 1ps

// ---------------------------------------------------------------------------
// nn_core.v  -  2-layer MLP forward pass on a single time-multiplexed MAC
//
//   x (N_IN) -> Dense(N_HID) -> ReLU -> Dense(N_OUT) -> logits, pred
//
// One MAC does every multiply. A small FSM walks the weight ROM in the order
// of the memory map written by python/train_xor.py:
//
//   W1_BASE = 0                      W1[j][i], row-major (N_HID x N_IN)
//   B1_BASE = N_HID*N_IN             b1[j]
//   W2_BASE = B1_BASE + N_HID        W2[o][j], row-major (N_OUT x N_HID)
//   B2_BASE = W2_BASE + N_OUT*N_HID  b2[o]
//
// Each neuron is one dot product of (N_prev + 1) terms, the last term being
// bias * 1.0. Terms stream into the MAC one per clock, neurons back-to-back.
// Layer 2 can only start once every hidden activation is back, so the
// pipeline drains once between layers.
//
// Pipeline alignment:
//   cycle t   : FSM drives rom addr, registers {valid,start,last,activation}
//   cycle t+1 : rom q (weight) + registered activation enter the MAC
//   cycle t+3 : MAC out_done for a neuron's last term -> result captured
//
// For XOR (2-4-1): 4*3 + 5 = 17 MAC terms, 27 clocks from start to done
// (0.54 us at 50 MHz), measured by tb/tb_nn_core.v.
//
// pred: N_OUT == 1  -> 1 bit, logit > 0 (sigmoid(z) > 0.5 without a sigmoid)
//       N_OUT  > 1  -> index of the largest logit (argmax), for MNIST later
// ---------------------------------------------------------------------------
module nn_core #(
    parameter DATA_W   = 16,
    parameter FRAC_W   = 8,
    parameter ACC_W    = 40,
    parameter N_IN     = 2,
    parameter N_HID    = 4,
    parameter N_OUT    = 1,
    parameter HEX_FILE = "../weights/xor_weights.hex"   // simulation only
) (
    input  wire                      clk,
    input  wire                      rst_n,

    input  wire                      start,     // pulse while !busy to run one inference
    input  wire [N_IN*DATA_W-1:0]    x_flat,    // input i at [i*DATA_W +: DATA_W], Q8.8
    output reg                       busy,
    output reg                       done,      // 1-cycle pulse, outputs valid from here
    output wire [N_HID*DATA_W-1:0]   hidden_flat,
    output wire [N_OUT*DATA_W-1:0]   logits_flat,
    output wire [((N_OUT > 1) ? $clog2(N_OUT) : 1)-1:0] pred
);

    // ---------------- sizes ----------------
    localparam DEPTH   = N_HID*N_IN + N_HID + N_OUT*N_HID + N_OUT;
    localparam ADDR_W  = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam W1_BASE = 0;
    localparam B1_BASE = N_HID*N_IN;
    localparam W2_BASE = B1_BASE + N_HID;
    localparam B2_BASE = W2_BASE + N_OUT*N_HID;

    localparam MAXN    = (N_IN > N_HID) ? ((N_IN > N_OUT) ? N_IN : N_OUT)
                                        : ((N_HID > N_OUT) ? N_HID : N_OUT);
    localparam CNT_W   = $clog2(MAXN + 2);
    localparam CLS_W   = (N_OUT > 1) ? $clog2(N_OUT) : 1;

    localparam [DATA_W-1:0] ONE = 1 << FRAC_W;

    // ---------------- storage ----------------
    reg signed [DATA_W-1:0] x_reg   [0:N_IN-1];
    reg signed [DATA_W-1:0] hidden  [0:N_HID-1];
    reg signed [DATA_W-1:0] logit   [0:N_OUT-1];

    // ---------------- FSM ----------------
    localparam S_IDLE  = 2'd0,
               S_ISSUE = 2'd1,   // stream one term per clock into the pipeline
               S_WAIT  = 2'd2;   // drain: wait for all of this layer's results

    reg [1:0]       state;
    reg             layer;       // 0 = hidden layer, 1 = output layer
    reg [CNT_W-1:0] neuron;      // neuron within the current layer
    reg [CNT_W-1:0] k;           // term within the current neuron (last = bias)
    reg [CNT_W-1:0] cap_idx;     // next neuron whose result comes out of the MAC

    wire [CNT_W-1:0] n_prev    = layer ? N_HID : N_IN;   // inputs to this layer
    wire [CNT_W-1:0] n_neurons = layer ? N_OUT : N_HID;
    wire             is_bias   = (k == n_prev);
    wire             last_term = is_bias;
    wire             last_neur = (neuron == n_neurons - 1'b1);

    // ROM address for the current (layer, neuron, k)
    reg [ADDR_W-1:0] rom_addr;
    always @(*) begin
        if (!layer)
            rom_addr = is_bias ? B1_BASE + neuron : W1_BASE + neuron*N_IN + k;
        else
            rom_addr = is_bias ? B2_BASE + neuron : W2_BASE + neuron*N_HID + k;
    end

    // Activation paired with that weight
    reg signed [DATA_W-1:0] act;
    always @(*) begin
        if (is_bias)     act = ONE;
        else if (!layer) act = x_reg[k];
        else             act = hidden[k];
    end

    // ---------------- ROM + stage-0 control registers ----------------
    wire [DATA_W-1:0] weight;

    weight_rom #(.DATA_W(DATA_W), .DEPTH(DEPTH), .ADDR_W(ADDR_W), .HEX_FILE(HEX_FILE))
    u_rom (.clk(clk), .addr(rom_addr), .q(weight));

    reg                     s_valid, s_start, s_last;
    reg signed [DATA_W-1:0] s_act;

    // ---------------- MAC ----------------
    wire                     mac_done;
    wire signed [ACC_W-1:0]  mac_acc;
    wire signed [DATA_W-1:0] mac_result;

    mac #(.DATA_W(DATA_W), .FRAC_W(FRAC_W), .ACC_W(ACC_W)) u_mac (
        .clk(clk), .rst_n(rst_n),
        .in_valid(s_valid), .in_start(s_start), .in_last(s_last),
        .a(weight), .b(s_act),
        .out_done(mac_done), .acc(mac_acc), .result(mac_result)
    );

    // ---------------- control ----------------
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            busy    <= 1'b0;
            done    <= 1'b0;
            layer   <= 1'b0;
            neuron  <= 0;
            k       <= 0;
            cap_idx <= 0;
            s_valid <= 1'b0;
            s_start <= 1'b0;
            s_last  <= 1'b0;
            s_act   <= 0;
            for (i = 0; i < N_HID; i = i + 1) hidden[i] <= 0;
            for (i = 0; i < N_OUT; i = i + 1) logit[i]  <= 0;
        end else begin
            done    <= 1'b0;
            s_valid <= 1'b0;
            s_start <= 1'b0;
            s_last  <= 1'b0;

            // results come out of the MAC in issue order
            if (mac_done) begin
                if (!layer) hidden[cap_idx] <= mac_result[DATA_W-1] ? {DATA_W{1'b0}} : mac_result; // ReLU
                else        logit[cap_idx]  <= mac_result;
                cap_idx <= cap_idx + 1'b1;
            end

            case (state)
            S_IDLE: begin
                if (start) begin
                    for (i = 0; i < N_IN; i = i + 1)
                        x_reg[i] <= x_flat[i*DATA_W +: DATA_W];
                    busy    <= 1'b1;
                    layer   <= 1'b0;
                    neuron  <= 0;
                    k       <= 0;
                    cap_idx <= 0;
                    state   <= S_ISSUE;
                end
            end

            S_ISSUE: begin
                s_valid <= 1'b1;
                s_start <= (k == 0);
                s_last  <= last_term;
                s_act   <= act;
                if (last_term) begin
                    k <= 0;
                    if (last_neur) state <= S_WAIT;
                    else           neuron <= neuron + 1'b1;
                end else begin
                    k <= k + 1'b1;
                end
            end

            S_WAIT: begin
                // cap_idx reaches n_neurons the cycle after the last capture
                if (cap_idx == n_neurons) begin
                    cap_idx <= 0;
                    neuron  <= 0;
                    if (!layer) begin
                        layer <= 1'b1;
                        state <= S_ISSUE;
                    end else begin
                        busy  <= 1'b0;
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end

    // ---------------- outputs ----------------
    genvar g;
    generate
        for (g = 0; g < N_HID; g = g + 1) begin : g_hid
            assign hidden_flat[g*DATA_W +: DATA_W] = hidden[g];
        end
        for (g = 0; g < N_OUT; g = g + 1) begin : g_out
            assign logits_flat[g*DATA_W +: DATA_W] = logit[g];
        end

        if (N_OUT == 1) begin : g_pred_sign
            assign pred = (logit[0] > 0);
        end else begin : g_pred_argmax
            reg [CLS_W-1:0]         best_idx;
            reg signed [DATA_W-1:0] best_val;
            integer o;
            always @(*) begin
                best_idx = 0;
                best_val = logit[0];
                for (o = 1; o < N_OUT; o = o + 1)
                    if (logit[o] > best_val) begin
                        best_val = logit[o];
                        best_idx = o;
                    end
            end
            assign pred = best_idx;
        end
    endgenerate

endmodule
