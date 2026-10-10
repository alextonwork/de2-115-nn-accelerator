`timescale 1ns / 1ps

// ---------------------------------------------------------------------------
// de2_115_mnist_top.v  -  MNIST digit classifier on the Terasic DE2-115
//
// 16 test digits (14x14, Q8.8) live in an on-chip image ROM. The core
// (rtl/nn_core_par.v, N_MAC parallel MACs) classifies them round-robin,
// forever, one after another (249 clocks each at N_MAC = 32, 6644 at
// N_MAC = 1) and remembers each prediction.
//
// N_MAC picks the number of MACs (1, 2, 4, 8, 16 or 32) and the matching
// wide weight ROM. quartus/sweep_n_mac.tcl compiles every value in turn.
//
// USE_CNN = 1 swaps the MLP for rtl/cnn_core.v (conv 3x3x16 + max-pool +
// dense, 2037 clocks); everything else, including the displays, stays the
// same. quartus/de2_115_cnn.qsf builds that version.
//
//   KEY[0]      reset (press to reset)
//   SW[3:0]     which stored digit to show (0-15)
//   SW[17]      HEX7-4 mode: down = latency in clock cycles, up = how many of
//               the 16 digits are classified correctly
//   HEX0        predicted digit for the selected image
//   HEX2        true label of the selected image
//   HEX7..HEX4  latency (decimal clock cycles, measured by a hardware counter
//               from start to done) or the correct count, per SW[17]
//   LEDG[0]     selected image classified correctly
//   LEDR[15:0]  one lamp per stored image, lit = classified correctly
//   LEDR[17]    selected image misclassified
//
// Port names match Terasic's DE2_115 pin assignment file, same as de2_115_top.
// ---------------------------------------------------------------------------
module de2_115_mnist_top #(
    parameter N_MAC       = 32,
    parameter USE_CNN     = 0,    // 1: rtl/cnn_core.v instead of nn_core_par
    parameter WEIGHT_FILE = "",   // default: ../weights/mnist_weights_par<NN>.hex
    parameter IMAGE_FILE  = "../weights/mnist_images.hex",
    parameter LABEL_FILE  = "../weights/mnist_labels.hex"
) (
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire [17:0] SW,
    output wire [8:0]  LEDG,
    output wire [17:0] LEDR,
    output wire [6:0]  HEX0, HEX1, HEX2, HEX3, HEX4, HEX5, HEX6, HEX7
);

    localparam DATA_W  = 16;
    localparam N_IN    = 196;
    localparam N_HID   = 32;
    localparam N_OUT   = 10;
    localparam N_IMG   = 16;
    localparam IMG_W   = 4;     // log2(N_IMG)
    localparam PIX_W   = 8;     // 256 words per image in the image ROM

    wire clk = CLOCK_50;

    // ---------------- reset + switch synchronizers ----------------
    reg [1:0] rst_sync = 2'b00;
    always @(posedge clk) rst_sync <= {rst_sync[0], KEY[0]};
    wire rst_n = rst_sync[1];

    reg [4:0] sw_meta, sw_sync;
    always @(posedge clk) begin
        sw_meta <= {SW[17], SW[3:0]};
        sw_sync <= sw_meta;
    end
    wire [IMG_W-1:0] sel       = sw_sync[3:0];
    wire             show_hits = sw_sync[4];

    // ---------------- image + label ROMs ----------------
    wire [PIX_W-1:0]  x_addr;
    wire [DATA_W-1:0] x_data;
    reg  [IMG_W-1:0]  cur_img;     // image being classified right now

    weight_rom #(.DATA_W(DATA_W), .DEPTH(N_IMG << PIX_W), .ADDR_W(IMG_W + PIX_W),
                 .HEX_FILE(IMAGE_FILE))
    u_images (.clk(clk), .addr({cur_img, x_addr}), .q(x_data));

    reg [3:0] label [0:N_IMG-1];
    initial $readmemh(LABEL_FILE, label);

    // ---------------- accelerator ----------------
    wire                     busy, done;
    wire [N_HID*DATA_W-1:0]  hidden_flat;
    wire [N_OUT*DATA_W-1:0]  logits_flat;
    wire [3:0]               pred;
    wire                     start = rst_n && !busy && !done;

    // file names all have the same length so this string mux works in both
    // Icarus and Quartus (shorter strings would be zero-padded)
    localparam WFILE = (WEIGHT_FILE != "") ? WEIGHT_FILE :
                       (N_MAC ==  1) ? "../weights/mnist_weights_par01.hex" :
                       (N_MAC ==  2) ? "../weights/mnist_weights_par02.hex" :
                       (N_MAC ==  4) ? "../weights/mnist_weights_par04.hex" :
                       (N_MAC ==  8) ? "../weights/mnist_weights_par08.hex" :
                       (N_MAC == 16) ? "../weights/mnist_weights_par16.hex" :
                                       "../weights/mnist_weights_par32.hex";

    generate if (USE_CNN) begin : g_cnn
        assign hidden_flat = {N_HID*DATA_W{1'b0}};
        cnn_core u_core (
            .clk(clk), .rst_n(rst_n),
            .start(start), .x_addr(x_addr), .x_data(x_data),
            .busy(busy), .done(done),
            .logits_flat(logits_flat), .pred(pred)
        );
    end else begin : g_mlp
        nn_core_par #(.N_IN(N_IN), .N_HID(N_HID), .N_OUT(N_OUT), .N_MAC(N_MAC),
                      .HEX_FILE(WFILE)) u_core (
            .clk(clk), .rst_n(rst_n),
            .start(start), .x_addr(x_addr), .x_data(x_data),
            .busy(busy), .done(done),
            .hidden_flat(hidden_flat), .logits_flat(logits_flat), .pred(pred)
        );
    end endgenerate

    // ---------------- round-robin over the stored images ----------------
    reg [3:0]       pred_mem [0:N_IMG-1];
    reg [N_IMG-1:0] hit;           // hit[i] = image i classified correctly
    reg [N_IMG-1:0] seen;          // image i has a prediction since reset
    reg [15:0]      cyc, latency;  // hardware latency counter
    integer i;

    always @(posedge clk) begin
        if (!rst_n) begin
            cur_img <= 0;
            hit     <= 0;
            seen    <= 0;
            cyc     <= 0;
            latency <= 0;
            for (i = 0; i < N_IMG; i = i + 1) pred_mem[i] <= 4'd0;
        end else begin
            // count clocks from the start pulse to the done pulse, inclusive
            if (start)     cyc <= 16'd1;
            else if (busy) cyc <= cyc + 1'b1;

            if (done) begin
                pred_mem[cur_img] <= pred;
                hit[cur_img]      <= (pred == label[cur_img]);
                seen[cur_img]     <= 1'b1;
                latency           <= cyc + 1'b1;
                cur_img           <= cur_img + 1'b1;   // wraps 15 -> 0
            end
        end
    end

    // ---------------- display ----------------
    wire       sel_seen = seen[sel];
    wire       sel_hit  = hit[sel];
    wire [3:0] sel_pred = pred_mem[sel];

    // popcount of hit
    reg [4:0] n_hit;
    always @(*) begin
        n_hit = 0;
        for (i = 0; i < N_IMG; i = i + 1) n_hit = n_hit + hit[i];
    end

    // HEX7-4: latency (4 decimal digits) or correct count (2 digits)
    wire [13:0] shown  = show_hits ? {9'b0, n_hit} : latency[13:0];
    wire [15:0] bcd;
    bin2bcd14 u_bcd (.bin(shown), .bcd(bcd));

    // blank leading zeros
    wire blank7 = (bcd[15:12] == 0);
    wire blank6 = blank7 && (bcd[11:8] == 0);
    wire blank5 = blank6 && (bcd[7:4] == 0);

    assign LEDG = {8'b0, sel_seen && sel_hit};
    assign LEDR = {sel_seen && !sel_hit, 1'b0, hit};

    hex7seg h0 (.value(sel_pred),    .blank(!sel_seen), .minus(1'b0), .seg(HEX0));
    hex7seg h1 (.value(4'h0),        .blank(1'b1),      .minus(1'b0), .seg(HEX1));
    hex7seg h2 (.value(label[sel]),  .blank(1'b0),      .minus(1'b0), .seg(HEX2));
    hex7seg h3 (.value(4'h0),        .blank(1'b1),      .minus(1'b0), .seg(HEX3));
    hex7seg h4 (.value(bcd[3:0]),    .blank(1'b0),      .minus(1'b0), .seg(HEX4));
    hex7seg h5 (.value(bcd[7:4]),    .blank(blank5),    .minus(1'b0), .seg(HEX5));
    hex7seg h6 (.value(bcd[11:8]),   .blank(blank6),    .minus(1'b0), .seg(HEX6));
    hex7seg h7 (.value(bcd[15:12]),  .blank(blank7),    .minus(1'b0), .seg(HEX7));

endmodule


// 14-bit binary -> 4 BCD digits (0-9999; larger values are out of range),
// combinational double dabble.
module bin2bcd14 (
    input  wire [13:0] bin,
    output reg  [15:0] bcd
);
    integer i;
    always @(*) begin
        bcd = 16'd0;
        for (i = 13; i >= 0; i = i - 1) begin
            if (bcd[3:0]   >= 5) bcd[3:0]   = bcd[3:0]   + 4'd3;
            if (bcd[7:4]   >= 5) bcd[7:4]   = bcd[7:4]   + 4'd3;
            if (bcd[11:8]  >= 5) bcd[11:8]  = bcd[11:8]  + 4'd3;
            if (bcd[15:12] >= 5) bcd[15:12] = bcd[15:12] + 4'd3;
            bcd = {bcd[14:0], bin[i]};
        end
    end
endmodule
