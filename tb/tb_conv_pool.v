`timescale 1ns / 1ps

// tb_conv_pool: conv_pool on the 16 board demo digits vs python/train_cnn.py,
// bit for bit. Uses the board's image ROM file, so the pixels are the same
// words the MNIST board top already stores.
module tb_conv_pool;
    parameter N_CH = 8;
    localparam DATA_W = 16;
    localparam N_IMG  = 16;
    localparam N_WIN  = 36;

    reg clk = 0, rst_n = 0, start = 0;
    always #10 clk = ~clk;

    reg  [3:0]  img;
    wire [7:0]  x_addr;
    wire [DATA_W-1:0] x_data;
    wire        busy, done, fm_we;
    wire [5:0]  fm_addr;
    wire [N_CH*DATA_W-1:0] fm_wdata;

    weight_rom #(.DATA_W(16), .DEPTH(4096), .ADDR_W(12), .HEX_FILE("../weights/mnist_images.hex"))
    u_img (.clk(clk), .addr({img, x_addr}), .q(x_data));

    conv_pool #(.N_CH(N_CH), .HEX_FILE("../weights/cnn_c8_conv.hex")) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done),
        .x_addr(x_addr), .x_data(x_data),
        .fm_we(fm_we), .fm_addr(fm_addr), .fm_wdata(fm_wdata));

    reg [N_CH*DATA_W-1:0] exp_mem [0:N_IMG*N_WIN-1];
    initial $readmemh("../tb/vectors/cnn_c8_pool_exp.hex", exp_mem);

    integer errors = 0, writes, i, cycles;

    always @(posedge clk)
        if (fm_we) begin
            if (fm_wdata !== exp_mem[img*N_WIN + fm_addr]) begin
                errors = errors + 1;
                if (errors < 10)
                    $display("FAIL img %0d window %0d: got %h exp %h",
                             img, fm_addr, fm_wdata, exp_mem[img*N_WIN + fm_addr]);
            end
            writes = writes + 1;
        end

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1;
        for (i = 0; i < N_IMG; i = i + 1) begin
            @(posedge clk); img = i; writes = 0; start <= 1;
            @(posedge clk); start <= 0;
            cycles = 1;
            while (!done) begin @(posedge clk); cycles = cycles + 1; end
            @(negedge clk);   // let the monitor count the last write (same edge as done)
            if (writes != N_WIN) begin
                errors = errors + 1;
                $display("FAIL img %0d: %0d feature-map writes", i, writes);
            end
        end
        $display("latency %0d cycles per image", cycles);
        if (errors == 0) $display("TEST PASSED: %0d images x %0d windows bit-exact", N_IMG, N_WIN);
        else             $display("TEST FAILED: %0d errors", errors);
        $finish;
    end
endmodule
