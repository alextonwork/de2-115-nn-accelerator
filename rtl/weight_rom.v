`timescale 1ns / 1ps

// ---------------------------------------------------------------------------
// weight_rom.v  -  synchronous-read ROM initialized from a $readmemh file
//
// Used for the weight ROM inside nn_core and for the MNIST image ROM on the
// board. The same $readmemh call initializes it in simulation and in Quartus
// (Quartus supports $readmemh for inferred memory contents), so one parameter
// selects the network and there is no separate synthesis-only path to drift.
// HEX_FILE paths are relative to the tool's working directory: sim/ for
// iverilog/ModelSim and quartus/ for Quartus, which are both one level below
// the repo root, so "../weights/..." works for both. The .mif twins in
// weights/ hold the same data with comments, for reading and for the
// In-System Memory Content Editor.
//
// Registered read (1 clock latency) is what lets Quartus map it to M9K blocks.
// ---------------------------------------------------------------------------
module weight_rom #(
    parameter DATA_W   = 16,
    parameter DEPTH    = 17,
    parameter ADDR_W   = 5,
    parameter HEX_FILE = "../weights/xor_weights.hex"
) (
    input  wire              clk,
    input  wire [ADDR_W-1:0] addr,
    output reg  [DATA_W-1:0] q
);

    (* romstyle = "M9K" *)
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    initial $readmemh(HEX_FILE, mem);

    always @(posedge clk)
        q <= mem[addr];

endmodule
