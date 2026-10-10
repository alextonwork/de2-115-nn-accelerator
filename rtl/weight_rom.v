`timescale 1ns / 1ps

// ---------------------------------------------------------------------------
// weight_rom.v  -  synchronous-read ROM holding all weights and biases
//
// Synthesis (Quartus): initialized from weights/xor_weights.mif through the
//   ram_init_file attribute and placed in an M9K block (romstyle). The path is
//   relative to the Quartus project directory (quartus/).
// Simulation: initialized from the .hex twin of the same data via $readmemh
//   (hidden from synthesis with translate_off, so Quartus uses the .mif).
//
// Registered read (1 clock latency) is what lets Quartus map it to block RAM.
// ---------------------------------------------------------------------------
module weight_rom #(
    parameter DATA_W   = 16,
    parameter DEPTH    = 17,
    parameter ADDR_W   = 5,
    parameter HEX_FILE = "../weights/xor_weights.hex"   // simulation only
) (
    input  wire              clk,
    input  wire [ADDR_W-1:0] addr,
    output reg  [DATA_W-1:0] q
);

    (* romstyle = "M9K", ram_init_file = "../weights/xor_weights.mif" *)
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    // synthesis translate_off
    initial $readmemh(HEX_FILE, mem);
    // synthesis translate_on

    always @(posedge clk)
        q <= mem[addr];

endmodule
