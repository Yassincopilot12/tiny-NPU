// =============================================================================
// Tensor Descriptor Table
// 256-entry register array, 256-bit each (32 bytes per descriptor)
// Combinational read, clocked write
// =============================================================================
`default_nettype none

module tensor_table
    import graph_isa_pkg::*;
#(
    parameter int NUM_ENTRIES = 256,
    parameter int ENTRY_BITS = 256,
    parameter int ADDR_W     = $clog2(NUM_ENTRIES)
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Write port (used by TB to load descriptors)
    input  wire                    wr_en,
    input  wire  [ADDR_W-1:0]     wr_addr,
    input  wire  [ENTRY_BITS-1:0] wr_data,

    // Read port 0 (combinational)
    input  wire  [ADDR_W-1:0]     rd0_addr,
    output wire  [ENTRY_BITS-1:0] rd0_data,

    // Read port 1 (combinational)
    input  wire  [ADDR_W-1:0]     rd1_addr,
    output wire  [ENTRY_BITS-1:0] rd1_data,

    // Read port 2 (combinational)
    input  wire  [ADDR_W-1:0]     rd2_addr,
    output wire  [ENTRY_BITS-1:0] rd2_data
);

    // Storage
    logic [ENTRY_BITS-1:0] tbl_mem [NUM_ENTRIES];

    // Clocked write
    always_ff @(posedge clk) begin
        if (wr_en)
            tbl_mem[wr_addr] <= wr_data;
    end

    // Combinational reads
    assign rd0_data = tbl_mem[rd0_addr];
    assign rd1_data = tbl_mem[rd1_addr];
    assign rd2_data = tbl_mem[rd2_addr];

endmodule

`default_nettype wire
