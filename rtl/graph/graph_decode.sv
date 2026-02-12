// =============================================================================
// Graph Decode - Thin combinational decode: raw 128-bit → graph_instr_t struct
// =============================================================================
`default_nettype none

module graph_decode
    import graph_isa_pkg::*;
(
    input  wire  [127:0]       raw_instr,
    input  wire                valid_in,

    output graph_instr_t       instr_out,
    output logic               valid_out
);

    assign instr_out = decode_graph_instr(raw_instr);
    assign valid_out = valid_in;

endmodule

`default_nettype wire
