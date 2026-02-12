// =============================================================================
// Graph Fetch - Sequential read from program SRAM
// Reads 128-bit instructions one at a time. Detects OP_G_END (opcode==0x00).
// =============================================================================
`default_nettype none

module graph_fetch
    import graph_isa_pkg::*;
#(
    parameter int SRAM_ADDR_W = 10
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Control
    input  wire                     start,
    input  wire  [15:0]             prog_len,     // number of instructions

    // Program SRAM read port
    output logic                    rd_en,
    output logic [SRAM_ADDR_W-1:0] rd_addr,
    input  wire  [127:0]           rd_data,

    // Output instruction
    output logic                    instr_valid,
    output logic [127:0]           instr_data,
    input  wire                     instr_ready,  // downstream consumed

    // Status
    output logic [15:0]            pc,
    output logic                    done,
    output logic                    busy
);

    typedef enum logic [2:0] {
        GF_IDLE,
        GF_READ,
        GF_WAIT_SRAM,
        GF_CHECK,
        GF_PRESENT,
        GF_DONE
    } gf_state_t;

    gf_state_t state;
    logic [15:0] prog_len_r;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= GF_IDLE;
            pc          <= 16'd0;
            prog_len_r  <= 16'd0;
            rd_en       <= 1'b0;
            rd_addr     <= '0;
            instr_valid <= 1'b0;
            instr_data  <= 128'd0;
            done        <= 1'b0;
            busy        <= 1'b0;
        end else begin
            case (state)
                GF_IDLE: begin
                    done        <= 1'b0;
                    instr_valid <= 1'b0;
                    rd_en       <= 1'b0;
                    if (start) begin
                        state      <= GF_READ;
                        pc         <= 16'd0;
                        prog_len_r <= prog_len;
                        busy       <= 1'b1;
                    end
                end

                GF_READ: begin
                    // Issue read for instruction at pc
                    rd_en   <= 1'b1;
                    rd_addr <= pc[SRAM_ADDR_W-1:0];
                    state   <= GF_WAIT_SRAM;
                end

                GF_WAIT_SRAM: begin
                    // SRAM is latching data on this edge; deassert rd_en
                    rd_en <= 1'b0;
                    state <= GF_CHECK;
                end

                GF_CHECK: begin
                    // rd_data (dout_a) is now stable from previous cycle
                    instr_data  <= rd_data;
                    instr_valid <= 1'b1;
                    // Check for OP_G_END (opcode in bits [7:0])
                    if (rd_data[7:0] == OP_G_END) begin
                        state <= GF_DONE;
                    end else begin
                        state <= GF_PRESENT;
                    end
                end

                GF_PRESENT: begin
                    // Wait for downstream to consume
                    if (instr_ready) begin
                        instr_valid <= 1'b0;
                        pc          <= pc + 16'd1;
                        if (pc + 16'd1 >= prog_len_r) begin
                            state <= GF_DONE;
                        end else begin
                            state <= GF_READ;
                        end
                    end
                end

                GF_DONE: begin
                    instr_valid <= 1'b0;
                    done        <= 1'b1;
                    busy        <= 1'b0;
                    state       <= GF_IDLE;
                end

                default: state <= GF_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
