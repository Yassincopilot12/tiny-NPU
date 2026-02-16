// =============================================================================
// cast_engine.sv - Dtype conversion engine (INT8 <-> FP16)
// Phase 4 simplified: element-by-element INT8->INT8 copy (passthrough)
// Actual dtype conversion (INT8->FP16, FP16->INT8) will be added when
// fp16_utils.sv is created in a later phase.
// FSM: CE_IDLE -> CE_READ -> CE_SRAM_WAIT -> CE_WRITE -> loop/CE_DONE
// =============================================================================
`default_nettype none

module cast_engine
    import graph_isa_pkg::*;
#(
    parameter int SRAM0_AW = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // -- Command interface ------------------------------------------------
    input  wire                     cmd_valid,
    input  wire  [15:0]             cmd_src_base,
    input  wire  [15:0]             cmd_dst_base,
    input  wire  [15:0]             cmd_length,      // number of elements
    input  wire  [7:0]              cmd_src_dtype,   // source dtype (0=INT8, 1=FP16, 2=INT32)
    input  wire  [7:0]              cmd_dst_dtype,   // destination dtype

    // -- SRAM read port ---------------------------------------------------
    output logic                    sram_rd_en,
    output logic [SRAM0_AW-1:0]    sram_rd_addr,
    input  wire  [7:0]             sram_rd_data,

    // -- SRAM write port --------------------------------------------------
    output logic                    sram_wr_en,
    output logic [SRAM0_AW-1:0]    sram_wr_addr,
    output logic [7:0]             sram_wr_data,

    // -- Status -----------------------------------------------------------
    output logic                    busy,
    output logic                    done
);

    // =====================================================================
    // FSM states
    // =====================================================================
    typedef enum logic [2:0] {
        CE_IDLE      = 3'd0,
        CE_READ      = 3'd1,
        CE_SRAM_WAIT = 3'd2,
        CE_WRITE     = 3'd3,
        CE_DONE      = 3'd4
    } ce_state_t;

    ce_state_t r_state, w_state;

    // =====================================================================
    // Internal registers
    // =====================================================================
    logic [15:0] r_src_base;
    logic [15:0] r_dst_base;
    logic [15:0] r_length;
    logic [15:0] r_index;
    logic [7:0]  r_src_dtype;
    logic [7:0]  r_dst_dtype;

    // =====================================================================
    // FSM next-state logic
    // =====================================================================
    always_comb begin
        w_state = r_state;
        case (r_state)
            CE_IDLE: begin
                if (cmd_valid)
                    w_state = CE_READ;
            end
            CE_READ: begin
                // SRAM read issued this cycle; need 1 cycle for registered output
                w_state = CE_SRAM_WAIT;
            end
            CE_SRAM_WAIT: begin
                // SRAM registered output now valid on next cycle
                w_state = CE_WRITE;
            end
            CE_WRITE: begin
                // Write data to SRAM, advance to next element or finish
                if (r_index + 16'd1 >= r_length)
                    w_state = CE_DONE;
                else
                    w_state = CE_READ;
            end
            CE_DONE: begin
                w_state = CE_IDLE;
            end
            default: w_state = CE_IDLE;
        endcase
    end

    // =====================================================================
    // FSM registered outputs
    // =====================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_state      <= CE_IDLE;
            r_src_base   <= 16'h0000;
            r_dst_base   <= 16'h0000;
            r_length     <= 16'h0000;
            r_index      <= 16'h0000;
            r_src_dtype  <= 8'h00;
            r_dst_dtype  <= 8'h00;
            sram_rd_en   <= 1'b0;
            sram_rd_addr <= '0;
            sram_wr_en   <= 1'b0;
            sram_wr_addr <= '0;
            sram_wr_data <= 8'h00;
        end else begin
            r_state <= w_state;

            // Default: de-assert strobes
            sram_rd_en <= 1'b0;
            sram_wr_en <= 1'b0;

            case (r_state)
                CE_IDLE: begin
                    if (cmd_valid) begin
                        r_src_base  <= cmd_src_base;
                        r_dst_base  <= cmd_dst_base;
                        r_length    <= cmd_length;
                        r_index     <= 16'h0000;
                        r_src_dtype <= cmd_src_dtype;
                        r_dst_dtype <= cmd_dst_dtype;
                    end
                end

                CE_READ: begin
                    // Issue SRAM read for current element
                    sram_rd_en   <= 1'b1;
                    sram_rd_addr <= r_src_base[SRAM0_AW-1:0] + r_index[SRAM0_AW-1:0];
                end

                CE_SRAM_WAIT: begin
                    // Wait for SRAM synchronous read (registered output)
                    // sram_rd_data will be valid at start of next state
                end

                CE_WRITE: begin
                    // Phase 4: simple passthrough (INT8 -> INT8 copy)
                    // TODO: add dtype conversion when fp16_utils.sv is available
                    //   INT8  -> FP16: expand 1 byte to 2 bytes
                    //   FP16  -> INT8: compress 2 bytes to 1 byte
                    sram_wr_en   <= 1'b1;
                    sram_wr_addr <= r_dst_base[SRAM0_AW-1:0] + r_index[SRAM0_AW-1:0];
                    sram_wr_data <= sram_rd_data;
                    r_index      <= r_index + 16'd1;
                end

                CE_DONE: begin
                    // Single-cycle done pulse, return to idle
                end

                default: ;
            endcase
        end
    end

    // =====================================================================
    // Status outputs
    // =====================================================================
    assign busy = (r_state != CE_IDLE);
    assign done = (r_state == CE_DONE);

endmodule

`default_nettype wire
