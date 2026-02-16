// =============================================================================
// math_engine.sv - Element-wise LUT-based math engine
// Supports EXP, LOG, SQRT, RSQRT operations on INT8 and FP16 data via LUTs
// INT8: 5 cycles/element, FP16: 9 cycles/element (2-byte read, LUT, 2-byte write)
// =============================================================================
`default_nettype none

module math_engine
    import graph_isa_pkg::*;
#(
    parameter int SRAM0_AW = 16
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // -- Command interface ------------------------------------------------
    input  wire                    cmd_valid,
    input  wire  [7:0]             cmd_opcode,
    input  wire  [15:0]            cmd_src_base,
    input  wire  [15:0]            cmd_dst_base,
    input  wire  [15:0]            cmd_length,
    input  wire  [1:0]             cmd_dtype,    // 0=INT8, 1=FP16

    // -- SRAM read port ---------------------------------------------------
    output logic                   sram_rd_en,
    output logic [SRAM0_AW-1:0]   sram_rd_addr,
    input  wire  [7:0]             sram_rd_data,

    // -- SRAM write port --------------------------------------------------
    output logic                   sram_wr_en,
    output logic [SRAM0_AW-1:0]   sram_wr_addr,
    output logic [7:0]             sram_wr_data,

    // -- Status -----------------------------------------------------------
    output logic                   busy,
    output logic                   done
);

    // =====================================================================
    // FSM states
    // =====================================================================
    typedef enum logic [3:0] {
        ME_IDLE        = 4'd0,
        ME_READ        = 4'd1,
        ME_SRAM_WAIT   = 4'd2,
        ME_READ_HI     = 4'd3,   // FP16: capture lo, issue hi read
        ME_SRAM_WAIT_HI= 4'd4,   // FP16: wait for hi byte
        ME_LUT_ADDR    = 4'd5,
        ME_LUT_WAIT    = 4'd6,
        ME_WRITE       = 4'd7,
        ME_WRITE_HI    = 4'd8,   // FP16: write high byte
        ME_DONE        = 4'd9
    } me_state_t;

    me_state_t r_state, w_state;

    // =====================================================================
    // Internal registers
    // =====================================================================
    logic [7:0]            r_opcode;
    logic [1:0]            r_dtype;
    logic [15:0]           r_src_base;
    logic [15:0]           r_dst_base;
    logic [15:0]           r_length;
    logic [15:0]           r_index;       // current element index

    logic [7:0]            r_rd_data;     // latched SRAM read data (INT8) / hi byte
    logic [7:0]            r_lo_byte;     // FP16: latched low byte
    logic [15:0]           r_fp16_val;    // FP16: assembled input value

    // =====================================================================
    // INT8 LUT instances
    // =====================================================================
    logic [7:0] lut_addr;
    logic [7:0] exp_out;
    logic [7:0] log_out;
    logic [7:0] sqrt_out;
    logic [7:0] rsqrt_out;
    logic [7:0] lut_result_int8;

    graph_exp_lut u_exp_lut (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (exp_out)
    );

    graph_log_lut u_log_lut (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (log_out)
    );

    graph_sqrt_lut u_sqrt_lut (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (sqrt_out)
    );

    graph_rsqrt_lut u_rsqrt_lut (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (rsqrt_out)
    );

    // =====================================================================
    // FP16 LUT instances
    // =====================================================================
    logic [15:0] exp_out_fp16;
    logic [15:0] log_out_fp16;
    logic [15:0] sqrt_out_fp16;
    logic [15:0] rsqrt_out_fp16;
    logic [15:0] lut_result_fp16;

    graph_exp_lut_fp16 u_exp_lut_fp16 (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (exp_out_fp16)
    );

    graph_log_lut_fp16 u_log_lut_fp16 (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (log_out_fp16)
    );

    graph_sqrt_lut_fp16 u_sqrt_lut_fp16 (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (sqrt_out_fp16)
    );

    graph_rsqrt_lut_fp16 u_rsqrt_lut_fp16 (
        .clk      (clk),
        .addr     (lut_addr),
        .data_out (rsqrt_out_fp16)
    );

    // =====================================================================
    // LUT output mux based on opcode and dtype
    // =====================================================================
    always_comb begin
        case (r_opcode)
            OP_G_EXP:   begin lut_result_int8 = exp_out;   lut_result_fp16 = exp_out_fp16;   end
            OP_G_LOG:   begin lut_result_int8 = log_out;   lut_result_fp16 = log_out_fp16;   end
            OP_G_SQRT:  begin lut_result_int8 = sqrt_out;  lut_result_fp16 = sqrt_out_fp16;  end
            OP_G_RSQRT: begin lut_result_int8 = rsqrt_out; lut_result_fp16 = rsqrt_out_fp16; end
            default:     begin lut_result_int8 = 8'h00;    lut_result_fp16 = 16'h0000;       end
        endcase
    end

    // =====================================================================
    // FSM next-state logic
    // =====================================================================
    always_comb begin
        w_state = r_state;
        case (r_state)
            ME_IDLE: begin
                if (cmd_valid)
                    w_state = ME_READ;
            end
            ME_READ: begin
                w_state = ME_SRAM_WAIT;
            end
            ME_SRAM_WAIT: begin
                if (r_dtype == 2'd1)
                    w_state = ME_READ_HI;    // FP16: need high byte
                else
                    w_state = ME_LUT_ADDR;   // INT8: go to LUT
            end
            ME_READ_HI: begin
                w_state = ME_SRAM_WAIT_HI;
            end
            ME_SRAM_WAIT_HI: begin
                w_state = ME_LUT_ADDR;
            end
            ME_LUT_ADDR: begin
                w_state = ME_LUT_WAIT;
            end
            ME_LUT_WAIT: begin
                w_state = ME_WRITE;
            end
            ME_WRITE: begin
                if (r_dtype == 2'd1) begin
                    w_state = ME_WRITE_HI;   // FP16: write high byte
                end else begin
                    if (r_index + 1 >= r_length)
                        w_state = ME_DONE;
                    else
                        w_state = ME_READ;
                end
            end
            ME_WRITE_HI: begin
                if (r_index + 1 >= r_length)
                    w_state = ME_DONE;
                else
                    w_state = ME_READ;
            end
            ME_DONE: begin
                w_state = ME_IDLE;
            end
            default: w_state = ME_IDLE;
        endcase
    end

    // =====================================================================
    // FSM registered outputs
    // =====================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_state      <= ME_IDLE;
            r_opcode     <= 8'h00;
            r_dtype      <= 2'b00;
            r_src_base   <= 16'h0000;
            r_dst_base   <= 16'h0000;
            r_length     <= 16'h0000;
            r_index      <= 16'h0000;
            r_rd_data    <= 8'h00;
            r_lo_byte    <= 8'h00;
            r_fp16_val   <= 16'h0000;
            sram_rd_en   <= 1'b0;
            sram_rd_addr <= '0;
            sram_wr_en   <= 1'b0;
            sram_wr_addr <= '0;
            sram_wr_data <= 8'h00;
            lut_addr     <= 8'h00;
        end else begin
            r_state <= w_state;

            // Default: de-assert strobes
            sram_rd_en <= 1'b0;
            sram_wr_en <= 1'b0;

            case (r_state)
                ME_IDLE: begin
                    if (cmd_valid) begin
                        r_opcode   <= cmd_opcode;
                        r_dtype    <= cmd_dtype;
                        r_src_base <= cmd_src_base;
                        r_dst_base <= cmd_dst_base;
                        r_length   <= cmd_length;
                        r_index    <= 16'h0000;
                    end
                end

                ME_READ: begin
                    // Issue SRAM read for low byte (or only byte in INT8)
                    sram_rd_en   <= 1'b1;
                    if (r_dtype == 2'd1)
                        sram_rd_addr <= r_src_base[SRAM0_AW-1:0] + (r_index[SRAM0_AW-1:0] << 1);  // FP16: element*2
                    else
                        sram_rd_addr <= r_src_base[SRAM0_AW-1:0] + r_index[SRAM0_AW-1:0];
                end

                ME_SRAM_WAIT: begin
                    // Wait for SRAM synchronous read
                end

                ME_READ_HI: begin
                    // FP16: capture low byte, issue high byte read
                    r_lo_byte    <= sram_rd_data;
                    sram_rd_en   <= 1'b1;
                    sram_rd_addr <= r_src_base[SRAM0_AW-1:0] + (r_index[SRAM0_AW-1:0] << 1) + 1;
                end

                ME_SRAM_WAIT_HI: begin
                    // Wait for high byte SRAM read
                end

                ME_LUT_ADDR: begin
                    if (r_dtype == 2'd1) begin
                        // FP16: assemble 16-bit value, use upper 8 bits as LUT index
                        r_fp16_val <= {sram_rd_data, r_lo_byte};
                        lut_addr   <= sram_rd_data;  // upper 8 bits of FP16
                    end else begin
                        // INT8: use byte directly
                        r_rd_data <= sram_rd_data;
                        lut_addr  <= sram_rd_data;
                    end
                end

                ME_LUT_WAIT: begin
                    // Wait one cycle for LUT registered output
                end

                ME_WRITE: begin
                    sram_wr_en <= 1'b1;
                    if (r_dtype == 2'd1) begin
                        // FP16: write low byte of FP16 LUT result
                        sram_wr_addr <= r_dst_base[SRAM0_AW-1:0] + (r_index[SRAM0_AW-1:0] << 1);
                        sram_wr_data <= lut_result_fp16[7:0];
                    end else begin
                        // INT8: write single byte
                        sram_wr_addr <= r_dst_base[SRAM0_AW-1:0] + r_index[SRAM0_AW-1:0];
                        sram_wr_data <= lut_result_int8;
                        r_index      <= r_index + 16'd1;
                    end
                end

                ME_WRITE_HI: begin
                    // FP16: write high byte of FP16 LUT result
                    sram_wr_en   <= 1'b1;
                    sram_wr_addr <= r_dst_base[SRAM0_AW-1:0] + (r_index[SRAM0_AW-1:0] << 1) + 1;
                    sram_wr_data <= lut_result_fp16[15:8];
                    r_index      <= r_index + 16'd1;
                end

                ME_DONE: begin
                    // Single-cycle done pulse
                end

                default: ;
            endcase
        end
    end

    // =====================================================================
    // Status outputs
    // =====================================================================
    assign busy = (r_state != ME_IDLE);
    assign done = (r_state == ME_DONE);

endmodule

`default_nettype wire
