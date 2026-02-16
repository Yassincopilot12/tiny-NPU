// =============================================================================
// resize_nearest_engine.sv - Nearest-Neighbor Upsampling engine
// Computes nearest-neighbor resize on INT8 NCHW data in SRAM
// FSM: RZ_IDLE -> RZ_READ -> RZ_SRAM_WAIT -> RZ_WRITE -> (next pixel or RZ_DONE)
// For each output pixel: map to nearest source pixel, read it, write to output
// =============================================================================
`default_nettype none

module resize_nearest_engine #(
    parameter int SRAM0_AW = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // -- Command interface ----------------------------------------------------
    input  wire                     cmd_valid,
    input  wire  [15:0]             cmd_src_base,   // SRAM base of input [N,C,in_H,in_W]
    input  wire  [15:0]             cmd_dst_base,   // SRAM base of output [N,C,out_H,out_W]
    input  wire  [15:0]             cmd_C,          // number of channels
    input  wire  [15:0]             cmd_in_H,       // input height
    input  wire  [15:0]             cmd_in_W,       // input width
    input  wire  [15:0]             cmd_out_H,      // output height
    input  wire  [15:0]             cmd_out_W,      // output width

    // -- SRAM0 read port ------------------------------------------------------
    output logic                    sram_rd_en,
    output logic [SRAM0_AW-1:0]    sram_rd_addr,
    input  wire  [7:0]              sram_rd_data,

    // -- SRAM0 write port -----------------------------------------------------
    output logic                    sram_wr_en,
    output logic [SRAM0_AW-1:0]    sram_wr_addr,
    output logic [7:0]             sram_wr_data,

    // -- Status ---------------------------------------------------------------
    output logic                    busy,
    output logic                    done
);

    // =========================================================================
    // FSM states
    // =========================================================================
    typedef enum logic [2:0] {
        RZ_IDLE      = 3'd0,
        RZ_READ      = 3'd1,
        RZ_SRAM_WAIT = 3'd2,
        RZ_WRITE     = 3'd3,
        RZ_DONE      = 3'd4
    } rz_state_t;

    rz_state_t r_state, w_state;

    // =========================================================================
    // Internal registers - latched command parameters
    // =========================================================================
    logic [15:0]           r_src_base;
    logic [15:0]           r_dst_base;
    logic [15:0]           r_C;
    logic [15:0]           r_in_H;
    logic [15:0]           r_in_W;
    logic [15:0]           r_out_H;
    logic [15:0]           r_out_W;

    // Precomputed strides
    logic [15:0]           r_in_channel_stride;    // in_H * in_W
    logic [15:0]           r_out_channel_stride;   // out_H * out_W

    // Loop counters
    logic [15:0]           r_c;    // channel counter
    logic [15:0]           r_oy;   // output y counter
    logic [15:0]           r_ox;   // output x counter

    // =========================================================================
    // Nearest-neighbor coordinate mapping (combinational)
    // =========================================================================
    logic [31:0]           w_mul_y;   // oy * in_H (32-bit to avoid overflow)
    logic [31:0]           w_mul_x;   // ox * in_W (32-bit to avoid overflow)
    logic [15:0]           w_src_y;   // mapped source y
    logic [15:0]           w_src_x;   // mapped source x

    assign w_mul_y = {16'd0, r_oy} * {16'd0, r_in_H};
    assign w_mul_x = {16'd0, r_ox} * {16'd0, r_in_W};
    assign w_src_y = w_mul_y[15:0] / r_out_H;  // (oy * in_H) / out_H -- verilator integer division
    assign w_src_x = w_mul_x[15:0] / r_out_W;  // (ox * in_W) / out_W -- verilator integer division

    // =========================================================================
    // FSM next-state logic
    // =========================================================================
    always_comb begin
        w_state = r_state;
        case (r_state)
            RZ_IDLE: begin
                if (cmd_valid)
                    w_state = RZ_READ;
            end
            RZ_READ: begin
                // SRAM read issued this cycle; need 1 wait cycle for registered output
                w_state = RZ_SRAM_WAIT;
            end
            RZ_SRAM_WAIT: begin
                // SRAM registered output now valid at start of next state
                w_state = RZ_WRITE;
            end
            RZ_WRITE: begin
                // Write result, advance to next output pixel or finish
                if (r_ox + 16'd1 >= r_out_W && r_oy + 16'd1 >= r_out_H && r_c + 16'd1 >= r_C)
                    w_state = RZ_DONE;
                else
                    w_state = RZ_READ;
            end
            RZ_DONE: begin
                w_state = RZ_IDLE;
            end
            default: w_state = RZ_IDLE;
        endcase
    end

    // =========================================================================
    // FSM registered outputs
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_state              <= RZ_IDLE;
            r_src_base           <= 16'h0000;
            r_dst_base           <= 16'h0000;
            r_C                  <= 16'h0000;
            r_in_H               <= 16'h0000;
            r_in_W               <= 16'h0000;
            r_out_H              <= 16'h0000;
            r_out_W              <= 16'h0000;
            r_in_channel_stride  <= 16'h0000;
            r_out_channel_stride <= 16'h0000;
            r_c                  <= 16'h0000;
            r_oy                 <= 16'h0000;
            r_ox                 <= 16'h0000;
            sram_rd_en           <= 1'b0;
            sram_rd_addr         <= '0;
            sram_wr_en           <= 1'b0;
            sram_wr_addr         <= '0;
            sram_wr_data         <= 8'h00;
        end else begin
            r_state <= w_state;

            // Default: de-assert strobes
            sram_rd_en <= 1'b0;
            sram_wr_en <= 1'b0;

            case (r_state)
                RZ_IDLE: begin
                    if (cmd_valid) begin
                        // Latch command parameters
                        r_src_base           <= cmd_src_base;
                        r_dst_base           <= cmd_dst_base;
                        r_C                  <= cmd_C;
                        r_in_H               <= cmd_in_H;
                        r_in_W               <= cmd_in_W;
                        r_out_H              <= cmd_out_H;
                        r_out_W              <= cmd_out_W;

                        // Precompute strides
                        r_in_channel_stride  <= cmd_in_H * cmd_in_W;
                        r_out_channel_stride <= cmd_out_H * cmd_out_W;

                        // Initialize counters
                        r_c  <= 16'h0000;
                        r_oy <= 16'h0000;
                        r_ox <= 16'h0000;
                    end
                end

                RZ_READ: begin
                    // Issue SRAM read for nearest source pixel
                    sram_rd_en   <= 1'b1;
                    sram_rd_addr <= r_src_base[SRAM0_AW-1:0]
                                  + r_c * r_in_channel_stride
                                  + w_src_y * r_in_W
                                  + w_src_x;
                end

                RZ_SRAM_WAIT: begin
                    // Wait for SRAM synchronous read (registered output)
                    // sram_rd_data will be valid at start of next state (RZ_WRITE)
                end

                RZ_WRITE: begin
                    // Write the read pixel value to the output location
                    sram_wr_en   <= 1'b1;
                    sram_wr_addr <= r_dst_base[SRAM0_AW-1:0]
                                  + r_c * r_out_channel_stride
                                  + r_oy * r_out_W
                                  + r_ox;
                    sram_wr_data <= sram_rd_data;

                    // Advance output position: ox -> oy -> c
                    if (r_ox + 16'd1 >= r_out_W) begin
                        r_ox <= 16'h0000;
                        if (r_oy + 16'd1 >= r_out_H) begin
                            r_oy <= 16'h0000;
                            r_c  <= r_c + 16'd1;
                            // If c+1 >= C, all done -> RZ_DONE (handled in next-state)
                        end else begin
                            r_oy <= r_oy + 16'd1;
                        end
                    end else begin
                        r_ox <= r_ox + 16'd1;
                    end
                end

                RZ_DONE: begin
                    // Single-cycle done pulse, return to idle
                end

                default: ;
            endcase
        end
    end

    // =========================================================================
    // Status outputs
    // =========================================================================
    assign busy = (r_state != RZ_IDLE);
    assign done = (r_state == RZ_DONE);

endmodule

`default_nettype wire
