// =============================================================================
// pad_engine.sv - Constant-0 padding engine
// Pads INT8 NCHW tensors in SRAM with zeros
// FSM: PD_IDLE -> PD_CHECK -> PD_READ -> PD_SRAM_WAIT -> PD_WRITE -> (next pixel or PD_DONE)
// For each output pixel: write 0 if in padded region, else copy source pixel
// =============================================================================
`default_nettype none

module pad_engine #(
    parameter int SRAM0_AW = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // -- Command interface ----------------------------------------------------
    input  wire                     cmd_valid,
    input  wire  [15:0]             cmd_src_base,   // SRAM base of input [C,H,W]
    input  wire  [15:0]             cmd_dst_base,   // SRAM base of output [C,out_H,out_W]
    input  wire  [15:0]             cmd_C,          // number of channels
    input  wire  [15:0]             cmd_H,          // input height
    input  wire  [15:0]             cmd_W,          // input width
    input  wire  [7:0]              cmd_pad_top,    // padding rows above
    input  wire  [7:0]              cmd_pad_bottom, // padding rows below
    input  wire  [7:0]              cmd_pad_left,   // padding columns left
    input  wire  [7:0]              cmd_pad_right,  // padding columns right

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
        PD_IDLE      = 3'd0,
        PD_CHECK     = 3'd1,
        PD_READ      = 3'd2,
        PD_SRAM_WAIT = 3'd3,
        PD_WRITE     = 3'd4,
        PD_DONE      = 3'd5
    } pd_state_t;

    pd_state_t r_state, w_state;

    // =========================================================================
    // Internal registers - latched command parameters
    // =========================================================================
    logic [15:0]           r_src_base;
    logic [15:0]           r_dst_base;
    logic [15:0]           r_C;
    logic [15:0]           r_H;
    logic [15:0]           r_W;
    logic [7:0]            r_pad_top;
    logic [7:0]            r_pad_bottom;
    logic [7:0]            r_pad_left;
    logic [7:0]            r_pad_right;

    // Precomputed dimensions
    logic [15:0]           r_out_h;
    logic [15:0]           r_out_w;
    logic [15:0]           r_channel_stride;       // H * W
    logic [15:0]           r_out_channel_stride;   // out_h * out_w

    // Loop counters
    logic [15:0]           r_c;    // channel counter
    logic [15:0]           r_oy;   // output height counter
    logic [15:0]           r_ox;   // output width counter

    // Pad-region flag
    logic                  r_is_pad;

    // =========================================================================
    // FSM next-state logic
    // =========================================================================
    always_comb begin
        w_state = r_state;
        case (r_state)
            PD_IDLE: begin
                if (cmd_valid)
                    w_state = PD_CHECK;
            end
            PD_CHECK: begin
                // Determine if current output pixel is in padded region or data region
                if (r_oy < {8'd0, r_pad_top} ||
                    r_oy >= r_H + {8'd0, r_pad_top} ||
                    r_ox < {8'd0, r_pad_left} ||
                    r_ox >= r_W + {8'd0, r_pad_left})
                    w_state = PD_WRITE;  // padded region, write 0 directly
                else
                    w_state = PD_READ;   // data region, need to read source
            end
            PD_READ: begin
                // SRAM read issued this cycle; need 1 wait cycle for registered output
                w_state = PD_SRAM_WAIT;
            end
            PD_SRAM_WAIT: begin
                // SRAM registered output now valid at start of next state
                w_state = PD_WRITE;
            end
            PD_WRITE: begin
                // Write result, advance to next output pixel or finish
                if (r_ox + 16'd1 >= r_out_w && r_oy + 16'd1 >= r_out_h && r_c + 16'd1 >= r_C)
                    w_state = PD_DONE;
                else
                    w_state = PD_CHECK;
            end
            PD_DONE: begin
                w_state = PD_IDLE;
            end
            default: w_state = PD_IDLE;
        endcase
    end

    // =========================================================================
    // FSM registered outputs
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_state              <= PD_IDLE;
            r_src_base           <= 16'h0000;
            r_dst_base           <= 16'h0000;
            r_C                  <= 16'h0000;
            r_H                  <= 16'h0000;
            r_W                  <= 16'h0000;
            r_pad_top            <= 8'h00;
            r_pad_bottom         <= 8'h00;
            r_pad_left           <= 8'h00;
            r_pad_right          <= 8'h00;
            r_out_h              <= 16'h0000;
            r_out_w              <= 16'h0000;
            r_channel_stride     <= 16'h0000;
            r_out_channel_stride <= 16'h0000;
            r_c                  <= 16'h0000;
            r_oy                 <= 16'h0000;
            r_ox                 <= 16'h0000;
            r_is_pad             <= 1'b0;
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
                PD_IDLE: begin
                    if (cmd_valid) begin
                        // Latch command parameters
                        r_src_base           <= cmd_src_base;
                        r_dst_base           <= cmd_dst_base;
                        r_C                  <= cmd_C;
                        r_H                  <= cmd_H;
                        r_W                  <= cmd_W;
                        r_pad_top            <= cmd_pad_top;
                        r_pad_bottom         <= cmd_pad_bottom;
                        r_pad_left           <= cmd_pad_left;
                        r_pad_right          <= cmd_pad_right;

                        // Precompute dimensions
                        r_out_h              <= cmd_H + {8'd0, cmd_pad_top} + {8'd0, cmd_pad_bottom};
                        r_out_w              <= cmd_W + {8'd0, cmd_pad_left} + {8'd0, cmd_pad_right};
                        r_channel_stride     <= cmd_H * cmd_W;
                        r_out_channel_stride <= (cmd_H + {8'd0, cmd_pad_top} + {8'd0, cmd_pad_bottom}) *
                                                (cmd_W + {8'd0, cmd_pad_left} + {8'd0, cmd_pad_right});

                        // Initialize counters
                        r_c  <= 16'h0000;
                        r_oy <= 16'h0000;
                        r_ox <= 16'h0000;

                        // Clear pad flag
                        r_is_pad <= 1'b0;
                    end
                end

                PD_CHECK: begin
                    // Set pad flag based on whether current pixel is in the padded region
                    if (r_oy < {8'd0, r_pad_top} ||
                        r_oy >= r_H + {8'd0, r_pad_top} ||
                        r_ox < {8'd0, r_pad_left} ||
                        r_ox >= r_W + {8'd0, r_pad_left}) begin
                        r_is_pad <= 1'b1;
                    end else begin
                        r_is_pad <= 1'b0;
                    end
                end

                PD_READ: begin
                    // Issue SRAM read for source pixel
                    sram_rd_en   <= 1'b1;
                    sram_rd_addr <= r_src_base[SRAM0_AW-1:0]
                                  + r_c * r_channel_stride
                                  + (r_oy - {8'd0, r_pad_top}) * r_W
                                  + (r_ox - {8'd0, r_pad_left});
                end

                PD_SRAM_WAIT: begin
                    // Wait for SRAM synchronous read (registered output)
                    // sram_rd_data will be valid at start of next state (PD_WRITE)
                end

                PD_WRITE: begin
                    // Write data: 0 for padded region, sram_rd_data for copied region
                    sram_wr_en   <= 1'b1;
                    sram_wr_addr <= r_dst_base[SRAM0_AW-1:0]
                                  + r_c * r_out_channel_stride
                                  + r_oy * r_out_w
                                  + r_ox;

                    if (r_is_pad)
                        sram_wr_data <= 8'h00;
                    else
                        sram_wr_data <= sram_rd_data;

                    // Advance output position: ox -> oy -> c
                    if (r_ox + 16'd1 >= r_out_w) begin
                        r_ox <= 16'h0000;
                        if (r_oy + 16'd1 >= r_out_h) begin
                            r_oy <= 16'h0000;
                            r_c  <= r_c + 16'd1;
                            // If c+1 >= C, all done -> PD_DONE (handled in next-state)
                        end else begin
                            r_oy <= r_oy + 16'd1;
                        end
                    end else begin
                        r_ox <= r_ox + 16'd1;
                    end
                end

                PD_DONE: begin
                    // Single-cycle done pulse, return to idle
                end

                default: ;
            endcase
        end
    end

    // =========================================================================
    // Status outputs
    // =========================================================================
    assign busy = (r_state != PD_IDLE);
    assign done = (r_state == PD_DONE);

endmodule

`default_nettype wire
