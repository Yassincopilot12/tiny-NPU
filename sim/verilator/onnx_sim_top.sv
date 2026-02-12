// =============================================================================
// onnx_sim_top.sv - Graph Mode simulation wrapper
// Contains: program SRAM, tensor table, SRAM0 (65536), ACC SRAM, scratch SRAM,
//           gemm_ctrl, systolic_array, softmax_engine, graph pipeline, DMA shim
//
// SRAM0 mux priority: graph_dispatch_ew > gemm > softmax > TB
// =============================================================================
`default_nettype none

module onnx_sim_top
    import npu_pkg::*;
    import graph_isa_pkg::*;
#(
    parameter int SRAM0_DEPTH   = 65536,
    parameter int SCRATCH_DEPTH = 4096,
    parameter int PROG_DEPTH    = 1024,
    parameter int SRAM0_AW      = $clog2(SRAM0_DEPTH),
    parameter int SCR_AW        = $clog2(SCRATCH_DEPTH),
    parameter int PROG_AW       = $clog2(PROG_DEPTH)
)(
    input  wire                clk,
    input  wire                rst_n,

    // --- Control ---
    input  wire                start_pulse,
    input  wire  [15:0]        prog_len,
    output wire                graph_done,

    // --- Program SRAM write (TB loads program via port B) ---
    input  wire                prog_wr_en,
    input  wire  [PROG_AW-1:0] prog_wr_addr,
    input  wire  [127:0]       prog_wr_data,

    // --- Tensor Table write (TB loads descriptors) ---
    input  wire                tdesc_wr_en,
    input  wire  [7:0]         tdesc_wr_addr,
    input  wire  [255:0]       tdesc_wr_data,

    // --- DATA SRAM0 TB access ---
    input  wire                tb_sram0_wr_en,
    input  wire  [SRAM0_AW-1:0] tb_sram0_wr_addr,
    input  wire  [7:0]         tb_sram0_wr_data,
    input  wire                tb_sram0_rd_en,
    input  wire  [SRAM0_AW-1:0] tb_sram0_rd_addr,
    output wire  [7:0]         tb_sram0_rd_data,

    // --- DMA command capture (exposed to C++) ---
    output wire                dma_cmd_captured,
    output wire  [31:0]        dma_ddr_addr,
    output wire  [15:0]        dma_sram_addr,
    output wire  [15:0]        dma_length,
    output wire                dma_direction,
    output wire                dma_strided,
    output wire  [31:0]        dma_stride,
    output wire  [15:0]        dma_count,
    output wire  [15:0]        dma_block_len,
    input  wire                dma_done_pulse,

    // --- Debug ---
    output wire  [31:0]        graph_status,
    output wire  [15:0]        graph_pc,
    output wire  [7:0]         graph_last_op,
    output wire                graph_busy
);

    // ================================================================
    // Program SRAM (128-bit x PROG_DEPTH)
    // ================================================================
    logic                  prog_rd_en;
    logic [PROG_AW-1:0]   prog_rd_addr;
    logic [127:0]          prog_rd_data;

    sram_dp #(.DEPTH(PROG_DEPTH), .WIDTH(128)) u_prog_sram (
        .clk    (clk),
        .en_a   (prog_rd_en),
        .we_a   (1'b0),
        .addr_a (prog_rd_addr),
        .din_a  (128'd0),
        .dout_a (prog_rd_data),
        .en_b   (prog_wr_en),
        .we_b   (prog_wr_en),
        .addr_b (prog_wr_addr),
        .din_b  (prog_wr_data),
        .dout_b ()
    );

    // ================================================================
    // Tensor Descriptor Table
    // ================================================================
    logic [7:0]   td_rd0_addr, td_rd1_addr, td_rd2_addr;
    logic [255:0] td_rd0_data, td_rd1_data, td_rd2_data;

    tensor_table u_tensor_table (
        .clk      (clk),
        .rst_n    (rst_n),
        .wr_en    (tdesc_wr_en),
        .wr_addr  (tdesc_wr_addr),
        .wr_data  (tdesc_wr_data),
        .rd0_addr (td_rd0_addr),
        .rd0_data (td_rd0_data),
        .rd1_addr (td_rd1_addr),
        .rd1_data (td_rd1_data),
        .rd2_addr (td_rd2_addr),
        .rd2_data (td_rd2_data)
    );

    // ================================================================
    // GEMM Engine: gemm_ctrl + systolic_array + ACC SRAM
    // ================================================================
    logic        gm_busy, gm_done;
    logic        gm_rd_en, gm_wr_en;
    logic [15:0] gm_rd_addr, gm_wr_addr;
    logic [7:0]  gm_wr_data;
    logic        gm_sa_clear, gm_sa_en;
    logic signed [7:0]  gm_sa_a_col [16];
    logic signed [7:0]  gm_sa_b_row [16];
    logic signed [31:0] gm_sa_acc   [16][16];

    logic        gm_acc_rd_en, gm_acc_wr_en;
    logic [7:0]  gm_acc_rd_addr, gm_acc_wr_addr;
    logic signed [31:0] gm_acc_rd_data, gm_acc_wr_data;

    // Graph pipeline signals
    logic        gp_gm_cmd_valid;
    logic [15:0] gp_gm_cmd_src0, gp_gm_cmd_src1, gp_gm_cmd_dst;
    logic [15:0] gp_gm_cmd_M, gp_gm_cmd_N, gp_gm_cmd_K;
    logic [7:0]  gp_gm_cmd_flags;
    logic [15:0] gp_gm_cmd_imm;

    gemm_ctrl #(
        .ARRAY_M     (16),
        .ARRAY_N     (16),
        .DATA_W      (8),
        .ACC_W       (32),
        .SRAM_ADDR_W (16)
    ) u_gemm_ctrl (
        .clk          (clk),
        .rst_n        (rst_n),
        .cmd_valid    (gp_gm_cmd_valid),
        .cmd_src0     (gp_gm_cmd_src0),
        .cmd_src1     (gp_gm_cmd_src1),
        .cmd_dst      (gp_gm_cmd_dst),
        .cmd_M        (gp_gm_cmd_M),
        .cmd_N        (gp_gm_cmd_N),
        .cmd_K        (gp_gm_cmd_K),
        .cmd_flags    (gp_gm_cmd_flags),
        .cmd_imm      (gp_gm_cmd_imm),
        .sram_rd_en   (gm_rd_en),
        .sram_rd_addr (gm_rd_addr),
        .sram_rd_data (s0_a_dout),
        .sram_wr_en   (gm_wr_en),
        .sram_wr_addr (gm_wr_addr),
        .sram_wr_data (gm_wr_data),
        .acc_rd_en    (gm_acc_rd_en),
        .acc_rd_addr  (gm_acc_rd_addr),
        .acc_rd_data  (gm_acc_rd_data),
        .acc_wr_en    (gm_acc_wr_en),
        .acc_wr_addr  (gm_acc_wr_addr),
        .acc_wr_data  (gm_acc_wr_data),
        .sa_clear     (gm_sa_clear),
        .sa_en        (gm_sa_en),
        .sa_a_col     (gm_sa_a_col),
        .sa_b_row     (gm_sa_b_row),
        .sa_acc       (gm_sa_acc),
        .busy         (gm_busy),
        .done         (gm_done)
    );

    systolic_array #(
        .M      (16),
        .N      (16),
        .DATA_W (8),
        .ACC_W  (32)
    ) u_systolic (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear_acc (gm_sa_clear),
        .en        (gm_sa_en),
        .a_col     (gm_sa_a_col),
        .b_row     (gm_sa_b_row),
        .acc_out   (gm_sa_acc),
        .acc_valid ()
    );

    // ACC SRAM (32-bit x 256 entries)
    sram_dp #(.DEPTH(256), .WIDTH(32)) u_acc_sram (
        .clk    (clk),
        .en_a   (gm_acc_rd_en),
        .we_a   (1'b0),
        .addr_a (gm_acc_rd_addr),
        .din_a  (32'd0),
        .dout_a (gm_acc_rd_data),
        .en_b   (gm_acc_wr_en),
        .we_b   (gm_acc_wr_en),
        .addr_b (gm_acc_wr_addr),
        .din_b  (gm_acc_wr_data),
        .dout_b ()
    );

    // ================================================================
    // Softmax Engine
    // ================================================================
    logic        sm_busy, sm_done;
    logic        sm_rd_en;
    logic [15:0] sm_rd_addr;
    logic [7:0]  sm_rd_data;
    logic        sm_wr_en;
    logic [15:0] sm_wr_addr;
    logic [7:0]  sm_wr_data;
    logic        sm_scr_wr_en;
    logic [15:0] sm_scr_wr_addr, sm_scr_wr_data;
    logic        sm_scr_rd_en;
    logic [15:0] sm_scr_rd_addr, sm_scr_rd_data;

    logic        gp_sm_cmd_valid;
    logic [15:0] gp_sm_src_base, gp_sm_dst_base, gp_sm_length;

    softmax_engine u_softmax (
        .clk             (clk),
        .rst_n           (rst_n),
        .cmd_valid       (gp_sm_cmd_valid),
        .cmd_ready       (),
        .length          (gp_sm_length),
        .src_base        (gp_sm_src_base),
        .dst_base        (gp_sm_dst_base),
        .scale_factor    (16'd256),  // 1.0 in Q8.8
        .causal_mask_en  (1'b0),
        .causal_limit    (16'd0),
        .sram_rd_en      (sm_rd_en),
        .sram_rd_addr    (sm_rd_addr),
        .sram_rd_data    (sm_rd_data),
        .sram_wr_en      (sm_wr_en),
        .sram_wr_addr    (sm_wr_addr),
        .sram_wr_data    (sm_wr_data),
        .scratch_wr_en   (sm_scr_wr_en),
        .scratch_wr_addr (sm_scr_wr_addr),
        .scratch_wr_data (sm_scr_wr_data),
        .scratch_rd_en   (sm_scr_rd_en),
        .scratch_rd_addr (sm_scr_rd_addr),
        .scratch_rd_data (sm_scr_rd_data),
        .busy            (sm_busy),
        .done            (sm_done)
    );

    // SCRATCH SRAM (16-bit x SCRATCH_DEPTH)
    sram_dp #(.DEPTH(SCRATCH_DEPTH), .WIDTH(16)) u_scratch_sram (
        .clk    (clk),
        .en_a   (sm_scr_rd_en),
        .we_a   (1'b0),
        .addr_a (sm_scr_rd_addr[SCR_AW-1:0]),
        .din_a  (16'd0),
        .dout_a (sm_scr_rd_data),
        .en_b   (sm_scr_wr_en),
        .we_b   (sm_scr_wr_en),
        .addr_b (sm_scr_wr_addr[SCR_AW-1:0]),
        .din_b  (sm_scr_wr_data),
        .dout_b ()
    );

    // ================================================================
    // Graph Pipeline (graph_top)
    // ================================================================
    logic        gp_ew_rd_en, gp_ew_wr_en;
    logic [SRAM0_AW-1:0] gp_ew_rd_addr, gp_ew_wr_addr;
    logic [7:0]  gp_ew_wr_data;
    logic        gp_ew_busy;

    logic        gp_dma_cmd_valid;
    logic [31:0] gp_dma_ddr_addr;
    logic [15:0] gp_dma_sram_addr, gp_dma_length;
    logic        gp_dma_direction, gp_dma_strided;
    logic [31:0] gp_dma_stride;
    logic [15:0] gp_dma_count, gp_dma_block_len;

    logic        gp_done, gp_busy;
    logic [31:0] gp_status;
    logic [15:0] gp_pc;
    logic [7:0]  gp_last_op;

    graph_top #(
        .PROG_SRAM_AW (PROG_AW),
        .SRAM0_AW     (SRAM0_AW)
    ) u_graph_top (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (start_pulse),
        .prog_len      (prog_len),
        .prog_rd_en    (prog_rd_en),
        .prog_rd_addr  (prog_rd_addr),
        .prog_rd_data  (prog_rd_data),
        .td_rd0_addr   (td_rd0_addr),
        .td_rd0_data   (td_rd0_data),
        .td_rd1_addr   (td_rd1_addr),
        .td_rd1_data   (td_rd1_data),
        .td_rd2_addr   (td_rd2_addr),
        .td_rd2_data   (td_rd2_data),
        .gm_cmd_valid  (gp_gm_cmd_valid),
        .gm_cmd_src0   (gp_gm_cmd_src0),
        .gm_cmd_src1   (gp_gm_cmd_src1),
        .gm_cmd_dst    (gp_gm_cmd_dst),
        .gm_cmd_M      (gp_gm_cmd_M),
        .gm_cmd_N      (gp_gm_cmd_N),
        .gm_cmd_K      (gp_gm_cmd_K),
        .gm_cmd_flags  (gp_gm_cmd_flags),
        .gm_cmd_imm    (gp_gm_cmd_imm),
        .gm_done       (gm_done),
        .sm_cmd_valid  (gp_sm_cmd_valid),
        .sm_src_base   (gp_sm_src_base),
        .sm_dst_base   (gp_sm_dst_base),
        .sm_length     (gp_sm_length),
        .sm_done       (sm_done),
        .dma_cmd_valid (gp_dma_cmd_valid),
        .dma_ddr_addr  (gp_dma_ddr_addr),
        .dma_sram_addr (gp_dma_sram_addr),
        .dma_length    (gp_dma_length),
        .dma_direction (gp_dma_direction),
        .dma_strided   (gp_dma_strided),
        .dma_stride    (gp_dma_stride),
        .dma_count     (gp_dma_count),
        .dma_block_len (gp_dma_block_len),
        .dma_done      (dma_done_internal),
        .ew_rd_en      (gp_ew_rd_en),
        .ew_rd_addr    (gp_ew_rd_addr),
        .ew_rd_data    (s0_a_dout),
        .ew_wr_en      (gp_ew_wr_en),
        .ew_wr_addr    (gp_ew_wr_addr),
        .ew_wr_data    (gp_ew_wr_data),
        .ew_busy       (gp_ew_busy),
        .graph_done    (gp_done),
        .graph_busy    (gp_busy),
        .graph_status  (gp_status),
        .graph_pc      (gp_pc),
        .graph_last_op (gp_last_op)
    );

    // ================================================================
    // DMA Shim - capture DMA commands for C++ TB handling
    // ================================================================
    logic        dma_active;
    logic        dma_captured_r;
    logic [31:0] dma_ddr_addr_r;
    logic [15:0] dma_sram_addr_r, dma_length_r;
    logic        dma_direction_r, dma_strided_r;
    logic [31:0] dma_stride_r;
    logic [15:0] dma_count_r, dma_block_len_r;
    logic        dma_done_internal;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_active      <= 1'b0;
            dma_captured_r  <= 1'b0;
            dma_ddr_addr_r  <= '0;
            dma_sram_addr_r <= '0;
            dma_length_r    <= '0;
            dma_direction_r <= 1'b0;
            dma_strided_r   <= 1'b0;
            dma_stride_r    <= '0;
            dma_count_r     <= '0;
            dma_block_len_r <= '0;
        end else begin
            dma_captured_r <= 1'b0;

            if (gp_dma_cmd_valid && !dma_active) begin
                dma_active      <= 1'b1;
                dma_captured_r  <= 1'b1;
                dma_ddr_addr_r  <= gp_dma_ddr_addr;
                dma_sram_addr_r <= gp_dma_sram_addr;
                dma_length_r    <= gp_dma_length;
                dma_direction_r <= gp_dma_direction;
                dma_strided_r   <= gp_dma_strided;
                dma_stride_r    <= gp_dma_stride;
                dma_count_r     <= gp_dma_count;
                dma_block_len_r <= gp_dma_block_len;
            end else if (dma_done_pulse && dma_active) begin
                dma_active <= 1'b0;
            end
        end
    end

    assign dma_done_internal = dma_done_pulse && dma_active;
    assign dma_cmd_captured  = dma_captured_r;
    assign dma_ddr_addr      = dma_ddr_addr_r;
    assign dma_sram_addr     = dma_sram_addr_r;
    assign dma_length        = dma_length_r;
    assign dma_direction     = dma_direction_r;
    assign dma_strided        = dma_strided_r;
    assign dma_stride        = dma_stride_r;
    assign dma_count         = dma_count_r;
    assign dma_block_len     = dma_block_len_r;

    // ================================================================
    // DATA SRAM0 (8-bit x SRAM0_DEPTH)
    // Mux priority: graph_dispatch_ew > gemm > softmax > TB
    // ================================================================
    logic                 s0_a_en;
    logic [SRAM0_AW-1:0] s0_a_addr;
    logic [7:0]           s0_a_dout;
    logic                 s0_b_en, s0_b_we;
    logic [SRAM0_AW-1:0] s0_b_addr;
    logic [7:0]           s0_b_din;

    // SRAM0 read mux (port A)
    always_comb begin
        if (gp_ew_busy && gp_ew_rd_en) begin
            s0_a_en   = 1'b1;
            s0_a_addr = gp_ew_rd_addr;
        end else if (gm_busy) begin
            s0_a_en   = gm_rd_en;
            s0_a_addr = gm_rd_addr[SRAM0_AW-1:0];
        end else if (sm_busy) begin
            s0_a_en   = sm_rd_en;
            s0_a_addr = sm_rd_addr[SRAM0_AW-1:0];
        end else begin
            s0_a_en   = tb_sram0_rd_en;
            s0_a_addr = tb_sram0_rd_addr;
        end
    end

    // SRAM0 write mux (port B)
    always_comb begin
        if (gp_ew_busy && gp_ew_wr_en) begin
            s0_b_en   = 1'b1;
            s0_b_we   = 1'b1;
            s0_b_addr = gp_ew_wr_addr;
            s0_b_din  = gp_ew_wr_data;
        end else if (gm_busy) begin
            s0_b_en   = gm_wr_en;
            s0_b_we   = gm_wr_en;
            s0_b_addr = gm_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = gm_wr_data;
        end else if (sm_busy) begin
            s0_b_en   = sm_wr_en;
            s0_b_we   = sm_wr_en;
            s0_b_addr = sm_wr_addr[SRAM0_AW-1:0];
            s0_b_din  = sm_wr_data;
        end else begin
            s0_b_en   = tb_sram0_wr_en;
            s0_b_we   = tb_sram0_wr_en;
            s0_b_addr = tb_sram0_wr_addr;
            s0_b_din  = tb_sram0_wr_data;
        end
    end

    sram_dp #(.DEPTH(SRAM0_DEPTH), .WIDTH(8)) u_sram0 (
        .clk    (clk),
        .en_a   (s0_a_en),
        .we_a   (1'b0),
        .addr_a (s0_a_addr),
        .din_a  (8'd0),
        .dout_a (s0_a_dout),
        .en_b   (s0_b_en),
        .we_b   (s0_b_we),
        .addr_b (s0_b_addr),
        .din_b  (s0_b_din),
        .dout_b ()
    );

    // Broadcast SRAM0 read data
    assign sm_rd_data       = s0_a_dout;
    assign tb_sram0_rd_data = s0_a_dout;

    // ================================================================
    // Output assignments
    // ================================================================
    assign graph_done    = gp_done;
    assign graph_busy    = gp_busy;
    assign graph_status  = gp_status;
    assign graph_pc      = gp_pc;
    assign graph_last_op = gp_last_op;

endmodule

`default_nettype wire
