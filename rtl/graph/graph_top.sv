// =============================================================================
// Graph Top - Wrapper for Graph Mode pipeline
// Connects: graph_fetch → graph_decode → graph_dispatch
// Exposes engine/DMA/SRAM/debug ports to parent sim wrapper
// =============================================================================
`default_nettype none

module graph_top
    import npu_pkg::*;
    import graph_isa_pkg::*;
#(
    parameter int PROG_SRAM_AW = 10,
    parameter int SRAM0_AW     = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Control
    input  wire                     start,
    input  wire  [15:0]             prog_len,

    // Program SRAM read port
    output logic                    prog_rd_en,
    output logic [PROG_SRAM_AW-1:0] prog_rd_addr,
    input  wire  [127:0]           prog_rd_data,

    // Tensor table read ports (directly to tensor_table)
    output logic [7:0]              td_rd0_addr,
    input  wire  [255:0]            td_rd0_data,
    output logic [7:0]              td_rd1_addr,
    input  wire  [255:0]            td_rd1_data,
    output logic [7:0]              td_rd2_addr,
    input  wire  [255:0]            td_rd2_data,

    // GEMM engine command
    output logic                    gm_cmd_valid,
    output logic [15:0]             gm_cmd_src0,
    output logic [15:0]             gm_cmd_src1,
    output logic [15:0]             gm_cmd_dst,
    output logic [15:0]             gm_cmd_M,
    output logic [15:0]             gm_cmd_N,
    output logic [15:0]             gm_cmd_K,
    output logic [7:0]              gm_cmd_flags,
    output logic [15:0]             gm_cmd_imm,
    input  wire                     gm_done,

    // Softmax engine command
    output logic                    sm_cmd_valid,
    output logic [15:0]             sm_src_base,
    output logic [15:0]             sm_dst_base,
    output logic [15:0]             sm_length,
    input  wire                     sm_done,

    // DMA shim
    output logic                    dma_cmd_valid,
    output logic [31:0]             dma_ddr_addr,
    output logic [15:0]             dma_sram_addr,
    output logic [15:0]             dma_length,
    output logic                    dma_direction,
    output logic                    dma_strided,
    output logic [31:0]             dma_stride,
    output logic [15:0]             dma_count,
    output logic [15:0]             dma_block_len,
    input  wire                     dma_done,

    // SRAM0 element-wise ports
    output logic                    ew_rd_en,
    output logic [SRAM0_AW-1:0]    ew_rd_addr,
    input  wire  [7:0]             ew_rd_data,
    output logic                    ew_wr_en,
    output logic [SRAM0_AW-1:0]    ew_wr_addr,
    output logic [7:0]             ew_wr_data,

    // EW busy (for SRAM mux priority)
    output logic                    ew_busy,

    // Status / debug
    output logic                    graph_done,
    output logic                    graph_busy,
    output logic [31:0]             graph_status,
    output logic [15:0]             graph_pc,
    output logic [7:0]              graph_last_op
);

    // =========================================================================
    // Internal wires
    // =========================================================================
    // Fetch → Decode
    logic        fetch_instr_valid;
    logic [127:0] fetch_instr_data;
    logic        fetch_instr_ready;
    logic [15:0] fetch_pc;
    logic        fetch_done;
    logic        fetch_busy;

    // Decode → Dispatch
    graph_instr_t decoded_instr;
    logic         decoded_valid;

    // Dispatch status
    logic        disp_done, disp_busy, disp_error;
    logic [7:0]  disp_error_code;
    logic [15:0] disp_pc;
    logic [7:0]  disp_last_op;

    // =========================================================================
    // Graph Fetch
    // =========================================================================
    graph_fetch #(
        .SRAM_ADDR_W (PROG_SRAM_AW)
    ) u_fetch (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .prog_len    (prog_len),
        .rd_en       (prog_rd_en),
        .rd_addr     (prog_rd_addr),
        .rd_data     (prog_rd_data),
        .instr_valid (fetch_instr_valid),
        .instr_data  (fetch_instr_data),
        .instr_ready (fetch_instr_ready),
        .pc          (fetch_pc),
        .done        (fetch_done),
        .busy        (fetch_busy)
    );

    // =========================================================================
    // Graph Decode (combinational)
    // =========================================================================
    graph_decode u_decode (
        .raw_instr  (fetch_instr_data),
        .valid_in   (fetch_instr_valid),
        .instr_out  (decoded_instr),
        .valid_out  (decoded_valid)
    );

    // =========================================================================
    // Graph Dispatch FSM
    // =========================================================================
    graph_dispatch #(
        .SRAM0_AW (SRAM0_AW)
    ) u_dispatch (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (start),
        .instr_valid   (decoded_valid),
        .instr_in      (decoded_instr),
        .instr_ready   (fetch_instr_ready),
        .td_rd0_addr   (td_rd0_addr),
        .td_rd0_data   (td_rd0_data),
        .td_rd1_addr   (td_rd1_addr),
        .td_rd1_data   (td_rd1_data),
        .td_rd2_addr   (td_rd2_addr),
        .td_rd2_data   (td_rd2_data),
        .gm_cmd_valid  (gm_cmd_valid),
        .gm_cmd_src0   (gm_cmd_src0),
        .gm_cmd_src1   (gm_cmd_src1),
        .gm_cmd_dst    (gm_cmd_dst),
        .gm_cmd_M      (gm_cmd_M),
        .gm_cmd_N      (gm_cmd_N),
        .gm_cmd_K      (gm_cmd_K),
        .gm_cmd_flags  (gm_cmd_flags),
        .gm_cmd_imm    (gm_cmd_imm),
        .gm_done       (gm_done),
        .sm_cmd_valid  (sm_cmd_valid),
        .sm_src_base   (sm_src_base),
        .sm_dst_base   (sm_dst_base),
        .sm_length     (sm_length),
        .sm_done       (sm_done),
        .dma_cmd_valid (dma_cmd_valid),
        .dma_ddr_addr  (dma_ddr_addr),
        .dma_sram_addr (dma_sram_addr),
        .dma_length    (dma_length),
        .dma_direction (dma_direction),
        .dma_strided   (dma_strided),
        .dma_stride    (dma_stride),
        .dma_count     (dma_count),
        .dma_block_len (dma_block_len),
        .dma_done      (dma_done),
        .ew_rd_en      (ew_rd_en),
        .ew_rd_addr    (ew_rd_addr),
        .ew_rd_data    (ew_rd_data),
        .ew_wr_en      (ew_wr_en),
        .ew_wr_addr    (ew_wr_addr),
        .ew_wr_data    (ew_wr_data),
        .graph_done    (disp_done),
        .graph_busy    (disp_busy),
        .graph_error   (disp_error),
        .error_code    (disp_error_code),
        .dbg_pc        (disp_pc),
        .dbg_last_op   (disp_last_op)
    );

    // =========================================================================
    // Status aggregation
    // =========================================================================
    // Done latching: hold done until next start
    logic done_latch;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done_latch <= 1'b0;
        else if (start)
            done_latch <= 1'b0;
        else if (disp_done || fetch_done)
            done_latch <= 1'b1;
    end

    assign graph_done   = done_latch;
    assign graph_busy   = disp_busy || fetch_busy;
    // EW busy only when dispatch is actively doing element-wise SRAM access
    assign ew_busy      = ew_rd_en || ew_wr_en;
    assign graph_pc     = disp_pc;
    assign graph_last_op = disp_last_op;

    // Status register: [2]=error, [1]=busy, [0]=done
    assign graph_status = {24'd0, disp_error_code[4:0], disp_error, graph_busy, done_latch};

endmodule

`default_nettype wire
