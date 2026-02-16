// =============================================================================
// Graph Dispatch FSM (Phase 4)
// Main execution engine for Graph Mode. Supports SERIAL and OVERLAP scheduling.
//   - Tensor descriptor lookup
//   - Engine programming (gemm_ctrl, softmax_engine, reduce/math/gather/slice/concat/
//     avgpool/maxpool/pad/resize/cast)
//   - Internal element-wise loops (EW_ADD/MUL/SUB/MIN/MAX, RELU)
//   - DMA shim interface + PREFETCH
//   - Tensor scoreboard for OVERLAP mode dependency tracking
//   - Error detection (bad opcode, shape mismatch, timeout)
//   - Performance counters (including overlap, stall, ln)
// =============================================================================
`default_nettype none

module graph_dispatch
    import npu_pkg::*;
    import graph_isa_pkg::*;
#(
    parameter int SRAM0_AW = 16,
    parameter int MAX_CYCLES_PER_OP = 1000000
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Control
    input  wire                     start,
    input  wire                     scheduler_mode,  // 0=SERIAL, 1=OVERLAP

    // Instruction input from fetch/decode
    input  wire                     instr_valid,
    input  graph_instr_t            instr_in,
    output logic                    instr_ready,

    // Tensor table read ports
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
    output logic [7:0]              gm_cmd_dtype,
    input  wire                     gm_done,

    // Softmax engine command
    output logic                    sm_cmd_valid,
    output logic [15:0]             sm_src_base,
    output logic [15:0]             sm_dst_base,
    output logic [15:0]             sm_length,
    output logic [7:0]              sm_cmd_dtype,
    input  wire                     sm_done,

    // Reduce engine command
    output logic                    re_cmd_valid,
    output logic [7:0]              re_cmd_opcode,
    output logic [15:0]             re_cmd_src_base,
    output logic [15:0]             re_cmd_dst_base,
    output logic [15:0]             re_cmd_reduce_dim,
    output logic [15:0]             re_cmd_outer_count,
    input  wire                     re_done,

    // Math engine command
    output logic                    me_cmd_valid,
    output logic [7:0]              me_cmd_opcode,
    output logic [15:0]             me_cmd_src_base,
    output logic [15:0]             me_cmd_dst_base,
    output logic [15:0]             me_cmd_length,
    output logic [7:0]              me_cmd_dtype,
    input  wire                     me_done,

    // Gather engine command
    output logic                    ga_cmd_valid,
    output logic [15:0]             ga_cmd_src_base,
    output logic [15:0]             ga_cmd_idx_base,
    output logic [15:0]             ga_cmd_dst_base,
    output logic [15:0]             ga_cmd_num_indices,
    output logic [15:0]             ga_cmd_row_size,
    output logic [15:0]             ga_cmd_num_rows,
    input  wire                     ga_done,

    // Slice engine command
    output logic                    sl_cmd_valid,
    output logic [15:0]             sl_cmd_src_base,
    output logic [15:0]             sl_cmd_dst_base,
    output logic [15:0]             sl_cmd_src_row_len,
    output logic [15:0]             sl_cmd_dst_row_len,
    output logic [15:0]             sl_cmd_start_offset,
    output logic [15:0]             sl_cmd_num_rows,
    input  wire                     sl_done,

    // Concat engine command
    output logic                    ct_cmd_valid,
    output logic [15:0]             ct_cmd_src0_base,
    output logic [15:0]             ct_cmd_src1_base,
    output logic [15:0]             ct_cmd_dst_base,
    output logic [15:0]             ct_cmd_src0_row_len,
    output logic [15:0]             ct_cmd_src1_row_len,
    output logic [15:0]             ct_cmd_num_rows,
    input  wire                     ct_done,

    // AvgPool2D engine command
    output logic                    ap_cmd_valid,
    output logic [15:0]             ap_cmd_src_base,
    output logic [15:0]             ap_cmd_dst_base,
    output logic [15:0]             ap_cmd_C,
    output logic [15:0]             ap_cmd_H,
    output logic [15:0]             ap_cmd_W,
    output logic [7:0]              ap_cmd_kh,
    output logic [7:0]              ap_cmd_kw,
    output logic [7:0]              ap_cmd_sh,
    output logic [7:0]              ap_cmd_sw,
    input  wire                     ap_done,

    // MaxPool2D engine command
    output logic                    mp_cmd_valid,
    output logic [15:0]             mp_cmd_src_base,
    output logic [15:0]             mp_cmd_dst_base,
    output logic [15:0]             mp_cmd_C,
    output logic [15:0]             mp_cmd_H,
    output logic [15:0]             mp_cmd_W,
    output logic [7:0]              mp_cmd_kh,
    output logic [7:0]              mp_cmd_kw,
    output logic [7:0]              mp_cmd_sh,
    output logic [7:0]              mp_cmd_sw,
    input  wire                     mp_done,

    // Pad engine command
    output logic                    pd_cmd_valid,
    output logic [15:0]             pd_cmd_src_base,
    output logic [15:0]             pd_cmd_dst_base,
    output logic [15:0]             pd_cmd_C,
    output logic [15:0]             pd_cmd_H,
    output logic [15:0]             pd_cmd_W,
    output logic [7:0]              pd_cmd_pad_top,
    output logic [7:0]              pd_cmd_pad_bottom,
    output logic [7:0]              pd_cmd_pad_left,
    output logic [7:0]              pd_cmd_pad_right,
    input  wire                     pd_done,

    // Resize nearest engine command
    output logic                    rz_cmd_valid,
    output logic [15:0]             rz_cmd_src_base,
    output logic [15:0]             rz_cmd_dst_base,
    output logic [15:0]             rz_cmd_C,
    output logic [15:0]             rz_cmd_in_H,
    output logic [15:0]             rz_cmd_in_W,
    output logic [15:0]             rz_cmd_out_H,
    output logic [15:0]             rz_cmd_out_W,
    input  wire                     rz_done,

    // Cast engine command
    output logic                    ca_cmd_valid,
    output logic [15:0]             ca_cmd_src_base,
    output logic [15:0]             ca_cmd_dst_base,
    output logic [15:0]             ca_cmd_length,
    output logic [7:0]              ca_cmd_src_dtype,
    output logic [7:0]              ca_cmd_dst_dtype,
    input  wire                     ca_done,

    // DMA shim interface
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

    // SRAM0 element-wise read port
    output logic                    ew_rd_en,
    output logic [SRAM0_AW-1:0]    ew_rd_addr,
    input  wire  [7:0]             ew_rd_data,

    // SRAM0 element-wise write port
    output logic                    ew_wr_en,
    output logic [SRAM0_AW-1:0]    ew_wr_addr,
    output logic [7:0]             ew_wr_data,

    // Performance counters
    output logic [31:0]             perf_total_cycles,
    output logic [31:0]             perf_gemm_cycles,
    output logic [31:0]             perf_softmax_cycles,
    output logic [31:0]             perf_dma_cycles,
    output logic [31:0]             perf_reduce_cycles,
    output logic [31:0]             perf_math_cycles,
    output logic [31:0]             perf_gather_cycles,
    output logic [31:0]             perf_slice_cycles,
    output logic [31:0]             perf_concat_cycles,
    output logic [31:0]             perf_avgpool_cycles,
    output logic [31:0]             perf_ew_cycles,
    output logic [31:0]             perf_overlap_cycles,
    output logic [31:0]             perf_stall_cycles,

    // Status / debug
    output logic                    graph_done,
    output logic                    graph_busy,
    output logic                    graph_error,
    output logic [7:0]              error_code,
    output logic [15:0]             dbg_pc,
    output logic [7:0]              dbg_last_op
);

    // =========================================================================
    // FSM states
    // =========================================================================
    typedef enum logic [4:0] {
        GD_IDLE,
        GD_FETCH_WAIT,
        GD_DECODE,
        GD_TDESC0,
        GD_TDESC1,
        GD_TDESC2,
        GD_EXEC_DMA,
        GD_EXEC_GEMM,
        GD_EXEC_EW_INIT,
        GD_EXEC_EW_RD_A,
        GD_EXEC_EW_RD_B,
        GD_EXEC_EW_COMPUTE,
        GD_EXEC_RELU_INIT,
        GD_EXEC_RELU_RD,
        GD_EXEC_RELU_WR,
        GD_EXEC_SOFTMAX,
        GD_EXEC_REDUCE,
        GD_EXEC_MATH,
        GD_EXEC_GATHER,
        GD_EXEC_SLICE,
        GD_EXEC_CONCAT,
        GD_EXEC_AVGPOOL,
        GD_EXEC_MAXPOOL,
        GD_EXEC_PAD,
        GD_EXEC_RESIZE,
        GD_EXEC_CAST,
        GD_WAIT_DONE,
        GD_NEXT,
        GD_DONE,
        GD_ERROR
    } gd_state_t;

    gd_state_t state, state_next;

    // =========================================================================
    // Registered instruction and descriptor data
    // =========================================================================
    graph_instr_t  r_instr;
    tensor_desc_t  r_td0, r_td1, r_td2;

    // Element-wise loop state
    logic [15:0] ew_idx;
    logic [15:0] ew_count;
    logic [7:0]  ew_val_a;
    logic [7:0]  ew_val_b;

    // Which engine we're waiting on
    typedef enum logic [3:0] {
        WAIT_NONE,
        WAIT_GEMM,
        WAIT_SOFTMAX,
        WAIT_DMA,
        WAIT_REDUCE,
        WAIT_MATH,
        WAIT_GATHER,
        WAIT_SLICE,
        WAIT_CONCAT,
        WAIT_AVGPOOL,
        WAIT_MAXPOOL,
        WAIT_PAD,
        WAIT_RESIZE,
        WAIT_CAST
    } wait_target_t;

    wait_target_t wait_target;

    // Timeout counter
    logic [31:0] timeout_cnt;

    // =========================================================================
    // Tensor scoreboard (Phase 4)
    // 0=INVALID, 1=PRODUCING, 2=READY
    // =========================================================================
    logic [1:0] tensor_state [0:255];

    // Engine in-flight trackers
    logic dma_in_flight;   logic [7:0] dma_dst_tid;
    logic gemm_in_flight;  logic [7:0] gemm_dst_tid;
    logic sm_in_flight;    logic [7:0] sm_dst_tid;
    logic re_in_flight;    logic [7:0] re_dst_tid;
    logic me_in_flight;    logic [7:0] me_dst_tid;
    logic ga_in_flight;    logic [7:0] ga_dst_tid;
    logic sl_in_flight;    logic [7:0] sl_dst_tid;
    logic ct_in_flight;    logic [7:0] ct_dst_tid;
    logic ap_in_flight;    logic [7:0] ap_dst_tid;
    logic mp_in_flight;    logic [7:0] mp_dst_tid;
    logic pd_in_flight;    logic [7:0] pd_dst_tid;
    logic rz_in_flight;    logic [7:0] rz_dst_tid;
    logic ca_in_flight;    logic [7:0] ca_dst_tid;

    // Scheduler mode register
    logic r_scheduler_mode;

    // =========================================================================
    // State register
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= GD_IDLE;
        else
            state <= state_next;
    end

    // =========================================================================
    // Next-state logic
    // =========================================================================
    always_comb begin
        state_next = state;

        case (state)
            GD_IDLE: begin
                if (start)
                    state_next = GD_FETCH_WAIT;
            end

            GD_FETCH_WAIT: begin
                if (instr_valid)
                    state_next = GD_DECODE;
            end

            GD_DECODE: begin
                if (r_instr.opcode == OP_G_END) begin
                    // In OVERLAP mode: wait for all in-flight engines before ending
                    if (r_scheduler_mode && (dma_in_flight || gemm_in_flight || sm_in_flight ||
                        re_in_flight || me_in_flight || ga_in_flight || sl_in_flight ||
                        ct_in_flight || ap_in_flight || mp_in_flight || pd_in_flight ||
                        rz_in_flight || ca_in_flight))
                        state_next = GD_DECODE; // stall
                    else
                        state_next = GD_DONE;
                end else if (r_instr.opcode == OP_G_BARRIER) begin
                    // Wait for all in-flight to complete
                    if (dma_in_flight || gemm_in_flight || sm_in_flight ||
                        re_in_flight || me_in_flight || ga_in_flight || sl_in_flight ||
                        ct_in_flight || ap_in_flight || mp_in_flight || pd_in_flight ||
                        rz_in_flight || ca_in_flight)
                        state_next = GD_DECODE; // stall
                    else
                        state_next = GD_NEXT;
                end else if (r_instr.opcode == OP_G_PREFETCH) begin
                    // PREFETCH: if DMA idle, issue; otherwise NOP
                    if (!dma_in_flight)
                        state_next = GD_TDESC0;
                    else
                        state_next = GD_NEXT; // silently skip
                end else begin
                    state_next = GD_TDESC0;
                end
            end

            GD_TDESC0: state_next = GD_TDESC1;
            GD_TDESC1: state_next = GD_TDESC2;

            GD_TDESC2: begin
                case (r_instr.opcode)
                    OP_G_DMA_LOAD,
                    OP_G_DMA_STORE,
                    OP_G_DMA_STRIDED:  state_next = GD_EXEC_DMA;
                    OP_G_PREFETCH:     state_next = GD_EXEC_DMA; // treat like DMA_LOAD
                    OP_G_GEMM:         begin
                        if (r_instr.flags[0]
                            ? (r_td0.shape1 != r_td1.shape1)
                            : (r_td0.shape1 != r_td1.shape0))
                            state_next = GD_ERROR;
                        else
                            state_next = GD_EXEC_GEMM;
                    end
                    OP_G_EW_ADD,
                    OP_G_EW_MUL,
                    OP_G_EW_SUB,
                    OP_G_EW_MIN,
                    OP_G_EW_MAX:       state_next = GD_EXEC_EW_INIT;
                    OP_G_RELU:         state_next = GD_EXEC_RELU_INIT;
                    OP_G_SOFTMAX:      state_next = GD_EXEC_SOFTMAX;
                    OP_G_REDUCE_SUM,
                    OP_G_REDUCE_MAX,
                    OP_G_REDUCE_MEAN:  state_next = GD_EXEC_REDUCE;
                    OP_G_EXP,
                    OP_G_LOG,
                    OP_G_SQRT,
                    OP_G_RSQRT:        state_next = GD_EXEC_MATH;
                    OP_G_GATHER:       state_next = GD_EXEC_GATHER;
                    OP_G_SLICE:        state_next = GD_EXEC_SLICE;
                    OP_G_CONCAT:       state_next = GD_EXEC_CONCAT;
                    OP_G_AVGPOOL2D:    state_next = GD_EXEC_AVGPOOL;
                    OP_G_MAXPOOL2D:    state_next = GD_EXEC_MAXPOOL;
                    OP_G_PAD:          state_next = GD_EXEC_PAD;
                    OP_G_RESIZE_NEAREST: state_next = GD_EXEC_RESIZE;
                    OP_G_CAST:         state_next = GD_EXEC_CAST;
                    OP_G_BARRIER:      state_next = GD_NEXT;
                    default:           state_next = GD_ERROR;
                endcase
            end

            GD_EXEC_DMA:     state_next = GD_WAIT_DONE;
            GD_EXEC_GEMM:    state_next = GD_WAIT_DONE;
            GD_EXEC_SOFTMAX: state_next = GD_WAIT_DONE;
            GD_EXEC_REDUCE:  state_next = GD_WAIT_DONE;
            GD_EXEC_MATH:    state_next = GD_WAIT_DONE;
            GD_EXEC_GATHER:  state_next = GD_WAIT_DONE;
            GD_EXEC_SLICE:   state_next = GD_WAIT_DONE;
            GD_EXEC_CONCAT:  state_next = GD_WAIT_DONE;
            GD_EXEC_AVGPOOL: state_next = GD_WAIT_DONE;
            GD_EXEC_MAXPOOL: state_next = GD_WAIT_DONE;
            GD_EXEC_PAD:     state_next = GD_WAIT_DONE;
            GD_EXEC_RESIZE:  state_next = GD_WAIT_DONE;
            GD_EXEC_CAST:    state_next = GD_WAIT_DONE;

            GD_EXEC_EW_INIT: begin
                if (r_td0.size_bytes == 16'd0)
                    state_next = GD_NEXT;
                else
                    state_next = GD_EXEC_EW_RD_A;
            end

            GD_EXEC_EW_RD_A: state_next = GD_EXEC_EW_RD_B;
            GD_EXEC_EW_RD_B: state_next = GD_EXEC_EW_COMPUTE;

            GD_EXEC_EW_COMPUTE: begin
                if (ew_idx >= ew_count - 16'd1)
                    state_next = GD_NEXT;
                else
                    state_next = GD_EXEC_EW_RD_A;
            end

            GD_EXEC_RELU_INIT: begin
                if (r_td0.size_bytes == 16'd0)
                    state_next = GD_NEXT;
                else
                    state_next = GD_EXEC_RELU_RD;
            end

            GD_EXEC_RELU_RD: state_next = GD_EXEC_RELU_WR;

            GD_EXEC_RELU_WR: begin
                if (ew_idx >= ew_count - 16'd1)
                    state_next = GD_NEXT;
                else
                    state_next = GD_EXEC_RELU_RD;
            end

            GD_WAIT_DONE: begin
                if (timeout_cnt >= MAX_CYCLES_PER_OP[31:0])
                    state_next = GD_ERROR;
                else begin
                    case (wait_target)
                        WAIT_GEMM:    if (gm_done)  state_next = GD_NEXT;
                        WAIT_SOFTMAX: if (sm_done)  state_next = GD_NEXT;
                        WAIT_DMA:     if (dma_done) state_next = GD_NEXT;
                        WAIT_REDUCE:  if (re_done)  state_next = GD_NEXT;
                        WAIT_MATH:    if (me_done)  state_next = GD_NEXT;
                        WAIT_GATHER:  if (ga_done)  state_next = GD_NEXT;
                        WAIT_SLICE:   if (sl_done)  state_next = GD_NEXT;
                        WAIT_CONCAT:  if (ct_done)  state_next = GD_NEXT;
                        WAIT_AVGPOOL: if (ap_done)  state_next = GD_NEXT;
                        WAIT_MAXPOOL: if (mp_done)  state_next = GD_NEXT;
                        WAIT_PAD:     if (pd_done)  state_next = GD_NEXT;
                        WAIT_RESIZE:  if (rz_done)  state_next = GD_NEXT;
                        WAIT_CAST:    if (ca_done)  state_next = GD_NEXT;
                        default:      state_next = GD_NEXT;
                    endcase
                end
            end

            GD_NEXT: state_next = GD_FETCH_WAIT;

            GD_DONE: state_next = GD_IDLE;

            GD_ERROR: state_next = GD_ERROR;

            default: state_next = GD_IDLE;
        endcase
    end

    // =========================================================================
    // Output logic
    // =========================================================================
    always_comb begin
        // Defaults
        instr_ready   = 1'b0;
        gm_cmd_valid  = 1'b0;
        gm_cmd_src0   = '0;
        gm_cmd_src1   = '0;
        gm_cmd_dst    = '0;
        gm_cmd_M      = '0;
        gm_cmd_N      = '0;
        gm_cmd_K      = '0;
        gm_cmd_flags  = '0;
        gm_cmd_imm    = '0;
        gm_cmd_dtype  = '0;
        sm_cmd_valid  = 1'b0;
        sm_src_base   = '0;
        sm_dst_base   = '0;
        sm_length     = '0;
        sm_cmd_dtype  = '0;
        re_cmd_valid      = 1'b0;
        re_cmd_opcode     = '0;
        re_cmd_src_base   = '0;
        re_cmd_dst_base   = '0;
        re_cmd_reduce_dim = '0;
        re_cmd_outer_count = '0;
        me_cmd_valid  = 1'b0;
        me_cmd_opcode = '0;
        me_cmd_src_base = '0;
        me_cmd_dst_base = '0;
        me_cmd_length = '0;
        me_cmd_dtype  = '0;
        ga_cmd_valid  = 1'b0;
        ga_cmd_src_base   = '0;
        ga_cmd_idx_base   = '0;
        ga_cmd_dst_base   = '0;
        ga_cmd_num_indices = '0;
        ga_cmd_row_size   = '0;
        ga_cmd_num_rows   = '0;
        sl_cmd_valid  = 1'b0;
        sl_cmd_src_base     = '0;
        sl_cmd_dst_base     = '0;
        sl_cmd_src_row_len  = '0;
        sl_cmd_dst_row_len  = '0;
        sl_cmd_start_offset = '0;
        sl_cmd_num_rows     = '0;
        ct_cmd_valid  = 1'b0;
        ct_cmd_src0_base    = '0;
        ct_cmd_src1_base    = '0;
        ct_cmd_dst_base     = '0;
        ct_cmd_src0_row_len = '0;
        ct_cmd_src1_row_len = '0;
        ct_cmd_num_rows     = '0;
        ap_cmd_valid      = 1'b0;
        ap_cmd_src_base   = '0;
        ap_cmd_dst_base   = '0;
        ap_cmd_C          = '0;
        ap_cmd_H          = '0;
        ap_cmd_W          = '0;
        ap_cmd_kh         = '0;
        ap_cmd_kw         = '0;
        ap_cmd_sh         = '0;
        ap_cmd_sw         = '0;
        mp_cmd_valid      = 1'b0;
        mp_cmd_src_base   = '0;
        mp_cmd_dst_base   = '0;
        mp_cmd_C          = '0;
        mp_cmd_H          = '0;
        mp_cmd_W          = '0;
        mp_cmd_kh         = '0;
        mp_cmd_kw         = '0;
        mp_cmd_sh         = '0;
        mp_cmd_sw         = '0;
        pd_cmd_valid      = 1'b0;
        pd_cmd_src_base   = '0;
        pd_cmd_dst_base   = '0;
        pd_cmd_C          = '0;
        pd_cmd_H          = '0;
        pd_cmd_W          = '0;
        pd_cmd_pad_top    = '0;
        pd_cmd_pad_bottom = '0;
        pd_cmd_pad_left   = '0;
        pd_cmd_pad_right  = '0;
        rz_cmd_valid      = 1'b0;
        rz_cmd_src_base   = '0;
        rz_cmd_dst_base   = '0;
        rz_cmd_C          = '0;
        rz_cmd_in_H       = '0;
        rz_cmd_in_W       = '0;
        rz_cmd_out_H      = '0;
        rz_cmd_out_W      = '0;
        ca_cmd_valid      = 1'b0;
        ca_cmd_src_base   = '0;
        ca_cmd_dst_base   = '0;
        ca_cmd_length     = '0;
        ca_cmd_src_dtype  = '0;
        ca_cmd_dst_dtype  = '0;
        dma_cmd_valid = 1'b0;
        dma_ddr_addr  = '0;
        dma_sram_addr = '0;
        dma_length    = '0;
        dma_direction = 1'b0;
        dma_strided   = 1'b0;
        dma_stride    = '0;
        dma_count     = '0;
        dma_block_len = '0;
        ew_rd_en      = 1'b0;
        ew_rd_addr    = '0;
        ew_wr_en      = 1'b0;
        ew_wr_addr    = '0;
        ew_wr_data    = '0;
        td_rd0_addr   = r_instr.src0;
        td_rd1_addr   = r_instr.src1;
        td_rd2_addr   = r_instr.dst;
        graph_done    = 1'b0;
        graph_busy    = (state != GD_IDLE);
        graph_error   = (state == GD_ERROR);

        case (state)
            GD_IDLE: begin
                graph_busy = 1'b0;
            end

            GD_FETCH_WAIT: begin
                instr_ready = 1'b1;
            end

            GD_TDESC0: begin
                td_rd0_addr = r_instr.src0;
                td_rd1_addr = r_instr.src1;
                td_rd2_addr = r_instr.dst;
            end

            GD_EXEC_DMA: begin
                dma_cmd_valid = 1'b1;
                if (r_instr.opcode == OP_G_DMA_LOAD || r_instr.opcode == OP_G_PREFETCH) begin
                    dma_ddr_addr  = r_td0.ddr_addr;
                    dma_sram_addr = r_td0.sram_addr;
                    dma_length    = r_td0.size_bytes;
                    dma_direction = 1'b0;
                end else if (r_instr.opcode == OP_G_DMA_STORE) begin
                    dma_ddr_addr  = r_td0.ddr_addr;
                    dma_sram_addr = r_td0.sram_addr;
                    dma_length    = r_td0.size_bytes;
                    dma_direction = 1'b1;
                end else begin
                    dma_ddr_addr  = r_td0.ddr_addr;
                    dma_sram_addr = r_td0.sram_addr;
                    dma_length    = r_td0.size_bytes;
                    dma_direction = (r_instr.flags[0]) ? 1'b1 : 1'b0;
                    dma_strided   = 1'b1;
                    dma_stride    = r_instr.imm1;
                    dma_count     = r_instr.imm0;
                    dma_block_len = r_instr.imm2[15:0];
                end
            end

            GD_EXEC_GEMM: begin
                gm_cmd_valid = 1'b1;
                gm_cmd_src0  = r_td0.sram_addr;
                gm_cmd_src1  = r_td1.sram_addr;
                gm_cmd_dst   = r_td2.sram_addr;
                gm_cmd_M     = r_td0.shape0;
                gm_cmd_N     = r_instr.flags[0] ? r_td1.shape0 : r_td1.shape1;
                gm_cmd_K     = r_td0.shape1;
                gm_cmd_flags = r_instr.flags;
                gm_cmd_imm   = r_instr.imm0;
                gm_cmd_dtype = r_td0.dtype;
            end

            GD_EXEC_SOFTMAX: begin
                sm_cmd_valid = 1'b1;
                sm_src_base  = r_td0.sram_addr;
                sm_dst_base  = r_td2.sram_addr;
                sm_length    = r_td0.size_bytes;
                sm_cmd_dtype = r_td0.dtype;
            end

            GD_EXEC_REDUCE: begin
                re_cmd_valid      = 1'b1;
                re_cmd_opcode     = r_instr.opcode;
                re_cmd_src_base   = r_td0.sram_addr;
                re_cmd_dst_base   = r_td2.sram_addr;
                re_cmd_reduce_dim = r_instr.imm0;
                re_cmd_outer_count = r_instr.imm1[15:0];
            end

            GD_EXEC_MATH: begin
                me_cmd_valid  = 1'b1;
                me_cmd_opcode = r_instr.opcode;
                me_cmd_src_base = r_td0.sram_addr;
                me_cmd_dst_base = r_td2.sram_addr;
                me_cmd_length = r_td0.size_bytes;
                me_cmd_dtype  = r_td0.dtype;
            end

            GD_EXEC_GATHER: begin
                ga_cmd_valid       = 1'b1;
                ga_cmd_src_base    = r_td0.sram_addr;
                ga_cmd_idx_base    = r_td1.sram_addr;
                ga_cmd_dst_base    = r_td2.sram_addr;
                ga_cmd_row_size    = r_instr.imm0;
                ga_cmd_num_rows    = r_instr.imm1[15:0];
                ga_cmd_num_indices = r_instr.imm2[15:0];
            end

            GD_EXEC_SLICE: begin
                sl_cmd_valid       = 1'b1;
                sl_cmd_src_base    = r_td0.sram_addr;
                sl_cmd_dst_base    = r_td2.sram_addr;
                sl_cmd_src_row_len = r_instr.imm2[15:0];
                sl_cmd_dst_row_len = r_instr.imm1[15:0];
                sl_cmd_start_offset = r_instr.imm0;
                sl_cmd_num_rows    = r_instr.imm1[31:16];
            end

            GD_EXEC_CONCAT: begin
                ct_cmd_valid       = 1'b1;
                ct_cmd_src0_base   = r_td0.sram_addr;
                ct_cmd_src1_base   = r_td1.sram_addr;
                ct_cmd_dst_base    = r_td2.sram_addr;
                ct_cmd_src0_row_len = r_instr.imm0;
                ct_cmd_src1_row_len = r_instr.imm1[15:0];
                ct_cmd_num_rows    = r_instr.imm2[15:0];
            end

            GD_EXEC_AVGPOOL: begin
                ap_cmd_valid    = 1'b1;
                ap_cmd_src_base = r_td0.sram_addr;
                ap_cmd_dst_base = r_td2.sram_addr;
                ap_cmd_C        = r_td0.shape1;
                ap_cmd_H        = r_td0.shape2;
                ap_cmd_W        = r_td0.shape3;
                ap_cmd_kh       = r_instr.imm0[15:8];
                ap_cmd_kw       = r_instr.imm0[7:0];
                ap_cmd_sh       = r_instr.imm1[15:8];
                ap_cmd_sw       = r_instr.imm1[7:0];
            end

            GD_EXEC_MAXPOOL: begin
                mp_cmd_valid    = 1'b1;
                mp_cmd_src_base = r_td0.sram_addr;
                mp_cmd_dst_base = r_td2.sram_addr;
                mp_cmd_C        = r_td0.shape1;
                mp_cmd_H        = r_td0.shape2;
                mp_cmd_W        = r_td0.shape3;
                mp_cmd_kh       = r_instr.imm0[15:8];
                mp_cmd_kw       = r_instr.imm0[7:0];
                mp_cmd_sh       = r_instr.imm1[15:8];
                mp_cmd_sw       = r_instr.imm1[7:0];
            end

            GD_EXEC_PAD: begin
                pd_cmd_valid      = 1'b1;
                pd_cmd_src_base   = r_td0.sram_addr;
                pd_cmd_dst_base   = r_td2.sram_addr;
                pd_cmd_C          = r_td0.shape1;
                pd_cmd_H          = r_td0.shape2;
                pd_cmd_W          = r_td0.shape3;
                pd_cmd_pad_top    = r_instr.imm0[15:8];
                pd_cmd_pad_bottom = r_instr.imm0[7:0];
                pd_cmd_pad_left   = r_instr.imm1[15:8];
                pd_cmd_pad_right  = r_instr.imm1[7:0];
            end

            GD_EXEC_RESIZE: begin
                rz_cmd_valid    = 1'b1;
                rz_cmd_src_base = r_td0.sram_addr;
                rz_cmd_dst_base = r_td2.sram_addr;
                rz_cmd_C        = r_td0.shape1;
                rz_cmd_in_H     = r_td0.shape2;
                rz_cmd_in_W     = r_td0.shape3;
                rz_cmd_out_H    = r_instr.imm0;
                rz_cmd_out_W    = r_instr.imm1[15:0];
            end

            GD_EXEC_CAST: begin
                ca_cmd_valid    = 1'b1;
                ca_cmd_src_base = r_td0.sram_addr;
                ca_cmd_dst_base = r_td2.sram_addr;
                ca_cmd_length   = r_td0.size_bytes;
                ca_cmd_src_dtype = r_td0.dtype;
                ca_cmd_dst_dtype = r_td2.dtype;
            end

            // Element-wise read A
            GD_EXEC_EW_RD_A: begin
                ew_rd_en   = 1'b1;
                ew_rd_addr = r_td0.sram_addr[SRAM0_AW-1:0] + ew_idx[SRAM0_AW-1:0];
            end

            // Element-wise read B
            GD_EXEC_EW_RD_B: begin
                ew_rd_en   = 1'b1;
                ew_rd_addr = r_td1.sram_addr[SRAM0_AW-1:0] + ew_idx[SRAM0_AW-1:0];
            end

            // Element-wise compute + write
            GD_EXEC_EW_COMPUTE: begin
                ew_wr_en   = 1'b1;
                ew_wr_addr = r_td2.sram_addr[SRAM0_AW-1:0] + ew_idx[SRAM0_AW-1:0];
                begin
                    automatic logic signed [15:0] a_ext = {{8{ew_val_a[7]}}, ew_val_a};
                    automatic logic signed [15:0] b_ext = {{8{ew_rd_data[7]}}, ew_rd_data};
                    automatic logic signed [15:0] result;
                    case (r_instr.opcode)
                        OP_G_EW_ADD: result = a_ext + b_ext;
                        OP_G_EW_MUL: result = (a_ext * b_ext) >>> 7;
                        OP_G_EW_SUB: result = a_ext - b_ext;
                        OP_G_EW_MIN: result = (a_ext < b_ext) ? a_ext : b_ext;
                        OP_G_EW_MAX: result = (a_ext > b_ext) ? a_ext : b_ext;
                        default:     result = a_ext;
                    endcase
                    if (result > 16'sd127)
                        ew_wr_data = 8'sd127;
                    else if (result < -16'sd128)
                        ew_wr_data = -8'sd128;
                    else
                        ew_wr_data = result[7:0];
                end
            end

            // RELU read
            GD_EXEC_RELU_RD: begin
                ew_rd_en   = 1'b1;
                ew_rd_addr = r_td0.sram_addr[SRAM0_AW-1:0] + ew_idx[SRAM0_AW-1:0];
            end

            // RELU write
            GD_EXEC_RELU_WR: begin
                ew_wr_en   = 1'b1;
                ew_wr_addr = r_td2.sram_addr[SRAM0_AW-1:0] + ew_idx[SRAM0_AW-1:0];
                ew_wr_data = ew_rd_data[7] ? 8'd0 : ew_rd_data;
            end

            GD_DONE: begin
                graph_done = 1'b1;
                graph_busy = 1'b0;
            end

            default: ;
        endcase
    end

    // =========================================================================
    // Sequential logic: instruction capture, descriptors, loops, scoreboard,
    // engine trackers, perf counters
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_instr     <= '0;
            r_td0       <= '0;
            r_td1       <= '0;
            r_td2       <= '0;
            ew_idx      <= '0;
            ew_count    <= '0;
            ew_val_a    <= '0;
            ew_val_b    <= '0;
            wait_target <= WAIT_NONE;
            error_code  <= GERR_NONE;
            dbg_pc      <= '0;
            dbg_last_op <= '0;
            timeout_cnt <= '0;
            r_scheduler_mode <= 1'b0;
            perf_total_cycles   <= '0;
            perf_gemm_cycles    <= '0;
            perf_softmax_cycles <= '0;
            perf_dma_cycles     <= '0;
            perf_reduce_cycles  <= '0;
            perf_math_cycles    <= '0;
            perf_gather_cycles  <= '0;
            perf_slice_cycles   <= '0;
            perf_concat_cycles  <= '0;
            perf_avgpool_cycles <= '0;
            perf_ew_cycles      <= '0;
            perf_overlap_cycles <= '0;
            perf_stall_cycles   <= '0;
            // Engine trackers
            dma_in_flight  <= 1'b0; dma_dst_tid  <= '0;
            gemm_in_flight <= 1'b0; gemm_dst_tid <= '0;
            sm_in_flight   <= 1'b0; sm_dst_tid   <= '0;
            re_in_flight   <= 1'b0; re_dst_tid   <= '0;
            me_in_flight   <= 1'b0; me_dst_tid   <= '0;
            ga_in_flight   <= 1'b0; ga_dst_tid   <= '0;
            sl_in_flight   <= 1'b0; sl_dst_tid   <= '0;
            ct_in_flight   <= 1'b0; ct_dst_tid   <= '0;
            ap_in_flight   <= 1'b0; ap_dst_tid   <= '0;
            mp_in_flight   <= 1'b0; mp_dst_tid   <= '0;
            pd_in_flight   <= 1'b0; pd_dst_tid   <= '0;
            rz_in_flight   <= 1'b0; rz_dst_tid   <= '0;
            ca_in_flight   <= 1'b0; ca_dst_tid   <= '0;
            // Scoreboard reset
            /* verilator lint_off BLKLOOPINIT */
            for (int i = 0; i < 256; i++)
                tensor_state[i] <= 2'd0;
            /* verilator lint_on BLKLOOPINIT */
        end else begin
            // ---- Engine done tracking (clear in-flight, update scoreboard) ----
            if (dma_in_flight && dma_done) begin
                dma_in_flight <= 1'b0;
                tensor_state[dma_dst_tid] <= 2'd2; // READY
            end
            if (gemm_in_flight && gm_done) begin
                gemm_in_flight <= 1'b0;
                tensor_state[gemm_dst_tid] <= 2'd2;
            end
            if (sm_in_flight && sm_done) begin
                sm_in_flight <= 1'b0;
                tensor_state[sm_dst_tid] <= 2'd2;
            end
            if (re_in_flight && re_done) begin
                re_in_flight <= 1'b0;
                tensor_state[re_dst_tid] <= 2'd2;
            end
            if (me_in_flight && me_done) begin
                me_in_flight <= 1'b0;
                tensor_state[me_dst_tid] <= 2'd2;
            end
            if (ga_in_flight && ga_done) begin
                ga_in_flight <= 1'b0;
                tensor_state[ga_dst_tid] <= 2'd2;
            end
            if (sl_in_flight && sl_done) begin
                sl_in_flight <= 1'b0;
                tensor_state[sl_dst_tid] <= 2'd2;
            end
            if (ct_in_flight && ct_done) begin
                ct_in_flight <= 1'b0;
                tensor_state[ct_dst_tid] <= 2'd2;
            end
            if (ap_in_flight && ap_done) begin
                ap_in_flight <= 1'b0;
                tensor_state[ap_dst_tid] <= 2'd2;
            end
            if (mp_in_flight && mp_done) begin
                mp_in_flight <= 1'b0;
                tensor_state[mp_dst_tid] <= 2'd2;
            end
            if (pd_in_flight && pd_done) begin
                pd_in_flight <= 1'b0;
                tensor_state[pd_dst_tid] <= 2'd2;
            end
            if (rz_in_flight && rz_done) begin
                rz_in_flight <= 1'b0;
                tensor_state[rz_dst_tid] <= 2'd2;
            end
            if (ca_in_flight && ca_done) begin
                ca_in_flight <= 1'b0;
                tensor_state[ca_dst_tid] <= 2'd2;
            end

            // ---- Performance counters ----
            if (state != GD_IDLE && state != GD_DONE)
                perf_total_cycles <= perf_total_cycles + 32'd1;

            // Per-engine performance counters
            if (state == GD_WAIT_DONE) begin
                case (wait_target)
                    WAIT_GEMM:    perf_gemm_cycles    <= perf_gemm_cycles    + 32'd1;
                    WAIT_SOFTMAX: perf_softmax_cycles <= perf_softmax_cycles + 32'd1;
                    WAIT_DMA:     perf_dma_cycles     <= perf_dma_cycles     + 32'd1;
                    WAIT_REDUCE:  perf_reduce_cycles  <= perf_reduce_cycles  + 32'd1;
                    WAIT_MATH:    perf_math_cycles    <= perf_math_cycles    + 32'd1;
                    WAIT_GATHER:  perf_gather_cycles  <= perf_gather_cycles  + 32'd1;
                    WAIT_SLICE:   perf_slice_cycles   <= perf_slice_cycles   + 32'd1;
                    WAIT_CONCAT:  perf_concat_cycles  <= perf_concat_cycles  + 32'd1;
                    WAIT_AVGPOOL: perf_avgpool_cycles <= perf_avgpool_cycles + 32'd1;
                    default: ;
                endcase
            end
            // EW cycles
            if (state inside {GD_EXEC_EW_INIT, GD_EXEC_EW_RD_A, GD_EXEC_EW_RD_B,
                              GD_EXEC_EW_COMPUTE, GD_EXEC_RELU_INIT, GD_EXEC_RELU_RD,
                              GD_EXEC_RELU_WR})
                perf_ew_cycles <= perf_ew_cycles + 32'd1;
            // Overlap cycles: DMA and GEMM both in-flight simultaneously
            if (dma_in_flight && gemm_in_flight)
                perf_overlap_cycles <= perf_overlap_cycles + 32'd1;
            // Stall cycles: dispatcher blocked waiting in GD_DECODE for deps/barriers
            if (state == GD_DECODE && state_next == GD_DECODE)
                perf_stall_cycles <= perf_stall_cycles + 32'd1;

            case (state)
                GD_IDLE: begin
                    if (start) begin
                        error_code  <= GERR_NONE;
                        dbg_pc      <= '0;
                        dbg_last_op <= '0;
                        r_scheduler_mode <= scheduler_mode;
                        perf_total_cycles   <= '0;
                        perf_gemm_cycles    <= '0;
                        perf_softmax_cycles <= '0;
                        perf_dma_cycles     <= '0;
                        perf_reduce_cycles  <= '0;
                        perf_math_cycles    <= '0;
                        perf_gather_cycles  <= '0;
                        perf_slice_cycles   <= '0;
                        perf_concat_cycles  <= '0;
                        perf_avgpool_cycles <= '0;
                        perf_ew_cycles      <= '0;
                        perf_overlap_cycles <= '0;
                        perf_stall_cycles   <= '0;
                        // Clear in-flight flags
                        dma_in_flight  <= 1'b0;
                        gemm_in_flight <= 1'b0;
                        sm_in_flight   <= 1'b0;
                        re_in_flight   <= 1'b0;
                        me_in_flight   <= 1'b0;
                        ga_in_flight   <= 1'b0;
                        sl_in_flight   <= 1'b0;
                        ct_in_flight   <= 1'b0;
                        ap_in_flight   <= 1'b0;
                        mp_in_flight   <= 1'b0;
                        pd_in_flight   <= 1'b0;
                        rz_in_flight   <= 1'b0;
                        ca_in_flight   <= 1'b0;
                        // Reset scoreboard
                        /* verilator lint_off BLKLOOPINIT */
                        for (int i = 0; i < 256; i++)
                            tensor_state[i] <= 2'd0;
                        /* verilator lint_on BLKLOOPINIT */
                    end
                end

                GD_FETCH_WAIT: begin
                    if (instr_valid) begin
                        r_instr     <= instr_in;
                        dbg_last_op <= instr_in.opcode;
                    end
                end

                GD_TDESC0: begin
                    r_td0 <= tensor_desc_t'(td_rd0_data);
                end

                GD_TDESC1: begin
                    r_td1 <= tensor_desc_t'(td_rd1_data);
                end

                GD_TDESC2: begin
                    r_td2 <= tensor_desc_t'(td_rd2_data);
                    // Valid opcode check
                    case (r_instr.opcode)
                        OP_G_DMA_LOAD, OP_G_DMA_STORE, OP_G_DMA_STRIDED,
                        OP_G_GEMM,
                        OP_G_EW_ADD, OP_G_EW_MUL, OP_G_EW_SUB,
                        OP_G_EW_MIN, OP_G_EW_MAX,
                        OP_G_RELU,
                        OP_G_SOFTMAX,
                        OP_G_REDUCE_SUM, OP_G_REDUCE_MAX, OP_G_REDUCE_MEAN,
                        OP_G_EXP, OP_G_LOG, OP_G_SQRT, OP_G_RSQRT,
                        OP_G_GATHER,
                        OP_G_SLICE, OP_G_CONCAT, OP_G_PAD,
                        OP_G_AVGPOOL2D,
                        OP_G_MAXPOOL2D, OP_G_RESIZE_NEAREST,
                        OP_G_PREFETCH, OP_G_CAST,
                        OP_G_BARRIER: ; // valid
                        default: error_code <= GERR_BAD_OPCODE;
                    endcase
                    // Shape mismatch check for GEMM
                    if (r_instr.opcode == OP_G_GEMM) begin
                        if (r_instr.flags[0]
                            ? (r_td0.shape1 != r_td1.shape1)
                            : (r_td0.shape1 != r_td1.shape0))
                            error_code <= GERR_SHAPE_MISMATCH;
                    end
                end

                GD_EXEC_DMA: begin
                    wait_target <= WAIT_DMA;
                    timeout_cnt <= '0;
                    dma_in_flight <= 1'b1;
                    dma_dst_tid   <= r_instr.src0;
                    tensor_state[r_instr.src0] <= 2'd1; // PRODUCING
                end

                GD_EXEC_GEMM: begin
                    wait_target <= WAIT_GEMM;
                    timeout_cnt <= '0;
                    gemm_in_flight <= 1'b1;
                    gemm_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_SOFTMAX: begin
                    wait_target <= WAIT_SOFTMAX;
                    timeout_cnt <= '0;
                    sm_in_flight <= 1'b1;
                    sm_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_REDUCE: begin
                    wait_target <= WAIT_REDUCE;
                    timeout_cnt <= '0;
                    re_in_flight <= 1'b1;
                    re_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_MATH: begin
                    wait_target <= WAIT_MATH;
                    timeout_cnt <= '0;
                    me_in_flight <= 1'b1;
                    me_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_GATHER: begin
                    wait_target <= WAIT_GATHER;
                    timeout_cnt <= '0;
                    ga_in_flight <= 1'b1;
                    ga_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_SLICE: begin
                    wait_target <= WAIT_SLICE;
                    timeout_cnt <= '0;
                    sl_in_flight <= 1'b1;
                    sl_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_CONCAT: begin
                    wait_target <= WAIT_CONCAT;
                    timeout_cnt <= '0;
                    ct_in_flight <= 1'b1;
                    ct_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_AVGPOOL: begin
                    wait_target <= WAIT_AVGPOOL;
                    timeout_cnt <= '0;
                    ap_in_flight <= 1'b1;
                    ap_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_MAXPOOL: begin
                    wait_target <= WAIT_MAXPOOL;
                    timeout_cnt <= '0;
                    mp_in_flight <= 1'b1;
                    mp_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_PAD: begin
                    wait_target <= WAIT_PAD;
                    timeout_cnt <= '0;
                    pd_in_flight <= 1'b1;
                    pd_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_RESIZE: begin
                    wait_target <= WAIT_RESIZE;
                    timeout_cnt <= '0;
                    rz_in_flight <= 1'b1;
                    rz_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_CAST: begin
                    wait_target <= WAIT_CAST;
                    timeout_cnt <= '0;
                    ca_in_flight <= 1'b1;
                    ca_dst_tid   <= r_instr.dst;
                    tensor_state[r_instr.dst] <= 2'd1;
                end

                GD_EXEC_EW_INIT: begin
                    ew_idx   <= 16'd0;
                    ew_count <= r_td0.size_bytes;
                end

                GD_EXEC_EW_RD_A: begin
                    // Read issued combinationally
                end

                GD_EXEC_EW_RD_B: begin
                    ew_val_a <= ew_rd_data;
                end

                GD_EXEC_EW_COMPUTE: begin
                    ew_val_b <= ew_rd_data;
                    ew_idx <= ew_idx + 16'd1;
                end

                GD_EXEC_RELU_INIT: begin
                    ew_idx   <= 16'd0;
                    ew_count <= r_td0.size_bytes;
                end

                GD_EXEC_RELU_RD: begin
                    // Read issued combinationally
                end

                GD_EXEC_RELU_WR: begin
                    ew_val_a <= ew_rd_data;
                    ew_idx   <= ew_idx + 16'd1;
                end

                GD_WAIT_DONE: begin
                    timeout_cnt <= timeout_cnt + 32'd1;
                    if (timeout_cnt >= MAX_CYCLES_PER_OP[31:0])
                        error_code <= GERR_TIMEOUT;
                end

                GD_NEXT: begin
                    dbg_pc <= dbg_pc + 16'd1;
                end

                GD_ERROR: begin
                    // Hold error state
                end

                default: ;
            endcase
        end
    end

endmodule

`default_nettype wire
