// =============================================================================
// Graph Dispatch FSM
// Main execution engine for Graph Mode. Serialized: one operation at a time.
//   - Tensor descriptor lookup
//   - Engine programming (gemm_ctrl, softmax_engine)
//   - Internal element-wise loops (EW_ADD/MUL/SUB, RELU)
//   - DMA shim interface
//   - Error detection (bad opcode, shape mismatch)
// =============================================================================
`default_nettype none

module graph_dispatch
    import npu_pkg::*;
    import graph_isa_pkg::*;
#(
    parameter int SRAM0_AW = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Control
    input  wire                     start,

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
    input  wire                     gm_done,

    // Softmax engine command
    output logic                    sm_cmd_valid,
    output logic [15:0]             sm_src_base,
    output logic [15:0]             sm_dst_base,
    output logic [15:0]             sm_length,
    input  wire                     sm_done,

    // DMA shim interface
    output logic                    dma_cmd_valid,
    output logic [31:0]             dma_ddr_addr,
    output logic [15:0]             dma_sram_addr,
    output logic [15:0]             dma_length,
    output logic                    dma_direction,  // 0=DDR→SRAM (load), 1=SRAM→DDR (store)
    output logic                    dma_strided,
    output logic [31:0]             dma_stride,
    output logic [15:0]             dma_count,
    output logic [15:0]             dma_block_len,
    input  wire                     dma_done,

    // SRAM0 element-wise read port (shared with other engines via mux)
    output logic                    ew_rd_en,
    output logic [SRAM0_AW-1:0]    ew_rd_addr,
    input  wire  [7:0]             ew_rd_data,

    // SRAM0 element-wise write port
    output logic                    ew_wr_en,
    output logic [SRAM0_AW-1:0]    ew_wr_addr,
    output logic [7:0]             ew_wr_data,

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
    tensor_desc_t  r_td0, r_td1, r_td2;  // src0, src1, dst descriptors

    // Element-wise loop state
    logic [15:0] ew_idx;
    logic [15:0] ew_count;
    logic [7:0]  ew_val_a;
    logic [7:0]  ew_val_b;

    // Which engine we're waiting on
    typedef enum logic [2:0] {
        WAIT_NONE,
        WAIT_GEMM,
        WAIT_SOFTMAX,
        WAIT_DMA
    } wait_target_t;

    wait_target_t wait_target;

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
                // Check for END instruction
                if (r_instr.opcode == OP_G_END)
                    state_next = GD_DONE;
                else
                    state_next = GD_TDESC0;
            end

            GD_TDESC0: begin
                // Tensor descriptors are combinational reads, advance immediately
                state_next = GD_TDESC1;
            end

            GD_TDESC1: begin
                state_next = GD_TDESC2;
            end

            GD_TDESC2: begin
                // Route to appropriate execution state based on opcode
                case (r_instr.opcode)
                    OP_G_DMA_LOAD,
                    OP_G_DMA_STORE,
                    OP_G_DMA_STRIDED:  state_next = GD_EXEC_DMA;
                    OP_G_GEMM:         begin
                        // Shape check: src0.shape1 must == src1 inner dim
                        // If TRANSPOSE_B, weight is [N,K] so check shape1; else [K,N] so check shape0
                        if (r_instr.flags[0]
                            ? (r_td0.shape1 != r_td1.shape1)
                            : (r_td0.shape1 != r_td1.shape0))
                            state_next = GD_ERROR;
                        else
                            state_next = GD_EXEC_GEMM;
                    end
                    OP_G_EW_ADD,
                    OP_G_EW_MUL,
                    OP_G_EW_SUB:       state_next = GD_EXEC_EW_INIT;
                    OP_G_RELU:         state_next = GD_EXEC_RELU_INIT;
                    OP_G_SOFTMAX:      state_next = GD_EXEC_SOFTMAX;
                    OP_G_BARRIER:      state_next = GD_NEXT; // no-op in serialized mode
                    default:           state_next = GD_ERROR;
                endcase
            end

            GD_EXEC_DMA: begin
                state_next = GD_WAIT_DONE;
            end

            GD_EXEC_GEMM: begin
                state_next = GD_WAIT_DONE;
            end

            GD_EXEC_SOFTMAX: begin
                state_next = GD_WAIT_DONE;
            end

            GD_EXEC_EW_INIT: begin
                // Use r_td0.size_bytes directly, NOT ew_count (which is
                // set by sequential logic in this same state and not yet valid)
                if (r_td0.size_bytes == 16'd0)
                    state_next = GD_NEXT;
                else
                    state_next = GD_EXEC_EW_RD_A;
            end

            GD_EXEC_EW_RD_A: begin
                state_next = GD_EXEC_EW_RD_B;
            end

            GD_EXEC_EW_RD_B: begin
                state_next = GD_EXEC_EW_COMPUTE;
            end

            GD_EXEC_EW_COMPUTE: begin
                if (ew_idx >= ew_count - 16'd1)
                    state_next = GD_NEXT;
                else
                    state_next = GD_EXEC_EW_RD_A;
            end

            GD_EXEC_RELU_INIT: begin
                // Use r_td0.size_bytes directly, NOT ew_count (same reason)
                if (r_td0.size_bytes == 16'd0)
                    state_next = GD_NEXT;
                else
                    state_next = GD_EXEC_RELU_RD;
            end

            GD_EXEC_RELU_RD: begin
                state_next = GD_EXEC_RELU_WR;
            end

            GD_EXEC_RELU_WR: begin
                if (ew_idx >= ew_count - 16'd1)
                    state_next = GD_NEXT;
                else
                    state_next = GD_EXEC_RELU_RD;
            end

            GD_WAIT_DONE: begin
                case (wait_target)
                    WAIT_GEMM:    if (gm_done)  state_next = GD_NEXT;
                    WAIT_SOFTMAX: if (sm_done)  state_next = GD_NEXT;
                    WAIT_DMA:     if (dma_done) state_next = GD_NEXT;
                    default:      state_next = GD_NEXT;
                endcase
            end

            GD_NEXT: begin
                state_next = GD_FETCH_WAIT;
            end

            GD_DONE: begin
                state_next = GD_IDLE;
            end

            GD_ERROR: begin
                // Stay in error until reset
                state_next = GD_ERROR;
            end

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
        sm_cmd_valid  = 1'b0;
        sm_src_base   = '0;
        sm_dst_base   = '0;
        sm_length     = '0;
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
                instr_ready = 1'b1; // Ready to accept instruction
            end

            GD_TDESC0: begin
                td_rd0_addr = r_instr.src0;
                td_rd1_addr = r_instr.src1;
                td_rd2_addr = r_instr.dst;
            end

            GD_EXEC_DMA: begin
                dma_cmd_valid = 1'b1;
                if (r_instr.opcode == OP_G_DMA_LOAD) begin
                    dma_ddr_addr  = r_td0.ddr_addr;
                    dma_sram_addr = r_td0.sram_addr;
                    dma_length    = r_td0.size_bytes;
                    dma_direction = 1'b0; // DDR→SRAM
                end else if (r_instr.opcode == OP_G_DMA_STORE) begin
                    dma_ddr_addr  = r_td0.ddr_addr;
                    dma_sram_addr = r_td0.sram_addr;
                    dma_length    = r_td0.size_bytes;
                    dma_direction = 1'b1; // SRAM→DDR
                end else begin
                    // DMA_STRIDED
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
                // When TRANSPOSE_B, weight is [N,K]: N=shape0, K=shape1
                // When normal,      weight is [K,N]: K=shape0, N=shape1
                gm_cmd_N     = r_instr.flags[0] ? r_td1.shape0 : r_td1.shape1;
                gm_cmd_K     = r_td0.shape1;
                gm_cmd_flags = r_instr.flags;
                gm_cmd_imm   = r_instr.imm0;
            end

            GD_EXEC_SOFTMAX: begin
                sm_cmd_valid = 1'b1;
                sm_src_base  = r_td0.sram_addr;
                sm_dst_base  = r_td2.sram_addr;
                sm_length    = r_td0.size_bytes;
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
            // Note: ew_val_a was captured in GD_EXEC_EW_RD_B (from A read),
            // and ew_rd_data currently holds B read result (1-cycle latency)
            GD_EXEC_EW_COMPUTE: begin
                ew_wr_en   = 1'b1;
                ew_wr_addr = r_td2.sram_addr[SRAM0_AW-1:0] + ew_idx[SRAM0_AW-1:0];
                begin
                    automatic logic signed [15:0] a_ext = {{8{ew_val_a[7]}}, ew_val_a};
                    // Use ew_rd_data directly for B (current SRAM output from RD_B cycle)
                    automatic logic signed [15:0] b_ext = {{8{ew_rd_data[7]}}, ew_rd_data};
                    automatic logic signed [15:0] result;
                    case (r_instr.opcode)
                        OP_G_EW_ADD: result = a_ext + b_ext;
                        OP_G_EW_MUL: result = (a_ext * b_ext) >>> 7; // Q7 multiply
                        OP_G_EW_SUB: result = a_ext - b_ext;
                        default:     result = a_ext;
                    endcase
                    // Saturate to int8
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
            // ew_rd_data holds the value read in GD_EXEC_RELU_RD (1-cycle latency)
            GD_EXEC_RELU_WR: begin
                ew_wr_en   = 1'b1;
                ew_wr_addr = r_td2.sram_addr[SRAM0_AW-1:0] + ew_idx[SRAM0_AW-1:0];
                // max(0, val) for signed int8
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
    // Sequential logic: capture instruction, descriptors, run loops
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
        end else begin
            case (state)
                GD_IDLE: begin
                    if (start) begin
                        error_code  <= GERR_NONE;
                        dbg_pc      <= '0;
                        dbg_last_op <= '0;
                    end
                end

                GD_FETCH_WAIT: begin
                    if (instr_valid) begin
                        r_instr     <= instr_in;
                        dbg_last_op <= instr_in.opcode;
                    end
                end

                GD_TDESC0: begin
                    // Combinational read results are available this cycle
                    r_td0 <= tensor_desc_t'(td_rd0_data);
                end

                GD_TDESC1: begin
                    r_td1 <= tensor_desc_t'(td_rd1_data);
                end

                GD_TDESC2: begin
                    r_td2 <= tensor_desc_t'(td_rd2_data);
                    // Set up error for bad opcode
                    case (r_instr.opcode)
                        OP_G_DMA_LOAD, OP_G_DMA_STORE, OP_G_DMA_STRIDED,
                        OP_G_GEMM,
                        OP_G_EW_ADD, OP_G_EW_MUL, OP_G_EW_SUB,
                        OP_G_RELU,
                        OP_G_SOFTMAX,
                        OP_G_BARRIER: ; // valid
                        default: error_code <= GERR_BAD_OPCODE;
                    endcase
                    // Shape mismatch check for GEMM
                    // r_td0 was captured in TDESC0, r_td1 in TDESC1 — both available now
                    if (r_instr.opcode == OP_G_GEMM) begin
                        if (r_instr.flags[0]
                            ? (r_td0.shape1 != r_td1.shape1)
                            : (r_td0.shape1 != r_td1.shape0))
                            error_code <= GERR_SHAPE_MISMATCH;
                    end
                end

                GD_EXEC_DMA: begin
                    wait_target <= WAIT_DMA;
                end

                GD_EXEC_GEMM: begin
                    wait_target <= WAIT_GEMM;
                end

                GD_EXEC_SOFTMAX: begin
                    wait_target <= WAIT_SOFTMAX;
                end

                GD_EXEC_EW_INIT: begin
                    ew_idx   <= 16'd0;
                    ew_count <= r_td0.size_bytes;
                end

                GD_EXEC_EW_RD_A: begin
                    // Nothing—read is issued combinationally
                end

                GD_EXEC_EW_RD_B: begin
                    // Capture read-A result (1-cycle latency from SRAM)
                    ew_val_a <= ew_rd_data;
                end

                GD_EXEC_EW_COMPUTE: begin
                    // Capture read-B result
                    ew_val_b <= ew_rd_data;
                    // Note: ew_val_b used in combo output is actually captured
                    // here for the NEXT iteration; current combo uses registered value
                    // Advance index
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
                    // Capture read result
                    ew_val_a <= ew_rd_data;
                    ew_idx   <= ew_idx + 16'd1;
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
