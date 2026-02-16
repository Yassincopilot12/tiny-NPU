// =============================================================================
// MxN Systolic Array (default 16x16 = 256 MACs, parameterizable to 24x24)
// Weight-stationary style: A flows east, B flows south
// Supports INT8 and FP16 data types via dtype_fp16 select
//
// Pipeline registers are inserted every PIPE_STAGE rows/columns to break
// long combinational paths and enable larger array sizes without timing
// violations. A-flow is pipelined at column boundaries that are multiples
// of PIPE_STAGE; B-flow is pipelined at row boundaries that are multiples
// of PIPE_STAGE.
// =============================================================================
`default_nettype none

module systolic_array #(
    parameter int M          = 16,   // rows (default 16, can override to 24)
    parameter int N          = 16,   // cols (default 16, can override to 24)
    parameter int DATA_W     = 8,
    parameter int ACC_W      = 32,
    parameter int PIPE_STAGE = 32    // disabled for now (must be > max(M,N) to avoid A/B desync)
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        clear_acc,
    input  wire                        en,
    input  wire                        dtype_fp16,
    // INT8 input: one column of A per row, one row of B per column
    input  wire  signed [DATA_W-1:0]   a_col  [M],
    input  wire  signed [DATA_W-1:0]   b_row  [N],
    // FP16 input
    input  wire  signed [15:0]         a_col_fp16 [M],
    input  wire  signed [15:0]         b_row_fp16 [N],
    // Output: MxN accumulator grid (INT32 for both modes)
    output logic signed [ACC_W-1:0]    acc_out [M][N],
    output logic                       acc_valid
);

    // =========================================================================
    // Inter-PE wiring (directly driven by PE outputs and external inputs)
    // =========================================================================

    // A flows east: a_wire[row][col], col ranges 0..N
    //   a_wire[gi][0]   = external input
    //   a_wire[gi][gj+1]= PE(gi,gj) a_out
    logic signed [DATA_W-1:0] a_wire      [M][N+1];
    logic signed [15:0]       a_wire_fp16 [M][N+1];

    // B flows south: b_wire[row][col], row ranges 0..M
    //   b_wire[0][gj]   = external input
    //   b_wire[gi+1][gj]= PE(gi,gj) b_out
    logic signed [DATA_W-1:0] b_wire      [M+1][N];
    logic signed [15:0]       b_wire_fp16 [M+1][N];

    // =========================================================================
    // Pipeline-registered versions of inter-PE wiring
    //
    // a_pipe[gi][gj] is what PE(gi,gj) actually reads as its a_in.
    // b_pipe[gi][gj] is what PE(gi,gj) actually reads as its b_in.
    //
    // At pipeline boundaries (column index is a nonzero multiple of
    // PIPE_STAGE for A; row index is a nonzero multiple of PIPE_STAGE
    // for B), a registered stage is inserted. Otherwise, the piped
    // signal is a direct pass-through of the corresponding wire.
    // =========================================================================
    logic signed [DATA_W-1:0] a_pipe      [M][N];
    logic signed [15:0]       a_pipe_fp16 [M][N];

    logic signed [DATA_W-1:0] b_pipe      [M][N];
    logic signed [15:0]       b_pipe_fp16 [M][N];

    logic signed [ACC_W-1:0]  pe_acc      [M][N];

    // =========================================================================
    // Connect external inputs to wire arrays (column 0 for A, row 0 for B)
    // =========================================================================
    genvar gi, gj;
    generate
        for (gi = 0; gi < M; gi++) begin : gen_a_input
            assign a_wire[gi][0]      = a_col[gi];
            assign a_wire_fp16[gi][0] = a_col_fp16[gi];
        end
        for (gj = 0; gj < N; gj++) begin : gen_b_input
            assign b_wire[0][gj]      = b_row[gj];
            assign b_wire_fp16[0][gj] = b_row_fp16[gj];
        end
    endgenerate

    // =========================================================================
    // A-flow pipeline registers (horizontal, at column boundaries)
    //
    // For PE at column gj, its a input comes from a_wire[gi][gj].
    // If gj is a nonzero multiple of PIPE_STAGE, we register that value;
    // otherwise we pass it through combinationally.
    // =========================================================================
    generate
        for (gi = 0; gi < M; gi++) begin : gen_a_pipe_row
            for (gj = 0; gj < N; gj++) begin : gen_a_pipe_col
                if (gj != 0 && (gj % PIPE_STAGE == 0)) begin : a_pipe_reg
                    // Pipeline register at this column boundary
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            a_pipe[gi][gj]      <= '0;
                            a_pipe_fp16[gi][gj] <= '0;
                        end else if (en) begin
                            a_pipe[gi][gj]      <= a_wire[gi][gj];
                            a_pipe_fp16[gi][gj] <= a_wire_fp16[gi][gj];
                        end
                    end
                end else begin : a_pipe_thru
                    // Direct pass-through (no extra latency)
                    assign a_pipe[gi][gj]      = a_wire[gi][gj];
                    assign a_pipe_fp16[gi][gj] = a_wire_fp16[gi][gj];
                end
            end
        end
    endgenerate

    // =========================================================================
    // B-flow pipeline registers (vertical, at row boundaries)
    //
    // For PE at row gi, its b input comes from b_wire[gi][gj].
    // If gi is a nonzero multiple of PIPE_STAGE, we register that value;
    // otherwise we pass it through combinationally.
    // =========================================================================
    generate
        for (gi = 0; gi < M; gi++) begin : gen_b_pipe_row
            for (gj = 0; gj < N; gj++) begin : gen_b_pipe_col
                if (gi != 0 && (gi % PIPE_STAGE == 0)) begin : b_pipe_reg
                    // Pipeline register at this row boundary
                    always_ff @(posedge clk or negedge rst_n) begin
                        if (!rst_n) begin
                            b_pipe[gi][gj]      <= '0;
                            b_pipe_fp16[gi][gj] <= '0;
                        end else if (en) begin
                            b_pipe[gi][gj]      <= b_wire[gi][gj];
                            b_pipe_fp16[gi][gj] <= b_wire_fp16[gi][gj];
                        end
                    end
                end else begin : b_pipe_thru
                    // Direct pass-through (no extra latency)
                    assign b_pipe[gi][gj]      = b_wire[gi][gj];
                    assign b_pipe_fp16[gi][gj] = b_wire_fp16[gi][gj];
                end
            end
        end
    endgenerate

    // =========================================================================
    // Instantiate MxN PE grid
    //
    // Each PE reads from the piped arrays (which include pipeline registers
    // at boundaries) and writes its pass-through outputs to the raw wire
    // arrays. This means:
    //   - PE input:  a_pipe[gi][gj],  b_pipe[gi][gj]
    //   - PE output: a_wire[gi][gj+1], b_wire[gi+1][gj]
    // =========================================================================
    generate
        for (gi = 0; gi < M; gi++) begin : gen_row
            for (gj = 0; gj < N; gj++) begin : gen_col
                pe #(
                    .DATA_W (DATA_W),
                    .ACC_W  (ACC_W)
                ) u_pe (
                    .clk         (clk),
                    .rst_n       (rst_n),
                    .clear_acc   (clear_acc),
                    .en          (en),
                    .dtype_fp16  (dtype_fp16),
                    // INT8: read from piped, write to raw wire
                    .a_in        (a_pipe[gi][gj]),
                    .b_in        (b_pipe[gi][gj]),
                    .a_out       (a_wire[gi][gj+1]),
                    .b_out       (b_wire[gi+1][gj]),
                    // FP16: read from piped, write to raw wire
                    .a_in_fp16   (a_pipe_fp16[gi][gj]),
                    .b_in_fp16   (b_pipe_fp16[gi][gj]),
                    .a_out_fp16  (a_wire_fp16[gi][gj+1]),
                    .b_out_fp16  (b_wire_fp16[gi+1][gj]),
                    .acc_out     (pe_acc[gi][gj])
                );
            end
        end
    endgenerate

    // =========================================================================
    // Output accumulator grid
    // =========================================================================
    generate
        for (gi = 0; gi < M; gi++) begin : gen_acc_row
            for (gj = 0; gj < N; gj++) begin : gen_acc_col
                assign acc_out[gi][gj] = pe_acc[gi][gj];
            end
        end
    endgenerate

    // =========================================================================
    // acc_valid: assert after data has propagated through the entire array,
    // accounting for additional pipeline register latency.
    //
    // Without pipeline registers the drain latency is M + N cycles (each PE
    // adds one cycle of latency for the pass-through).
    //
    // Each pipeline boundary crossed adds one extra cycle. For A-flow across
    // N columns, the number of pipeline boundaries is floor((N-1)/PIPE_STAGE).
    // For B-flow across M rows, the number is floor((M-1)/PIPE_STAGE).
    // =========================================================================
    localparam int NUM_A_PIPES    = (N > 1) ? ((N - 1) / PIPE_STAGE) : 0;
    localparam int NUM_B_PIPES    = (M > 1) ? ((M - 1) / PIPE_STAGE) : 0;
    localparam int DRAIN_LATENCY  = M + N + NUM_A_PIPES + NUM_B_PIPES;
    localparam int CNT_W          = $clog2(DRAIN_LATENCY + 1);
    logic [CNT_W-1:0] valid_cnt;
    localparam [CNT_W-1:0] DRAIN_MAX = CNT_W'(DRAIN_LATENCY);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_cnt <= '0;
            acc_valid <= 1'b0;
        end else if (clear_acc) begin
            valid_cnt <= '0;
            acc_valid <= 1'b0;
        end else if (en) begin
            if (valid_cnt < DRAIN_MAX)
                valid_cnt <= valid_cnt + 1;
            else
                acc_valid <= 1'b1;
        end
    end

endmodule

`default_nettype wire
