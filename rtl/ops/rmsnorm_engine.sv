// =============================================================================
// rmsnorm_engine.sv - RMSNorm over hidden dimension (2-pass)
// Pass 1: Stream input to accumulate sum(x^2), divide by N -> E[x^2]
// Compute inv_rms = rsqrt(E[x^2])
// Pass 2: Re-read input + gamma, apply: out[i] = (x[i] * gamma[i]) * inv_rms >> 16
//         Clamped to int8
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module rmsnorm_engine (
    input  logic               clk,
    input  logic               rst_n,

    // Command interface
    input  logic               cmd_valid,
    output logic               cmd_ready,
    input  logic [15:0]        length,       // hidden dimension
    input  logic [15:0]        src_base,     // input activation base address (SRAM0)
    input  logic [15:0]        dst_base,     // output base address (SRAM0)
    input  logic [15:0]        gamma_base,   // scale parameter base address (SRAM1)

    // SRAM0 read port (input x)
    output logic               sram_rd0_en,
    output logic [15:0]        sram_rd0_addr,
    input  logic [DATA_W-1:0]  sram_rd0_data,

    // SRAM1 read port (gamma)
    output logic               sram_rd1_en,
    output logic [15:0]        sram_rd1_addr,
    input  logic [DATA_W-1:0]  sram_rd1_data,

    // SRAM0 write port (output)
    output logic               sram_wr_en,
    output logic [15:0]        sram_wr_addr,
    output logic [DATA_W-1:0]  sram_wr_data,

    // Status
    output logic               busy,
    output logic               done
);

    // ----------------------------------------------------------------
    // FSM States
    // ----------------------------------------------------------------
    typedef enum logic [3:0] {
        S_IDLE,
        S_P1_READ,       // Pass 1: read input x[i] from SRAM0
        S_P1_ACCUM,      // Pass 1: accumulate x[i]^2
        S_P1_DIVIDE,     // Pass 1: divide sum_sq by N (iterative)
        S_RSQRT,         // Feed E[x^2] top bits to rsqrt LUT
        S_RSQRT_LATCH,   // Latch rsqrt result (1-cycle LUT latency)
        S_P2_READ,       // Pass 2: read x[i] from SRAM0, gamma[i] from SRAM1
        S_P2_COMPUTE,    // Pass 2: compute normalized output
        S_DONE
    } state_t;

    state_t state, state_nxt;

    // ----------------------------------------------------------------
    // Registered command parameters
    // ----------------------------------------------------------------
    logic [15:0] r_length, r_src_base, r_dst_base, r_gamma_base;
    logic [15:0] r_idx;

    // ----------------------------------------------------------------
    // Pass 1 accumulation
    // ----------------------------------------------------------------
    logic [31:0] r_sum_sq;       // running sum of x^2 (unsigned, max 128^2 * 64 = 1048576)
    logic [31:0] r_rms_val;      // E[x^2] = sum_sq / N

    // Division: simple restoring divider
    logic [5:0]  div_count;
    logic [47:0] div_remainder;
    logic [31:0] div_quotient;

    // ----------------------------------------------------------------
    // rsqrt LUT signals
    // ----------------------------------------------------------------
    logic [7:0]  rsqrt_addr;
    logic [15:0] rsqrt_data;
    logic [15:0] r_inv_rms;      // Q0.16 from rsqrt LUT

    // Pipeline registers for pass 2
    logic signed [7:0]  p_input;       // latched input value
    logic signed [7:0]  p_gamma;       // latched gamma value

    // ----------------------------------------------------------------
    // rsqrt LUT instantiation
    // ----------------------------------------------------------------
    rsqrt_lut u_rsqrt (
        .clk      (clk),
        .addr     (rsqrt_addr),
        .data_out (rsqrt_data)
    );

    // ----------------------------------------------------------------
    // FSM transition
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) state <= S_IDLE;
        else        state <= state_nxt;

    always_comb begin
        state_nxt = state;
        case (state)
            S_IDLE:         if (cmd_valid) state_nxt = S_P1_READ;
            S_P1_READ:      state_nxt = S_P1_ACCUM;
            S_P1_ACCUM:     state_nxt = (r_idx == r_length - 16'd1) ? S_P1_DIVIDE : S_P1_READ;
            S_P1_DIVIDE:    if (div_count == 6'd0) state_nxt = S_RSQRT;
            S_RSQRT:        state_nxt = S_RSQRT_LATCH;
            S_RSQRT_LATCH:  state_nxt = S_P2_READ;
            S_P2_READ:      state_nxt = S_P2_COMPUTE;
            S_P2_COMPUTE:   state_nxt = (r_idx == r_length - 16'd1) ? S_DONE : S_P2_READ;
            S_DONE:         state_nxt = S_IDLE;
            default:        state_nxt = S_IDLE;
        endcase
    end

    // ----------------------------------------------------------------
    // Command capture
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_length     <= '0;
            r_src_base   <= '0;
            r_dst_base   <= '0;
            r_gamma_base <= '0;
        end else if (state == S_IDLE && cmd_valid) begin
            r_length     <= length;
            r_src_base   <= src_base;
            r_dst_base   <= dst_base;
            r_gamma_base <= gamma_base;
        end
    end

    // ----------------------------------------------------------------
    // Index counter (reused across passes)
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_idx <= '0;
        else if (state == S_IDLE && cmd_valid)
            r_idx <= '0;
        else if (state == S_P1_ACCUM && state_nxt == S_P1_READ)
            r_idx <= r_idx + 16'd1;
        else if (state == S_P1_DIVIDE && div_count == 6'd0)
            r_idx <= '0;  // reset for pass 2
        else if (state == S_P2_COMPUTE && state_nxt == S_P2_READ)
            r_idx <= r_idx + 16'd1;
    end

    // ----------------------------------------------------------------
    // Pass 1: Accumulate sum of squares
    // ----------------------------------------------------------------
    logic signed [7:0] x_signed;
    logic [15:0] sq_val;  // x^2 (signed square, always >= 0)
    assign x_signed = sram_rd0_data;
    assign sq_val   = x_signed * x_signed;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_sum_sq <= '0;
        end else if (state == S_IDLE && cmd_valid) begin
            r_sum_sq <= '0;
        end else if (state == S_P1_ACCUM) begin
            // Accumulate x[i]^2 treating x as signed: |x|^2 = x*x (always positive)
            // sram_rd0_data is int8, cast to signed then square
            r_sum_sq <= r_sum_sq + 32'(sq_val);
        end
    end

    // ----------------------------------------------------------------
    // Pass 1: Division - compute sum_sq / N
    // Simple restoring divider, 32 cycles
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            div_count     <= '0;
            div_remainder <= '0;
            div_quotient  <= '0;
        end else if (state == S_P1_ACCUM && state_nxt == S_P1_DIVIDE) begin
            // Initialize division: sum_sq << 8 to get Q8.8 result alignment
            div_remainder <= 48'(r_sum_sq + 32'(sq_val)) << 8;
            div_quotient  <= '0;
            div_count     <= 6'd32;
        end else if (state == S_P1_DIVIDE && div_count > 6'd0) begin
            div_quotient <= div_quotient << 1;
            if (div_remainder >= 48'(r_length)) begin
                div_remainder <= div_remainder - 48'(r_length);
                div_quotient[0] <= 1'b1;
            end else begin
                div_quotient[0] <= 1'b0;
            end
            div_count <= div_count - 6'd1;
        end
    end

    // Latch E[x^2] when division completes
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_rms_val <= '0;
        else if (state == S_P1_DIVIDE && div_count == 6'd1)
            // Will be ready next cycle when div_count reaches 0
            r_rms_val <= div_quotient;
    end

    // ----------------------------------------------------------------
    // rsqrt computation
    // Feed top 8 bits of E[x^2] to rsqrt LUT (same as layernorm)
    // ----------------------------------------------------------------
    assign rsqrt_addr = r_rms_val[15:8];

    // Latch rsqrt result
    always_ff @(posedge clk) begin
        if (state == S_RSQRT_LATCH)
            r_inv_rms <= rsqrt_data;
    end

    // ----------------------------------------------------------------
    // SRAM read port control
    // ----------------------------------------------------------------
    always_comb begin
        sram_rd0_en   = 1'b0;
        sram_rd0_addr = '0;
        sram_rd1_en   = 1'b0;
        sram_rd1_addr = '0;

        if (state == S_P1_READ) begin
            // Pass 1: read input x[i]
            sram_rd0_en   = 1'b1;
            sram_rd0_addr = r_src_base + r_idx;
        end else if (state == S_P2_READ) begin
            // Pass 2: read input x[i] from SRAM0, gamma[i] from SRAM1
            sram_rd0_en   = 1'b1;
            sram_rd0_addr = r_src_base + r_idx;
            sram_rd1_en   = 1'b1;
            sram_rd1_addr = r_gamma_base + r_idx;
        end
    end

    // Pipeline registers for pass 2 data
    always_ff @(posedge clk) begin
        if (state == S_P2_READ) begin
            p_input <= signed'(sram_rd0_data);
            p_gamma <= signed'(sram_rd1_data);
        end
    end

    // ----------------------------------------------------------------
    // Pass 2: RMSNorm computation (fused: multiply gamma first)
    // out[i] = (x[i] * gamma[i]) * inv_rms >> 16
    // ----------------------------------------------------------------
    logic signed [15:0] xg;
    logic signed [31:0] gamma_applied;
    logic signed [7:0]  norm_result;

    always_comb begin
        // Multiply x * gamma first (int8 * int8 = int16, no shift)
        xg = 16'(signed'(p_input)) * 16'(signed'(p_gamma));

        // Then scale by inv_rms (Q0.16) and shift >> 16
        gamma_applied = (32'(xg) * 32'(signed'({1'b0, r_inv_rms}))) >>> 16;

        // Clamp to int8
        if (gamma_applied > 32'sd127)
            norm_result = 8'sd127;
        else if (gamma_applied < -32'sd128)
            norm_result = -8'sd128;
        else
            norm_result = gamma_applied[7:0];
    end

    // ----------------------------------------------------------------
    // SRAM write - during S_P2_COMPUTE
    // ----------------------------------------------------------------
    assign sram_wr_en   = (state == S_P2_COMPUTE);
    assign sram_wr_addr = r_dst_base + r_idx;
    assign sram_wr_data = norm_result;

    // ----------------------------------------------------------------
    // Status and handshake
    // ----------------------------------------------------------------
    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
