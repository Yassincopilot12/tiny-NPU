// =============================================================================
// layernorm_engine.sv - Full LayerNorm over hidden dimension (2-pass)
// Pass 1: Stream input through mean_var_engine to get mean and variance
// Compute inv_std = rsqrt(variance + epsilon)
// Pass 2: Re-read input, apply: out[i] = ((in[i]-mean)*inv_std*gamma[i]+beta[i])
//         requantized to int8
//
// FP16 mode (cmd_dtype==1):
//   Pass 1: Read 2-byte FP16 elements, accumulate sum and sum_sq inline.
//           mean = sum * recip_N, var = sum_sq*recip_N - mean^2
//           inv_std = rsqrt_fp16(var)
//   Pass 2: Read input(2B) + gamma(2B) + beta(2B), normalize in FP16,
//           write 2-byte FP16 result.
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;
import fp16_utils_pkg::*;

module layernorm_engine (
    input  logic               clk,
    input  logic               rst_n,

    // Command interface
    input  logic               cmd_valid,
    output logic               cmd_ready,
    input  logic [15:0]        length,       // hidden dimension
    input  logic [15:0]        src_base,     // input activation base address
    input  logic [15:0]        dst_base,     // output base address
    input  logic [15:0]        gamma_base,   // scale parameter base address
    input  logic [15:0]        beta_base,    // bias parameter base address
    input  logic [1:0]         cmd_dtype,    // 0=INT8, 1=FP16

    // SRAM read port 0 (input / gamma)
    output logic               sram_rd0_en,
    output logic [15:0]        sram_rd0_addr,
    input  logic [DATA_W-1:0]  sram_rd0_data,

    // SRAM read port 1 (beta)
    output logic               sram_rd1_en,
    output logic [15:0]        sram_rd1_addr,
    input  logic [DATA_W-1:0]  sram_rd1_data,

    // SRAM write port (output)
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
    typedef enum logic [4:0] {
        S_IDLE,
        S_P1_READ,           // Pass 1: read input (INT8: 1 byte; FP16: low byte)
        S_P1_READ_HI,        // Pass 1 FP16: read high byte of input element
        S_P1_FEED,           // Pass 1: feed element to mean_var_engine / FP16 accum
        S_P1_WAIT,           // Pass 1: wait for mean/var result (INT8 only)
        S_P1_FP16_FINALIZE,  // Pass 1 FP16: compute mean and variance from sums
        S_RSQRT,             // Compute rsqrt of variance
        S_RSQRT_LATCH,       // Latch rsqrt result (1-cycle LUT latency)
        S_P2_READ,           // Pass 2: read input low byte (INT8: input+beta)
        S_P2_READ_HI,        // Pass 2 FP16: read input high byte
        S_P2_READ_GAMMA,     // Pass 2 FP16: read gamma low byte
        S_P2_READ_GAMMA_HI,  // Pass 2 FP16: read gamma high byte
        S_P2_READ_BETA,      // Pass 2 FP16: read beta low byte
        S_P2_READ_BETA_HI,   // Pass 2 FP16: read beta high byte
        S_P2_COMPUTE,        // Pass 2: compute normalized output
        S_P2_WRITE_HI,       // Pass 2 FP16: write high byte of output
        S_DONE
    } state_t;

    state_t state, state_nxt;

    // ----------------------------------------------------------------
    // Registered command parameters
    // ----------------------------------------------------------------
    logic [15:0] r_length, r_src_base, r_dst_base, r_gamma_base, r_beta_base;
    logic [15:0] r_idx;
    logic [1:0]  r_dtype;   // latched dtype: 0=INT8, 1=FP16

    // ----------------------------------------------------------------
    // FP16 internal registers
    // ----------------------------------------------------------------
    logic [15:0] r_mean_fp16;       // FP16 mean value
    logic [15:0] r_inv_std_fp16;    // FP16 inverse std
    logic [15:0] r_fp16_sum;        // running sum for mean (FP16)
    logic [15:0] r_fp16_sq_sum;     // running sum of squares (FP16)
    logic [7:0]  lo_byte;           // staging register for 2-byte SRAM reads
    logic [15:0] fp16_rd_val;       // assembled FP16 value from 2 reads
    logic [15:0] fp16_gamma_val;    // gamma read as FP16
    logic [15:0] fp16_beta_val;     // beta read as FP16
    logic [15:0] r_recip_n_fp16;    // precomputed 1/N in FP16

    // ----------------------------------------------------------------
    // mean_var_engine signals (INT8 path)
    // ----------------------------------------------------------------
    logic               mv_start;
    logic               mv_din_valid;
    logic signed [7:0]  mv_din;
    logic               mv_din_last;
    logic signed [15:0] mv_mean;       // Q8.8
    logic        [31:0] mv_var;        // variance (unsigned)
    logic               mv_valid;

    // Latched results (INT8 path)
    logic signed [15:0] r_mean;        // Q8.8
    logic        [15:0] r_inv_std;     // Q0.16 from rsqrt LUT

    // rsqrt LUT signals (INT8 path)
    logic [7:0]         rsqrt_addr;
    logic [15:0]        rsqrt_data;

    // rsqrt FP16 LUT signals
    logic [7:0]         rsqrt_fp16_addr;
    logic [15:0]        rsqrt_fp16_data;

    // Pipeline registers (INT8 path)
    logic signed [7:0]  p_input;       // latched input value
    logic signed [7:0]  p_gamma;       // latched gamma value
    logic signed [7:0]  p_beta;        // latched beta value

    // ----------------------------------------------------------------
    // Sub-module instantiations
    // ----------------------------------------------------------------
    mean_var_engine u_mean_var (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (mv_start),
        .length       (r_length),
        .din_valid    (mv_din_valid),
        .din          (mv_din),
        .din_last     (mv_din_last),
        .mean_out     (mv_mean),
        .var_out      (mv_var),
        .result_valid (mv_valid)
    );

    rsqrt_lut u_rsqrt (
        .clk      (clk),
        .addr     (rsqrt_addr),
        .data_out (rsqrt_data)
    );

    graph_rsqrt_lut_fp16 u_rsqrt_fp16 (
        .clk      (clk),
        .addr     (rsqrt_fp16_addr),
        .data_out (rsqrt_fp16_data)
    );

    // ----------------------------------------------------------------
    // Reciprocal of N LUT (FP16)
    // Maps small integer N -> FP16(1/N)
    // Covers N = 1..256 (common hidden dimensions)
    // Uses upper 8 bits of length as index
    // ----------------------------------------------------------------
    logic [15:0] recip_n_lut_out;

    always_comb begin
        // Compute 1/N for common layer sizes via case statement
        // FP16 encoding: sign(1) | exponent(5) | mantissa(10)
        // 1/1   = 1.0     = 0x3C00
        // 1/2   = 0.5     = 0x3800
        // 1/4   = 0.25    = 0x3400
        // 1/8   = 0.125   = 0x3000
        // 1/16  = 0.0625  = 0x2C00
        // 1/32  = 0.03125 = 0x2800
        // 1/64  = 0x2400
        // 1/128 = 0x2000
        // 1/256 = 0x1C00
        // 1/512 = 0x1800
        // For non-power-of-2 values, approximate via nearest power-of-2
        case (r_length)
            16'd1:   recip_n_lut_out = 16'h3C00; // 1.0
            16'd2:   recip_n_lut_out = 16'h3800; // 0.5
            16'd3:   recip_n_lut_out = 16'h3555; // ~0.3333
            16'd4:   recip_n_lut_out = 16'h3400; // 0.25
            16'd5:   recip_n_lut_out = 16'h3266; // ~0.2
            16'd6:   recip_n_lut_out = 16'h3155; // ~0.1667
            16'd7:   recip_n_lut_out = 16'h3092; // ~0.1429
            16'd8:   recip_n_lut_out = 16'h3000; // 0.125
            16'd10:  recip_n_lut_out = 16'h2E66; // 0.1
            16'd12:  recip_n_lut_out = 16'h2D55; // ~0.0833
            16'd16:  recip_n_lut_out = 16'h2C00; // 0.0625
            16'd20:  recip_n_lut_out = 16'h2A66; // 0.05
            16'd24:  recip_n_lut_out = 16'h2955; // ~0.0417
            16'd32:  recip_n_lut_out = 16'h2800; // 0.03125
            16'd48:  recip_n_lut_out = 16'h2555; // ~0.0208
            16'd64:  recip_n_lut_out = 16'h2400; // 0.015625
            16'd96:  recip_n_lut_out = 16'h2155; // ~0.0104
            16'd128: recip_n_lut_out = 16'h2000; // 0.0078125
            16'd192: recip_n_lut_out = 16'h1D55; // ~0.0052
            16'd256: recip_n_lut_out = 16'h1C00; // 0.00390625
            16'd384: recip_n_lut_out = 16'h1955; // ~0.0026
            16'd512: recip_n_lut_out = 16'h1800; // 0.001953125
            16'd768: recip_n_lut_out = 16'h1555; // ~0.0013
            16'd1024:recip_n_lut_out = 16'h1400; // ~0.000977
            16'd2048:recip_n_lut_out = 16'h1000; // ~0.000488
            16'd4096:recip_n_lut_out = 16'h0C00; // ~0.000244
            default: begin
                // Fallback: use int8_to_fp16 of upper byte as rough approximation
                // For lengths that are not in the LUT, approximate by nearest power-of-2
                if (r_length >= 16'd2048)
                    recip_n_lut_out = 16'h1000;
                else if (r_length >= 16'd1024)
                    recip_n_lut_out = 16'h1400;
                else if (r_length >= 16'd512)
                    recip_n_lut_out = 16'h1800;
                else if (r_length >= 16'd256)
                    recip_n_lut_out = 16'h1C00;
                else if (r_length >= 16'd128)
                    recip_n_lut_out = 16'h2000;
                else if (r_length >= 16'd64)
                    recip_n_lut_out = 16'h2400;
                else if (r_length >= 16'd32)
                    recip_n_lut_out = 16'h2800;
                else if (r_length >= 16'd16)
                    recip_n_lut_out = 16'h2C00;
                else if (r_length >= 16'd8)
                    recip_n_lut_out = 16'h3000;
                else if (r_length >= 16'd4)
                    recip_n_lut_out = 16'h3400;
                else if (r_length >= 16'd2)
                    recip_n_lut_out = 16'h3800;
                else
                    recip_n_lut_out = 16'h3C00;
            end
        endcase
    end

    // ----------------------------------------------------------------
    // FSM transition
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) state <= S_IDLE;
        else        state <= state_nxt;

    always_comb begin
        state_nxt = state;
        case (state)
            S_IDLE: if (cmd_valid) state_nxt = S_P1_READ;

            // === Pass 1 ===
            S_P1_READ: begin
                if (r_dtype == 2'd1)
                    state_nxt = S_P1_READ_HI;  // FP16: need high byte
                else
                    state_nxt = S_P1_FEED;      // INT8: data ready
            end

            S_P1_READ_HI:
                state_nxt = S_P1_FEED;  // FP16: both bytes read, feed

            S_P1_FEED: begin
                if (r_idx == r_length - 16'd1) begin
                    if (r_dtype == 2'd1)
                        state_nxt = S_P1_FP16_FINALIZE;  // FP16: compute mean/var inline
                    else
                        state_nxt = S_P1_WAIT;            // INT8: wait for mean_var_engine
                end else begin
                    state_nxt = S_P1_READ;
                end
            end

            S_P1_WAIT:
                if (mv_valid) state_nxt = S_RSQRT;

            S_P1_FP16_FINALIZE:
                state_nxt = S_RSQRT;

            // === rsqrt ===
            S_RSQRT:        state_nxt = S_RSQRT_LATCH;
            S_RSQRT_LATCH:  state_nxt = S_P2_READ;

            // === Pass 2 ===
            S_P2_READ: begin
                if (r_dtype == 2'd1)
                    state_nxt = S_P2_READ_HI;      // FP16: read input high byte
                else
                    state_nxt = S_P2_COMPUTE;       // INT8: data ready, compute
            end

            S_P2_READ_HI:
                state_nxt = S_P2_READ_GAMMA;        // FP16: input done, read gamma

            S_P2_READ_GAMMA:
                state_nxt = S_P2_READ_GAMMA_HI;     // FP16: gamma low byte read

            S_P2_READ_GAMMA_HI:
                state_nxt = S_P2_READ_BETA;          // FP16: gamma done, read beta

            S_P2_READ_BETA:
                state_nxt = S_P2_READ_BETA_HI;       // FP16: beta low byte read

            S_P2_READ_BETA_HI:
                state_nxt = S_P2_COMPUTE;             // FP16: all params read, compute

            S_P2_COMPUTE: begin
                if (r_dtype == 2'd1) begin
                    state_nxt = S_P2_WRITE_HI;       // FP16: need to write high byte too
                end else begin
                    state_nxt = (r_idx == r_length - 16'd1) ? S_DONE : S_P2_READ;
                end
            end

            S_P2_WRITE_HI:
                state_nxt = (r_idx == r_length - 16'd1) ? S_DONE : S_P2_READ;

            S_DONE: state_nxt = S_IDLE;
            default: state_nxt = S_IDLE;
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
            r_beta_base  <= '0;
            r_dtype      <= '0;
            r_recip_n_fp16 <= '0;
        end else if (state == S_IDLE && cmd_valid) begin
            r_length     <= length;
            r_src_base   <= src_base;
            r_dst_base   <= dst_base;
            r_gamma_base <= gamma_base;
            r_beta_base  <= beta_base;
            r_dtype      <= cmd_dtype;
            r_recip_n_fp16 <= recip_n_lut_out;
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
        else if (state == S_P1_FEED && state_nxt == S_P1_READ)
            r_idx <= r_idx + 16'd1;
        else if (state == S_P1_WAIT && mv_valid)
            r_idx <= '0;  // reset for pass 2 (INT8)
        else if (state == S_P1_FP16_FINALIZE)
            r_idx <= '0;  // reset for pass 2 (FP16)
        else if (state == S_P2_COMPUTE && r_dtype == 2'd0 && state_nxt == S_P2_READ)
            r_idx <= r_idx + 16'd1;
        else if (state == S_P2_WRITE_HI && state_nxt == S_P2_READ)
            r_idx <= r_idx + 16'd1;  // FP16: advance after writing both bytes
        else if (state == S_P2_COMPUTE && r_dtype == 2'd0 && state_nxt == S_DONE)
            r_idx <= r_idx;  // hold at final index
    end

    // ----------------------------------------------------------------
    // FP16: Low-byte staging and value assembly
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lo_byte <= '0;
        end else begin
            // Capture low byte on the read state preceding the high-byte read
            if (state == S_P1_READ && r_dtype == 2'd1)
                lo_byte <= sram_rd0_data;
            else if (state == S_P2_READ && r_dtype == 2'd1)
                lo_byte <= sram_rd0_data;
            else if (state == S_P2_READ_GAMMA && r_dtype == 2'd1)
                lo_byte <= sram_rd0_data;
            else if (state == S_P2_READ_BETA && r_dtype == 2'd1)
                lo_byte <= sram_rd1_data;
        end
    end

    // Assemble FP16 value from {hi, lo} bytes
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fp16_rd_val   <= '0;
            fp16_gamma_val <= '0;
            fp16_beta_val  <= '0;
        end else begin
            // Pass 1: assemble input FP16 value
            if (state == S_P1_READ_HI)
                fp16_rd_val <= {sram_rd0_data, lo_byte};

            // Pass 2: assemble input FP16 value
            if (state == S_P2_READ_HI)
                fp16_rd_val <= {sram_rd0_data, lo_byte};

            // Pass 2: assemble gamma FP16 value
            if (state == S_P2_READ_GAMMA_HI)
                fp16_gamma_val <= {sram_rd0_data, lo_byte};

            // Pass 2: assemble beta FP16 value
            if (state == S_P2_READ_BETA_HI)
                fp16_beta_val <= {sram_rd1_data, lo_byte};
        end
    end

    // ----------------------------------------------------------------
    // Pass 1: Feed elements to mean_var_engine (INT8 path)
    // ----------------------------------------------------------------
    assign mv_start     = (state == S_IDLE && cmd_valid && cmd_dtype == 2'd0);
    assign mv_din_valid = (state == S_P1_FEED && r_dtype == 2'd0);
    assign mv_din       = signed'(sram_rd0_data);
    assign mv_din_last  = (state == S_P1_FEED) && (r_idx == r_length - 16'd1) && (r_dtype == 2'd0);

    // Latch mean after pass 1 (INT8 path)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_mean <= '0;
        else if (mv_valid)
            r_mean <= mv_mean;
    end

    // ----------------------------------------------------------------
    // Pass 1: FP16 accumulation (bypass mean_var_engine)
    // Accumulate sum and sum_sq using fp16_add and fp16_mul
    // After all elements:
    //   mean = sum * recip_N
    //   var  = sum_sq * recip_N - mean * mean
    // ----------------------------------------------------------------
    logic [15:0] fp16_x_sq;   // x*x in FP16

    always_comb begin
        fp16_x_sq = fp16_mul(fp16_rd_val, fp16_rd_val);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_fp16_sum    <= 16'h0000;  // FP16 zero
            r_fp16_sq_sum <= 16'h0000;
        end else if (state == S_IDLE && cmd_valid) begin
            r_fp16_sum    <= 16'h0000;
            r_fp16_sq_sum <= 16'h0000;
        end else if (state == S_P1_FEED && r_dtype == 2'd1) begin
            r_fp16_sum    <= fp16_add(r_fp16_sum, fp16_rd_val);
            r_fp16_sq_sum <= fp16_add(r_fp16_sq_sum, fp16_x_sq);
        end
    end

    // FP16 finalize: compute mean and variance from accumulated sums
    logic [15:0] fp16_mean_val;
    logic [15:0] fp16_e_x2;
    logic [15:0] fp16_mean_sq;
    logic [15:0] fp16_var_val;

    always_comb begin
        // mean = sum * (1/N)
        fp16_mean_val = fp16_mul(r_fp16_sum, r_recip_n_fp16);
        // E[x^2] = sum_sq * (1/N)
        fp16_e_x2     = fp16_mul(r_fp16_sq_sum, r_recip_n_fp16);
        // mean^2
        fp16_mean_sq  = fp16_mul(fp16_mean_val, fp16_mean_val);
        // var = E[x^2] - mean^2
        fp16_var_val  = fp16_sub(fp16_e_x2, fp16_mean_sq);
    end

    // Latch FP16 mean on finalize
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_mean_fp16 <= '0;
        else if (state == S_P1_FP16_FINALIZE)
            r_mean_fp16 <= fp16_mean_val;
    end

    // ----------------------------------------------------------------
    // rsqrt computation
    // INT8: Feed top 8 bits of variance to rsqrt LUT
    // FP16: Feed top 8 bits of fp16 variance to rsqrt_fp16 LUT
    // ----------------------------------------------------------------
    assign rsqrt_addr      = mv_var[31:24];  // INT8 path: top 8 bits as index
    assign rsqrt_fp16_addr = (state == S_RSQRT && r_dtype == 2'd1) ? fp16_var_val[15:8] : 8'd0;

    // Latch rsqrt result
    always_ff @(posedge clk) begin
        if (state == S_RSQRT_LATCH) begin
            if (r_dtype == 2'd0)
                r_inv_std <= rsqrt_data;       // INT8: Q0.16
            else
                r_inv_std_fp16 <= rsqrt_fp16_data;  // FP16: FP16 value
        end
    end

    // ----------------------------------------------------------------
    // SRAM read port control
    // ----------------------------------------------------------------
    always_comb begin
        sram_rd0_en   = 1'b0;
        sram_rd0_addr = '0;
        sram_rd1_en   = 1'b0;
        sram_rd1_addr = '0;

        if (r_dtype == 2'd0) begin
            // ============ INT8 path (unchanged) ============
            if (state == S_P1_READ) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + r_idx;
            end else if (state == S_P2_READ) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + r_idx;
                sram_rd1_en   = 1'b1;
                sram_rd1_addr = r_beta_base + r_idx;
            end
        end else begin
            // ============ FP16 path ============
            // Pass 1: read input as 2 bytes (little-endian: low byte first)
            // Each FP16 element at byte address = src_base + idx*2 + 0/1
            if (state == S_P1_READ) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + {r_idx[14:0], 1'b0};  // idx*2 + 0
            end else if (state == S_P1_READ_HI) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + {r_idx[14:0], 1'b0} + 16'd1;  // idx*2 + 1
            end

            // Pass 2: read input (2B), gamma (2B on rd0), beta (2B on rd1)
            else if (state == S_P2_READ) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + {r_idx[14:0], 1'b0};  // input low
            end else if (state == S_P2_READ_HI) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_src_base + {r_idx[14:0], 1'b0} + 16'd1;  // input high
            end else if (state == S_P2_READ_GAMMA) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_gamma_base + {r_idx[14:0], 1'b0};  // gamma low
            end else if (state == S_P2_READ_GAMMA_HI) begin
                sram_rd0_en   = 1'b1;
                sram_rd0_addr = r_gamma_base + {r_idx[14:0], 1'b0} + 16'd1;  // gamma high
            end else if (state == S_P2_READ_BETA) begin
                sram_rd1_en   = 1'b1;
                sram_rd1_addr = r_beta_base + {r_idx[14:0], 1'b0};  // beta low
            end else if (state == S_P2_READ_BETA_HI) begin
                sram_rd1_en   = 1'b1;
                sram_rd1_addr = r_beta_base + {r_idx[14:0], 1'b0} + 16'd1;  // beta high
            end
        end
    end

    // Pipeline registers for pass 2 data (INT8 path)
    always_ff @(posedge clk) begin
        if (state == S_P2_READ && r_dtype == 2'd0) begin
            p_input <= signed'(sram_rd0_data);
            p_beta  <= signed'(sram_rd1_data);
            // For gamma, we assume it's unity-scaled or stored with input
            // In a full implementation, a third read port or time-multiplexing
            // would fetch gamma. Here we use a default gamma of 1 (=127 in int8 scale).
            p_gamma <= 8'sd127;  // placeholder: unity gamma in int8 scale
        end
    end

    // ----------------------------------------------------------------
    // Pass 2: Normalize computation (INT8 path - unchanged)
    // out = ((in - mean) * inv_std) * gamma + beta -> requant to int8
    // ----------------------------------------------------------------
    logic signed [15:0] centered;       // in - mean
    logic signed [31:0] scaled;         // centered * inv_std
    logic signed [31:0] gamma_applied;  // scaled * gamma
    logic signed [31:0] bias_added;     // + beta
    logic signed [7:0]  norm_result;

    always_comb begin
        // Center: (in[i] - mean) where in is int8 and mean is Q8.8
        // Convert input to Q8.8 first: in << 8
        centered = (16'(signed'(p_input)) <<< 8) - r_mean;  // Q8.8

        // Scale by inv_std (Q0.16): result is Q8.24, take upper bits
        scaled = 32'(signed'(centered)) * 32'(signed'({1'b0, r_inv_std}));
        // scaled is Q8.24 (8.8 * 0.16 = 8.24)
        // Shift right by 16 to get Q8.8
        scaled = scaled >>> 16;

        // Apply gamma (int8, representing scale ~1.0 when gamma=127)
        // gamma_applied = scaled * gamma / 128 (normalize gamma to ~1.0)
        gamma_applied = (scaled * 32'(signed'(p_gamma))) >>> 7;

        // Add beta (int8, sign-extend to Q8.8 by shifting left 8)
        bias_added = gamma_applied + (32'(signed'(p_beta)) <<< 8);

        // Requantize: shift right by 8 to get int8 from Q8.8
        if ((bias_added >>> 8) > 32'sd127)
            norm_result = 8'sd127;
        else if ((bias_added >>> 8) < -32'sd128)
            norm_result = -8'sd128;
        else
            norm_result = bias_added[15:8];
    end

    // ----------------------------------------------------------------
    // Pass 2: Normalize computation (FP16 path)
    // centered = fp16_sub(x, mean)
    // scaled   = fp16_mul(centered, inv_std)
    // gamma_ap = fp16_mul(scaled, gamma)
    // result   = fp16_add(gamma_ap, beta)
    // ----------------------------------------------------------------
    logic [15:0] fp16_centered;
    logic [15:0] fp16_scaled;
    logic [15:0] fp16_gamma_applied;
    logic [15:0] fp16_norm_result;

    always_comb begin
        fp16_centered     = fp16_sub(fp16_rd_val, r_mean_fp16);
        fp16_scaled       = fp16_mul(fp16_centered, r_inv_std_fp16);
        fp16_gamma_applied = fp16_mul(fp16_scaled, fp16_gamma_val);
        fp16_norm_result  = fp16_add(fp16_gamma_applied, fp16_beta_val);
    end

    // Register to hold FP16 result for 2-byte write
    logic [15:0] r_fp16_wr_val;

    always_ff @(posedge clk) begin
        if (state == S_P2_COMPUTE && r_dtype == 2'd1)
            r_fp16_wr_val <= fp16_norm_result;
    end

    // ----------------------------------------------------------------
    // SRAM write
    // INT8: write during S_P2_COMPUTE (1 byte)
    // FP16: write low byte during S_P2_COMPUTE, high byte during S_P2_WRITE_HI
    // ----------------------------------------------------------------
    always_comb begin
        sram_wr_en   = 1'b0;
        sram_wr_addr = '0;
        sram_wr_data = '0;

        if (r_dtype == 2'd0) begin
            // INT8 path
            sram_wr_en   = (state == S_P2_COMPUTE);
            sram_wr_addr = r_dst_base + r_idx;
            sram_wr_data = norm_result;
        end else begin
            // FP16 path: write 2 bytes per element (little-endian)
            if (state == S_P2_COMPUTE) begin
                sram_wr_en   = 1'b1;
                sram_wr_addr = r_dst_base + {r_idx[14:0], 1'b0};      // low byte
                sram_wr_data = fp16_norm_result[7:0];
            end else if (state == S_P2_WRITE_HI) begin
                sram_wr_en   = 1'b1;
                sram_wr_addr = r_dst_base + {r_idx[14:0], 1'b0} + 16'd1;  // high byte
                sram_wr_data = r_fp16_wr_val[15:8];
            end
        end
    end

    // ----------------------------------------------------------------
    // Status and handshake
    // ----------------------------------------------------------------
    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
