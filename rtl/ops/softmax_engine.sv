// =============================================================================
// softmax_engine.sv - Full softmax over a row (3-pass architecture)
// Pass 1: Find max value (reduce_max for INT8, inline fp16_cmp_gt for FP16)
// Pass 2: Subtract max, exp via LUT, accumulate sum
// Pass 3: Normalize by reciprocal of sum, requantize to int8 / fp16
//
// Supports both INT8 (r_dtype==0) and FP16 (r_dtype==1) data paths.
// SRAM is 8-bit wide, so FP16 values require 2 reads/writes per element
// (low byte first, then high byte).
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;
import fp16_utils_pkg::*;

module softmax_engine (
    input  logic               clk,
    input  logic               rst_n,

    // Command interface
    input  logic               cmd_valid,
    output logic               cmd_ready,
    input  logic [15:0]        length,          // number of elements in row
    input  logic [15:0]        src_base,        // SRAM base address for input scores
    input  logic [15:0]        dst_base,        // SRAM base address for output probs
    input  logic [15:0]        scale_factor,    // attention scale (Q8.8)
    input  logic               causal_mask_en,  // enable causal masking
    input  logic [15:0]        causal_limit,    // mask positions > limit to -128
    input  logic [1:0]         cmd_dtype,       // 0=INT8, 1=FP16

    // SRAM read port (shared for all passes)
    output logic               sram_rd_en,
    output logic [15:0]        sram_rd_addr,
    input  logic [DATA_W-1:0]  sram_rd_data,

    // SRAM write port (shared for exp scratch and final output)
    output logic               sram_wr_en,
    output logic [15:0]        sram_wr_addr,
    output logic [DATA_W-1:0]  sram_wr_data,

    // Scratch SRAM write port (for intermediate exp values, 16-bit packed)
    output logic               scratch_wr_en,
    output logic [15:0]        scratch_wr_addr,
    output logic [15:0]        scratch_wr_data,

    // Scratch SRAM read port
    output logic               scratch_rd_en,
    output logic [15:0]        scratch_rd_addr,
    input  logic [15:0]        scratch_rd_data,

    // Status
    output logic               busy,
    output logic               done
);

    // ----------------------------------------------------------------
    // FSM States
    // ----------------------------------------------------------------
    typedef enum logic [4:0] {
        S_IDLE,
        // Pass 1: find max
        S_P1_READ,          // Drive SRAM read address (low byte / INT8 byte)
        S_P1_READ_HI,       // FP16: drive SRAM read address for high byte
        S_P1_FEED,          // SRAM data valid, feed to reduce_max / FP16 compare
        S_P1_WAIT,          // Wait for max_valid (INT8 only)
        // Pass 2: exp + sum
        S_P2_READ,          // Drive SRAM read address (low byte / INT8 byte)
        S_P2_READ_HI,       // FP16: drive SRAM read address for high byte
        S_P2_EXP,           // SRAM data valid, drive exp LUT address
        S_P2_FEED,          // Exp data valid, feed to sum, write scratch
        S_P2_WAIT,          // Wait for sum_valid (INT8 only)
        // Pass 3: normalize
        S_P3_RECIP,         // Drive recip LUT address
        S_P3_RECIP_WAIT,    // Recip data valid, latch
        S_P3_READ,          // Drive scratch read address (or SRAM low byte for FP16 output read-back)
        S_P3_READ_HI,       // FP16: read high byte of scratch (unused - scratch is 16-bit)
        S_P3_NORM,          // Scratch data valid, normalize and write
        S_P3_WRITE_HI,      // FP16: write high byte of output
        S_DONE
    } state_t;

    state_t state, state_nxt;

    // ----------------------------------------------------------------
    // Registered command parameters
    // ----------------------------------------------------------------
    logic [15:0] r_length, r_src_base, r_dst_base, r_scale_factor, r_causal_limit;
    logic        r_causal_mask_en;
    logic [15:0] r_idx;
    logic [1:0]  r_dtype;           // Latched dtype: 0=INT8, 1=FP16

    // ----------------------------------------------------------------
    // FP16 internal registers
    // ----------------------------------------------------------------
    logic [15:0] r_max_val_fp16;    // FP16 max value for pass 1
    logic [15:0] r_sum_val_fp16;    // FP16 sum for pass 2
    logic [15:0] r_recip_val_fp16;  // FP16 reciprocal for pass 3
    logic [7:0]  lo_byte;           // Low byte staging for 2-byte SRAM reads
    logic [15:0] fp16_rd_val;       // Assembled FP16 value from 2 reads

    // ----------------------------------------------------------------
    // Sub-module wires (INT8 path)
    // ----------------------------------------------------------------
    // reduce_max
    logic              rmax_start;
    logic              rmax_din_valid;
    logic signed [7:0] rmax_din;
    logic              rmax_din_last;
    logic signed [7:0] rmax_result;
    logic              rmax_valid;
    logic signed [7:0] r_max_val;  // latched max (INT8)

    // exp_lut (INT8)
    logic [7:0]        exp_addr;
    logic [15:0]       exp_data;

    // graph_exp_lut_fp16
    logic [7:0]        exp_fp16_addr;
    logic [15:0]       exp_fp16_data;

    // reduce_sum
    logic              rsum_start;
    logic              rsum_din_valid;
    logic signed [15:0] rsum_din;
    logic              rsum_din_last;
    logic signed [31:0] rsum_result;
    logic              rsum_valid;
    logic [15:0]       r_sum_val;  // latched sum (unsigned portion, INT8)

    // recip_lut
    logic [7:0]        recip_addr;
    logic [15:0]       recip_data;
    logic [15:0]       r_recip_val; // latched reciprocal (INT8)

    // Pipeline registers
    logic signed [7:0] p_rd_data;   // latched SRAM read data
    logic [15:0]       p_exp_val;   // latched exp LUT output
    logic [15:0]       p_scratch;   // latched scratch read data

    // FP16 intermediate computation wires
    logic [15:0]       fp16_diff;        // fp16_sub result (x - max)
    logic [15:0]       fp16_norm_result; // normalized FP16 output

    // ----------------------------------------------------------------
    // Convenience: is this FP16 mode?
    // ----------------------------------------------------------------
    wire is_fp16 = (r_dtype == 2'd1);

    // ----------------------------------------------------------------
    // Sub-module instantiations
    // ----------------------------------------------------------------
    reduce_max u_reduce_max (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (rmax_start),
        .din_valid (rmax_din_valid),
        .din       (rmax_din),
        .din_last  (rmax_din_last),
        .max_val   (rmax_result),
        .max_valid (rmax_valid)
    );

    exp_lut u_exp_lut (
        .clk      (clk),
        .addr     (exp_addr),
        .data_out (exp_data)
    );

    graph_exp_lut_fp16 u_exp_lut_fp16 (
        .clk      (clk),
        .addr     (exp_fp16_addr),
        .data_out (exp_fp16_data)
    );

    reduce_sum u_reduce_sum (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (rsum_start),
        .din_valid (rsum_din_valid),
        .din       (rsum_din),
        .din_last  (rsum_din_last),
        .sum_val   (rsum_result),
        .sum_valid (rsum_valid)
    );

    recip_lut u_recip_lut (
        .clk      (clk),
        .addr     (recip_addr),
        .data_out (recip_data)
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
            S_IDLE: if (cmd_valid) state_nxt = S_P1_READ;

            // ----- Pass 1: Find max -----
            S_P1_READ: begin
                if (is_fp16)
                    state_nxt = S_P1_READ_HI;   // need second byte
                else
                    state_nxt = S_P1_FEED;
            end

            S_P1_READ_HI: state_nxt = S_P1_FEED;

            S_P1_FEED: begin
                if (r_idx == r_length - 16'd1) begin
                    if (is_fp16)
                        state_nxt = S_P2_READ;   // FP16: max tracked inline, skip S_P1_WAIT
                    else
                        state_nxt = S_P1_WAIT;
                end else begin
                    state_nxt = S_P1_READ;
                end
            end

            S_P1_WAIT: if (rmax_valid) state_nxt = S_P2_READ;

            // ----- Pass 2: exp + sum -----
            S_P2_READ: begin
                if (is_fp16)
                    state_nxt = S_P2_READ_HI;
                else
                    state_nxt = S_P2_EXP;
            end

            S_P2_READ_HI: state_nxt = S_P2_EXP;

            S_P2_EXP:  state_nxt = S_P2_FEED;

            S_P2_FEED: begin
                if (r_idx == r_length - 16'd1) begin
                    if (is_fp16)
                        state_nxt = S_P3_RECIP;  // FP16: sum tracked inline, skip S_P2_WAIT
                    else
                        state_nxt = S_P2_WAIT;
                end else begin
                    state_nxt = S_P2_READ;
                end
            end

            S_P2_WAIT: if (rsum_valid) state_nxt = S_P3_RECIP;

            // ----- Pass 3: normalize -----
            S_P3_RECIP:      state_nxt = S_P3_RECIP_WAIT;
            S_P3_RECIP_WAIT: state_nxt = S_P3_READ;

            S_P3_READ: begin
                // Scratch SRAM is 16-bit wide, so no S_P3_READ_HI needed
                state_nxt = S_P3_NORM;
            end

            S_P3_READ_HI: state_nxt = S_P3_NORM; // reserved, not used currently

            S_P3_NORM: begin
                if (is_fp16)
                    state_nxt = S_P3_WRITE_HI;  // write low byte now, high byte next
                else begin
                    if (r_idx == r_length - 16'd1)
                        state_nxt = S_DONE;
                    else
                        state_nxt = S_P3_READ;
                end
            end

            S_P3_WRITE_HI: begin
                if (r_idx == r_length - 16'd1)
                    state_nxt = S_DONE;
                else
                    state_nxt = S_P3_READ;
            end

            S_DONE:    state_nxt = S_IDLE;
            default:   state_nxt = S_IDLE;
        endcase
    end

    // ----------------------------------------------------------------
    // Command capture
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_length        <= '0;
            r_src_base      <= '0;
            r_dst_base      <= '0;
            r_scale_factor  <= '0;
            r_causal_mask_en<= 1'b0;
            r_causal_limit  <= '0;
            r_dtype         <= '0;
        end else if (state == S_IDLE && cmd_valid) begin
            r_length        <= length;
            r_src_base      <= src_base;
            r_dst_base      <= dst_base;
            r_scale_factor  <= scale_factor;
            r_causal_mask_en<= causal_mask_en;
            r_causal_limit  <= causal_limit;
            r_dtype         <= cmd_dtype;
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
        // Pass 1 -> next element
        else if (state == S_P1_FEED && state_nxt == S_P1_READ)
            r_idx <= r_idx + 16'd1;
        // Pass 1 done -> reset for pass 2
        else if (!is_fp16 && state == S_P1_WAIT && rmax_valid)
            r_idx <= '0;
        else if (is_fp16 && state == S_P1_FEED && r_idx == r_length - 16'd1)
            r_idx <= '0;  // FP16: reset for pass 2 at end of pass 1
        // Pass 2 -> next element
        else if (state == S_P2_FEED && state_nxt == S_P2_READ)
            r_idx <= r_idx + 16'd1;
        // Pass 2 done -> reset for pass 3
        else if (!is_fp16 && state == S_P2_WAIT && rsum_valid)
            r_idx <= '0;
        else if (is_fp16 && state == S_P2_FEED && r_idx == r_length - 16'd1)
            r_idx <= '0;  // FP16: reset for pass 3 at end of pass 2
        // Pass 3 -> next element (INT8: after S_P3_NORM, FP16: after S_P3_WRITE_HI)
        else if (!is_fp16 && state == S_P3_NORM && state_nxt == S_P3_READ)
            r_idx <= r_idx + 16'd1;
        else if (is_fp16 && state == S_P3_WRITE_HI && state_nxt == S_P3_READ)
            r_idx <= r_idx + 16'd1;
    end

    // ----------------------------------------------------------------
    // Latch max value after pass 1 (INT8)
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_max_val <= -8'sd128;
        else if (rmax_valid)
            r_max_val <= rmax_result;
    end

    // ----------------------------------------------------------------
    // FP16 max tracking (pass 1) - inline using fp16_cmp_gt
    // Initialize to 0xFC00 (negative infinity in FP16)
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_max_val_fp16 <= 16'hFC00;
        else if (state == S_IDLE && cmd_valid)
            r_max_val_fp16 <= 16'hFC00;  // reset to -inf at start
        else if (is_fp16 && state == S_P1_FEED) begin
            // Compare current element with running max
            // Use fp16_assembled (combinational) since fp16_rd_val hasn't
            // been registered yet at this point in the cycle
            if (r_causal_mask_en && r_idx > r_causal_limit) begin
                // Masked position: treat as -inf, don't update max
            end else if (fp16_cmp_gt(fp16_assembled, r_max_val_fp16)) begin
                r_max_val_fp16 <= fp16_assembled;
            end
        end
    end

    // ----------------------------------------------------------------
    // FP16 sum accumulation (pass 2) - inline using fp16_add
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_sum_val_fp16 <= 16'h0000;
        else if (state == S_P1_FEED && is_fp16 && r_idx == r_length - 16'd1)
            r_sum_val_fp16 <= 16'h0000;  // reset sum at end of pass 1
        else if (!is_fp16 && state == S_P1_WAIT && rmax_valid)
            r_sum_val_fp16 <= 16'h0000;  // also reset on INT8 path transition (benign)
        else if (is_fp16 && state == S_P2_FEED)
            r_sum_val_fp16 <= fp16_add(r_sum_val_fp16, exp_fp16_data);
    end

    // ----------------------------------------------------------------
    // FP16 reciprocal latch (pass 3)
    // Convert recip_lut output (Q0.16 unsigned) to FP16.
    // recip_lut[i] = round(65536 / i), so the value represents 1/sum_approx.
    // We convert this Q0.16 value to FP16 for use with fp16_mul.
    // ----------------------------------------------------------------
    // Q0.16 to FP16 conversion helper: value / 65536.0
    // The recip_lut output is an unsigned 16-bit integer representing a Q0.16
    // fractional value. To convert to FP16: treat it as an integer, convert
    // to FP16, then multiply by FP16(1/256) twice (or FP16(1/65536)).
    // Simpler: use the int-to-fp16 approach for the upper byte, then adjust exponent.
    //
    // Actually, simplest practical approach: we use the recip_lut output as-is
    // for INT8, and for FP16 we convert the 16-bit Q0.16 recip value to FP16
    // by finding the MSB and building the float.
    // ----------------------------------------------------------------
    function automatic logic [15:0] q016_to_fp16(input logic [15:0] val);
        // Convert unsigned Q0.16 fixed-point to FP16
        // Q0.16 means the value = val / 65536.0, range [0, 1)
        // Special case: val=0 -> +0.0
        logic [15:0] result;
        logic [4:0]  msb_pos;
        logic [4:0]  fp16_exp;
        logic [9:0]  fp16_mant;
        integer i;

        if (val == 16'h0000) begin
            result = 16'h0000;
            return result;
        end

        // Find MSB position (highest set bit, 0..15)
        msb_pos = 5'd0;
        for (i = 15; i >= 0; i = i - 1) begin
            if (val[i]) begin
                msb_pos = i[4:0];
                break;
            end
        end

        // The true value is val * 2^(-16)
        // If MSB is at position msb_pos, then val = 1.frac * 2^msb_pos
        // So true value = 1.frac * 2^(msb_pos - 16)
        // FP16 exponent = (msb_pos - 16) + 15 = msb_pos - 1
        // Since msb_pos is 0..15, fp16_exp ranges from -1 to 14
        // For msb_pos == 0: exponent = -1 -> underflow to zero (denormal)
        if (msb_pos == 5'd0) begin
            // Value = 1 * 2^(-16), too small for FP16 normals -> flush to zero
            result = 16'h0000;
            return result;
        end

        fp16_exp = msb_pos - 5'd1; // biased exponent (bias=15, unbiased = msb_pos-16, biased = msb_pos-16+15 = msb_pos-1)

        // Extract mantissa: remove implicit 1, left-align into 10 bits
        // The bits below msb_pos form the fraction
        case (msb_pos)
            5'd1:  fp16_mant = {val[0],    9'h000};
            5'd2:  fp16_mant = {val[1:0],  8'h00};
            5'd3:  fp16_mant = {val[2:0],  7'h00};
            5'd4:  fp16_mant = {val[3:0],  6'h00};
            5'd5:  fp16_mant = {val[4:0],  5'h00};
            5'd6:  fp16_mant = {val[5:0],  4'h0};
            5'd7:  fp16_mant = {val[6:0],  3'h0};
            5'd8:  fp16_mant = {val[7:0],  2'h0};
            5'd9:  fp16_mant = {val[8:0],  1'h0};
            5'd10: fp16_mant = val[9:0];
            5'd11: fp16_mant = val[10:1];
            5'd12: fp16_mant = val[11:2];
            5'd13: fp16_mant = val[12:3];
            5'd14: fp16_mant = val[13:4];
            5'd15: fp16_mant = val[14:5];
            default: fp16_mant = 10'h000;
        endcase

        result = {1'b0, fp16_exp, fp16_mant}; // always positive
        return result;
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_recip_val_fp16 <= 16'h0000;
        else if (state == S_P3_RECIP_WAIT && is_fp16)
            r_recip_val_fp16 <= q016_to_fp16(recip_data);
    end

    // ----------------------------------------------------------------
    // Latch sum value after pass 2 (take lower 16 bits as Q8.8) - INT8
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            r_sum_val <= '0;
        else if (rsum_valid)
            r_sum_val <= rsum_result[15:0];
    end

    // ----------------------------------------------------------------
    // Latch reciprocal value after pass 3 recip LUT read - INT8
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state == S_P3_RECIP_WAIT && !is_fp16)
            r_recip_val <= recip_data;
    end

    // ----------------------------------------------------------------
    // FP16 2-byte read staging: latch low byte, assemble on high byte
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        // Latch low byte when it arrives (1 cycle after S_P1_READ / S_P2_READ)
        // Data from S_P1_READ arrives when state == S_P1_READ_HI
        // Data from S_P2_READ arrives when state == S_P2_READ_HI
        if (state == S_P1_READ_HI || state == S_P2_READ_HI)
            lo_byte <= sram_rd_data;
    end

    // Combinational assembly of the full FP16 value from {high_byte, lo_byte}.
    // High byte arrives from SRAM 1 cycle after S_P1_READ_HI / S_P2_READ_HI,
    // i.e. when state == S_P1_FEED or S_P2_EXP respectively.
    // This combinational signal is used immediately for exp LUT addressing (pass 2)
    // and max comparison (pass 1).
    logic [15:0] fp16_assembled;
    always_comb begin
        fp16_assembled = {sram_rd_data, lo_byte};
    end

    // Register the assembled value for use in subsequent cycles
    always_ff @(posedge clk) begin
        if (is_fp16 && (state == S_P1_FEED || state == S_P2_EXP))
            fp16_rd_val <= fp16_assembled;
    end

    // ----------------------------------------------------------------
    // Pipeline register: latch SRAM read data (INT8 path)
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!is_fp16 && (state == S_P1_FEED || state == S_P2_EXP))
            p_rd_data <= signed'(sram_rd_data);
    end

    // Latch exp LUT output (available 1 cycle after addr driven)
    always_ff @(posedge clk) begin
        p_exp_val <= exp_data;
    end

    // Latch scratch read data
    always_ff @(posedge clk) begin
        if (state == S_P3_READ)
            p_scratch <= scratch_rd_data;
    end

    // ----------------------------------------------------------------
    // SRAM read port control
    // ----------------------------------------------------------------
    // In FP16 mode, each element occupies 2 bytes at addr = base + idx*2
    // and base + idx*2 + 1. Read low byte first, then high byte.
    always_comb begin
        sram_rd_en   = 1'b0;
        sram_rd_addr = 16'd0;

        case (state)
            S_P1_READ: begin
                sram_rd_en = 1'b1;
                if (is_fp16)
                    sram_rd_addr = r_src_base + {r_idx[14:0], 1'b0};       // idx*2 (low byte)
                else
                    sram_rd_addr = r_src_base + r_idx;
            end

            S_P1_READ_HI: begin
                sram_rd_en   = 1'b1;
                sram_rd_addr = r_src_base + {r_idx[14:0], 1'b0} + 16'd1;  // idx*2+1 (high byte)
            end

            S_P2_READ: begin
                sram_rd_en = 1'b1;
                if (is_fp16)
                    sram_rd_addr = r_src_base + {r_idx[14:0], 1'b0};
                else
                    sram_rd_addr = r_src_base + r_idx;
            end

            S_P2_READ_HI: begin
                sram_rd_en   = 1'b1;
                sram_rd_addr = r_src_base + {r_idx[14:0], 1'b0} + 16'd1;
            end

            default: begin
                sram_rd_en   = 1'b0;
                sram_rd_addr = 16'd0;
            end
        endcase
    end

    // ----------------------------------------------------------------
    // Pass 1: reduce_max control (INT8 only)
    // ----------------------------------------------------------------
    assign rmax_start     = (state == S_IDLE && cmd_valid && cmd_dtype == 2'd0);
    assign rmax_din_valid = (!is_fp16 && state == S_P1_FEED);

    // Apply causal mask: if position > causal_limit, feed -128 (INT8) or -inf (FP16)
    logic signed [7:0] masked_input;
    always_comb begin
        if (r_causal_mask_en && r_idx > r_causal_limit)
            masked_input = -8'sd128;
        else
            masked_input = signed'(sram_rd_data);
    end

    assign rmax_din      = masked_input;
    assign rmax_din_last = (!is_fp16 && state == S_P1_FEED) && (r_idx == r_length - 16'd1);

    // ----------------------------------------------------------------
    // FP16 causal-masked input value
    // Uses fp16_assembled (combinational) for immediate availability
    // during S_P1_FEED and S_P2_EXP
    // ----------------------------------------------------------------
    logic [15:0] fp16_masked_input;
    always_comb begin
        if (r_causal_mask_en && r_idx > r_causal_limit)
            fp16_masked_input = 16'hFC00;  // -inf in FP16
        else
            fp16_masked_input = fp16_assembled;
    end

    // ----------------------------------------------------------------
    // Pass 2: exp + sum
    // ----------------------------------------------------------------
    // INT8: Subtract max and apply scale, then feed to exp LUT
    logic signed [15:0] scaled_diff;
    logic signed [7:0]  exp_input;

    always_comb begin
        // (x - max): both are int8
        scaled_diff = 16'(signed'(masked_input)) - 16'(signed'(r_max_val));
        // Clamp to int8 range for exp LUT input
        if (scaled_diff < -16'sd128)
            exp_input = -8'sd128;
        else if (scaled_diff > 16'sd127)
            exp_input = 8'sd127;
        else
            exp_input = scaled_diff[7:0];
    end

    // Drive INT8 exp LUT address during pass 2
    assign exp_addr = (!is_fp16 && state == S_P2_EXP) ? exp_input : 8'd0;

    // FP16 exp LUT: compute diff = x - max, use upper 8 bits as index
    // fp16_diff is computed combinationally for use in S_P2_EXP
    always_comb begin
        if (is_fp16)
            fp16_diff = fp16_sub(fp16_masked_input, r_max_val_fp16);
        else
            fp16_diff = 16'h0000;
    end

    // Drive FP16 exp LUT address: upper 8 bits of the difference value
    assign exp_fp16_addr = (is_fp16 && state == S_P2_EXP) ? fp16_diff[15:8] : 8'd0;

    // reduce_sum control (INT8 only)
    assign rsum_start     = (!is_fp16 && state == S_P1_WAIT && rmax_valid);
    assign rsum_din_valid = (!is_fp16 && state == S_P2_FEED);
    assign rsum_din       = 16'(signed'(exp_data));  // exp output is unsigned Q8.8, treat as int16
    assign rsum_din_last  = (!is_fp16 && state == S_P2_FEED) && (r_idx == r_length - 16'd1);

    // Write exp values to scratch SRAM during pass 2
    // INT8: write exp_data (16-bit Q8.8)
    // FP16: write exp_fp16_data (16-bit FP16)
    always_comb begin
        scratch_wr_en   = (state == S_P2_FEED);
        scratch_wr_addr = r_idx;
        if (is_fp16)
            scratch_wr_data = exp_fp16_data;
        else
            scratch_wr_data = exp_data;
    end

    // ----------------------------------------------------------------
    // Pass 3: normalize
    // ----------------------------------------------------------------
    // Drive reciprocal LUT with top 8 bits of sum
    // INT8: top 8 bits of Q8.8 sum
    // FP16: top 8 bits of FP16 sum (sign + exponent + top 2 mantissa bits)
    always_comb begin
        if (is_fp16)
            recip_addr = r_sum_val_fp16[15:8];
        else
            recip_addr = r_sum_val[15:8];
    end

    // Read scratch values during pass 3
    assign scratch_rd_en   = (state == S_P3_READ);
    assign scratch_rd_addr = r_idx;

    // ----------------------------------------------------------------
    // INT8 normalize: out = (exp_val * recip) >> 17, clamp to int8
    // exp_val is Q8.8, recip is Q0.16, product is Q8.24
    // >>17 maps [0.0, 1.0] to [0, ~128], clamped to [0, 127]
    // ----------------------------------------------------------------
    logic [31:0] norm_product;
    logic [14:0] norm_trunc;
    logic signed [7:0] norm_result;

    always_comb begin
        norm_product = 32'(scratch_rd_data) * 32'(r_recip_val);
        norm_trunc = norm_product[31:17];  // >>17 for Q8.24 -> Q0.7
        // Clamp to [0, 127] since softmax outputs are non-negative
        if (norm_trunc > 15'd127)
            norm_result = 8'sd127;
        else
            norm_result = norm_trunc[7:0];
    end

    // ----------------------------------------------------------------
    // FP16 normalize: out = scratch_val * recip_fp16
    // Both are FP16, result is FP16
    // ----------------------------------------------------------------
    always_comb begin
        fp16_norm_result = fp16_mul(scratch_rd_data, r_recip_val_fp16);
    end

    // ----------------------------------------------------------------
    // Latch the FP16 norm result so it persists into S_P3_WRITE_HI
    // scratch_rd_data is only valid during S_P3_NORM (1 cycle after
    // S_P3_READ), so we must register the result for the high byte write.
    // ----------------------------------------------------------------
    logic [15:0] r_fp16_norm_out;
    always_ff @(posedge clk) begin
        if (is_fp16 && state == S_P3_NORM)
            r_fp16_norm_out <= fp16_norm_result;
    end

    // ----------------------------------------------------------------
    // SRAM write port control
    // INT8: single byte write during S_P3_NORM
    // FP16: low byte during S_P3_NORM, high byte during S_P3_WRITE_HI
    //       (high byte uses registered r_fp16_norm_out since scratch_rd_data
    //        is no longer valid in S_P3_WRITE_HI)
    // ----------------------------------------------------------------
    always_comb begin
        sram_wr_en   = 1'b0;
        sram_wr_addr = 16'd0;
        sram_wr_data = 8'd0;

        if (!is_fp16 && state == S_P3_NORM) begin
            // INT8: write single byte
            sram_wr_en   = 1'b1;
            sram_wr_addr = r_dst_base + r_idx;
            sram_wr_data = norm_result;
        end else if (is_fp16 && state == S_P3_NORM) begin
            // FP16: write low byte (fp16_norm_result valid combinationally)
            sram_wr_en   = 1'b1;
            sram_wr_addr = r_dst_base + {r_idx[14:0], 1'b0};       // idx*2
            sram_wr_data = fp16_norm_result[7:0];
        end else if (is_fp16 && state == S_P3_WRITE_HI) begin
            // FP16: write high byte (use registered value)
            sram_wr_en   = 1'b1;
            sram_wr_addr = r_dst_base + {r_idx[14:0], 1'b0} + 16'd1; // idx*2+1
            sram_wr_data = r_fp16_norm_out[15:8];
        end
    end

    // ----------------------------------------------------------------
    // Status and handshake
    // ----------------------------------------------------------------
    assign cmd_ready = (state == S_IDLE);
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
