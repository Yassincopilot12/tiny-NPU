`default_nettype none

package fp16_utils_pkg;

    // ----------------------------------------------------------------
    // FP16 add: a + b
    // IEEE 754 half-precision addition, combinational.
    // Uses 14-bit working mantissa (11 significant + 3 guard/round/sticky).
    // ----------------------------------------------------------------
    function automatic logic [15:0] fp16_add(input logic [15:0] a, input logic [15:0] b);
        logic        a_sign, b_sign;
        logic [4:0]  a_exp,  b_exp;
        logic [9:0]  a_mant, b_mant;
        logic        a_is_nan, b_is_nan;
        logic        a_is_inf, b_is_inf;
        logic        a_is_zero, b_is_zero;

        // Working variables
        logic        lg_sign, sm_sign;       // larger / smaller operand signs
        logic [4:0]  lg_exp,  sm_exp;        // larger / smaller exponents
        logic [13:0] lg_mant, sm_mant;       // 14-bit working mantissa: {1, mant[9:0], 3'b000}
        logic [4:0]  exp_diff;
        logic        swap;

        logic [14:0] sum_mant;               // 15-bit to capture carry
        logic        res_sign;
        logic signed [6:0] res_exp;          // signed to detect underflow
        logic [13:0] norm_mant;
        logic [9:0]  final_mant;
        logic [4:0]  final_exp;
        logic [15:0] result;

        logic        do_sub;
        logic [3:0]  lzc;                    // leading zero count
        logic        sticky;

        integer i;

        // Extract fields
        a_sign = a[15];
        a_exp  = a[14:10];
        a_mant = a[9:0];
        b_sign = b[15];
        b_exp  = b[14:10];
        b_mant = b[9:0];

        // Special value detection
        a_is_nan  = (a_exp == 5'h1F) && (a_mant != 10'h0);
        b_is_nan  = (b_exp == 5'h1F) && (b_mant != 10'h0);
        a_is_inf  = (a_exp == 5'h1F) && (a_mant == 10'h0);
        b_is_inf  = (b_exp == 5'h1F) && (b_mant == 10'h0);
        a_is_zero = (a_exp == 5'h00) && (a_mant == 10'h0);
        b_is_zero = (b_exp == 5'h00) && (b_mant == 10'h0);

        // NaN propagation
        if (a_is_nan || b_is_nan) begin
            result = 16'h7FFF; // canonical NaN
            return result;
        end

        // Infinity handling
        if (a_is_inf && b_is_inf) begin
            if (a_sign != b_sign) begin
                result = 16'h7FFF; // inf - inf = NaN
            end else begin
                result = {a_sign, 5'h1F, 10'h0}; // inf + inf = inf (same sign)
            end
            return result;
        end
        if (a_is_inf) begin
            result = a;
            return result;
        end
        if (b_is_inf) begin
            result = b;
            return result;
        end

        // Zero handling
        if (a_is_zero && b_is_zero) begin
            // +0 + -0 = +0 (IEEE rule)
            result = {a_sign & b_sign, 15'h0};
            return result;
        end
        if (a_is_zero) begin
            result = b;
            return result;
        end
        if (b_is_zero) begin
            result = a;
            return result;
        end

        // Flush denormals to zero
        if (a_exp == 5'h00) begin
            result = b;
            return result;
        end
        if (b_exp == 5'h00) begin
            result = a;
            return result;
        end

        // Determine which operand is larger (by exponent, then mantissa)
        if (a_exp > b_exp || (a_exp == b_exp && a_mant >= b_mant)) begin
            swap = 1'b0;
        end else begin
            swap = 1'b1;
        end

        if (swap) begin
            lg_sign = b_sign; lg_exp = b_exp; lg_mant = {1'b1, b_mant, 3'b000};
            sm_sign = a_sign; sm_exp = a_exp; sm_mant = {1'b1, a_mant, 3'b000};
        end else begin
            lg_sign = a_sign; lg_exp = a_exp; lg_mant = {1'b1, a_mant, 3'b000};
            sm_sign = b_sign; sm_exp = b_exp; sm_mant = {1'b1, b_mant, 3'b000};
        end

        exp_diff = lg_exp - sm_exp;

        // Shift smaller mantissa right, collect sticky bits
        sticky = 1'b0;
        for (i = 0; i < exp_diff; i = i + 1) begin
            sticky = sticky | sm_mant[0];
            sm_mant = {1'b0, sm_mant[13:1]};
        end
        sm_mant[0] = sm_mant[0] | sticky;

        // Effective operation
        do_sub = (lg_sign != sm_sign);

        if (do_sub) begin
            sum_mant = {1'b0, lg_mant} - {1'b0, sm_mant};
        end else begin
            sum_mant = {1'b0, lg_mant} + {1'b0, sm_mant};
        end

        res_sign = lg_sign;
        res_exp  = {2'b00, lg_exp};

        // Result is zero
        if (sum_mant == 15'h0) begin
            result = 16'h0000; // +0
            return result;
        end

        // Normalize
        if (sum_mant[14]) begin
            // Overflow: shift right by 1
            sticky = sum_mant[0];
            sum_mant = {1'b0, sum_mant[14:1]};
            sum_mant[0] = sum_mant[0] | sticky;
            res_exp = res_exp + 7'sd1;
        end else begin
            // Count leading zeros in sum_mant[13:0] and shift left
            lzc = 4'd0;
            if      (!sum_mant[13]) begin
                // Need to find leading 1
                lzc = 4'd0;
                for (i = 13; i >= 0; i = i - 1) begin
                    if (!sum_mant[i] && (lzc == (4'd13 - i[3:0])))
                        lzc = lzc + 4'd1;
                end
            end

            // Limit shift so exponent doesn't go below 1
            if (lzc > 0) begin
                if (res_exp - {3'b000, lzc} < 7'sd1) begin
                    // Would underflow -- shift only as much as possible
                    if (res_exp > 7'sd1) begin
                        sum_mant = sum_mant << (res_exp - 7'sd1);
                        res_exp  = 7'sd1;
                    end
                end else begin
                    sum_mant = sum_mant << lzc;
                    res_exp  = res_exp - {3'b000, lzc};
                end
            end
        end

        norm_mant = sum_mant[13:0];

        // Rounding (round-to-nearest-even)
        // GRS bits are norm_mant[2:0]
        // Guard = norm_mant[2], Round = norm_mant[1], Sticky = norm_mant[0]
        begin
            logic guard, round_bit, sticky_bit;
            logic round_up;
            guard     = norm_mant[2];
            round_bit = norm_mant[1];
            sticky_bit = norm_mant[0];

            // Round to nearest even
            round_up = guard & (round_bit | sticky_bit | norm_mant[3]);

            final_mant = norm_mant[12:3];
            if (round_up) begin
                if (final_mant == 10'h3FF) begin
                    final_mant = 10'h000;
                    res_exp = res_exp + 7'sd1;
                end else begin
                    final_mant = final_mant + 10'h001;
                end
            end
        end

        // Overflow to infinity
        if (res_exp >= 7'sd31) begin
            result = {res_sign, 5'h1F, 10'h0};
            return result;
        end

        // Underflow to zero
        if (res_exp <= 7'sd0) begin
            result = {res_sign, 15'h0};
            return result;
        end

        final_exp = res_exp[4:0];
        result = {res_sign, final_exp, final_mant};
        return result;
    endfunction


    // ----------------------------------------------------------------
    // FP16 subtract: a - b (flip sign of b and add)
    // ----------------------------------------------------------------
    function automatic logic [15:0] fp16_sub(input logic [15:0] a, input logic [15:0] b);
        return fp16_add(a, {~b[15], b[14:0]});
    endfunction


    // ----------------------------------------------------------------
    // FP16 multiply: a * b
    // IEEE 754 half-precision multiplication, combinational.
    // ----------------------------------------------------------------
    function automatic logic [15:0] fp16_mul(input logic [15:0] a, input logic [15:0] b);
        logic        a_sign, b_sign, res_sign;
        logic [4:0]  a_exp,  b_exp;
        logic [9:0]  a_mant, b_mant;
        logic        a_is_nan, b_is_nan;
        logic        a_is_inf, b_is_inf;
        logic        a_is_zero, b_is_zero;

        logic [10:0] a_full, b_full;        // 11-bit mantissa with implicit 1
        logic [21:0] product;               // 22-bit product
        logic signed [6:0] res_exp;
        logic [9:0]  final_mant;
        logic [4:0]  final_exp;
        logic [15:0] result;

        // Extract fields
        a_sign = a[15];
        a_exp  = a[14:10];
        a_mant = a[9:0];
        b_sign = b[15];
        b_exp  = b[14:10];
        b_mant = b[9:0];

        res_sign = a_sign ^ b_sign;

        // Special value detection
        a_is_nan  = (a_exp == 5'h1F) && (a_mant != 10'h0);
        b_is_nan  = (b_exp == 5'h1F) && (b_mant != 10'h0);
        a_is_inf  = (a_exp == 5'h1F) && (a_mant == 10'h0);
        b_is_inf  = (b_exp == 5'h1F) && (b_mant == 10'h0);
        a_is_zero = (a_exp == 5'h00);  // includes denormals flushed to zero
        b_is_zero = (b_exp == 5'h00);

        // NaN propagation
        if (a_is_nan || b_is_nan) begin
            result = 16'h7FFF;
            return result;
        end

        // inf * 0 = NaN
        if ((a_is_inf && b_is_zero) || (b_is_inf && a_is_zero)) begin
            result = 16'h7FFF;
            return result;
        end

        // Infinity
        if (a_is_inf || b_is_inf) begin
            result = {res_sign, 5'h1F, 10'h0};
            return result;
        end

        // Zero
        if (a_is_zero || b_is_zero) begin
            result = {res_sign, 15'h0};
            return result;
        end

        // Normal multiplication
        a_full = {1'b1, a_mant};
        b_full = {1'b1, b_mant};
        product = a_full * b_full; // 22-bit result: range [1.0, ~4.0) in Q1.21 or Q2.20

        // Compute raw exponent: ea + eb - bias
        res_exp = {2'b00, a_exp} + {2'b00, b_exp} - 7'sd15;

        // Normalize: product is in [1.0*1.0, ~2.0*2.0) = [1.0, ~4.0)
        // If bit 21 is set, product >= 2.0: shift right and increment exponent
        if (product[21]) begin
            // Product in range [2.0, 4.0)
            // Mantissa bits: product[20:11], GRS from product[10:0]
            begin
                logic guard, round_bit, sticky_bit, round_up;
                guard     = product[10];
                round_bit = product[9];
                sticky_bit = |product[8:0];
                round_up  = guard & (round_bit | sticky_bit | product[11]);

                final_mant = product[20:11];
                if (round_up) begin
                    if (final_mant == 10'h3FF) begin
                        final_mant = 10'h000;
                        res_exp = res_exp + 7'sd1;
                    end else begin
                        final_mant = final_mant + 10'h001;
                    end
                end
            end
            res_exp = res_exp + 7'sd1;
        end else begin
            // Product in range [1.0, 2.0)
            // Mantissa bits: product[19:10], GRS from product[9:0]
            begin
                logic guard, round_bit, sticky_bit, round_up;
                guard     = product[9];
                round_bit = product[8];
                sticky_bit = |product[7:0];
                round_up  = guard & (round_bit | sticky_bit | product[10]);

                final_mant = product[19:10];
                if (round_up) begin
                    if (final_mant == 10'h3FF) begin
                        final_mant = 10'h000;
                        res_exp = res_exp + 7'sd1;
                    end else begin
                        final_mant = final_mant + 10'h001;
                    end
                end
            end
        end

        // Overflow
        if (res_exp >= 7'sd31) begin
            result = {res_sign, 5'h1F, 10'h0};
            return result;
        end

        // Underflow
        if (res_exp <= 7'sd0) begin
            result = {res_sign, 15'h0};
            return result;
        end

        final_exp = res_exp[4:0];
        result = {res_sign, final_exp, final_mant};
        return result;
    endfunction


    // ----------------------------------------------------------------
    // FP16 compare greater than (signed magnitude)
    // Returns 1 if a > b, 0 otherwise. NaN is not greater than anything.
    // ----------------------------------------------------------------
    function automatic logic fp16_cmp_gt(input logic [15:0] a, input logic [15:0] b);
        logic        a_sign, b_sign;
        logic [4:0]  a_exp,  b_exp;
        logic [9:0]  a_mant, b_mant;
        logic        a_is_nan, b_is_nan;
        logic        result;

        a_sign = a[15];
        a_exp  = a[14:10];
        a_mant = a[9:0];
        b_sign = b[15];
        b_exp  = b[14:10];
        b_mant = b[9:0];

        a_is_nan = (a_exp == 5'h1F) && (a_mant != 10'h0);
        b_is_nan = (b_exp == 5'h1F) && (b_mant != 10'h0);

        // NaN is unordered
        if (a_is_nan || b_is_nan) begin
            result = 1'b0;
            return result;
        end

        // Both zeros (regardless of sign)
        if ((a_exp == 5'h00 && a_mant == 10'h0) && (b_exp == 5'h00 && b_mant == 10'h0)) begin
            result = 1'b0;
            return result;
        end

        // Different signs: positive > negative
        if (a_sign != b_sign) begin
            result = b_sign; // a > b iff b is negative (b_sign == 1)
            return result;
        end

        // Same sign: compare magnitude
        if (a_sign == 1'b0) begin
            // Both positive: larger magnitude is greater
            if (a_exp != b_exp)
                result = (a_exp > b_exp);
            else
                result = (a_mant > b_mant);
        end else begin
            // Both negative: smaller magnitude is greater
            if (a_exp != b_exp)
                result = (a_exp < b_exp);
            else
                result = (a_mant < b_mant);
        end
        return result;
    endfunction


    // ----------------------------------------------------------------
    // INT8 (signed) to FP16: exact conversion
    // Signed int8 range [-128, 127] is always exactly representable in FP16.
    // ----------------------------------------------------------------
    function automatic logic [15:0] int8_to_fp16(input logic [7:0] val);
        logic        sign;
        logic [7:0]  abs_val;
        logic [4:0]  exp_out;
        logic [9:0]  mant_out;
        logic [15:0] result;
        logic [3:0]  msb_pos;
        integer i;

        // Zero
        if (val == 8'h00) begin
            result = 16'h0000;
            return result;
        end

        // Determine sign and absolute value
        sign = val[7];
        if (sign) begin
            abs_val = (~val) + 8'h01; // two's complement negate
        end else begin
            abs_val = val;
        end

        // abs_val is now in [1, 128]
        // Find position of MSB (highest set bit)
        msb_pos = 4'd0;
        for (i = 7; i >= 0; i = i - 1) begin
            if (abs_val[i]) begin
                msb_pos = i[3:0];
                break;
            end
        end

        // Exponent = msb_pos + bias(15)
        exp_out = {1'b0, msb_pos} + 5'd15;

        // Mantissa: remove implicit 1 and left-align into 10-bit field
        // Shift abs_val left so that the MSB (implicit 1) is removed
        // and remaining bits fill mantissa from MSB side
        mant_out = 10'h000;
        case (msb_pos)
            4'd0: mant_out = 10'h000; // value = 1, no fraction bits
            4'd1: mant_out = {abs_val[0],   9'h000};
            4'd2: mant_out = {abs_val[1:0], 8'h00};
            4'd3: mant_out = {abs_val[2:0], 7'h00};
            4'd4: mant_out = {abs_val[3:0], 6'h00};
            4'd5: mant_out = {abs_val[4:0], 5'h00};
            4'd6: mant_out = {abs_val[5:0], 4'h0};
            4'd7: mant_out = {abs_val[6:0], 3'h0};
            default: mant_out = 10'h000;
        endcase

        result = {sign, exp_out, mant_out};
        return result;
    endfunction


    // ----------------------------------------------------------------
    // FP32 to FP16: truncate with round-to-nearest-even
    // ----------------------------------------------------------------
    function automatic logic [15:0] fp32_to_fp16(input logic [31:0] val);
        logic        sign;
        logic [7:0]  fp32_exp;
        logic [22:0] fp32_mant;
        logic        is_nan, is_inf, is_zero;
        logic signed [8:0] unbiased_exp;
        logic signed [8:0] fp16_exp_s;
        logic [4:0]  fp16_exp;
        logic [9:0]  fp16_mant;
        logic [15:0] result;

        // Extract FP32 fields
        sign      = val[31];
        fp32_exp  = val[30:23];
        fp32_mant = val[22:0];

        is_nan  = (fp32_exp == 8'hFF) && (fp32_mant != 23'h0);
        is_inf  = (fp32_exp == 8'hFF) && (fp32_mant == 23'h0);
        is_zero = (fp32_exp == 8'h00) && (fp32_mant == 23'h0);

        // NaN -> canonical FP16 NaN
        if (is_nan) begin
            result = 16'h7FFF;
            return result;
        end

        // Infinity -> FP16 infinity
        if (is_inf) begin
            result = {sign, 5'h1F, 10'h0};
            return result;
        end

        // Zero
        if (is_zero || fp32_exp == 8'h00) begin
            result = {sign, 15'h0};
            return result;
        end

        // Unbiased exponent: fp32_exp - 127
        unbiased_exp = signed'({1'b0, fp32_exp}) - signed'(9'd127);

        // FP16 biased exponent: unbiased + 15
        fp16_exp_s = unbiased_exp + signed'(9'd15);

        // Overflow -> infinity
        if (fp16_exp_s >= signed'(9'd31)) begin
            result = {sign, 5'h1F, 10'h0};
            return result;
        end

        // Underflow -> zero
        if (fp16_exp_s <= signed'(9'd0)) begin
            result = {sign, 15'h0};
            return result;
        end

        fp16_exp = fp16_exp_s[4:0];

        // Round-to-nearest-even: GRS from fp32_mant[12:0]
        // fp32_mant[22:13] = 10-bit mantissa for FP16
        // fp32_mant[12]    = guard, fp32_mant[11] = round, fp32_mant[10:0] = sticky
        begin
            logic guard, round_bit, sticky_bit, round_up;
            guard      = fp32_mant[12];
            round_bit  = fp32_mant[11];
            sticky_bit = |fp32_mant[10:0];
            round_up   = guard & (round_bit | sticky_bit | fp32_mant[13]);

            fp16_mant = fp32_mant[22:13];
            if (round_up) begin
                if (fp16_mant == 10'h3FF) begin
                    fp16_mant = 10'h000;
                    fp16_exp  = fp16_exp + 5'd1;
                    if (fp16_exp == 5'h1F) begin
                        result = {sign, 5'h1F, 10'h0}; // overflow to inf
                        return result;
                    end
                end else begin
                    fp16_mant = fp16_mant + 10'h001;
                end
            end
        end

        result = {sign, fp16_exp, fp16_mant};
        return result;
    endfunction


    // ----------------------------------------------------------------
    // FP16 to INT8 (signed): clamp to [-128, 127], round toward zero
    // ----------------------------------------------------------------
    function automatic logic [7:0] fp16_to_int8(input logic [15:0] val);
        logic        sign;
        logic [4:0]  exp_val;
        logic [9:0]  mant_val;
        logic        is_nan, is_inf;
        logic signed [8:0] int_result; // extra bit for range
        logic [7:0]  result;
        logic [4:0]  shift_amount;
        logic [10:0] full_mant; // {1, mantissa}

        sign     = val[15];
        exp_val  = val[14:10];
        mant_val = val[9:0];

        is_nan = (exp_val == 5'h1F) && (mant_val != 10'h0);
        is_inf = (exp_val == 5'h1F) && (mant_val == 10'h0);

        // NaN -> 0
        if (is_nan) begin
            result = 8'h00;
            return result;
        end

        // Infinity -> clamp to max/min
        if (is_inf) begin
            if (sign)
                result = 8'h80; // -128
            else
                result = 8'h7F; // 127
            return result;
        end

        // Zero or denormal -> 0
        if (exp_val == 5'h00) begin
            result = 8'h00;
            return result;
        end

        // Unbiased exponent
        // exp_val - 15 gives unbiased exponent
        // If unbiased exp < 0 (exp_val < 15), |value| < 1.0, truncate to 0
        if (exp_val < 5'd15) begin
            result = 8'h00;
            return result;
        end

        // Unbiased exponent >= 0
        // Value = 1.mantissa * 2^(exp_val - 15)
        full_mant = {1'b1, mant_val}; // 11 bits: 1.mant

        // The integer part is obtained by shifting full_mant right by (10 - (exp_val - 15))
        // or equivalently full_mant >> (25 - exp_val) when exp_val <= 25
        // But if exp_val - 15 >= 10, the entire mantissa contributes to integer
        // If exp_val - 15 >= 8, result magnitude >= 256 which overflows int8

        // Check for overflow: if unbiased exp >= 8, magnitude >= 256 (except 128 exactly for negative)
        if (exp_val >= 5'd23) begin
            // Magnitude >= 256, definitely clamp
            if (sign)
                result = 8'h80; // -128
            else
                result = 8'h7F; // 127
            return result;
        end

        // exp_val in [15, 22]
        // shift = exp_val - 15 gives how many bits of mantissa are integer part
        // integer value = full_mant >> (10 - shift) when shift <= 10
        shift_amount = exp_val - 5'd15; // range [0, 7]

        // Shift right to get integer (truncate toward zero)
        int_result = 9'sd0;
        case (shift_amount)
            5'd0: int_result = {8'h00, 1'b1};             // 1.xxx -> 1
            5'd1: int_result = {7'h00, full_mant[10:9]};   // integer part is top 2 bits
            5'd2: int_result = {6'h00, full_mant[10:8]};
            5'd3: int_result = {5'h00, full_mant[10:7]};
            5'd4: int_result = {4'h0,  full_mant[10:6]};
            5'd5: int_result = {3'h0,  full_mant[10:5]};
            5'd6: int_result = {2'h0,  full_mant[10:4]};
            5'd7: int_result = {1'b0,  full_mant[10:3]};
            default: int_result = 9'sd0;
        endcase

        // Clamp positive to 127
        if (!sign && int_result > 9'sd127) begin
            result = 8'h7F;
            return result;
        end

        // Clamp negative: allow up to 128
        if (sign && int_result > 9'sd128) begin
            result = 8'h80;
            return result;
        end

        // Apply sign
        if (sign) begin
            result = (~int_result[7:0]) + 8'h01; // negate
        end else begin
            result = int_result[7:0];
        end

        return result;
    endfunction

endpackage

`default_nettype wire
