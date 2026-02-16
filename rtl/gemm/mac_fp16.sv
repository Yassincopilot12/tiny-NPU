// =============================================================================
// FP16 Multiply-Accumulate Unit (2-stage pipeline)
//   Stage 1: FP16 x FP16 -> FP32 product
//   Stage 2: FP32 accumulate (product + acc_reg)
// Matches mac_int8 latency (2 pipeline stages).
// =============================================================================
`default_nettype none

module mac_fp16 #(
    parameter int DATA_W = 16,  // FP16 = 16 bits
    parameter int ACC_W  = 32   // FP32 accumulator
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear_acc,
    input  wire                    en,
    input  wire  [DATA_W-1:0]     a_in,    // FP16 input
    input  wire  [DATA_W-1:0]     b_in,    // FP16 input
    output logic [ACC_W-1:0]      acc_out  // FP32 accumulator
);

    // =========================================================================
    // FP16 field extraction helpers
    // =========================================================================
    // FP16: sign[15], exponent[14:10] (5-bit, bias=15), mantissa[9:0]

    logic        a_sign, b_sign;
    logic [4:0]  a_exp,  b_exp;
    logic [9:0]  a_mant, b_mant;
    logic        a_is_zero, b_is_zero;
    logic        a_is_inf,  b_is_inf;
    logic        a_is_nan,  b_is_nan;

    always_comb begin
        a_sign    = a_in[15];
        a_exp     = a_in[14:10];
        a_mant    = a_in[9:0];
        b_sign    = b_in[15];
        b_exp     = b_in[14:10];
        b_mant    = b_in[9:0];

        a_is_zero = (a_exp == 5'h00) && (a_mant == 10'h000);
        b_is_zero = (b_exp == 5'h00) && (b_mant == 10'h000);
        a_is_inf  = (a_exp == 5'h1F) && (a_mant == 10'h000);
        b_is_inf  = (b_exp == 5'h1F) && (b_mant == 10'h000);
        a_is_nan  = (a_exp == 5'h1F) && (a_mant != 10'h000);
        b_is_nan  = (b_exp == 5'h1F) && (b_mant != 10'h000);
    end

    // =========================================================================
    // Stage 1: FP16 multiply -> FP32 product (registered)
    // =========================================================================
    logic [ACC_W-1:0] product_fp32;
    logic             en_d1;
    logic             clear_d1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product_fp32 <= 32'h0000_0000;
            en_d1        <= 1'b0;
            clear_d1     <= 1'b0;
        end else begin
            en_d1    <= en;
            clear_d1 <= clear_acc;
            if (en) begin
                product_fp32 <= fp16_mul(a_sign, a_exp, a_mant,
                                         b_sign, b_exp, b_mant,
                                         a_is_zero, b_is_zero,
                                         a_is_inf,  b_is_inf,
                                         a_is_nan,  b_is_nan);
            end else begin
                product_fp32 <= 32'h0000_0000;
            end
        end
    end

    // =========================================================================
    // Stage 2: FP32 accumulate (product_fp32 + acc_reg)
    // =========================================================================
    logic [ACC_W-1:0] acc_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= 32'h0000_0000;
        end else if (clear_d1) begin
            acc_reg <= 32'h0000_0000;
        end else if (en_d1) begin
            acc_reg <= fp32_add(product_fp32, acc_reg);
        end
    end

    assign acc_out = acc_reg;

    // =========================================================================
    // Function: FP16 multiply -> FP32 result
    // =========================================================================
    function automatic logic [31:0] fp16_mul(
        input logic        sa,
        input logic [4:0]  ea,
        input logic [9:0]  ma,
        input logic        sb,
        input logic [4:0]  eb,
        input logic [9:0]  mb,
        input logic        a_zero,
        input logic        b_zero,
        input logic        a_inf,
        input logic        b_inf,
        input logic        a_nan,
        input logic        b_nan
    );
        logic        res_sign;
        logic [21:0] mant_prod;   // 11-bit x 11-bit = 22-bit product
        logic [10:0] full_ma, full_mb;
        logic signed [7:0]  raw_exp;  // intermediate exponent (signed)
        logic [7:0]  fp32_exp;
        logic [22:0] fp32_mant;
        logic        prod_overflow;

        // Result sign
        res_sign = sa ^ sb;

        // ------------------------------------------------------------------
        // Special cases
        // ------------------------------------------------------------------
        if (a_nan || b_nan) begin
            // NaN propagation -> quiet NaN
            fp16_mul = {1'b0, 8'hFF, 23'h40_0000};  // canonical qNaN
            return fp16_mul;
        end
        if ((a_inf && b_zero) || (b_inf && a_zero)) begin
            // inf * 0 = NaN
            fp16_mul = {1'b0, 8'hFF, 23'h40_0000};
            return fp16_mul;
        end
        if (a_inf || b_inf) begin
            // inf * finite = inf
            fp16_mul = {res_sign, 8'hFF, 23'h000000};
            return fp16_mul;
        end
        if (a_zero || b_zero) begin
            // +/- zero
            fp16_mul = {res_sign, 31'h0000_0000};
            return fp16_mul;
        end

        // ------------------------------------------------------------------
        // Normal / subnormal multiply
        // ------------------------------------------------------------------
        // Implicit leading 1 for normals, 0 for subnormals
        full_ma = (ea != 5'h00) ? {1'b1, ma} : {1'b0, ma};
        full_mb = (eb != 5'h00) ? {1'b1, mb} : {1'b0, mb};

        mant_prod = full_ma * full_mb;  // 22-bit result

        // Raw exponent in FP16 domain:
        //   normal:    ea + eb - 15 (FP16 bias)
        //   subnormal: treat effective exponent as 1 (not 0)
        raw_exp = signed'({3'b0, (ea != 5'h00) ? ea : 5'h01})
                + signed'({3'b0, (eb != 5'h00) ? eb : 5'h01})
                - signed'(8'd15);

        // Normalise the 22-bit product.
        // If mant_prod[21] is set, the product is in [2, 4) -> shift right,
        // increment exponent.  Otherwise product is in [1, 2) -> no shift.
        if (mant_prod[21]) begin
            prod_overflow = 1'b1;
            raw_exp       = raw_exp + signed'(8'd1);
        end else begin
            prod_overflow = 1'b0;
        end

        // Convert to FP32 exponent: add bias difference (127 - 15 = 112)
        raw_exp = raw_exp + signed'(8'd112);

        // Clamp / detect underflow / overflow
        if (raw_exp <= 0) begin
            // Flush to zero (simplified: no FP32 subnormal generation)
            fp16_mul = {res_sign, 31'h0000_0000};
            return fp16_mul;
        end else if (raw_exp >= signed'(8'd255)) begin
            // Overflow -> infinity
            fp16_mul = {res_sign, 8'hFF, 23'h000000};
            return fp16_mul;
        end

        fp32_exp = raw_exp[7:0];

        // Map mantissa bits into FP32 23-bit mantissa (drop implicit 1).
        // mant_prod is 22 bits.  After normalisation:
        //   overflow case: significant bits are mant_prod[20:0], need 23 bits
        //                  -> {mant_prod[20:0], 2'b00}
        //   normal  case:  significant bits are mant_prod[19:0], need 23 bits
        //                  -> {mant_prod[19:0], 3'b000}
        if (prod_overflow)
            fp32_mant = {mant_prod[20:0], 2'b00};
        else
            fp32_mant = {mant_prod[19:0], 3'b000};

        fp16_mul = {res_sign, fp32_exp, fp32_mant};
    endfunction

    // =========================================================================
    // Function: FP32 addition (for accumulation)
    //   Uses 27-bit working mantissa (24 significant + 3 guard/round/sticky).
    // =========================================================================
    function automatic logic [31:0] fp32_add(
        input logic [31:0] x,
        input logic [31:0] y
    );
        logic        sx, sy, sr;
        logic [7:0]  ex, ey, er;
        logic [22:0] mx, my;
        logic [26:0] wx, wy;       // 27-bit working mantissa (1.23 + 3 GRS)
        logic [27:0] wsum;          // 28-bit to catch carry
        logic        x_zero, y_zero;
        logic        x_inf,  y_inf;
        logic        x_nan,  y_nan;
        logic [7:0]  exp_diff;
        logic        swap;
        // Temporaries for the larger / smaller operand
        logic        sl, ss;
        logic [7:0]  el, es;
        logic [26:0] wl, ws;
        logic [4:0]  lzc;          // leading-zero count
        int          i;
        logic        sticky;

        // Unpack
        sx = x[31]; ex = x[30:23]; mx = x[22:0];
        sy = y[31]; ey = y[30:23]; my = y[22:0];

        x_zero = (ex == 8'h00) && (mx == 23'h0);
        y_zero = (ey == 8'h00) && (my == 23'h0);
        x_inf  = (ex == 8'hFF) && (mx == 23'h0);
        y_inf  = (ey == 8'hFF) && (my == 23'h0);
        x_nan  = (ex == 8'hFF) && (mx != 23'h0);
        y_nan  = (ey == 8'hFF) && (my != 23'h0);

        // ------------------------------------------------------------------
        // Special cases
        // ------------------------------------------------------------------
        if (x_nan || y_nan) begin
            fp32_add = {1'b0, 8'hFF, 23'h40_0000};  // qNaN
            return fp32_add;
        end
        if (x_inf && y_inf) begin
            if (sx != sy) begin
                fp32_add = {1'b0, 8'hFF, 23'h40_0000};  // inf - inf = NaN
            end else begin
                fp32_add = x;  // same-sign inf
            end
            return fp32_add;
        end
        if (x_inf) begin fp32_add = x; return fp32_add; end
        if (y_inf) begin fp32_add = y; return fp32_add; end
        if (x_zero && y_zero) begin
            // -0 + -0 = -0, otherwise +0
            fp32_add = {sx & sy, 31'h0};
            return fp32_add;
        end
        if (x_zero) begin fp32_add = y; return fp32_add; end
        if (y_zero) begin fp32_add = x; return fp32_add; end

        // ------------------------------------------------------------------
        // Build working mantissas: {implicit_1, mantissa[22:0], 3'b000}
        // ------------------------------------------------------------------
        wx = {1'b1, mx, 3'b000};
        wy = {1'b1, my, 3'b000};

        // ------------------------------------------------------------------
        // Sort so that el >= es  (larger exponent first)
        // ------------------------------------------------------------------
        swap = (ey > ex) || ((ey == ex) && (my > mx));
        if (swap) begin
            sl = sy; el = ey; wl = wy;
            ss = sx; es = ex; ws = wx;
        end else begin
            sl = sx; el = ex; wl = wx;
            ss = sy; es = ey; ws = wy;
        end

        // ------------------------------------------------------------------
        // Align smaller mantissa
        // ------------------------------------------------------------------
        exp_diff = el - es;

        // Compute sticky bit from shifted-out bits
        sticky = 1'b0;
        if (exp_diff > 8'd27) begin
            sticky = |ws;
            ws     = 27'b0;
        end else begin
            for (i = 0; i < 27; i++) begin
                if (i[7:0] < exp_diff)
                    sticky = sticky | ws[i];
            end
            ws = ws >> exp_diff;
        end
        ws[0] = ws[0] | sticky;

        // ------------------------------------------------------------------
        // Add or subtract mantissas
        // ------------------------------------------------------------------
        if (sl == ss) begin
            // Same sign: add magnitudes
            wsum = {1'b0, wl} + {1'b0, ws};
            sr   = sl;
        end else begin
            // Different sign: subtract smaller from larger
            wsum = {1'b0, wl} - {1'b0, ws};
            sr   = sl;
            if (wsum == 28'b0) begin
                fp32_add = 32'h0000_0000;  // exact zero -> +0
                return fp32_add;
            end
        end

        // ------------------------------------------------------------------
        // Normalise
        // ------------------------------------------------------------------
        er = el;

        // Handle carry out (same-sign addition overflow)
        if (wsum[27]) begin
            sticky  = wsum[0];
            wsum    = wsum >> 1;
            wsum[0] = wsum[0] | sticky;
            er      = er + 8'd1;
            if (er == 8'hFF) begin
                // Overflow -> infinity
                fp32_add = {sr, 8'hFF, 23'h000000};
                return fp32_add;
            end
        end

        // Count leading zeros and shift left to normalise
        lzc = 5'd0;
        for (i = 26; i >= 0; i--) begin
            if (!wsum[i] && (lzc == (5'd26 - i[4:0])))
                lzc = lzc + 5'd1;
        end

        if (lzc > 0) begin
            if ({3'b0, lzc} >= er) begin
                // Underflow -> flush to zero
                fp32_add = {sr, 31'h0};
                return fp32_add;
            end
            wsum = wsum << lzc;
            er   = er - {3'b0, lzc};
        end

        // ------------------------------------------------------------------
        // Round (round-to-nearest-even using GRS bits)
        // ------------------------------------------------------------------
        // wsum[26:3] = 1.mantissa (24 bits), wsum[2:0] = GRS
        begin
            logic guard, round_bit, sticky_bit;
            logic round_up;
            guard     = wsum[2];
            round_bit = wsum[1];
            sticky_bit= wsum[0];
            round_up  = guard & (round_bit | sticky_bit | wsum[3]);
            if (round_up) begin
                wsum[27:3] = wsum[27:3] + 25'd1;
                // Check for carry into bit 27 after rounding
                if (wsum[27]) begin
                    wsum = wsum >> 1;
                    er   = er + 8'd1;
                    if (er == 8'hFF) begin
                        fp32_add = {sr, 8'hFF, 23'h000000};
                        return fp32_add;
                    end
                end
            end
        end

        fp32_add = {sr, er, wsum[25:3]};
    endfunction

endmodule

`default_nettype wire
