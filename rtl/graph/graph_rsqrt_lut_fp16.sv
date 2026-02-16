// =============================================================================
// graph_rsqrt_lut_fp16.sv - Reciprocal square root function via 256-entry ROM lookup table (FP16)
// Input:  addr[7:0] = upper 8 bits of FP16 input value
// Output: data_out[15:0] = FP16 result
// Formula: LUT[i] = fp16(1/sqrt(fp16_input))
// Pipeline: 1-cycle registered output
// =============================================================================
`default_nettype none

module graph_rsqrt_lut_fp16 (
    input  wire        clk,
    input  wire  [7:0] addr,       // upper 8 bits of FP16 input
    output logic [15:0] data_out    // FP16 result
);

    // 256-entry ROM - inferred as block RAM
    (* rom_style = "block" *)
    logic [15:0] rom [0:255];

    // ----------------------------------------------------------------
    // ROM initialization - rsqrt(fp16_value) lookup
    //
    // Key values (addr -> fp16 input -> rsqrt output):
    //   0x3C (+1.0)  -> rsqrt(1)  = 1.0    -> 0x3C00
    //   0x40 (+2.0)  -> rsqrt(2)  = 0.707  -> 0x39A8
    //   0x44 (+4.0)  -> rsqrt(4)  = 0.5    -> 0x3800
    //   0x00 (+0.0)  -> rsqrt(0)  = +Inf   -> 0x7C00
    //   0x7C (+Inf)  -> rsqrt(Inf)= 0      -> 0x0000
    //   0xBC (-1.0)  -> rsqrt(-1) = NaN    -> 0x7E00
    //   0x7E (NaN)   -> NaN                -> 0x7E00
    // ----------------------------------------------------------------
    initial begin
        // Entries 0..15 (fp16: +0.0 .. 0.000427246)
        rom[  0] = 16'h7C00;
        rom[  1] = 16'h5C00;
        rom[  2] = 16'h59A8;
        rom[  3] = 16'h589E;
        rom[  4] = 16'h5800;
        rom[  5] = 16'h5728;
        rom[  6] = 16'h5688;
        rom[  7] = 16'h560C;
        rom[  8] = 16'h55A8;
        rom[  9] = 16'h550F;
        rom[ 10] = 16'h549E;
        rom[ 11] = 16'h5447;
        rom[ 12] = 16'h5400;
        rom[ 13] = 16'h5328;
        rom[ 14] = 16'h5288;
        rom[ 15] = 16'h520C;

        // Entries 16..31 (fp16: 0.000488281 .. 0.00683594)
        rom[ 16] = 16'h51A8;
        rom[ 17] = 16'h510F;
        rom[ 18] = 16'h509E;
        rom[ 19] = 16'h5047;
        rom[ 20] = 16'h5000;
        rom[ 21] = 16'h4F28;
        rom[ 22] = 16'h4E88;
        rom[ 23] = 16'h4E0C;
        rom[ 24] = 16'h4DA8;
        rom[ 25] = 16'h4D0F;
        rom[ 26] = 16'h4C9E;
        rom[ 27] = 16'h4C47;
        rom[ 28] = 16'h4C00;
        rom[ 29] = 16'h4B28;
        rom[ 30] = 16'h4A88;
        rom[ 31] = 16'h4A0C;

        // Entries 32..47 (fp16: 0.0078125 .. 0.109375)
        rom[ 32] = 16'h49A8;
        rom[ 33] = 16'h490F;
        rom[ 34] = 16'h489E;
        rom[ 35] = 16'h4847;
        rom[ 36] = 16'h4800;
        rom[ 37] = 16'h4728;
        rom[ 38] = 16'h4688;
        rom[ 39] = 16'h460C;
        rom[ 40] = 16'h45A8;
        rom[ 41] = 16'h450F;
        rom[ 42] = 16'h449E;
        rom[ 43] = 16'h4447;
        rom[ 44] = 16'h4400;
        rom[ 45] = 16'h4328;
        rom[ 46] = 16'h4288;
        rom[ 47] = 16'h420C;

        // Entries 48..63 (fp16: 0.125 .. 1.75)
        rom[ 48] = 16'h41A8;
        rom[ 49] = 16'h410F;
        rom[ 50] = 16'h409E;
        rom[ 51] = 16'h4047;
        rom[ 52] = 16'h4000;
        rom[ 53] = 16'h3F28;
        rom[ 54] = 16'h3E88;
        rom[ 55] = 16'h3E0C;
        rom[ 56] = 16'h3DA8;
        rom[ 57] = 16'h3D0F;
        rom[ 58] = 16'h3C9E;
        rom[ 59] = 16'h3C47;
        rom[ 60] = 16'h3C00;
        rom[ 61] = 16'h3B28;
        rom[ 62] = 16'h3A88;
        rom[ 63] = 16'h3A0C;

        // Entries 64..79 (fp16: 2 .. 28)
        rom[ 64] = 16'h39A8;
        rom[ 65] = 16'h390F;
        rom[ 66] = 16'h389E;
        rom[ 67] = 16'h3847;
        rom[ 68] = 16'h3800;
        rom[ 69] = 16'h3728;
        rom[ 70] = 16'h3688;
        rom[ 71] = 16'h360C;
        rom[ 72] = 16'h35A8;
        rom[ 73] = 16'h350F;
        rom[ 74] = 16'h349E;
        rom[ 75] = 16'h3447;
        rom[ 76] = 16'h3400;
        rom[ 77] = 16'h3328;
        rom[ 78] = 16'h3288;
        rom[ 79] = 16'h320C;

        // Entries 80..95 (fp16: 32 .. 448)
        rom[ 80] = 16'h31A8;
        rom[ 81] = 16'h310F;
        rom[ 82] = 16'h309E;
        rom[ 83] = 16'h3047;
        rom[ 84] = 16'h3000;
        rom[ 85] = 16'h2F28;
        rom[ 86] = 16'h2E88;
        rom[ 87] = 16'h2E0C;
        rom[ 88] = 16'h2DA8;
        rom[ 89] = 16'h2D0F;
        rom[ 90] = 16'h2C9E;
        rom[ 91] = 16'h2C47;
        rom[ 92] = 16'h2C00;
        rom[ 93] = 16'h2B28;
        rom[ 94] = 16'h2A88;
        rom[ 95] = 16'h2A0C;

        // Entries 96..111 (fp16: 512 .. 7168)
        rom[ 96] = 16'h29A8;
        rom[ 97] = 16'h290F;
        rom[ 98] = 16'h289E;
        rom[ 99] = 16'h2847;
        rom[100] = 16'h2800;
        rom[101] = 16'h2728;
        rom[102] = 16'h2688;
        rom[103] = 16'h260C;
        rom[104] = 16'h25A8;
        rom[105] = 16'h250F;
        rom[106] = 16'h249E;
        rom[107] = 16'h2447;
        rom[108] = 16'h2400;
        rom[109] = 16'h2328;
        rom[110] = 16'h2288;
        rom[111] = 16'h220C;

        // Entries 112..127 (fp16: 8192 .. NaN)
        rom[112] = 16'h21A8;
        rom[113] = 16'h210F;
        rom[114] = 16'h209E;
        rom[115] = 16'h2047;
        rom[116] = 16'h2000;
        rom[117] = 16'h1F28;
        rom[118] = 16'h1E88;
        rom[119] = 16'h1E0C;
        rom[120] = 16'h1DA8;
        rom[121] = 16'h1D0F;
        rom[122] = 16'h1C9E;
        rom[123] = 16'h1C47;
        rom[124] = 16'h0000;
        rom[125] = 16'h7E00;
        rom[126] = 16'h7E00;
        rom[127] = 16'h7E00;

        // Entries 128..143 (fp16: -0.0 .. -0.000427246)
        rom[128] = 16'h7C00;
        rom[129] = 16'h7E00;
        rom[130] = 16'h7E00;
        rom[131] = 16'h7E00;
        rom[132] = 16'h7E00;
        rom[133] = 16'h7E00;
        rom[134] = 16'h7E00;
        rom[135] = 16'h7E00;
        rom[136] = 16'h7E00;
        rom[137] = 16'h7E00;
        rom[138] = 16'h7E00;
        rom[139] = 16'h7E00;
        rom[140] = 16'h7E00;
        rom[141] = 16'h7E00;
        rom[142] = 16'h7E00;
        rom[143] = 16'h7E00;

        // Entries 144..159 (fp16: -0.000488281 .. -0.00683594)
        rom[144] = 16'h7E00;
        rom[145] = 16'h7E00;
        rom[146] = 16'h7E00;
        rom[147] = 16'h7E00;
        rom[148] = 16'h7E00;
        rom[149] = 16'h7E00;
        rom[150] = 16'h7E00;
        rom[151] = 16'h7E00;
        rom[152] = 16'h7E00;
        rom[153] = 16'h7E00;
        rom[154] = 16'h7E00;
        rom[155] = 16'h7E00;
        rom[156] = 16'h7E00;
        rom[157] = 16'h7E00;
        rom[158] = 16'h7E00;
        rom[159] = 16'h7E00;

        // Entries 160..175 (fp16: -0.0078125 .. -0.109375)
        rom[160] = 16'h7E00;
        rom[161] = 16'h7E00;
        rom[162] = 16'h7E00;
        rom[163] = 16'h7E00;
        rom[164] = 16'h7E00;
        rom[165] = 16'h7E00;
        rom[166] = 16'h7E00;
        rom[167] = 16'h7E00;
        rom[168] = 16'h7E00;
        rom[169] = 16'h7E00;
        rom[170] = 16'h7E00;
        rom[171] = 16'h7E00;
        rom[172] = 16'h7E00;
        rom[173] = 16'h7E00;
        rom[174] = 16'h7E00;
        rom[175] = 16'h7E00;

        // Entries 176..191 (fp16: -0.125 .. -1.75)
        rom[176] = 16'h7E00;
        rom[177] = 16'h7E00;
        rom[178] = 16'h7E00;
        rom[179] = 16'h7E00;
        rom[180] = 16'h7E00;
        rom[181] = 16'h7E00;
        rom[182] = 16'h7E00;
        rom[183] = 16'h7E00;
        rom[184] = 16'h7E00;
        rom[185] = 16'h7E00;
        rom[186] = 16'h7E00;
        rom[187] = 16'h7E00;
        rom[188] = 16'h7E00;
        rom[189] = 16'h7E00;
        rom[190] = 16'h7E00;
        rom[191] = 16'h7E00;

        // Entries 192..207 (fp16: -2 .. -28)
        rom[192] = 16'h7E00;
        rom[193] = 16'h7E00;
        rom[194] = 16'h7E00;
        rom[195] = 16'h7E00;
        rom[196] = 16'h7E00;
        rom[197] = 16'h7E00;
        rom[198] = 16'h7E00;
        rom[199] = 16'h7E00;
        rom[200] = 16'h7E00;
        rom[201] = 16'h7E00;
        rom[202] = 16'h7E00;
        rom[203] = 16'h7E00;
        rom[204] = 16'h7E00;
        rom[205] = 16'h7E00;
        rom[206] = 16'h7E00;
        rom[207] = 16'h7E00;

        // Entries 208..223 (fp16: -32 .. -448)
        rom[208] = 16'h7E00;
        rom[209] = 16'h7E00;
        rom[210] = 16'h7E00;
        rom[211] = 16'h7E00;
        rom[212] = 16'h7E00;
        rom[213] = 16'h7E00;
        rom[214] = 16'h7E00;
        rom[215] = 16'h7E00;
        rom[216] = 16'h7E00;
        rom[217] = 16'h7E00;
        rom[218] = 16'h7E00;
        rom[219] = 16'h7E00;
        rom[220] = 16'h7E00;
        rom[221] = 16'h7E00;
        rom[222] = 16'h7E00;
        rom[223] = 16'h7E00;

        // Entries 224..239 (fp16: -512 .. -7168)
        rom[224] = 16'h7E00;
        rom[225] = 16'h7E00;
        rom[226] = 16'h7E00;
        rom[227] = 16'h7E00;
        rom[228] = 16'h7E00;
        rom[229] = 16'h7E00;
        rom[230] = 16'h7E00;
        rom[231] = 16'h7E00;
        rom[232] = 16'h7E00;
        rom[233] = 16'h7E00;
        rom[234] = 16'h7E00;
        rom[235] = 16'h7E00;
        rom[236] = 16'h7E00;
        rom[237] = 16'h7E00;
        rom[238] = 16'h7E00;
        rom[239] = 16'h7E00;

        // Entries 240..255 (fp16: -8192 .. NaN)
        rom[240] = 16'h7E00;
        rom[241] = 16'h7E00;
        rom[242] = 16'h7E00;
        rom[243] = 16'h7E00;
        rom[244] = 16'h7E00;
        rom[245] = 16'h7E00;
        rom[246] = 16'h7E00;
        rom[247] = 16'h7E00;
        rom[248] = 16'h7E00;
        rom[249] = 16'h7E00;
        rom[250] = 16'h7E00;
        rom[251] = 16'h7E00;
        rom[252] = 16'h7E00;
        rom[253] = 16'h7E00;
        rom[254] = 16'h7E00;
        rom[255] = 16'h7E00;

    end

    // ----------------------------------------------------------------
    // Registered read (1-cycle latency)
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        data_out <= rom[addr];
    end

endmodule

`default_nettype wire
