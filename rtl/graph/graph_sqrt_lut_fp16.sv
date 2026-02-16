// =============================================================================
// graph_sqrt_lut_fp16.sv - Square root function via 256-entry ROM lookup table (FP16)
// Input:  addr[7:0] = upper 8 bits of FP16 input value
// Output: data_out[15:0] = FP16 result
// Formula: LUT[i] = fp16(sqrt(fp16_input))
// Pipeline: 1-cycle registered output
// =============================================================================
`default_nettype none

module graph_sqrt_lut_fp16 (
    input  wire        clk,
    input  wire  [7:0] addr,       // upper 8 bits of FP16 input
    output logic [15:0] data_out    // FP16 result
);

    // 256-entry ROM - inferred as block RAM
    (* rom_style = "block" *)
    logic [15:0] rom [0:255];

    // ----------------------------------------------------------------
    // ROM initialization - sqrt(fp16_value) lookup
    //
    // Key values (addr -> fp16 input -> sqrt output):
    //   0x3C (+1.0)  -> sqrt(1)   = 1.0    -> 0x3C00
    //   0x40 (+2.0)  -> sqrt(2)   = 1.414  -> 0x3DA8
    //   0x44 (+4.0)  -> sqrt(4)   = 2.0    -> 0x4000
    //   0x00 (+0.0)  -> sqrt(0)   = 0      -> 0x0000
    //   0x7C (+Inf)  -> sqrt(Inf) = +Inf   -> 0x7C00
    //   0xBC (-1.0)  -> sqrt(-1)  = NaN    -> 0x7E00
    //   0x7E (NaN)   -> NaN                -> 0x7E00
    // ----------------------------------------------------------------
    initial begin
        // Entries 0..15 (fp16: +0.0 .. 0.000427246)
        rom[  0] = 16'h0000;
        rom[  1] = 16'h1C00;
        rom[  2] = 16'h1DA8;
        rom[  3] = 16'h1EEE;
        rom[  4] = 16'h2000;
        rom[  5] = 16'h2079;
        rom[  6] = 16'h20E6;
        rom[  7] = 16'h214B;
        rom[  8] = 16'h21A8;
        rom[  9] = 16'h2253;
        rom[ 10] = 16'h22EE;
        rom[ 11] = 16'h237C;
        rom[ 12] = 16'h2400;
        rom[ 13] = 16'h2479;
        rom[ 14] = 16'h24E6;
        rom[ 15] = 16'h254B;

        // Entries 16..31 (fp16: 0.000488281 .. 0.00683594)
        rom[ 16] = 16'h25A8;
        rom[ 17] = 16'h2653;
        rom[ 18] = 16'h26EE;
        rom[ 19] = 16'h277C;
        rom[ 20] = 16'h2800;
        rom[ 21] = 16'h2879;
        rom[ 22] = 16'h28E6;
        rom[ 23] = 16'h294B;
        rom[ 24] = 16'h29A8;
        rom[ 25] = 16'h2A53;
        rom[ 26] = 16'h2AEE;
        rom[ 27] = 16'h2B7C;
        rom[ 28] = 16'h2C00;
        rom[ 29] = 16'h2C79;
        rom[ 30] = 16'h2CE6;
        rom[ 31] = 16'h2D4B;

        // Entries 32..47 (fp16: 0.0078125 .. 0.109375)
        rom[ 32] = 16'h2DA8;
        rom[ 33] = 16'h2E53;
        rom[ 34] = 16'h2EEE;
        rom[ 35] = 16'h2F7C;
        rom[ 36] = 16'h3000;
        rom[ 37] = 16'h3079;
        rom[ 38] = 16'h30E6;
        rom[ 39] = 16'h314B;
        rom[ 40] = 16'h31A8;
        rom[ 41] = 16'h3253;
        rom[ 42] = 16'h32EE;
        rom[ 43] = 16'h337C;
        rom[ 44] = 16'h3400;
        rom[ 45] = 16'h3479;
        rom[ 46] = 16'h34E6;
        rom[ 47] = 16'h354B;

        // Entries 48..63 (fp16: 0.125 .. 1.75)
        rom[ 48] = 16'h35A8;
        rom[ 49] = 16'h3653;
        rom[ 50] = 16'h36EE;
        rom[ 51] = 16'h377C;
        rom[ 52] = 16'h3800;
        rom[ 53] = 16'h3879;
        rom[ 54] = 16'h38E6;
        rom[ 55] = 16'h394B;
        rom[ 56] = 16'h39A8;
        rom[ 57] = 16'h3A53;
        rom[ 58] = 16'h3AEE;
        rom[ 59] = 16'h3B7C;
        rom[ 60] = 16'h3C00;
        rom[ 61] = 16'h3C79;
        rom[ 62] = 16'h3CE6;
        rom[ 63] = 16'h3D4B;

        // Entries 64..79 (fp16: 2 .. 28)
        rom[ 64] = 16'h3DA8;
        rom[ 65] = 16'h3E53;
        rom[ 66] = 16'h3EEE;
        rom[ 67] = 16'h3F7C;
        rom[ 68] = 16'h4000;
        rom[ 69] = 16'h4079;
        rom[ 70] = 16'h40E6;
        rom[ 71] = 16'h414B;
        rom[ 72] = 16'h41A8;
        rom[ 73] = 16'h4253;
        rom[ 74] = 16'h42EE;
        rom[ 75] = 16'h437C;
        rom[ 76] = 16'h4400;
        rom[ 77] = 16'h4479;
        rom[ 78] = 16'h44E6;
        rom[ 79] = 16'h454B;

        // Entries 80..95 (fp16: 32 .. 448)
        rom[ 80] = 16'h45A8;
        rom[ 81] = 16'h4653;
        rom[ 82] = 16'h46EE;
        rom[ 83] = 16'h477C;
        rom[ 84] = 16'h4800;
        rom[ 85] = 16'h4879;
        rom[ 86] = 16'h48E6;
        rom[ 87] = 16'h494B;
        rom[ 88] = 16'h49A8;
        rom[ 89] = 16'h4A53;
        rom[ 90] = 16'h4AEE;
        rom[ 91] = 16'h4B7C;
        rom[ 92] = 16'h4C00;
        rom[ 93] = 16'h4C79;
        rom[ 94] = 16'h4CE6;
        rom[ 95] = 16'h4D4B;

        // Entries 96..111 (fp16: 512 .. 7168)
        rom[ 96] = 16'h4DA8;
        rom[ 97] = 16'h4E53;
        rom[ 98] = 16'h4EEE;
        rom[ 99] = 16'h4F7C;
        rom[100] = 16'h5000;
        rom[101] = 16'h5079;
        rom[102] = 16'h50E6;
        rom[103] = 16'h514B;
        rom[104] = 16'h51A8;
        rom[105] = 16'h5253;
        rom[106] = 16'h52EE;
        rom[107] = 16'h537C;
        rom[108] = 16'h5400;
        rom[109] = 16'h5479;
        rom[110] = 16'h54E6;
        rom[111] = 16'h554B;

        // Entries 112..127 (fp16: 8192 .. NaN)
        rom[112] = 16'h55A8;
        rom[113] = 16'h5653;
        rom[114] = 16'h56EE;
        rom[115] = 16'h577C;
        rom[116] = 16'h5800;
        rom[117] = 16'h5879;
        rom[118] = 16'h58E6;
        rom[119] = 16'h594B;
        rom[120] = 16'h59A8;
        rom[121] = 16'h5A53;
        rom[122] = 16'h5AEE;
        rom[123] = 16'h5B7C;
        rom[124] = 16'h7C00;
        rom[125] = 16'h7E00;
        rom[126] = 16'h7E00;
        rom[127] = 16'h7E00;

        // Entries 128..143 (fp16: -0.0 .. -0.000427246)
        rom[128] = 16'h0000;
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
