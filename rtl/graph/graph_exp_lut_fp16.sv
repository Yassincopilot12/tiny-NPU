// =============================================================================
// graph_exp_lut_fp16.sv - EXP activation function via 256-entry ROM lookup table (FP16)
// Input:  addr[7:0] = upper 8 bits of FP16 input value
// Output: data_out[15:0] = FP16 result
// Formula: LUT[i] = fp16(exp(fp16_input))
// Pipeline: 1-cycle registered output
// =============================================================================
`default_nettype none

module graph_exp_lut_fp16 (
    input  wire        clk,
    input  wire  [7:0] addr,       // upper 8 bits of FP16 input
    output logic [15:0] data_out    // FP16 result
);

    // 256-entry ROM - inferred as block RAM
    (* rom_style = "block" *)
    logic [15:0] rom [0:255];

    // ----------------------------------------------------------------
    // ROM initialization - exp(fp16_value) lookup
    //
    // Key values (addr -> fp16 input -> exp output):
    //   0x00 (+0.0)  -> exp(0)    = 1.0    -> 0x3C00
    //   0x80 (-0.0)  -> exp(0)    = 1.0    -> 0x3C00
    //   0x3C (+1.0)  -> exp(1)    = 2.718  -> 0x4170
    //   0xBC (-1.0)  -> exp(-1)   = 0.368  -> 0x35E3
    //   0x7C (+Inf)  -> exp(+Inf) = +Inf   -> 0x7C00
    //   0xFC (-Inf)  -> exp(-Inf) = 0      -> 0x0000
    //   0x7E (NaN)   -> NaN                -> 0x7E00
    // ----------------------------------------------------------------
    initial begin
        // Entries 0..15 (fp16: +0.0 .. 0.000427246)
        rom[  0] = 16'h3C00;
        rom[  1] = 16'h3C00;
        rom[  2] = 16'h3C00;
        rom[  3] = 16'h3C00;
        rom[  4] = 16'h3C00;
        rom[  5] = 16'h3C00;
        rom[  6] = 16'h3C00;
        rom[  7] = 16'h3C00;
        rom[  8] = 16'h3C00;
        rom[  9] = 16'h3C00;
        rom[ 10] = 16'h3C00;
        rom[ 11] = 16'h3C00;
        rom[ 12] = 16'h3C00;
        rom[ 13] = 16'h3C00;
        rom[ 14] = 16'h3C00;
        rom[ 15] = 16'h3C00;

        // Entries 16..31 (fp16: 0.000488281 .. 0.00683594)
        rom[ 16] = 16'h3C01;
        rom[ 17] = 16'h3C01;
        rom[ 18] = 16'h3C01;
        rom[ 19] = 16'h3C01;
        rom[ 20] = 16'h3C01;
        rom[ 21] = 16'h3C01;
        rom[ 22] = 16'h3C02;
        rom[ 23] = 16'h3C02;
        rom[ 24] = 16'h3C02;
        rom[ 25] = 16'h3C03;
        rom[ 26] = 16'h3C03;
        rom[ 27] = 16'h3C04;
        rom[ 28] = 16'h3C04;
        rom[ 29] = 16'h3C05;
        rom[ 30] = 16'h3C06;
        rom[ 31] = 16'h3C07;

        // Entries 32..47 (fp16: 0.0078125 .. 0.109375)
        rom[ 32] = 16'h3C08;
        rom[ 33] = 16'h3C0A;
        rom[ 34] = 16'h3C0C;
        rom[ 35] = 16'h3C0E;
        rom[ 36] = 16'h3C10;
        rom[ 37] = 16'h3C14;
        rom[ 38] = 16'h3C18;
        rom[ 39] = 16'h3C1C;
        rom[ 40] = 16'h3C21;
        rom[ 41] = 16'h3C29;
        rom[ 42] = 16'h3C31;
        rom[ 43] = 16'h3C3A;
        rom[ 44] = 16'h3C42;
        rom[ 45] = 16'h3C53;
        rom[ 46] = 16'h3C65;
        rom[ 47] = 16'h3C76;

        // Entries 48..63 (fp16: 0.125 .. 1.75)
        rom[ 48] = 16'h3C88;
        rom[ 49] = 16'h3CAD;
        rom[ 50] = 16'h3CD3;
        rom[ 51] = 16'h3CFA;
        rom[ 52] = 16'h3D23;
        rom[ 53] = 16'h3D78;
        rom[ 54] = 16'h3DD2;
        rom[ 55] = 16'h3E32;
        rom[ 56] = 16'h3E98;
        rom[ 57] = 16'h3F79;
        rom[ 58] = 16'h403C;
        rom[ 59] = 16'h40CC;
        rom[ 60] = 16'h4170;
        rom[ 61] = 16'h42FB;
        rom[ 62] = 16'h447B;
        rom[ 63] = 16'h45C1;

        // Entries 64..79 (fp16: 2 .. 28)
        rom[ 64] = 16'h4764;
        rom[ 65] = 16'h4A17;
        rom[ 66] = 16'h4D05;
        rom[ 67] = 16'h5024;
        rom[ 68] = 16'h52D3;
        rom[ 69] = 16'h58A3;
        rom[ 70] = 16'h5E4E;
        rom[ 71] = 16'h6449;
        rom[ 72] = 16'h69D2;
        rom[ 73] = 16'h7561;
        rom[ 74] = 16'h7C00;
        rom[ 75] = 16'h7C00;
        rom[ 76] = 16'h7C00;
        rom[ 77] = 16'h7C00;
        rom[ 78] = 16'h7C00;
        rom[ 79] = 16'h7C00;

        // Entries 80..95 (fp16: 32 .. 448)
        rom[ 80] = 16'h7C00;
        rom[ 81] = 16'h7C00;
        rom[ 82] = 16'h7C00;
        rom[ 83] = 16'h7C00;
        rom[ 84] = 16'h7C00;
        rom[ 85] = 16'h7C00;
        rom[ 86] = 16'h7C00;
        rom[ 87] = 16'h7C00;
        rom[ 88] = 16'h7C00;
        rom[ 89] = 16'h7C00;
        rom[ 90] = 16'h7C00;
        rom[ 91] = 16'h7C00;
        rom[ 92] = 16'h7C00;
        rom[ 93] = 16'h7C00;
        rom[ 94] = 16'h7C00;
        rom[ 95] = 16'h7C00;

        // Entries 96..111 (fp16: 512 .. 7168)
        rom[ 96] = 16'h7C00;
        rom[ 97] = 16'h7C00;
        rom[ 98] = 16'h7C00;
        rom[ 99] = 16'h7C00;
        rom[100] = 16'h7C00;
        rom[101] = 16'h7C00;
        rom[102] = 16'h7C00;
        rom[103] = 16'h7C00;
        rom[104] = 16'h7C00;
        rom[105] = 16'h7C00;
        rom[106] = 16'h7C00;
        rom[107] = 16'h7C00;
        rom[108] = 16'h7C00;
        rom[109] = 16'h7C00;
        rom[110] = 16'h7C00;
        rom[111] = 16'h7C00;

        // Entries 112..127 (fp16: 8192 .. NaN)
        rom[112] = 16'h7C00;
        rom[113] = 16'h7C00;
        rom[114] = 16'h7C00;
        rom[115] = 16'h7C00;
        rom[116] = 16'h7C00;
        rom[117] = 16'h7C00;
        rom[118] = 16'h7C00;
        rom[119] = 16'h7C00;
        rom[120] = 16'h7C00;
        rom[121] = 16'h7C00;
        rom[122] = 16'h7C00;
        rom[123] = 16'h7C00;
        rom[124] = 16'h7C00;
        rom[125] = 16'h7E00;
        rom[126] = 16'h7E00;
        rom[127] = 16'h7E00;

        // Entries 128..143 (fp16: -0.0 .. -0.000427246)
        rom[128] = 16'h3C00;
        rom[129] = 16'h3C00;
        rom[130] = 16'h3C00;
        rom[131] = 16'h3C00;
        rom[132] = 16'h3C00;
        rom[133] = 16'h3C00;
        rom[134] = 16'h3C00;
        rom[135] = 16'h3C00;
        rom[136] = 16'h3C00;
        rom[137] = 16'h3C00;
        rom[138] = 16'h3C00;
        rom[139] = 16'h3C00;
        rom[140] = 16'h3C00;
        rom[141] = 16'h3BFF;
        rom[142] = 16'h3BFF;
        rom[143] = 16'h3BFF;

        // Entries 144..159 (fp16: -0.000488281 .. -0.00683594)
        rom[144] = 16'h3BFF;
        rom[145] = 16'h3BFF;
        rom[146] = 16'h3BFF;
        rom[147] = 16'h3BFE;
        rom[148] = 16'h3BFE;
        rom[149] = 16'h3BFE;
        rom[150] = 16'h3BFD;
        rom[151] = 16'h3BFD;
        rom[152] = 16'h3BFC;
        rom[153] = 16'h3BFB;
        rom[154] = 16'h3BFA;
        rom[155] = 16'h3BF9;
        rom[156] = 16'h3BF8;
        rom[157] = 16'h3BF6;
        rom[158] = 16'h3BF4;
        rom[159] = 16'h3BF2;

        // Entries 160..175 (fp16: -0.0078125 .. -0.109375)
        rom[160] = 16'h3BF0;
        rom[161] = 16'h3BEC;
        rom[162] = 16'h3BE8;
        rom[163] = 16'h3BE4;
        rom[164] = 16'h3BE0;
        rom[165] = 16'h3BD8;
        rom[166] = 16'h3BD1;
        rom[167] = 16'h3BC9;
        rom[168] = 16'h3BC1;
        rom[169] = 16'h3BB2;
        rom[170] = 16'h3BA2;
        rom[171] = 16'h3B93;
        rom[172] = 16'h3B84;
        rom[173] = 16'h3B66;
        rom[174] = 16'h3B49;
        rom[175] = 16'h3B2C;

        // Entries 176..191 (fp16: -0.125 .. -1.75)
        rom[176] = 16'h3B0F;
        rom[177] = 16'h3AD8;
        rom[178] = 16'h3AA2;
        rom[179] = 16'h3A6E;
        rom[180] = 16'h3A3B;
        rom[181] = 16'h39DA;
        rom[182] = 16'h3980;
        rom[183] = 16'h392A;
        rom[184] = 16'h38DA;
        rom[185] = 16'h3848;
        rom[186] = 16'h378F;
        rom[187] = 16'h36AB;
        rom[188] = 16'h35E3;
        rom[189] = 16'h3496;
        rom[190] = 16'h3324;
        rom[191] = 16'h3190;

        // Entries 192..207 (fp16: -2 .. -28)
        rom[192] = 16'h3055;
        rom[193] = 16'h2D41;
        rom[194] = 16'h2A5F;
        rom[195] = 16'h27BB;
        rom[196] = 16'h24B0;
        rom[197] = 16'h1EE6;
        rom[198] = 16'h1914;
        rom[199] = 16'h1378;
        rom[200] = 16'h0D7F;
        rom[201] = 16'h02FA;
        rom[202] = 16'h0067;
        rom[203] = 16'h000E;
        rom[204] = 16'h0002;
        rom[205] = 16'h0000;
        rom[206] = 16'h0000;
        rom[207] = 16'h0000;

        // Entries 208..223 (fp16: -32 .. -448)
        rom[208] = 16'h0000;
        rom[209] = 16'h0000;
        rom[210] = 16'h0000;
        rom[211] = 16'h0000;
        rom[212] = 16'h0000;
        rom[213] = 16'h0000;
        rom[214] = 16'h0000;
        rom[215] = 16'h0000;
        rom[216] = 16'h0000;
        rom[217] = 16'h0000;
        rom[218] = 16'h0000;
        rom[219] = 16'h0000;
        rom[220] = 16'h0000;
        rom[221] = 16'h0000;
        rom[222] = 16'h0000;
        rom[223] = 16'h0000;

        // Entries 224..239 (fp16: -512 .. -7168)
        rom[224] = 16'h0000;
        rom[225] = 16'h0000;
        rom[226] = 16'h0000;
        rom[227] = 16'h0000;
        rom[228] = 16'h0000;
        rom[229] = 16'h0000;
        rom[230] = 16'h0000;
        rom[231] = 16'h0000;
        rom[232] = 16'h0000;
        rom[233] = 16'h0000;
        rom[234] = 16'h0000;
        rom[235] = 16'h0000;
        rom[236] = 16'h0000;
        rom[237] = 16'h0000;
        rom[238] = 16'h0000;
        rom[239] = 16'h0000;

        // Entries 240..255 (fp16: -8192 .. NaN)
        rom[240] = 16'h0000;
        rom[241] = 16'h0000;
        rom[242] = 16'h0000;
        rom[243] = 16'h0000;
        rom[244] = 16'h0000;
        rom[245] = 16'h0000;
        rom[246] = 16'h0000;
        rom[247] = 16'h0000;
        rom[248] = 16'h0000;
        rom[249] = 16'h0000;
        rom[250] = 16'h0000;
        rom[251] = 16'h0000;
        rom[252] = 16'h0000;
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
