// =============================================================================
// graph_log_lut_fp16.sv - Natural LOG function via 256-entry ROM lookup table (FP16)
// Input:  addr[7:0] = upper 8 bits of FP16 input value
// Output: data_out[15:0] = FP16 result
// Formula: LUT[i] = fp16(ln(fp16_input))
// Pipeline: 1-cycle registered output
// =============================================================================
`default_nettype none

module graph_log_lut_fp16 (
    input  wire        clk,
    input  wire  [7:0] addr,       // upper 8 bits of FP16 input
    output logic [15:0] data_out    // FP16 result
);

    // 256-entry ROM - inferred as block RAM
    (* rom_style = "block" *)
    logic [15:0] rom [0:255];

    // ----------------------------------------------------------------
    // ROM initialization - log(fp16_value) lookup
    //
    // Key values (addr -> fp16 input -> log output):
    //   0x3C (+1.0)  -> log(1)    = 0.0    -> 0x0000
    //   0x40 (+2.0)  -> log(2)    = 0.693  -> 0x398C
    //   0x00 (+0.0)  -> log(0)    = -Inf   -> 0xFC00
    //   0x7C (+Inf)  -> log(+Inf) = +Inf   -> 0x7C00
    //   0xBC (-1.0)  -> log(-1)   = NaN    -> 0x7E00
    //   0x7E (NaN)   -> NaN                -> 0x7E00
    // ----------------------------------------------------------------
    initial begin
        // Entries 0..15 (fp16: +0.0 .. 0.000427246)
        rom[  0] = 16'hFC00;
        rom[  1] = 16'hC98C;
        rom[  2] = 16'hC933;
        rom[  3] = 16'hC8FF;
        rom[  4] = 16'hC8DA;
        rom[  5] = 16'hC8BE;
        rom[  6] = 16'hC8A6;
        rom[  7] = 16'hC892;
        rom[  8] = 16'hC881;
        rom[  9] = 16'hC865;
        rom[ 10] = 16'hC84D;
        rom[ 11] = 16'hC83A;
        rom[ 12] = 16'hC829;
        rom[ 13] = 16'hC80C;
        rom[ 14] = 16'hC7EA;
        rom[ 15] = 16'hC7C2;

        // Entries 16..31 (fp16: 0.000488281 .. 0.00683594)
        rom[ 16] = 16'hC7A0;
        rom[ 17] = 16'hC767;
        rom[ 18] = 16'hC738;
        rom[ 19] = 16'hC711;
        rom[ 20] = 16'hC6EE;
        rom[ 21] = 16'hC6B5;
        rom[ 22] = 16'hC687;
        rom[ 23] = 16'hC65F;
        rom[ 24] = 16'hC63D;
        rom[ 25] = 16'hC604;
        rom[ 26] = 16'hC5D5;
        rom[ 27] = 16'hC5AE;
        rom[ 28] = 16'hC58C;
        rom[ 29] = 16'hC552;
        rom[ 30] = 16'hC524;
        rom[ 31] = 16'hC4FC;

        // Entries 32..47 (fp16: 0.0078125 .. 0.109375)
        rom[ 32] = 16'hC4DA;
        rom[ 33] = 16'hC4A1;
        rom[ 34] = 16'hC472;
        rom[ 35] = 16'hC44B;
        rom[ 36] = 16'hC429;
        rom[ 37] = 16'hC3DF;
        rom[ 38] = 16'hC382;
        rom[ 39] = 16'hC333;
        rom[ 40] = 16'hC2EE;
        rom[ 41] = 16'hC27C;
        rom[ 42] = 16'hC21F;
        rom[ 43] = 16'hC1D0;
        rom[ 44] = 16'hC18C;
        rom[ 45] = 16'hC119;
        rom[ 46] = 16'hC0BC;
        rom[ 47] = 16'hC06D;

        // Entries 48..63 (fp16: 0.125 .. 1.75)
        rom[ 48] = 16'hC029;
        rom[ 49] = 16'hBF6D;
        rom[ 50] = 16'hBEB2;
        rom[ 51] = 16'hBE14;
        rom[ 52] = 16'hBD8C;
        rom[ 53] = 16'hBCA7;
        rom[ 54] = 16'hBBD9;
        rom[ 55] = 16'hBA9D;
        rom[ 56] = 16'hB98C;
        rom[ 57] = 16'hB785;
        rom[ 58] = 16'hB49A;
        rom[ 59] = 16'hB046;
        rom[ 60] = 16'h0000;
        rom[ 61] = 16'h3324;
        rom[ 62] = 16'h367D;
        rom[ 63] = 16'h387A;

        // Entries 64..79 (fp16: 2 .. 28)
        rom[ 64] = 16'h398C;
        rom[ 65] = 16'h3B55;
        rom[ 66] = 16'h3C65;
        rom[ 67] = 16'h3D03;
        rom[ 68] = 16'h3D8C;
        rom[ 69] = 16'h3E70;
        rom[ 70] = 16'h3F2B;
        rom[ 71] = 16'h3FC9;
        rom[ 72] = 16'h4029;
        rom[ 73] = 16'h409B;
        rom[ 74] = 16'h40F8;
        rom[ 75] = 16'h4147;
        rom[ 76] = 16'h418C;
        rom[ 77] = 16'h41FE;
        rom[ 78] = 16'h425B;
        rom[ 79] = 16'h42AA;

        // Entries 80..95 (fp16: 32 .. 448)
        rom[ 80] = 16'h42EE;
        rom[ 81] = 16'h4361;
        rom[ 82] = 16'h43BE;
        rom[ 83] = 16'h4406;
        rom[ 84] = 16'h4429;
        rom[ 85] = 16'h4462;
        rom[ 86] = 16'h4490;
        rom[ 87] = 16'h44B8;
        rom[ 88] = 16'h44DA;
        rom[ 89] = 16'h4513;
        rom[ 90] = 16'h4542;
        rom[ 91] = 16'h4569;
        rom[ 92] = 16'h458C;
        rom[ 93] = 16'h45C5;
        rom[ 94] = 16'h45F3;
        rom[ 95] = 16'h461B;

        // Entries 96..111 (fp16: 512 .. 7168)
        rom[ 96] = 16'h463D;
        rom[ 97] = 16'h4676;
        rom[ 98] = 16'h46A5;
        rom[ 99] = 16'h46CC;
        rom[100] = 16'h46EE;
        rom[101] = 16'h4728;
        rom[102] = 16'h4756;
        rom[103] = 16'h477E;
        rom[104] = 16'h47A0;
        rom[105] = 16'h47D9;
        rom[106] = 16'h4804;
        rom[107] = 16'h4818;
        rom[108] = 16'h4829;
        rom[109] = 16'h4845;
        rom[110] = 16'h485D;
        rom[111] = 16'h4870;

        // Entries 112..127 (fp16: 8192 .. NaN)
        rom[112] = 16'h4881;
        rom[113] = 16'h489E;
        rom[114] = 16'h48B5;
        rom[115] = 16'h48C9;
        rom[116] = 16'h48DA;
        rom[117] = 16'h48F7;
        rom[118] = 16'h490E;
        rom[119] = 16'h4922;
        rom[120] = 16'h4933;
        rom[121] = 16'h494F;
        rom[122] = 16'h4967;
        rom[123] = 16'h497A;
        rom[124] = 16'h7C00;
        rom[125] = 16'h7E00;
        rom[126] = 16'h7E00;
        rom[127] = 16'h7E00;

        // Entries 128..143 (fp16: -0.0 .. -0.000427246)
        rom[128] = 16'hFC00;
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
