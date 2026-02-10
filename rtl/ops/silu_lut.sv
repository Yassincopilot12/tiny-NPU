// =============================================================================
// silu_lut.sv - SiLU activation function via 256-entry ROM lookup table
// Input:  x[7:0] signed int8
// Output: silu_out[7:0] signed int8
// Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Scale: int8 maps to float via /32, so int8 range [-128,127] -> [-4.0, 3.97]
// LUT[i] = round(SiLU(signed_i / 32.0) * 32.0) clamped to [-128, 127]
// Pipeline: 1-cycle registered output
// =============================================================================
import npu_pkg::*;
import fixed_pkg::*;

module silu_lut (
    input  logic       clk,
    input  logic [7:0] addr,       // treated as signed int8 index
    output logic [7:0] data_out    // signed int8 SiLU result
);

    // 256-entry ROM - inferred as block RAM
    (* rom_style = "block" *)
    logic [7:0] rom [0:255];

    // ----------------------------------------------------------------
    // ROM initialization
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // Scale: x_float = signed_i / 32.0
    // LUT[i] = clamp(round(SiLU(x_float) * 32.0), -128, 127)
    //
    // Key values (signed int8 index -> SiLU output):
    //   0 -> 0           (SiLU(0) = 0)
    //  32 -> 23          (SiLU(1.0) ~ 0.731 -> 23.4)
    //  64 -> 56          (SiLU(2.0) ~ 1.762 -> 56.4)
    //  96 -> 91          (SiLU(3.0) ~ 2.858 -> 91.4)
    // -32 -> -9          (SiLU(-1.0) ~ -0.269 -> -8.6)
    // -64 -> -8          (SiLU(-2.0) ~ -0.238 -> -7.6)
    // -128-> -2          (SiLU(-4.0) ~ -0.072 -> -2.3)
    // ----------------------------------------------------------------
    initial begin
        // Positive indices 0..127 (signed 0 to +127)
        // x_float = i/32.0, range [0, 3.97]
        rom[  0] = 8'd0;     // SiLU(0) = 0
        rom[  1] = 8'd1;
        rom[  2] = 8'd1;
        rom[  3] = 8'd2;
        rom[  4] = 8'd2;
        rom[  5] = 8'd3;
        rom[  6] = 8'd3;
        rom[  7] = 8'd4;
        rom[  8] = 8'd4;     // SiLU(0.25) ~ 0.138
        rom[  9] = 8'd5;
        rom[ 10] = 8'd6;
        rom[ 11] = 8'd6;
        rom[ 12] = 8'd7;
        rom[ 13] = 8'd8;
        rom[ 14] = 8'd9;
        rom[ 15] = 8'd9;
        rom[ 16] = 8'd10;    // SiLU(0.5) ~ 0.311
        rom[ 17] = 8'd11;
        rom[ 18] = 8'd11;
        rom[ 19] = 8'd12;
        rom[ 20] = 8'd13;
        rom[ 21] = 8'd14;
        rom[ 22] = 8'd15;
        rom[ 23] = 8'd15;
        rom[ 24] = 8'd16;    // SiLU(0.75) ~ 0.498
        rom[ 25] = 8'd17;
        rom[ 26] = 8'd18;
        rom[ 27] = 8'd19;
        rom[ 28] = 8'd20;
        rom[ 29] = 8'd21;
        rom[ 30] = 8'd22;
        rom[ 31] = 8'd22;
        rom[ 32] = 8'd23;    // SiLU(1.0) ~ 0.731
        rom[ 33] = 8'd24;
        rom[ 34] = 8'd25;
        rom[ 35] = 8'd26;
        rom[ 36] = 8'd27;
        rom[ 37] = 8'd28;
        rom[ 38] = 8'd29;
        rom[ 39] = 8'd30;
        rom[ 40] = 8'd31;
        rom[ 41] = 8'd32;
        rom[ 42] = 8'd33;
        rom[ 43] = 8'd34;
        rom[ 44] = 8'd35;
        rom[ 45] = 8'd36;
        rom[ 46] = 8'd37;
        rom[ 47] = 8'd38;
        rom[ 48] = 8'd39;    // SiLU(1.5) ~ 1.243
        rom[ 49] = 8'd40;
        rom[ 50] = 8'd41;
        rom[ 51] = 8'd42;
        rom[ 52] = 8'd43;
        rom[ 53] = 8'd45;
        rom[ 54] = 8'd46;
        rom[ 55] = 8'd47;
        rom[ 56] = 8'd48;
        rom[ 57] = 8'd49;
        rom[ 58] = 8'd50;
        rom[ 59] = 8'd51;
        rom[ 60] = 8'd52;
        rom[ 61] = 8'd53;
        rom[ 62] = 8'd54;
        rom[ 63] = 8'd55;
        rom[ 64] = 8'd56;    // SiLU(2.0) ~ 1.762
        rom[ 65] = 8'd57;
        rom[ 66] = 8'd59;
        rom[ 67] = 8'd60;
        rom[ 68] = 8'd61;
        rom[ 69] = 8'd62;
        rom[ 70] = 8'd63;
        rom[ 71] = 8'd64;
        rom[ 72] = 8'd65;
        rom[ 73] = 8'd66;
        rom[ 74] = 8'd67;
        rom[ 75] = 8'd68;
        rom[ 76] = 8'd70;
        rom[ 77] = 8'd71;
        rom[ 78] = 8'd72;
        rom[ 79] = 8'd73;
        rom[ 80] = 8'd74;
        rom[ 81] = 8'd75;
        rom[ 82] = 8'd76;
        rom[ 83] = 8'd77;
        rom[ 84] = 8'd78;
        rom[ 85] = 8'd79;
        rom[ 86] = 8'd81;
        rom[ 87] = 8'd82;
        rom[ 88] = 8'd83;
        rom[ 89] = 8'd84;
        rom[ 90] = 8'd85;
        rom[ 91] = 8'd86;
        rom[ 92] = 8'd87;
        rom[ 93] = 8'd88;
        rom[ 94] = 8'd89;
        rom[ 95] = 8'd90;
        rom[ 96] = 8'd91;    // SiLU(3.0) ~ 2.858
        rom[ 97] = 8'd93;
        rom[ 98] = 8'd94;
        rom[ 99] = 8'd95;
        rom[100] = 8'd96;
        rom[101] = 8'd97;
        rom[102] = 8'd98;
        rom[103] = 8'd99;
        rom[104] = 8'd100;
        rom[105] = 8'd101;
        rom[106] = 8'd102;
        rom[107] = 8'd103;
        rom[108] = 8'd104;
        rom[109] = 8'd106;
        rom[110] = 8'd107;
        rom[111] = 8'd108;
        rom[112] = 8'd109;
        rom[113] = 8'd110;
        rom[114] = 8'd111;
        rom[115] = 8'd112;
        rom[116] = 8'd113;
        rom[117] = 8'd114;
        rom[118] = 8'd115;
        rom[119] = 8'd116;
        rom[120] = 8'd117;
        rom[121] = 8'd118;
        rom[122] = 8'd119;
        rom[123] = 8'd120;
        rom[124] = 8'd121;
        rom[125] = 8'd123;
        rom[126] = 8'd124;
        rom[127] = 8'd125;

        // Negative indices 128..255 (signed -128 to -1)
        // x_float = (i-256)/32.0, range [-4.0, -0.03125]
        // SiLU has a shallow trough near x=-1.28 (~-0.278)
        rom[128] = 8'hFE;    // SiLU(-4.0) ~ -0.072 -> -2
        rom[129] = 8'hFE;
        rom[130] = 8'hFE;
        rom[131] = 8'hFE;
        rom[132] = 8'hFD;
        rom[133] = 8'hFD;
        rom[134] = 8'hFD;
        rom[135] = 8'hFD;
        rom[136] = 8'hFD;
        rom[137] = 8'hFD;
        rom[138] = 8'hFD;
        rom[139] = 8'hFD;
        rom[140] = 8'hFD;
        rom[141] = 8'hFD;
        rom[142] = 8'hFD;
        rom[143] = 8'hFD;
        rom[144] = 8'hFD;
        rom[145] = 8'hFD;
        rom[146] = 8'hFD;
        rom[147] = 8'hFD;
        rom[148] = 8'hFC;
        rom[149] = 8'hFC;
        rom[150] = 8'hFC;
        rom[151] = 8'hFC;
        rom[152] = 8'hFC;
        rom[153] = 8'hFC;
        rom[154] = 8'hFC;
        rom[155] = 8'hFC;
        rom[156] = 8'hFC;
        rom[157] = 8'hFC;
        rom[158] = 8'hFC;
        rom[159] = 8'hFC;
        rom[160] = 8'hFB;    // SiLU(-3.0) ~ -0.142 -> -5
        rom[161] = 8'hFB;
        rom[162] = 8'hFB;
        rom[163] = 8'hFB;
        rom[164] = 8'hFB;
        rom[165] = 8'hFB;
        rom[166] = 8'hFB;
        rom[167] = 8'hFB;
        rom[168] = 8'hFB;
        rom[169] = 8'hFB;
        rom[170] = 8'hFB;
        rom[171] = 8'hFA;
        rom[172] = 8'hFA;
        rom[173] = 8'hFA;
        rom[174] = 8'hFA;
        rom[175] = 8'hFA;
        rom[176] = 8'hFA;
        rom[177] = 8'hFA;
        rom[178] = 8'hFA;
        rom[179] = 8'hFA;
        rom[180] = 8'hFA;
        rom[181] = 8'hF9;
        rom[182] = 8'hF9;
        rom[183] = 8'hF9;
        rom[184] = 8'hF9;
        rom[185] = 8'hF9;
        rom[186] = 8'hF9;
        rom[187] = 8'hF9;
        rom[188] = 8'hF9;
        rom[189] = 8'hF9;
        rom[190] = 8'hF9;
        rom[191] = 8'hF8;
        rom[192] = 8'hF8;    // SiLU(-2.0) ~ -0.238 -> -8
        rom[193] = 8'hF8;
        rom[194] = 8'hF8;
        rom[195] = 8'hF8;
        rom[196] = 8'hF8;
        rom[197] = 8'hF8;
        rom[198] = 8'hF8;
        rom[199] = 8'hF8;
        rom[200] = 8'hF8;
        rom[201] = 8'hF8;
        rom[202] = 8'hF8;
        rom[203] = 8'hF8;
        rom[204] = 8'hF7;
        rom[205] = 8'hF7;
        rom[206] = 8'hF7;
        rom[207] = 8'hF7;
        rom[208] = 8'hF7;
        rom[209] = 8'hF7;
        rom[210] = 8'hF7;
        rom[211] = 8'hF7;
        rom[212] = 8'hF7;
        rom[213] = 8'hF7;
        rom[214] = 8'hF7;
        rom[215] = 8'hF7;
        rom[216] = 8'hF7;
        rom[217] = 8'hF7;
        rom[218] = 8'hF7;
        rom[219] = 8'hF7;
        rom[220] = 8'hF7;
        rom[221] = 8'hF7;
        rom[222] = 8'hF7;
        rom[223] = 8'hF7;
        rom[224] = 8'hF7;    // SiLU(-1.0) ~ -0.269 -> -9
        rom[225] = 8'hF7;
        rom[226] = 8'hF8;
        rom[227] = 8'hF8;
        rom[228] = 8'hF8;
        rom[229] = 8'hF8;
        rom[230] = 8'hF8;
        rom[231] = 8'hF8;
        rom[232] = 8'hF8;
        rom[233] = 8'hF8;
        rom[234] = 8'hF9;
        rom[235] = 8'hF9;
        rom[236] = 8'hF9;
        rom[237] = 8'hF9;
        rom[238] = 8'hF9;
        rom[239] = 8'hFA;
        rom[240] = 8'hFA;    // SiLU(-0.5) ~ -0.154
        rom[241] = 8'hFA;
        rom[242] = 8'hFB;
        rom[243] = 8'hFB;
        rom[244] = 8'hFB;
        rom[245] = 8'hFB;
        rom[246] = 8'hFC;
        rom[247] = 8'hFC;
        rom[248] = 8'hFC;    // SiLU(-0.25)
        rom[249] = 8'hFD;
        rom[250] = 8'hFD;
        rom[251] = 8'hFE;
        rom[252] = 8'hFE;
        rom[253] = 8'hFF;
        rom[254] = 8'hFF;
        rom[255] = 8'h00;    // SiLU(-0.03125) ~ 0.0 -> rounds to 0
    end

    // ----------------------------------------------------------------
    // Registered read (1-cycle latency)
    // ----------------------------------------------------------------
    always_ff @(posedge clk) begin
        data_out <= rom[addr];
    end

endmodule
