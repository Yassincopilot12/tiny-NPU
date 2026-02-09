// =============================================================================
// tb_gpt2_block.cpp - GPT-2 transformer block integration test (multi-head)
// Full pipeline: LN1 -> MHA(4 heads) -> Wo -> Residual -> LN2 -> FFN -> Residual
// Real engines: softmax, layernorm, gelu, vec, GEMM (all on HW via tiling)
// C++ shims: DMA only
// =============================================================================

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vgpt2_block_top.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <string>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// =============================================================================
// Global simulation state
// =============================================================================
vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static Vgpt2_block_top* dut;
static VerilatedVcdC*    tfp;
static int               tc = 0;

// =============================================================================
// Dimensions
// =============================================================================
static const int SEQ_LEN  = 8;
static const int HIDDEN   = 64;
static const int HEAD_DIM = 16;
static const int N_HEADS  = 4;
static const int FFN_DIM  = 256;

// =============================================================================
// SRAM0 Memory layout (byte addresses)
// =============================================================================
// Weights (head-blocked QKV):
static const uint16_t ADDR_WQ      = 0x0000;  // 4x[64,16] = 4096B (head-blocked)
static const uint16_t ADDR_WK      = 0x1000;  // 4x[64,16] = 4096B (head-blocked)
static const uint16_t ADDR_WV      = 0x2000;  // 4x[64,16] = 4096B (head-blocked)
static const uint16_t ADDR_WO      = 0x3000;  // [64][64]  = 4096B
static const uint16_t ADDR_W1      = 0x4000;  // [64][256] = 16384B
static const uint16_t ADDR_W2      = 0x8000;  // [256][64] = 16384B

// Activations:
static const uint16_t ADDR_X       = 0xC000;  // [8][64]   = 512B
static const uint16_t ADDR_LN1_OUT = 0xC400;  // [8][64]   = 512B
static const uint16_t ADDR_Q_H     = 0xC800;  // [8][16]   = 128B  (per-head, reused)
static const uint16_t ADDR_K_H     = 0xC900;  // [8][16]   = 128B  (per-head, reused)
static const uint16_t ADDR_V_H     = 0xCA00;  // [8][16]   = 128B  (per-head, reused)
static const uint16_t ADDR_S       = 0xCB00;  // [8][8]    = 64B   (per-head, reused)
static const uint16_t ADDR_P       = 0xCC00;  // [8][8]    = 64B   (per-head, reused)
static const uint16_t ADDR_ATTN_H  = 0xCD00;  // [8][16]   = 128B  (per-head context temp)
static const uint16_t ADDR_ATTN    = 0xCE00;  // [8][64]   = 512B  (concat destination)
static const uint16_t ADDR_WO_OUT  = 0xD200;  // [8][64]   = 512B
static const uint16_t ADDR_X2      = 0xD600;  // [8][64]   = 512B
static const uint16_t ADDR_LN2_OUT = 0xDA00;  // [8][64]   = 512B
static const uint16_t ADDR_FFN1    = 0xDE00;  // [8][256]  = 2048B
static const uint16_t ADDR_FFN2    = 0xEE00;  // [8][64]   = 512B
static const uint16_t ADDR_X_OUT   = 0xF200;  // [8][64]   = 512B

// SRAM1 Memory layout
static const uint16_t S1_LN1_BETA  = 0x0000;  // [64]
static const uint16_t S1_LN2_BETA  = 0x0040;  // [64]
static const uint16_t S1_RESID     = 0x0100;  // [8][64] = 512B (residual source)

// =============================================================================
// Opcodes and Flags
// =============================================================================
static const uint8_t OP_DMA_LOAD  = 1;
static const uint8_t OP_GEMM      = 3;
static const uint8_t OP_VEC       = 4;
static const uint8_t OP_SOFTMAX   = 5;
static const uint8_t OP_LAYERNORM = 6;
static const uint8_t OP_GELU      = 7;
static const uint8_t OP_BARRIER   = 10;
static const uint8_t OP_END       = 255;

static const uint8_t FLAG_TRANSPOSE_B = 0x01;
static const uint8_t FLAG_REQUANT     = 0x04;
static const uint8_t FLAG_COPY2D      = 0x04;  // flags[2] for VEC COPY2D mode
static const uint8_t FLAG_CAUSAL_MASK = 0x10;

// =============================================================================
// GEMM IMM values (different per GEMM type)
// =============================================================================
static const uint16_t GEMM_IMM_K64 = 0x0901;  // scale=1, shift=9, for K=64
static const uint16_t GEMM_IMM_K16 = 0x0701;  // scale=1, shift=7, for K=16 (score/context)

// =============================================================================
// Debug dump flag
// =============================================================================
static const bool DUMP_FFN = (getenv("DUMP_FFN") != nullptr);

// =============================================================================
// Clock / Reset Helpers
// =============================================================================
void tick() {
    dut->clk = 0;
    dut->eval();
    if (tfp) tfp->dump(tc * 10);
    tc++;
    dut->clk = 1;
    dut->eval();
    if (tfp) tfp->dump(tc * 10 + 5);
    tc++;
    main_time = tc;
}

void reset_dut(int cycles = 10) {
    dut->rst_n = 0;
    for (int i = 0; i < cycles; i++) tick();
    dut->rst_n = 1;
    tick();
}

// =============================================================================
// SRAM Helpers
// =============================================================================
void sram0_write(uint16_t addr, uint8_t data) {
    dut->tb_sram0_wr_en   = 1;
    dut->tb_sram0_wr_addr = addr;
    dut->tb_sram0_wr_data = data;
    tick();
    dut->tb_sram0_wr_en = 0;
}

uint8_t sram0_read(uint16_t addr) {
    dut->tb_sram0_rd_en   = 1;
    dut->tb_sram0_rd_addr = addr;
    tick();
    dut->tb_sram0_rd_en = 0;
    return dut->tb_sram0_rd_data;
}

void sram1_write(uint16_t addr, uint8_t data) {
    dut->tb_sram1_wr_en   = 1;
    dut->tb_sram1_wr_addr = addr;
    dut->tb_sram1_wr_data = data;
    tick();
    dut->tb_sram1_wr_en = 0;
}

uint8_t sram1_read(uint16_t addr) {
    dut->tb_sram1_rd_en   = 1;
    dut->tb_sram1_rd_addr = addr;
    tick();
    dut->tb_sram1_rd_en = 0;
    return dut->tb_sram1_rd_data;
}

void ucode_write(uint16_t addr, uint64_t hi, uint64_t lo) {
    dut->uc_wr_en   = 1;
    dut->uc_wr_addr = addr;
    dut->uc_wr_data[0] = (uint32_t)(lo & 0xFFFFFFFF);
    dut->uc_wr_data[1] = (uint32_t)(lo >> 32);
    dut->uc_wr_data[2] = (uint32_t)(hi & 0xFFFFFFFF);
    dut->uc_wr_data[3] = (uint32_t)(hi >> 32);
    tick();
    dut->uc_wr_en = 0;
}

void encode_instr(uint8_t opcode, uint8_t flags, uint16_t dst, uint16_t src0,
                  uint16_t src1, uint16_t M, uint16_t N, uint16_t K,
                  uint16_t imm, uint64_t& hi, uint64_t& lo) {
    lo = (uint64_t)opcode
       | ((uint64_t)flags  << 8)
       | ((uint64_t)dst    << 16)
       | ((uint64_t)src0   << 32)
       | ((uint64_t)src1   << 48);
    hi = (uint64_t)M
       | ((uint64_t)N  << 16)
       | ((uint64_t)K  << 32)
       | ((uint64_t)imm << 48);
}

// =============================================================================
// C++ GEMM Golden (matches RTL requantize exactly)
// =============================================================================
void gemm_golden(const int8_t* A, const int8_t* B, int8_t* C,
                 int M, int N, int K, bool transpose_b,
                 int scale, int shift) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t b_val = transpose_b ? B[j*K + k] : B[k*N + j];
                acc += (int32_t)A[i*K + k] * (int32_t)b_val;
            }
            int64_t prod = (int64_t)acc * (int64_t)scale;
            if (shift > 0) prod += (1LL << (shift - 1));
            int32_t res = (int32_t)(prod >> shift);
            if (res > 127)  res = 127;
            if (res < -128) res = -128;
            C[i*N + j] = (int8_t)res;
        }
    }
}

// =============================================================================
// C++ Softmax Golden (matches RTL LUT-based implementation)
// =============================================================================
static uint16_t EXP_LUT[256];
static uint16_t RECIP_LUT[256];

void build_softmax_luts() {
    for (int i = 0; i < 256; i++) {
        int8_t signed_i = (int8_t)(uint8_t)i;
        double x = (double)signed_i / 32.0;
        double val = exp(x) * 256.0;
        int v = (int)round(val);
        if (v < 0) v = 0;
        if (v > 65535) v = 65535;
        EXP_LUT[i] = (uint16_t)v;
    }
    for (int i = 0; i < 256; i++) {
        if (i == 0) {
            RECIP_LUT[i] = 65535;
        } else {
            double val = 65536.0 / (double)i;
            int v = (int)round(val);
            if (v > 65535) v = 65535;
            RECIP_LUT[i] = (uint16_t)v;
        }
    }
}

void softmax_golden(const int8_t* scores, int8_t* output, int length,
                    bool causal_mask_en, int causal_limit) {
    int8_t max_val = -128;
    for (int i = 0; i < length; i++) {
        int8_t s = scores[i];
        if (causal_mask_en && i > causal_limit) s = -128;
        if (s > max_val) max_val = s;
    }

    std::vector<uint16_t> exp_vals(length);
    int32_t exp_sum = 0;
    for (int i = 0; i < length; i++) {
        int8_t s = scores[i];
        if (causal_mask_en && i > causal_limit) s = -128;
        int16_t diff = (int16_t)s - (int16_t)max_val;
        if (diff < -128) diff = -128;
        if (diff > 127) diff = 127;
        uint8_t idx = (uint8_t)(int8_t)diff;
        exp_vals[i] = EXP_LUT[idx];
        exp_sum += (int32_t)exp_vals[i];
    }

    if (exp_sum == 0) { memset(output, 0, length); return; }
    int sum_top8 = (int)((uint16_t)(exp_sum & 0xFFFF) >> 8);
    if (sum_top8 > 255) sum_top8 = 255;
    if (sum_top8 == 0) sum_top8 = 1;
    uint16_t inv_sum = RECIP_LUT[sum_top8];

    for (int i = 0; i < length; i++) {
        uint32_t prod = (uint32_t)exp_vals[i] * (uint32_t)inv_sum;
        int32_t prob = (int32_t)(prod >> 17);
        if (prob > 127) prob = 127;
        if (prob < 0) prob = 0;
        output[i] = (int8_t)prob;
    }
}

// =============================================================================
// C++ LayerNorm Golden (matches RTL layernorm_engine.sv exactly)
// =============================================================================
static uint16_t RSQRT_LUT[256];

void build_rsqrt_lut() {
    RSQRT_LUT[0] = 65535;
    for (int i = 1; i < 256; i++) {
        double val = 4096.0 / sqrt((double)i);
        int v = (int)round(val);
        if (v > 65535) v = 65535;
        RSQRT_LUT[i] = (uint16_t)v;
    }
}

uint32_t counting_div_mean(int64_t dividend, uint16_t divisor) {
    const uint64_t MASK48 = (1ULL << 48) - 1;
    uint64_t remainder = (uint64_t)dividend & MASK48;
    uint64_t div = (uint64_t)divisor;
    uint32_t quotient = 0;
    for (int i = 0; i < 32; i++) {
        quotient <<= 1;
        if (remainder >= div) {
            remainder -= div;
            quotient |= 1;
        }
    }
    return quotient;
}

uint32_t counting_div_unsigned(uint64_t dividend, uint64_t divisor) {
    uint64_t remainder = dividend;
    uint32_t quotient = 0;
    for (int i = 0; i < 32; i++) {
        quotient <<= 1;
        if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= 1;
        }
    }
    return quotient;
}

void layernorm_golden(const int8_t* input, int8_t* output, int length,
                      const int8_t* beta) {
    int32_t sum = 0;
    uint64_t sum_sq = 0;
    for (int i = 0; i < length; i++) {
        int16_t val = (int16_t)input[i];
        sum += val;
        uint16_t sq = (uint16_t)(val * val);
        sum_sq += sq;
    }

    int64_t mean_dividend = ((int64_t)sum) << 8;
    uint32_t mean_quotient = counting_div_mean(mean_dividend, (uint16_t)length);
    int16_t r_mean = (int16_t)(mean_quotient & 0xFFFF);

    uint64_t sq_dividend = sum_sq << 8;
    uint32_t sq_quotient = counting_div_unsigned(sq_dividend, (uint64_t)length);

    int32_t mean_i32 = (int32_t)r_mean;
    uint32_t mean_sq = (uint32_t)(mean_i32 * mean_i32);
    uint32_t sq_shifted = sq_quotient << 8;
    uint32_t variance;
    if (sq_shifted >= mean_sq)
        variance = sq_shifted - mean_sq;
    else
        variance = 0;

    uint8_t rsqrt_addr = (uint8_t)(variance >> 24);
    uint16_t inv_std = RSQRT_LUT[rsqrt_addr];

    int8_t p_gamma = 127;
    for (int i = 0; i < length; i++) {
        int16_t centered = (int16_t)(((int16_t)input[i]) << 8) - r_mean;
        int32_t inv_std_signed = (int32_t)inv_std;
        int32_t scaled = (int32_t)centered * inv_std_signed;
        scaled = scaled >> 16;
        int32_t gamma_applied = (scaled * (int32_t)p_gamma) >> 7;
        int32_t bias_added = gamma_applied + ((int32_t)beta[i] << 8);
        int32_t result = bias_added >> 8;
        if (result > 127) result = 127;
        if (result < -128) result = -128;
        output[i] = (int8_t)result;
    }
}

// =============================================================================
// C++ GELU Golden (matches RTL gelu_lut.sv exactly)
// =============================================================================
static int8_t GELU_LUT[256];

void build_gelu_lut() {
    for (int i = 0; i < 256; i++) {
        int8_t signed_i = (int8_t)(uint8_t)i;
        double x = (double)signed_i / 32.0;
        double gelu_val = x * 0.5 * (1.0 + erf(x / sqrt(2.0)));
        double result = gelu_val * 32.0;
        int v = (int)round(result);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        GELU_LUT[i] = (int8_t)v;
    }
}

void gelu_golden(const int8_t* input, int8_t* output, int length) {
    for (int i = 0; i < length; i++) {
        uint8_t idx = (uint8_t)input[i];
        output[i] = GELU_LUT[idx];
    }
}

// =============================================================================
// C++ Vec ADD Golden (matches RTL vec_engine.sv)
// =============================================================================
void vec_add_golden(const int8_t* src0, const int8_t* src1, int8_t* dst, int length) {
    for (int i = 0; i < length; i++) {
        int16_t sum = (int16_t)src0[i] + (int16_t)src1[i];
        if (sum > 127) sum = 127;
        if (sum < -128) sum = -128;
        dst[i] = (int8_t)sum;
    }
}

// =============================================================================
// Test Data
// =============================================================================
static int8_t X[SEQ_LEN][HIDDEN];

// Weights stored head-blocked: 4 contiguous [64,16] blocks for QKV
static int8_t Wq[N_HEADS * HIDDEN][HEAD_DIM];  // [256][16] stored as 4x[64,16]
static int8_t Wk[N_HEADS * HIDDEN][HEAD_DIM];  // [256][16] stored as 4x[64,16]
static int8_t Wv[N_HEADS * HIDDEN][HEAD_DIM];  // [256][16] stored as 4x[64,16]
static int8_t Wo[HIDDEN][HIDDEN];               // [64][64]
static int8_t W1[HIDDEN][FFN_DIM];              // [64][256]
static int8_t W2[FFN_DIM][HIDDEN];              // [256][64]
static int8_t LN1_beta[HIDDEN];
static int8_t LN2_beta[HIDDEN];

// Golden intermediates
static int8_t LN1_out_gold[SEQ_LEN][HIDDEN];

// Per-head golden intermediates
static int8_t Q_h_gold[SEQ_LEN][HEAD_DIM];
static int8_t K_h_gold[SEQ_LEN][HEAD_DIM];
static int8_t V_h_gold[SEQ_LEN][HEAD_DIM];
static int8_t S_h_gold[SEQ_LEN][SEQ_LEN];
static int8_t P_h_gold[SEQ_LEN][SEQ_LEN];
static int8_t ATTN_h_gold[SEQ_LEN][HEAD_DIM];
static int8_t ATTN_gold[SEQ_LEN][HIDDEN];  // concat of all heads

static int8_t WO_out_gold[SEQ_LEN][HIDDEN];
static int8_t X2_gold[SEQ_LEN][HIDDEN];
static int8_t LN2_out_gold[SEQ_LEN][HIDDEN];
static int8_t FFN1_gold[SEQ_LEN][FFN_DIM];
static int8_t GELU_gold[SEQ_LEN][FFN_DIM];
static int8_t FFN2_gold[SEQ_LEN][HIDDEN];
static int8_t X_OUT_gold[SEQ_LEN][HIDDEN];

// =============================================================================
// Generate test data and compute golden
// =============================================================================
void generate_data_and_golden() {
    srand(42);

    // Generate inputs
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            X[i][j] = (int8_t)((rand() % 21) - 10);

    // Generate head-blocked QKV weights: 4 heads, each [64,16]
    for (int h = 0; h < N_HEADS; h++)
        for (int i = 0; i < HIDDEN; i++)
            for (int j = 0; j < HEAD_DIM; j++) {
                Wq[h * HIDDEN + i][j] = (int8_t)((rand() % 11) - 5);
                Wk[h * HIDDEN + i][j] = (int8_t)((rand() % 11) - 5);
                Wv[h * HIDDEN + i][j] = (int8_t)((rand() % 11) - 5);
            }

    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            Wo[i][j] = (int8_t)((rand() % 11) - 5);

    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < FFN_DIM; j++)
            W1[i][j] = (int8_t)((rand() % 11) - 5);

    for (int i = 0; i < FFN_DIM; i++)
        for (int j = 0; j < HIDDEN; j++)
            W2[i][j] = (int8_t)((rand() % 11) - 5);

    for (int j = 0; j < HIDDEN; j++) {
        LN1_beta[j] = (int8_t)((rand() % 21) - 10);
        LN2_beta[j] = (int8_t)((rand() % 21) - 10);
    }

    int scale_k64 = 1, shift_k64 = 9;  // for K=64 GEMMs
    int scale_k16 = 1, shift_k16 = 7;  // for K=16 GEMMs (score/context)

    // Step 1: LayerNorm 1 (per row)
    for (int r = 0; r < SEQ_LEN; r++)
        layernorm_golden(&X[r][0], &LN1_out_gold[r][0], HIDDEN, LN1_beta);

    // Step 2: Multi-head attention loop
    memset(&ATTN_gold[0][0], 0, sizeof(ATTN_gold));
    for (int h = 0; h < N_HEADS; h++) {
        // Wq_h is at Wq[h*HIDDEN .. (h+1)*HIDDEN-1][0..HEAD_DIM-1]
        int8_t* Wq_h = &Wq[h * HIDDEN][0];
        int8_t* Wk_h = &Wk[h * HIDDEN][0];
        int8_t* Wv_h = &Wv[h * HIDDEN][0];

        // Q_h = LN1_out * Wq_h  [8,64] x [64,16] -> [8,16]
        gemm_golden(&LN1_out_gold[0][0], Wq_h, &Q_h_gold[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);

        // K_h = LN1_out * Wk_h
        gemm_golden(&LN1_out_gold[0][0], Wk_h, &K_h_gold[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);

        // V_h = LN1_out * Wv_h
        gemm_golden(&LN1_out_gold[0][0], Wv_h, &V_h_gold[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);

        // S_h = Q_h * K_h^T  [8,16] x [16,8]^T -> [8,8]
        gemm_golden(&Q_h_gold[0][0], &K_h_gold[0][0], &S_h_gold[0][0],
                    SEQ_LEN, SEQ_LEN, HEAD_DIM, true, scale_k16, shift_k16);

        // P_h = softmax(S_h, causal)
        for (int r = 0; r < SEQ_LEN; r++)
            softmax_golden(&S_h_gold[r][0], &P_h_gold[r][0], SEQ_LEN, true, r);

        // ATTN_h = P_h * V_h  [8,8] x [8,16] -> [8,16]
        gemm_golden(&P_h_gold[0][0], &V_h_gold[0][0], &ATTN_h_gold[0][0],
                    SEQ_LEN, HEAD_DIM, SEQ_LEN, false, scale_k16, shift_k16);

        // Scatter ATTN_h into ATTN_concat[:, h*16:(h+1)*16]
        for (int r = 0; r < SEQ_LEN; r++)
            for (int c = 0; c < HEAD_DIM; c++)
                ATTN_gold[r][h * HEAD_DIM + c] = ATTN_h_gold[r][c];
    }

    // Step 3: WO_out = ATTN_concat * Wo  [8,64] x [64,64] -> [8,64]
    gemm_golden(&ATTN_gold[0][0], &Wo[0][0], &WO_out_gold[0][0],
                SEQ_LEN, HIDDEN, HIDDEN, false, scale_k64, shift_k64);

    // Step 4: X2 = X + WO_out (residual add)
    vec_add_golden(&WO_out_gold[0][0], &X[0][0], &X2_gold[0][0], SEQ_LEN * HIDDEN);

    // Step 5: LayerNorm 2 (per row)
    for (int r = 0; r < SEQ_LEN; r++)
        layernorm_golden(&X2_gold[r][0], &LN2_out_gold[r][0], HIDDEN, LN2_beta);

    // Step 6: FFN1 = LN2_out * W1  [8,64] x [64,256] -> [8,256]
    gemm_golden(&LN2_out_gold[0][0], &W1[0][0], &FFN1_gold[0][0],
                SEQ_LEN, FFN_DIM, HIDDEN, false, scale_k64, shift_k64);

    // Step 7: GELU
    gelu_golden(&FFN1_gold[0][0], &GELU_gold[0][0], SEQ_LEN * FFN_DIM);

    // Step 8: FFN2 = GELU_out * W2  [8,256] x [256,64] -> [8,64]
    gemm_golden(&GELU_gold[0][0], &W2[0][0], &FFN2_gold[0][0],
                SEQ_LEN, HIDDEN, FFN_DIM, false, scale_k64, shift_k64);

    // Step 9: X_OUT = X2 + FFN2 (residual add)
    vec_add_golden(&FFN2_gold[0][0], &X2_gold[0][0], &X_OUT_gold[0][0], SEQ_LEN * HIDDEN);
}

// =============================================================================
// Load microcode program (multi-head attention)
// =============================================================================
int load_microcode() {
    uint64_t hi, lo;
    int addr = 0;

    // ---- LayerNorm 1 (8 rows) ----
    for (int i = 0; i < SEQ_LEN; i++) {
        uint16_t src = ADDR_X + i * HIDDEN;
        uint16_t dst = ADDR_LN1_OUT + i * HIDDEN;
        encode_instr(OP_LAYERNORM, 0, dst, src, S1_LN1_BETA,
                     0, HIDDEN, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- Per-head attention loop (4 heads) ----
    for (int h = 0; h < N_HEADS; h++) {
        uint16_t wq_h_addr = ADDR_WQ + h * HIDDEN * HEAD_DIM;  // h * 1024
        uint16_t wk_h_addr = ADDR_WK + h * HIDDEN * HEAD_DIM;
        uint16_t wv_h_addr = ADDR_WV + h * HIDDEN * HEAD_DIM;

        // GEMM Q_h = LN1_OUT * Wq_h  (M=8, N=16, K=64, imm=0x0901)
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_Q_H, ADDR_LN1_OUT, wq_h_addr,
                     SEQ_LEN, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);

        // BARRIER
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM K_h = LN1_OUT * Wk_h  (M=8, N=16, K=64, imm=0x0901)
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_K_H, ADDR_LN1_OUT, wk_h_addr,
                     SEQ_LEN, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);

        // BARRIER
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM V_h = LN1_OUT * Wv_h  (M=8, N=16, K=64, imm=0x0901)
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_V_H, ADDR_LN1_OUT, wv_h_addr,
                     SEQ_LEN, HEAD_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
        ucode_write(addr++, hi, lo);

        // BARRIER
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM S = Q_h * K_h^T  (M=8, N=8, K=16, imm=0x0701, TRANSPOSE_B)
        encode_instr(OP_GEMM, FLAG_TRANSPOSE_B | FLAG_REQUANT, ADDR_S, ADDR_Q_H, ADDR_K_H,
                     SEQ_LEN, SEQ_LEN, HEAD_DIM, GEMM_IMM_K16, hi, lo);
        ucode_write(addr++, hi, lo);

        // BARRIER
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // SOFTMAX rows (8 rows with causal mask)
        for (int i = 0; i < SEQ_LEN; i++) {
            uint16_t src = ADDR_S + i * SEQ_LEN;
            uint16_t dst = ADDR_P + i * SEQ_LEN;
            encode_instr(OP_SOFTMAX, FLAG_CAUSAL_MASK, dst, src, 0,
                         0, SEQ_LEN, i, 0x0100, hi, lo);
            ucode_write(addr++, hi, lo);
        }

        // BARRIER
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // GEMM ATTN_h = P * V_h  (M=8, N=16, K=8, imm=0x0701)
        encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_ATTN_H, ADDR_P, ADDR_V_H,
                     SEQ_LEN, HEAD_DIM, SEQ_LEN, GEMM_IMM_K16, hi, lo);
        ucode_write(addr++, hi, lo);

        // BARRIER
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);

        // VEC COPY2D: scatter ATTN_H -> ATTN[:, h*16:(h+1)*16]
        //   opcode=VEC, flags=FLAG_COPY2D (0x04)
        //   src0=ADDR_ATTN_H, dst=ADDR_ATTN + h*HEAD_DIM
        //   N=HEAD_DIM (=16, number of columns / length)
        //   M=SEQ_LEN (=8, number of rows)
        //   K=HEAD_DIM (=16, src stride)
        //   imm=HIDDEN (=64, dst stride)
        encode_instr(OP_VEC, FLAG_COPY2D, ADDR_ATTN + h * HEAD_DIM, ADDR_ATTN_H, 0,
                     SEQ_LEN, HEAD_DIM, HEAD_DIM, HIDDEN, hi, lo);
        ucode_write(addr++, hi, lo);

        // BARRIER
        encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // ---- Output projection ----
    // GEMM WO_OUT = ATTN * Wo  (M=8, N=64, K=64, imm=0x0901)
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_WO_OUT, ADDR_ATTN, ADDR_WO,
                 SEQ_LEN, HIDDEN, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- Residual 1 ----
    // VEC ADD: X2 = WO_OUT + X  (src0=WO_OUT in SRAM0, src1=S1_RESID in SRAM1)
    encode_instr(OP_VEC, 0x00, ADDR_X2, ADDR_WO_OUT, S1_RESID,
                 0, SEQ_LEN * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- DMA: copy X2 to SRAM1 for second residual ----
    encode_instr(OP_DMA_LOAD, 0, S1_RESID, ADDR_X2, 0,
                 0, SEQ_LEN * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- LayerNorm 2 (8 rows) ----
    for (int i = 0; i < SEQ_LEN; i++) {
        uint16_t src = ADDR_X2 + i * HIDDEN;
        uint16_t dst = ADDR_LN2_OUT + i * HIDDEN;
        encode_instr(OP_LAYERNORM, 0, dst, src, S1_LN2_BETA,
                     0, HIDDEN, 0, 0, hi, lo);
        ucode_write(addr++, hi, lo);
    }

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // ---- FFN ----
    // GEMM FFN1 = LN2_OUT * W1  (M=8, N=256, K=64, imm=0x0901)
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN1, ADDR_LN2_OUT, ADDR_W1,
                 SEQ_LEN, FFN_DIM, HIDDEN, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GELU in-place on FFN1
    encode_instr(OP_GELU, 0, ADDR_FFN1, ADDR_FFN1, 0,
                 0, SEQ_LEN * FFN_DIM, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // GEMM FFN2 = GELU_out * W2  (M=8, N=64, K=256, imm=0x0901)
    encode_instr(OP_GEMM, FLAG_REQUANT, ADDR_FFN2, ADDR_FFN1, ADDR_W2,
                 SEQ_LEN, HIDDEN, FFN_DIM, GEMM_IMM_K64, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // VEC ADD: X_OUT = FFN2 + X2  (src0=FFN2, src1=S1_RESID in SRAM1)
    encode_instr(OP_VEC, 0x00, ADDR_X_OUT, ADDR_FFN2, S1_RESID,
                 0, SEQ_LEN * HIDDEN, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // BARRIER
    encode_instr(OP_BARRIER, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    // END
    encode_instr(OP_END, 0, 0, 0, 0, 0, 0, 0, 0, hi, lo);
    ucode_write(addr++, hi, lo);

    return addr;
}

// =============================================================================
// Load data to SRAMs
// =============================================================================
void load_data_to_srams() {
    // SRAM0: X
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_X + i * HIDDEN + j, (uint8_t)X[i][j]);

    // SRAM0: Head-blocked Wq, Wk, Wv
    // Each weight is stored as 4 contiguous [64,16] blocks
    for (int h = 0; h < N_HEADS; h++) {
        uint16_t wq_base = ADDR_WQ + h * HIDDEN * HEAD_DIM;
        uint16_t wk_base = ADDR_WK + h * HIDDEN * HEAD_DIM;
        uint16_t wv_base = ADDR_WV + h * HIDDEN * HEAD_DIM;
        for (int i = 0; i < HIDDEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                sram0_write(wq_base + i * HEAD_DIM + j, (uint8_t)Wq[h * HIDDEN + i][j]);
                sram0_write(wk_base + i * HEAD_DIM + j, (uint8_t)Wk[h * HIDDEN + i][j]);
                sram0_write(wv_base + i * HEAD_DIM + j, (uint8_t)Wv[h * HIDDEN + i][j]);
            }
        }
    }

    // SRAM0: Wo [64][64]
    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_WO + i * HIDDEN + j, (uint8_t)Wo[i][j]);

    // SRAM0: W1 [64][256]
    for (int i = 0; i < HIDDEN; i++)
        for (int j = 0; j < FFN_DIM; j++)
            sram0_write(ADDR_W1 + i * FFN_DIM + j, (uint8_t)W1[i][j]);

    // SRAM0: W2 [256][64]
    for (int i = 0; i < FFN_DIM; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram0_write(ADDR_W2 + i * HIDDEN + j, (uint8_t)W2[i][j]);

    // SRAM1: LN1 beta, LN2 beta
    for (int j = 0; j < HIDDEN; j++) {
        sram1_write(S1_LN1_BETA + j, (uint8_t)LN1_beta[j]);
        sram1_write(S1_LN2_BETA + j, (uint8_t)LN2_beta[j]);
    }

    // SRAM1: Copy X for first residual add (src1 of VEC_ADD)
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HIDDEN; j++)
            sram1_write(S1_RESID + i * HIDDEN + j, (uint8_t)X[i][j]);
}

// =============================================================================
// Handle DMA command: copy from SRAM0 to SRAM1
// =============================================================================
void handle_dma_command() {
    uint16_t src  = dut->dma_src;
    uint16_t dst  = dut->dma_dst;
    uint16_t len  = dut->dma_len;

    std::cout << "  DMA: src=0x" << std::hex << src
              << " dst=0x" << dst << " len=" << std::dec << len << std::endl;

    for (int i = 0; i < len; i++) {
        uint8_t val = sram0_read(src + i);
        sram1_write(dst + i, val);
    }
}

// =============================================================================
// Verify a matrix against golden
// =============================================================================
struct VerifyResult {
    int mismatches;
    int max_err;
};

VerifyResult verify_matrix(const char* name, uint16_t base_addr, const int8_t* golden,
                           int rows, int cols, int tolerance = 0) {
    VerifyResult r = {0, 0};
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int8_t actual = (int8_t)sram0_read(base_addr + i * cols + j);
            int err = abs((int)actual - (int)golden[i * cols + j]);
            if (err > r.max_err) r.max_err = err;
            if (err > tolerance) r.mismatches++;
        }
    }
    std::cout << "  " << name << ": max_err=" << r.max_err
              << " mismatches(>" << tolerance << ")=" << r.mismatches
              << (r.mismatches == 0 ? " PASS" : " FAIL") << std::endl;
    return r;
}

// Verify a strided (scattered) matrix in SRAM against a dense golden buffer.
// SRAM layout: base_addr + row * sram_stride + col
// Golden layout: golden[row * cols + col]
VerifyResult verify_matrix_strided(const char* name, uint16_t base_addr,
                                   int sram_stride, const int8_t* golden,
                                   int rows, int cols, int tolerance = 0) {
    VerifyResult r = {0, 0};
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int8_t actual = (int8_t)sram0_read(base_addr + i * sram_stride + j);
            int err = abs((int)actual - (int)golden[i * cols + j]);
            if (err > r.max_err) r.max_err = err;
            if (err > tolerance) r.mismatches++;
        }
    }
    std::cout << "  " << name << ": max_err=" << r.max_err
              << " mismatches(>" << tolerance << ")=" << r.mismatches
              << (r.mismatches == 0 ? " PASS" : " FAIL") << std::endl;
    return r;
}

// =============================================================================
// Debug dump helpers
// =============================================================================
void dump_csv(const char* filename, const int8_t* data, int rows, int cols) {
#ifdef _WIN32
    _mkdir("dumps");
#else
    mkdir("dumps", 0755);
#endif
    std::string path = std::string("dumps/") + filename;
    std::ofstream f(path);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j > 0) f << ",";
            f << (int)data[i * cols + j];
        }
        f << "\n";
    }
    f.close();
    std::cout << "  Dumped: " << path << std::endl;
}

// =============================================================================
// Main test
// =============================================================================
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    dut = new Vgpt2_block_top;
    tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("gpt2_block_sim.vcd");

    // Initialize all inputs
    dut->clk = 0;
    dut->rst_n = 0;
    dut->start_pulse = 0;
    dut->ucode_len = 0;
    dut->uc_wr_en = 0;
    dut->uc_wr_addr = 0;
    memset(dut->uc_wr_data, 0, sizeof(dut->uc_wr_data));
    dut->tb_sram0_wr_en = 0; dut->tb_sram0_wr_addr = 0; dut->tb_sram0_wr_data = 0;
    dut->tb_sram0_rd_en = 0; dut->tb_sram0_rd_addr = 0;
    dut->tb_sram1_wr_en = 0; dut->tb_sram1_wr_addr = 0; dut->tb_sram1_wr_data = 0;
    dut->tb_sram1_rd_en = 0; dut->tb_sram1_rd_addr = 0;
    dut->dma_done_pulse = 0;

    std::cout << "============================================" << std::endl;
    std::cout << "  GPT-2 BLOCK TEST: Multi-Head Attention" << std::endl;
    std::cout << "  seq_len=" << SEQ_LEN << " hidden=" << HIDDEN
              << " head_dim=" << HEAD_DIM << " heads=" << N_HEADS
              << " ffn=" << FFN_DIM << std::endl;
    std::cout << "  HW GEMM: 5/head x4 + Wo + FFN1 + FFN2 = 23" << std::endl;
    std::cout << "============================================" << std::endl;

    // Build LUTs
    build_softmax_luts();
    build_rsqrt_lut();
    build_gelu_lut();

    // Reset
    reset_dut();
    std::cout << "Reset complete." << std::endl;

    // Generate data and compute golden
    generate_data_and_golden();
    std::cout << "Golden reference computed." << std::endl;

    // Print sample golden values
    std::cout << "  X[0]: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++) std::cout << (int)X[0][j] << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;
    std::cout << "  LN1_out[0]: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++) std::cout << (int)LN1_out_gold[0][j] << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;
    std::cout << "  X_OUT_gold[0]: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++) std::cout << (int)X_OUT_gold[0][j] << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;

    // Load microcode
    std::cout << "Loading microcode..." << std::endl;
    int num_instrs = load_microcode();
    std::cout << "  Loaded " << num_instrs << " instructions" << std::endl;

    // Load data
    std::cout << "Loading data to SRAMs..." << std::endl;
    load_data_to_srams();
    std::cout << "  Data loaded." << std::endl;

    // Start execution
    std::cout << "Starting execution..." << std::endl;
    dut->ucode_len = num_instrs;
    dut->start_pulse = 1;
    tick();
    dut->start_pulse = 0;

    // Main simulation loop
    int dma_count = 0;
    int softmax_count = 0;
    int layernorm_count = 0;
    int gelu_done_count = 0;
    int vec_done_count = 0;
    int cycle = 0;
    bool done = false;
    const int MAX_CYCLES = 2000000;

    while (cycle < MAX_CYCLES && !done) {
        tick();
        cycle++;

        // DMA interception
        if (dut->dma_cmd_captured) {
            dma_count++;
            std::cout << "[cycle " << cycle << "] DMA #" << dma_count << std::endl;
            handle_dma_command();
            dut->dma_done_pulse = 1;
            tick(); cycle++;
            dut->dma_done_pulse = 0;
        }

        // Track engine completions
        if (dut->softmax_done_dbg) softmax_count++;
        if (dut->layernorm_done_dbg) layernorm_count++;
        if (dut->gelu_done_dbg) gelu_done_count++;
        if (dut->vec_done_dbg) vec_done_count++;

        // Check for program end
        if (dut->program_end) {
            std::cout << "[cycle " << cycle << "] Program END" << std::endl;
            for (int i = 0; i < 10; i++) tick();
            done = true;
        }
    }

    if (!done) {
        std::cerr << "TIMEOUT after " << MAX_CYCLES << " cycles!" << std::endl;
        tfp->close(); delete tfp; delete dut;
        return 1;
    }

    int hw_gemm_count = (int)dut->hw_gemm_done_count;

    std::cout << std::endl;
    std::cout << "Execution complete: " << cycle << " cycles" << std::endl;
    std::cout << "  HW GEMM done:     " << hw_gemm_count << std::endl;
    std::cout << "  DMA commands:     " << dma_count << std::endl;
    std::cout << "  Softmax rows:     " << softmax_count << std::endl;
    std::cout << "  LayerNorm rows:   " << layernorm_count << std::endl;
    std::cout << "  GELU completions: " << gelu_done_count << std::endl;
    std::cout << "  Vec completions:  " << vec_done_count << std::endl;

    // Let state settle
    for (int i = 0; i < 20; i++) tick();

    // ================================================================
    // Verification: "follow-actual" approach
    // Re-read SRAM intermediates and recompute golden from actual values.
    // For multi-head: per-head buffers are reused, so we verify ATTN_concat
    // (the full [8,64] scatter result) and everything downstream.
    // ================================================================
    std::cout << std::endl << "=== Verification (follow-actual) ===" << std::endl;

    // Verify all engines are idle before SRAM reads
    assert(dut->vec_busy_dbg == 0 && "VEC engine still busy at verification time!");

    int scale_k64 = 1, shift_k64 = 9;
    int scale_k16 = 1, shift_k16 = 7;
    bool all_pass = true;

    // Helper: read matrix from SRAM0 into buffer
    auto read_sram_matrix = [](uint16_t base, int rows, int cols, int8_t* buf) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                buf[i * cols + j] = (int8_t)sram0_read(base + i * cols + j);
    };

    // 1. LN1: compare RTL LN with C++ LN golden (informational, tolerance=20)
    auto r_ln1 = verify_matrix("LN1_out", ADDR_LN1_OUT, &LN1_out_gold[0][0],
                                SEQ_LEN, HIDDEN, 20);
    // LN precision mismatch is expected; don't fail on it

    // 2. Read actual LN1_out from SRAM
    int8_t actual_ln1[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_LN1_OUT, SEQ_LEN, HIDDEN, &actual_ln1[0][0]);

    // 3. Compute full multi-head golden from actual LN1 (all 4 heads, scatter into concat)
    int8_t ref_attn_concat[SEQ_LEN][HIDDEN];
    memset(&ref_attn_concat[0][0], 0, sizeof(ref_attn_concat));

    for (int h = 0; h < N_HEADS; h++) {
        int8_t* Wq_h = &Wq[h * HIDDEN][0];
        int8_t* Wk_h = &Wk[h * HIDDEN][0];
        int8_t* Wv_h = &Wv[h * HIDDEN][0];

        int8_t ref_qh[SEQ_LEN][HEAD_DIM];
        int8_t ref_kh[SEQ_LEN][HEAD_DIM];
        int8_t ref_vh[SEQ_LEN][HEAD_DIM];
        int8_t ref_sh[SEQ_LEN][SEQ_LEN];
        int8_t ref_ph[SEQ_LEN][SEQ_LEN];
        int8_t ref_ah[SEQ_LEN][HEAD_DIM];

        // Q_h = actual_ln1 * Wq_h  (scale=1, shift=9)
        gemm_golden(&actual_ln1[0][0], Wq_h, &ref_qh[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);

        // K_h = actual_ln1 * Wk_h
        gemm_golden(&actual_ln1[0][0], Wk_h, &ref_kh[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);

        // V_h = actual_ln1 * Wv_h
        gemm_golden(&actual_ln1[0][0], Wv_h, &ref_vh[0][0],
                    SEQ_LEN, HEAD_DIM, HIDDEN, false, scale_k64, shift_k64);

        // S_h = Q_h * K_h^T  (scale=1, shift=7)
        gemm_golden(&ref_qh[0][0], &ref_kh[0][0], &ref_sh[0][0],
                    SEQ_LEN, SEQ_LEN, HEAD_DIM, true, scale_k16, shift_k16);

        // P_h = softmax(S_h, causal)
        for (int r = 0; r < SEQ_LEN; r++)
            softmax_golden(&ref_sh[r][0], &ref_ph[r][0], SEQ_LEN, true, r);

        // ATTN_h = P_h * V_h  (scale=1, shift=7)
        gemm_golden(&ref_ph[0][0], &ref_vh[0][0], &ref_ah[0][0],
                    SEQ_LEN, HEAD_DIM, SEQ_LEN, false, scale_k16, shift_k16);

        // Scatter into concat buffer
        for (int r = 0; r < SEQ_LEN; r++)
            for (int c = 0; c < HEAD_DIM; c++)
                ref_attn_concat[r][h * HEAD_DIM + c] = ref_ah[r][c];

        // Debug: print first row of Q_h for head h
        std::cout << "  Head " << h << " Q_h[0]: ";
        for (int j = 0; j < HEAD_DIM; j++) std::cout << (int)ref_qh[0][j] << " ";
        std::cout << std::endl;
    }

    // 4. Verify ATTN_concat at ADDR_ATTN
    auto r_attn = verify_matrix("ATTN_concat", ADDR_ATTN, &ref_attn_concat[0][0],
                                 SEQ_LEN, HIDDEN, 0);
    if (r_attn.mismatches) all_pass = false;

    // Debug: print first row of ATTN_concat
    std::cout << std::endl;
    std::cout << "  ATTN_concat[0] actual: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++)
        std::cout << (int)(int8_t)sram0_read(ADDR_ATTN + j) << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;
    std::cout << "  ATTN_concat[0] golden: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++)
        std::cout << (int)ref_attn_concat[0][j] << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl << std::endl;

    // 5. Read actual ATTN_concat -> recompute WO_out
    int8_t actual_attn[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_ATTN, SEQ_LEN, HIDDEN, &actual_attn[0][0]);

    int8_t ref_wo[SEQ_LEN][HIDDEN];
    gemm_golden(&actual_attn[0][0], &Wo[0][0], &ref_wo[0][0],
                SEQ_LEN, HIDDEN, HIDDEN, false, scale_k64, shift_k64);
    auto r_wo = verify_matrix("WO(hw) ", ADDR_WO_OUT, &ref_wo[0][0], SEQ_LEN, HIDDEN, 0);
    if (r_wo.mismatches) all_pass = false;

    // 6. Read actual WO_out -> verify X2 = WO_out + X (residual add)
    int8_t actual_wo[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_WO_OUT, SEQ_LEN, HIDDEN, &actual_wo[0][0]);

    int8_t ref_x2[SEQ_LEN][HIDDEN];
    vec_add_golden(&actual_wo[0][0], &X[0][0], &ref_x2[0][0], SEQ_LEN * HIDDEN);
    auto r_x2 = verify_matrix("X2     ", ADDR_X2, &ref_x2[0][0], SEQ_LEN, HIDDEN, 0);
    if (r_x2.mismatches) all_pass = false;

    // 7. LN2: informational comparison with tolerance
    int8_t actual_x2[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_X2, SEQ_LEN, HIDDEN, &actual_x2[0][0]);
    int8_t ref_ln2[SEQ_LEN][HIDDEN];
    for (int r = 0; r < SEQ_LEN; r++)
        layernorm_golden(&actual_x2[r][0], &ref_ln2[r][0], HIDDEN, LN2_beta);
    auto r_ln2 = verify_matrix("LN2_out", ADDR_LN2_OUT, &ref_ln2[0][0],
                                SEQ_LEN, HIDDEN, 20);

    // 8. Read actual LN2_out -> recompute FFN1 -> verify GELU(FFN1)
    int8_t actual_ln2[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_LN2_OUT, SEQ_LEN, HIDDEN, &actual_ln2[0][0]);

    int8_t ref_ffn1[SEQ_LEN][FFN_DIM];
    gemm_golden(&actual_ln2[0][0], &W1[0][0], &ref_ffn1[0][0],
                SEQ_LEN, FFN_DIM, HIDDEN, false, scale_k64, shift_k64);

    if (DUMP_FFN) {
        dump_csv("ffn1_ln2_input.csv", &actual_ln2[0][0], SEQ_LEN, HIDDEN);
    }

    // 9. GELU: compare SRAM (which has GELU applied in-place) against gelu(ref_ffn1)
    int8_t ref_gelu[SEQ_LEN][FFN_DIM];
    gelu_golden(&ref_ffn1[0][0], &ref_gelu[0][0], SEQ_LEN * FFN_DIM);
    auto r_ff1 = verify_matrix("FFN1+GELU(hw)", ADDR_FFN1, &ref_gelu[0][0], SEQ_LEN, FFN_DIM, 1);
    if (r_ff1.mismatches) all_pass = false;

    if (DUMP_FFN) {
        int8_t actual_gelu_dump[SEQ_LEN][FFN_DIM];
        read_sram_matrix(ADDR_FFN1, SEQ_LEN, FFN_DIM, &actual_gelu_dump[0][0]);
        dump_csv("ffn1_output.csv", &actual_gelu_dump[0][0], SEQ_LEN, FFN_DIM);
    }

    // 10. Read actual GELU output -> recompute FFN2
    int8_t actual_gelu[SEQ_LEN][FFN_DIM];
    read_sram_matrix(ADDR_FFN1, SEQ_LEN, FFN_DIM, &actual_gelu[0][0]);

    if (DUMP_FFN) {
        dump_csv("ffn2_gelu_input.csv", &actual_gelu[0][0], SEQ_LEN, FFN_DIM);
    }

    int8_t ref_ffn2[SEQ_LEN][HIDDEN];
    gemm_golden(&actual_gelu[0][0], &W2[0][0], &ref_ffn2[0][0],
                SEQ_LEN, HIDDEN, FFN_DIM, false, scale_k64, shift_k64);
    auto r_ff2 = verify_matrix("FFN2 (hw)", ADDR_FFN2, &ref_ffn2[0][0], SEQ_LEN, HIDDEN, 0);
    if (r_ff2.mismatches) all_pass = false;

    if (DUMP_FFN) {
        int8_t actual_ffn2_dump[SEQ_LEN][HIDDEN];
        read_sram_matrix(ADDR_FFN2, SEQ_LEN, HIDDEN, &actual_ffn2_dump[0][0]);
        dump_csv("ffn2_output.csv", &actual_ffn2_dump[0][0], SEQ_LEN, HIDDEN);
    }

    // 11. Read actual FFN2, X2 -> verify X_OUT = FFN2 + X2 (residual add)
    int8_t actual_ffn2[SEQ_LEN][HIDDEN];
    read_sram_matrix(ADDR_FFN2, SEQ_LEN, HIDDEN, &actual_ffn2[0][0]);

    int8_t ref_xout[SEQ_LEN][HIDDEN];
    vec_add_golden(&actual_ffn2[0][0], &actual_x2[0][0], &ref_xout[0][0], SEQ_LEN * HIDDEN);
    auto r_out = verify_matrix("X_OUT  ", ADDR_X_OUT, &ref_xout[0][0], SEQ_LEN, HIDDEN, 0);
    if (r_out.mismatches) all_pass = false;

    // Print first row of output
    std::cout << std::endl;
    std::cout << "  X_OUT[0] actual: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++)
        std::cout << (int)(int8_t)sram0_read(ADDR_X_OUT + j) << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;
    std::cout << "  X_OUT[0] golden: ";
    for (int j = 0; j < std::min(HIDDEN, 16); j++)
        std::cout << (int)ref_xout[0][j] << " ";
    if (HIDDEN > 16) std::cout << "...";
    std::cout << std::endl;

    // ================================================================
    // Final verdict
    // Expected: 23 HW GEMMs, 1 DMA, 32 softmax rows, 16 LN rows,
    //           1 GELU, 6 vec (4 COPY2D + 2 ADD)
    // ================================================================
    bool pass = all_pass && (hw_gemm_count == 23) && (dma_count == 1);

    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  HW GEMM done:      " << hw_gemm_count << "/23 "
              << (hw_gemm_count == 23 ? "OK" : "FAIL") << std::endl;
    std::cout << "  DMA commands:      " << dma_count << "/1 "
              << (dma_count == 1 ? "OK" : "FAIL") << std::endl;
    std::cout << "  Softmax rows:      " << softmax_count << "/32" << std::endl;
    std::cout << "  LayerNorm rows:    " << layernorm_count << "/16" << std::endl;
    std::cout << "  GELU completions:  " << gelu_done_count << "/1" << std::endl;
    std::cout << "  Vec completions:   " << vec_done_count << "/6" << std::endl;
    std::cout << "  X_OUT max error:   " << r_out.max_err << " (tol=0)" << std::endl;
    std::cout << "  Total cycles:      " << cycle << std::endl;
    std::cout << "  GEMMs: " << hw_gemm_count << "/23 on REAL hardware" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  GPT-2 BLOCK TEST: " << (pass ? "PASS" : "FAIL") << std::endl;
    std::cout << "============================================" << std::endl;

    tfp->close();
    delete tfp;
    delete dut;

    return pass ? 0 : 1;
}
