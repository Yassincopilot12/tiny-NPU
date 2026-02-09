"""Shared constants for NPU GPT-2 demo: model config, SRAM layout, quant params."""
import numpy as np

# ── Model configuration (gpt2-tiny: sliced from real GPT-2) ──────────────
HIDDEN      = 64
N_HEADS     = 4
HEAD_DIM    = 16     # = HIDDEN // N_HEADS
FFN_DIM     = 256
N_LAYERS    = 4
VOCAB_SIZE  = 256    # first 256 tokens from GPT-2 tokenizer
MAX_SEQ     = 16     # reduced from 32 to fit in 64KB SRAM0

# ── Quantization ─────────────────────────────────────────────────────────
QUANT_WEIGHT_MAX = 4     # map max |weight| to this int8 value
GEMM_SCALE       = 1
GEMM_SHIFT       = 9     # requant: (acc * 1 + 256) >> 9 for K=64
GEMM_IMM         = (GEMM_SHIFT << 8) | GEMM_SCALE   # 0x0901
GEMM_IMM_K64     = GEMM_IMM                           # K=64 projections
GEMM_SHIFT_K16   = 7     # requant for K=16 (score/context GEMMs)
GEMM_IMM_K16     = (GEMM_SHIFT_K16 << 8) | GEMM_SCALE  # 0x0701
GEMM_IMM_KS      = GEMM_IMM_K16                       # context P*V (K=S≤16)

# ── SRAM0 addresses (block execution, max seq_len=16) ────────────────────
# Weights (48KB total, head-blocked QKV layout):
ADDR_WQ       = 0x0000   # 4x[64,16] = 4096B  (head-blocked)
ADDR_WK       = 0x1000   # 4x[64,16] = 4096B  (head-blocked)
ADDR_WV       = 0x2000   # 4x[64,16] = 4096B  (head-blocked)
ADDR_WO       = 0x3000   # [64][64]  = 4096B
ADDR_W1       = 0x4000   # [64][256] = 16384B
ADDR_W2       = 0x8000   # [256][64] = 16384B

# Activations (multi-head):
ADDR_X        = 0xC000   # [S,64]   = 1024B
ADDR_LN1_OUT  = 0xC400   # [S,64]   = 1024B
ADDR_Q_H      = 0xC800   # [S,16]   = 256B   (per-head, reused)
ADDR_K_H      = 0xC900   # [S,16]   = 256B   (per-head, reused)
ADDR_V_H      = 0xCA00   # [S,16]   = 256B   (per-head, reused)
ADDR_S        = 0xCB00   # [S,S]    = 256B   (per-head, reused)
ADDR_P        = 0xCC00   # [S,S]    = 256B   (per-head, reused)
ADDR_ATTN_H   = 0xCD00   # [S,16]   = 256B   (per-head context temp)
ADDR_ATTN     = 0xCE00   # [S,64]   = 1024B  (concat destination)
ADDR_WO_OUT   = 0xD200   # [S,64]   = 1024B
ADDR_X2       = 0xD600   # [S,64]   = 1024B
ADDR_LN2_OUT  = 0xDA00   # [S,64]   = 1024B
ADDR_FFN1     = 0xDE00   # [S,256]  = 4096B
ADDR_FFN2     = 0xEE00   # [S,64]   = 1024B
ADDR_X_OUT    = 0xF200   # [S,64]   = 1024B

# Legacy aliases for backward compatibility in tests that use full-size Q/K/V
ADDR_Q = ADDR_Q_H
ADDR_K = ADDR_K_H
ADDR_V = ADDR_V_H

def wq_head_addr(h):
    """SRAM0 address for head h's WQ slice [64,16]."""
    return ADDR_WQ + h * HIDDEN * HEAD_DIM

def wk_head_addr(h):
    """SRAM0 address for head h's WK slice [64,16]."""
    return ADDR_WK + h * HIDDEN * HEAD_DIM

def wv_head_addr(h):
    """SRAM0 address for head h's WV slice [64,16]."""
    return ADDR_WV + h * HIDDEN * HEAD_DIM

# ── SRAM0 addresses for lm_head phase ────────────────────────────────────
ADDR_LM_INPUT  = 0x0000   # last-token hidden state [1][64] = 64B
ADDR_LM_WEIGHT = 0x0100   # lm_head weight [256][64] = 16384B
ADDR_LM_OUTPUT = 0x4100   # logits [1][256] = 256B

# ── SRAM1 addresses ──────────────────────────────────────────────────────
S1_LN1_BETA  = 0x0000   # [64]
S1_LN2_BETA  = 0x0040   # [64]
S1_LN_F_BETA = 0x0080   # [64]
S1_RESID     = 0x0100   # [16][64] = 1024B

# ── weights.bin layout (offsets in bytes) ─────────────────────────────────
WTE_OFFSET      = 0
WTE_SIZE        = VOCAB_SIZE * HIDDEN                        # 16384
WPE_OFFSET      = WTE_OFFSET + WTE_SIZE
WPE_SIZE        = MAX_SEQ * HIDDEN                           # 1024

BLOCK_SIZE      = (64 + 4096 + 4096 + 4096 + 4096 + 64 + 16384 + 16384)  # 49280
BLOCKS_OFFSET   = WPE_OFFSET + WPE_SIZE                      # 17408
# Within each block:
BLK_LN1_BETA  = 0         # 64B
BLK_WQ        = 64        # 4096B
BLK_WK        = 4160      # 4096B
BLK_WV        = 8256      # 4096B
BLK_WO        = 12352     # 4096B
BLK_LN2_BETA  = 16448     # 64B
BLK_W1        = 16512     # 16384B
BLK_W2        = 32896     # 16384B

LN_F_OFFSET     = BLOCKS_OFFSET + N_LAYERS * BLOCK_SIZE      # 214528
LN_F_SIZE       = 64
LM_HEAD_OFFSET  = LN_F_OFFSET + LN_F_SIZE                    # 214592
LM_HEAD_SIZE    = VOCAB_SIZE * HIDDEN                         # 16384
WEIGHTS_TOTAL   = LM_HEAD_OFFSET + LM_HEAD_SIZE              # 230976


def quantize_tensor(w_fp32, target_max=QUANT_WEIGHT_MAX):
    """Per-tensor symmetric int8 quantization.
    Maps max |w| to target_max (default 4) in int8."""
    w = np.asarray(w_fp32, dtype=np.float32)
    amax = np.max(np.abs(w))
    if amax < 1e-10:
        return np.zeros_like(w, dtype=np.int8)
    scale = target_max / amax
    return np.clip(np.round(w * scale), -127, 127).astype(np.int8)


def quantize_beta(beta_fp32):
    """Quantize LN beta to int8 (direct rounding, since beta adds directly)."""
    return np.clip(np.round(np.asarray(beta_fp32, dtype=np.float32)),
                   -128, 127).astype(np.int8)
