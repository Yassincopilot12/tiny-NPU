"""Shared constants for NPU LLaMA demo: model config, SRAM layout, quant params."""
import numpy as np

# ── Model configuration (tiny LLaMA) ──────────────────────────────────
HIDDEN      = 64
N_Q_HEADS   = 4
N_KV_HEADS  = 2
HEAD_DIM    = 16     # = HIDDEN // N_Q_HEADS
GQA_RATIO   = N_Q_HEADS // N_KV_HEADS  # = 2
FFN_DIM     = 128
N_LAYERS    = 4
VOCAB_SIZE  = 256
MAX_SEQ     = 16

# ── Quantization ─────────────────────────────────────────────────────────
QUANT_WEIGHT_MAX = 4
GEMM_SCALE       = 1
GEMM_SHIFT       = 2
GEMM_IMM         = (GEMM_SHIFT << 8) | GEMM_SCALE   # 0x0201
GEMM_IMM_K64     = GEMM_IMM
GEMM_SHIFT_K16   = 7
GEMM_IMM_K16     = (GEMM_SHIFT_K16 << 8) | GEMM_SCALE  # 0x0701
GEMM_IMM_KS      = GEMM_IMM_K16
GEMM_SHIFT_K128  = 2
GEMM_IMM_K128    = (GEMM_SHIFT_K128 << 8) | GEMM_SCALE  # 0x0201

# ── SRAM0 addresses (LLaMA block) ─────────────────────────────────────
# Weights (36KB):
ADDR_WQ       = 0x0000   # 4x[64,16] = 4096B  (4 Q-heads, head-blocked)
ADDR_WK       = 0x1000   # 2x[64,16] = 2048B  (2 KV-heads, GQA)
ADDR_WV       = 0x1800   # 2x[64,16] = 2048B  (2 KV-heads, GQA)
ADDR_WO       = 0x2000   # [64,64]   = 4096B
ADDR_W_GATE   = 0x3000   # [64,128]  = 8192B  (SwiGLU gate)
ADDR_W_UP     = 0x5000   # [64,128]  = 8192B  (SwiGLU up)
ADDR_W_DOWN   = 0x7000   # [128,64]  = 8192B  (SwiGLU down)

# Activations (S=16):
ADDR_X        = 0xA000   # [S,64]   = 1024B
ADDR_RMS1_OUT = 0xA400   # [S,64]   = 1024B
ADDR_Q_H      = 0xA800   # [S,16]   =  256B   (reused per head)
ADDR_K_H      = 0xA900   # [S,16]   =  256B   (reused per head)
ADDR_V_H      = 0xAA00   # [S,16]   =  256B   (reused per head)
ADDR_S_MAT    = 0xAB00   # [S,S]    =  256B   (reused per head)
ADDR_P_MAT    = 0xAC00   # [S,S]    =  256B   (reused per head)
ADDR_ATTN_H   = 0xAD00   # [S,16]   =  256B   (per-head temp)
ADDR_ATTN     = 0xAE00   # [S,64]   = 1024B   (concat destination)
ADDR_WO_OUT   = 0xB200   # [S,64]   = 1024B
ADDR_X2       = 0xB600   # [S,64]   = 1024B   (residual 1)
ADDR_RMS2_OUT = 0xBA00   # [S,64]   = 1024B
ADDR_FFN_GATE = 0xBE00   # [S,128]  = 2048B
ADDR_FFN_UP   = 0xC600   # [S,128]  = 2048B
ADDR_FFN_DOWN = 0xCE00   # [S,64]   = 1024B
ADDR_X_OUT    = 0xD200   # [S,64]   = 1024B

# ── SRAM1 addresses ──────────────────────────────────────────────────────
S1_RMS1_GAMMA = 0x0000   # [64]     =  64B   (RMSNorm scale)
S1_RMS2_GAMMA = 0x0040   # [64]     =  64B
S1_ROPE_SIN   = 0x0080   # [16,8]   = 128B   (int8, Q1.7)
S1_ROPE_COS   = 0x0100   # [16,8]   = 128B   (int8, Q1.7)
S1_RESID      = 0x0180   # [S,64]   = 1024B  (residual source)

# ── Per-head address helpers ─────────────────────────────────────────────
def wq_head_addr(h):
    """SRAM0 address for Q-head h's WQ slice [64,16]."""
    return ADDR_WQ + h * HIDDEN * HEAD_DIM

def wk_head_addr(kv_h):
    """SRAM0 address for KV-head kv_h's WK slice [64,16]."""
    return ADDR_WK + kv_h * HIDDEN * HEAD_DIM

def wv_head_addr(kv_h):
    """SRAM0 address for KV-head kv_h's WV slice [64,16]."""
    return ADDR_WV + kv_h * HIDDEN * HEAD_DIM

# ── weights.bin layout (offsets in bytes) ─────────────────────────────────
WTE_OFFSET      = 0
WTE_SIZE        = VOCAB_SIZE * HIDDEN                        # 16384

LLAMA_BLOCK_SIZE = (64 + 4096 + 2048 + 2048 + 4096 + 64 + 8192 + 8192 + 8192)  # 36992
BLOCKS_OFFSET    = WTE_OFFSET + WTE_SIZE                     # 16384
# Within each block:
BLK_RMS1_GAMMA = 0         # 64B
BLK_WQ         = 64        # 4096B (4x[64,16])
BLK_WK         = 4160      # 2048B (2x[64,16])
BLK_WV         = 6208      # 2048B (2x[64,16])
BLK_WO         = 8256      # 4096B
BLK_RMS2_GAMMA = 12352     # 64B
BLK_W_GATE     = 12416     # 8192B
BLK_W_UP       = 20608     # 8192B
BLK_W_DOWN     = 28800     # 8192B

LN_F_OFFSET     = BLOCKS_OFFSET + N_LAYERS * LLAMA_BLOCK_SIZE  # 164352
LN_F_SIZE       = 64
LM_HEAD_OFFSET  = LN_F_OFFSET + LN_F_SIZE                      # 164416
LM_HEAD_SIZE    = VOCAB_SIZE * HIDDEN                           # 16384
WEIGHTS_TOTAL   = LM_HEAD_OFFSET + LM_HEAD_SIZE                # 180800


def quantize_tensor(w_fp32, target_max=QUANT_WEIGHT_MAX):
    """Per-tensor symmetric int8 quantization."""
    w = np.asarray(w_fp32, dtype=np.float32)
    amax = np.max(np.abs(w))
    if amax < 1e-10:
        return np.zeros_like(w, dtype=np.int8)
    scale = target_max / amax
    return np.clip(np.round(w * scale), -127, 127).astype(np.int8)


def quantize_gamma(gamma_fp32):
    """Quantize RMSNorm gamma to int8 (Q1.7 scale)."""
    return np.clip(np.round(np.asarray(gamma_fp32, dtype=np.float32) * 127),
                   -128, 127).astype(np.int8)
