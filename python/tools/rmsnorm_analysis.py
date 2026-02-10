#!/usr/bin/env python3
"""Analyze RMSNorm precision: current two-stage vs fused approach."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from golden.layernorm_ref import RSQRT_LUT
from golden.rmsnorm_ref import rmsnorm_fixed
from golden.gemm_ref import gemm_int8

def fused_rmsnorm(x, gamma):
    """Proposed: multiply x*gamma FIRST (int16), then normalize with single >>16."""
    x = x.astype(np.int32)
    N = len(x)
    sum_sq = np.int64(0)
    for i in range(N):
        sum_sq += np.int64(x[i]) * np.int64(x[i])
    rms_val_q8 = int((sum_sq << 8) // N)
    rms_top8 = min((rms_val_q8 >> 8) & 0xFF, 255) if rms_val_q8 > 0 else 0
    inv_rms = int(RSQRT_LUT[rms_top8])

    result = np.zeros(N, dtype=np.int8)
    for i in range(N):
        if gamma is not None:
            xg = np.int64(x[i]) * np.int64(gamma[i])  # int16, no shift
            scaled = (xg * np.int64(inv_rms)) >> 16     # single >>16
        else:
            scaled = (np.int64(x[i]) * np.int64(inv_rms)) >> 16
        result[i] = np.clip(scaled, -128, 127).astype(np.int8)
    return result


# Generate test data matching random weight generator (seed=42)
rng = np.random.RandomState(42)
wte = rng.randint(-32, 33, (256, 64)).astype(np.int8)

# First block weights (consumed in same order as llama_gen_weights.py)
gamma1 = rng.randint(-64, 64, 64).astype(np.int8)
Wq_heads = [rng.randint(-4, 5, (64, 16)).astype(np.int8) for _ in range(4)]

x = wte[1]  # embedding for token 1

print("=== Element-level comparison (token 1 embedding) ===")
print("inv_rms/LUT computation:")
x32 = x.astype(np.int32)
sum_sq = sum(np.int64(x32[i])**2 for i in range(64))
rms_val_q8 = int((sum_sq << 8) // 64)
rms_top8 = min((rms_val_q8 >> 8) & 0xFF, 255)
inv = int(RSQRT_LUT[rms_top8])
print("  sum_sq=%d  rms_val_q8=%d  rms_top8=%d  inv_rms=%d" % (sum_sq, rms_val_q8, rms_top8, inv))
print()

# Per-element comparison (first 16 elements)
print("%3s %4s %4s | %7s %7s | %7s" % ("i", "x", "g", "old_s1", "old_out", "fused"))
print("-" * 50)
for i in range(16):
    xi, gi = int(x32[i]), int(gamma1[i])
    old_s1 = int((np.int64(xi) * np.int64(inv)) >> 16)
    old_out = int((old_s1 * np.int64(gi)) >> 7)
    fused_out = int((np.int64(xi) * np.int64(gi) * np.int64(inv)) >> 16)
    old_out = np.clip(old_out, -128, 127)
    fused_out = np.clip(fused_out, -128, 127)
    print("%3d %4d %4d | %7d %7d | %7d" % (i, xi, gi, old_s1, old_out, fused_out))

# Full vectors
old_out = rmsnorm_fixed(x, gamma1)
new_out = fused_rmsnorm(x, gamma1)
print()
print("OLD RMSNorm: nonzero=%d/64  absmax=%d  vals[:16]=%s" % (
    np.count_nonzero(old_out), np.abs(old_out).max(), old_out[:16].tolist()))
print("NEW RMSNorm: nonzero=%d/64  absmax=%d  vals[:16]=%s" % (
    np.count_nonzero(new_out), np.abs(new_out).max(), new_out[:16].tolist()))

# Downstream GEMM comparison (Q head 0, shift=9)
print()
print("=== Downstream Q-projection GEMM (head 0, K=64, shift=9) ===")
# Full sequence (4 tokens)
x_seq = wte[[1,2,3,4]]  # [4, 64]
old_rms = np.zeros_like(x_seq)
new_rms = np.zeros_like(x_seq)
for s in range(4):
    old_rms[s] = rmsnorm_fixed(x_seq[s], gamma1)
    new_rms[s] = fused_rmsnorm(x_seq[s], gamma1)

Wq0 = Wq_heads[0]
old_q = gemm_int8(old_rms, Wq0, scale=1, shift=9)
new_q = gemm_int8(new_rms, Wq0, scale=1, shift=9)
print("OLD Q[0]: nonzero=%d/%d  absmax=%d  row0=%s" % (
    np.count_nonzero(old_q), old_q.size, np.abs(old_q).max(), old_q[0].tolist()))
print("NEW Q[0]: nonzero=%d/%d  absmax=%d  row0=%s" % (
    np.count_nonzero(new_q), new_q.size, np.abs(new_q).max(), new_q[0].tolist()))

# Check if the fused approach would cause GEMM overflow
print()
print("=== Overflow check ===")
# Worst case: max RMSNorm output * max weight * K
max_rms = np.abs(new_rms).max()
max_dot = int(max_rms) * 4 * 64
print("Max fused RMSNorm output: %d" % max_rms)
print("Worst-case GEMM accumulator: %d * 4 * 64 = %d (int16 max=32767)" % (max_rms, max_dot))
print("After >>9: %d (fits int8: %s)" % (max_dot >> 9, "YES" if (max_dot >> 9) <= 127 else "NO"))
