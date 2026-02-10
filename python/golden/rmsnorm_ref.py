"""Reference fixed-point RMSNorm matching RTL rmsnorm_engine.sv."""
import numpy as np
from .quant import clamp_i8
from .layernorm_ref import RSQRT_LUT


def rmsnorm_fixed(x, gamma=None):
    """
    Fixed-point RMSNorm matching RTL.

    x: int8 array [hidden_dim]
    gamma: int8 array [hidden_dim] (scale, Q1.7 format, optional)
    Returns: int8 array [hidden_dim]

    Formula: y[i] = x[i] * rsqrt(mean(x^2)) * gamma[i]
    """
    x = x.astype(np.int32)
    N = len(x)

    # Compute sum of squares (unsigned: x*x is always non-negative for signed)
    sum_sq = np.int64(0)
    for i in range(N):
        sum_sq += np.int64(x[i]) * np.int64(x[i])

    # Divide by N to get E[x^2], with <<8 alignment for Q8.8
    # Match RTL: restoring divider on (sum_sq << 8) / N
    rms_val_q8 = int((sum_sq << 8) // N)  # rough Q8.8

    # Extract top 8 bits for rsqrt LUT index (same as layernorm variance indexing)
    rms_top8 = min((rms_val_q8 >> 8) & 0xFF, 255) if rms_val_q8 > 0 else 0
    inv_rms = int(RSQRT_LUT[rms_top8])  # Q0.16

    # Apply: out[i] = (x[i] * gamma[i]) * inv_rms >> 16   (fused: multiply gamma first)
    result = np.zeros(N, dtype=np.int8)
    for i in range(N):
        if gamma is not None:
            xg = np.int64(x[i]) * np.int64(gamma[i])       # int8 * int8 = int16, no shift
            scaled = (xg * np.int64(inv_rms)) >> 16         # single >>16
        else:
            scaled = (np.int64(x[i]) * np.int64(inv_rms)) >> 16

        result[i] = np.clip(scaled, -128, 127).astype(np.int8)

    return result
