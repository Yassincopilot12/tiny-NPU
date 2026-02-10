"""Reference fixed-point RoPE matching RTL rope_engine.sv."""
import numpy as np
from .quant import clamp_i8


def make_rope_tables(max_seq, head_dim, base=10000.0):
    """
    Generate int8 sin/cos tables for RoPE.

    Returns:
        sin_table: int8 [max_seq, head_dim//2] in Q1.7 format
        cos_table: int8 [max_seq, head_dim//2] in Q1.7 format
    """
    half_dim = head_dim // 2
    sin_table = np.zeros((max_seq, half_dim), dtype=np.int8)
    cos_table = np.zeros((max_seq, half_dim), dtype=np.int8)

    for pos in range(max_seq):
        for i in range(half_dim):
            freq = 1.0 / (base ** (2.0 * i / head_dim))
            angle = pos * freq
            # Q1.7: multiply by 128, clamp to [-128, 127]
            sin_val = np.sin(angle) * 128.0
            cos_val = np.cos(angle) * 128.0
            sin_table[pos, i] = np.clip(int(round(sin_val)), -128, 127)
            cos_table[pos, i] = np.clip(int(round(cos_val)), -128, 127)

    return sin_table, cos_table


def rope_fixed(x, sin_table, cos_table, pos_offset=0):
    """
    Apply RoPE to Q or K vectors using fixed-point arithmetic.

    x: int8 [num_rows, head_dim]
    sin_table: int8 [max_seq, head_dim//2] (Q1.7)
    cos_table: int8 [max_seq, head_dim//2] (Q1.7)
    pos_offset: starting position index (0 for prefill)

    Returns: int8 [num_rows, head_dim]
    """
    num_rows, head_dim = x.shape
    half_dim = head_dim // 2
    result = np.zeros_like(x, dtype=np.int8)

    for row in range(num_rows):
        pos = row + pos_offset
        for i in range(half_dim):
            even = int(x[row, 2 * i])
            odd = int(x[row, 2 * i + 1])
            cos_val = int(cos_table[pos, i])
            sin_val = int(sin_table[pos, i])

            # Fixed-point rotation: (a*cos - b*sin + 64) >> 7
            rot_even = (even * cos_val - odd * sin_val + 64) >> 7
            rot_odd = (even * sin_val + odd * cos_val + 64) >> 7

            result[row, 2 * i] = np.clip(rot_even, -128, 127)
            result[row, 2 * i + 1] = np.clip(rot_odd, -128, 127)

    return result.astype(np.int8)
