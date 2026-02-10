"""Reference SiLU activation matching RTL silu_lut.sv."""
import numpy as np
from .quant import clamp_i8


def make_silu_lut(input_scale=32.0):
    """Generate SiLU LUT matching RTL silu_lut.sv.
    Maps int8 -> int8 with SiLU activation.
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    input_scale: maps int8 to float (x_float = x_int8 / input_scale)
    """
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        signed_i = np.array(i, dtype=np.uint8).view(np.int8)
        x = float(signed_i) / input_scale
        # SiLU(x) = x * sigmoid(x)
        sigmoid_x = 1.0 / (1.0 + np.exp(-x))
        silu_val = x * sigmoid_x
        result = silu_val * input_scale
        lut[i] = np.clip(int(round(result)), -128, 127)
    return lut


SILU_LUT = None  # Lazy init


def _get_silu_lut():
    global SILU_LUT
    if SILU_LUT is None:
        SILU_LUT = make_silu_lut()
    return SILU_LUT


def silu_fixed(x):
    """
    Fixed-point SiLU matching RTL.
    x: int8 array
    Returns: int8 array
    """
    lut = _get_silu_lut()
    x = x.astype(np.int8)
    result = np.zeros_like(x, dtype=np.int8)
    for idx in np.ndindex(x.shape):
        unsigned_idx = np.uint8(x[idx])
        result[idx] = lut[unsigned_idx]
    return result
