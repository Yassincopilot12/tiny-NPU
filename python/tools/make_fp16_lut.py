#!/usr/bin/env python3
"""Generate FP16 lookup tables for math operations.
Each table: 256 entries, indexed by upper 8 bits of FP16 input, output is 16-bit FP16.
"""
import numpy as np
import struct

def float_to_fp16_bits(f):
    """Convert float to IEEE 754 FP16 bit pattern (uint16)."""
    return int(np.float16(f).view(np.uint16))

def fp16_bits_to_float(bits):
    """Convert FP16 bit pattern to float."""
    return float(np.array(bits, dtype=np.uint16).view(np.float16))

def gen_lut(func):
    """Generate 256-entry LUT."""
    entries = []
    for idx in range(256):
        # Reconstruct FP16 value from upper 8 bits (lower 8 bits = 0)
        fp16_bits = idx << 8
        try:
            x = fp16_bits_to_float(fp16_bits)
            if np.isnan(x) or np.isinf(x):
                result = float_to_fp16_bits(0.0)
            else:
                y = func(float(x))
                if np.isnan(y) or np.isinf(y):
                    result = float_to_fp16_bits(0.0)
                else:
                    result = float_to_fp16_bits(y)
            entries.append(result)
        except:
            entries.append(0)
    return entries

def safe_exp(x):
    x = float(x)
    if x > 11.0: return 65504.0  # FP16 max
    if x < -17.0: return 0.0
    return np.exp(x)

def safe_log(x):
    if x <= 0: return float(np.float16(-65504.0))
    return np.log(float(x))

def safe_sqrt(x):
    if x < 0: return 0.0
    return np.sqrt(float(x))

def safe_rsqrt(x):
    if x <= 0: return 0.0
    return 1.0 / np.sqrt(float(x))

# Print LUT as SystemVerilog initialization
def print_sv_lut(name, entries):
    print(f"// {name} FP16 LUT - 256 entries")
    for i, val in enumerate(entries):
        print(f"    8'd{i}: data_out <= 16'h{val:04X};")

if __name__ == '__main__':
    import os
    funcs = {
        'exp':   safe_exp,
        'log':   safe_log,
        'sqrt':  safe_sqrt,
        'rsqrt': safe_rsqrt,
    }

    for name, func in funcs.items():
        entries = gen_lut(func)
        print(f"\n// ===== {name.upper()} FP16 LUT =====")
        print_sv_lut(name, entries)
        print()

    # Also save as binary for golden model
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fp16_luts')
    os.makedirs(out_dir, exist_ok=True)
    for name, func in funcs.items():
        entries = gen_lut(func)
        path = os.path.join(out_dir, f'{name}_fp16_lut.bin')
        with open(path, 'wb') as f:
            for val in entries:
                f.write(struct.pack('<H', val))
        print(f"Saved {path}")
