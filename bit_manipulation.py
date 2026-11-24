"""
bit_manipulation.py
===================
Utilities for low-level bit representation, extraction, and bit-flipping
across IEEE 754 Floating Point (FP32, FP16) and Fixed-Point (INT8) weights.

Hardware Security Context:
In Rowhammer (DRAM read-disturbance) attacks, an electrical disturbance
in adjacent DRAM rows forces charge leakage, flipping stored capacitor states
from 0->1 or 1->0.

IEEE 754 Float32 Layout:
  [31]       Sign (1 bit)
  [30:23]    Exponent (8 bits, bias = 127)
  [22:0]     Mantissa/Fraction (23 bits)

IEEE 754 Float16 (Half-Precision) Layout:
  [15]       Sign (1 bit)
  [14:10]    Exponent (5 bits, bias = 15)
  [9:0]      Mantissa/Fraction (10 bits)
"""

import struct
import numpy as np
import torch

def float32_to_bits(f: float) -> int:
    """Pack a 32-bit float into its uint32 bit integer representation."""
    return struct.unpack('>I', struct.pack('>f', float(f)))[0]

def bits_to_float32(b: int) -> float:
    """Unpack a uint32 bit integer back into a 32-bit float."""
    return struct.unpack('>f', struct.pack('>I', b & 0xFFFFFFFF))[0]

def flip_float32_bit(f: float, bit_idx: int) -> float:
    """Flip bit at index bit_idx (0 to 31) of a 32-bit float."""
    assert 0 <= bit_idx <= 31, "Bit index for float32 must be in [0, 31]"
    b = float32_to_bits(f)
    b_flipped = b ^ (1 << bit_idx)
    return bits_to_float32(b_flipped)

def float16_to_bits(f: float) -> int:
    """Pack a 16-bit float into its uint16 bit integer representation."""
    return int(np.float16(f).view(np.uint16))

def bits_to_float16(b: int) -> float:
    """Unpack a uint16 bit integer back into a 16-bit float."""
    return float(np.uint16(b & 0xFFFF).view(np.float16))

def flip_float16_bit(f: float, bit_idx: int) -> float:
    """Flip bit at index bit_idx (0 to 15) of a 16-bit float."""
    assert 0 <= bit_idx <= 15, "Bit index for float16 must be in [0, 15]"
    b = float16_to_bits(f)
    b_flipped = b ^ (1 << bit_idx)
    return bits_to_float16(b_flipped)

def flip_int8_bit(val: int, bit_idx: int) -> int:
    """Flip bit at index bit_idx (0 to 7) of a signed 8-bit integer."""
    assert 0 <= bit_idx <= 7, "Bit index for int8 must be in [0, 7]"
    # Map signed [-128, 127] to uint8 [0, 255]
    u = val & 0xFF
    u_flipped = u ^ (1 << bit_idx)
    # Convert back to signed int8
    return u_flipped if u_flipped < 128 else u_flipped - 256

def analyze_flip_impact(val: float, bit_idx: int, dtype: str = "float32"):
    """
    Analyze the theoretical and numerical shift of flipping a specific bit.
    Distinguishes whether the flipped bit falls in Sign, Exponent, or Mantissa.
    """
    if dtype == "float32":
        flipped = flip_float32_bit(val, bit_idx)
        if bit_idx == 31:
            segment = "Sign Bit [31]"
        elif 23 <= bit_idx <= 30:
            segment = f"Exponent Bit [{bit_idx}] (exp bit {bit_idx - 23})"
        else:
            segment = f"Mantissa Bit [{bit_idx}]"
    elif dtype == "float16":
        flipped = flip_float16_bit(val, bit_idx)
        if bit_idx == 15:
            segment = "Sign Bit [15]"
        elif 10 <= bit_idx <= 14:
            segment = f"Exponent Bit [{bit_idx}] (exp bit {bit_idx - 10})"
        else:
            segment = f"Mantissa Bit [{bit_idx}]"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    delta = flipped - val
    ratio = abs(flipped / (val + 1e-12))
    return {
        "original": val,
        "flipped": flipped,
        "delta": delta,
        "ratio": ratio,
        "segment": segment,
        "is_outlier": abs(delta) > 10.0
    }

if __name__ == "__main__":
    # Quick sanity check
    test_w = 0.0523  # Typical small neural network weight
    print(f"Original weight: {test_w:.6f}\n")
    print(f"{'Bit':<5} | {'Segment':<32} | {'Flipped Val':<16} | {'Delta':<16} | {'Outlier?'}")
    print("-" * 80)
    for bit in [31, 30, 29, 28, 23, 22, 10, 0]:
        res = analyze_flip_impact(test_w, bit, "float32")
        print(f"{bit:<5} | {res['segment']:<32} | {res['flipped']:<16.4e} | {res['delta']:<16.4e} | {res['is_outlier']}")
