"""
visualize_neuron.py
===================
Interactive Terminal Diagnostic Tool for Inspecting IEEE 754 Floating-Point
Bit Flips and Neuron Activation Dynamics.

Demonstrates:
- Bit-by-bit inspection of Sign, Exponent, and Mantissa fields
- Real-time simulation of neuron dot-product and activation explosion
"""

import sys
import argparse
from bit_manipulation import float32_to_bits, bits_to_float32, flip_float32_bit

def format_bits_32(val: float) -> str:
    """Format a 32-bit float into grouped binary representation."""
    b = float32_to_bits(val)
    bin_str = f"{b:032b}"
    sign = bin_str[0]
    exp = bin_str[1:9]
    mantissa = bin_str[9:]
    return f"{sign} | {exp} | {mantissa}"

def inspect_bit_flip(val: float, bit_idx: int):
    """Print detailed ASCII breakdown of a single bit flip."""
    b_orig = float32_to_bits(val)
    f_new = flip_float32_bit(val, bit_idx)
    b_new = float32_to_bits(f_new)

    bin_orig = f"{b_orig:032b}"
    bin_new = f"{b_new:032b}"

    # Extract fields
    exp_orig = (b_orig >> 23) & 0xFF
    exp_new = (b_new >> 23) & 0xFF
    mant_orig = b_orig & 0x7FFFFF
    mant_new = b_new & 0x7FFFFF

    if bit_idx == 31:
        seg = "SIGN BIT (inverts sign)"
    elif 23 <= bit_idx <= 30:
        seg = f"EXPONENT BIT (power-of-2 scale multiplier: 2^{bit_idx - 23})"
    else:
        seg = f"MANTISSA BIT (fractional precision: 2^{- (23 - bit_idx)})"

    print("=" * 75)
    print(f" IEEE 754 BIT-FLIP INSPECTOR: Bit {bit_idx} ({seg})")
    print("=" * 75)

    print("\n[Bit Layout: S | Exponent (8b) | Mantissa (23b)]")
    print(f"Original:  {format_bits_32(val)}  ==>  {val:+.6e}")
    
    # Highlight inverted bit position
    arrow_pos = 0
    # Map bit_idx (0..31 where 31 is leftmost)
    char_idx = 31 - bit_idx
    if char_idx > 0:
        char_idx += 3  # skip " | "
    if char_idx > 11:
        char_idx += 3  # skip second " | "
    pointer = " " * (11 + char_idx) + "^ (Flipped)"

    print(f"Corrupted: {format_bits_32(f_new)}  ==>  {f_new:+.6e}")
    print(pointer)

    print("\n[Field Decodings]")
    print(f"  Exponent : {exp_orig:3d} (bias -127: {exp_orig - 127:+4d})  ==>  {exp_new:3d} (bias -127: {exp_new - 127:+4d})")
    print(f"  Mantissa : 0x{mant_orig:06X}               ==>  0x{mant_new:06X}")
    print(f"  Weight Δ : {f_new - val:+.6e}")

    # Simulated Neuron Dot Product
    print("\n[Neuron Dot-Product Impact Simulation (x = 1.0)]")
    clean_out = val * 1.0
    corrupt_out = f_new * 1.0
    print(f"  Clean Neuron Output     : {clean_out:+.6f}")
    print(f"  Corrupted Neuron Output : {corrupt_out:+.4e}")
    if abs(corrupt_out) > 1e10:
        print("  [!] EXPLOSIVE ACTIVATION: Overwhelms all layer activations, forcing Winner-Group collapse!")

    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect IEEE 754 Bit Flips")
    parser.add_argument("--weight", type=float, default=0.04754, help="Original floating-point weight")
    parser.add_argument("--bit", type=int, default=30, help="Bit index to flip (0..31)")
    args = parser.parse_args()

    inspect_bit_flip(args.weight, args.bit)
