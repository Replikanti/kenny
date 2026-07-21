//! bf16 <-> f32 conversion, hand-rolled per ADR-0021 (the `half` crate would
//! be a dependency for ~15 lines). bf16 is the upper 16 bits of an IEEE f32;
//! conversion down is round-to-nearest-even on the truncated 16 bits.

/// f32 -> bf16 with round-to-nearest-even. NaN is quietened (sign and payload
/// top bit preserved) so a signalling NaN can never round into an infinity.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x0040;
    }
    // Adding 0x7FFF plus the lowest kept bit implements RNE; overflow into
    // the exponent naturally rounds large finite values to infinity.
    let round = 0x7FFF + ((b >> 16) & 1);
    (b.wrapping_add(round) >> 16) as u16
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vectors cross-checked against an independent Python implementation.
    #[test]
    fn known_vectors() {
        let cases: &[(u32, u16)] = &[
            (0x3F80_0000, 0x3F80), // 1.0
            (0x3F80_8000, 0x3F80), // tie -> even (down)
            (0x3F81_8000, 0x3F82), // tie -> even (up)
            (0x3F80_8001, 0x3F81), // just above tie -> up
            (0x7F7F_FFFF, 0x7F80), // f32::MAX -> +inf
            (0xBF80_0000, 0xBF80), // -1.0
            (0x0000_0000, 0x0000), // +0.0
            (0x8000_0000, 0x8000), // -0.0
            (0x4020_0000, 0x4020), // 2.5 (exactly representable)
        ];
        for &(bits, want) in cases {
            assert_eq!(f32_to_bf16(f32::from_bits(bits)), want, "bits {bits:#010X}");
        }
    }

    #[test]
    fn nan_and_inf() {
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xFF80);
        // Signalling NaN (upper mantissa bits zero) must not become inf.
        let snan = f32::from_bits(0x7F80_0001);
        assert_eq!(f32_to_bf16(snan), 0x7FC0);
        let neg_snan = f32::from_bits(0xFF80_0001);
        assert_eq!(f32_to_bf16(neg_snan), 0xFFC0);
        assert!(bf16_to_f32(0x7FC0).is_nan());
    }

    #[test]
    fn roundtrip_representable() {
        // Values whose f32 form has a zero low half round-trip exactly.
        for h in [0x0000u16, 0x3F80, 0xC2F7, 0x7F7F, 0x0080, 0x8001] {
            assert_eq!(f32_to_bf16(bf16_to_f32(h)), h, "bf16 {h:#06X}");
        }
    }
}
