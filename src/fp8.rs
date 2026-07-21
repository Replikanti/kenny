//! e4m3fn — the fp8 flavor used for weights (ADR-0018 groundwork): 1 sign,
//! 4 exponent bits (bias 7), 3 mantissa bits; NO infinities, max finite 448,
//! `0x7F`/`0xFF` are NaN. Hand-rolled per ADR-0021; round-to-nearest-even,
//! saturating at ±448. Cross-checked against an independent Python model and
//! locked by an exhaustive all-codes round-trip test.

/// 2^e as f32 built from bits — no libm, exact, deterministic.
fn pow2(e: i32) -> f32 {
    debug_assert!((-126..=127).contains(&e));
    f32::from_bits(((e + 127) as u32) << 23)
}

pub fn f32_to_e4m3(x: f32) -> u8 {
    let sign = ((x.to_bits() >> 31) as u8) << 7;
    if x.is_nan() {
        return sign | 0x7F;
    }
    let a = x.abs().min(448.0);
    if a == 0.0 {
        return sign;
    }
    if a < pow2(-6) {
        // Subnormal: value = m * 2^-9, m in 0..=7. A round-up to m == 8 is
        // exactly the minimum normal, whose encoding is (1 << 3) | 0 = 8.
        let m = (a * pow2(9)).round_ties_even() as u8;
        return sign | m;
    }
    let bits = a.to_bits();
    let mut e = ((bits >> 23) & 0xFF) as i32 - 127;
    let frac = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000); // [1, 2)
    let mut m = ((frac - 1.0) * 8.0).round_ties_even() as u32;
    if m == 8 {
        e += 1;
        m = 0;
    }
    if e > 8 {
        return sign | 0x7E; // defensive: pre-clamp makes this unreachable
    }
    sign | (((e + 7) as u8) << 3) | m as u8
}

pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0x0F;
    let m = (b & 0x07) as f32;
    if e == 0x0F && b & 0x07 == 0x07 {
        return f32::NAN;
    }
    if e == 0 {
        sign * m * pow2(-9)
    } else {
        sign * (1.0 + m / 8.0) * pow2(e as i32 - 7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vectors cross-checked against an independent Python implementation.
    #[test]
    fn golden_vectors() {
        let cases: &[(f32, u8)] = &[
            (1.0, 0x38),
            (0.5, 0x30),
            (448.0, 0x7E),
            (0.001953125, 0x01), // min subnormal 2^-9
            (240.0, 0x77),
            (-1.5, 0xBC),
            (1.0625, 0x38), // tie -> even (down)
            (1.1875, 0x3A), // tie -> even (up)
            (0.0, 0x00),
            (500.0, 0x7E),        // saturates
            (3.2, 0x45),          // -> 3.25
            (0.0009765625, 0x00), // tie with zero -> even
            (0.0029296875, 0x02), // subnormal tie -> even
        ];
        for &(x, want) in cases {
            assert_eq!(f32_to_e4m3(x), want, "input {x}");
        }
        assert_eq!(f32_to_e4m3(-0.0), 0x80);
    }

    #[test]
    fn exhaustive_code_roundtrip() {
        for b in 0u8..=255 {
            if b & 0x7F == 0x7F {
                assert!(e4m3_to_f32(b).is_nan(), "code {b:#04x} is NaN");
                continue;
            }
            assert_eq!(f32_to_e4m3(e4m3_to_f32(b)), b, "code {b:#04x}");
        }
    }

    #[test]
    fn nan_and_saturation() {
        assert_eq!(f32_to_e4m3(f32::NAN) & 0x7F, 0x7F);
        assert_eq!(f32_to_e4m3(f32::INFINITY), 0x7E);
        assert_eq!(f32_to_e4m3(f32::NEG_INFINITY), 0xFE);
        assert_eq!(f32_to_e4m3(1e10), 0x7E);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
    }
}
