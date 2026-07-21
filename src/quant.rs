//! Per-output-channel weight quantization — carve-time and central, per
//! ADR-0012: nodes serve canonical bits, they never transform weights.
//!
//! For each output row r of a matrix: scale `s_r = max|w[r,:]| / limit`
//! (448 for fp8 e4m3, 127 for int8; `s_r = 1.0` for an all-zero row), stored
//! as f32 LE in the blob's scale block. Quantized element:
//! `q[r,c] = round(w[r,c] / s_r)` in the target format; dequantization is
//! `q[r,c] * s_r`. Plain IEEE f32 arithmetic throughout — deterministic
//! across platforms, so identical sources always produce identical CIDs.

use crate::bf16::bf16_to_f32;
use crate::blob::Dtype;
use crate::error::{Error, Result};
use crate::fp8;

pub fn bf16_to_f32_vec(src: &[u8]) -> Result<Vec<f32>> {
    if !src.len().is_multiple_of(2) {
        return Err(Error::parse("bf16 buffer has odd byte length"));
    }
    Ok(src
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

/// Quantize a row-major `rows x cols` bf16 matrix. Returns (scales_le, data).
pub fn quantize_matrix(
    dtype: Dtype,
    src_bf16: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let limit = match dtype {
        Dtype::Fp8 => 448.0f32,
        Dtype::Int8 => 127.0,
        Dtype::Bf16 => return Err(Error::parse("quantize: bf16 is passthrough, not quantized")),
    };
    let w = bf16_to_f32_vec(src_bf16)?;
    if w.len() != rows * cols {
        return Err(Error::parse(format!(
            "quantize: {} values for a {rows}x{cols} matrix",
            w.len()
        )));
    }
    let mut scales = Vec::with_capacity(rows * 4);
    let mut data = Vec::with_capacity(rows * cols);
    for row in w.chunks_exact(cols) {
        let mut max = 0.0f32;
        for &v in row {
            if !v.is_finite() {
                return Err(Error::parse("quantize: non-finite weight in source tensor"));
            }
            max = max.max(v.abs());
        }
        let s = if max == 0.0 { 1.0 } else { max / limit };
        scales.extend_from_slice(&s.to_le_bytes());
        match dtype {
            Dtype::Fp8 => {
                for &v in row {
                    data.push(fp8::f32_to_e4m3(v / s));
                }
            }
            Dtype::Int8 => {
                for &v in row {
                    let q = (v / s).round_ties_even().clamp(-127.0, 127.0) as i8;
                    data.push(q as u8);
                }
            }
            Dtype::Bf16 => unreachable!("rejected above"),
        }
    }
    Ok((scales, data))
}

pub fn dequantize_matrix(
    dtype: Dtype,
    scales_le: &[u8],
    data: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    if dtype == Dtype::Bf16 {
        return Err(Error::parse(
            "dequantize: bf16 is passthrough, not quantized",
        ));
    }
    if scales_le.len() != rows * 4 || data.len() != rows * cols {
        return Err(Error::parse(format!(
            "dequantize: {} scale bytes / {} data bytes for a {rows}x{cols} matrix",
            scales_le.len(),
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(rows * cols);
    for (sb, row) in scales_le.chunks_exact(4).zip(data.chunks_exact(cols)) {
        let s = f32::from_le_bytes(sb.try_into().expect("4-byte chunk"));
        match dtype {
            Dtype::Fp8 => {
                for &b in row {
                    out.push(fp8::e4m3_to_f32(b) * s);
                }
            }
            Dtype::Int8 => {
                for &b in row {
                    out.push((b as i8) as f32 * s);
                }
            }
            Dtype::Bf16 => unreachable!("rejected above"),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16::f32_to_bf16;
    use crate::rng::SplitMix64;

    fn bf16_matrix(rows: usize, cols: usize, tag: &str) -> Vec<u8> {
        let mut rng = SplitMix64::for_name(7, tag);
        let mut out = Vec::with_capacity(rows * cols * 2);
        for _ in 0..rows * cols {
            out.extend_from_slice(&f32_to_bf16(rng.next_unit_f32()).to_le_bytes());
        }
        out
    }

    #[test]
    fn int8_error_bound() {
        let (rows, cols) = (5, 16);
        let src = bf16_matrix(rows, cols, "int8");
        let w = bf16_to_f32_vec(&src).unwrap();
        let (scales, data) = quantize_matrix(Dtype::Int8, &src, rows, cols).unwrap();
        let d = dequantize_matrix(Dtype::Int8, &scales, &data, rows, cols).unwrap();
        for r in 0..rows {
            let s = f32::from_le_bytes(scales[r * 4..r * 4 + 4].try_into().unwrap());
            for c in 0..cols {
                let (a, b) = (w[r * cols + c], d[r * cols + c]);
                // Rounding to the nearest int8 step keeps the error <= s/2.
                assert!(
                    (a - b).abs() <= s / 2.0 + f32::EPSILON,
                    "({r},{c}): {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn fp8_error_bound() {
        let (rows, cols) = (5, 16);
        let src = bf16_matrix(rows, cols, "fp8");
        let w = bf16_to_f32_vec(&src).unwrap();
        let (scales, data) = quantize_matrix(Dtype::Fp8, &src, rows, cols).unwrap();
        let d = dequantize_matrix(Dtype::Fp8, &scales, &data, rows, cols).unwrap();
        for r in 0..rows {
            let s = f32::from_le_bytes(scales[r * 4..r * 4 + 4].try_into().unwrap());
            for c in 0..cols {
                let (a, b) = (w[r * cols + c], d[r * cols + c]);
                // e4m3 RNE: relative error <= 2^-4 for normals, plus a small
                // absolute floor in the subnormal range.
                let bound = a.abs() * 0.0625 + s * 0.002;
                assert!(
                    (a - b).abs() <= bound,
                    "({r},{c}): {a} vs {b}, bound {bound}"
                );
            }
        }
    }

    #[test]
    fn zero_row_gets_unit_scale() {
        let src = vec![0u8; 2 * 8]; // one row of 8 bf16 zeros
        let (scales, data) = quantize_matrix(Dtype::Int8, &src, 1, 8).unwrap();
        assert_eq!(f32::from_le_bytes(scales[0..4].try_into().unwrap()), 1.0);
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn rejects_bad_input() {
        assert!(quantize_matrix(Dtype::Bf16, &[0, 0], 1, 1).is_err(), "bf16");
        assert!(quantize_matrix(Dtype::Int8, &[0, 0], 1, 2).is_err(), "size");
        assert!(
            quantize_matrix(Dtype::Int8, &[0], 1, 1).is_err(),
            "odd bytes"
        );
        let nan = f32_to_bf16(f32::NAN).to_le_bytes();
        assert!(
            quantize_matrix(Dtype::Int8, &nan, 1, 1).is_err(),
            "non-finite source"
        );
        assert!(dequantize_matrix(Dtype::Int8, &[0; 4], &[0; 3], 1, 2).is_err());
    }
}
