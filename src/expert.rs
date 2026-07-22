//! The expert FFN kernel — the one shared implementation of a MoE expert's
//! forward pass and the blob -> f32 reconstruction that feeds it.
//!
//! M0 kept both inline in `diff.rs`; M1 distributes the same computation to the
//! node process, so the kernel lives here and both the in-process diff and the
//! dispatched node path call it. One kernel means determinism is owned once
//! (ADR-0018): a bf16-passthrough carve must match the source BIT-FOR-BIT
//! because the identical code runs on both sides.

use crate::Result;
use crate::blob::{Decoded, Dtype};
use crate::quant;

/// `y = down . (silu(gate . x) * (up . x))`; gate/up are [inter, hidden],
/// down is [hidden, inter], all row-major f32.
pub fn forward(gate: &[f32], up: &[f32], down: &[f32], hidden: usize, x: &[f32], y: &mut [f32]) {
    let inter = gate.len() / hidden;
    let mut a = vec![0f32; inter];
    for (ar, (grow, urow)) in a
        .iter_mut()
        .zip(gate.chunks_exact(hidden).zip(up.chunks_exact(hidden)))
    {
        let mut g = 0f32;
        let mut u = 0f32;
        for ((&gw, &uw), &xv) in grow.iter().zip(urow).zip(x) {
            g += gw * xv;
            u += uw * xv;
        }
        let silu = g / (1.0 + (-g).exp());
        *ar = silu * u;
    }
    for (yr, drow) in y.iter_mut().zip(down.chunks_exact(inter)) {
        let mut acc = 0f32;
        for (&dw, &av) in drow.iter().zip(&a) {
            acc += dw * av;
        }
        *yr = acc;
    }
}

/// Reconstruct an expert's (gate, up, down) f32 matrices from a decoded blob:
/// bf16 passthrough, or per-row dequantization for fp8/int8 carves. The dims
/// come from the blob header the decoder already validated.
pub fn reconstruct(d: &Decoded) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let (hidden, inter) = (d.header.hidden as usize, d.header.inter as usize);
    match d.header.dtype {
        Dtype::Bf16 => Ok((
            quant::bf16_to_f32_vec(d.gate)?,
            quant::bf16_to_f32_vec(d.up)?,
            quant::bf16_to_f32_vec(d.down)?,
        )),
        dt => {
            let (sg, su, sd) = d.scale_parts()?;
            Ok((
                quant::dequantize_matrix(dt, sg, d.gate, inter, hidden)?,
                quant::dequantize_matrix(dt, su, d.up, inter, hidden)?,
                quant::dequantize_matrix(dt, sd, d.down, hidden, inter)?,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_matches_hand_computation() {
        // hidden = 2, inter = 2. gate/up are [inter, hidden] row-major,
        // down is [hidden, inter]. Identity-ish weights make the algebra
        // hand-checkable: gate = up = I, down = I.
        let gate = [1.0f32, 0.0, 0.0, 1.0];
        let up = [1.0f32, 0.0, 0.0, 1.0];
        let down = [1.0f32, 0.0, 0.0, 1.0];
        let x = [1.5f32, -0.5];
        let mut y = [0f32; 2];
        forward(&gate, &up, &down, 2, &x, &mut y);
        // a_i = silu(x_i) * x_i ; y = down . a = a.
        for (i, &xi) in x.iter().enumerate() {
            let silu = xi / (1.0 + (-xi).exp());
            assert_eq!(y[i], silu * xi);
        }
    }
}
