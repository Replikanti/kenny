//! KNY1 expert blob — consensus format, version 1 (ADR-0005).
//!
//! Fixed little-endian layout; every field below is part of the hashed bytes.
//! Change protocol: a STRUCTURAL change (field offsets, widths, order, or the
//! meaning of existing bytes) is CID-breaking — bump `VERSION`, write the
//! ADR, update the golden tests, in that order, in one PR. ADDITIVE dtype-tag
//! registration (a new tag value, loudly rejected by older decoders, zero
//! byte impact on existing blobs) needs the authorizing ADR and a row in the
//! table below, but NOT a version bump — bumping would re-encode the version
//! field of every future blob and orphan nothing but goldens. Tags 1/2 were
//! added under ADR-0012 this way; bf16 goldens are unchanged.
//!
//! ```text
//! offset  size  field
//! 0       4     magic "KNY1"
//! 4       2     version (u16) = 1
//! 6       2     layer (u16)
//! 8       2     expert (u16)
//! 10      1     dtype (u8): 0 = bf16, 1 = fp8 e4m3 (per-channel),
//!               2 = int8 (per-channel)
//! 11      1     pad, must be 0
//! 12      4     hidden (u32)
//! 16      4     inter (u32) — moe_intermediate
//! 20      4     scale_len (u32) — 0 for bf16;
//!               4 * (inter + inter + hidden) for fp8/int8
//! 24      ..    scale block: per-output-row f32 LE scales — gate rows
//!               (inter), then up rows (inter), then down rows (hidden);
//!               dequant is w[r, c] = decode(q[r, c]) * scale[r]
//! 24+s    ..    payload: gate_proj, up_proj, down_proj — row-major bytes,
//!               each hidden*inter*dtype_size long
//! ```
//!
//! CID = lowercase blake3 hex of the entire blob (header + scale + payload).

use std::path::PathBuf;

use crate::error::{Error, Result};

pub const MAGIC: [u8; 4] = *b"KNY1";
pub const VERSION: u16 = 1;
pub const HEADER_LEN: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16 = 0,
    Fp8 = 1,
    Int8 = 2,
}

impl Dtype {
    pub fn from_u8(v: u8) -> Result<Dtype> {
        match v {
            0 => Ok(Dtype::Bf16),
            1 => Ok(Dtype::Fp8),
            2 => Ok(Dtype::Int8),
            other => Err(Error::parse(format!("blob: unknown dtype tag {other}"))),
        }
    }

    pub fn from_name(name: &str) -> Result<Dtype> {
        match name {
            "bf16" => Ok(Dtype::Bf16),
            "fp8" => Ok(Dtype::Fp8),
            "int8" => Ok(Dtype::Int8),
            other => Err(Error::parse(format!(
                "unknown dtype {other:?} (expected bf16, fp8 or int8)"
            ))),
        }
    }

    pub fn size(self) -> u64 {
        match self {
            Dtype::Bf16 => 2,
            Dtype::Fp8 | Dtype::Int8 => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Dtype::Bf16 => "bf16",
            Dtype::Fp8 => "fp8",
            Dtype::Int8 => "int8",
        }
    }

    /// Source safetensors dtype this carve mode accepts (quantized modes read
    /// bf16 sources and quantize centrally, ADR-0012).
    pub fn source_dtype(self) -> &'static str {
        match self {
            Dtype::Bf16 | Dtype::Fp8 | Dtype::Int8 => "BF16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub layer: u16,
    pub expert: u16,
    pub dtype: Dtype,
    pub hidden: u32,
    pub inter: u32,
}

impl Header {
    /// Bytes of one matrix (gate, up and down are all hidden*inter elements).
    pub fn matrix_len(&self) -> Result<usize> {
        (self.hidden as u64)
            .checked_mul(self.inter as u64)
            .and_then(|n| n.checked_mul(self.dtype.size()))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::parse("blob: matrix size overflows"))
    }
}

/// Scale-block length implied by the header: 0 for bf16, one f32 per output
/// row (gate: inter, up: inter, down: hidden) for quantized dtypes.
pub fn expected_scale_len(dtype: Dtype, hidden: u32, inter: u32) -> Result<usize> {
    match dtype {
        Dtype::Bf16 => Ok(0),
        Dtype::Fp8 | Dtype::Int8 => (inter as u64)
            .checked_add(inter as u64)
            .and_then(|n| n.checked_add(hidden as u64))
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::parse("blob: scale block size overflows")),
    }
}

pub fn encode(h: &Header, scale: &[u8], gate: &[u8], up: &[u8], down: &[u8]) -> Result<Vec<u8>> {
    if h.hidden == 0 || h.inter == 0 {
        return Err(Error::parse("blob: zero dimension"));
    }
    let want_scale = expected_scale_len(h.dtype, h.hidden, h.inter)?;
    if scale.len() != want_scale {
        return Err(Error::parse(format!(
            "blob: {} scale bytes, {} dtype needs exactly {want_scale}",
            scale.len(),
            h.dtype.name()
        )));
    }
    let msize = h.matrix_len()?;
    for (name, m) in [("gate", gate), ("up", up), ("down", down)] {
        if m.len() != msize {
            return Err(Error::parse(format!(
                "blob: {name}_proj has {} bytes, dims say {msize}",
                m.len()
            )));
        }
    }
    let mut out = Vec::with_capacity(HEADER_LEN + scale.len() + 3 * msize);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&h.layer.to_le_bytes());
    out.extend_from_slice(&h.expert.to_le_bytes());
    out.push(h.dtype as u8);
    out.push(0); // pad
    out.extend_from_slice(&h.hidden.to_le_bytes());
    out.extend_from_slice(&h.inter.to_le_bytes());
    out.extend_from_slice(&(scale.len() as u32).to_le_bytes());
    out.extend_from_slice(scale);
    out.extend_from_slice(gate);
    out.extend_from_slice(up);
    out.extend_from_slice(down);
    Ok(out)
}

#[derive(Debug)]
pub struct Decoded<'a> {
    pub header: Header,
    pub scale: &'a [u8],
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub down: &'a [u8],
}

pub fn decode(bytes: &[u8]) -> Result<Decoded<'_>> {
    if bytes.len() < HEADER_LEN {
        return Err(Error::parse(format!(
            "blob: {} bytes, header needs {HEADER_LEN}",
            bytes.len()
        )));
    }
    if bytes[0..4] != MAGIC {
        return Err(Error::parse("blob: bad magic (not a KNY1 blob)"));
    }
    let version = u16::from_le_bytes(bytes[4..6].try_into().expect("2 bytes"));
    if version != VERSION {
        return Err(Error::parse(format!(
            "blob: version {version}, this build speaks {VERSION}"
        )));
    }
    let layer = u16::from_le_bytes(bytes[6..8].try_into().expect("2 bytes"));
    let expert = u16::from_le_bytes(bytes[8..10].try_into().expect("2 bytes"));
    let dtype = Dtype::from_u8(bytes[10])?;
    if bytes[11] != 0 {
        return Err(Error::parse("blob: nonzero pad byte"));
    }
    let hidden = u32::from_le_bytes(bytes[12..16].try_into().expect("4 bytes"));
    let inter = u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes"));
    let scale_len = u32::from_le_bytes(bytes[20..24].try_into().expect("4 bytes")) as usize;
    let header = Header {
        layer,
        expert,
        dtype,
        hidden,
        inter,
    };
    if hidden == 0 || inter == 0 {
        return Err(Error::parse("blob: zero dimension"));
    }
    if scale_len != expected_scale_len(dtype, hidden, inter)? {
        return Err(Error::parse(format!(
            "blob: scale block is {scale_len} bytes, {} dtype implies {}",
            dtype.name(),
            expected_scale_len(dtype, hidden, inter)?
        )));
    }
    let msize = header.matrix_len()?;
    // Checked u64 math: a crafted header must yield a clean error, never a
    // debug-overflow panic — this decoder is the seam that will face
    // pool-fetched bytes from M1 on.
    let expect = (msize as u64)
        .checked_mul(3)
        .and_then(|p| p.checked_add(HEADER_LEN as u64))
        .and_then(|p| p.checked_add(scale_len as u64))
        .ok_or_else(|| Error::parse("blob: header implies an impossible size"))?;
    if bytes.len() as u64 != expect {
        return Err(Error::parse(format!(
            "blob: {} bytes, header implies exactly {expect}",
            bytes.len()
        )));
    }
    let scale = &bytes[HEADER_LEN..HEADER_LEN + scale_len];
    let p = HEADER_LEN + scale_len;
    Ok(Decoded {
        header,
        scale,
        gate: &bytes[p..p + msize],
        up: &bytes[p + msize..p + 2 * msize],
        down: &bytes[p + 2 * msize..p + 3 * msize],
    })
}

impl<'a> Decoded<'a> {
    /// Split the scale block into (gate, up, down) per-row f32 LE scales.
    /// Errors for bf16 blobs, which carry none.
    pub fn scale_parts(&self) -> Result<(&'a [u8], &'a [u8], &'a [u8])> {
        if self.header.dtype == Dtype::Bf16 {
            return Err(Error::parse("blob: bf16 blobs have no scale block"));
        }
        let i4 = self.header.inter as usize * 4;
        let h4 = self.header.hidden as usize * 4;
        // decode() guarantees this for blobs it produced; a hand-built
        // Decoded must not be able to trigger a slice panic here.
        if self.scale.len() != 2 * i4 + h4 {
            return Err(Error::parse("blob: scale block length disagrees with dims"));
        }
        Ok((
            &self.scale[..i4],
            &self.scale[i4..2 * i4],
            &self.scale[2 * i4..2 * i4 + h4],
        ))
    }
}

pub fn cid(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

/// Storage path relative to the blobs root: `<first-2-hex>/<full-cid>`.
pub fn rel_path(cid: &str) -> PathBuf {
    PathBuf::from(&cid[..2]).join(cid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (Header, Vec<u8>, Vec<u8>, Vec<u8>) {
        let h = Header {
            layer: 3,
            expert: 200,
            dtype: Dtype::Bf16,
            hidden: 4,
            inter: 2,
        };
        let m = 4 * 2 * 2; // hidden * inter * dtype size
        let gate: Vec<u8> = (0..m as u8).collect();
        let up: Vec<u8> = (0..m as u8).map(|b| b ^ 0xFF).collect();
        let down: Vec<u8> = (0..m as u8).map(|b| b.wrapping_mul(7)).collect();
        (h, gate, up, down)
    }

    #[test]
    fn roundtrip() {
        let (h, gate, up, down) = sample();
        let blob = encode(&h, &[], &gate, &up, &down).unwrap();
        let d = decode(&blob).unwrap();
        assert_eq!(d.header, h);
        assert_eq!(d.scale, &[] as &[u8]);
        assert_eq!(d.gate, &gate[..]);
        assert_eq!(d.up, &up[..]);
        assert_eq!(d.down, &down[..]);
    }

    #[test]
    fn rejects_corruption() {
        let (h, gate, up, down) = sample();
        let blob = encode(&h, &[], &gate, &up, &down).unwrap();

        let mut bad = blob.clone();
        bad[0] = b'X';
        assert!(decode(&bad).is_err(), "magic");

        let mut bad = blob.clone();
        bad[4] = 9;
        assert!(decode(&bad).is_err(), "version");

        let mut bad = blob.clone();
        bad[11] = 1;
        assert!(decode(&bad).is_err(), "pad");

        let mut bad = blob.clone();
        bad.pop();
        assert!(decode(&bad).is_err(), "truncated");

        let mut bad = blob.clone();
        bad.push(0);
        assert!(decode(&bad).is_err(), "oversized");

        assert!(decode(&blob[..10]).is_err(), "short header");
    }

    #[test]
    fn overflow_header_is_rejected_not_panicking() {
        // hidden * inter chosen so matrix_len fits u64 but 3 * msize wraps.
        let mut bad = Vec::new();
        bad.extend_from_slice(&MAGIC);
        bad.extend_from_slice(&VERSION.to_le_bytes());
        bad.extend_from_slice(&0u16.to_le_bytes()); // layer
        bad.extend_from_slice(&0u16.to_le_bytes()); // expert
        bad.push(0); // dtype bf16
        bad.push(0); // pad
        bad.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // hidden
        bad.extend_from_slice(&0x8000_0000u32.to_le_bytes()); // inter
        bad.extend_from_slice(&0u32.to_le_bytes()); // scale_len
        bad.extend_from_slice(&[0u8; 16]);
        assert!(decode(&bad).is_err());
    }

    #[test]
    fn rejects_bad_encode_input() {
        let (h, gate, up, down) = sample();
        assert!(
            encode(&h, &[1], &gate, &up, &down).is_err(),
            "scale on bf16"
        );
        assert!(
            encode(&h, &[], &gate[..gate.len() - 1], &up, &down).is_err(),
            "short matrix"
        );
        let zero = Header { hidden: 0, ..h };
        assert!(encode(&zero, &[], &[], &[], &[]).is_err(), "zero dim");
    }

    // Golden CID: locks KNY1 v1 byte layout. If this changes, the format
    // changed — that requires a VERSION bump plus an ADR, never a test edit
    // alone (kenny-format-auditor enforces this).
    #[test]
    fn golden_cid_v1() {
        let (h, gate, up, down) = sample();
        let blob = encode(&h, &[], &gate, &up, &down).unwrap();
        assert_eq!(blob.len(), HEADER_LEN + 3 * 16);
        assert_eq!(
            cid(&blob),
            "7494d99e6b53f11e6fcef0868f84add97ba0ee7d1bd3fa8e08017859af6cedb9"
        );
    }

    #[test]
    fn quantized_roundtrip_and_scale_split() {
        let h = Header {
            layer: 1,
            expert: 2,
            dtype: Dtype::Int8,
            hidden: 4,
            inter: 2,
        };
        // gate/up: 2x4 = 8 bytes each; down: 4x2 = 8 bytes; scales: (2+2+4)*4.
        let scale: Vec<u8> = (0..32u8).collect();
        let m: Vec<u8> = (0..8u8).collect();
        let blob = encode(&h, &scale, &m, &m, &m).unwrap();
        let d = decode(&blob).unwrap();
        assert_eq!(d.header.dtype, Dtype::Int8);
        let (sg, su, sd) = d.scale_parts().unwrap();
        assert_eq!(sg, &scale[..8]);
        assert_eq!(su, &scale[8..16]);
        assert_eq!(sd, &scale[16..32]);
        // Wrong scale length is rejected on both sides.
        assert!(encode(&h, &scale[..31], &m, &m, &m).is_err());
        let bf = Header {
            dtype: Dtype::Bf16,
            ..h
        };
        let m2: Vec<u8> = (0..16u8).collect();
        let bf_blob = encode(&bf, &[], &m2, &m2, &m2).unwrap();
        assert!(
            decode(&bf_blob).unwrap().scale_parts().is_err(),
            "bf16 has no scales"
        );
    }

    // Golden CIDs for the quantized layouts: lock dtype tags 1/2, the scale
    // block, and payload order directly at the blob level (the manifest
    // goldens lock them only transitively). Same change protocol as
    // golden_cid_v1.
    #[test]
    fn golden_cid_quantized() {
        let h = Header {
            layer: 1,
            expert: 2,
            dtype: Dtype::Fp8,
            hidden: 4,
            inter: 2,
        };
        let scale: Vec<u8> = (0..32u8).collect(); // (2 + 2 + 4) rows * 4 bytes
        let m: Vec<u8> = (0..8u8).collect(); // 2x4 / 4x2 at 1 byte per elem
        let fp8_blob = encode(&h, &scale, &m, &m, &m).unwrap();
        assert_eq!(
            cid(&fp8_blob),
            "be278cbdfb536beaba145a27b7d62acb68b9ef7e2de24289e2a98d175090fd68"
        );

        let h = Header {
            dtype: Dtype::Int8,
            ..h
        };
        let int8_blob = encode(&h, &scale, &m, &m, &m).unwrap();
        assert_eq!(
            cid(&int8_blob),
            "37fb559e354cbbdea02198a1b810cba2361c79eba95dbed903954c56d3533ed9"
        );
        assert_ne!(cid(&fp8_blob), cid(&int8_blob), "dtype tag is hashed");
    }

    #[test]
    fn rel_path_shape() {
        let c = "ab".to_string() + &"0".repeat(62);
        assert_eq!(rel_path(&c), PathBuf::from("ab").join(c));
    }
}
