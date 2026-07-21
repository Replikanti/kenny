//! KNY1 expert blob — consensus format, version 1 (ADR-0005).
//!
//! Fixed little-endian layout; every field below is part of the hashed bytes,
//! so any change here is a CID-breaking format change: bump `VERSION`, write
//! the ADR, update the golden tests — in that order, in one PR.
//!
//! ```text
//! offset  size  field
//! 0       4     magic "KNY1"
//! 4       2     version (u16) = 1
//! 6       2     layer (u16)
//! 8       2     expert (u16)
//! 10      1     dtype (u8): 0 = bf16
//! 11      1     pad, must be 0
//! 12      4     hidden (u32)
//! 16      4     inter (u32) — moe_intermediate
//! 20      4     scale_len (u32) — 0 for bf16
//! 24      ..    scale block (scale_len bytes)
//! 24+s    ..    payload: gate_proj, up_proj, down_proj — row-major source
//!               bytes, each hidden*inter*dtype_size long
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
}

impl Dtype {
    pub fn from_u8(v: u8) -> Result<Dtype> {
        match v {
            0 => Ok(Dtype::Bf16),
            other => Err(Error::parse(format!("blob: unknown dtype tag {other}"))),
        }
    }

    pub fn from_name(name: &str) -> Result<Dtype> {
        match name {
            "bf16" => Ok(Dtype::Bf16),
            "fp8" | "int8" => Err(Error::parse(format!(
                "dtype {name:?} is not implemented yet (M0 is bf16 passthrough; fp8/int8 arrive \
                 with the kenny diff milestone, ADR-0018 pending)"
            ))),
            other => Err(Error::parse(format!(
                "unknown dtype {other:?} (expected bf16)"
            ))),
        }
    }

    pub fn size(self) -> u64 {
        match self {
            Dtype::Bf16 => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Dtype::Bf16 => "bf16",
        }
    }

    /// Source safetensors dtype this carve mode accepts.
    pub fn source_dtype(self) -> &'static str {
        match self {
            Dtype::Bf16 => "BF16",
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

pub fn encode(h: &Header, scale: &[u8], gate: &[u8], up: &[u8], down: &[u8]) -> Result<Vec<u8>> {
    if h.hidden == 0 || h.inter == 0 {
        return Err(Error::parse("blob: zero dimension"));
    }
    if h.dtype == Dtype::Bf16 && !scale.is_empty() {
        return Err(Error::parse("blob: bf16 blobs carry no scale block"));
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
    if dtype == Dtype::Bf16 && scale_len != 0 {
        return Err(Error::parse("blob: bf16 blob with nonempty scale block"));
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
    fn rel_path_shape() {
        let c = "ab".to_string() + &"0".repeat(62);
        assert_eq!(rel_path(&c), PathBuf::from("ab").join(c));
    }
}
