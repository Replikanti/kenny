//! The wire layer — activation codec + dispatch/gather framing + a byte-counting
//! transport, all behind one module boundary (ADR-0016).
//!
//! Two consensus surfaces live here, both with the same change protocol as the
//! blob format (`src/blob.rs`): the byte layout below is protocol consensus
//! (ADR-0011 — "wire bytes are canonical"), so a STRUCTURAL change (field
//! offsets, widths, order, magic values, or codec byte production) is a
//! version event — bump `WIRE_VERSION` (framing) or the codec's `codec_version`
//! (activation bytes), write the ADR, update the golden tests, in that order,
//! in one PR. A node may never choose its own encoding or "equivalent" bytes.
//!
//! Codec version travels in the handshake (ADR-0011): peers agree on the exact
//! activation encoding before any dispatch, and unknown magic / wire version /
//! codec is rejected loudly.
//!
//! The transport is interim sync TCP (ADR-0016, proposed): std threads, blocking
//! I/O, no async runtime; it sits behind this one boundary so M3 can swap it.
//! Each successfully completed read/write advances the byte counters by its full
//! length, so wire bytes are measured, not estimated (MANIFESTO §4.3 / BENCH "no
//! vibes"); framing bytes are accounted exactly and separately from payload so
//! tests can assert up/down independently. A call that errors mid-transfer ends
//! the session (the connection is torn down and the counters are not consumed),
//! so the counters are exact for the successful path — the only path BENCH reads.
//!
//! ```text
//! Handshake frame (fixed 44 bytes, no length prefix):
//! offset  size  field
//! 0       4     magic "KNYW"
//! 4       2     wire_version (u16 LE) = 1
//! 6       2     codec_id (u16 LE): 1 = fp8 e4m3, 2 = bf16
//! 8       2     codec_version (u16 LE)
//! 10      2     pad, must be 0
//! 12      32    manifest identity (raw blake3 bytes, ADR-0005)
//!
//! Dispatch frame (spine -> node): send x ONCE, list the experts to run:
//! offset  size  field
//! 0       4     magic "KNYD"
//! 4       2     layer (u16 LE)
//! 6       2     n_experts (u16 LE)
//! 8       4     x_len (u32 LE) — encoded activation bytes = hidden * codec_bytes
//! 12      ..    x: x_len bytes of codec-encoded activation
//! 12+x    ..    expert_ids: n_experts * u16 LE
//!
//! Gather frame (node -> spine): one record per requested expert:
//! offset  size  field
//! 0       4     magic "KNYG"
//! 4       2     layer (u16 LE)
//! 6       2     n_results (u16 LE)
//! 8       4     y_len (u32 LE) — encoded bytes of ONE answered y = hidden * codec_bytes
//! 12      ..    records: n_results * record
//!   record: expert_id (u16 LE), status (u8: 0 = ok, 1 = not-held),
//!           then y_len bytes of encoded y iff status == ok (none for not-held)
//! ```

use std::io::{Read, Write};

use crate::bf16;
use crate::error::{Error, Result};
use crate::fp8;

pub const MAGIC_HANDSHAKE: [u8; 4] = *b"KNYW";
pub const MAGIC_DISPATCH: [u8; 4] = *b"KNYD";
pub const MAGIC_GATHER: [u8; 4] = *b"KNYG";

/// Framing version — the dispatch/gather/handshake byte layout. A structural
/// change bumps this (consensus surface, ADR-0011).
pub const WIRE_VERSION: u16 = 1;

/// Fixed sizes so byte accounting is exact and separable from payload (A5): a
/// caller can compute `total = payload + known framing` from these alone.
pub const HANDSHAKE_LEN: usize = 44;
pub const DISPATCH_HEADER_LEN: usize = 12;
pub const GATHER_HEADER_LEN: usize = 12;
/// expert_id (u16) + status (u8) that precedes each gather record's optional y.
pub const GATHER_RECORD_HEADER_LEN: usize = 3;

/// Registered codec ids. fp8 is the baseline (MANIFESTO §4.3); bf16 is the
/// lossless validation-reference codec (ADR-0018 groundwork).
pub const FP8_CODEC_ID: u16 = 1;
pub const BF16_CODEC_ID: u16 = 2;

// -------------------------------------------------------------------------
// Codecs (ADR-0011)
// -------------------------------------------------------------------------

/// The activation wire format. For a given `(codec_id, codec_version)` the bytes
/// for a given activation are defined exactly — no per-node freedom (ADR-0011).
pub trait WireCodec {
    fn codec_id(&self) -> u16;
    fn codec_version(&self) -> u16;
    /// Encoded bytes per hidden element (the unit of the wire-byte accounting).
    fn elem_bytes(&self) -> usize;
    /// Encode an activation vector, appending to `out`.
    fn encode(&self, x: &[f32], out: &mut Vec<u8>);
    /// Decode `bytes` back to an activation vector; length must be a whole
    /// number of elements.
    fn decode(&self, bytes: &[u8]) -> Result<Vec<f32>>;
}

/// fp8 E4M3 activations — the baseline wire codec (MANIFESTO §4.3).
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp8Codec;

impl WireCodec for Fp8Codec {
    fn codec_id(&self) -> u16 {
        FP8_CODEC_ID
    }
    fn codec_version(&self) -> u16 {
        1
    }
    fn elem_bytes(&self) -> usize {
        1
    }
    fn encode(&self, x: &[f32], out: &mut Vec<u8>) {
        out.reserve(x.len());
        for &v in x {
            out.push(fp8::f32_to_e4m3(v));
        }
    }
    fn decode(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        Ok(bytes.iter().map(|&b| fp8::e4m3_to_f32(b)).collect())
    }
}

/// bf16 activations — the lossless validation-reference codec. Both paths share
/// this identical encoding, so a matched-codec local≡node run is bit-exact by
/// construction (ADR-0018).
#[derive(Debug, Clone, Copy, Default)]
pub struct Bf16Codec;

impl WireCodec for Bf16Codec {
    fn codec_id(&self) -> u16 {
        BF16_CODEC_ID
    }
    fn codec_version(&self) -> u16 {
        1
    }
    fn elem_bytes(&self) -> usize {
        2
    }
    fn encode(&self, x: &[f32], out: &mut Vec<u8>) {
        out.reserve(2 * x.len());
        for &v in x {
            out.extend_from_slice(&bf16::f32_to_bf16(v).to_le_bytes());
        }
    }
    fn decode(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        if !bytes.len().is_multiple_of(2) {
            return Err(Error::parse(format!(
                "wire: bf16 payload is {} bytes, not a whole number of elements",
                bytes.len()
            )));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|c| bf16::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }
}

/// Resolve a codec from the ids carried in the handshake; unknown codec /
/// version is rejected loudly (ADR-0011: no per-link negotiation).
pub fn codec_for(id: u16, version: u16) -> Result<Box<dyn WireCodec>> {
    let codec: Box<dyn WireCodec> = match id {
        FP8_CODEC_ID => Box::new(Fp8Codec),
        BF16_CODEC_ID => Box::new(Bf16Codec),
        other => {
            return Err(Error::parse(format!(
                "wire: unknown codec id {other} (this build speaks fp8={FP8_CODEC_ID}, \
                 bf16={BF16_CODEC_ID})"
            )));
        }
    };
    if codec.codec_version() != version {
        return Err(Error::parse(format!(
            "wire: codec {id} version {version}, this build speaks {}",
            codec.codec_version()
        )));
    }
    Ok(codec)
}

// -------------------------------------------------------------------------
// Frames
// -------------------------------------------------------------------------

/// The connect handshake: peers agree on framing + codec + which model before
/// any dispatch flows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Handshake {
    pub wire_version: u16,
    pub codec_id: u16,
    pub codec_version: u16,
    /// Raw 32-byte manifest identity (blake3 of the canonical manifest bytes,
    /// ADR-0005) — the "which model" agreement.
    pub identity: [u8; 32],
}

impl Handshake {
    /// A handshake for this build's wire version and the given codec + model.
    pub fn new(codec: &dyn WireCodec, identity: [u8; 32]) -> Handshake {
        Handshake {
            wire_version: WIRE_VERSION,
            codec_id: codec.codec_id(),
            codec_version: codec.codec_version(),
            identity,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HANDSHAKE_LEN);
        out.extend_from_slice(&MAGIC_HANDSHAKE);
        out.extend_from_slice(&self.wire_version.to_le_bytes());
        out.extend_from_slice(&self.codec_id.to_le_bytes());
        out.extend_from_slice(&self.codec_version.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // pad
        out.extend_from_slice(&self.identity);
        debug_assert_eq!(out.len(), HANDSHAKE_LEN);
        out
    }

    /// Parse a handshake; rejects wrong magic, wire version, and nonzero pad.
    /// Codec / model agreement is a separate loud step, `verify`.
    pub fn decode(bytes: &[u8]) -> Result<Handshake> {
        if bytes.len() != HANDSHAKE_LEN {
            return Err(Error::parse(format!(
                "wire: handshake is {} bytes, expected {HANDSHAKE_LEN}",
                bytes.len()
            )));
        }
        if bytes[0..4] != MAGIC_HANDSHAKE {
            return Err(Error::parse("wire: bad handshake magic (not KNYW)"));
        }
        let wire_version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if wire_version != WIRE_VERSION {
            return Err(Error::parse(format!(
                "wire: handshake wire_version {wire_version}, this build speaks {WIRE_VERSION}"
            )));
        }
        if bytes[10..12] != [0, 0] {
            return Err(Error::parse("wire: nonzero handshake pad"));
        }
        let mut identity = [0u8; 32];
        identity.copy_from_slice(&bytes[12..44]);
        Ok(Handshake {
            wire_version,
            codec_id: u16::from_le_bytes([bytes[6], bytes[7]]),
            codec_version: u16::from_le_bytes([bytes[8], bytes[9]]),
            identity,
        })
    }

    /// Confirm the peer speaks our codec and serves our model, returning the
    /// agreed codec. Mismatched identity or codec is rejected loudly (ADR-0011).
    pub fn verify(&self, codec: &dyn WireCodec, identity: &[u8; 32]) -> Result<Box<dyn WireCodec>> {
        if &self.identity != identity {
            return Err(Error::parse(
                "wire: handshake manifest identity mismatch — peer is serving a different model",
            ));
        }
        if self.codec_id != codec.codec_id() || self.codec_version != codec.codec_version() {
            return Err(Error::parse(format!(
                "wire: handshake offers codec {}v{}, we speak {}v{}",
                self.codec_id,
                self.codec_version,
                codec.codec_id(),
                codec.codec_version()
            )));
        }
        codec_for(self.codec_id, self.codec_version)
    }
}

/// A dispatch: run `experts` of `layer` on the single activation `x` (encoded
/// codec bytes). x is sent once per node (A5 up-byte accounting).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dispatch {
    pub layer: u16,
    pub x: Vec<u8>,
    pub experts: Vec<u16>,
}

impl Dispatch {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let n = u16::try_from(self.experts.len())
            .map_err(|_| Error::parse("wire: too many experts in one dispatch (max 65535)"))?;
        let x_len = u32::try_from(self.x.len())
            .map_err(|_| Error::parse("wire: dispatch activation too large"))?;
        let mut out =
            Vec::with_capacity(DISPATCH_HEADER_LEN + self.x.len() + 2 * self.experts.len());
        out.extend_from_slice(&MAGIC_DISPATCH);
        out.extend_from_slice(&self.layer.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&x_len.to_le_bytes());
        out.extend_from_slice(&self.x);
        for &e in &self.experts {
            out.extend_from_slice(&e.to_le_bytes());
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Dispatch> {
        if bytes.len() < DISPATCH_HEADER_LEN {
            return Err(Error::parse(format!(
                "wire: dispatch is {} bytes, header needs {DISPATCH_HEADER_LEN}",
                bytes.len()
            )));
        }
        if bytes[0..4] != MAGIC_DISPATCH {
            return Err(Error::parse("wire: bad dispatch magic (not KNYD)"));
        }
        let layer = u16::from_le_bytes([bytes[4], bytes[5]]);
        let n_experts = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
        let x_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let want = DISPATCH_HEADER_LEN
            .checked_add(x_len)
            .and_then(|p| p.checked_add(2 * n_experts))
            .ok_or_else(|| Error::parse("wire: dispatch header implies an impossible size"))?;
        if bytes.len() != want {
            return Err(Error::parse(format!(
                "wire: dispatch is {} bytes, header implies exactly {want}",
                bytes.len()
            )));
        }
        let x = bytes[DISPATCH_HEADER_LEN..DISPATCH_HEADER_LEN + x_len].to_vec();
        let mut experts = Vec::with_capacity(n_experts);
        let mut p = DISPATCH_HEADER_LEN + x_len;
        for _ in 0..n_experts {
            experts.push(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
            p += 2;
        }
        Ok(Dispatch { layer, x, experts })
    }
}

/// Whether a node held and ran the expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertStatus {
    /// Held and computed; `y` is present.
    Ok = 0,
    /// Not held by this node; no `y` (feeds the spine's ADR-0008 renorm).
    NotHeld = 1,
}

impl ExpertStatus {
    fn from_u8(v: u8) -> Result<ExpertStatus> {
        match v {
            0 => Ok(ExpertStatus::Ok),
            1 => Ok(ExpertStatus::NotHeld),
            other => Err(Error::parse(format!("wire: unknown gather status {other}"))),
        }
    }
}

/// One expert's result. `y` is the encoded output (empty iff `NotHeld`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatherResult {
    pub expert: u16,
    pub status: ExpertStatus,
    pub y: Vec<u8>,
}

/// A gather: one record per requested expert. Each answered expert returns its
/// own `y` of `y_len` bytes (A5 down-byte accounting — NOT symmetric with up).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gather {
    pub layer: u16,
    /// Encoded bytes of one answered `y` (= hidden * codec_bytes).
    pub y_len: u32,
    pub results: Vec<GatherResult>,
}

impl Gather {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let n = u16::try_from(self.results.len())
            .map_err(|_| Error::parse("wire: too many gather results (max 65535)"))?;
        let y_len = self.y_len as usize;
        let mut out = Vec::with_capacity(GATHER_HEADER_LEN);
        out.extend_from_slice(&MAGIC_GATHER);
        out.extend_from_slice(&self.layer.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&self.y_len.to_le_bytes());
        for r in &self.results {
            match r.status {
                ExpertStatus::Ok if r.y.len() != y_len => {
                    return Err(Error::parse(format!(
                        "wire: gather ok result for expert {} has {} y bytes, header says {y_len}",
                        r.expert,
                        r.y.len()
                    )));
                }
                ExpertStatus::NotHeld if !r.y.is_empty() => {
                    return Err(Error::parse(format!(
                        "wire: gather not-held result for expert {} carries {} y bytes",
                        r.expert,
                        r.y.len()
                    )));
                }
                _ => {}
            }
            out.extend_from_slice(&r.expert.to_le_bytes());
            out.push(r.status as u8);
            if r.status == ExpertStatus::Ok {
                out.extend_from_slice(&r.y);
            }
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Gather> {
        if bytes.len() < GATHER_HEADER_LEN {
            return Err(Error::parse(format!(
                "wire: gather is {} bytes, header needs {GATHER_HEADER_LEN}",
                bytes.len()
            )));
        }
        if bytes[0..4] != MAGIC_GATHER {
            return Err(Error::parse("wire: bad gather magic (not KNYG)"));
        }
        let layer = u16::from_le_bytes([bytes[4], bytes[5]]);
        let n_results = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
        let y_len_u32 = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let y_len = y_len_u32 as usize;
        let mut results = Vec::with_capacity(n_results);
        let mut p = GATHER_HEADER_LEN;
        for _ in 0..n_results {
            if p + GATHER_RECORD_HEADER_LEN > bytes.len() {
                return Err(Error::parse("wire: gather truncated in a record header"));
            }
            let expert = u16::from_le_bytes([bytes[p], bytes[p + 1]]);
            let status = ExpertStatus::from_u8(bytes[p + 2])?;
            p += GATHER_RECORD_HEADER_LEN;
            let y = if status == ExpertStatus::Ok {
                if p + y_len > bytes.len() {
                    return Err(Error::parse("wire: gather truncated in a y payload"));
                }
                let y = bytes[p..p + y_len].to_vec();
                p += y_len;
                y
            } else {
                Vec::new()
            };
            results.push(GatherResult { expert, status, y });
        }
        if p != bytes.len() {
            return Err(Error::parse(format!(
                "wire: gather has {} trailing bytes after {n_results} records",
                bytes.len() - p
            )));
        }
        Ok(Gather {
            layer,
            y_len: y_len_u32,
            results,
        })
    }
}

// -------------------------------------------------------------------------
// Transport (ADR-0016 interim: sync, blocking, byte-counted)
// -------------------------------------------------------------------------

/// A blocking byte-counted transport over any `Read + Write` (a
/// `std::net::TcpStream` in production, an in-memory duplex in tests). Each
/// successfully completed send/recv advances `up` / `down` by its full length —
/// handshake and framing included — so BENCH numbers are measured at the wire,
/// not estimated. A send/recv that errors mid-transfer ends the session, so the
/// counters are exact for completed calls (the only counts BENCH consumes); they
/// are not a partial-progress meter for a failed, torn-down connection.
#[derive(Debug)]
pub struct Transport<S> {
    stream: S,
    /// Bytes written to the peer (spine -> node payload + framing).
    pub up: u64,
    /// Bytes read from the peer (node -> spine payload + framing).
    pub down: u64,
}

impl<S: Read + Write> Transport<S> {
    pub fn new(stream: S) -> Transport<S> {
        Transport {
            stream,
            up: 0,
            down: 0,
        }
    }

    /// The inner stream (e.g. to close or clone it) — counters keep their state.
    pub fn get_ref(&self) -> &S {
        &self.stream
    }

    fn write_counted(&mut self, buf: &[u8]) -> Result<()> {
        self.stream
            .write_all(buf)
            .map_err(|e| Error::parse(format!("wire: write failed: {e}")))?;
        self.up += buf.len() as u64;
        Ok(())
    }

    fn read_counted(&mut self, buf: &mut [u8]) -> Result<()> {
        self.stream
            .read_exact(buf)
            .map_err(|e| Error::parse(format!("wire: read failed: {e}")))?;
        self.down += buf.len() as u64;
        Ok(())
    }

    pub fn send_handshake(&mut self, h: &Handshake) -> Result<()> {
        self.write_counted(&h.encode())
    }

    pub fn recv_handshake(&mut self) -> Result<Handshake> {
        let mut buf = [0u8; HANDSHAKE_LEN];
        self.read_counted(&mut buf)?;
        Handshake::decode(&buf)
    }

    pub fn send_dispatch(&mut self, d: &Dispatch) -> Result<()> {
        self.write_counted(&d.encode()?)
    }

    /// Receive a dispatch. `expect_x_len` is the only legal activation size —
    /// `hidden * codec.elem_bytes()` from the negotiated handshake codec and the
    /// receiver's manifest. The header's `x_len` is checked against it BEFORE
    /// any buffer is allocated, so a hostile or corrupt length prefix (e.g.
    /// `x_len = 2_000_000_000`) is rejected without committing memory. The
    /// expert-id list is `u16`-bounded by `n_experts` and needs no extra cap.
    pub fn recv_dispatch(&mut self, expect_x_len: usize) -> Result<Dispatch> {
        let mut head = [0u8; DISPATCH_HEADER_LEN];
        self.read_counted(&mut head)?;
        if head[0..4] != MAGIC_DISPATCH {
            return Err(Error::parse("wire: bad dispatch magic (not KNYD)"));
        }
        let n_experts = u16::from_le_bytes([head[6], head[7]]) as usize;
        let x_len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
        if x_len != expect_x_len {
            return Err(Error::parse(format!(
                "wire: dispatch x_len {x_len}, negotiated codec + model imply exactly \
                 {expect_x_len} (refusing to allocate an untrusted length)"
            )));
        }
        let body_len = x_len
            .checked_add(2 * n_experts)
            .ok_or_else(|| Error::parse("wire: dispatch header implies an impossible size"))?;
        let mut frame = vec![0u8; DISPATCH_HEADER_LEN + body_len];
        frame[..DISPATCH_HEADER_LEN].copy_from_slice(&head);
        self.read_counted(&mut frame[DISPATCH_HEADER_LEN..])?;
        Dispatch::decode(&frame)
    }

    pub fn send_gather(&mut self, g: &Gather) -> Result<()> {
        self.write_counted(&g.encode()?)
    }

    /// Receive a gather. `expect_y_len` is the only legal per-answered-y size —
    /// `hidden * codec.elem_bytes()` from the negotiated handshake codec and the
    /// receiver's manifest. The header's `y_len` is checked against it BEFORE
    /// any record buffer is allocated, so a hostile `y_len = 2_000_000_000`
    /// prefix is rejected without committing ~2 GB of zeros. `n_results` is
    /// `u16`-bounded and each record's bytes must actually arrive on the socket,
    /// so no single allocation can outrun the payload.
    pub fn recv_gather(&mut self, expect_y_len: usize) -> Result<Gather> {
        let mut head = [0u8; GATHER_HEADER_LEN];
        self.read_counted(&mut head)?;
        if head[0..4] != MAGIC_GATHER {
            return Err(Error::parse("wire: bad gather magic (not KNYG)"));
        }
        let n_results = u16::from_le_bytes([head[6], head[7]]) as usize;
        let y_len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
        if y_len != expect_y_len {
            return Err(Error::parse(format!(
                "wire: gather y_len {y_len}, negotiated codec + model imply exactly \
                 {expect_y_len} (refusing to allocate an untrusted length)"
            )));
        }
        // Read records one at a time — a record's size depends on its status,
        // so we cannot know the frame length from the header alone.
        let mut frame = head.to_vec();
        for _ in 0..n_results {
            let base = frame.len();
            frame.resize(base + GATHER_RECORD_HEADER_LEN, 0);
            self.read_counted(&mut frame[base..])?;
            if ExpertStatus::from_u8(frame[base + 2])? == ExpertStatus::Ok {
                let ybase = frame.len();
                frame.resize(ybase + y_len, 0);
                self.read_counted(&mut frame[ybase..])?;
            }
        }
        Gather::decode(&frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    // --- codecs -----------------------------------------------------------

    #[test]
    fn codec_roundtrip() {
        let x = [1.0f32, -0.5, 0.0, 3.25, -2.0];
        for codec in [&Fp8Codec as &dyn WireCodec, &Bf16Codec as &dyn WireCodec] {
            let mut buf = Vec::new();
            codec.encode(&x, &mut buf);
            assert_eq!(buf.len(), x.len() * codec.elem_bytes());
            let back = codec.decode(&buf).unwrap();
            // Both codecs are exact for these values (fp8-representable / bf16
            // upper-half exact), so the round-trip is bit-identical.
            assert_eq!(back, x);
        }
    }

    #[test]
    fn codec_registry_rejects_unknown() {
        assert_eq!(codec_for(FP8_CODEC_ID, 1).unwrap().codec_id(), FP8_CODEC_ID);
        assert_eq!(
            codec_for(BF16_CODEC_ID, 1).unwrap().codec_id(),
            BF16_CODEC_ID
        );
        assert!(codec_for(999, 1).is_err(), "unknown codec id");
        assert!(codec_for(FP8_CODEC_ID, 2).is_err(), "unknown codec version");
    }

    #[test]
    fn bf16_decode_rejects_odd_length() {
        assert!(Bf16Codec.decode(&[0u8; 3]).is_err());
    }

    // --- CONSENSUS GOLDENS ------------------------------------------------
    // These lock the canonical wire bytes (ADR-0011). A change here is a
    // PROTOCOL change: bump WIRE_VERSION (framing) or the codec_version
    // (activation bytes) + write the ADR + update these goldens, in one PR —
    // never a test edit alone. kenny-format-auditor enforces this.

    #[test]
    fn golden_fp8_activation_bytes() {
        // Canonical fp8 e4m3 encoding of a fixed activation vector.
        let x = [1.0f32, 0.5, -1.5, 448.0, 0.0, 240.0];
        let mut buf = Vec::new();
        Fp8Codec.encode(&x, &mut buf);
        assert_eq!(hex(&buf), "3830bc7e0077");
    }

    #[test]
    fn golden_bf16_activation_bytes() {
        let x = [1.0f32, -1.0, 2.5];
        let mut buf = Vec::new();
        Bf16Codec.encode(&x, &mut buf);
        assert_eq!(hex(&buf), "803f80bf2040");
    }

    #[test]
    fn golden_handshake_bytes() {
        let identity: [u8; 32] = std::array::from_fn(|i| i as u8);
        let h = Handshake::new(&Fp8Codec, identity);
        let bytes = h.encode();
        assert_eq!(bytes.len(), HANDSHAKE_LEN);
        assert_eq!(
            hex(&bytes),
            "4b4e59570100010001000000000102030405060708090a0b0c0d0e0f\
             101112131415161718191a1b1c1d1e1f"
        );
        assert_eq!(Handshake::decode(&bytes).unwrap(), h);
    }

    #[test]
    fn golden_dispatch_bytes() {
        let mut x = Vec::new();
        Fp8Codec.encode(&[1.0f32, 0.5, -1.5], &mut x);
        let d = Dispatch {
            layer: 7,
            x,
            experts: vec![2, 200],
        };
        let bytes = d.encode().unwrap();
        assert_eq!(hex(&bytes), "4b4e594407000200030000003830bc0200c800");
        assert_eq!(Dispatch::decode(&bytes).unwrap(), d);
    }

    #[test]
    fn golden_gather_bytes() {
        let mut y = Vec::new();
        Fp8Codec.encode(&[1.0f32, 0.5], &mut y);
        let g = Gather {
            layer: 7,
            y_len: 2,
            results: vec![
                GatherResult {
                    expert: 2,
                    status: ExpertStatus::Ok,
                    y,
                },
                GatherResult {
                    expert: 5,
                    status: ExpertStatus::NotHeld,
                    y: Vec::new(),
                },
            ],
        };
        let bytes = g.encode().unwrap();
        assert_eq!(hex(&bytes), "4b4e594707000200020000000200003830050001");
        assert_eq!(Gather::decode(&bytes).unwrap(), g);
    }

    // --- frame round-trips + validation -----------------------------------

    #[test]
    fn frame_roundtrips_in_memory() {
        let mut x = Vec::new();
        Bf16Codec.encode(&[0.25f32, -0.75, 1.0, 2.0], &mut x);
        let d = Dispatch {
            layer: 3,
            x,
            experts: vec![0, 1, 2, 3],
        };
        assert_eq!(Dispatch::decode(&d.encode().unwrap()).unwrap(), d);

        let mut y = Vec::new();
        Bf16Codec.encode(&[1.0f32, 2.0], &mut y);
        let g = Gather {
            layer: 3,
            y_len: 4,
            results: vec![
                GatherResult {
                    expert: 1,
                    status: ExpertStatus::Ok,
                    y: y.clone(),
                },
                GatherResult {
                    expert: 9,
                    status: ExpertStatus::NotHeld,
                    y: Vec::new(),
                },
                GatherResult {
                    expert: 4,
                    status: ExpertStatus::Ok,
                    y,
                },
            ],
        };
        assert_eq!(Gather::decode(&g.encode().unwrap()).unwrap(), g);
    }

    #[test]
    fn dispatch_decode_rejects_corruption() {
        let d = Dispatch {
            layer: 1,
            x: vec![1, 2, 3, 4],
            experts: vec![7],
        };
        let good = d.encode().unwrap();
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(Dispatch::decode(&bad).is_err(), "magic");
        let mut bad = good.clone();
        bad.pop();
        assert!(Dispatch::decode(&bad).is_err(), "truncated");
        let mut bad = good.clone();
        bad.push(0);
        assert!(Dispatch::decode(&bad).is_err(), "oversized");
        assert!(Dispatch::decode(&good[..8]).is_err(), "short header");
    }

    #[test]
    fn gather_encode_rejects_inconsistent_y() {
        let g = Gather {
            layer: 0,
            y_len: 4,
            results: vec![GatherResult {
                expert: 0,
                status: ExpertStatus::Ok,
                y: vec![1, 2], // header says 4
            }],
        };
        assert!(g.encode().is_err(), "ok y disagrees with y_len");
        let g = Gather {
            layer: 0,
            y_len: 4,
            results: vec![GatherResult {
                expert: 0,
                status: ExpertStatus::NotHeld,
                y: vec![1, 2, 3, 4], // not-held must carry none
            }],
        };
        assert!(g.encode().is_err(), "not-held carries y");
    }

    #[test]
    fn gather_decode_rejects_bad_status() {
        let mut y = Vec::new();
        Fp8Codec.encode(&[1.0f32], &mut y);
        let g = Gather {
            layer: 0,
            y_len: 1,
            results: vec![GatherResult {
                expert: 0,
                status: ExpertStatus::Ok,
                y,
            }],
        };
        let mut bad = g.encode().unwrap();
        // Flip the status byte (offset 12 + 2) to an unknown value.
        bad[GATHER_HEADER_LEN + 2] = 9;
        assert!(Gather::decode(&bad).is_err());
    }

    // --- handshake verify -------------------------------------------------

    #[test]
    fn handshake_verify_matches_and_rejects() {
        let id: [u8; 32] = std::array::from_fn(|i| i as u8);
        let h = Handshake::new(&Fp8Codec, id);
        assert!(h.verify(&Fp8Codec, &id).is_ok());

        let other = std::array::from_fn(|i| (i as u8) ^ 0xFF);
        assert!(h.verify(&Fp8Codec, &other).is_err(), "wrong model identity");
        assert!(h.verify(&Bf16Codec, &id).is_err(), "wrong codec");
    }

    #[test]
    fn handshake_decode_rejects_corruption() {
        let id = [0u8; 32];
        let good = Handshake::new(&Fp8Codec, id).encode();
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(Handshake::decode(&bad).is_err(), "magic");
        let mut bad = good.clone();
        bad[4] = 9; // wire_version
        assert!(Handshake::decode(&bad).is_err(), "wire_version");
        let mut bad = good.clone();
        bad[10] = 1; // pad
        assert!(Handshake::decode(&bad).is_err(), "pad");
        assert!(
            Handshake::decode(&good[..HANDSHAKE_LEN - 1]).is_err(),
            "short"
        );
    }

    // --- transport byte counters ------------------------------------------

    #[test]
    fn transport_counts_in_memory_bytes() {
        // A Cursor is Read+Write; drive send then read back to check counters.
        let mut t = Transport::new(Cursor::new(Vec::new()));
        let id = [1u8; 32];
        let hs = Handshake::new(&Fp8Codec, id);
        let mut x = Vec::new();
        Fp8Codec.encode(&[1.0f32, 0.5, -1.5], &mut x);
        let disp = Dispatch {
            layer: 0,
            x: x.clone(),
            experts: vec![1, 2],
        };
        t.send_handshake(&hs).unwrap();
        t.send_dispatch(&disp).unwrap();
        let hs_len = hs.encode().len() as u64;
        let disp_len = disp.encode().unwrap().len() as u64;
        assert_eq!(t.up, hs_len + disp_len);
        assert_eq!(t.down, 0);

        // Rewind and read them back through a fresh transport.
        let mut buf = t.stream.into_inner();
        // Append a gather for the read side.
        let mut y = Vec::new();
        Fp8Codec.encode(&[2.0f32, 3.0, 4.0], &mut y);
        let gath = Gather {
            layer: 0,
            y_len: 3,
            results: vec![GatherResult {
                expert: 1,
                status: ExpertStatus::Ok,
                y,
            }],
        };
        buf.extend_from_slice(&gath.encode().unwrap());
        let mut r = Transport::new(Cursor::new(buf));
        assert_eq!(r.recv_handshake().unwrap(), hs);
        // 3 fp8 elements = 3 bytes per activation / answered y.
        assert_eq!(r.recv_dispatch(3).unwrap(), disp);
        assert_eq!(r.recv_gather(3).unwrap(), gath);
        assert_eq!(
            r.down,
            hs_len + disp_len + gath.encode().unwrap().len() as u64
        );
    }

    #[test]
    fn recv_rejects_oversized_length_prefix() {
        // A 12-byte KNYD header claiming a 2 GB activation: recv_dispatch must
        // reject it against the negotiated size BEFORE allocating any buffer,
        // so only the header (not ~2 GB of zeros) is ever touched. We assert on
        // the length-prefix error message to prove rejection is pre-allocation,
        // not a later truncation error.
        let mut d_head = Vec::new();
        d_head.extend_from_slice(&MAGIC_DISPATCH);
        d_head.extend_from_slice(&3u16.to_le_bytes()); // layer
        d_head.extend_from_slice(&1u16.to_le_bytes()); // n_experts
        d_head.extend_from_slice(&2_000_000_000u32.to_le_bytes()); // hostile x_len
        assert_eq!(d_head.len(), DISPATCH_HEADER_LEN);
        let mut t = Transport::new(Cursor::new(d_head));
        let err = format!("{}", t.recv_dispatch(8).unwrap_err());
        assert!(
            err.contains("x_len"),
            "expected a length-prefix rejection, got: {err}"
        );

        // Same allocation-DoS class for KNYG y_len.
        let mut g_head = Vec::new();
        g_head.extend_from_slice(&MAGIC_GATHER);
        g_head.extend_from_slice(&3u16.to_le_bytes()); // layer
        g_head.extend_from_slice(&1u16.to_le_bytes()); // n_results
        g_head.extend_from_slice(&2_000_000_000u32.to_le_bytes()); // hostile y_len
        assert_eq!(g_head.len(), GATHER_HEADER_LEN);
        let mut t = Transport::new(Cursor::new(g_head));
        let err = format!("{}", t.recv_gather(8).unwrap_err());
        assert!(
            err.contains("y_len"),
            "expected a length-prefix rejection, got: {err}"
        );
    }

    #[test]
    fn transport_loopback_socket_byte_counts() {
        // The literal socket path: the counters must equal the bytes actually
        // written / read on a real 127.0.0.1 stream (A5 — counted at the socket).
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let id = [7u8; 32];
        let hs = Handshake::new(&Bf16Codec, id);
        let mut x = Vec::new();
        Bf16Codec.encode(&[1.0f32, 2.0, 3.0, 4.0], &mut x);
        let disp = Dispatch {
            layer: 5,
            x,
            experts: vec![10, 20, 30],
        };
        let hs_c = hs.clone();
        let disp_c = disp.clone();

        // Node side: accept, read handshake + dispatch, answer with one gather.
        let server = thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            let mut t = Transport::new(sock);
            let got_hs = t.recv_handshake().unwrap();
            // 4 bf16 elements = 8 bytes per activation / answered y.
            let got_disp = t.recv_dispatch(8).unwrap();
            let mut y = Vec::new();
            Bf16Codec.encode(&[9.0f32, 8.0, 7.0, 6.0], &mut y);
            let g = Gather {
                layer: got_disp.layer,
                y_len: 8,
                results: vec![
                    GatherResult {
                        expert: got_disp.experts[0],
                        status: ExpertStatus::Ok,
                        y,
                    },
                    GatherResult {
                        expert: got_disp.experts[1],
                        status: ExpertStatus::NotHeld,
                        y: Vec::new(),
                    },
                ],
            };
            t.send_gather(&g).unwrap();
            // Node read exactly the handshake + dispatch; wrote the gather.
            assert_eq!(
                t.down,
                (got_hs.encode().len() + got_disp.encode().unwrap().len()) as u64
            );
            assert_eq!(t.up, g.encode().unwrap().len() as u64);
            g
        });

        let mut t = Transport::new(TcpStream::connect(addr).unwrap());
        t.send_handshake(&hs_c).unwrap();
        t.send_dispatch(&disp_c).unwrap();
        let g = t.recv_gather(8).unwrap();
        let g_expected = server.join().unwrap();
        assert_eq!(g, g_expected);

        // Client counters equal the exact bytes it wrote / read.
        assert_eq!(
            t.up,
            (hs_c.encode().len() + disp_c.encode().unwrap().len()) as u64
        );
        assert_eq!(t.down, g.encode().unwrap().len() as u64);

        // A5: framing is exactly accountable. Up = handshake + dispatch header
        // + x payload + expert-id list; the payload term is hidden*codec_bytes.
        let up_payload = disp_c.x.len() as u64; // hidden * codec_bytes, sent once
        let up_framing =
            HANDSHAKE_LEN as u64 + DISPATCH_HEADER_LEN as u64 + 2 * disp_c.experts.len() as u64;
        assert_eq!(t.up, up_payload + up_framing);
        // Down = gather header + per-record headers + one y per answered expert.
        let answered = g
            .results
            .iter()
            .filter(|r| r.status == ExpertStatus::Ok)
            .count();
        let down_payload = answered as u64 * g.y_len as u64;
        let down_framing =
            GATHER_HEADER_LEN as u64 + GATHER_RECORD_HEADER_LEN as u64 * g.results.len() as u64;
        assert_eq!(t.down, down_payload + down_framing);
    }
}
