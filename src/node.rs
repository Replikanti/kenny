//! `kenny node` — the expert-holding blob server (ADR-0004 / ADR-0013).
//!
//! A node loads a carve's manifest, indexes `(layer, expert) -> CID`, and
//! serves dispatches: for each requested expert it lazily mmaps the blob by
//! CID, reconstructs the `(gate, up, down)` matrices (`expert::reconstruct`),
//! runs the shared expert FFN (`expert::forward`) on CPU, and returns the
//! encoded `y`. Experts it does not hold answer `not-held`, which feeds the
//! spine's ADR-0008 renorm. Nothing stateful lives here — a node is a very
//! distributed hot cache of stateless pure functions (MANIFESTO §2).
//!
//! RAM stays bounded: blobs are memory-mapped on first use (the OS pages them
//! in and out), so the working set is the page cache plus one expert's
//! transient f32 buffers, never the whole carve. That is what lets the M1
//! two-process run share one box.
//!
//! Transport is the interim sync TCP of ADR-0016 via `crate::wire`: one
//! connection carries a handshake then a stream of dispatch/gather pairs. The
//! handshake agreement is verified loudly before any dispatch — the peer must
//! serve the SAME model (manifest identity, ADR-0005) and speak a codec this
//! build knows (ADR-0011). A recv error at a dispatch boundary is taken as the
//! peer having hung up: for the M1 trusted-localhost topology the session ends
//! and its stats are returned. The wire layer already rejects hostile length
//! prefixes before allocating (`recv_dispatch`), so a torn-down session leaks
//! neither memory nor state.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::blob;
use crate::error::{Error, Result};
use crate::expert;
use crate::manifest::{self, Manifest};
use crate::wire::{ExpertStatus, Gather, GatherResult, Transport, WireCodec, codec_for};

/// Per-connection serving stats — measured, so a run can be reported without
/// vibes (BENCH convention). Byte counts come straight off the wire counters.
#[derive(Debug, Default, Clone, Copy)]
pub struct ServeStats {
    pub dispatches: u64,
    pub experts_ok: u64,
    pub experts_not_held: u64,
    /// Bytes read from the spine (handshake + dispatch framing + payload).
    pub bytes_in: u64,
    /// Bytes written to the spine (gather framing + payload).
    pub bytes_out: u64,
}

/// An expert-holding node backed by one carve directory.
pub struct Node {
    manifest: Manifest,
    /// Raw 32-byte manifest identity (blake3 of the canonical bytes, ADR-0005).
    identity: [u8; 32],
    blobs_dir: PathBuf,
    /// `(layer, expert) -> CID` for every expert the manifest lists. An expert
    /// absent here is answered `not-held`.
    index: HashMap<(u16, u16), String>,
    /// Lazily populated `CID -> mmapped blob bytes`; each blob is hashed against
    /// its CID exactly once, on first map (integrity check for pool-fetched
    /// bytes), then reused across dispatches.
    cache: HashMap<String, Mmap>,
}

impl Node {
    /// Load a node from a carved directory (`manifest.json` + `blobs/`).
    pub fn load(carved_dir: &Path) -> Result<Node> {
        let manifest = Manifest::load(&carved_dir.join(manifest::FILE_NAME))?;
        let identity = *blake3::hash(&manifest.canonical_bytes()).as_bytes();
        let mut index = HashMap::with_capacity(manifest.experts.len());
        for e in &manifest.experts {
            index.insert((e.layer, e.expert), e.cid.clone());
        }
        Ok(Node {
            blobs_dir: carved_dir.join("blobs"),
            manifest,
            identity,
            index,
            cache: HashMap::new(),
        })
    }

    /// Number of experts this node holds.
    pub fn held(&self) -> usize {
        self.index.len()
    }

    /// The mmapped bytes of a blob by CID, mapping (and CID-verifying) on first
    /// use. Returns a borrow into the cache, valid until the next `&mut self`.
    fn blob_bytes(&mut self, cid: &str) -> Result<&[u8]> {
        if !self.cache.contains_key(cid) {
            let path = self.blobs_dir.join(blob::rel_path(cid));
            let file = File::open(&path).map_err(|e| Error::io(&path, e))?;
            // SAFETY: the blob file is read-only for the node's lifetime; the
            // same discipline safetensors shards are mapped under (src/safetensors.rs).
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(&path, e))?;
            if blob::cid(&mmap) != cid {
                return Err(Error::parse(format!(
                    "node: blob {cid} does not hash to its CID — corrupt store"
                )));
            }
            self.cache.insert(cid.to_string(), mmap);
        }
        Ok(&self.cache[cid])
    }

    /// Run one expert on `x`, returning its encoded `y`, or `None` if this node
    /// does not hold that `(layer, expert)`.
    fn run_expert(
        &mut self,
        layer: u16,
        expert: u16,
        x: &[f32],
        codec: &dyn WireCodec,
    ) -> Result<Option<Vec<u8>>> {
        let cid = match self.index.get(&(layer, expert)) {
            Some(c) => c.clone(),
            None => return Ok(None),
        };
        // Copy the Copy dims out before borrowing the cache (disjoint-field
        // access does not reach across the blob_bytes method boundary).
        let hidden = self.manifest.model.hidden as usize;
        let inter = self.manifest.model.inter as usize;
        let dtype = self.manifest.model.dtype;
        let (gate, up, down) = {
            let bytes = self.blob_bytes(&cid)?;
            let d = blob::decode(bytes)?;
            if (d.header.layer, d.header.expert) != (layer, expert)
                || (d.header.hidden as usize, d.header.inter as usize) != (hidden, inter)
                || d.header.dtype != dtype
            {
                return Err(Error::parse(format!(
                    "node: blob header disagrees with the manifest for (layer {layer}, expert {expert})"
                )));
            }
            expert::reconstruct(&d)?
        };
        let mut y = vec![0f32; hidden];
        expert::forward(&gate, &up, &down, hidden, x, &mut y);
        let mut yb = Vec::with_capacity(hidden * codec.elem_bytes());
        codec.encode(&y, &mut yb);
        Ok(Some(yb))
    }

    /// Serve one connection: verify the handshake, then answer dispatches until
    /// the peer hangs up. Returns the session stats.
    pub fn serve_connection<S: Read + Write>(&mut self, stream: S) -> Result<ServeStats> {
        let mut t = Transport::new(stream);
        let hs = t.recv_handshake()?;
        if hs.identity != self.identity {
            return Err(Error::parse(
                "node: handshake manifest identity mismatch — spine is serving a different model",
            ));
        }
        let codec = codec_for(hs.codec_id, hs.codec_version)?;
        let hidden = self.manifest.model.hidden as usize;
        let elem = codec.elem_bytes();
        let expect_x_len = hidden * elem;
        let y_len = (hidden * elem) as u32;

        let mut stats = ServeStats::default();
        // A recv error ends the loop: no more dispatches (clean hang-up on the
        // trusted localhost link, or a rejected frame) — the session is over.
        while let Ok(dispatch) = t.recv_dispatch(expect_x_len) {
            let x = codec.decode(&dispatch.x)?;
            let mut results = Vec::with_capacity(dispatch.experts.len());
            for &expert in &dispatch.experts {
                match self.run_expert(dispatch.layer, expert, &x, codec.as_ref())? {
                    Some(y) => {
                        stats.experts_ok += 1;
                        results.push(GatherResult {
                            expert,
                            status: ExpertStatus::Ok,
                            y,
                        });
                    }
                    None => {
                        stats.experts_not_held += 1;
                        results.push(GatherResult {
                            expert,
                            status: ExpertStatus::NotHeld,
                            y: Vec::new(),
                        });
                    }
                }
            }
            t.send_gather(&Gather {
                layer: dispatch.layer,
                y_len,
                results,
            })?;
            stats.dispatches += 1;
        }
        // The wire counters are labeled from the spine's perspective (up =
        // spine -> node); on the node side that flips — what we read is the
        // dispatch stream, what we wrote is the gather stream.
        stats.bytes_in = t.down;
        stats.bytes_out = t.up;
        Ok(stats)
    }
}

/// Bind, announce the address, and serve forever — the `kenny node` entry point.
///
/// A5: the node binds the requested address (default `127.0.0.1:0`, an
/// OS-assigned free port) and prints `listening <addr>` to stdout, flushed, so a
/// launcher can discover the port before the spine connects. A fixed port is
/// flaky under CI concurrency, hence the ephemeral default.
pub fn serve(carved_dir: &Path, listen: &str) -> Result<()> {
    let mut node = Node::load(carved_dir)?;
    let listener = TcpListener::bind(listen)
        .map_err(|e| Error::parse(format!("node: cannot bind {listen}: {e}")))?;
    let addr = listener
        .local_addr()
        .map_err(|e| Error::parse(format!("node: local_addr failed: {e}")))?;
    println!("listening {addr}");
    // stdout is block-buffered when piped; flush so the launcher sees the
    // address before `accept()` blocks.
    io::stdout().flush().ok();
    eprintln!(
        "node: holding {} experts across {} MoE layers (identity {})",
        node.held(),
        node.manifest.model.moe_layers,
        node.manifest.identity()
    );

    for stream in listener.incoming() {
        match stream {
            Ok(sock) => match node.serve_connection(sock) {
                Ok(s) => eprintln!(
                    "node: session served {} dispatches ({} ok, {} not-held, in {} B, out {} B)",
                    s.dispatches, s.experts_ok, s.experts_not_held, s.bytes_in, s.bytes_out
                ),
                Err(e) => eprintln!("node: session error: {e}"),
            },
            Err(e) => eprintln!("node: accept failed: {e}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::Dtype;
    use crate::carve::{self, Options};
    use crate::fixture::{self, Params};
    use crate::wire::{Bf16Codec, Dispatch, Fp8Codec, Handshake, WIRE_VERSION};
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("kenny-node-{name}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Build the default fixture and a bf16 carve; return the carved dir.
    fn carved(root: &Path) -> PathBuf {
        let model = root.join("model");
        fixture::generate(&Params::default(), &model).unwrap();
        let out = root.join("carved");
        carve::run(
            &model,
            &Options {
                out: out.clone(),
                model_name: "fixture".into(),
                model_rev: String::new(),
                dtype: Dtype::Bf16,
            },
        )
        .unwrap();
        out
    }

    fn identity_of(carved_dir: &Path) -> [u8; 32] {
        let m = Manifest::load(&carved_dir.join(manifest::FILE_NAME)).unwrap();
        *blake3::hash(&m.canonical_bytes()).as_bytes()
    }

    #[test]
    fn loads_fixture_and_indexes_every_expert() {
        let out = carved(&tmp("load"));
        let node = Node::load(&out).unwrap();
        assert_eq!(node.held(), 8, "2 MoE layers x 4 experts");
        assert!(node.index.contains_key(&(0, 0)));
        assert!(node.index.contains_key(&(1, 3)));
        assert!(!node.index.contains_key(&(0, 99)), "unheld expert absent");
    }

    #[test]
    fn dispatch_matches_local_forward_and_reports_not_held() {
        let out = carved(&tmp("dispatch"));
        let manifest = Manifest::load(&out.join(manifest::FILE_NAME)).unwrap();
        let hidden = manifest.model.hidden as usize;
        let id = identity_of(&out);
        // Deterministic activation.
        let x_raw: Vec<f32> = (0..hidden).map(|k| (k as f32) * 0.3 - 1.0).collect();

        for codec in [&Fp8Codec as &dyn WireCodec, &Bf16Codec as &dyn WireCodec] {
            // Encode x once; both the node and the local reference decode the
            // SAME bytes, so the forward input is identical (mirrors the wire).
            let mut xb = Vec::new();
            codec.encode(&x_raw, &mut xb);
            let x = codec.decode(&xb).unwrap();

            // Local reference for a held expert, with the identical codec
            // round-trip applied to y that the wire path applies.
            let want = |layer: u16, expert: u16| -> Vec<f32> {
                let cid = manifest
                    .experts
                    .iter()
                    .find(|e| e.layer == layer && e.expert == expert)
                    .unwrap()
                    .cid
                    .clone();
                let bytes = std::fs::read(out.join("blobs").join(blob::rel_path(&cid))).unwrap();
                let d = blob::decode(&bytes).unwrap();
                let (g, u, dn) = expert::reconstruct(&d).unwrap();
                let mut y = vec![0f32; hidden];
                expert::forward(&g, &u, &dn, hidden, &x, &mut y);
                let mut yb = Vec::new();
                codec.encode(&y, &mut yb);
                codec.decode(&yb).unwrap()
            };
            let want1 = want(0, 1);

            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let out_c = out.clone();
            let server = thread::spawn(move || {
                let mut node = Node::load(&out_c).unwrap();
                let (sock, _) = listener.accept().unwrap();
                node.serve_connection(sock).unwrap()
            });

            let mut t = Transport::new(TcpStream::connect(addr).unwrap());
            t.send_handshake(&Handshake::new(codec, id)).unwrap();
            // Expert 1 is held; expert 99 is not.
            let d = Dispatch {
                layer: 0,
                x: xb.clone(),
                experts: vec![1, 99],
            };
            t.send_dispatch(&d).unwrap();
            let g = t.recv_gather(hidden * codec.elem_bytes()).unwrap();
            drop(t); // hang up so the node's serve loop ends
            let stats = server.join().unwrap();

            assert_eq!(g.layer, 0);
            assert_eq!(g.results.len(), 2);
            assert_eq!(g.results[0].expert, 1);
            assert_eq!(g.results[0].status, ExpertStatus::Ok);
            assert_eq!(
                codec.decode(&g.results[0].y).unwrap(),
                want1,
                "held expert y matches the local forward through the same codec"
            );
            assert_eq!(g.results[1].expert, 99);
            assert_eq!(g.results[1].status, ExpertStatus::NotHeld);
            assert!(g.results[1].y.is_empty(), "not-held carries no y");

            assert_eq!(stats.dispatches, 1);
            assert_eq!(stats.experts_ok, 1);
            assert_eq!(stats.experts_not_held, 1);
        }
    }

    #[test]
    fn rejects_mismatched_manifest_identity() {
        let out = carved(&tmp("wrongid"));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let out_c = out.clone();
        let server = thread::spawn(move || {
            let mut node = Node::load(&out_c).unwrap();
            let (sock, _) = listener.accept().unwrap();
            node.serve_connection(sock)
        });

        let mut t = Transport::new(TcpStream::connect(addr).unwrap());
        t.send_handshake(&Handshake::new(&Fp8Codec, [0xAB; 32]))
            .unwrap();
        assert!(
            server.join().unwrap().is_err(),
            "node must reject a peer serving a different model"
        );
    }

    #[test]
    fn rejects_unknown_codec() {
        let out = carved(&tmp("badcodec"));
        let id = identity_of(&out);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let out_c = out.clone();
        let server = thread::spawn(move || {
            let mut node = Node::load(&out_c).unwrap();
            let (sock, _) = listener.accept().unwrap();
            node.serve_connection(sock)
        });

        let mut t = Transport::new(TcpStream::connect(addr).unwrap());
        let hs = Handshake {
            wire_version: WIRE_VERSION,
            codec_id: 999,
            codec_version: 1,
            identity: id,
        };
        t.send_handshake(&hs).unwrap();
        assert!(
            server.join().unwrap().is_err(),
            "node must reject a codec it does not speak"
        );
    }
}
