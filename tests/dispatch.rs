//! M1 dispatch/gather equivalence: the distributed (`NodeDispatch`) path must
//! reproduce the in-process (`LocalDispatch`) path BIT-FOR-BIT under a matched
//! codec. That protocol self-consistency is the M1 correctness gate (ADR-0020):
//! it isolates wire/protocol faithfulness from codec lossiness and holds
//! regardless of whether the spine's attention matches HuggingFace logits
//! (model-quality validation is the deferred perplexity canary, ADR-0008).
//!
//! Everything runs on the synthetic fixture — CI never downloads a model. The
//! fixture's attention is square, so the spine loads at num_heads =
//! num_kv_heads = 1, head_dim = hidden (A4); GQA head repetition is unit-tested
//! in `src/spine.rs`. The real-model two-process run is `KENNY_MODEL_DIR`-gated
//! and lives in a later PR (S7).

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;

use kenny::blob::Dtype;
use kenny::carve::{self, Options};
use kenny::fixture::{self, Params};
use kenny::manifest::{self, Manifest};
use kenny::node::Node;
use kenny::spine::{self, Config, LocalDispatch, NodeDispatch, Spine, eps_from_ppm};
use kenny::wire::{
    Bf16Codec, DISPATCH_HEADER_LEN, Fp8Codec, GATHER_HEADER_LEN, GATHER_RECORD_HEADER_LEN,
    HANDSHAKE_LEN, WireCodec,
};

fn tmp(name: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn fixture_and_carve(root: &Path) -> (PathBuf, PathBuf) {
    let model = root.join("model");
    fixture::generate(&Params::default(), &model).unwrap();
    let carved = root.join("carved");
    carve::run(
        &model,
        &Options {
            out: carved.clone(),
            model_name: "fixture".into(),
            model_rev: String::new(),
            dtype: Dtype::Bf16,
        },
    )
    .unwrap();
    (model, carved)
}

/// Fixture-shaped config: square attention forces a single head with
/// head_dim = hidden; top_k defaults to 2 of the fixture's 4 experts.
fn config(hidden: usize, top_k: usize) -> Config {
    Config {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: hidden,
        rope_theta: 1_000_000.0,
        rms_eps: eps_from_ppm(1),
        top_k,
    }
}

fn make_codec(which: &str) -> Box<dyn WireCodec> {
    match which {
        "fp8" => Box::new(Fp8Codec),
        "bf16" => Box::new(Bf16Codec),
        other => panic!("unknown codec {other}"),
    }
}

/// Generate through an in-process node, optionally having it drop experts
/// (simulating lost replicas, for the renorm path).
fn via_local(
    spine: &Spine,
    carved: &Path,
    which: &str,
    prompt: &[u32],
    tokens: usize,
    forget: &[(u16, u16)],
) -> (Vec<u32>, spine::GenStats) {
    let mut d = LocalDispatch::new(carved, make_codec(which)).unwrap();
    for &(l, e) in forget {
        assert!(d.node_mut().drop_expert(l, e), "expert to drop must exist");
    }
    spine.generate(&mut d, prompt, tokens).unwrap()
}

/// Generate through a real TCP node in a background thread. The node's manifest
/// identity and hidden size are derived from the carve, keeping the arg list lean.
fn via_node(
    spine: &Spine,
    carved: &Path,
    which: &str,
    prompt: &[u32],
    tokens: usize,
    forget: &[(u16, u16)],
) -> (Vec<u32>, spine::GenStats) {
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let identity = *blake3::hash(&manifest.canonical_bytes()).as_bytes();
    let hidden = manifest.model.hidden as usize;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let carved = carved.to_path_buf();
    let forget = forget.to_vec();
    let server = thread::spawn(move || {
        let mut node = Node::load(&carved).unwrap();
        for &(l, e) in &forget {
            assert!(node.drop_expert(l, e), "expert to drop must exist");
        }
        let (sock, _) = listener.accept().unwrap();
        node.serve_connection(sock).unwrap()
    });

    let mut dispatch =
        NodeDispatch::connect(&addr.to_string(), make_codec(which), identity, hidden).unwrap();
    let out = spine.generate(&mut dispatch, prompt, tokens).unwrap();
    drop(dispatch); // hang up so the node's serve loop ends
    server.join().unwrap();
    out
}

/// THE M1 GATE: local ≡ node, bit-for-bit, under BOTH codecs.
#[test]
fn local_equals_node_bit_exact() {
    let root = tmp("equiv");
    let (model, carved) = fixture_and_carve(&root);
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let spine = Spine::load(&model, &manifest, config(hidden, 2)).unwrap();
    let prompt = [1u32, 2, 3];

    for which in ["fp8", "bf16"] {
        let (local, _) = via_local(&spine, &carved, which, &prompt, 6, &[]);
        let (node, _) = via_node(&spine, &carved, which, &prompt, 6, &[]);
        assert_eq!(
            local, node,
            "codec {which}: dispatched path must reproduce the in-process path bit-for-bit"
        );
    }
}

/// Wire bytes are counted at the socket and accountable PER DIRECTION with exact
/// framing (A5): x is sent once per dispatch (up), each answering expert returns
/// its own y (down) — the two directions are NOT symmetric.
#[test]
fn wire_bytes_match_per_direction_accounting() {
    let root = tmp("wirebytes");
    let (model, carved) = fixture_and_carve(&root);
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let spine = Spine::load(&model, &manifest, config(hidden, 2)).unwrap();
    let prompt = [1u32, 2, 3];
    let tokens = 6usize;

    for which in ["fp8", "bf16"] {
        let (_out, stats) = via_node(&spine, &carved, which, &prompt, tokens, &[]);

        let elem = make_codec(which).elem_bytes();
        let k = spine.experts_per_step(); // experts requested per dispatch
        let moe_layers = manifest.model.moe_layers as u64;
        // One forward per prompt token + (tokens - 1) generation forwards.
        let forwards = (prompt.len() + tokens - 1) as u64;
        let dispatches = forwards * moe_layers;
        assert_eq!(
            stats.dispatches, dispatches,
            "codec {which}: dispatch count"
        );

        let payload = (hidden * elem) as u64; // one encoded activation / y
        // Up = one handshake + per-dispatch (header + x sent once + expert-id list).
        let up = HANDSHAKE_LEN as u64
            + dispatches * (DISPATCH_HEADER_LEN as u64 + payload + 2 * k as u64);
        // Down = per-dispatch (gather header + per-record header * k + one y per
        // ANSWERED expert). All experts present here, so answered == k.
        let down = dispatches
            * (GATHER_HEADER_LEN as u64
                + GATHER_RECORD_HEADER_LEN as u64 * k as u64
                + payload * k as u64);
        assert_eq!(stats.wire_up, up, "codec {which}: up bytes");
        assert_eq!(stats.wire_down, down, "codec {which}: down bytes");
        // The asymmetry is real: down carries k payloads per dispatch, up one.
        assert!(stats.wire_down > stats.wire_up, "codec {which}: down > up");
    }
}

/// ADR-0008 renorm end-to-end: with selected experts missing on BOTH paths, the
/// `not-held` gather status flows through the spine's renorm and the two paths
/// STILL agree bit-for-bit — the property this integration test exists for.
/// top_k = experts so every replica is routed every step, and layer 0 loses
/// three of its four replicas, so the renorm fires on every forward. The
/// *numeric* renorm formula (that dropping an expert changes the mixed output)
/// is proven deterministically in `src/spine.rs::moe_renorms_over_answered_subset`
/// — the fixture's argmax margins are wide enough that a real renorm need not
/// flip a greedy token, so a token-level "differs" assertion here would be flaky.
#[test]
fn renorm_over_answered_subset() {
    let root = tmp("renorm");
    let (model, carved) = fixture_and_carve(&root);
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let experts = manifest.model.experts_per_layer as usize;
    let spine = Spine::load(&model, &manifest, config(hidden, experts)).unwrap();
    let prompt = [1u32, 2, 3];
    // Three of layer 0's four replicas are down; only expert 3 survives there.
    let lost = [(0u16, 0u16), (0, 1), (0, 2)];

    let (_all, all_stats) = via_local(&spine, &carved, "bf16", &prompt, 6, &[]);
    assert_eq!(all_stats.renorm_steps, 0, "nothing missing -> no renorm");

    let (drop_local, drop_stats) = via_local(&spine, &carved, "bf16", &prompt, 6, &lost);
    let (drop_node, _) = via_node(&spine, &carved, "bf16", &prompt, 6, &lost);

    assert_eq!(drop_local, drop_node, "renorm path: local must equal node");
    // The run is observably a renorm run (all-present had zero renorm steps).
    assert!(drop_stats.renorm_steps > 0, "renorm must have fired");
    assert!(
        drop_stats.experts_answered < drop_stats.experts_requested,
        "an expert was requested but not answered"
    );
    assert!(
        drop_local.iter().all(|&t| (t as usize) < spine.vocab()),
        "renormed output stays finite / in-vocab"
    );
}

// REGRESSION LOCK — end-to-end spine forward determinism. This is NOT a
// consensus surface (contrast the KNYW/KNYD/KNYG + codec goldens in
// src/wire.rs): it may be re-baselined whenever the spine math legitimately
// changes (e.g. the A1 router order, RoPE, attention) with NO ADR / wire_version
// / codec_version event. It only catches UNINTENDED drift in the forward pass.
const GOLDEN_SPINE_TOKENS: &[u32] = &[1, 2, 3, 7, 26, 26, 26, 26, 26];

#[test]
fn golden_token_sequence_is_stable() {
    let root = tmp("golden");
    let (model, carved) = fixture_and_carve(&root);
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let spine = Spine::load(&model, &manifest, config(hidden, 2)).unwrap();
    // Fixed prompt / flags / codec -> a fixed greedy token sequence.
    let (out, _) = via_local(&spine, &carved, "fp8", &[1, 2, 3], 6, &[]);
    assert_eq!(
        out, GOLDEN_SPINE_TOKENS,
        "spine forward drifted; if intentional, re-baseline this REGRESSION LOCK"
    );
}

/// The literal "two real processes over localhost" gate: spawn `kenny node`,
/// read its OS-assigned address off stdout, then run `kenny spine` against it.
#[test]
fn two_process_cli_smoke() {
    let root = tmp("cli");
    let (model, carved) = fixture_and_carve(&root);
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let bin = env!("CARGO_BIN_EXE_kenny");

    let mut node = Command::new(bin)
        .args([
            "node",
            "--carved",
            carved.to_str().unwrap(),
            "--listen",
            "127.0.0.1:0",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    // The node prints `listening <addr>` on stdout before it blocks on accept.
    let mut reader = BufReader::new(node.stdout.take().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let addr = line
        .trim()
        .strip_prefix("listening ")
        .expect("node must print 'listening <addr>' on stdout")
        .to_string();

    let out = Command::new(bin)
        .args([
            "spine",
            "--carved",
            carved.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--node",
            &addr,
            "--tokens",
            "3",
            "--prompt",
            "1,2",
            "--codec",
            "fp8",
            "--num-heads",
            "1",
            "--num-kv-heads",
            "1",
            "--head-dim",
            &hidden.to_string(),
            "--top-k",
            "2",
            "--rope-theta",
            "1000000",
            "--rms-eps-ppm",
            "1",
        ])
        .output()
        .unwrap();

    let _ = node.kill();
    let _ = node.wait();

    assert!(
        out.status.success(),
        "kenny spine failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("tok/s"), "stats missing tok/s:\n{stdout}");
    assert!(
        stdout.contains("wire:"),
        "stats missing wire bytes:\n{stdout}"
    );
}
