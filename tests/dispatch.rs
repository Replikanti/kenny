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

/// Batched twin of `via_local`: advance B streams in lockstep through an
/// in-process node, optionally dropping experts (lost replicas, renorm path).
fn via_local_batch(
    spine: &Spine,
    carved: &Path,
    which: &str,
    prompts: &[&[u32]],
    tokens: usize,
    forget: &[(u16, u16)],
) -> (Vec<Vec<u32>>, spine::GenStats) {
    let mut d = LocalDispatch::new(carved, make_codec(which)).unwrap();
    for &(l, e) in forget {
        assert!(d.node_mut().drop_expert(l, e), "expert to drop must exist");
    }
    spine.generate_batch(&mut d, prompts, tokens).unwrap()
}

/// Batched twin of `via_node`: B streams over ONE TCP connection to a real node
/// in a background thread. The batched `NodeDispatch` pipelines the B round-trips
/// (ADR-0023); the node's serve loop is unchanged (a faster-arriving stream of
/// the same dispatch/gather frames).
fn via_node_batch(
    spine: &Spine,
    carved: &Path,
    which: &str,
    prompts: &[&[u32]],
    tokens: usize,
    forget: &[(u16, u16)],
) -> (Vec<Vec<u32>>, spine::GenStats) {
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
    let out = spine
        .generate_batch(&mut dispatch, prompts, tokens)
        .unwrap();
    drop(dispatch); // hang up so the node's serve loop ends
    server.join().unwrap();
    out
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

/// THE M2 GATE: batched local ≡ batched node, bit-for-bit, under BOTH codecs.
/// The pipelined `NodeDispatch::dispatch_batch` (writer thread + concurrent
/// gather drain, ADR-0023) must reproduce the sequential `LocalDispatch` default
/// batch loop exactly — proving the composed-wire pipeline is output-faithful.
#[test]
fn local_equals_node_bit_exact_batched() {
    let root = tmp("equiv-batch");
    let (model, carved) = fixture_and_carve(&root);
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let spine = Spine::load(&model, &manifest, config(hidden, 2)).unwrap();
    // Four independent streams, equal length (rectangular batch).
    let prompts: Vec<&[u32]> = vec![&[1, 2, 3], &[3, 2, 1], &[2, 3, 1], &[1, 3, 2]];

    for which in ["fp8", "bf16"] {
        let (local, _) = via_local_batch(&spine, &carved, which, &prompts, 6, &[]);
        let (node, _) = via_node_batch(&spine, &carved, which, &prompts, 6, &[]);
        assert_eq!(
            local, node,
            "codec {which}: batched dispatched path must reproduce the in-process path bit-for-bit"
        );
    }
}

/// THE strong M2 invariant: batching is OUTPUT-INVARIANT. A batch of B independent
/// streams reproduces each stream generated ALONE (B = 1), token-for-token — the
/// property that makes `GOLDEN_SPINE_TOKENS` immune to the batch path. Any drift
/// here is a batching bug, not a legitimate spine-math change.
#[test]
fn batch_equals_serial() {
    let root = tmp("batch-serial");
    let (model, carved) = fixture_and_carve(&root);
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let spine = Spine::load(&model, &manifest, config(hidden, 2)).unwrap();
    let prompts: Vec<&[u32]> = vec![&[1, 2, 3], &[3, 2, 1], &[2, 3, 1]];
    let tokens = 6usize;

    // Each stream generated alone (single-stream B = 1 path).
    let solo: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| via_local(&spine, &carved, "fp8", p, tokens, &[]).0)
        .collect();

    // The whole batch together, both dispatch paths.
    let (batched_local, _) = via_local_batch(&spine, &carved, "fp8", &prompts, tokens, &[]);
    let (batched_node, _) = via_node_batch(&spine, &carved, "fp8", &prompts, tokens, &[]);

    assert_eq!(
        batched_local, solo,
        "batched local == each stream generated alone"
    );
    assert_eq!(
        batched_node, solo,
        "batched node == each stream generated alone"
    );
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

/// Batched wire accounting (M2): B streams compose as B independent frame pairs
/// per MoE layer per step (ADR-0023), so the M1 per-direction formula holds with
/// the dispatch count scaled by B — one handshake, then B x the single-stream
/// dispatch/gather framing. This is the byte-level proof that batching adds no
/// new wire shape.
#[test]
fn wire_bytes_match_per_direction_accounting_batched() {
    let root = tmp("wirebytes-batch");
    let (model, carved) = fixture_and_carve(&root);
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let spine = Spine::load(&model, &manifest, config(hidden, 2)).unwrap();
    let prompts: Vec<&[u32]> = vec![&[1, 2, 3], &[3, 2, 1], &[2, 3, 1], &[1, 3, 2]];
    let b = prompts.len() as u64;
    let tokens = 6usize;

    for which in ["fp8", "bf16"] {
        let (_out, stats) = via_node_batch(&spine, &carved, which, &prompts, tokens, &[]);

        let elem = make_codec(which).elem_bytes();
        let k = spine.experts_per_step() as u64;
        let moe_layers = manifest.model.moe_layers as u64;
        // Per-stream forwards; one dispatch FRAME per stream per MoE layer.
        let forwards = (prompts[0].len() + tokens - 1) as u64;
        let dispatches = b * forwards * moe_layers;
        assert_eq!(
            stats.dispatches, dispatches,
            "codec {which}: B x dispatch count"
        );

        let payload = (hidden * elem) as u64;
        // One handshake for the connection, then B x the M1 per-dispatch framing.
        let up = HANDSHAKE_LEN as u64 + dispatches * (DISPATCH_HEADER_LEN as u64 + payload + 2 * k);
        let down = dispatches
            * (GATHER_HEADER_LEN as u64 + GATHER_RECORD_HEADER_LEN as u64 * k + payload * k);
        assert_eq!(stats.wire_up, up, "codec {which}: batched up bytes");
        assert_eq!(stats.wire_down, down, "codec {which}: batched down bytes");
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

    let tokens = 6usize;
    let (_all, all_stats) = via_local(&spine, &carved, "bf16", &prompt, tokens, &[]);
    assert_eq!(all_stats.renorm_steps, 0, "nothing missing -> no renorm");

    let (drop_local, drop_stats) = via_local(&spine, &carved, "bf16", &prompt, tokens, &lost);
    let (drop_node, node_stats) = via_node(&spine, &carved, "bf16", &prompt, tokens, &lost);

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

    // Down-byte accounting for the answered<k (replica-loss) branch (PR #17
    // follow-up): a not-held expert returns a record HEADER but NO y payload, so
    // the down stream is exactly the all-present down MINUS the missing experts'
    // y payloads — the asymmetry the not-held path exists to produce. top_k =
    // experts here, so every one of the k replicas is routed every step; layer 0
    // holds only 1 of its 4 (three dropped), every other MoE layer holds all k.
    let elem = make_codec("bf16").elem_bytes();
    let payload = (hidden * elem) as u64; // one encoded y = hidden * codec_bytes
    let k = spine.experts_per_step() as u64;
    let moe_layers = manifest.model.moe_layers as u64;
    let forwards = (prompt.len() + tokens - 1) as u64;
    let dispatches = forwards * moe_layers;
    // Every dispatch carries the gather header + one record header per requested
    // expert (present or not); only ANSWERED experts add a y payload.
    let frame = dispatches * (GATHER_HEADER_LEN as u64 + GATHER_RECORD_HEADER_LEN as u64 * k);
    // Answered y payloads: 1 on layer 0 (only the surviving replica), k on each
    // of the remaining MoE layers.
    let answered_ys = forwards * (1 + k * (moe_layers - 1));
    let expect_down = frame + answered_ys * payload;
    assert_eq!(
        node_stats.wire_down, expect_down,
        "renorm down bytes: header per requested expert, y only for the answered ones"
    );
    // And it is strictly fewer down bytes than an all-present run would carry:
    // the (k - 1) dropped replicas of layer 0 each drop one y payload per forward.
    let all_present_down = frame + forwards * k * moe_layers * payload;
    assert!(
        node_stats.wire_down < all_present_down,
        "the replica loss must SHRINK the down stream by the missing y payloads"
    );
    assert_eq!(
        all_present_down - node_stats.wire_down,
        forwards * (k - 1) * payload,
        "the down-byte shortfall equals exactly the missing experts' y payloads"
    );
}

/// Batched renorm (M2): the ADR-0008 not-held path is PER STREAM. With layer 0
/// missing three of four replicas, every stream renorms every forward at layer 0
/// and nowhere else; batched local ≡ batched node, and the aggregate down-byte
/// shortfall is exactly the missing y payloads summed over all B streams.
#[test]
fn renorm_over_answered_subset_batched() {
    let root = tmp("renorm-batch");
    let (model, carved) = fixture_and_carve(&root);
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let experts = manifest.model.experts_per_layer as usize;
    let spine = Spine::load(&model, &manifest, config(hidden, experts)).unwrap();
    let prompts: Vec<&[u32]> = vec![&[1, 2, 3], &[3, 2, 1], &[2, 3, 1]];
    let b = prompts.len() as u64;
    // Three of layer 0's four replicas are down on the (single) node.
    let lost = [(0u16, 0u16), (0, 1), (0, 2)];
    let tokens = 6usize;

    let (drop_local, drop_stats) =
        via_local_batch(&spine, &carved, "bf16", &prompts, tokens, &lost);
    let (drop_node, node_stats) = via_node_batch(&spine, &carved, "bf16", &prompts, tokens, &lost);

    assert_eq!(
        drop_local, drop_node,
        "batched renorm path: local must equal node"
    );
    assert!(drop_stats.renorm_steps > 0, "renorm must have fired");
    assert!(
        drop_local
            .iter()
            .flatten()
            .all(|&t| (t as usize) < spine.vocab()),
        "renormed batched output stays finite / in-vocab"
    );

    let elem = make_codec("bf16").elem_bytes();
    let payload = (hidden * elem) as u64;
    let k = spine.experts_per_step() as u64;
    let moe_layers = manifest.model.moe_layers as u64;
    let forwards = (prompts[0].len() + tokens - 1) as u64;
    // Every stream renorms layer 0 on every forward, and only there.
    assert_eq!(
        drop_stats.renorm_steps,
        b * forwards,
        "one renorm/stream/forward at layer 0"
    );

    // Aggregate framing: B streams x (gather header + one record header per
    // requested expert) per dispatch; y payloads only for the answered experts.
    let frame = b
        * forwards
        * moe_layers
        * (GATHER_HEADER_LEN as u64 + GATHER_RECORD_HEADER_LEN as u64 * k);
    let answered_ys = b * forwards * (1 + k * (moe_layers - 1));
    let expect_down = frame + answered_ys * payload;
    assert_eq!(
        node_stats.wire_down, expect_down,
        "batched renorm down bytes: header per requested expert, y only for the answered ones"
    );
    // Strictly fewer down bytes than an all-present run: each stream drops (k-1)
    // layer-0 y payloads per forward.
    let all_present_down = frame + b * forwards * k * moe_layers * payload;
    assert_eq!(
        all_present_down - node_stats.wire_down,
        b * forwards * (k - 1) * payload,
        "the down-byte shortfall equals exactly the missing experts' y payloads, per stream"
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

// -------------------------------------------------------------------------
// S7 — the M1 exit criterion, against the REAL Qwen3-30B-A3B (KENNY_MODEL_DIR).
// CI never downloads a model, so this is gated; it is the demonstration the
// milestone closes on. Two things are proven: (1) the distributed fp8 path
// (fp8 blobs + fp8 wire) reproduces the in-process path bit-for-bit on the real
// model — the same protocol self-consistency gate the fixture proves, now with
// real numerics — and (2) an OUTPUT-SANITY cosine of the fp8-dispatched logits
// vs a reference forward that reads the ORIGINAL bf16 weights (no blob quant, no
// codec — the diff.rs::source_matrix path, A6), the first end-to-end ADR-0018
// signal that mirrors M0's fp8-vs-bf16 methodology.
// -------------------------------------------------------------------------

use kenny::spine::Dispatcher;
use kenny::{expert, quant, safetensors};
use std::collections::BTreeMap;

/// Reference dispatcher (A6): every expert reconstructed from the ORIGINAL bf16
/// model tensors, run with NO quantization and NO wire codec — the exact
/// `diff.rs::source_matrix` reference M0 measured fp8 against. It holds every
/// expert, so it never answers not-held. Shards are mmapped and cached.
struct SourceDispatch {
    dir: PathBuf,
    tensor_shard: BTreeMap<String, String>,
    shards: BTreeMap<String, safetensors::ShardFile>,
    hidden: usize,
    inter: usize,
}

impl SourceDispatch {
    fn new(model_dir: &Path, hidden: usize, inter: usize) -> SourceDispatch {
        let model = safetensors::open_model(model_dir).unwrap();
        let tensor_shard = model.weight_map.iter().cloned().collect();
        SourceDispatch {
            dir: model.dir,
            tensor_shard,
            shards: BTreeMap::new(),
            hidden,
            inter,
        }
    }

    fn matrix(&mut self, layer: u16, expert: u16, proj: &str) -> Vec<f32> {
        let name = format!("model.layers.{layer}.mlp.experts.{expert}.{proj}.weight");
        let shard_name = self.tensor_shard[&name].clone();
        let shard = self
            .shards
            .entry(shard_name.clone())
            .or_insert_with(|| safetensors::ShardFile::open(&self.dir.join(&shard_name)).unwrap());
        let meta = shard.tensor(&name).unwrap();
        quant::bf16_to_f32_vec(shard.bytes(meta)).unwrap()
    }
}

impl Dispatcher for SourceDispatch {
    fn dispatch(
        &mut self,
        layer: u16,
        x: &[f32],
        experts: &[u16],
    ) -> kenny::error::Result<Vec<Option<Vec<f32>>>> {
        let hidden = self.hidden;
        let mut out = Vec::with_capacity(experts.len());
        for &e in experts {
            let gate = self.matrix(layer, e, "gate_proj");
            let up = self.matrix(layer, e, "up_proj");
            let down = self.matrix(layer, e, "down_proj");
            let mut y = vec![0f32; hidden];
            expert::forward(&gate, &up, &down, hidden, x, &mut y);
            out.push(Some(y));
        }
        let _ = self.inter;
        Ok(out)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn real_model_two_process_dispatch() {
    let Some(dir) = std::env::var_os("KENNY_MODEL_DIR") else {
        eprintln!("KENNY_MODEL_DIR unset — skipping S7 real-model two-process run");
        return;
    };
    let model_dir = PathBuf::from(dir);

    // fp8 carve of the real model (the node's blob store). Re-runs dedup-skip,
    // so this is cheap on a warm carve.
    let carved = Path::new(env!("CARGO_TARGET_TMPDIR")).join("real-carve-fp8");
    std::fs::create_dir_all(&carved).unwrap();
    let t0 = std::time::Instant::now();
    let s = carve::run(
        &model_dir,
        &Options {
            out: carved.clone(),
            model_name: "qwen3-30b-a3b".into(),
            model_rev: String::new(),
            dtype: Dtype::Fp8,
        },
    )
    .unwrap();
    eprintln!(
        "S7 fp8 carve: {} blobs ({} new bytes, {} dedup) in {:.1?}",
        s.blobs,
        s.blob_bytes,
        s.dedup_skipped,
        t0.elapsed()
    );
    assert_eq!(s.blobs, 6144, "Qwen3-30B-A3B routed expert count");
    assert_eq!((s.moe_layers, s.experts_per_layer), (48, 128));

    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let inter = manifest.model.inter as usize;

    // Real Qwen3-30B-A3B hyperparameters (the model card == Config::default()).
    let cfg = Config::default();
    let t_load = std::time::Instant::now();
    let spine = Spine::load(&model_dir, &manifest, cfg).unwrap();
    eprintln!(
        "S7 spine load (always-on tensors): {:.1?}",
        t_load.elapsed()
    );

    let prompt = [40u32, 1207, 264, 3405]; // arbitrary in-vocab ids
    let gen_n = 8usize;

    // (1) THE GATE on the real model: fp8 local ≡ fp8 node, bit-for-bit.
    let (local, local_stats) = via_local(&spine, &carved, "fp8", &prompt, gen_n, &[]);
    let (node, node_stats) = via_node(&spine, &carved, "fp8", &prompt, gen_n, &[]);
    assert_eq!(
        local, node,
        "real model: dispatched fp8 path must reproduce the in-process path bit-for-bit"
    );
    assert!(
        node_stats.wire_up > 0 && node_stats.wire_down > 0,
        "wire moved"
    );

    let secs = node_stats.elapsed.as_secs_f64();
    let tok_s = node_stats.generated_tokens as f64 / secs;
    let wire_per_tok =
        (node_stats.wire_up + node_stats.wire_down) as f64 / node_stats.generated_tokens as f64;
    let (median, p99) = node_stats.latency_median_p99();
    eprintln!(
        "S7 two-process: {gen_n} tok in {secs:.3}s = {tok_s:.2} tok/s | per-forward median \
         {median:.1?} p99 {p99:.1?} | wire up {} B down {} B ({wire_per_tok:.0} B/tok) | local {:.3}s",
        node_stats.wire_up,
        node_stats.wire_down,
        local_stats.elapsed.as_secs_f64(),
    );

    // (2) OUTPUT SANITY (A6): fp8-dispatched logits vs a bf16-source reference
    // (no blob quant, no wire codec), teacher-forced on the same prompt.
    let mut fp8 = LocalDispatch::new(&carved, Box::new(Fp8Codec)).unwrap();
    let logits_fp8 = spine.logits(&mut fp8, &prompt).unwrap();
    let mut reference = SourceDispatch::new(&model_dir, hidden, inter);
    let logits_ref = spine.logits(&mut reference, &prompt).unwrap();
    let cos = cosine(&logits_fp8, &logits_ref);
    eprintln!("S7 output-sanity cosine (fp8-blob+fp8-wire vs bf16-source): {cos:.6}");
    // A loose floor only — this is a MEASURED signal for BENCH.md, not a tuned
    // gate; the exact number is what the milestone reports.
    assert!(
        cos > 0.9,
        "end-to-end fp8 forward degraded far past sanity: cosine {cos}"
    );
    // The greedy next token may or may not agree: a >0.999 cosine still perturbs
    // a 151936-way argmax when the top candidates are close, so token-level match
    // is NOT the quality gate (that is the deferred perplexity canary, ADR-0008).
    // Print both argmaxes as an observation, assert nothing about their equality.
    let arg = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    };
    eprintln!(
        "S7 next-token argmax: fp8 {} vs bf16-source {}",
        arg(&logits_fp8),
        arg(&logits_ref)
    );
}

// -------------------------------------------------------------------------
// M2 — localhost batching B-sweep against the REAL Qwen3-30B-A3B
// (KENNY_MODEL_DIR). The M2 deliverable is BENCH.md numbers, not a pass/fail:
// aggregate tok/s and per-STEP median/p99 as batch size B rises, plus exact
// per-direction wire bytes reconciled to the framing constants. Loopback
// topology (`spine ⇄ 127.0.0.1 ⇄ node`, the S7 harness): RTT≈0, so the §4.4
// per-layer barrier has nothing to amortize — the number recorded honestly
// baselines the real-LAN / M3 `tc netem` re-run where the amortization win
// appears. Gated: CI never downloads a model.
// -------------------------------------------------------------------------

/// Distinct short prompts, one per stream index (mirrors the CLI's seed-derived
/// batch prompts): stream `s` routes independently so B streams are genuinely
/// independent work, not B copies of one.
fn bench_prompts(b: usize, len: usize, vocab: usize, seed: u64) -> Vec<Vec<u32>> {
    (0..b)
        .map(|s| {
            let mut rng = kenny::rng::SplitMix64::for_name(seed, &format!("m2.bench.{s}"));
            (0..len)
                .map(|_| (rng.next_u64() % vocab as u64) as u32)
                .collect()
        })
        .collect()
}

#[test]
fn batch_sweep_localhost() {
    let Some(dir) = std::env::var_os("KENNY_MODEL_DIR") else {
        eprintln!("KENNY_MODEL_DIR unset — skipping M2 localhost batch sweep");
        return;
    };
    let model_dir = PathBuf::from(dir);

    // fp8 carve of the real model (shared with the S7 harness; re-runs dedup-skip
    // so a warm carve is cheap).
    let carved = Path::new(env!("CARGO_TARGET_TMPDIR")).join("real-carve-fp8");
    std::fs::create_dir_all(&carved).unwrap();
    let t0 = std::time::Instant::now();
    let s = carve::run(
        &model_dir,
        &Options {
            out: carved.clone(),
            model_name: "qwen3-30b-a3b".into(),
            model_rev: String::new(),
            dtype: Dtype::Fp8,
        },
    )
    .unwrap();
    eprintln!(
        "M2 fp8 carve: {} blobs ({} new bytes, {} dedup) in {:.1?}",
        s.blobs,
        s.blob_bytes,
        s.dedup_skipped,
        t0.elapsed()
    );

    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME)).unwrap();
    let hidden = manifest.model.hidden as usize;
    let moe_layers = manifest.model.moe_layers as u64;

    let cfg = Config::default();
    let t_load = std::time::Instant::now();
    let spine = Spine::load(&model_dir, &manifest, cfg).unwrap();
    eprintln!(
        "M2 spine load (always-on tensors): {:.1?}",
        t_load.elapsed()
    );

    // Short prompt + tiny max_new bound the wall time: the spine-sim forward is a
    // CPU-bound dense pure-Rust f32 pass, and a batched step runs B of them, so
    // per-step cost scales ~linearly in B (this is exactly what the sweep shows).
    let prompt_len = 2usize;
    let max_new = 2usize;
    let vocab = spine.vocab();
    let k = spine.experts_per_step() as u64;
    let elem = 1u64; // fp8: one byte per element
    let payload = hidden as u64 * elem;
    // forwards per stream = prompt_len primes + (max_new - 1) generations.
    let forwards = (prompt_len + max_new - 1) as u64;

    eprintln!(
        "M2 sweep: prompt {prompt_len} tok, max_new {max_new}, top-k {k}, {moe_layers} MoE layers, \
         fp8 blobs + fp8 wire, loopback"
    );
    eprintln!("B\ttok/s\tstep_median\tstep_p99\tup_B\tdown_B\tup/tok\tdown/tok");

    for b in [1usize, 2, 4, 8, 16, 32, 64, 128] {
        let prompts = bench_prompts(b, prompt_len, vocab, 42);
        let refs: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
        let (_outs, stats) = via_node_batch(&spine, &carved, "fp8", &refs, max_new, &[]);

        // Exact per-direction wire accounting — the sweep double-checks that
        // batching adds NO new wire shape (ADR-0023): D independent dispatch/gather
        // pairs, D = B × forwards × moe_layers, all experts present (answered == k).
        let d = b as u64 * forwards * moe_layers;
        let up = HANDSHAKE_LEN as u64 + d * (DISPATCH_HEADER_LEN as u64 + payload + 2 * k);
        let down =
            d * (GATHER_HEADER_LEN as u64 + GATHER_RECORD_HEADER_LEN as u64 * k + payload * k);
        assert_eq!(stats.wire_up, up, "B={b}: up bytes reconcile to framing");
        assert_eq!(
            stats.wire_down, down,
            "B={b}: down bytes reconcile to framing"
        );
        assert_eq!(
            stats.dispatches, d,
            "B={b}: dispatch count = B × forwards × moe_layers"
        );
        assert_eq!(
            stats.generated_tokens,
            b * max_new,
            "B={b}: aggregate generated tokens"
        );

        let secs = stats.elapsed.as_secs_f64();
        let tok_s = stats.generated_tokens as f64 / secs;
        let (median, p99) = stats.latency_median_p99();
        let up_tok = stats.wire_up as f64 / stats.generated_tokens as f64;
        let down_tok = stats.wire_down as f64 / stats.generated_tokens as f64;
        eprintln!(
            "{b}\t{tok_s:.3}\t{median:.2?}\t{p99:.2?}\t{}\t{}\t{up_tok:.0}\t{down_tok:.0}",
            stats.wire_up, stats.wire_down
        );
    }
}
