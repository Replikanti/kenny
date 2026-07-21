//! The M0 stop condition: fixture -> carve -> blobs + manifest -> reload ->
//! bit-exact diff vs source tensors. Golden constants lock the consensus
//! encodings — if one changes, the format changed, which requires a version
//! bump plus an ADR, never a test edit alone (kenny-format-auditor enforces
//! this).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use kenny::blob::{self, Dtype};
use kenny::carve::{self, Options};
use kenny::diff;
use kenny::fixture::{self, Params};
use kenny::manifest::Manifest;
use kenny::safetensors::{self, ShardFile};

const GOLDEN_MANIFEST_IDENTITY: &str =
    "f3776ff47bf10cdd9e5c849d1b8f596f9e44300b7d6e4f45d41b4998aa00bde5";
// Locks the fixture generator end to end: safetensors writer layout, RNG
// streams, bf16 rounding. Same change protocol as the other goldens.
const GOLDEN_FIXTURE_SHARD1: &str =
    "0ec0a9fb04c8c74d64da66b20436779f184a29b50b0ec9c5b71e90fbf2cb50b3";

fn tmp(name: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

fn opts(out: PathBuf) -> Options {
    Options {
        out,
        model_name: "fixture".into(),
        model_rev: String::new(),
        dtype: Dtype::Bf16,
    }
}

/// blake3 of every file under `dir`, keyed by path relative to `dir`.
fn dir_hashes(dir: &Path) -> BTreeMap<String, String> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                out.insert(rel, blob::cid(&fs::read(&path).unwrap()));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(dir, dir, &mut out);
    out
}

fn open_shards(model_dir: &Path) -> (Vec<(String, String)>, BTreeMap<String, ShardFile>) {
    let model = safetensors::open_model(model_dir).unwrap();
    let mut shards = BTreeMap::new();
    for (_, shard_name) in &model.weight_map {
        if !shards.contains_key(shard_name) {
            shards.insert(
                shard_name.clone(),
                ShardFile::open(&model.dir.join(shard_name)).unwrap(),
            );
        }
    }
    (model.weight_map, shards)
}

#[test]
fn fixture_is_deterministic() {
    let root = tmp("fixture-det");
    let (a, b, c) = (root.join("a"), root.join("b"), root.join("c"));
    fixture::generate(&Params::default(), &a).unwrap();
    fixture::generate(&Params::default(), &b).unwrap();
    assert_eq!(
        dir_hashes(&a),
        dir_hashes(&b),
        "same seed => identical bytes"
    );

    fixture::generate(
        &Params {
            seed: 43,
            ..Params::default()
        },
        &c,
    )
    .unwrap();
    let (ha, hc) = (dir_hashes(&a), dir_hashes(&c));
    assert_eq!(
        ha.keys().collect::<Vec<_>>(),
        hc.keys().collect::<Vec<_>>(),
        "same file set regardless of seed"
    );
    assert_ne!(
        ha["model-00001-of-00002.safetensors"], hc["model-00001-of-00002.safetensors"],
        "different seed => different tensor bytes"
    );
}

#[test]
fn roundtrip_bit_exact() {
    let root = tmp("roundtrip");
    let model_dir = root.join("model");
    fixture::generate(&Params::default(), &model_dir).unwrap();
    let out = root.join("carved");
    let summary = carve::run(&model_dir, &opts(out.clone())).unwrap();
    assert_eq!(summary.blobs, 8);
    assert_eq!(summary.dedup_skipped, 0);

    let m = Manifest::load(&out.join("manifest.json")).unwrap();
    assert_eq!(
        (
            m.model.hidden,
            m.model.inter,
            m.model.moe_layers,
            m.model.experts_per_layer
        ),
        (8, 4, 2, 4)
    );
    assert_eq!(m.experts.len(), 8);

    let (weight_map, shards) = open_shards(&model_dir);
    let tensor_shard: BTreeMap<&str, &str> = weight_map
        .iter()
        .map(|(t, s)| (t.as_str(), s.as_str()))
        .collect();

    // Every expert matrix reloaded from its blob is byte-identical to the
    // source tensor, and the stored blob hashes to its own CID.
    for e in &m.experts {
        let blob_path = out.join("blobs").join(blob::rel_path(&e.cid));
        let bytes = fs::read(&blob_path).unwrap();
        assert_eq!(blob::cid(&bytes), e.cid, "blob content matches its CID");
        let d = blob::decode(&bytes).unwrap();
        assert_eq!((d.header.layer, d.header.expert), (e.layer, e.expert));
        assert_eq!((d.header.hidden, d.header.inter), (8, 4));
        for (proj, got) in [
            ("gate_proj", d.gate),
            ("up_proj", d.up),
            ("down_proj", d.down),
        ] {
            let name = format!(
                "model.layers.{}.mlp.experts.{}.{proj}.weight",
                e.layer, e.expert
            );
            let shard = &shards[tensor_shard[name.as_str()]];
            let want = shard.bytes(shard.tensor(&name).unwrap());
            assert!(got == want, "byte mismatch in {name}");
        }
    }

    // Every non-expert tensor is recorded on the spine side with the right
    // hash and absolute byte range.
    assert_eq!(m.spine.len(), weight_map.len() - 8 * 3);
    for s in &m.spine {
        let shard = &shards[s.shard.as_str()];
        let meta = shard.tensor(&s.name).unwrap();
        assert_eq!(
            blob::cid(shard.bytes(meta)),
            s.cid,
            "spine cid for {}",
            s.name
        );
        assert_eq!(
            (s.begin, s.end),
            shard.abs_range(meta),
            "abs range for {}",
            s.name
        );
        assert_eq!(s.dtype, meta.dtype);
        assert_eq!(s.shape, meta.shape);
    }
}

#[test]
fn carve_is_deterministic_and_dedups() {
    let root = tmp("carve-det");
    let model_dir = root.join("model");
    fixture::generate(&Params::default(), &model_dir).unwrap();

    let (out1, out2) = (root.join("c1"), root.join("c2"));
    let s1 = carve::run(&model_dir, &opts(out1.clone())).unwrap();
    let s2 = carve::run(&model_dir, &opts(out2.clone())).unwrap();
    assert_eq!(
        fs::read(out1.join("manifest.json")).unwrap(),
        fs::read(out2.join("manifest.json")).unwrap(),
        "carve is deterministic"
    );
    assert_eq!(s1.manifest_identity, s2.manifest_identity);
    assert_eq!(dir_hashes(&out1), dir_hashes(&out2));

    // Re-carving into the same output skips every existing blob.
    let s3 = carve::run(&model_dir, &opts(out1.clone())).unwrap();
    assert_eq!(s3.dedup_skipped, 8);
    assert_eq!(s3.blob_bytes, 0);
    assert_eq!(s3.manifest_identity, s1.manifest_identity);

    // The manifest identity IS the blake3 of the file bytes as written.
    assert_eq!(
        blob::cid(&fs::read(out1.join("manifest.json")).unwrap()),
        s1.manifest_identity
    );
}

#[test]
fn golden_manifest_identity() {
    let root = tmp("golden");
    let model_dir = root.join("model");
    fixture::generate(&Params::default(), &model_dir).unwrap();
    let summary = carve::run(&model_dir, &opts(root.join("carved"))).unwrap();
    assert_eq!(
        blob::cid(&fs::read(model_dir.join("model-00001-of-00002.safetensors")).unwrap()),
        GOLDEN_FIXTURE_SHARD1,
        "fixture shard bytes changed — generator, safetensors writer, RNG or bf16 moved"
    );
    assert_eq!(
        summary.manifest_identity, GOLDEN_MANIFEST_IDENTITY,
        "manifest identity for the default fixture changed — that means the canonical \
         encoding or the fixture generator changed, which is a consensus event: bump the \
         format/codec version and write the ADR before touching this constant"
    );
}

#[test]
fn cli_smoke() {
    let bin = env!("CARGO_BIN_EXE_kenny");
    let root = tmp("cli");
    let model = root.join("model");
    let carved = root.join("carved");

    let st = Command::new(bin)
        .args(["fixture", "--out"])
        .arg(&model)
        .status()
        .unwrap();
    assert!(st.success(), "fixture");

    let out = Command::new(bin)
        .args(["carve", "--dump-names"])
        .arg(&model)
        .output()
        .unwrap();
    assert!(out.status.success(), "dump-names");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(
            "model.layers.0.mlp.experts.0.gate_proj.weight\tmodel-00001-of-00002.safetensors"
        ),
        "dump-names lists expert tensors with shards"
    );

    let out = Command::new(bin)
        .args(["carve"])
        .arg(&model)
        .args(["--out"])
        .arg(&carved)
        .args(["--model-name", "fixture"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "carve: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(carved.join("manifest.json").is_file());

    let out2 = Command::new(bin)
        .args(["carve"])
        .arg(&model)
        .args(["--out"])
        .arg(&carved)
        .args(["--model-name", "fixture"])
        .output()
        .unwrap();
    assert!(out2.status.success());
    assert!(
        String::from_utf8_lossy(&out2.stdout).contains("8 deduplicated"),
        "second carve reports dedup"
    );

    // Usage errors exit 2, not 1.
    let st = Command::new(bin).args(["carve"]).status().unwrap();
    assert_eq!(st.code(), Some(2), "missing args is a usage error");
    let st = Command::new(bin).args(["frobnicate"]).status().unwrap();
    assert_eq!(st.code(), Some(2), "unknown command is a usage error");
    let st = Command::new(bin)
        .args(["carve"])
        .arg(&model)
        .args(["--out"])
        .arg(root.join("x"))
        .args(["--dtype", "bogus"])
        .status()
        .unwrap();
    assert_eq!(st.code(), Some(2), "unknown dtype is refused");
}

const GOLDEN_MANIFEST_IDENTITY_FP8: &str =
    "7d00d51564a26942182a9aa3df3bcf73f44d9fc3cf95bd020625db175b36d6e5";
const GOLDEN_MANIFEST_IDENTITY_INT8: &str =
    "e465b37199d63782ce02841e5bde7ad053c0a383579c34414e60f61840756264";
// Thresholds measured on the default fixture, set with margin below the
// observed floor; a regression past these means the quantizer moved.
const FP8_MIN_COSINE: f64 = 0.995;
const INT8_MIN_COSINE: f64 = 0.9998;

#[test]
fn quantized_carve_and_diff() {
    let root = tmp("qdiff");
    let model_dir = root.join("model");
    fixture::generate(&Params::default(), &model_dir).unwrap();

    let mut identities = Vec::new();
    for dtype in [Dtype::Bf16, Dtype::Fp8, Dtype::Int8] {
        let out = root.join(dtype.name());
        let s = carve::run(
            &model_dir,
            &Options {
                out: out.clone(),
                model_name: "fixture".into(),
                model_rev: String::new(),
                dtype,
            },
        )
        .unwrap();
        identities.push(s.manifest_identity.clone());

        for layer in [0u16, 1] {
            let r = diff::run(
                &model_dir,
                &out,
                &diff::DiffOptions {
                    layer,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(r.per_expert.len(), 4);
            eprintln!(
                "{} layer {layer}: exact={} max_abs={:.3e} cosine={:.8}",
                dtype.name(),
                r.bitwise_exact,
                r.worst_max_abs,
                r.worst_cosine
            );
            match dtype {
                Dtype::Bf16 => {
                    assert!(r.bitwise_exact, "bf16 passthrough must be bit-exact");
                    assert_eq!(r.worst_max_abs, 0.0);
                }
                Dtype::Fp8 => {
                    assert!(!r.per_expert.is_empty());
                    assert!(
                        r.worst_cosine >= FP8_MIN_COSINE,
                        "fp8 cosine {}",
                        r.worst_cosine
                    );
                }
                Dtype::Int8 => {
                    assert!(
                        r.worst_cosine >= INT8_MIN_COSINE,
                        "int8 cosine {}",
                        r.worst_cosine
                    );
                }
            }
        }
    }
    assert_ne!(
        identities[0], identities[1],
        "dtype changes the model identity"
    );
    assert_ne!(identities[1], identities[2]);
    assert_eq!(identities[1], GOLDEN_MANIFEST_IDENTITY_FP8, "fp8 identity");
    assert_eq!(
        identities[2], GOLDEN_MANIFEST_IDENTITY_INT8,
        "int8 identity"
    );

    // Asking for a layer that has no experts is a clear error.
    assert!(
        diff::run(
            &model_dir,
            &root.join("bf16"),
            &diff::DiffOptions {
                layer: 7,
                ..Default::default()
            }
        )
        .is_err()
    );
}

/// Full real-model round-trip: carve all 6,144 Qwen3-30B-A3B experts (bf16
/// passthrough) and bit-exactly diff layer 0. Gated on KENNY_MODEL_DIR; a
/// re-run is cheap because existing blobs dedup-skip (hash-only pass).
#[test]
fn real_model_full_carve_and_diff() {
    let Some(dir) = std::env::var_os("KENNY_MODEL_DIR") else {
        eprintln!("KENNY_MODEL_DIR unset — skipping real-model carve + diff");
        return;
    };
    let model_dir = PathBuf::from(dir);
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("real-carve-bf16");
    fs::create_dir_all(&out).unwrap();
    let t0 = std::time::Instant::now();
    let s = carve::run(
        &model_dir,
        &Options {
            out: out.clone(),
            model_name: "qwen3-30b-a3b".into(),
            model_rev: String::new(),
            dtype: Dtype::Bf16,
        },
    )
    .unwrap();
    eprintln!(
        "real carve: {} blobs ({} new bytes, {} dedup) in {:.1?}",
        s.blobs,
        s.blob_bytes,
        s.dedup_skipped,
        t0.elapsed()
    );
    assert_eq!(s.blobs, 6144, "Qwen3-30B-A3B routed expert count");
    assert_eq!((s.moe_layers, s.experts_per_layer), (48, 128));
    assert_eq!((s.hidden, s.inter), (2048, 768));

    let t1 = std::time::Instant::now();
    let r = diff::run(&model_dir, &out, &diff::DiffOptions::default()).unwrap();
    eprintln!(
        "real diff layer 0: exact={} max_abs={:.3e} cosine={:.8} in {:.1?}",
        r.bitwise_exact,
        r.worst_max_abs,
        r.worst_cosine,
        t1.elapsed()
    );
    assert_eq!(r.per_expert.len(), 128);
    assert!(
        r.bitwise_exact,
        "bf16 passthrough must be bit-exact on the real model"
    );
}

/// Gated on KENNY_MODEL_DIR (repo convention: CI never downloads models).
/// Light schema gate for issue #1 scope — the full layer-0 carve + numeric
/// diff belongs to the kenny-diff milestone.
#[test]
fn real_model_schema_gate() {
    let Some(dir) = std::env::var_os("KENNY_MODEL_DIR") else {
        eprintln!("KENNY_MODEL_DIR unset — skipping real-model schema gate");
        return;
    };
    let names = carve::dump_names(Path::new(&dir)).unwrap();
    assert!(!names.is_empty(), "model dir lists no tensors");
    let experts = names
        .iter()
        .filter(|(n, _)| n.contains(".mlp.experts."))
        .count();
    assert!(
        experts > 0,
        "no routed-expert tensors found — wrong dir or schema drift"
    );
    eprintln!(
        "real model: {} tensors, {} expert-family",
        names.len(),
        experts
    );
}
