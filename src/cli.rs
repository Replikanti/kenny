//! Hand-rolled argument parsing (ADR-0021: no CLI framework for a surface
//! this small; the whole grammar fits in one screen of match arms).

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::blob::Dtype;
use crate::carve;
use crate::diff;
use crate::error::{Error, Result};
use crate::fixture;
use crate::manifest::{self, Manifest};
use crate::node;
use crate::rng::SplitMix64;
use crate::spine::{self, Config, LocalDispatch, NodeDispatch, Spine};
use crate::wire::{Bf16Codec, Fp8Codec, WireCodec};

const USAGE: &str = "\
kenny — distributed MoE expert pool (M0 carve tooling)

USAGE:
    kenny fixture --out <dir> [--layers N] [--experts N] [--hidden N]
                  [--inter N] [--vocab N] [--seed N]
        Generate a tiny synthetic safetensors model (Qwen3 naming schema,
        deterministic bytes). Defaults: 2 layers x 4 experts x hidden 8 x
        inter 4, vocab 32, seed 42.

    kenny carve --dump-names <model_dir>
        Print every tensor name (natural order) with its shard file. Run this
        against a real model before carving it — trust the schema, but verify.

    kenny carve <model_dir> --out <dir> [--dtype bf16|fp8|int8]
                [--model-name NAME] [--model-rev REV]
        Cut routed experts into content-addressed blobs (blobs/<xx>/<cid>)
        and write the canonical manifest (manifest.json). bf16 = byte-exact
        passthrough; fp8/int8 = central per-channel quantization (ADR-0012).
        --model-name defaults to the model directory name.

    kenny diff <model_dir> <carved_dir> [--layer N] [--batch N] [--seed N]
        Recompute y = down(silu(gate.x) * (up.x)) for every expert of one MoE
        layer, from source tensors and from carved blobs, and compare. bf16
        carves must match bit-for-bit; fp8/int8 report max-abs and cosine.

    kenny node --carved <dir> [--listen <addr>]
        Serve the carve's experts (ADR-0013): load the manifest, then answer
        dispatch/gather over sync TCP (ADR-0016). Prints `listening <addr>` on
        stdout — the OS-assigned address, since --listen defaults to
        127.0.0.1:0 (a fixed port is flaky under concurrency). Experts the node
        does not hold answer not-held (feeds the spine's renorm, ADR-0008).

    kenny spine --carved <dir> --model <model_dir> (--node <addr> | --local)
                [--tokens N] [--prompt id,id,...] [--batch N] [--codec fp8|bf16]
                [--seed N] [--num-heads N] [--num-kv-heads N] [--head-dim N]
                [--rope-theta N] [--rms-eps-ppm N] [--top-k N] [--layer-timeout-ms N]
        Run the Qwen3-30B-A3B spine-sim (ADR-0020): a pure-Rust dense forward
        whose MoE FFN is dispatched to a node (--node) or run in-process
        (--local). Reads the always-on tensors from <model_dir> by the manifest's
        byte ranges and greedily generates --tokens tokens, printing tok/s and
        wire bytes counted at the socket. --batch N advances N independent streams
        in lockstep (ADR-0006 / ADR-0023: aggregate tok/s is the product); N > 1
        derives N distinct seed-keyed prompts (so --prompt is single-stream only)
        and reports aggregate tok/s over N x --tokens tokens. Hyperparameter flags
        default to the Qwen3-30B-A3B card (head-dim 128 != hidden/num-heads;
        rope-theta 10000000; rms-eps-ppm 1 = 1e-6); the fixture (square attention)
        loads only at --num-heads 1 --num-kv-heads 1 --head-dim <hidden>.
        --layer-timeout-ms N caps each MoE layer's wait on a --node (ADR-0010):
        a straggler layer is dropped to the renorm (ADR-0008) and the connection
        reconnected; default off (wait indefinitely, the M1/M2 behavior).
";

pub fn run(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        None | Some("-h" | "--help" | "help") => {
            print!("{USAGE}");
            Ok(())
        }
        Some("fixture") => run_fixture(&args[1..]),
        Some("carve") => run_carve(&args[1..]),
        Some("diff") => run_diff(&args[1..]),
        Some("node") => run_node(&args[1..]),
        Some("spine") => run_spine(&args[1..]),
        Some(other) => Err(Error::usage(format!("unknown command {other:?}"))),
    }
}

fn value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| Error::usage(format!("{flag} needs a value")))
}

fn parse_num(s: &str, flag: &str, min: u64, max: u64) -> Result<u64> {
    let n = s
        .parse::<u64>()
        .map_err(|_| Error::usage(format!("{flag}: {s:?} is not a non-negative integer")))?;
    if n < min || n > max {
        return Err(Error::usage(format!(
            "{flag}: {n} out of range [{min}, {max}]"
        )));
    }
    Ok(n)
}

fn run_fixture(args: &[String]) -> Result<()> {
    let mut p = fixture::Params::default();
    let mut out: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => out = Some(PathBuf::from(value(args, &mut i, "--out")?)),
            "--layers" => {
                p.layers = parse_num(
                    value(args, &mut i, "--layers")?,
                    "--layers",
                    1,
                    u16::MAX as u64,
                )? as u16;
            }
            "--experts" => {
                p.experts = parse_num(
                    value(args, &mut i, "--experts")?,
                    "--experts",
                    1,
                    u16::MAX as u64,
                )? as u16;
            }
            "--hidden" => {
                p.hidden =
                    parse_num(value(args, &mut i, "--hidden")?, "--hidden", 1, 1 << 24)? as u32;
            }
            "--inter" => {
                p.inter = parse_num(value(args, &mut i, "--inter")?, "--inter", 1, 1 << 24)? as u32;
            }
            "--vocab" => {
                p.vocab = parse_num(value(args, &mut i, "--vocab")?, "--vocab", 1, 1 << 24)? as u32;
            }
            "--seed" => p.seed = parse_num(value(args, &mut i, "--seed")?, "--seed", 0, u64::MAX)?,
            other => {
                return Err(Error::usage(format!(
                    "fixture: unexpected argument {other:?}"
                )));
            }
        }
        i += 1;
    }
    let out = out.ok_or_else(|| Error::usage("fixture: --out is required"))?;
    let s = fixture::generate(&p, &out)?;
    println!(
        "fixture: {} tensors in {} shards, {} bytes -> {}",
        s.tensors,
        s.shards,
        s.bytes,
        out.display()
    );
    Ok(())
}

fn run_carve(args: &[String]) -> Result<()> {
    let mut dump_names = false;
    let mut model_dir: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut dtype = Dtype::Bf16;
    let mut model_name: Option<String> = None;
    let mut model_rev = String::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dump-names" => dump_names = true,
            "--out" => out = Some(PathBuf::from(value(args, &mut i, "--out")?)),
            "--dtype" => {
                dtype = Dtype::from_name(value(args, &mut i, "--dtype")?)
                    .map_err(|e| Error::usage(format!("--dtype: {e}")))?;
            }
            "--model-name" => model_name = Some(value(args, &mut i, "--model-name")?.to_string()),
            "--model-rev" => model_rev = value(args, &mut i, "--model-rev")?.to_string(),
            flag if flag.starts_with('-') => {
                return Err(Error::usage(format!("carve: unknown flag {flag:?}")));
            }
            path => {
                if model_dir.is_some() {
                    return Err(Error::usage("carve: more than one model directory given"));
                }
                model_dir = Some(PathBuf::from(path));
            }
        }
        i += 1;
    }
    let model_dir =
        model_dir.ok_or_else(|| Error::usage("carve: a model directory is required"))?;

    if dump_names {
        let names = carve::dump_names(&model_dir)?;
        let shards: BTreeSet<&str> = names.iter().map(|(_, s)| s.as_str()).collect();
        for (tensor, shard) in &names {
            println!("{tensor}\t{shard}");
        }
        eprintln!("{} tensors across {} shards", names.len(), shards.len());
        return Ok(());
    }

    let out = out.ok_or_else(|| Error::usage("carve: --out is required (or use --dump-names)"))?;
    let model_name = model_name.unwrap_or_else(|| {
        model_dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".into())
    });
    let opts = carve::Options {
        out,
        model_name,
        model_rev,
        dtype,
    };
    let s = carve::run(&model_dir, &opts)?;
    println!(
        "carved:   {} MoE layers x {} experts (hidden {}, inter {})",
        s.moe_layers, s.experts_per_layer, s.hidden, s.inter
    );
    println!(
        "blobs:    {} experts, {} new bytes, {} deduplicated",
        s.blobs, s.blob_bytes, s.dedup_skipped
    );
    println!("spine:    {} tensors recorded by range", s.spine_tensors);
    println!("manifest: {}", s.manifest_path.display());
    println!("identity: {}", s.manifest_identity);
    Ok(())
}

fn run_diff(args: &[String]) -> Result<()> {
    let mut opts = diff::DiffOptions::default();
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--layer" => {
                opts.layer = parse_num(
                    value(args, &mut i, "--layer")?,
                    "--layer",
                    0,
                    u16::MAX as u64,
                )? as u16;
            }
            "--batch" => {
                opts.batch =
                    parse_num(value(args, &mut i, "--batch")?, "--batch", 1, 4096)? as usize;
            }
            "--seed" => {
                opts.seed = parse_num(value(args, &mut i, "--seed")?, "--seed", 0, u64::MAX)?
            }
            flag if flag.starts_with('-') => {
                return Err(Error::usage(format!("diff: unknown flag {flag:?}")));
            }
            path => dirs.push(PathBuf::from(path)),
        }
        i += 1;
    }
    let [model_dir, carved_dir] = dirs.as_slice() else {
        return Err(Error::usage(
            "diff: exactly two directories required: <model_dir> <carved_dir>",
        ));
    };
    let r = diff::run(model_dir, carved_dir, &opts)?;
    let worst_abs = r
        .per_expert
        .iter()
        .max_by(|a, b| a.max_abs.total_cmp(&b.max_abs))
        .expect("at least one expert");
    let worst_cos = r
        .per_expert
        .iter()
        .min_by(|a, b| a.cosine.total_cmp(&b.cosine))
        .expect("at least one expert");
    println!(
        "diff:     layer {} ({}), {} experts x batch {}",
        r.layer,
        r.dtype.name(),
        r.per_expert.len(),
        r.batch
    );
    println!(
        "exact:    {}",
        if r.bitwise_exact {
            "yes (bit-for-bit)"
        } else {
            "no"
        }
    );
    println!(
        "max-abs:  {:.3e} (worst: expert {})",
        r.worst_max_abs, worst_abs.expert
    );
    println!(
        "cosine:   {:.8} (worst: expert {})",
        r.worst_cosine, worst_cos.expert
    );
    Ok(())
}

fn run_node(args: &[String]) -> Result<()> {
    let mut carved: Option<PathBuf> = None;
    // A5: default to an OS-assigned port; a fixed port flakes under CI
    // concurrency. The bound address is printed on stdout for discovery.
    let mut listen = String::from("127.0.0.1:0");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--carved" => carved = Some(PathBuf::from(value(args, &mut i, "--carved")?)),
            "--listen" => listen = value(args, &mut i, "--listen")?.to_string(),
            other => {
                return Err(Error::usage(format!("node: unexpected argument {other:?}")));
            }
        }
        i += 1;
    }
    let carved = carved.ok_or_else(|| Error::usage("node: --carved is required"))?;
    node::serve(&carved, &listen)
}

/// Parse a comma-separated token-id list, e.g. `--prompt 1,2,3`.
fn parse_prompt(s: &str) -> Result<Vec<u32>> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<u32>()
                .map_err(|_| Error::usage(format!("--prompt: {p:?} is not a token id")))
        })
        .collect()
}

fn run_spine(args: &[String]) -> Result<()> {
    let mut carved: Option<PathBuf> = None;
    let mut model: Option<PathBuf> = None;
    let mut nodes: Vec<String> = Vec::new();
    let mut local = false;
    let mut tokens = 16usize;
    let mut batch = 1usize;
    let mut prompt: Option<Vec<u32>> = None;
    let mut codec_name = String::from("fp8");
    let mut seed = 42u64;
    let mut cfg = Config::default();
    let mut rms_eps_ppm = 1u64; // 1e-6
    let mut layer_timeout_ms: Option<u64> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--carved" => carved = Some(PathBuf::from(value(args, &mut i, "--carved")?)),
            "--model" => model = Some(PathBuf::from(value(args, &mut i, "--model")?)),
            "--node" => nodes.push(value(args, &mut i, "--node")?.to_string()),
            "--local" => local = true,
            "--tokens" => {
                tokens =
                    parse_num(value(args, &mut i, "--tokens")?, "--tokens", 1, 1 << 20)? as usize;
            }
            "--batch" => {
                // Cap at 4096: past that the fp8 payloads bound B_max well before
                // the framing does (MANIFESTO §4.4), and it keeps the per-stream
                // Vec allocations sane on the spine.
                batch = parse_num(value(args, &mut i, "--batch")?, "--batch", 1, 4096)? as usize;
            }
            "--prompt" => prompt = Some(parse_prompt(value(args, &mut i, "--prompt")?)?),
            "--codec" => codec_name = value(args, &mut i, "--codec")?.to_string(),
            "--seed" => seed = parse_num(value(args, &mut i, "--seed")?, "--seed", 0, u64::MAX)?,
            "--num-heads" => {
                cfg.num_heads = parse_num(
                    value(args, &mut i, "--num-heads")?,
                    "--num-heads",
                    1,
                    1 << 16,
                )? as usize;
            }
            "--num-kv-heads" => {
                cfg.num_kv_heads = parse_num(
                    value(args, &mut i, "--num-kv-heads")?,
                    "--num-kv-heads",
                    1,
                    1 << 16,
                )? as usize;
            }
            "--head-dim" => {
                cfg.head_dim =
                    parse_num(value(args, &mut i, "--head-dim")?, "--head-dim", 2, 1 << 16)?
                        as usize;
            }
            "--rope-theta" => {
                cfg.rope_theta = parse_num(
                    value(args, &mut i, "--rope-theta")?,
                    "--rope-theta",
                    1,
                    u64::MAX,
                )? as f64;
            }
            "--rms-eps-ppm" => {
                rms_eps_ppm = parse_num(
                    value(args, &mut i, "--rms-eps-ppm")?,
                    "--rms-eps-ppm",
                    0,
                    1 << 30,
                )?;
            }
            "--top-k" => {
                cfg.top_k =
                    parse_num(value(args, &mut i, "--top-k")?, "--top-k", 1, 1 << 16)? as usize;
            }
            "--layer-timeout-ms" => {
                layer_timeout_ms = Some(parse_num(
                    value(args, &mut i, "--layer-timeout-ms")?,
                    "--layer-timeout-ms",
                    1,
                    1 << 30,
                )?);
            }
            other => {
                return Err(Error::usage(format!(
                    "spine: unexpected argument {other:?}"
                )));
            }
        }
        i += 1;
    }
    cfg.rms_eps = spine::eps_from_ppm(rms_eps_ppm);

    let carved = carved.ok_or_else(|| Error::usage("spine: --carved is required"))?;
    let model = model.ok_or_else(|| Error::usage("spine: --model is required"))?;
    let has_node = !nodes.is_empty();
    if local == has_node {
        return Err(Error::usage(
            "spine: pass exactly one of --local or --node <addr>",
        ));
    }
    if nodes.len() > 1 {
        // Multi-node dispatch needs a placement/replication policy (ADR-0009),
        // which is out of scope for M1's single-node two-process gate.
        return Err(Error::usage(
            "spine: M1 supports a single --node (multi-node placement is ADR-0009, post-M1)",
        ));
    }

    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME))?;
    let identity = *blake3::hash(&manifest.canonical_bytes()).as_bytes();
    let hidden = manifest.model.hidden as usize;
    let spine = Spine::load(&model, &manifest, cfg)?;

    let codec: Box<dyn WireCodec> = match codec_name.as_str() {
        "fp8" => Box::new(Fp8Codec),
        "bf16" => Box::new(Bf16Codec),
        other => {
            return Err(Error::usage(format!(
                "spine: unknown --codec {other:?} (expected fp8 or bf16)"
            )));
        }
    };

    // Build the batch of prompts. B = 1 keeps the single-stream path (explicit
    // --prompt or a seed-derived default). B > 1 derives B DISTINCT prompts, one
    // per stream index keyed off --seed, so the streams route independently; an
    // explicit --prompt is single-stream only (batched prompts are seed-derived).
    let prompts: Vec<Vec<u32>> = if batch == 1 {
        vec![prompt.unwrap_or_else(|| seed_prompt(seed, 0, spine.vocab()))]
    } else {
        if prompt.is_some() {
            return Err(Error::usage(
                "spine: --prompt is single-stream; with --batch N > 1 prompts are seed-derived",
            ));
        }
        (0..batch)
            .map(|s| seed_prompt(seed, s, spine.vocab()))
            .collect()
    };
    let prompt_refs: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();

    let mut dispatcher: Box<dyn spine::Dispatcher> = if local {
        if layer_timeout_ms.is_some() {
            return Err(Error::usage(
                "spine: --layer-timeout-ms applies to --node dispatch, not --local",
            ));
        }
        Box::new(LocalDispatch::new(&carved, codec)?)
    } else {
        let mut nd = NodeDispatch::connect(&nodes[0], codec, identity, hidden)?;
        if let Some(ms) = layer_timeout_ms {
            nd = nd.with_layer_timeout(std::time::Duration::from_millis(ms));
        }
        Box::new(nd)
    };

    let (outs, stats) = spine.generate_batch(dispatcher.as_mut(), &prompt_refs, tokens)?;

    let where_ = if local {
        "local (in-process)".to_string()
    } else {
        format!("node {}", nodes[0])
    };
    let secs = stats.elapsed.as_secs_f64();
    let tok_s = if secs > 0.0 {
        stats.generated_tokens as f64 / secs
    } else {
        f64::INFINITY
    };
    println!(
        "spine:    {} MoE layers, {} experts/layer, top-k {} ({} per step), codec {}",
        spine.moe_layers(),
        manifest.model.experts_per_layer,
        cfg.top_k,
        spine.experts_per_step(),
        codec_name
    );
    println!("dispatch: {where_}");
    if batch == 1 {
        println!(
            "prompt:   {} tokens -> generated {} (vocab {})",
            stats.prompt_tokens,
            stats.generated_tokens,
            spine.vocab()
        );
        println!("tokens:   {:?}", outs[0]);
    } else {
        println!(
            "batch:    {} streams x {} prompt tokens -> {} generated total (vocab {})",
            batch,
            stats.prompt_tokens,
            stats.generated_tokens,
            spine.vocab()
        );
        println!("tokens:   stream 0 {:?}", outs[0]);
    }
    println!(
        "dispatch: {} frames, {}/{} experts answered, {} renorm steps, {} layer timeouts",
        stats.dispatches,
        stats.experts_answered,
        stats.experts_requested,
        stats.renorm_steps,
        stats.layer_timeouts
    );
    println!(
        "wire:     up {} B, down {} B",
        stats.wire_up, stats.wire_down
    );
    let (median, p99) = stats.latency_median_p99();
    let rate = if batch == 1 {
        "tok/s"
    } else {
        "tok/s aggregate"
    };
    println!(
        "speed:    {tok_s:.1} {rate} ({:.3} s); per-step median {:.1?}, p99 {:.1?}",
        secs, median, p99
    );
    Ok(())
}

/// A short deterministic prompt for stream `s`, keyed by `(seed, s)` so distinct
/// streams route independently and a run is reproducible without an explicit
/// `--prompt`. Stream 0 keeps the pre-batch key so single-stream runs are
/// byte-stable across this change.
fn seed_prompt(seed: u64, s: usize, vocab: usize) -> Vec<u32> {
    let name = if s == 0 {
        "spine.prompt".to_string()
    } else {
        format!("spine.prompt.{s}")
    };
    let mut rng = SplitMix64::for_name(seed, &name);
    (0..4)
        .map(|_| (rng.next_u64() % vocab as u64) as u32)
        .collect()
}
