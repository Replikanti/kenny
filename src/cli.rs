//! Hand-rolled argument parsing (ADR-0021: no CLI framework for a surface
//! this small; the whole grammar fits in one screen of match arms).

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::blob::Dtype;
use crate::canary;
use crate::carve;
use crate::diff;
use crate::error::{Error, Result};
use crate::fixture;
use crate::manifest::{self, Manifest};
use crate::node::{self, Hold};
use crate::placement::{HeatMap, NodeDesc, build_placement};
use crate::prefix;
use crate::rng::SplitMix64;
use crate::spine::{self, Config, LocalDispatch, NodeDispatch, PlacedDispatch, Spine};
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
               [--hold <l:e,l:e,...> | --shard <i/n>]
        Serve the carve's experts (ADR-0013): load the manifest, then answer
        dispatch/gather over sync TCP (ADR-0016). Prints `listening <addr>` on
        stdout — the OS-assigned address, since --listen defaults to
        127.0.0.1:0 (a fixed port is flaky under concurrency). Experts the node
        does not hold answer not-held (feeds the spine's renorm, ADR-0008).
        --hold restricts the node to a SUBSET of the carve (the ADR-0009 held
        subset a placement map assigns): --hold 0:1,1:3 holds exactly those two
        experts, everything else answers not-held. --shard i/n instead holds
        shard i of n by a stable hash of (layer,expert), so n nodes launched with
        --shard 0/n .. (n-1)/n partition the catalog into DISJOINT subsets with no
        placement file — the sim's distinct-holding-nodes knob. The serve loop and
        the wire are untouched (ADR-0024): a subset node just answers not-held more.

    kenny spine --carved <dir> --model <model_dir> (--node <addr>... | --local)
                [--tokens N] [--prompt id,id,...] [--batch N] [--codec fp8|bf16]
                [--seed N] [--num-heads N] [--num-kv-heads N] [--head-dim N]
                [--rope-theta N] [--rms-eps-ppm N] [--top-k N] [--layer-timeout-ms N]
                [--replicas N] [--hedge-ms N]
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
        --layer-timeout-ms N caps each MoE layer's wait on a single --node
        (ADR-0010): a straggler layer is dropped to the renorm (ADR-0008) and the
        connection reconnected; default off (wait indefinitely, the M1/M2
        behavior). Pass --node more than once for MULTI-NODE placed dispatch
        (ADR-0009 / ADR-0024): the routed experts are fanned across the nodes by a
        placement map (uniform consistent-hash bootstrap over the node set, each
        expert replicated --replicas N ways — default 2, clamped to the node
        count), each holder gets its sub-list on the existing wire (no frame
        change), and an expert no node holds renorms. Dead replicas surface on the
        ADR-0008 alarm at the end of the run. --hedge-ms N arms the ADR-0010
        replica-set hedge on the placed path: a stalled primary's experts spill to
        their next replica, first-answer-wins (the multi-node analogue of the
        single-node --layer-timeout-ms; default off).

    kenny canary --carved <fp8_dir> --model <model_dir>
                 [--prompts N] [--len N] [--seed N] [--codec fp8|bf16]
                 [--num-heads N] [--num-kv-heads N] [--head-dim N]
                 [--rope-theta N] [--rms-eps-ppm N] [--top-k N]
        The ADR-0008 perplexity canary: teacher-forced perplexity of the carved
        blob+wire path (default fp8) vs the bf16-source reference (the diff.rs
        source-matrix path, no quant, no codec), over a fixed seed-keyed prompt
        set, and print Δppl = ppl(test) − ppl(ref). This is the deciding QUALITY
        axis ADR-0018 is blocked on. --prompts N sequences x --len N tokens
        (default 4 x 16); both paths hold every expert so nothing renorms and the
        number is pure quantization quality. Hyperparameter flags default to the
        Qwen3-30B-A3B card, exactly like `kenny spine`; the fixture (square
        attention) loads only at --num-heads 1 --num-kv-heads 1 --head-dim <hidden>.
        CI never downloads a model — run this against a real --model for a BENCH
        Δppl (KENNY_MODEL_DIR in the gated test arm).

    kenny prefix --carved <dir>
                 [--streams N] [--system-len N] [--user-len N] [--block N]
                 [--seed N] [--vocab N] [--num-kv-heads N] [--head-dim N]
        The ADR-0022 prefix-cache hit-rate on a SHARED-SYSTEM-PROMPT fixture: N
        streams share a --system-len system prompt and each carry a distinct
        seed-derived --user-len user tail; prompt tokens are chunked into --block
        blocks whose blake3 hash-chain keys (rooted in the manifest identity) are
        looked up in a spine-local radix, and prefix_hit_rate = reused / total
        prompt tokens is reported. Model-free: it reads only the carve's manifest
        (identity + MoE layer count), loads NO weights. The derived KV occupancy
        (B x ctx x layers x kv_elem, from the existing LayerKv) is printed
        alongside — --num-kv-heads / --head-dim default to the Qwen3-30B-A3B card
        (the fixture is --num-kv-heads 1 --head-dim <hidden>). Spine-local cache,
        NO consensus surface (WIRE_VERSION unchanged).
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
        Some("canary") => run_canary(&args[1..]),
        Some("prefix") => run_prefix(&args[1..]),
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
    let mut hold: Option<Hold> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--carved" => carved = Some(PathBuf::from(value(args, &mut i, "--carved")?)),
            "--listen" => listen = value(args, &mut i, "--listen")?.to_string(),
            "--hold" => {
                if hold.is_some() {
                    return Err(Error::usage("node: pass at most one of --hold or --shard"));
                }
                hold = Some(Hold::Only(parse_hold(value(args, &mut i, "--hold")?)?));
            }
            "--shard" => {
                if hold.is_some() {
                    return Err(Error::usage("node: pass at most one of --hold or --shard"));
                }
                hold = Some(parse_shard(value(args, &mut i, "--shard")?)?);
            }
            other => {
                return Err(Error::usage(format!("node: unexpected argument {other:?}")));
            }
        }
        i += 1;
    }
    let carved = carved.ok_or_else(|| Error::usage("node: --carved is required"))?;
    node::serve(&carved, &listen, hold.as_ref())
}

/// Parse a `--hold` keep-list, e.g. `0:1,1:3` -> `[(0,1),(1,3)]`.
fn parse_hold(s: &str) -> Result<Vec<(u16, u16)>> {
    s.split(',')
        .map(|pair| {
            let (l, e) = pair.trim().split_once(':').ok_or_else(|| {
                Error::usage(format!("--hold: {pair:?} is not a layer:expert pair"))
            })?;
            let layer = l
                .trim()
                .parse::<u16>()
                .map_err(|_| Error::usage(format!("--hold: {l:?} is not a layer id")))?;
            let expert = e
                .trim()
                .parse::<u16>()
                .map_err(|_| Error::usage(format!("--hold: {e:?} is not an expert id")))?;
            Ok((layer, expert))
        })
        .collect()
}

/// Parse a `--shard i/n` spec, requiring `0 <= i < n` and `n >= 1`.
fn parse_shard(s: &str) -> Result<Hold> {
    let (i, n) = s
        .split_once('/')
        .ok_or_else(|| Error::usage(format!("--shard: {s:?} is not i/n")))?;
    let i = i
        .trim()
        .parse::<u16>()
        .map_err(|_| Error::usage(format!("--shard: {i:?} is not a shard index")))?;
    let n = n
        .trim()
        .parse::<u16>()
        .map_err(|_| Error::usage(format!("--shard: {n:?} is not a shard count")))?;
    if n == 0 {
        return Err(Error::usage("--shard: n must be >= 1"));
    }
    if i >= n {
        return Err(Error::usage(format!(
            "--shard: shard index {i} out of range for n={n}"
        )));
    }
    Ok(Hold::Shard { i, n })
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
    let mut replicas = 2usize; // ADR-0009 default r; clamped to the node count
    let mut hedge_ms: Option<u64> = None;
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
            "--replicas" => {
                replicas = parse_num(value(args, &mut i, "--replicas")?, "--replicas", 1, 1 << 16)?
                    as usize;
            }
            "--hedge-ms" => {
                hedge_ms = Some(parse_num(
                    value(args, &mut i, "--hedge-ms")?,
                    "--hedge-ms",
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
    let manifest = Manifest::load(&carved.join(manifest::FILE_NAME))?;
    let identity = *blake3::hash(&manifest.canonical_bytes()).as_bytes();
    let hidden = manifest.model.hidden as usize;
    let spine = Spine::load(&model, &manifest, cfg)?;

    // Validate the codec up front, then hand out fresh boxes on demand — the
    // multi-node placed path needs one codec instance per node connection.
    match codec_name.as_str() {
        "fp8" | "bf16" => {}
        other => {
            return Err(Error::usage(format!(
                "spine: unknown --codec {other:?} (expected fp8 or bf16)"
            )));
        }
    }
    let make_codec = || -> Box<dyn WireCodec> {
        match codec_name.as_str() {
            "bf16" => Box::new(Bf16Codec),
            _ => Box::new(Fp8Codec),
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

    // The replica-set hedge (ADR-0010) is a multi-node knob only: a single node
    // has no second replica, so single-node tail latency is --layer-timeout-ms.
    if hedge_ms.is_some() && nodes.len() < 2 {
        return Err(Error::usage(
            "spine: --hedge-ms applies to multi-node placed dispatch (2+ --node); \
             single-node tail latency is --layer-timeout-ms",
        ));
    }
    let mut dispatcher: Box<dyn spine::Dispatcher> = if local {
        if layer_timeout_ms.is_some() {
            return Err(Error::usage(
                "spine: --layer-timeout-ms applies to --node dispatch, not --local",
            ));
        }
        Box::new(LocalDispatch::new(&carved, make_codec())?)
    } else if nodes.len() == 1 {
        let mut nd = NodeDispatch::connect(&nodes[0], make_codec(), identity, hidden)?;
        if let Some(ms) = layer_timeout_ms {
            nd = nd.with_layer_timeout(std::time::Duration::from_millis(ms));
        }
        Box::new(nd)
    } else {
        // Multi-node PLACED dispatch (ADR-0009 / ADR-0024). The per-layer timeout
        // is a single-node knob; multi-node tail latency is the ADR-0010
        // replica-set hedge (--hedge-ms), so reject the mix rather than silently
        // ignore it and point at the real mitigation.
        if layer_timeout_ms.is_some() {
            return Err(Error::usage(
                "spine: --layer-timeout-ms is single-node; multi-node tail latency is the \
                 ADR-0010 replica-set hedge — use --hedge-ms",
            ));
        }
        // Uniform consistent-hash bootstrap over the node set: no heat log exists
        // yet, so every node is an equal, distinct failure domain and every routed
        // expert is seeded cold (ADR-0009 bootstrap). build_placement then spreads
        // the catalog by rendezvous hash and replicates it `--replicas` ways
        // (clamped to the node count).
        let node_descs: Vec<NodeDesc> = nodes
            .iter()
            .map(|addr| NodeDesc {
                id: addr.clone(),
                failure_domain: addr.clone(),
                uplink_class: 1,
                ram_class: 1,
            })
            .collect();
        let mut heat = HeatMap::new();
        for e in &manifest.experts {
            heat.touch(e.layer, e.expert);
        }
        let map = build_placement(&node_descs, &heat, replicas)?;
        // --hedge-ms arms the replica-set second-send: a stalled primary's experts
        // spill to their next replica, first-answer-wins (ADR-0010 on placement).
        let hedge_delay = hedge_ms.map(std::time::Duration::from_millis);
        Box::new(PlacedDispatch::connect(
            &nodes,
            make_codec,
            identity,
            hidden,
            map,
            hedge_delay,
        )?)
    };

    let (outs, stats) = spine.generate_batch(dispatcher.as_mut(), &prompt_refs, tokens)?;

    let where_ = if local {
        "local (in-process)".to_string()
    } else if nodes.len() == 1 {
        format!("node {}", nodes[0])
    } else {
        format!(
            "{} nodes placed (ADR-0009, r={})",
            nodes.len(),
            replicas.min(nodes.len())
        )
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
    // ADR-0008 dead-replica alarm consumer: surface any expert whose holder(s)
    // answered not-held on every dispatch (a placed multi-node run only).
    let suspects = dispatcher.suspect_replicas();
    if !suspects.is_empty() {
        println!(
            "alarm:    {} dead replica(s) (ADR-0008): {}",
            suspects.len(),
            suspects
                .iter()
                .map(|(l, e)| format!("L{l}/E{e}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
}

fn run_canary(args: &[String]) -> Result<()> {
    let mut carved: Option<PathBuf> = None;
    let mut model: Option<PathBuf> = None;
    let mut opts = canary::CanaryOptions::default();
    let mut rms_eps_ppm = 1u64; // 1e-6, resolved into cfg below
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--carved" => carved = Some(PathBuf::from(value(args, &mut i, "--carved")?)),
            "--model" => model = Some(PathBuf::from(value(args, &mut i, "--model")?)),
            "--prompts" => {
                opts.prompts =
                    parse_num(value(args, &mut i, "--prompts")?, "--prompts", 1, 1 << 16)? as usize;
            }
            "--len" => {
                opts.len = parse_num(value(args, &mut i, "--len")?, "--len", 2, 1 << 16)? as usize;
            }
            "--seed" => {
                opts.seed = parse_num(value(args, &mut i, "--seed")?, "--seed", 0, u64::MAX)?
            }
            "--codec" => opts.codec = value(args, &mut i, "--codec")?.to_string(),
            "--num-heads" => {
                opts.config.num_heads = parse_num(
                    value(args, &mut i, "--num-heads")?,
                    "--num-heads",
                    1,
                    1 << 16,
                )? as usize;
            }
            "--num-kv-heads" => {
                opts.config.num_kv_heads = parse_num(
                    value(args, &mut i, "--num-kv-heads")?,
                    "--num-kv-heads",
                    1,
                    1 << 16,
                )? as usize;
            }
            "--head-dim" => {
                opts.config.head_dim =
                    parse_num(value(args, &mut i, "--head-dim")?, "--head-dim", 2, 1 << 16)?
                        as usize;
            }
            "--rope-theta" => {
                opts.config.rope_theta = parse_num(
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
                opts.config.top_k =
                    parse_num(value(args, &mut i, "--top-k")?, "--top-k", 1, 1 << 16)? as usize;
            }
            other => {
                return Err(Error::usage(format!(
                    "canary: unexpected argument {other:?}"
                )));
            }
        }
        i += 1;
    }
    opts.config.rms_eps = spine::eps_from_ppm(rms_eps_ppm);
    let carved = carved.ok_or_else(|| Error::usage("canary: --carved is required"))?;
    let model = model.ok_or_else(|| Error::usage("canary: --model is required"))?;

    let r = canary::run(&model, &carved, &opts)?;
    println!(
        "canary:   {} sequences x {} tokens ({} scored), seed {}, blobs {}, wire {}",
        r.prompts,
        r.prompt_len,
        r.scored_tokens,
        opts.seed,
        r.dtype.name(),
        r.codec
    );
    println!(
        "ppl:      test ({}) {:.6}   ref (bf16-source) {:.6}",
        r.codec, r.ppl_test, r.ppl_ref
    );
    println!(
        "nll:      test {:.6}   ref {:.6} (nats/token)",
        r.nll_test, r.nll_ref
    );
    println!(
        "delta:    Δppl {:+.6} (test − ref) — ADR-0008 canary / ADR-0018 quality axis",
        r.delta_ppl
    );
    Ok(())
}

fn run_prefix(args: &[String]) -> Result<()> {
    let mut carved: Option<PathBuf> = None;
    let mut opts = prefix::PrefixOptions::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--carved" => carved = Some(PathBuf::from(value(args, &mut i, "--carved")?)),
            "--streams" => {
                opts.streams =
                    parse_num(value(args, &mut i, "--streams")?, "--streams", 1, 1 << 24)? as usize;
            }
            "--system-len" => {
                opts.system_len = parse_num(
                    value(args, &mut i, "--system-len")?,
                    "--system-len",
                    0,
                    1 << 24,
                )? as usize;
            }
            "--user-len" => {
                opts.user_len =
                    parse_num(value(args, &mut i, "--user-len")?, "--user-len", 0, 1 << 24)?
                        as usize;
            }
            "--block" => {
                opts.block_tokens =
                    parse_num(value(args, &mut i, "--block")?, "--block", 1, 1 << 24)? as usize;
            }
            "--seed" => {
                opts.seed = parse_num(value(args, &mut i, "--seed")?, "--seed", 0, u64::MAX)?
            }
            "--vocab" => {
                opts.vocab = parse_num(value(args, &mut i, "--vocab")?, "--vocab", 1, u64::MAX)?;
            }
            "--num-kv-heads" => {
                opts.num_kv_heads = parse_num(
                    value(args, &mut i, "--num-kv-heads")?,
                    "--num-kv-heads",
                    1,
                    1 << 16,
                )? as usize;
            }
            "--head-dim" => {
                opts.head_dim =
                    parse_num(value(args, &mut i, "--head-dim")?, "--head-dim", 1, 1 << 16)?
                        as usize;
            }
            other => {
                return Err(Error::usage(format!(
                    "prefix: unexpected argument {other:?}"
                )));
            }
        }
        i += 1;
    }
    let carved = carved.ok_or_else(|| Error::usage("prefix: --carved is required"))?;

    let r = prefix::run(&carved, &opts)?;
    println!(
        "prefix:   {} streams, system {} + user {} tokens, block {} (seed {})",
        r.streams, r.system_len, r.user_len, r.block_tokens, opts.seed
    );
    println!(
        "hit-rate: {:.4} ({} reused / {} total prompt tokens), {} distinct blocks",
        r.hit_rate, r.reused_prompt_tokens, r.total_prompt_tokens, r.distinct_blocks
    );
    println!(
        "kv:       {} B occupancy (derived: {} streams x {} ctx x {} layers x kv_elem)",
        r.kv_occupancy_bytes,
        r.streams,
        r.system_len + r.user_len,
        r.kv_layers
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
