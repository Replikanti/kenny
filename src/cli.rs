//! Hand-rolled argument parsing (ADR-0021: no CLI framework for a surface
//! this small; the whole grammar fits in one screen of match arms).

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::blob::Dtype;
use crate::carve;
use crate::error::{Error, Result};
use crate::fixture;

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

    kenny carve <model_dir> --out <dir> [--dtype bf16]
                [--model-name NAME] [--model-rev REV]
        Cut routed experts into content-addressed blobs (blobs/<xx>/<cid>)
        and write the canonical manifest (manifest.json). M0 supports bf16
        passthrough; --model-name defaults to the model directory name.
";

pub fn run(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        None | Some("-h" | "--help" | "help") => {
            print!("{USAGE}");
            Ok(())
        }
        Some("fixture") => run_fixture(&args[1..]),
        Some("carve") => run_carve(&args[1..]),
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
