//! `kenny diff` — the M0 validation harness: prove that a carved layer
//! computes the same expert FFN as the source model.
//!
//! For every expert of one MoE layer, run
//! `y = down_proj(silu(gate_proj . x) * (up_proj . x))` (MANIFESTO §4.1) on a
//! deterministic batch of random activations, once with matrices read
//! straight from the source safetensors (bf16 -> f32) and once with matrices
//! reconstructed from the carved blobs (bf16 passthrough or dequantized
//! fp8/int8). Both sides use the identical forward code in the same process,
//! so bf16 passthrough must match BIT-FOR-BIT; quantized carves report
//! max-abs and cosine per expert.
//!
//! Router-weighted mixing is deliberately not diffed here: the router weights
//! live on the spine and are byte-identical by construction (recorded by
//! range, ADR-0005) — mixing them in would only blur per-expert fidelity.

use std::collections::BTreeMap;
use std::path::Path;

use crate::blob::{self, Dtype};
use crate::error::{Error, Result};
use crate::manifest::Manifest;
use crate::quant;
use crate::rng::SplitMix64;
use crate::safetensors::{self, ShardFile};

pub struct DiffOptions {
    pub layer: u16,
    pub batch: usize,
    pub seed: u64,
}

impl Default for DiffOptions {
    fn default() -> Self {
        DiffOptions {
            layer: 0,
            batch: 8,
            seed: 42,
        }
    }
}

#[derive(Debug)]
pub struct ExpertDiff {
    pub expert: u16,
    pub max_abs: f64,
    pub cosine: f64,
}

#[derive(Debug)]
pub struct DiffReport {
    pub layer: u16,
    pub dtype: Dtype,
    pub batch: usize,
    pub bitwise_exact: bool,
    pub worst_max_abs: f64,
    pub worst_cosine: f64,
    pub per_expert: Vec<ExpertDiff>,
}

/// `y = down . (silu(gate . x) * (up . x))`; gate/up are [inter, hidden],
/// down is [hidden, inter], all row-major f32.
fn forward(gate: &[f32], up: &[f32], down: &[f32], hidden: usize, x: &[f32], y: &mut [f32]) {
    let inter = gate.len() / hidden;
    let mut a = vec![0f32; inter];
    for (ar, (grow, urow)) in a
        .iter_mut()
        .zip(gate.chunks_exact(hidden).zip(up.chunks_exact(hidden)))
    {
        let mut g = 0f32;
        let mut u = 0f32;
        for ((&gw, &uw), &xv) in grow.iter().zip(urow).zip(x) {
            g += gw * xv;
            u += uw * xv;
        }
        let silu = g / (1.0 + (-g).exp());
        *ar = silu * u;
    }
    for (yr, drow) in y.iter_mut().zip(down.chunks_exact(inter)) {
        let mut acc = 0f32;
        for (&dw, &av) in drow.iter().zip(&a) {
            acc += dw * av;
        }
        *yr = acc;
    }
}

pub fn run(model_dir: &Path, carved_dir: &Path, opts: &DiffOptions) -> Result<DiffReport> {
    if opts.batch == 0 {
        return Err(Error::usage("diff: --batch must be at least 1"));
    }
    let m = Manifest::load(&carved_dir.join(crate::manifest::FILE_NAME))?;
    let (hidden, inter) = (m.model.hidden as usize, m.model.inter as usize);

    let mut layer_experts: Vec<_> = m.experts.iter().filter(|e| e.layer == opts.layer).collect();
    layer_experts.sort_by_key(|e| e.expert);
    if layer_experts.is_empty() {
        let mut layers: Vec<u16> = m.experts.iter().map(|e| e.layer).collect();
        layers.sort_unstable();
        layers.dedup();
        return Err(Error::usage(format!(
            "diff: layer {} has no experts in this manifest (MoE layers: {layers:?})",
            opts.layer
        )));
    }

    // Source tensors, straight from the safetensors shards.
    let model = safetensors::open_model(model_dir)?;
    let tensor_shard: BTreeMap<&str, &str> = model
        .weight_map
        .iter()
        .map(|(t, s)| (t.as_str(), s.as_str()))
        .collect();
    let mut shards: BTreeMap<String, ShardFile> = BTreeMap::new();
    // The expected shape is part of the lookup: a model dir that does not
    // match the manifest must produce a clean error, never a garbage verdict
    // (or a panic downstream in forward()).
    let mut source_matrix =
        |layer: u16, expert: u16, proj: &str, rows: usize, cols: usize| -> Result<Vec<f32>> {
            let name = format!("model.layers.{layer}.mlp.experts.{expert}.{proj}.weight");
            let shard_name = *tensor_shard.get(name.as_str()).ok_or_else(|| {
                Error::parse(format!("diff: tensor {name:?} not found in the model dir"))
            })?;
            if !shards.contains_key(shard_name) {
                shards.insert(
                    shard_name.to_string(),
                    ShardFile::open(&model.dir.join(shard_name))?,
                );
            }
            let shard = &shards[shard_name];
            let meta = shard.tensor(&name).ok_or_else(|| {
                Error::parse(format!("diff: {name:?} missing from {shard_name:?}"))
            })?;
            if meta.dtype != "BF16" {
                return Err(Error::parse(format!(
                    "diff: {name:?} is {}, expected BF16 sources",
                    meta.dtype
                )));
            }
            if meta.shape != [rows as u64, cols as u64] {
                return Err(Error::parse(format!(
                    "diff: {name:?} has shape {:?}, the manifest implies [{rows}, {cols}] — \
                     is this the model the carve came from?",
                    meta.shape
                )));
            }
            quant::bf16_to_f32_vec(shard.bytes(meta))
        };

    // Deterministic activation batch.
    let xs: Vec<Vec<f32>> = (0..opts.batch)
        .map(|b| {
            let mut rng = SplitMix64::for_name(opts.seed, &format!("diff.x.{b}"));
            (0..hidden).map(|_| rng.next_unit_f32()).collect()
        })
        .collect();

    let mut per_expert = Vec::with_capacity(layer_experts.len());
    let mut bitwise_exact = true;
    for entry in &layer_experts {
        let path = carved_dir.join("blobs").join(blob::rel_path(&entry.cid));
        let bytes = std::fs::read(&path).map_err(|e| Error::io(&path, e))?;
        if blob::cid(&bytes) != entry.cid {
            return Err(Error::parse(format!(
                "diff: blob for expert (layer {}, expert {}) does not hash to its CID — \
                 corrupt store",
                entry.layer, entry.expert
            )));
        }
        let d = blob::decode(&bytes)?;
        if (d.header.layer, d.header.expert) != (entry.layer, entry.expert)
            || (d.header.hidden as usize, d.header.inter as usize) != (hidden, inter)
            || d.header.dtype != m.model.dtype
        {
            return Err(Error::parse(format!(
                "diff: blob header disagrees with the manifest for expert (layer {}, expert {})",
                entry.layer, entry.expert
            )));
        }
        let (cg, cu, cd) = match d.header.dtype {
            Dtype::Bf16 => (
                quant::bf16_to_f32_vec(d.gate)?,
                quant::bf16_to_f32_vec(d.up)?,
                quant::bf16_to_f32_vec(d.down)?,
            ),
            dt => {
                let (sg, su, sd) = d.scale_parts()?;
                (
                    quant::dequantize_matrix(dt, sg, d.gate, inter, hidden)?,
                    quant::dequantize_matrix(dt, su, d.up, inter, hidden)?,
                    quant::dequantize_matrix(dt, sd, d.down, hidden, inter)?,
                )
            }
        };
        let rg = source_matrix(entry.layer, entry.expert, "gate_proj", inter, hidden)?;
        let ru = source_matrix(entry.layer, entry.expert, "up_proj", inter, hidden)?;
        let rd = source_matrix(entry.layer, entry.expert, "down_proj", hidden, inter)?;

        let mut max_abs = 0f64;
        let (mut dot, mut n_ref, mut n_car) = (0f64, 0f64, 0f64);
        let mut y_ref = vec![0f32; hidden];
        let mut y_car = vec![0f32; hidden];
        for x in &xs {
            forward(&rg, &ru, &rd, hidden, x, &mut y_ref);
            forward(&cg, &cu, &cd, hidden, x, &mut y_car);
            for (&a, &b) in y_ref.iter().zip(&y_car) {
                if a.to_bits() != b.to_bits() {
                    bitwise_exact = false;
                }
                max_abs = max_abs.max((a as f64 - b as f64).abs());
                dot += a as f64 * b as f64;
                n_ref += a as f64 * a as f64;
                n_car += b as f64 * b as f64;
            }
        }
        let cosine = if n_ref == 0.0 && n_car == 0.0 {
            1.0
        } else if n_ref == 0.0 || n_car == 0.0 {
            0.0
        } else {
            dot / (n_ref.sqrt() * n_car.sqrt())
        };
        per_expert.push(ExpertDiff {
            expert: entry.expert,
            max_abs,
            cosine,
        });
    }

    let worst_max_abs = per_expert.iter().map(|e| e.max_abs).fold(0.0, f64::max);
    let worst_cosine = per_expert.iter().map(|e| e.cosine).fold(1.0, f64::min);
    Ok(DiffReport {
        layer: opts.layer,
        dtype: m.model.dtype,
        batch: opts.batch,
        bitwise_exact,
        worst_max_abs,
        worst_cosine,
        per_expert,
    })
}
