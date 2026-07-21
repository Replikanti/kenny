//! Synthetic safetensors model matching the Qwen3-30B-A3B naming schema.
//!
//! Every unit and CI test runs on this — CI never downloads a model (§repo
//! conventions). Deterministic by construction: tensor values come from a
//! SplitMix64 stream keyed by (seed, tensor name), so bytes are independent
//! of generation order and identical across runs and machines.
//!
//! The tensor set mirrors the real schema shape: router gate + experts per
//! MoE layer, attention projections/norms, embeddings, final norm, lm_head,
//! split across two shards plus an index — the multi-shard code path is
//! always exercised.

use std::path::Path;

use crate::bf16::f32_to_bf16;
use crate::error::{Error, Result};
use crate::rng::SplitMix64;
use crate::safetensors::{self, TensorPayload};

#[derive(Debug, Clone)]
pub struct Params {
    pub layers: u16,
    pub experts: u16,
    pub hidden: u32,
    pub inter: u32,
    pub vocab: u32,
    pub seed: u64,
}

impl Default for Params {
    /// The M0 default: 2 layers x 4 experts x hidden 8 x inter 4.
    fn default() -> Self {
        Params {
            layers: 2,
            experts: 4,
            hidden: 8,
            inter: 4,
            vocab: 32,
            seed: 42,
        }
    }
}

#[derive(Debug)]
pub struct Summary {
    pub shards: usize,
    pub tensors: usize,
    pub bytes: u64,
}

pub fn generate(p: &Params, dir: &Path) -> Result<Summary> {
    if p.layers == 0 || p.experts == 0 || p.hidden == 0 || p.inter == 0 || p.vocab == 0 {
        return Err(Error::usage("fixture: all dimensions must be nonzero"));
    }
    std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;

    let (h, i, v, e) = (
        p.hidden as u64,
        p.inter as u64,
        p.vocab as u64,
        p.experts as u64,
    );
    let layer_tensors = |l: u16| -> Vec<(String, Vec<u64>)> {
        let base = format!("model.layers.{l}");
        let mut t = vec![
            (format!("{base}.input_layernorm.weight"), vec![h]),
            (format!("{base}.self_attn.q_proj.weight"), vec![h, h]),
            (format!("{base}.self_attn.k_proj.weight"), vec![h, h]),
            (format!("{base}.self_attn.v_proj.weight"), vec![h, h]),
            (format!("{base}.self_attn.o_proj.weight"), vec![h, h]),
            (format!("{base}.self_attn.q_norm.weight"), vec![h]),
            (format!("{base}.self_attn.k_norm.weight"), vec![h]),
            (format!("{base}.post_attention_layernorm.weight"), vec![h]),
            (format!("{base}.mlp.gate.weight"), vec![e, h]),
        ];
        for ex in 0..p.experts {
            t.push((
                format!("{base}.mlp.experts.{ex}.gate_proj.weight"),
                vec![i, h],
            ));
            t.push((
                format!("{base}.mlp.experts.{ex}.up_proj.weight"),
                vec![i, h],
            ));
            t.push((
                format!("{base}.mlp.experts.{ex}.down_proj.weight"),
                vec![h, i],
            ));
        }
        t
    };

    // Shard 1: embeddings + the first half of the layers.
    // Shard 2: the rest + final norm + lm_head. Always two shards so the
    // index/multi-shard path is exercised even at fixture scale.
    let split = p.layers / 2;
    let mut shard1: Vec<(String, Vec<u64>)> =
        vec![("model.embed_tokens.weight".into(), vec![v, h])];
    for l in 0..split {
        shard1.extend(layer_tensors(l));
    }
    let mut shard2: Vec<(String, Vec<u64>)> = Vec::new();
    for l in split..p.layers {
        shard2.extend(layer_tensors(l));
    }
    shard2.push(("model.norm.weight".into(), vec![h]));
    shard2.push(("lm_head.weight".into(), vec![v, h]));

    let shard_files = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    let mut weight_map: Vec<(String, String)> = Vec::new();
    let mut total_bytes = 0u64;
    let mut total_tensors = 0usize;

    for (file, tensors) in shard_files.iter().zip([&shard1, &shard2]) {
        let payloads: Vec<TensorPayload> = tensors
            .iter()
            .map(|(name, shape)| {
                let elems: u64 = shape.iter().product();
                let mut rng = SplitMix64::for_name(p.seed, name);
                let mut data = Vec::with_capacity((elems * 2) as usize);
                for _ in 0..elems {
                    data.extend_from_slice(&f32_to_bf16(rng.next_unit_f32()).to_le_bytes());
                }
                weight_map.push((name.clone(), file.to_string()));
                total_bytes += elems * 2;
                total_tensors += 1;
                TensorPayload {
                    name: name.clone(),
                    dtype: "BF16",
                    shape: shape.clone(),
                    data,
                }
            })
            .collect();
        safetensors::write_shard(&dir.join(file), &payloads)?;
    }

    weight_map.sort_by(|a, b| a.0.cmp(&b.0));
    safetensors::write_index(&dir.join(safetensors::INDEX_FILE), &weight_map, total_bytes)?;

    Ok(Summary {
        shards: shard_files.len(),
        tensors: total_tensors,
        bytes: total_bytes,
    })
}
