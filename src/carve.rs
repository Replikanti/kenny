//! `kenny carve` — cut a safetensors model into content-addressed expert
//! blobs plus the canonical manifest (ADR-0005, ADR-0012).
//!
//! M0 scope: bf16 passthrough (byte-exact slicing, no numeric
//! interpretation). Everything NOT matching the expert pattern is spine-side:
//! recorded in the manifest by CID + absolute byte range, payload left in the
//! original shards.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use crate::blob::{self, Dtype, Header};
use crate::error::{Error, Result};
use crate::manifest::{ExpertEntry, Manifest, ModelInfo, SpineEntry};
use crate::natsort::natural_cmp;
use crate::quant;
use crate::safetensors::{self, ShardFile};

pub struct Options {
    pub out: PathBuf,
    pub model_name: String,
    pub model_rev: String,
    pub dtype: Dtype,
}

#[derive(Debug)]
pub struct CarveSummary {
    pub blobs: usize,
    pub blob_bytes: u64,
    pub dedup_skipped: usize,
    pub spine_tensors: usize,
    pub moe_layers: u32,
    pub experts_per_layer: u32,
    pub hidden: u32,
    pub inter: u32,
    pub manifest_path: PathBuf,
    pub manifest_identity: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Proj {
    Gate = 0,
    Up = 1,
    Down = 2,
}

enum Class {
    Expert { layer: u16, expert: u16, proj: Proj },
    Spine,
}

/// (tensor name, shard name) for each of gate/up/down, keyed by (layer, expert).
type ExpertGrid = BTreeMap<(u16, u16), [Option<(String, String)>; 3]>;

/// Strict schema match for `model.layers.{L}.mlp.experts.{E}.{proj}.weight`.
/// Anything else touching `.mlp.experts.` is a hard error — an unrecognized
/// expert-family tensor (quant scales, renamed projections) must stop the
/// carve, not silently land on the spine ("trust but verify", ADR-0007).
/// Digits only, no leading zeros, no signs — `u16::parse` alone would accept
/// "007" and "+7", which could never round-trip back into a tensor name.
fn parse_index(s: &str) -> Option<u16> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if s.len() > 1 && s.starts_with('0') {
        return None;
    }
    s.parse::<u16>().ok()
}

fn classify(name: &str) -> Result<Class> {
    let segs: Vec<&str> = name.split('.').collect();
    if segs.len() == 8
        && segs[0] == "model"
        && segs[1] == "layers"
        && segs[3] == "mlp"
        && segs[4] == "experts"
        && segs[7] == "weight"
    {
        let proj = match segs[6] {
            "gate_proj" => Some(Proj::Gate),
            "up_proj" => Some(Proj::Up),
            "down_proj" => Some(Proj::Down),
            _ => None,
        };
        if let (Some(layer), Some(expert), Some(proj)) =
            (parse_index(segs[2]), parse_index(segs[5]), proj)
        {
            return Ok(Class::Expert {
                layer,
                expert,
                proj,
            });
        }
    }
    if name.contains(".mlp.experts.") {
        return Err(Error::parse(format!(
            "unrecognized expert-family tensor {name:?} — the naming schema differs from what \
             this carve understands; run --dump-names and extend the matcher deliberately"
        )));
    }
    Ok(Class::Spine)
}

/// Tensor names with their shard, natural-sorted — `--dump-names`.
pub fn dump_names(model_dir: &Path) -> Result<Vec<(String, String)>> {
    let model = safetensors::open_model(model_dir)?;
    let mut names = model.weight_map;
    names.sort_by(|a, b| natural_cmp(&a.0, &b.0));
    Ok(names)
}

pub fn run(model_dir: &Path, opts: &Options) -> Result<CarveSummary> {
    let model = safetensors::open_model(model_dir)?;

    // Open every referenced shard once; look tensors up through the index.
    let mut shards: HashMap<String, ShardFile> = HashMap::new();
    for (_, shard_name) in &model.weight_map {
        if !shards.contains_key(shard_name) {
            let shard = ShardFile::open(&model.dir.join(shard_name))?;
            shards.insert(shard_name.clone(), shard);
        }
    }
    let locate =
        |tensor: &str, shard_name: &str| -> Result<(&ShardFile, &safetensors::TensorMeta)> {
            let shard = shards
                .get(shard_name)
                .expect("shard opened above; weight_map is the only source of names");
            let meta = shard.tensor(tensor).ok_or_else(|| {
                Error::parse(format!(
                    "tensor {tensor:?} listed in the index but missing from shard {shard_name:?}"
                ))
            })?;
            Ok((shard, meta))
        };

    // Classify and group.
    let mut experts: ExpertGrid = ExpertGrid::new();
    let mut spine_names: Vec<(String, String)> = Vec::new();
    for (tensor, shard_name) in &model.weight_map {
        match classify(tensor)? {
            Class::Expert {
                layer,
                expert,
                proj,
            } => {
                let slot = &mut experts.entry((layer, expert)).or_default()[proj as usize];
                if slot.is_some() {
                    return Err(Error::parse(format!("duplicate expert tensor {tensor:?}")));
                }
                *slot = Some((tensor.clone(), shard_name.clone()));
            }
            Class::Spine => spine_names.push((tensor.clone(), shard_name.clone())),
        }
    }
    if experts.is_empty() {
        return Err(Error::parse(format!(
            "{}: no routed experts found — nothing to carve",
            model_dir.display()
        )));
    }

    // Validate triples, dims, dtypes, and grid uniformity up front (metadata
    // only — cheap even for GLM-scale models).
    let mut hidden_inter: Option<(u32, u32)> = None;
    for (&(layer, expert), triple) in &experts {
        let names = ["gate_proj", "up_proj", "down_proj"];
        let mut dims = [(0u64, 0u64); 3];
        for (i, slot) in triple.iter().enumerate() {
            let (tensor, shard_name) = slot.as_ref().ok_or_else(|| {
                Error::parse(format!(
                    "expert (layer {layer}, expert {expert}) is missing its {} tensor",
                    names[i]
                ))
            })?;
            let (_, meta) = locate(tensor, shard_name)?;
            if meta.dtype != opts.dtype.source_dtype() {
                return Err(Error::parse(format!(
                    "tensor {tensor:?} is {}, {} carve expects {} sources",
                    meta.dtype,
                    opts.dtype.name(),
                    opts.dtype.source_dtype()
                )));
            }
            if meta.shape.len() != 2 {
                return Err(Error::parse(format!("tensor {tensor:?}: expected 2 dims")));
            }
            dims[i] = (meta.shape[0], meta.shape[1]);
        }
        // gate/up are [inter, hidden]; down is [hidden, inter].
        let (i0, h0) = dims[0];
        if dims[1] != (i0, h0) || dims[2] != (h0, i0) {
            return Err(Error::parse(format!(
                "expert (layer {layer}, expert {expert}): inconsistent shapes \
                 gate {:?} up {:?} down {:?}",
                dims[0], dims[1], dims[2]
            )));
        }
        let (h, i) = (
            u32::try_from(h0).map_err(|_| Error::parse("hidden exceeds u32"))?,
            u32::try_from(i0).map_err(|_| Error::parse("inter exceeds u32"))?,
        );
        match hidden_inter {
            None => hidden_inter = Some((h, i)),
            Some(prev) if prev != (h, i) => {
                return Err(Error::parse(format!(
                    "expert (layer {layer}, expert {expert}) has dims ({h}, {i}), \
                     earlier experts had {prev:?}"
                )));
            }
            _ => {}
        }
    }
    let (hidden, inter) = hidden_inter.expect("experts is non-empty");

    // Uniform, gap-free expert grid: every MoE layer carries experts 0..=max.
    let mut per_layer: BTreeMap<u16, Vec<u16>> = BTreeMap::new();
    for &(layer, expert) in experts.keys() {
        per_layer.entry(layer).or_default().push(expert);
    }
    let mut experts_per_layer: Option<usize> = None;
    for (layer, list) in &per_layer {
        let n = list.len();
        // BTreeMap iteration is sorted, so the list is ascending already.
        if list[0] != 0 || list[n - 1] as usize != n - 1 {
            return Err(Error::parse(format!(
                "layer {layer}: expert indices are not contiguous from 0 ({n} tensors, \
                 max index {})",
                list[n - 1]
            )));
        }
        match experts_per_layer {
            None => experts_per_layer = Some(n),
            Some(prev) if prev != n => {
                return Err(Error::parse(format!(
                    "layer {layer} has {n} experts, earlier layers had {prev} — ragged grids \
                     are not supported"
                )));
            }
            _ => {}
        }
    }
    let experts_per_layer = experts_per_layer.expect("non-empty") as u32;
    let moe_layers = per_layer.len() as u32;

    // Write blobs — parallel over experts; writes go to distinct
    // content-addressed paths, identical content is skipped (dedup).
    let blobs_dir = opts.out.join("blobs");
    std::fs::create_dir_all(&blobs_dir).map_err(|e| Error::io(&blobs_dir, e))?;

    struct WorkerOut {
        entries: Vec<ExpertEntry>,
        bytes_written: u64,
        dedup_skipped: usize,
    }

    let keys: Vec<(u16, u16)> = experts.keys().copied().collect();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(keys.len().max(1));
    let chunk = keys.len().div_ceil(threads);

    let write_expert =
        |&(layer, expert): &(u16, u16), worker: usize| -> Result<(ExpertEntry, u64, bool)> {
            let triple = &experts[&(layer, expert)];
            let mut srcs: [(&[u8], usize, usize); 3] = [(&[], 0, 0); 3];
            for (i, slot) in triple.iter().enumerate() {
                let (tensor, shard_name) = slot.as_ref().expect("validated above");
                let (shard, meta) = locate(tensor, shard_name)?;
                srcs[i] = (
                    shard.bytes(meta),
                    meta.shape[0] as usize,
                    meta.shape[1] as usize,
                );
            }
            let header = Header {
                layer,
                expert,
                dtype: opts.dtype,
                hidden,
                inter,
            };
            let bytes = match opts.dtype {
                Dtype::Bf16 => blob::encode(&header, &[], srcs[0].0, srcs[1].0, srcs[2].0)?,
                dt => {
                    // Central quantization (ADR-0012): per-output-row scales,
                    // concatenated gate ++ up ++ down in the scale block.
                    let mut scale_block = Vec::new();
                    let mut mats: Vec<Vec<u8>> = Vec::with_capacity(3);
                    for &(src, rows, cols) in &srcs {
                        let (scales, data) = quant::quantize_matrix(dt, src, rows, cols)?;
                        scale_block.extend_from_slice(&scales);
                        mats.push(data);
                    }
                    blob::encode(&header, &scale_block, &mats[0], &mats[1], &mats[2])?
                }
            };
            let cid = blob::cid(&bytes);
            let path = blobs_dir.join(blob::rel_path(&cid));
            let entry = ExpertEntry { layer, expert, cid };
            if path.exists() {
                return Ok((entry, 0, true));
            }
            let dir = path.parent().expect("rel_path has a parent");
            std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
            // Temp-then-rename: concurrent writers of the same CID race benignly
            // (same bytes), and a crashed carve never leaves a torn blob at a
            // content-addressed path.
            let tmp = path.with_extension(format!("tmp{worker}"));
            std::fs::write(&tmp, &bytes).map_err(|e| Error::io(&tmp, e))?;
            std::fs::rename(&tmp, &path).map_err(|e| Error::io(&path, e))?;
            Ok((entry, bytes.len() as u64, false))
        };

    let outputs: Vec<WorkerOut> = std::thread::scope(|scope| {
        let handles: Vec<_> = keys
            .chunks(chunk.max(1))
            .enumerate()
            .map(|(worker, chunk_keys)| {
                let write_expert = &write_expert;
                scope.spawn(move || -> Result<WorkerOut> {
                    let mut out = WorkerOut {
                        entries: Vec::new(),
                        bytes_written: 0,
                        dedup_skipped: 0,
                    };
                    for key in chunk_keys {
                        let (entry, bytes, dedup) = write_expert(key, worker)?;
                        out.entries.push(entry);
                        out.bytes_written += bytes;
                        if dedup {
                            out.dedup_skipped += 1;
                        }
                    }
                    Ok(out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("carve worker panicked"))
            .collect::<Result<Vec<WorkerOut>>>()
    })?;

    let mut expert_entries = Vec::with_capacity(keys.len());
    let mut blob_bytes = 0u64;
    let mut dedup_skipped = 0usize;
    for out in outputs {
        expert_entries.extend(out.entries);
        blob_bytes += out.bytes_written;
        dedup_skipped += out.dedup_skipped;
    }

    // Spine tensors: record CID + absolute range, payload stays in the shard.
    let mut spine_entries = Vec::with_capacity(spine_names.len());
    for (tensor, shard_name) in &spine_names {
        let (shard, meta) = locate(tensor, shard_name)?;
        let bytes = shard.bytes(meta);
        let (begin, end) = shard.abs_range(meta);
        spine_entries.push(SpineEntry {
            name: tensor.clone(),
            dtype: meta.dtype.clone(),
            shape: meta.shape.clone(),
            shard: shard_name.clone(),
            begin,
            end,
            cid: blake3::hash(bytes).to_hex().to_string(),
        });
    }

    let manifest = Manifest {
        model: ModelInfo {
            name: opts.model_name.clone(),
            revision: opts.model_rev.clone(),
            dtype: opts.dtype,
            hidden,
            inter,
            moe_layers,
            experts_per_layer,
        },
        experts: expert_entries,
        spine: spine_entries,
    };
    let manifest_path = opts.out.join(crate::manifest::FILE_NAME);
    manifest.write(&manifest_path)?;

    Ok(CarveSummary {
        blobs: keys.len(),
        blob_bytes,
        dedup_skipped,
        spine_tensors: manifest.spine.len(),
        moe_layers,
        experts_per_layer,
        hidden,
        inter,
        manifest_path,
        manifest_identity: manifest.identity(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_expert_names() {
        match classify("model.layers.7.mlp.experts.127.gate_proj.weight").unwrap() {
            Class::Expert {
                layer: 7,
                expert: 127,
                proj: Proj::Gate,
            } => {}
            _ => panic!("expected expert classification"),
        }
        match classify("model.layers.0.mlp.experts.0.down_proj.weight").unwrap() {
            Class::Expert {
                proj: Proj::Down, ..
            } => {}
            _ => panic!("expected down_proj"),
        }
    }

    #[test]
    fn classify_spine_names() {
        for name in [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.norm.weight",
            "lm_head.weight",
        ] {
            assert!(matches!(classify(name).unwrap(), Class::Spine), "{name}");
        }
    }

    #[test]
    fn classify_rejects_unknown_expert_family() {
        for name in [
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
            "model.layers.0.mlp.experts.0.qkv_proj.weight",
            "model.layers.0.mlp.experts.70000.gate_proj.weight",
            "model.layers.007.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.+7.gate_proj.weight",
        ] {
            assert!(classify(name).is_err(), "{name}");
        }
    }
}
