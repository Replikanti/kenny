//! The manifest — model identity (ADR-0005).
//!
//! Canonical single-line JSON (sorted keys, no whitespace — the src/json.rs
//! writer); blake3 over the exact file bytes IS the model identity. Structure
//! (keys shown in canonical order):
//!
//! ```json
//! {
//!   "codec_version": 1,
//!   "experts": [[layer, expert, "cid"], ...],           // sorted by (layer, expert)
//!   "format": "kenny-manifest",
//!   "model": {"dtype": "bf16", "experts_per_layer": N, "hidden": H,
//!             "inter": I, "moe_layers": L, "name": "...", "revision": "..."},
//!   "spine": [{"cid": "...", "dtype": "BF16", "name": "...",
//!              "offsets": [abs_begin, abs_end], "shape": [...],
//!              "shard": "model-....safetensors"}, ...], // sorted by name
//!   "version": 1
//! }
//! ```
//!
//! Spine offsets are ABSOLUTE byte ranges in the named shard file — for M0
//! the spine payloads stay in the original shards, referenced by range.

use std::path::Path;

use crate::blob::Dtype;
use crate::error::{Error, Result};
use crate::json::{self, Value};

pub const FORMAT: &str = "kenny-manifest";
pub const VERSION: u64 = 1;
pub const CODEC_VERSION: u64 = 1;
pub const FILE_NAME: &str = "manifest.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertEntry {
    pub layer: u16,
    pub expert: u16,
    pub cid: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpineEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub shard: String,
    /// Absolute byte range in the shard file.
    pub begin: u64,
    pub end: u64,
    pub cid: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub name: String,
    pub revision: String,
    pub dtype: Dtype,
    pub hidden: u32,
    pub inter: u32,
    pub moe_layers: u32,
    pub experts_per_layer: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub model: ModelInfo,
    pub experts: Vec<ExpertEntry>,
    pub spine: Vec<SpineEntry>,
}

impl Manifest {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut experts = self.experts.clone();
        experts.sort_by_key(|e| (e.layer, e.expert));
        let experts = Value::Arr(
            experts
                .iter()
                .map(|e| {
                    Value::Arr(vec![
                        Value::Int(e.layer as u64),
                        Value::Int(e.expert as u64),
                        Value::Str(e.cid.clone()),
                    ])
                })
                .collect(),
        );
        let mut spine = self.spine.clone();
        spine.sort_by(|a, b| a.name.cmp(&b.name));
        let spine = Value::Arr(
            spine
                .iter()
                .map(|s| {
                    Value::Obj(vec![
                        ("cid".into(), Value::Str(s.cid.clone())),
                        ("dtype".into(), Value::Str(s.dtype.clone())),
                        ("name".into(), Value::Str(s.name.clone())),
                        (
                            "offsets".into(),
                            Value::Arr(vec![Value::Int(s.begin), Value::Int(s.end)]),
                        ),
                        (
                            "shape".into(),
                            Value::Arr(s.shape.iter().map(|&d| Value::Int(d)).collect()),
                        ),
                        ("shard".into(), Value::Str(s.shard.clone())),
                    ])
                })
                .collect(),
        );
        let model = Value::Obj(vec![
            ("dtype".into(), Value::Str(self.model.dtype.name().into())),
            (
                "experts_per_layer".into(),
                Value::Int(self.model.experts_per_layer as u64),
            ),
            ("hidden".into(), Value::Int(self.model.hidden as u64)),
            ("inter".into(), Value::Int(self.model.inter as u64)),
            (
                "moe_layers".into(),
                Value::Int(self.model.moe_layers as u64),
            ),
            ("name".into(), Value::Str(self.model.name.clone())),
            ("revision".into(), Value::Str(self.model.revision.clone())),
        ]);
        let root = Value::Obj(vec![
            ("codec_version".into(), Value::Int(CODEC_VERSION)),
            ("experts".into(), experts),
            ("format".into(), Value::Str(FORMAT.into())),
            ("model".into(), model),
            ("spine".into(), spine),
            ("version".into(), Value::Int(VERSION)),
        ]);
        json::to_canonical(&root)
    }

    /// blake3 hex of the canonical bytes — the model identity.
    pub fn identity(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.canonical_bytes()).map_err(|e| Error::io(path, e))
    }

    pub fn load(path: &Path) -> Result<Manifest> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        let ctx = path.display();
        let m = Self::from_value(&json::parse(&bytes)?)
            .map_err(|e| Error::parse(format!("{ctx}: {e}")))?;
        Ok(m)
    }

    fn from_value(v: &Value) -> Result<Manifest> {
        let field = |name: &str| {
            v.get(name)
                .ok_or_else(|| Error::parse(format!("manifest: missing {name:?}")))
        };
        if field("format")?.as_str() != Some(FORMAT) {
            return Err(Error::parse("manifest: wrong format tag"));
        }
        if field("version")?.as_u64() != Some(VERSION) {
            return Err(Error::parse(format!(
                "manifest: unsupported version (want {VERSION})"
            )));
        }
        if field("codec_version")?.as_u64() != Some(CODEC_VERSION) {
            return Err(Error::parse(format!(
                "manifest: unsupported codec_version (want {CODEC_VERSION})"
            )));
        }

        let mv = field("model")?;
        let mfield = |name: &str| {
            mv.get(name)
                .ok_or_else(|| Error::parse(format!("manifest: missing model.{name}")))
        };
        let mstr = |name: &str| -> Result<String> {
            Ok(mfield(name)?
                .as_str()
                .ok_or_else(|| Error::parse(format!("manifest: model.{name} is not a string")))?
                .to_string())
        };
        let mu32 = |name: &str| -> Result<u32> {
            mfield(name)?
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| Error::parse(format!("manifest: model.{name} is not a u32")))
        };
        let model = ModelInfo {
            name: mstr("name")?,
            revision: mstr("revision")?,
            dtype: Dtype::from_name(&mstr("dtype")?)?,
            hidden: mu32("hidden")?,
            inter: mu32("inter")?,
            moe_layers: mu32("moe_layers")?,
            experts_per_layer: mu32("experts_per_layer")?,
        };

        let mut experts = Vec::new();
        for e in field("experts")?
            .as_arr()
            .ok_or_else(|| Error::parse("manifest: experts is not an array"))?
        {
            let t = e.as_arr().filter(|t| t.len() == 3).ok_or_else(|| {
                Error::parse("manifest: expert entry is not [layer, expert, cid]")
            })?;
            let layer = t[0]
                .as_u64()
                .and_then(|n| u16::try_from(n).ok())
                .ok_or_else(|| Error::parse("manifest: expert layer out of range"))?;
            let expert = t[1]
                .as_u64()
                .and_then(|n| u16::try_from(n).ok())
                .ok_or_else(|| Error::parse("manifest: expert index out of range"))?;
            let cid = t[2]
                .as_str()
                .ok_or_else(|| Error::parse("manifest: expert cid is not a string"))?;
            check_cid(cid)?;
            experts.push(ExpertEntry {
                layer,
                expert,
                cid: cid.to_string(),
            });
        }
        let expect = model.moe_layers as u64 * model.experts_per_layer as u64;
        if experts.len() as u64 != expect {
            return Err(Error::parse(format!(
                "manifest: {} expert entries, model block promises {expect}",
                experts.len()
            )));
        }
        let mut seen: Vec<(u16, u16)> = experts.iter().map(|e| (e.layer, e.expert)).collect();
        seen.sort_unstable();
        if seen.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::parse("manifest: duplicate (layer, expert) entry"));
        }

        let mut spine = Vec::new();
        for s in field("spine")?
            .as_arr()
            .ok_or_else(|| Error::parse("manifest: spine is not an array"))?
        {
            let sfield = |name: &str| {
                s.get(name)
                    .ok_or_else(|| Error::parse(format!("manifest: spine entry missing {name:?}")))
            };
            let name = sfield("name")?
                .as_str()
                .ok_or_else(|| Error::parse("manifest: spine name is not a string"))?
                .to_string();
            let cid = sfield("cid")?.as_str().ok_or_else(|| {
                Error::parse(format!("manifest: spine {name:?}: cid not a string"))
            })?;
            check_cid(cid)?;
            let offs = sfield("offsets")?
                .as_arr()
                .filter(|o| o.len() == 2)
                .ok_or_else(|| Error::parse(format!("manifest: spine {name:?}: bad offsets")))?;
            let begin = offs[0]
                .as_u64()
                .ok_or_else(|| Error::parse(format!("manifest: spine {name:?}: bad offset")))?;
            let end = offs[1]
                .as_u64()
                .ok_or_else(|| Error::parse(format!("manifest: spine {name:?}: bad offset")))?;
            if begin > end {
                return Err(Error::parse(format!(
                    "manifest: spine {name:?}: begin > end"
                )));
            }
            let shape = sfield("shape")?
                .as_arr()
                .ok_or_else(|| Error::parse(format!("manifest: spine {name:?}: bad shape")))?
                .iter()
                .map(|d| {
                    d.as_u64()
                        .ok_or_else(|| Error::parse(format!("manifest: spine {name:?}: bad dim")))
                })
                .collect::<Result<Vec<u64>>>()?;
            let dtype = sfield("dtype")?
                .as_str()
                .ok_or_else(|| Error::parse(format!("manifest: spine {name:?}: bad dtype")))?
                .to_string();
            let shard = sfield("shard")?
                .as_str()
                .ok_or_else(|| Error::parse(format!("manifest: spine {name:?}: bad shard")))?
                .to_string();
            spine.push(SpineEntry {
                name,
                dtype,
                shape,
                shard,
                begin,
                end,
                cid: cid.to_string(),
            });
        }

        Ok(Manifest {
            model,
            experts,
            spine,
        })
    }
}

fn check_cid(cid: &str) -> Result<()> {
    let ok = cid.len() == 64 && cid.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if ok {
        Ok(())
    } else {
        Err(Error::parse(format!("manifest: malformed cid {cid:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid_of(tag: u8) -> String {
        blake3::hash(&[tag]).to_hex().to_string()
    }

    fn sample() -> Manifest {
        Manifest {
            model: ModelInfo {
                name: "fixture".into(),
                revision: String::new(),
                dtype: Dtype::Bf16,
                hidden: 8,
                inter: 4,
                moe_layers: 1,
                experts_per_layer: 2,
            },
            // Deliberately unsorted — canonical_bytes must sort.
            experts: vec![
                ExpertEntry {
                    layer: 0,
                    expert: 1,
                    cid: cid_of(1),
                },
                ExpertEntry {
                    layer: 0,
                    expert: 0,
                    cid: cid_of(0),
                },
            ],
            spine: vec![
                SpineEntry {
                    name: "model.norm.weight".into(),
                    dtype: "BF16".into(),
                    shape: vec![8],
                    shard: "model-00002-of-00002.safetensors".into(),
                    begin: 100,
                    end: 116,
                    cid: cid_of(2),
                },
                SpineEntry {
                    name: "lm_head.weight".into(),
                    dtype: "BF16".into(),
                    shape: vec![32, 8],
                    shard: "model-00002-of-00002.safetensors".into(),
                    begin: 116,
                    end: 628,
                    cid: cid_of(3),
                },
            ],
        }
    }

    #[test]
    fn canonical_roundtrip_is_fixed_point() {
        let m = sample();
        let bytes = m.canonical_bytes();
        let reparsed = Manifest::from_value(&json::parse(&bytes).unwrap()).unwrap();
        assert_eq!(reparsed.canonical_bytes(), bytes);
        // Entry order in the struct must not affect identity.
        let mut swapped = m.clone();
        swapped.experts.reverse();
        swapped.spine.reverse();
        assert_eq!(swapped.identity(), m.identity());
    }

    #[test]
    fn identity_tracks_content() {
        let m = sample();
        let mut changed = m.clone();
        changed.experts[0].cid = cid_of(9);
        assert_ne!(changed.identity(), m.identity());
    }

    #[test]
    fn load_validates() {
        let m = sample();
        let mut wrong_count = m.clone();
        wrong_count.model.experts_per_layer = 3;
        let v = json::parse(&wrong_count.canonical_bytes()).unwrap();
        assert!(Manifest::from_value(&v).is_err(), "count mismatch");

        let mut dup = m.clone();
        dup.experts[1] = dup.experts[0].clone();
        let v = json::parse(&dup.canonical_bytes()).unwrap();
        assert!(Manifest::from_value(&v).is_err(), "duplicate entry");

        let mut bad_cid = m;
        bad_cid.experts[0].cid = "zz".into();
        let v = json::parse(&bad_cid.canonical_bytes()).unwrap();
        assert!(Manifest::from_value(&v).is_err(), "malformed cid");
    }
}
