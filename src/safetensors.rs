//! safetensors reading (mmap + exact byte ranges) and writing (fixtures).
//!
//! File layout: `u64 LE header_len`, then `header_len` bytes of JSON mapping
//! `tensor name -> {dtype, shape, data_offsets}` (optionally padded with
//! spaces), then the raw payload. `data_offsets` are relative to the payload
//! start (byte `8 + header_len`). A multi-shard model adds
//! `model.safetensors.index.json` with a `weight_map` of tensor -> shard file.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::json::{self, Value};

/// Headers beyond this are treated as corruption, not data.
const MAX_HEADER_LEN: u64 = 256 * 1024 * 1024;

pub const INDEX_FILE: &str = "model.safetensors.index.json";
pub const SINGLE_FILE: &str = "model.safetensors";

pub fn dtype_size(dtype: &str) -> Result<u64> {
    Ok(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "F8_E4M3" | "F8_E5M2" | "I8" | "U8" | "BOOL" => 1,
        other => return Err(Error::parse(format!("unknown safetensors dtype {other:?}"))),
    })
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Byte range relative to the payload section.
    pub begin: u64,
    pub end: u64,
}

pub struct ShardFile {
    pub path: PathBuf,
    mmap: Mmap,
    pub data_offset: u64,
    tensors: Vec<TensorMeta>,
    by_name: HashMap<String, usize>,
}

impl ShardFile {
    pub fn open(path: &Path) -> Result<ShardFile> {
        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        // SAFETY: read-only mapping of a local file we treat as immutable for
        // the lifetime of the carve. Concurrent truncation by another process
        // would be UB; an accepted risk for an offline CLI over local model
        // files (the alternative is copying multi-GB shards through read()).
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(path, e))?;
        let ctx = path.display();
        if mmap.len() < 8 {
            return Err(Error::parse(format!(
                "{ctx}: file too short for safetensors"
            )));
        }
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().expect("8-byte slice"));
        if header_len > MAX_HEADER_LEN || 8 + header_len > mmap.len() as u64 {
            return Err(Error::parse(format!(
                "{ctx}: implausible header length {header_len}"
            )));
        }
        let data_offset = 8 + header_len;
        let data_len = mmap.len() as u64 - data_offset;
        let header = json::parse(&mmap[8..data_offset as usize])
            .map_err(|e| Error::parse(format!("{ctx}: {e}")))?;
        let tensors =
            parse_header(&header, data_len).map_err(|e| Error::parse(format!("{ctx}: {e}")))?;
        let by_name = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Ok(ShardFile {
            path: path.to_path_buf(),
            mmap,
            data_offset,
            tensors,
            by_name,
        })
    }

    pub fn tensors(&self) -> &[TensorMeta] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    pub fn bytes(&self, t: &TensorMeta) -> &[u8] {
        let begin = (self.data_offset + t.begin) as usize;
        let end = (self.data_offset + t.end) as usize;
        &self.mmap[begin..end]
    }

    /// Absolute byte range within the shard file (self-sufficient for later
    /// pread by the spine — recorded in the manifest).
    pub fn abs_range(&self, t: &TensorMeta) -> (u64, u64) {
        (self.data_offset + t.begin, self.data_offset + t.end)
    }
}

/// Header validation shared by the mmap path and unit tests.
fn parse_header(header: &Value, data_len: u64) -> Result<Vec<TensorMeta>> {
    let entries = header
        .as_obj()
        .ok_or_else(|| Error::parse("header is not a json object"))?;
    let mut tensors = Vec::new();
    for (name, spec) in entries {
        if name == "__metadata__" {
            spec.as_obj()
                .ok_or_else(|| Error::parse("__metadata__ is not an object"))?;
            continue;
        }
        let obj = spec
            .as_obj()
            .ok_or_else(|| Error::parse(format!("tensor {name:?}: spec is not an object")))?;
        for (k, _) in obj {
            if !matches!(k.as_str(), "dtype" | "shape" | "data_offsets") {
                return Err(Error::parse(format!("tensor {name:?}: unknown key {k:?}")));
            }
        }
        let dtype = spec
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::parse(format!("tensor {name:?}: missing dtype")))?
            .to_string();
        let dsize =
            dtype_size(&dtype).map_err(|e| Error::parse(format!("tensor {name:?}: {e}")))?;
        let shape: Vec<u64> = spec
            .get("shape")
            .and_then(Value::as_arr)
            .ok_or_else(|| Error::parse(format!("tensor {name:?}: missing shape")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| Error::parse(format!("tensor {name:?}: non-integer dim")))
            })
            .collect::<Result<_>>()?;
        let offs = spec
            .get("data_offsets")
            .and_then(Value::as_arr)
            .ok_or_else(|| Error::parse(format!("tensor {name:?}: missing data_offsets")))?;
        if offs.len() != 2 {
            return Err(Error::parse(format!(
                "tensor {name:?}: data_offsets must be [begin, end]"
            )));
        }
        let begin = offs[0]
            .as_u64()
            .ok_or_else(|| Error::parse(format!("tensor {name:?}: non-integer offset")))?;
        let end = offs[1]
            .as_u64()
            .ok_or_else(|| Error::parse(format!("tensor {name:?}: non-integer offset")))?;
        if begin > end || end > data_len {
            return Err(Error::parse(format!(
                "tensor {name:?}: offsets [{begin}, {end}] out of bounds (data len {data_len})"
            )));
        }
        let elems = shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| Error::parse(format!("tensor {name:?}: shape overflows u64")))?;
        let expect = elems
            .checked_mul(dsize)
            .ok_or_else(|| Error::parse(format!("tensor {name:?}: byte size overflows u64")))?;
        if expect != end - begin {
            return Err(Error::parse(format!(
                "tensor {name:?}: shape {shape:?} x {dtype} = {expect} bytes, offsets give {}",
                end - begin
            )));
        }
        tensors.push(TensorMeta {
            name: name.clone(),
            dtype,
            shape,
            begin,
            end,
        });
    }
    Ok(tensors)
}

/// A model directory: tensor name -> shard file name.
pub struct ModelDir {
    pub dir: PathBuf,
    pub weight_map: Vec<(String, String)>,
}

pub fn open_model(dir: &Path) -> Result<ModelDir> {
    let index_path = dir.join(INDEX_FILE);
    if index_path.is_file() {
        let bytes = std::fs::read(&index_path).map_err(|e| Error::io(&index_path, e))?;
        let v = json::parse(&bytes)
            .map_err(|e| Error::parse(format!("{}: {e}", index_path.display())))?;
        let map = v.get("weight_map").and_then(Value::as_obj).ok_or_else(|| {
            Error::parse(format!(
                "{}: missing weight_map object",
                index_path.display()
            ))
        })?;
        let mut weight_map = Vec::with_capacity(map.len());
        for (tensor, shard) in map {
            let shard = shard.as_str().ok_or_else(|| {
                Error::parse(format!(
                    "{}: weight_map[{tensor:?}] is not a string",
                    index_path.display()
                ))
            })?;
            weight_map.push((tensor.clone(), shard.to_string()));
        }
        return Ok(ModelDir {
            dir: dir.to_path_buf(),
            weight_map,
        });
    }
    let single = dir.join(SINGLE_FILE);
    if single.is_file() {
        let shard = ShardFile::open(&single)?;
        let weight_map = shard
            .tensors()
            .iter()
            .map(|t| (t.name.clone(), SINGLE_FILE.to_string()))
            .collect();
        return Ok(ModelDir {
            dir: dir.to_path_buf(),
            weight_map,
        });
    }
    Err(Error::parse(format!(
        "{}: neither {INDEX_FILE} nor {SINGLE_FILE} found — not a safetensors model dir",
        dir.display()
    )))
}

/// One tensor to be written into a fixture shard.
pub struct TensorPayload {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

pub fn write_shard(path: &Path, tensors: &[TensorPayload]) -> Result<()> {
    let mut entries: Vec<(String, Value)> = Vec::with_capacity(tensors.len());
    let mut offset = 0u64;
    for t in tensors {
        let elems = t
            .shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| Error::parse(format!("tensor {:?}: shape overflow", t.name)))?;
        let expect = elems * dtype_size(t.dtype)?;
        if expect != t.data.len() as u64 {
            return Err(Error::parse(format!(
                "tensor {:?}: {} data bytes but shape says {expect}",
                t.name,
                t.data.len()
            )));
        }
        let end = offset + expect;
        entries.push((
            t.name.clone(),
            Value::Obj(vec![
                ("dtype".into(), Value::Str(t.dtype.into())),
                (
                    "shape".into(),
                    Value::Arr(t.shape.iter().map(|&d| Value::Int(d)).collect()),
                ),
                (
                    "data_offsets".into(),
                    Value::Arr(vec![Value::Int(offset), Value::Int(end)]),
                ),
            ]),
        ));
        offset = end;
    }
    let mut header = json::to_canonical(&Value::Obj(entries));
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let file = File::create(path).map_err(|e| Error::io(path, e))?;
    let mut w = BufWriter::new(file);
    let io = |e| Error::io(path, e);
    w.write_all(&(header.len() as u64).to_le_bytes())
        .map_err(io)?;
    w.write_all(&header).map_err(io)?;
    for t in tensors {
        w.write_all(&t.data).map_err(io)?;
    }
    w.flush().map_err(io)
}

pub fn write_index(path: &Path, weight_map: &[(String, String)], total_size: u64) -> Result<()> {
    let map = Value::Obj(
        weight_map
            .iter()
            .map(|(t, s)| (t.clone(), Value::Str(s.clone())))
            .collect(),
    );
    let root = Value::Obj(vec![
        (
            "metadata".into(),
            Value::Obj(vec![("total_size".into(), Value::Int(total_size))]),
        ),
        ("weight_map".into(), map),
    ]);
    std::fs::write(path, json::to_canonical(&root)).map_err(|e| Error::io(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(json_text: &str) -> Value {
        json::parse(json_text.as_bytes()).unwrap()
    }

    #[test]
    fn parses_valid_header() {
        let h = header(
            r#"{"__metadata__":{"format":"pt"},
                "a":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},
                "b":{"dtype":"F32","shape":[3],"data_offsets":[16,28]}}"#,
        );
        let ts = parse_header(&h, 28).unwrap();
        assert_eq!(ts.len(), 2);
        assert_eq!(ts[0].name, "a");
        assert_eq!(ts[0].shape, [2, 4]);
        assert_eq!((ts[1].begin, ts[1].end), (16, 28));
    }

    #[test]
    fn rejects_bad_headers() {
        let cases = [
            // size mismatch
            r#"{"a":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,15]}}"#,
            // out of bounds
            r#"{"a":{"dtype":"BF16","shape":[2,4],"data_offsets":[16,32]}}"#,
            // unknown dtype
            r#"{"a":{"dtype":"Q4","shape":[2],"data_offsets":[0,2]}}"#,
            // unknown key
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4],"x":1}}"#,
            // malformed offsets
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0]}}"#,
        ];
        for c in cases {
            assert!(parse_header(&header(c), 28).is_err(), "should reject: {c}");
        }
    }

    #[test]
    fn zero_sized_tensor_is_valid() {
        let h = header(r#"{"a":{"dtype":"BF16","shape":[0,4],"data_offsets":[0,0]}}"#);
        let ts = parse_header(&h, 0).unwrap();
        assert_eq!(ts[0].begin, ts[0].end);
    }
}
