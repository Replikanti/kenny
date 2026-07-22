//! `kenny spine` — the Qwen3-30B-A3B spine-sim (ADR-0020).
//!
//! The spine is the one strong machine that owns everything stateful: the token
//! stream, attention + KV cache, routing and sampling (MANIFESTO §2). For M1–M4
//! it is a *spine-sim*: a pure-Rust dense Qwen3 forward whose MoE FFN call is
//! replaced by dispatch-to-nodes. This is a deliberate in-repo reinvention
//! rather than hooking `candle`/`llama.cpp`: both of those pull crates on
//! kenny's `deny.toml` denylist (`serde`, `rand`, `thiserror`) and neither
//! cleanly exposes the per-layer FFN boundary + router logits ADR-0020 names as
//! the selection criterion. The expert kernel is the shared `expert::forward`
//! (one kernel = determinism owned once, ADR-0018); the spine only does the
//! dense scaffolding around it.
//!
//! The M1 correctness gate is *protocol self-consistency*: the dispatched
//! (`NodeDispatch`) path must reproduce the in-process (`LocalDispatch`) path
//! bit-for-bit under a matched codec (`tests/dispatch.rs`). Both dispatchers
//! apply the identical `wire` codec around the identical `expert::forward`, so
//! equivalence is bit-exact by construction — any drift is a real bug, not
//! numeric slop. Perplexity-vs-HuggingFace validation is a later concern by
//! roadmap design (ADR-0008 canaries); the router math below is nonetheless
//! kept Qwen3-faithful so the deferred real-model number is meaningful.
//!
//! Router order is Qwen3, not DeepSeek/GLM: **softmax over ALL experts first**,
//! then top-k, then (`norm_topk_prob=true`) renormalize the selected weights to
//! sum 1. On top of that the spine applies the ADR-0008 renorm over the subset
//! that actually answered — a no-op when every selected expert is present.

use std::collections::{BTreeMap, HashMap};
use std::net::TcpStream;
use std::path::Path;
use std::time::{Duration, Instant};

use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::manifest::{Manifest, SpineEntry};
use crate::node::Node;
use crate::quant;
use crate::wire::{Dispatch, ExpertStatus, Handshake, Transport, WireCodec};

// -------------------------------------------------------------------------
// Hyperparameters (Qwen3-30B-A3B defaults — the authoritative model card)
// -------------------------------------------------------------------------

/// Spine hyperparameters that are NOT in the manifest. They cannot be read from
/// `config.json` because kenny's JSON subset rejects floats (ADR-0017) and
/// `rms_norm_eps` is a float; they arrive as CLI flags with the defaults below,
/// so `hidden`/`inter`/`experts_per_layer`/`moe_layers` come from the manifest
/// and the rest are pinned here.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    /// `head_dim` is its own value: for Qwen3-30B-A3B it is 128, which is NOT
    /// `hidden / num_heads` (= 64), so it must be carried explicitly.
    pub head_dim: usize,
    /// RoPE base. Qwen3-30B-A3B uses 10_000_000.
    pub rope_theta: f64,
    /// RMSNorm epsilon. Qwen3 uses 1e-6, expressed as an integer ppm on the CLI
    /// (`1e-6` cannot be an integer count of "milli"); stored resolved here.
    pub rms_eps: f32,
    /// Router top-k. Clamped to `experts_per_layer` at route time.
    pub top_k: usize,
}

impl Default for Config {
    /// Qwen3-30B-A3B, from the model card: 32 query heads, 4 KV heads (GQA),
    /// head_dim 128, RoPE theta 1e7, rms_norm_eps 1e-6, 8 experts routed.
    fn default() -> Self {
        Config {
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            top_k: 8,
        }
    }
}

/// Convert an integer ppm to an `f32` epsilon (`ppm × 1e-6`): Qwen3's `1` means
/// `1e-6`. This is the only float that reaches the spine from the CLI, and ppm
/// keeps it inside the integer argument parser (A2).
pub fn eps_from_ppm(ppm: u64) -> f32 {
    (ppm as f64 * 1e-6) as f32
}

// -------------------------------------------------------------------------
// Spine weights (always-on tensors, loaded from the source model by range)
// -------------------------------------------------------------------------

/// Per-layer attention + norm + router weights, all row-major f32.
struct LayerWeights {
    input_ln: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_ln: Vec<f32>,
    /// Router gate, `[experts_per_layer, hidden]`.
    gate: Vec<f32>,
}

/// The spine's always-on tensors: embeddings, per-layer attention/router, final
/// norm, lm_head. Loaded once from the SOURCE model dir via the manifest's
/// absolute byte ranges (ADR-0005) — the experts are the only thing carved out;
/// everything here stays in the original shards and is read by range.
struct SpineWeights {
    embed: Vec<f32>,
    norm: Vec<f32>,
    lm_head: Vec<f32>,
    layers: BTreeMap<u16, LayerWeights>,
}

impl SpineWeights {
    fn load(model_dir: &Path, manifest: &Manifest, cfg: &Config) -> Result<(SpineWeights, usize)> {
        let hidden = manifest.model.hidden as usize;
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        if nh == 0 || nkv == 0 || hd == 0 {
            return Err(Error::usage("spine: heads and head_dim must be nonzero"));
        }
        if nh % nkv != 0 {
            return Err(Error::usage(format!(
                "spine: num_heads {nh} is not a multiple of num_kv_heads {nkv}"
            )));
        }
        if !hd.is_multiple_of(2) {
            return Err(Error::usage(format!(
                "spine: head_dim {hd} must be even (RoPE)"
            )));
        }
        let (q_dim, kv_dim) = (nh * hd, nkv * hd);

        let by_name: HashMap<&str, &SpineEntry> = manifest
            .spine
            .iter()
            .map(|s| (s.name.as_str(), s))
            .collect();
        let mut shard_cache: HashMap<String, Mmap> = HashMap::new();

        // Load one BF16 spine tensor by name, verifying its dtype, shape, byte
        // range, and CID against the manifest before decoding to f32.
        let mut load = |name: &str, shape: &[u64]| -> Result<Vec<f32>> {
            let e = by_name.get(name).ok_or_else(|| {
                Error::parse(format!("spine: manifest has no spine tensor {name:?}"))
            })?;
            if e.dtype != "BF16" {
                return Err(Error::parse(format!(
                    "spine: {name} is {}, this build reads BF16 spine tensors",
                    e.dtype
                )));
            }
            if e.shape != shape {
                return Err(Error::parse(format!(
                    "spine: {name} has shape {:?}, the config implies {shape:?} — check the \
                     --num-heads / --num-kv-heads / --head-dim flags against this model",
                    e.shape
                )));
            }
            if !shard_cache.contains_key(&e.shard) {
                let path = model_dir.join(&e.shard);
                let file = std::fs::File::open(&path).map_err(|er| Error::io(&path, er))?;
                // SAFETY: read-only mapping of a source shard treated as
                // immutable for the spine's lifetime (same discipline as
                // src/safetensors.rs and src/node.rs).
                let mmap = unsafe { Mmap::map(&file) }.map_err(|er| Error::io(&path, er))?;
                shard_cache.insert(e.shard.clone(), mmap);
            }
            let mmap = &shard_cache[&e.shard];
            let (b, en) = (e.begin as usize, e.end as usize);
            if en > mmap.len() || b > en {
                return Err(Error::parse(format!(
                    "spine: {name} range [{b}, {en}] out of bounds in shard {} ({} bytes)",
                    e.shard,
                    mmap.len()
                )));
            }
            let bytes = &mmap[b..en];
            if blake3::hash(bytes).to_hex().as_str() != e.cid {
                return Err(Error::parse(format!(
                    "spine: {name} bytes do not hash to the manifest CID — the --model dir is \
                     not the one this carve came from"
                )));
            }
            quant::bf16_to_f32_vec(bytes)
        };

        // Vocab comes from the embedding's own shape (not in the manifest's
        // model block) — read it before validating that tensor's shape.
        let embed_meta = by_name
            .get("model.embed_tokens.weight")
            .ok_or_else(|| Error::parse("spine: manifest has no model.embed_tokens.weight"))?;
        if embed_meta.shape.len() != 2 || embed_meta.shape[1] != hidden as u64 {
            return Err(Error::parse(format!(
                "spine: model.embed_tokens.weight has shape {:?}, expected [vocab, {hidden}]",
                embed_meta.shape
            )));
        }
        let vocab = embed_meta.shape[0] as usize;

        let h = hidden as u64;
        let embed = load("model.embed_tokens.weight", &[vocab as u64, h])?;
        let norm = load("model.norm.weight", &[h])?;
        let lm_head = load("lm_head.weight", &[vocab as u64, h])?;

        let experts = manifest.model.experts_per_layer as u64;
        let mut layer_ids: Vec<u16> = manifest.experts.iter().map(|e| e.layer).collect();
        layer_ids.sort_unstable();
        layer_ids.dedup();

        let mut layers = BTreeMap::new();
        for &l in &layer_ids {
            let p = |t: &str| format!("model.layers.{l}.{t}");
            let lw = LayerWeights {
                input_ln: load(&p("input_layernorm.weight"), &[h])?,
                q_proj: load(&p("self_attn.q_proj.weight"), &[q_dim as u64, h])?,
                k_proj: load(&p("self_attn.k_proj.weight"), &[kv_dim as u64, h])?,
                v_proj: load(&p("self_attn.v_proj.weight"), &[kv_dim as u64, h])?,
                o_proj: load(&p("self_attn.o_proj.weight"), &[h, q_dim as u64])?,
                q_norm: load(&p("self_attn.q_norm.weight"), &[hd as u64])?,
                k_norm: load(&p("self_attn.k_norm.weight"), &[hd as u64])?,
                post_ln: load(&p("post_attention_layernorm.weight"), &[h])?,
                gate: load(&p("mlp.gate.weight"), &[experts, h])?,
            };
            layers.insert(l, lw);
        }

        Ok((
            SpineWeights {
                embed,
                norm,
                lm_head,
                layers,
            },
            vocab,
        ))
    }
}

// -------------------------------------------------------------------------
// Dense math (hand-rolled per ADR-0021; f32 throughout for determinism)
// -------------------------------------------------------------------------

/// `out = W · x`, W row-major `[rows, cols]`.
fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    let mut out = vec![0f32; rows];
    for (o, row) in out.iter_mut().zip(w.chunks_exact(cols)) {
        let mut acc = 0f32;
        for (&wv, &xv) in row.iter().zip(x) {
            acc += wv * xv;
        }
        *o = acc;
    }
    out
}

/// RMSNorm in place: `x_i <- x_i / sqrt(mean(x^2) + eps) * weight_i`.
fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len() as f32;
    let ms = x.iter().map(|&v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    for (v, &w) in x.iter_mut().zip(weight) {
        *v = *v * inv * w;
    }
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    rms_norm_inplace(&mut out, weight, eps);
    out
}

/// Rotary position embedding in place over one head vector (HF `rotate_half`
/// layout: the vector splits into two halves that rotate together). Angles are
/// computed in f64 for fidelity, applied in f32.
fn rope_inplace(v: &mut [f32], pos: usize, theta: f64) {
    let hd = v.len();
    let half = hd / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f64 / hd as f64);
        let (sin, cos) = (pos as f64 * freq).sin_cos();
        let (sin, cos) = (sin as f32, cos as f32);
        let (x0, x1) = (v[i], v[i + half]);
        v[i] = x0 * cos - x1 * sin;
        v[i + half] = x1 * cos + x0 * sin;
    }
}

/// Softmax in place (numerically stabilized by the max).
fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Greedy argmax; ties go to the lowest index (deterministic).
fn argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Qwen3 router: softmax over ALL experts, take the `top_k` highest by
/// probability (ties broken by ascending index — deterministic), then
/// renormalize the selected weights to sum 1 (`norm_topk_prob=true`). Returns
/// `(expert_index, weight)` for the selected experts. This is the ORDER that
/// matters (A1): softmax-then-top-k, NOT top-k-then-softmax.
fn route(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    // Descending probability, ascending index on ties (total_cmp = no NaN trap).
    idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    let k = top_k.min(probs.len());
    let selected = &idx[..k];
    let sum: f32 = selected.iter().map(|&i| probs[i]).sum();
    selected.iter().map(|&i| (i, probs[i] / sum)).collect()
}

// -------------------------------------------------------------------------
// Dispatch abstraction — LocalDispatch (in-process) and NodeDispatch (TCP)
// -------------------------------------------------------------------------

/// The one seam ADR-0020 names: run `experts` of `layer` on activation `x`,
/// returning one entry per requested expert (`None` = not held / not answered,
/// feeding the ADR-0008 renorm). Both impls apply the wire codec identically
/// around `expert::forward`, so a matched-codec local run and dispatched run
/// are bit-identical (the M1 gate).
pub trait Dispatcher {
    fn dispatch(&mut self, layer: u16, x: &[f32], experts: &[u16])
    -> Result<Vec<Option<Vec<f32>>>>;
    /// Wire bytes (up, down) measured at the socket; `(0, 0)` for in-process.
    fn wire_bytes(&self) -> (u64, u64) {
        (0, 0)
    }
}

/// In-process dispatch: runs experts through a `Node` loaded from the carve,
/// applying the same `codec.encode`/`decode` round-trip the wire applies, so it
/// mirrors `NodeDispatch` byte-for-byte on the compute side (ADR-0018).
pub struct LocalDispatch {
    node: Node,
    codec: Box<dyn WireCodec>,
}

impl LocalDispatch {
    pub fn new(carved_dir: &Path, codec: Box<dyn WireCodec>) -> Result<LocalDispatch> {
        Ok(LocalDispatch {
            node: Node::load(carved_dir)?,
            codec,
        })
    }

    /// Mutable access to the backing node — used in tests to drop an expert and
    /// exercise the renorm path against a node that lost a replica.
    pub fn node_mut(&mut self) -> &mut Node {
        &mut self.node
    }
}

impl Dispatcher for LocalDispatch {
    fn dispatch(
        &mut self,
        layer: u16,
        x: &[f32],
        experts: &[u16],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        // Round-trip x through the codec exactly as the node does off the wire,
        // so the forward input is identical on both paths.
        let mut xb = Vec::new();
        self.codec.encode(x, &mut xb);
        let xd = self.codec.decode(&xb)?;
        let mut out = Vec::with_capacity(experts.len());
        for &e in experts {
            out.push(self.node.run_local(layer, e, &xd, self.codec.as_ref())?);
        }
        Ok(out)
    }
}

/// TCP dispatch: one connection to a node, a handshake, then a dispatch/gather
/// per MoE layer (ADR-0016 interim sync transport via `crate::wire`).
pub struct NodeDispatch {
    transport: Transport<TcpStream>,
    codec: Box<dyn WireCodec>,
    /// `hidden * codec.elem_bytes()` — the only legal activation / answered-y
    /// size, checked by the transport before it allocates.
    elem_len: usize,
}

impl NodeDispatch {
    /// Connect, verify nothing yet, send the handshake (codec + model identity).
    pub fn connect(
        addr: &str,
        codec: Box<dyn WireCodec>,
        identity: [u8; 32],
        hidden: usize,
    ) -> Result<NodeDispatch> {
        let stream = TcpStream::connect(addr)
            .map_err(|e| Error::parse(format!("spine: cannot connect to node {addr}: {e}")))?;
        let mut transport = Transport::new(stream);
        transport.send_handshake(&Handshake::new(codec.as_ref(), identity))?;
        let elem_len = hidden * codec.elem_bytes();
        Ok(NodeDispatch {
            transport,
            codec,
            elem_len,
        })
    }
}

impl Dispatcher for NodeDispatch {
    fn dispatch(
        &mut self,
        layer: u16,
        x: &[f32],
        experts: &[u16],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let mut xb = Vec::with_capacity(self.elem_len);
        self.codec.encode(x, &mut xb);
        self.transport.send_dispatch(&Dispatch {
            layer,
            x: xb,
            experts: experts.to_vec(),
        })?;
        let gather = self.transport.recv_gather(self.elem_len)?;
        if gather.results.len() != experts.len() {
            return Err(Error::parse(format!(
                "spine: node answered {} results for {} requested experts",
                gather.results.len(),
                experts.len()
            )));
        }
        let mut out = Vec::with_capacity(experts.len());
        for (&want, r) in experts.iter().zip(&gather.results) {
            if r.expert != want {
                return Err(Error::parse(
                    "spine: node returned gather records out of dispatch order",
                ));
            }
            out.push(match r.status {
                ExpertStatus::Ok => Some(self.codec.decode(&r.y)?),
                ExpertStatus::NotHeld => None,
            });
        }
        Ok(out)
    }

    fn wire_bytes(&self) -> (u64, u64) {
        (self.transport.up, self.transport.down)
    }
}

// -------------------------------------------------------------------------
// The spine forward
// -------------------------------------------------------------------------

/// Per-layer KV cache — one (k, v) row per processed position.
#[derive(Default)]
struct LayerKv {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

/// Measured generation stats (BENCH convention: numbers, not vibes). Byte counts
/// come straight off the dispatcher's socket counters.
#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    /// Dispatch frames sent (one per MoE layer per forward pass).
    pub dispatches: u64,
    pub experts_requested: u64,
    pub experts_answered: u64,
    /// MoE steps where at least one selected expert did not answer (renorm ran).
    pub renorm_steps: u64,
    pub wire_up: u64,
    pub wire_down: u64,
    pub elapsed: Duration,
    /// Wall time of each forward pass, in call order (prompt-priming forwards
    /// then generation forwards). The per-token latency distribution BENCH
    /// reports median + p99 over (`no vibes`); empty for a `logits()` prefill.
    pub per_forward: Vec<Duration>,
}

impl GenStats {
    /// `(median, p99)` of the per-forward latencies (nearest-rank p99), or
    /// `(0, 0)` if none were recorded.
    pub fn latency_median_p99(&self) -> (Duration, Duration) {
        if self.per_forward.is_empty() {
            return (Duration::ZERO, Duration::ZERO);
        }
        let mut v = self.per_forward.clone();
        v.sort_unstable();
        let median = v[v.len() / 2];
        // Nearest-rank: ceil(0.99 * n) - 1, clamped into range.
        let rank = (((v.len() as f64) * 0.99).ceil() as usize).clamp(1, v.len()) - 1;
        (median, v[rank])
    }
}

/// The spine-sim: dense Qwen3 scaffolding around a dispatched MoE FFN.
pub struct Spine {
    config: Config,
    hidden: usize,
    vocab: usize,
    experts_per_layer: usize,
    /// MoE layer indices in ascending order (the model's own indexing).
    layers: Vec<u16>,
    weights: SpineWeights,
}

impl Spine {
    /// Load the always-on tensors from `model_dir` using `manifest`'s ranges and
    /// validate them against `config`.
    pub fn load(model_dir: &Path, manifest: &Manifest, config: Config) -> Result<Spine> {
        let hidden = manifest.model.hidden as usize;
        let experts_per_layer = manifest.model.experts_per_layer as usize;
        if config.top_k == 0 {
            return Err(Error::usage("spine: top-k must be at least 1"));
        }
        let (weights, vocab) = SpineWeights::load(model_dir, manifest, &config)?;
        let layers: Vec<u16> = weights.layers.keys().copied().collect();
        Ok(Spine {
            config,
            hidden,
            vocab,
            experts_per_layer,
            layers,
            weights,
        })
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn moe_layers(&self) -> usize {
        self.layers.len()
    }

    /// The number of experts routed per MoE step (top-k clamped to the layer's
    /// expert count) — the constant that makes wire-byte accounting exact (A5).
    pub fn experts_per_step(&self) -> usize {
        self.config.top_k.min(self.experts_per_layer)
    }

    /// Greedy generation of `max_new` tokens after `prompt`, dispatching every
    /// MoE FFN through `dispatcher`. Returns the full token sequence
    /// (prompt ++ generated) and the measured stats.
    pub fn generate(
        &self,
        dispatcher: &mut dyn Dispatcher,
        prompt: &[u32],
        max_new: usize,
    ) -> Result<(Vec<u32>, GenStats)> {
        if prompt.is_empty() {
            return Err(Error::usage("spine: prompt must have at least one token"));
        }
        if max_new == 0 {
            return Err(Error::usage("spine: --tokens must be at least 1"));
        }
        let mut stats = GenStats::default();
        let started = Instant::now();
        let mut kv: Vec<LayerKv> = (0..self.layers.len()).map(|_| LayerKv::default()).collect();
        let mut tokens = prompt.to_vec();
        let mut pos = 0usize;

        // Prime the KV cache with the prompt; the last prompt logits predict the
        // first generated token.
        let mut logits = Vec::new();
        for &tok in prompt {
            let t = Instant::now();
            logits = self.forward_token(tok, pos, &mut kv, dispatcher, &mut stats)?;
            stats.per_forward.push(t.elapsed());
            pos += 1;
        }
        // Emit max_new tokens; forward only to predict the NEXT one, so no extra
        // dispatch (and no extra wire bytes) is spent past the last token (A5).
        for i in 0..max_new {
            let next = argmax(&logits);
            tokens.push(next);
            if i + 1 < max_new {
                let t = Instant::now();
                logits = self.forward_token(next, pos, &mut kv, dispatcher, &mut stats)?;
                stats.per_forward.push(t.elapsed());
                pos += 1;
            }
        }

        stats.prompt_tokens = prompt.len();
        stats.generated_tokens = max_new;
        stats.elapsed = started.elapsed();
        let (up, down) = dispatcher.wire_bytes();
        stats.wire_up = up;
        stats.wire_down = down;
        Ok((tokens, stats))
    }

    /// Prefill `prompt` and return the logits at its final position — the
    /// distribution over the next token, without sampling or generating past it.
    /// This is the seam the S7 output-sanity check compares two dispatch paths
    /// on: run once through the fp8 path (fp8 blobs + fp8 wire) and once through
    /// a reference path that reconstructs experts from the ORIGINAL bf16 weights
    /// (no blob quant, no codec), then take the cosine of the two logit vectors —
    /// the first end-to-end ADR-0018 signal, mirroring M0's fp8-vs-bf16
    /// methodology (A6). Teacher-forced on the same prompt, so the two paths see
    /// identical input and the number is well defined (no greedy divergence).
    pub fn logits(&self, dispatcher: &mut dyn Dispatcher, prompt: &[u32]) -> Result<Vec<f32>> {
        if prompt.is_empty() {
            return Err(Error::usage("spine: prompt must have at least one token"));
        }
        let mut stats = GenStats::default();
        let mut kv: Vec<LayerKv> = (0..self.layers.len()).map(|_| LayerKv::default()).collect();
        let mut logits = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            logits = self.forward_token(tok, pos, &mut kv, dispatcher, &mut stats)?;
        }
        Ok(logits)
    }

    /// One forward pass for `tok` at `pos`, returning the vocab logits.
    fn forward_token(
        &self,
        tok: u32,
        pos: usize,
        kv: &mut [LayerKv],
        dispatcher: &mut dyn Dispatcher,
        stats: &mut GenStats,
    ) -> Result<Vec<f32>> {
        let eps = self.config.rms_eps;
        let mut h = self.embed_row(tok)?;
        for (li, &layer) in self.layers.iter().enumerate() {
            let lw = &self.weights.layers[&layer];
            // Attention block with a residual connection.
            let normed = rms_norm(&h, &lw.input_ln, eps);
            let attn = self.attention(lw, &normed, pos, &mut kv[li]);
            for (hv, av) in h.iter_mut().zip(&attn) {
                *hv += av;
            }
            // MoE block with a residual connection.
            let normed = rms_norm(&h, &lw.post_ln, eps);
            let moe = self.moe(layer, lw, &normed, dispatcher, stats)?;
            for (hv, mv) in h.iter_mut().zip(&moe) {
                *hv += mv;
            }
        }
        rms_norm_inplace(&mut h, &self.weights.norm, eps);
        Ok(matvec(&self.weights.lm_head, &h, self.vocab, self.hidden))
    }

    fn embed_row(&self, tok: u32) -> Result<Vec<f32>> {
        let t = tok as usize;
        if t >= self.vocab {
            return Err(Error::parse(format!(
                "spine: token {tok} out of range for vocab {}",
                self.vocab
            )));
        }
        let start = t * self.hidden;
        Ok(self.weights.embed[start..start + self.hidden].to_vec())
    }

    /// GQA attention for one token: per-head q_norm/k_norm then RoPE, causal
    /// scores over the KV cache scaled by `1/sqrt(head_dim)`, KV heads repeated
    /// across their query-head group.
    fn attention(&self, lw: &LayerWeights, x: &[f32], pos: usize, kv: &mut LayerKv) -> Vec<f32> {
        let (nh, nkv, hd) = (
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
        );
        let (q_dim, kv_dim) = (nh * hd, nkv * hd);
        let eps = self.config.rms_eps;
        let theta = self.config.rope_theta;

        let mut q = matvec(&lw.q_proj, x, q_dim, self.hidden);
        let mut k = matvec(&lw.k_proj, x, kv_dim, self.hidden);
        let v = matvec(&lw.v_proj, x, kv_dim, self.hidden);
        for head in q.chunks_exact_mut(hd) {
            rms_norm_inplace(head, &lw.q_norm, eps);
            rope_inplace(head, pos, theta);
        }
        for head in k.chunks_exact_mut(hd) {
            rms_norm_inplace(head, &lw.k_norm, eps);
            rope_inplace(head, pos, theta);
        }
        kv.k.push(k);
        kv.v.push(v);

        let scale = 1.0 / (hd as f32).sqrt();
        let groups = nh / nkv;
        let mut context = vec![0f32; q_dim];
        for (hqi, (qhead, ctx)) in q
            .chunks_exact(hd)
            .zip(context.chunks_exact_mut(hd))
            .enumerate()
        {
            let kv_head = hqi / groups;
            let mut scores = Vec::with_capacity(kv.k.len());
            for krow in &kv.k {
                let khead = &krow[kv_head * hd..kv_head * hd + hd];
                let dot: f32 = qhead.iter().zip(khead).map(|(&a, &b)| a * b).sum();
                scores.push(dot * scale);
            }
            softmax_inplace(&mut scores);
            for (t, &s) in scores.iter().enumerate() {
                let vhead = &kv.v[t][kv_head * hd..kv_head * hd + hd];
                for (c, &vv) in ctx.iter_mut().zip(vhead) {
                    *c += s * vv;
                }
            }
        }
        matvec(&lw.o_proj, &context, self.hidden, q_dim)
    }

    /// The dispatched MoE FFN: route (Qwen3 order, A1), dispatch the selected
    /// experts once, then mix `y`s weighted by the router — renormalizing over
    /// the subset that answered (ADR-0008; a no-op when all are present).
    fn moe(
        &self,
        layer: u16,
        lw: &LayerWeights,
        x: &[f32],
        dispatcher: &mut dyn Dispatcher,
        stats: &mut GenStats,
    ) -> Result<Vec<f32>> {
        let logits = matvec(&lw.gate, x, self.experts_per_layer, self.hidden);
        let routed = route(&logits, self.config.top_k);
        let experts: Vec<u16> = routed.iter().map(|&(e, _)| e as u16).collect();

        let ys = dispatcher.dispatch(layer, x, &experts)?;
        stats.dispatches += 1;
        stats.experts_requested += experts.len() as u64;

        // ADR-0008 renorm denominator: the router weight mass that actually
        // answered. When every selected expert answered this equals 1 (the
        // norm_topk_prob sum), so the division below is a no-op.
        let answered_mass: f32 = routed
            .iter()
            .zip(&ys)
            .filter(|(_, y)| y.is_some())
            .map(|(&(_, w), _)| w)
            .sum();

        let mut out = vec![0f32; self.hidden];
        let mut answered = 0u64;
        if answered_mass > 0.0 {
            for (&(_, w), y) in routed.iter().zip(&ys) {
                if let Some(yv) = y {
                    answered += 1;
                    let rw = w / answered_mass;
                    for (o, &yy) in out.iter_mut().zip(yv) {
                        *o += rw * yy;
                    }
                }
            }
        }
        stats.experts_answered += answered;
        if answered < experts.len() as u64 {
            stats.renorm_steps += 1;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::Dtype;
    use crate::carve::{self, Options};
    use crate::fixture::{self, Params};
    use crate::safetensors::ShardFile;
    use crate::wire::Bf16Codec;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("kenny-spine-{name}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Generate the default fixture and a bf16 carve; return (model_dir, carved).
    fn fixture_carve(root: &Path) -> (PathBuf, PathBuf) {
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

    /// The fixture's attention is square (`q/k/v/o_proj = [h, h]`,
    /// `q_norm/k_norm = [h]`), so it only loads at num_heads = num_kv_heads = 1,
    /// head_dim = hidden (A4).
    fn fixture_config(hidden: usize) -> Config {
        Config {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: hidden,
            rope_theta: 1_000_000.0,
            rms_eps: eps_from_ppm(1),
            top_k: 2,
        }
    }

    // --- dense math -------------------------------------------------------

    #[test]
    fn rms_norm_matches_hand_computation() {
        // x = [3, 4]; mean(x^2) = 12.5; inv = 1/sqrt(12.5 + 0). weight = [2, 0.5].
        let x = [3.0f32, 4.0];
        let w = [2.0f32, 0.5];
        let out = rms_norm(&x, &w, 0.0);
        let inv = 1.0f32 / 12.5f32.sqrt();
        assert!((out[0] - 3.0 * inv * 2.0).abs() < 1e-6);
        assert!((out[1] - 4.0 * inv * 0.5).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one_and_orders() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax_inplace(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[0] < x[1] && x[1] < x[2]);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        // cos(0) = 1, sin(0) = 0 -> no rotation at pos 0.
        let mut v = [1.0f32, 2.0, 3.0, 4.0];
        rope_inplace(&mut v, 0, 10_000.0);
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_rotates_by_expected_angle() {
        // head_dim 2 -> one pair, freq = theta^0 = 1, angle = pos. At pos 1:
        // out = (x0 cos1 - x1 sin1, x1 cos1 + x0 sin1).
        let mut v = [1.0f32, 0.0];
        rope_inplace(&mut v, 1, 10_000.0);
        let c = 1.0f32.cos();
        let s = 1.0f32.sin();
        assert!((v[0] - c).abs() < 1e-6);
        assert!((v[1] - s).abs() < 1e-6);
    }

    #[test]
    fn argmax_prefers_lowest_index_on_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    // --- router (A1: softmax over ALL experts, then top-k, then renorm) ----

    #[test]
    fn route_is_softmax_then_topk_then_renorm() {
        // Four experts; logits chosen so softmax order is 3 > 2 > 1 > 0.
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        let mut probs = logits.to_vec();
        softmax_inplace(&mut probs);
        let routed = route(&logits, 2);
        // Top-2 by probability are experts 3 and 2, in that order.
        assert_eq!(routed[0].0, 3);
        assert_eq!(routed[1].0, 2);
        // Weights are the ALL-experts softmax probs renormalized over the pair.
        let sum = probs[3] + probs[2];
        assert!((routed[0].1 - probs[3] / sum).abs() < 1e-6);
        assert!((routed[1].1 - probs[2] / sum).abs() < 1e-6);
        assert!((routed[0].1 + routed[1].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn route_clamps_topk_to_expert_count() {
        let logits = [1.0f32, 2.0, 3.0];
        let routed = route(&logits, 8);
        assert_eq!(routed.len(), 3, "top-k clamped to experts");
        let total: f32 = routed.iter().map(|&(_, w)| w).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    // --- spine weight loader ---------------------------------------------

    #[test]
    fn spine_loader_bytes_match_source_tensor() {
        let root = tmp("loader");
        let (model, carved) = fixture_carve(&root);
        let manifest = Manifest::load(&carved.join(crate::manifest::FILE_NAME)).unwrap();
        let hidden = manifest.model.hidden as usize;
        let (weights, _vocab) =
            SpineWeights::load(&model, &manifest, &fixture_config(hidden)).unwrap();

        // Compare the loaded final norm against the raw source tensor read via
        // the safetensors reader (independent path).
        let shard = ShardFile::open(&model.join("model-00002-of-00002.safetensors")).unwrap();
        let meta = shard.tensor("model.norm.weight").unwrap();
        let want = quant::bf16_to_f32_vec(shard.bytes(meta)).unwrap();
        assert_eq!(
            weights.norm, want,
            "loader must reproduce source bytes exactly"
        );
    }

    #[test]
    fn spine_load_rejects_wrong_head_geometry() {
        let root = tmp("geom");
        let (model, carved) = fixture_carve(&root);
        let manifest = Manifest::load(&carved.join(crate::manifest::FILE_NAME)).unwrap();
        // head_dim != hidden makes q_norm's [hidden] disagree with [head_dim].
        let bad = Config {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            ..fixture_config(manifest.model.hidden as usize)
        };
        assert!(Spine::load(&model, &manifest, bad).is_err());
    }

    // --- GQA head repetition (A4: unit-tested, not via the fixture) --------

    #[test]
    fn gqa_repeats_kv_heads_across_query_group() {
        // Synthetic 1-layer spine hand-built with num_heads = 4, num_kv_heads =
        // 2, head_dim = 2 (GQA groups of 2). Identity projections make the
        // grouping observable: query heads 0,1 share KV head 0; heads 2,3 share
        // KV head 1. At pos 0 (one position, softmax = 1) the context for each
        // query head equals its KV head's v — so heads in a group are identical.
        let hidden = 8usize; // = num_heads(4) * head_dim(2) = num_kv(2)*... no
        // q_dim = 4*2 = 8 = hidden; kv_dim = 2*2 = 4.
        let cfg = Config {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
            rope_theta: 10_000.0,
            rms_eps: 0.0,
            top_k: 1,
        };
        let eye = |n: usize| -> Vec<f32> {
            let mut m = vec![0f32; n * n];
            for i in 0..n {
                m[i * n + i] = 1.0;
            }
            m
        };
        let lw = LayerWeights {
            input_ln: vec![1.0; hidden],
            q_proj: eye(hidden), // [8,8]
            k_proj: {
                // [kv_dim=4, hidden=8]: pick x[0..4] into k.
                let mut m = vec![0f32; 4 * hidden];
                for i in 0..4 {
                    m[i * hidden + i] = 1.0;
                }
                m
            },
            v_proj: {
                // [4,8]: v = x[4..8], so the two KV heads carry distinct values.
                let mut m = vec![0f32; 4 * hidden];
                for i in 0..4 {
                    m[i * hidden + (4 + i)] = 1.0;
                }
                m
            },
            o_proj: eye(hidden), // [8,8]
            q_norm: vec![1.0; 2],
            k_norm: vec![1.0; 2],
            post_ln: vec![1.0; hidden],
            gate: vec![0.0; hidden], // unused here
        };
        let spine = Spine {
            config: cfg,
            hidden,
            vocab: 1,
            experts_per_layer: 1,
            layers: vec![0],
            weights: SpineWeights {
                embed: vec![0.0; hidden],
                norm: vec![1.0; hidden],
                lm_head: vec![0.0; hidden],
                layers: BTreeMap::from([(0u16, lw)]),
            },
        };
        let x: Vec<f32> = (0..hidden).map(|i| (i + 1) as f32).collect();
        let mut kv = LayerKv::default();
        let out = spine.attention(&spine.weights.layers[&0], &x, 0, &mut kv);
        // v = x[4..8] = [5,6,7,8]. KV head 0 = [5,6], KV head 1 = [7,8].
        // Query heads 0,1 -> KV head 0; heads 2,3 -> KV head 1 (o_proj = I).
        assert_eq!(&out[0..2], &out[2..4], "heads 0,1 share KV head 0");
        assert_eq!(&out[4..6], &out[6..8], "heads 2,3 share KV head 1");
        assert_eq!(&out[0..2], &[5.0, 6.0]);
        assert_eq!(&out[4..6], &[7.0, 8.0]);
    }

    // --- MoE renorm over the answered subset (ADR-0008) --------------------

    /// A dispatcher that answers every expert except a fixed `missing` set,
    /// returning a deterministic `y_e = [e + 1; hidden]` for the rest — so the
    /// renorm math is hand-checkable.
    struct MockDispatch {
        hidden: usize,
        missing: Vec<u16>,
    }

    impl Dispatcher for MockDispatch {
        fn dispatch(
            &mut self,
            _layer: u16,
            _x: &[f32],
            experts: &[u16],
        ) -> Result<Vec<Option<Vec<f32>>>> {
            Ok(experts
                .iter()
                .map(|&e| {
                    if self.missing.contains(&e) {
                        None
                    } else {
                        Some(vec![e as f32 + 1.0; self.hidden])
                    }
                })
                .collect())
        }
    }

    fn one_layer_router_spine(hidden: usize, experts: usize, gate: Vec<f32>) -> Spine {
        let z = |n: usize| vec![0f32; n];
        let lw = LayerWeights {
            input_ln: vec![1.0; hidden],
            q_proj: z(hidden * hidden),
            k_proj: z(hidden * hidden),
            v_proj: z(hidden * hidden),
            o_proj: z(hidden * hidden),
            q_norm: vec![1.0; hidden],
            k_norm: vec![1.0; hidden],
            post_ln: vec![1.0; hidden],
            gate,
        };
        Spine {
            config: Config {
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: hidden,
                rope_theta: 10_000.0,
                rms_eps: 0.0,
                top_k: experts,
            },
            hidden,
            vocab: 1,
            experts_per_layer: experts,
            layers: vec![0],
            weights: SpineWeights {
                embed: z(hidden),
                norm: vec![1.0; hidden],
                lm_head: z(hidden),
                layers: BTreeMap::from([(0u16, lw)]),
            },
        }
    }

    #[test]
    fn moe_renorms_over_answered_subset() {
        // hidden 2, 3 experts, top_k = 3 (all selected). gate rows pick logits
        // [1, 0, -1] for x = [1, 0], so weights are softmax([1, 0, -1]) and
        // norm_topk_prob (sum over all 3) leaves them as those probabilities.
        let hidden = 2;
        let gate = vec![1.0, 0.0, 0.0, 0.0, -1.0, 0.0]; // [3, 2] row-major
        let spine = one_layer_router_spine(hidden, 3, gate);
        let lw = &spine.weights.layers[&0];
        let x = [1.0f32, 0.0];

        let mut probs = vec![1.0f32, 0.0, -1.0];
        softmax_inplace(&mut probs);
        // y_e = [e + 1; hidden]: expert 0 -> 1, 1 -> 2, 2 -> 3.

        // Expert 1 down: renorm over {0, 2}.
        let mut mock = MockDispatch {
            hidden,
            missing: vec![1],
        };
        let mut stats = GenStats::default();
        let out = spine.moe(0, lw, &x, &mut mock, &mut stats).unwrap();
        let mass = probs[0] + probs[2];
        let expect = (probs[0] / mass) * 1.0 + (probs[2] / mass) * 3.0;
        assert!(
            (out[0] - expect).abs() < 1e-6,
            "renorm over answered subset"
        );
        assert_eq!((out[0], out[1]), (out[0], out[0]));
        assert_eq!(stats.experts_requested, 3);
        assert_eq!(stats.experts_answered, 2);
        assert_eq!(stats.renorm_steps, 1);

        // All present: full-mass weighted sum (mass = 1), a DIFFERENT value.
        let mut full_mock = MockDispatch {
            hidden,
            missing: vec![],
        };
        let mut full_stats = GenStats::default();
        let full = spine
            .moe(0, lw, &x, &mut full_mock, &mut full_stats)
            .unwrap();
        let expect_full = probs[0] * 1.0 + probs[1] * 2.0 + probs[2] * 3.0;
        assert!((full[0] - expect_full).abs() < 1e-6);
        assert_ne!(
            out[0], full[0],
            "the renorm changes the output vs all-present"
        );
        assert_eq!(full_stats.renorm_steps, 0);
    }

    // --- end-to-end LocalDispatch smoke (node<->local equivalence lives in
    //     tests/dispatch.rs; this only checks the forward runs + is finite) ---

    #[test]
    fn local_generate_is_finite_and_deterministic() {
        let root = tmp("gen");
        let (model, carved) = fixture_carve(&root);
        let manifest = Manifest::load(&carved.join(crate::manifest::FILE_NAME)).unwrap();
        let hidden = manifest.model.hidden as usize;
        let spine = Spine::load(&model, &manifest, fixture_config(hidden)).unwrap();

        let run = || {
            let mut d = LocalDispatch::new(&carved, Box::new(Bf16Codec)).unwrap();
            spine.generate(&mut d, &[1, 2, 3], 5).unwrap()
        };
        let (a, sa) = run();
        let (b, _sb) = run();
        assert_eq!(a, b, "generation is deterministic");
        assert_eq!(a.len(), 3 + 5);
        assert!(a.iter().all(|&t| (t as usize) < spine.vocab()));
        // All fixture experts present -> every selected expert answers, no renorm.
        assert_eq!(sa.experts_answered, sa.experts_requested);
        assert_eq!(sa.renorm_steps, 0);
        assert_eq!(sa.wire_up, 0, "local dispatch touches no socket");
    }
}
