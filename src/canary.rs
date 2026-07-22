//! `kenny canary` — the ADR-0008 perplexity canary.
//!
//! ADR-0008 makes canaries a *corollary of the renorm decision, not an optional
//! feature*: "quality degradation must be measured from day zero" via fixed
//! prompt sets scored against a known baseline. This is that measurement, and it
//! is also the deciding **quality axis** ADR-0018 is blocked on (fp8 vs int8 is
//! settled only by a per-path perplexity delta vs bf16).
//!
//! Method (the M0/M1 methodology, A6): teacher-force a fixed set of canary
//! sequences through TWO forwards that differ only at the dispatched MoE FFN —
//!
//! - the **test path**: the carved blobs + a wire codec (fp8 blobs + fp8 wire,
//!   the numeric path under test), via [`LocalDispatch`], which applies the exact
//!   `codec.encode`/`decode` round-trip the wire applies (so it is the on-wire
//!   fp8 path numerically, no live node required — the same construction S7 used);
//! - the **reference path**: every expert reconstructed straight from the
//!   ORIGINAL bf16 model tensors with NO quantization and NO codec — the exact
//!   `diff.rs::source_matrix` reference M0 measured fp8 against.
//!
//! At each position `i` the observed next token `t = tokens[i+1]` is scored by
//! its stable negative log-likelihood `logsumexp(logits) - logits[t]`; the
//! per-token mean NLL exponentiates to perplexity, and the reported number is
//! `Δppl = ppl(test) − ppl(ref)`. Teacher-forcing feeds the TRUE tokens to both
//! paths at every position, so the two see identical input and Δppl isolates the
//! carved+codec path's quality loss (the paths still diverge INTERNALLY — later
//! layers route on perturbed activations — which is exactly the end-to-end signal
//! ADR-0018 wants). Both dispatchers hold every expert, so nothing renorms and
//! the number is pure quantization quality, not dropout.
//!
//! CI never downloads a model (CLAUDE.md): the deterministic arm runs on the
//! synthetic fixture; the real Qwen3-30B-A3B Δppl is gated behind
//! `KENNY_MODEL_DIR` (BENCH.md "M4 — perplexity canary").

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::blob::Dtype;
use crate::error::{Error, Result};
use crate::expert;
use crate::manifest::Manifest;
use crate::quant;
use crate::rng::SplitMix64;
use crate::safetensors::{self, ShardFile};
use crate::spine::{Config, Dispatcher, LocalDispatch, Spine};
use crate::wire::{Bf16Codec, Fp8Codec, WireCodec};

pub struct CanaryOptions {
    /// Number of canary sequences in the fixed set.
    pub prompts: usize,
    /// Tokens per sequence (`len - 1` scored transitions each).
    pub len: usize,
    /// Seed keying the deterministic canary prompt set.
    pub seed: u64,
    /// Wire codec of the TEST path (`fp8` — the path under test — or `bf16`).
    pub codec: String,
    /// Spine hyperparameters (Qwen3-30B-A3B card by default; the fixture loads
    /// only at the square-attention config, exactly like `kenny spine`).
    pub config: Config,
}

impl Default for CanaryOptions {
    fn default() -> Self {
        CanaryOptions {
            prompts: 4,
            len: 16,
            seed: 42,
            codec: "fp8".into(),
            config: Config::default(),
        }
    }
}

#[derive(Debug)]
pub struct CanaryReport {
    /// Carve dtype of the test path's blobs (from the manifest).
    pub dtype: Dtype,
    /// Wire codec of the test path.
    pub codec: String,
    pub prompts: usize,
    pub prompt_len: usize,
    /// Total scored next-token transitions = `prompts × (len − 1)`.
    pub scored_tokens: usize,
    /// Perplexity of the carved+codec path.
    pub ppl_test: f64,
    /// Perplexity of the bf16-source reference.
    pub ppl_ref: f64,
    /// `ppl_test − ppl_ref` — the ADR-0018 deciding-axis number.
    pub delta_ppl: f64,
    /// Mean NLL (nats/token) of each path — the pre-exp figures.
    pub nll_test: f64,
    pub nll_ref: f64,
}

pub fn run(model_dir: &Path, carved_dir: &Path, opts: &CanaryOptions) -> Result<CanaryReport> {
    if opts.prompts == 0 {
        return Err(Error::usage("canary: --prompts must be at least 1"));
    }
    if opts.len < 2 {
        return Err(Error::usage(
            "canary: --len must be at least 2 (a token needs a next token to score)",
        ));
    }
    let codec_name = opts.codec.clone();
    // Validate the codec before touching the model so bad input fails fast.
    let _ = make_codec(&codec_name)?;

    let manifest = Manifest::load(&carved_dir.join(crate::manifest::FILE_NAME))?;
    let hidden = manifest.model.hidden as usize;
    let inter = manifest.model.inter as usize;
    let dtype = manifest.model.dtype;
    let spine = Spine::load(model_dir, &manifest, opts.config)?;
    let vocab = spine.vocab();

    // The fixed canary prompt set (ADR-0008): deterministic, seed-keyed, in-vocab.
    let prompts: Vec<Vec<u32>> = (0..opts.prompts)
        .map(|p| canary_prompt(opts.seed, p, opts.len, vocab))
        .collect();

    // Test path: carved blobs + the wire codec (the fp8 blob+wire path).
    let mut test = LocalDispatch::new(carved_dir, make_codec(&codec_name)?)?;
    let (nll_test_sum, count_test) = teacher_forced_nll(&spine, &mut test, &prompts)?;

    // Reference path: bf16 source matrices, no quant, no codec.
    let mut reference = SourceRefDispatch::new(model_dir, hidden, inter)?;
    let (nll_ref_sum, count_ref) = teacher_forced_nll(&spine, &mut reference, &prompts)?;

    // Same prompts, same scoring loop — the two counts are equal by construction.
    debug_assert_eq!(count_test, count_ref);
    let count = count_test as f64;
    let nll_test = nll_test_sum / count;
    let nll_ref = nll_ref_sum / count;
    let ppl_test = nll_test.exp();
    let ppl_ref = nll_ref.exp();
    Ok(CanaryReport {
        dtype,
        codec: codec_name,
        prompts: opts.prompts,
        prompt_len: opts.len,
        scored_tokens: count_test,
        ppl_test,
        ppl_ref,
        delta_ppl: ppl_test - ppl_ref,
        nll_test,
        nll_ref,
    })
}

/// Sum the teacher-forced NLL over every canary sequence, one sequence at a time
/// (so at most one sequence's per-position logits are resident — the logit
/// matrix is `len × vocab`). Returns `(nll_sum, scored_token_count)`; the mean
/// NLL is `nll_sum / count` and exponentiates to perplexity.
///
/// Public so the ADR-0008 renorm-quality-dip lock (`tests/dispatch.rs`) scores a
/// dropout-degraded dispatch path with the SAME teacher-forced NLL the canary
/// uses — the "quality dip WHILE a domain is down" is measured on the exact
/// canary metric, not a re-derived one.
pub fn teacher_forced_nll(
    spine: &Spine,
    dispatcher: &mut dyn Dispatcher,
    prompts: &[Vec<u32>],
) -> Result<(f64, usize)> {
    let mut nll = 0f64;
    let mut count = 0usize;
    for tokens in prompts {
        let per_pos = spine.logits_per_position(dispatcher, tokens)?;
        let (n, c) = score_tokens(&per_pos, tokens)?;
        nll += n;
        count += c;
        // `per_pos` drops here, bounding logit memory to a single sequence.
    }
    if count == 0 {
        return Err(Error::usage(
            "canary: no scored tokens (every sequence is shorter than 2 tokens)",
        ));
    }
    Ok((nll, count))
}

/// Teacher-forced NLL of `tokens[1..]` under `per_pos`: position `i`'s
/// distribution predicts `tokens[i+1]`, scored as the numerically stable
/// `logsumexp(l) - l[target]` (= −log softmax, computed in f64). Returns
/// `(nll_sum, count)` over the `len − 1` transitions.
fn score_tokens(per_pos: &[Vec<f32>], tokens: &[u32]) -> Result<(f64, usize)> {
    let mut nll = 0f64;
    let mut count = 0usize;
    for i in 0..tokens.len().saturating_sub(1) {
        let l = &per_pos[i];
        let target = tokens[i + 1] as usize;
        let lt = *l.get(target).ok_or_else(|| {
            Error::parse(format!(
                "canary: target token {target} out of range for vocab {}",
                l.len()
            ))
        })?;
        // logsumexp, stabilized by the max, accumulated in f64.
        let max = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut sum = 0f64;
        for &v in l {
            sum += (v as f64 - max).exp();
        }
        let lse = max + sum.ln();
        nll += lse - lt as f64;
        count += 1;
    }
    Ok((nll, count))
}

/// One deterministic in-vocab canary sequence of `len` tokens, keyed by
/// `(seed, index)` so the set is fixed and reproducible (ADR-0008 "fixed prompt
/// sets scored continuously against a baseline"). Both dispatch paths score the
/// SAME sequences, so Δppl is a clean difference.
fn canary_prompt(seed: u64, index: usize, len: usize, vocab: usize) -> Vec<u32> {
    let mut rng = SplitMix64::for_name(seed, &format!("canary.prompt.{index}"));
    (0..len)
        .map(|_| (rng.next_u64() % vocab as u64) as u32)
        .collect()
}

fn make_codec(name: &str) -> Result<Box<dyn WireCodec>> {
    match name {
        "fp8" => Ok(Box::new(Fp8Codec)),
        "bf16" => Ok(Box::new(Bf16Codec)),
        other => Err(Error::usage(format!(
            "canary: unknown --codec {other:?} (expected fp8 or bf16)"
        ))),
    }
}

/// The bf16-source reference dispatcher (A6): every requested expert
/// reconstructed straight from the ORIGINAL bf16 model tensors and run through
/// the shared [`expert::forward`] with NO blob quantization and NO wire codec —
/// the exact `diff.rs::source_matrix` reference M0/M1 measured fp8 against. It
/// holds every expert, so it never answers not-held (no renorm); shards are
/// mmapped and cached across experts.
struct SourceRefDispatch {
    dir: PathBuf,
    tensor_shard: BTreeMap<String, String>,
    shards: BTreeMap<String, ShardFile>,
    hidden: usize,
    inter: usize,
}

impl SourceRefDispatch {
    fn new(model_dir: &Path, hidden: usize, inter: usize) -> Result<SourceRefDispatch> {
        let model = safetensors::open_model(model_dir)?;
        let tensor_shard = model.weight_map.into_iter().collect();
        Ok(SourceRefDispatch {
            dir: model.dir,
            tensor_shard,
            shards: BTreeMap::new(),
            hidden,
            inter,
        })
    }

    /// Load one bf16 expert projection, verifying its dtype and shape against the
    /// manifest-implied `[rows, cols]` (a mismatched model dir errors cleanly,
    /// never panics or produces a garbage number — the same discipline as
    /// `diff.rs::source_matrix`).
    fn matrix(
        &mut self,
        layer: u16,
        expert: u16,
        proj: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        let name = format!("model.layers.{layer}.mlp.experts.{expert}.{proj}.weight");
        let shard_name = self.tensor_shard.get(&name).cloned().ok_or_else(|| {
            Error::parse(format!(
                "canary: tensor {name:?} not found in the model dir"
            ))
        })?;
        if !self.shards.contains_key(&shard_name) {
            let sf = ShardFile::open(&self.dir.join(&shard_name))?;
            self.shards.insert(shard_name.clone(), sf);
        }
        let shard = &self.shards[&shard_name];
        let meta = shard
            .tensor(&name)
            .ok_or_else(|| Error::parse(format!("canary: {name:?} missing from {shard_name:?}")))?;
        if meta.dtype != "BF16" {
            return Err(Error::parse(format!(
                "canary: {name:?} is {}, expected BF16 sources",
                meta.dtype
            )));
        }
        if meta.shape != [rows as u64, cols as u64] {
            return Err(Error::parse(format!(
                "canary: {name:?} has shape {:?}, the manifest implies [{rows}, {cols}] — \
                 is this the model the carve came from?",
                meta.shape
            )));
        }
        quant::bf16_to_f32_vec(shard.bytes(meta))
    }
}

impl Dispatcher for SourceRefDispatch {
    fn dispatch(
        &mut self,
        layer: u16,
        x: &[f32],
        experts: &[u16],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let (hidden, inter) = (self.hidden, self.inter);
        let mut out = Vec::with_capacity(experts.len());
        for &e in experts {
            // gate_proj / up_proj are [inter, hidden]; down_proj is [hidden, inter].
            let gate = self.matrix(layer, e, "gate_proj", inter, hidden)?;
            let up = self.matrix(layer, e, "up_proj", inter, hidden)?;
            let down = self.matrix(layer, e, "down_proj", hidden, inter)?;
            let mut y = vec![0f32; hidden];
            expert::forward(&gate, &up, &down, hidden, x, &mut y);
            out.push(Some(y));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carve::{self, Options};
    use crate::fixture::{self, Params};
    use crate::spine::eps_from_ppm;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("kenny-canary-{name}"));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// The fixture's attention is square, so it loads only at
    /// num_heads = num_kv_heads = 1, head_dim = hidden (A4) — same as the spine
    /// tests' `fixture_config`.
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

    fn fixture_carve(root: &Path, dtype: Dtype) -> (PathBuf, PathBuf) {
        let model = root.join("model");
        fixture::generate(&Params::default(), &model).unwrap();
        let carved = root.join("carved");
        carve::run(
            &model,
            &Options {
                out: carved.clone(),
                model_name: "fixture".into(),
                model_rev: String::new(),
                dtype,
            },
        )
        .unwrap();
        (model, carved)
    }

    // --- scoring math (hand-checked, no model) ----------------------------

    #[test]
    fn score_tokens_matches_hand_nll() {
        // Two positions, vocab 3. Position 0 predicts token 1, position 1 predicts
        // token 2. NLL_i = logsumexp(l) - l[target].
        let per_pos = vec![vec![0.0f32, 1.0, 2.0], vec![3.0f32, 0.0, 1.0]];
        let tokens = vec![0u32, 1, 2];
        let (nll, count) = score_tokens(&per_pos, &tokens).unwrap();
        assert_eq!(count, 2);

        let lse = |l: &[f32]| -> f64 {
            let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
        };
        let want = (lse(&per_pos[0]) - 1.0) + (lse(&per_pos[1]) - 1.0);
        assert!((nll - want).abs() < 1e-9, "nll {nll} vs {want}");
    }

    #[test]
    fn score_tokens_rejects_out_of_range_target() {
        let per_pos = vec![vec![0.0f32, 1.0]];
        let tokens = vec![0u32, 5]; // target 5 >= vocab 2
        assert!(score_tokens(&per_pos, &tokens).is_err());
    }

    #[test]
    fn score_tokens_uniform_logits_is_ln_vocab() {
        // Uniform logits over vocab V -> per-token NLL = ln(V) exactly.
        let per_pos = vec![vec![0.0f32; 8]];
        let tokens = vec![0u32, 3];
        let (nll, count) = score_tokens(&per_pos, &tokens).unwrap();
        assert_eq!(count, 1);
        assert!((nll - (8f64).ln()).abs() < 1e-9);
    }

    // --- reference path is deterministic (the baseline must be stable) -----

    #[test]
    fn reference_perplexity_is_exactly_reproducible() {
        let root = tmp("ref-determinism");
        let (model, carved) = fixture_carve(&root, Dtype::Bf16);
        let manifest = Manifest::load(&carved.join(crate::manifest::FILE_NAME)).unwrap();
        let hidden = manifest.model.hidden as usize;
        let inter = manifest.model.inter as usize;
        let spine = Spine::load(&model, &manifest, fixture_config(hidden)).unwrap();
        let prompts: Vec<Vec<u32>> = (0..3)
            .map(|p| canary_prompt(42, p, 6, spine.vocab()))
            .collect();

        let mut a = SourceRefDispatch::new(&model, hidden, inter).unwrap();
        let (na, ca) = teacher_forced_nll(&spine, &mut a, &prompts).unwrap();
        let mut b = SourceRefDispatch::new(&model, hidden, inter).unwrap();
        let (nb, cb) = teacher_forced_nll(&spine, &mut b, &prompts).unwrap();
        assert_eq!((ca, cb), (3 * 5, 3 * 5), "prompts x (len - 1) transitions");
        assert_eq!(
            na.to_bits(),
            nb.to_bits(),
            "reference NLL is bit-reproducible"
        );
    }

    // --- end-to-end canary on the fp8 fixture carve (deterministic, model-free)

    #[test]
    fn fixture_fp8_canary_is_finite_and_deterministic() {
        let root = tmp("fp8-e2e");
        let (model, carved) = fixture_carve(&root, Dtype::Fp8);
        let opts = CanaryOptions {
            prompts: 3,
            len: 8,
            seed: 7,
            codec: "fp8".into(),
            config: fixture_config(8),
        };
        let r1 = run(&model, &carved, &opts).unwrap();
        let r2 = run(&model, &carved, &opts).unwrap();

        assert_eq!(r1.dtype, Dtype::Fp8);
        assert_eq!(r1.scored_tokens, 3 * 7, "prompts x (len - 1)");
        assert!(
            r1.ppl_test.is_finite() && r1.ppl_test > 0.0,
            "ppl_test {}",
            r1.ppl_test
        );
        assert!(
            r1.ppl_ref.is_finite() && r1.ppl_ref > 0.0,
            "ppl_ref {}",
            r1.ppl_ref
        );
        // The delta is a real number for BENCH — sign not asserted (a random-weight
        // fixture is not a language model), only that the harness produces it.
        assert!(r1.delta_ppl.is_finite());
        // Deterministic to the bit across runs.
        assert_eq!(r1.ppl_test.to_bits(), r2.ppl_test.to_bits());
        assert_eq!(r1.ppl_ref.to_bits(), r2.ppl_ref.to_bits());
        assert_eq!(r1.delta_ppl.to_bits(), r2.delta_ppl.to_bits());
    }

    #[test]
    fn run_rejects_degenerate_args() {
        let root = tmp("bad-args");
        let (model, carved) = fixture_carve(&root, Dtype::Fp8);
        let cfg = fixture_config(8); // Config is Copy
        let opts = |f: &dyn Fn(&mut CanaryOptions)| -> CanaryOptions {
            let mut o = CanaryOptions {
                config: cfg,
                ..CanaryOptions::default()
            };
            f(&mut o);
            o
        };
        assert!(
            run(&model, &carved, &opts(&|o| o.len = 1)).is_err(),
            "len < 2 has no scored transition"
        );
        assert!(
            run(&model, &carved, &opts(&|o| o.prompts = 0)).is_err(),
            "zero prompts is empty"
        );
        assert!(
            run(&model, &carved, &opts(&|o| o.codec = "int8".into())).is_err(),
            "unknown codec is rejected"
        );
    }
}
