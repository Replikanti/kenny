//! `kenny prefix` — the ADR-0022 prefix-cache identity primitive + hit-rate.
//!
//! Prefill is existential (MANIFESTO §4.5): agent colonies sharing a system
//! prompt and tool definitions reach 80–90 %+ prompt reuse, and every reused
//! prompt token is prefill bytes that never touch the star. ADR-0022's survival
//! metric is the prefix-cache HIT RATE, and its correctness mechanism is content
//! addressing over token blocks — NOT client-declared cache ids.
//!
//! This module is that identity primitive plus the hit-rate meter. It is
//! spine-LOCAL (ADR-0004): the block keys, the lookup structure, and the tallies
//! never touch the wire, a manifest, or another node — losing the cache costs
//! recomputation, never correctness. It touches NO consensus surface (no
//! `WIRE_VERSION`, no codec, no manifest change); the block-key encoding IS a
//! canonical encoding, so it gets its own spine-local golden ([`block_key`]'s
//! chain), but it is explicitly not an interop format.
//!
//! Identity (ADR-0022): prompt tokens are chunked into fixed-size blocks, and
//! each block's key is a blake3 hash chain rooted in the model identity
//! (ADR-0005),
//!
//! ```text
//! key_{-1} = model_identity
//! key_n    = blake3(model_identity ++ key_{n-1} ++ canonical(tokens_n))
//! ```
//!
//! so two streams sharing a prompt prefix produce IDENTICAL block keys over the
//! shared blocks and diverge at the first differing block — dedup by
//! construction, zero client coordination, the same idiom the blob store uses.
//! A one-token difference invalidates every subsequent block (exact-match
//! semantics, ADR-0022) — that is precisely the pressure toward stable system
//! prompts.
//!
//! What lands here is the M4 rescoped slice: the identity primitive + the
//! hit-rate metric, measured on a shared-system-prompt fixture so the number is
//! real and non-zero. The KV memory hierarchy (int4/NVMe tiers, weighted-LRU
//! eviction, decode-first admission) stays deferred — it is only measurable
//! against a real memory-bound, concurrency-bound serving loop (the real party,
//! #6), and ADR-0022 stays `proposed` until then. **KV occupancy** (dashboard
//! #2) is reported now as a DERIVED number from the existing `LayerKv`
//! ([`crate::spine::kv_occupancy_bytes`]), not a new subsystem.

use std::collections::HashSet;
use std::path::Path;

use crate::error::{Error, Result};
use crate::manifest::Manifest;
use crate::rng::SplitMix64;
use crate::spine::kv_occupancy_bytes;

/// A 32-byte content-addressed block key — one link of the ADR-0022 hash chain.
pub type BlockKey = [u8; 32];

/// Canonical little-endian encoding of a token block: each `u32` token as 4 LE
/// bytes, concatenated (the x86/LE discipline the wire and manifest already use,
/// ADR-0011). The block length is implied by the byte count, so a partial
/// trailing block hashes to its own distinct key.
fn canonical_block(tokens: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(tokens.len() * 4);
    for &t in tokens {
        b.extend_from_slice(&t.to_le_bytes());
    }
    b
}

/// One link of the ADR-0022 hash chain:
/// `blake3(model_identity ++ prev ++ canonical(tokens))`, where `prev` is the
/// previous block's key (and the model identity itself for the first block —
/// `key_{-1} = model_identity`). The model identity is folded in at every link
/// so a block key is meaningless under a different model (a carve mismatch can
/// never alias another model's KV).
pub fn block_key(identity: &[u8; 32], prev: &BlockKey, tokens: &[u32]) -> BlockKey {
    let mut h = blake3::Hasher::new();
    h.update(identity);
    h.update(prev);
    h.update(&canonical_block(tokens));
    *h.finalize().as_bytes()
}

/// The reuse of one registered prompt: how many of its tokens were served from
/// the cache (a leading run of already-known blocks) out of its total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockStats {
    pub reused_tokens: usize,
    pub total_tokens: usize,
}

/// The spine-local prefix cache: a radix over content-addressed block keys.
///
/// Because each key chains its predecessor, walking a new stream's block keys in
/// order and testing membership finds the LONGEST shared prefix — the chain
/// collapses the trie into content-addressed membership, so a flat set over the
/// chained keys IS the radix lookup ADR-0022 names. It holds only block keys, no
/// KV payload: this is the identity + hit-rate primitive, not the (deferred) KV
/// memory hierarchy.
#[derive(Debug, Clone)]
pub struct PrefixCache {
    identity: [u8; 32],
    block_tokens: usize,
    known: HashSet<BlockKey>,
    reused: u64,
    total: u64,
}

impl PrefixCache {
    /// A cache rooted in `identity` (the manifest identity, ADR-0005) with a
    /// fixed block size. ADR-0022 puts the block size on the order of 256 tokens;
    /// it is a tunable constant here so a measurement can pick a granularity that
    /// exposes reuse on shorter fixtures (the golden pins the ENCODING, not the
    /// block size).
    pub fn new(identity: [u8; 32], block_tokens: usize) -> PrefixCache {
        PrefixCache {
            identity,
            // A zero block size would panic `slice::chunks`; clamp to a 1-token
            // floor so the constructor is infallible (the CLI/`run` reject 0 up
            // front with a clear message — this is only a safety floor).
            block_tokens: block_tokens.max(1),
            known: HashSet::new(),
            reused: 0,
            total: 0,
        }
    }

    /// Register a prompt's tokens, returning this prompt's reuse. `reused_tokens`
    /// is the leading run of blocks whose chained key was ALREADY known (a hit =
    /// that block's KV can be served from cache without an expert dispatch). By
    /// the hash chain a miss invalidates every later block, so hits are a
    /// contiguous prefix (ADR-0022 exact-match semantics). Every block — hit or
    /// miss — is then inserted, so a later stream sharing this prefix hits.
    pub fn register(&mut self, tokens: &[u32]) -> BlockStats {
        let mut prev = self.identity;
        let mut reused = 0usize;
        let mut still_hitting = true;
        for block in tokens.chunks(self.block_tokens) {
            let key = block_key(&self.identity, &prev, block);
            // A block hits only if every prior block hit too: once a block
            // misses, `prev` is a novel key so no later key can be known anyway,
            // but gate explicitly to make the exact-match invalidation clear.
            if still_hitting && self.known.contains(&key) {
                reused += block.len();
            } else {
                still_hitting = false;
            }
            self.known.insert(key);
            prev = key;
        }
        self.reused += reused as u64;
        self.total += tokens.len() as u64;
        BlockStats {
            reused_tokens: reused,
            total_tokens: tokens.len(),
        }
    }

    /// `prefix_hit_rate = reused_prompt_tokens / total_prompt_tokens` over every
    /// prompt registered so far (ADR-0022 dashboard #5). `0.0` when nothing has
    /// been registered.
    pub fn hit_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.reused as f64 / self.total as f64
        }
    }

    /// Total reused prompt tokens across every registration (the hit-rate
    /// numerator).
    pub fn reused_tokens(&self) -> u64 {
        self.reused
    }

    /// Total prompt tokens across every registration (the hit-rate denominator).
    pub fn total_tokens(&self) -> u64 {
        self.total
    }

    /// Distinct block keys resident — the size of the radix, a spine bookkeeping
    /// figure (ADR-0022: the cache is new state, but all of it is recomputable).
    pub fn distinct_blocks(&self) -> usize {
        self.known.len()
    }
}

/// Options for the `kenny prefix` shared-prompt hit-rate measurement.
pub struct PrefixOptions {
    /// Number of streams sharing the system prompt.
    pub streams: usize,
    /// Length of the system prompt every stream shares (its reusable prefix).
    pub system_len: usize,
    /// Length of each stream's distinct user tail (never shared).
    pub user_len: usize,
    /// Block size in tokens (ADR-0022 tunable; the granularity/overhead knob).
    pub block_tokens: usize,
    /// Seed keying the deterministic shared prompt + per-stream user tails.
    pub seed: u64,
    /// Vocabulary the synthetic token ids are drawn from — the cache operates on
    /// ids alone (no embedding lookup), so this is only for realistic ids and
    /// does not affect the hit-rate.
    pub vocab: u64,
    /// KV heads / head dim for the DERIVED KV-occupancy number (ADR-0022 #2).
    /// Default to the Qwen3-30B-A3B card; the fixture (square attention) is
    /// `--num-kv-heads 1 --head-dim <hidden>`.
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Default for PrefixOptions {
    fn default() -> Self {
        PrefixOptions {
            streams: 8,
            system_len: 256,
            user_len: 64,
            block_tokens: 64,
            seed: 42,
            // Qwen3-30B-A3B vocabulary (the model card) — realistic ids only.
            vocab: 151_936,
            // Qwen3-30B-A3B card: 4 KV heads (GQA), head_dim 128.
            num_kv_heads: 4,
            head_dim: 128,
        }
    }
}

/// The measured prefix-cache report (ADR-0022 dashboard #5 + the derived #2).
#[derive(Debug, Clone)]
pub struct PrefixReport {
    pub streams: usize,
    pub system_len: usize,
    pub user_len: usize,
    pub block_tokens: usize,
    pub total_prompt_tokens: u64,
    pub reused_prompt_tokens: u64,
    /// `reused / total` — the survival metric.
    pub hit_rate: f64,
    pub distinct_blocks: usize,
    /// MoE layers the KV cache spans (the spine's `LayerKv` count) — the derived
    /// occupancy's layer factor.
    pub kv_layers: usize,
    /// `B × ctx × layers × kv_elem`, ctx = system_len + user_len (ADR-0022 #2).
    pub kv_occupancy_bytes: u64,
}

/// Measure the prefix-cache hit-rate on a SHARED-SYSTEM-PROMPT fixture (the
/// number is only meaningful against shared prompts — random independent prompts
/// have no reuse). Model-free: it reads the carve's manifest for the model
/// identity (ADR-0005) and MoE layer count, generates `streams` prompts that all
/// share a `system_len`-token system prompt and each carry a distinct
/// seed-derived `user_len`-token tail, registers them into a [`PrefixCache`], and
/// reports the aggregate hit-rate alongside the derived KV occupancy. No model
/// weights are loaded, so this runs in plain CI.
pub fn run(carved_dir: &Path, opts: &PrefixOptions) -> Result<PrefixReport> {
    if opts.streams == 0 {
        return Err(Error::usage("prefix: --streams must be at least 1"));
    }
    if opts.system_len == 0 && opts.user_len == 0 {
        return Err(Error::usage(
            "prefix: a prompt needs at least one token (--system-len + --user-len > 0)",
        ));
    }
    if opts.block_tokens == 0 {
        return Err(Error::usage("prefix: --block must be at least 1 token"));
    }
    if opts.vocab == 0 {
        return Err(Error::usage("prefix: --vocab must be at least 1"));
    }

    let manifest = Manifest::load(&carved_dir.join(crate::manifest::FILE_NAME))?;
    let identity = *blake3::hash(&manifest.canonical_bytes()).as_bytes();

    // MoE layer count = the distinct layers the carve holds experts for, the same
    // count the spine's per-layer `LayerKv` cache spans.
    let mut layers: Vec<u16> = manifest.experts.iter().map(|e| e.layer).collect();
    layers.sort_unstable();
    layers.dedup();
    let kv_layers = layers.len();

    // The shared system prompt (identical across streams) + one distinct user
    // tail per stream, all deterministic from the seed.
    let system = synth_tokens(opts.seed, "prefix.system", opts.system_len, opts.vocab);
    let mut cache = PrefixCache::new(identity, opts.block_tokens);
    for s in 0..opts.streams {
        let tail = synth_tokens(
            opts.seed,
            &format!("prefix.user.{s}"),
            opts.user_len,
            opts.vocab,
        );
        let mut prompt = system.clone();
        prompt.extend_from_slice(&tail);
        cache.register(&prompt);
    }

    let ctx = opts.system_len + opts.user_len;
    let kv_occupancy_bytes = kv_occupancy_bytes(
        kv_layers,
        opts.num_kv_heads,
        opts.head_dim,
        opts.streams,
        ctx,
    );

    Ok(PrefixReport {
        streams: opts.streams,
        system_len: opts.system_len,
        user_len: opts.user_len,
        block_tokens: opts.block_tokens,
        total_prompt_tokens: cache.total_tokens(),
        reused_prompt_tokens: cache.reused_tokens(),
        hit_rate: cache.hit_rate(),
        distinct_blocks: cache.distinct_blocks(),
        kv_layers,
        kv_occupancy_bytes,
    })
}

/// A deterministic run of `len` in-vocab token ids keyed by `(seed, name)` — the
/// same `SplitMix64::for_name` idiom the spine and canary use for reproducible
/// fixtures.
fn synth_tokens(seed: u64, name: &str, len: usize, vocab: u64) -> Vec<u32> {
    let mut rng = SplitMix64::for_name(seed, name);
    (0..len).map(|_| (rng.next_u64() % vocab) as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    // --- CONSENSUS-ADJACENT GOLDEN ---------------------------------------
    // The block-key encoding is a spine-local CANONICAL encoding (ADR-0022): it
    // is not on the wire, but a change to it silently discards every cached KV
    // block. Treat it like the wire goldens — a change here is a cache-identity
    // change: update ADR-0022 + this golden together, never a bare test edit
    // (the kenny-format-auditor gate).

    #[test]
    fn golden_block_key_chain() {
        // Fixed identity [0,1,..,31] + fixed two-token blocks, so the exact
        // canonical bytes hashed are pinned. Block 0 chains from the identity
        // (key_{-1} = identity); block 1 chains from block 0's key.
        let identity: [u8; 32] = std::array::from_fn(|i| i as u8);
        let k0 = block_key(&identity, &identity, &[1, 2]);
        let k1 = block_key(&identity, &k0, &[3, 4]);
        assert_eq!(
            hex(&k0),
            "0a3f96f9253008a18cc97625915b4b2b63b3b201d1b347fda48a2f60ced95603",
        );
        assert_eq!(
            hex(&k1),
            "00d62d98104760c216fcca201de6a79e13c0811a2aeb35b9ea54103dd85b3b82",
        );
    }

    // --- shared-prompt hit-rate (the real, non-zero signal) --------------

    #[test]
    fn shared_system_prompt_hits_after_first_stream() {
        // block = 4 tokens; an 8-token (2-block) shared system prompt.
        let id = [7u8; 32];
        let system: Vec<u32> = (0..8).collect();
        let mut cache = PrefixCache::new(id, 4);

        // Stream 0 is cold: every block is a miss.
        let mut s0 = system.clone();
        s0.extend_from_slice(&[100, 101, 102, 103]);
        let r0 = cache.register(&s0);
        assert_eq!(r0.reused_tokens, 0, "the first stream primes, hits nothing");
        assert_eq!(r0.total_tokens, 12);

        // Stream 1 shares the system prompt, distinct tail: the 2 system blocks
        // (8 tokens) hit; the distinct tail block misses.
        let mut s1 = system.clone();
        s1.extend_from_slice(&[200, 201, 202, 203]);
        let r1 = cache.register(&s1);
        assert_eq!(r1.reused_tokens, 8, "both shared system blocks are hits");

        // Aggregate: 8 reused of 24 total = 1/3.
        assert_eq!(cache.reused_tokens(), 8);
        assert_eq!(cache.total_tokens(), 24);
        assert!((cache.hit_rate() - (8.0 / 24.0)).abs() < 1e-12);
    }

    #[test]
    fn one_token_divergence_invalidates_every_later_block() {
        // Warm the cache with a 3-block (12-token) prompt, then register a prompt
        // that matches the first block but diverges by ONE token in the second:
        // block 0 hits, blocks 1 and 2 both miss (the chain propagates).
        let id = [9u8; 32];
        let base: Vec<u32> = (0..12).collect();
        let mut cache = PrefixCache::new(id, 4);
        cache.register(&base);

        let mut diverged = base.clone();
        diverged[5] = 999; // a token in block 1 (indices 4..8)
        let r = cache.register(&diverged);
        assert_eq!(
            r.reused_tokens, 4,
            "only block 0 (unchanged) hits; the divergence kills blocks 1 and 2"
        );

        // A first-block divergence invalidates everything.
        let mut early = base.clone();
        early[0] = 999;
        let r2 = cache.register(&early);
        assert_eq!(r2.reused_tokens, 0, "a first-block divergence hits nothing");
    }

    #[test]
    fn hit_rate_is_deterministic() {
        // The same registration sequence reproduces the hit-rate to the bit.
        let build = || {
            let id = [3u8; 32];
            let system: Vec<u32> = (0..16).collect();
            let mut cache = PrefixCache::new(id, 4);
            for s in 0..5u32 {
                let mut p = system.clone();
                p.extend_from_slice(&[1000 + s, 2000 + s]);
                cache.register(&p);
            }
            cache.hit_rate()
        };
        assert_eq!(build().to_bits(), build().to_bits());
    }

    #[test]
    fn independent_prompts_never_hit() {
        // No shared prefix ⇒ a hit-rate of exactly 0 (the mitigation for the
        // "meaningless hit-rate on random prompts" risk: only shared prompts
        // produce signal, so the fixture deliberately shares a system prompt).
        let id = [1u8; 32];
        let mut cache = PrefixCache::new(id, 4);
        for s in 0..4u32 {
            let p = vec![s * 100, s * 100 + 1, s * 100 + 2, s * 100 + 3];
            let r = cache.register(&p);
            assert_eq!(r.reused_tokens, 0);
        }
        assert_eq!(cache.hit_rate(), 0.0);
    }

    // --- end-to-end run over a carved fixture (model-free, deterministic) ---

    #[test]
    fn run_measures_shared_prompt_hit_rate() {
        use crate::blob::Dtype;
        use crate::carve::{self, Options};
        use crate::fixture::{self, Params};

        let root = std::env::temp_dir().join("kenny-prefix-run");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
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

        let opts = PrefixOptions {
            streams: 8,
            system_len: 64,
            user_len: 16,
            block_tokens: 16,
            seed: 42,
            vocab: 32,
            // Fixture is square attention (num_kv_heads 1, head_dim = hidden 8).
            num_kv_heads: 1,
            head_dim: 8,
        };
        let r1 = run(&carved, &opts).unwrap();
        let r2 = run(&carved, &opts).unwrap();

        // 8 streams × 80 tokens = 640 total; stream 0 primes (miss), streams
        // 1..8 each reuse the full 64-token shared system prompt (4 blocks of
        // 16): reused = 7 × 64 = 448.
        assert_eq!(r1.total_prompt_tokens, 8 * 80);
        assert_eq!(r1.reused_prompt_tokens, 7 * 64);
        assert!((r1.hit_rate - (7.0 * 64.0) / (8.0 * 80.0)).abs() < 1e-12);
        assert_eq!(r1.kv_layers, 2, "fixture default has 2 MoE layers");
        // KV occupancy = B(8) × ctx(80) × layers(2) × kv_elem(2×1×8×4 = 64).
        assert_eq!(r1.kv_occupancy_bytes, 8 * 80 * 2 * 64);
        // Deterministic to the bit across runs.
        assert_eq!(r1.hit_rate.to_bits(), r2.hit_rate.to_bits());
    }

    #[test]
    fn run_rejects_degenerate_args() {
        use crate::blob::Dtype;
        use crate::carve::{self, Options};
        use crate::fixture::{self, Params};

        let root = std::env::temp_dir().join("kenny-prefix-badargs");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
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

        let bad = |f: &dyn Fn(&mut PrefixOptions)| -> PrefixOptions {
            let mut o = PrefixOptions {
                num_kv_heads: 1,
                head_dim: 8,
                vocab: 32,
                ..PrefixOptions::default()
            };
            f(&mut o);
            o
        };
        assert!(run(&carved, &bad(&|o| o.streams = 0)).is_err());
        assert!(
            run(
                &carved,
                &bad(&|o| {
                    o.system_len = 0;
                    o.user_len = 0;
                })
            )
            .is_err()
        );
        assert!(run(&carved, &bad(&|o| o.block_tokens = 0)).is_err());
    }
}
