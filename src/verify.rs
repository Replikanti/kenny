//! Verification spot-checks (ADR-0015): the trusted-pool integrity retrofit.
//!
//! ADR-0015 keeps the pool a SOCIAL boundary (trusted parties only) but designs
//! for a spine that occasionally re-checks a node's work: sample a fraction of
//! answered dispatches, recompute the same `(layer, expert)` locally from the
//! bf16 source, and accumulate per-node trust. This is that mechanism, built on
//! the recompute primitive the canary already carries ([`crate::canary::SourceRefDispatch`]
//! / `diff.rs::source_matrix`) — one bf16-source truth, now a second consumer.
//!
//! **Tolerance-based, not byte-exact.** ADR-0018 is still `proposed`: only the
//! fp8 half of the numeric-path table is measured, and fp8 FMA reordering means
//! two correct nodes can differ in bits. On top of that the node computes on the
//! codec-rounded activation while the oracle sees the raw f32, so even a perfectly
//! honest node is only tolerance-close, never bit-equal. The comparison is
//! therefore a cosine + relative-error envelope ([`Tolerance`]); the exact
//! byte-compare lane is blocked on ADR-0018's `Int8Codec` arm and is a labeled
//! follow-up, exactly as ADR-0015 scopes it ("whether comparison is exact or
//! tolerance-based follows ADR-0018").
//!
//! Everything here is spine-LOCAL (ADR-0004): the trust tally, the sampling RNG,
//! and the recompute never touch the wire, a manifest, or another node — the
//! exact posture of [`crate::placement::HeatMap`]. Wrapping a dispatcher does not
//! change its output (`WIRE_VERSION` stays 1, every wire golden byte-identical):
//! [`VerifyingDispatch`] returns the inner answers verbatim and only observes.
//!
//! Trust weighting is agreement-count ONLY: no stake, no reputation economy —
//! that stays deferred in ADR-0015 until a real party runs it.

use std::collections::BTreeMap;

use crate::error::Result;
use crate::placement::PlacementMap;
use crate::rng::SplitMix64;
use crate::spine::Dispatcher;

/// Per-node agree/disagree tallies from the spot-checks (mirror of
/// [`crate::placement::Counts`]). `checks == agreements + disagreements`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TrustCounts {
    pub checks: u64,
    pub agreements: u64,
    pub disagreements: u64,
}

/// Per-node accumulated verification trust (ADR-0015), keyed by node index into
/// the placement's `nodes` slice — the same index a [`PlacementMap`] replica set
/// carries. Spine-LOCAL: never serialized, never cross-node (ADR-0004), the exact
/// posture of the dispatch [`crate::placement::HeatMap`]. Ordered so a distrust
/// verdict is deterministic.
#[derive(Debug, Default, Clone)]
pub struct TrustTally {
    per_node: BTreeMap<usize, TrustCounts>,
}

impl TrustTally {
    pub fn new() -> TrustTally {
        TrustTally::default()
    }

    /// Record one spot-check outcome against `node`.
    fn record(&mut self, node: usize, agree: bool) {
        let c = self.per_node.entry(node).or_default();
        c.checks += 1;
        if agree {
            c.agreements += 1;
        } else {
            c.disagreements += 1;
        }
    }

    /// Counts for one node (zero if never spot-checked).
    pub fn get(&self, node: usize) -> TrustCounts {
        self.per_node.get(&node).copied().unwrap_or_default()
    }

    /// Ordered iteration over `(node index, counts)`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, TrustCounts)> + '_ {
        self.per_node.iter().map(|(&k, &c)| (k, c))
    }

    pub fn is_empty(&self) -> bool {
        self.per_node.is_empty()
    }

    /// The ADR-0015 distrust verdict: nodes spot-checked at least `min_checks`
    /// times whose disagreement fraction is at least `num / den`. Integer
    /// comparison (`disagreements × den >= checks × num`) keeps it exact, the
    /// mirror of [`crate::placement::HeatMap::suspect`]. A never-checked node
    /// (`checks == 0`) is NEVER distrusted regardless of `min_checks` — the
    /// explicit `checks > 0` guard keeps a 0/0 node off the alarm (an unchecked
    /// node is not a lying one).
    pub fn distrusted(&self, min_checks: u64, num: u64, den: u64) -> Vec<usize> {
        self.per_node
            .iter()
            .filter(|(_, c)| {
                c.checks > 0
                    && c.checks >= min_checks
                    && (c.disagreements as u128) * (den as u128)
                        >= (c.checks as u128) * (num as u128)
            })
            .map(|(&k, _)| k)
            .collect()
    }
}

/// The tolerance envelope for one spot-check comparison (ADR-0018): the node's
/// answered `y` and the bf16-source recompute AGREE when their cosine similarity
/// is at least `min_cosine` AND their relative L2 error is at most `max_rel`.
/// Cosine catches a direction lie, the relative-error bound catches a scale lie
/// (a scaled-up correct vector has cosine 1 but is still wrong). Two near-zero
/// vectors (both norms below `zero_norm`) agree vacuously — there is no direction
/// to compare.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub min_cosine: f64,
    pub max_rel: f64,
    pub zero_norm: f64,
}

impl Default for Tolerance {
    /// Comfortably admits an honest fp8 node (M0 per-expert 1−cos ≈ 1e-3, M1
    /// end-to-end wire cosine 0.99985) while a garbage answer — wrong direction
    /// or wrong scale — falls well outside. Deliberately loose because the axis
    /// is tolerance-based by ADR-0018 necessity, not tight verification.
    fn default() -> Self {
        Tolerance {
            min_cosine: 0.99,
            max_rel: 0.25,
            zero_norm: 1e-9,
        }
    }
}

impl Tolerance {
    /// Whether `got` (the node's answer) is within the envelope of `want` (the
    /// bf16-source recompute). Accumulates in f64 for a stable comparison.
    pub fn agrees(&self, got: &[f32], want: &[f32]) -> bool {
        if got.len() != want.len() {
            return false;
        }
        let (mut dot, mut ng, mut nw, mut nd) = (0f64, 0f64, 0f64, 0f64);
        for (&g, &w) in got.iter().zip(want) {
            let (g, w) = (g as f64, w as f64);
            dot += g * w;
            ng += g * g;
            nw += w * w;
            nd += (g - w) * (g - w);
        }
        // Both essentially zero: vacuously equal (no direction, no scale).
        if ng <= self.zero_norm && nw <= self.zero_norm {
            return true;
        }
        // Exactly one near-zero: a real mismatch (one side answered, one did not).
        if ng <= self.zero_norm || nw <= self.zero_norm {
            return false;
        }
        let cosine = dot / (ng.sqrt() * nw.sqrt());
        // Relative L2 error against the reference norm (the ground truth's scale).
        let rel = nd.sqrt() / nw.sqrt();
        cosine >= self.min_cosine && rel <= self.max_rel
    }
}

/// A [`Dispatcher`] decorator that spot-checks a sampled fraction of a wrapped
/// dispatcher's answers against a bf16-source recompute ORACLE and accumulates
/// per-node trust (ADR-0015). Transparent: it returns the inner answers verbatim,
/// so a spine driven through `VerifyingDispatch<PlacedDispatch>` generates exactly
/// what the bare `PlacedDispatch` would (verification is off the critical path —
/// no wire change, `WIRE_VERSION` 1, goldens frozen).
///
/// Attribution is the round-0 PRIMARY of the answered expert
/// (`map.replicas_of(layer, e)[0]`): on the unhedged placed path that is the node
/// that actually answered, so trust lands on the right node. Verification is
/// scoped to the unhedged path (hedging spills to other replicas, blurring "who
/// answered"); the CLI enforces that pairing.
pub struct VerifyingDispatch<D: Dispatcher> {
    inner: D,
    /// The bf16-source recompute oracle (ground truth), typically a
    /// [`crate::canary::SourceRefDispatch`] over the source model dir.
    oracle: Box<dyn Dispatcher>,
    /// Spine-local placement, for per-node attribution of an answered expert.
    map: PlacementMap,
    tol: Tolerance,
    /// Deterministic sampling stream: one draw per ANSWERED expert, in fixed
    /// (item, position) order, so a run's spot-check set is reproducible.
    rng: SplitMix64,
    /// Sampling fraction in per-mille: a draw `< sample_permille` (mod 1000)
    /// spot-checks that answer. `1000` checks every answer; `0` checks none.
    sample_permille: u64,
    trust: TrustTally,
    spot_checks: u64,
    disagreements: u64,
}

impl<D: Dispatcher> VerifyingDispatch<D> {
    /// Wrap `inner` with spot-checking against `oracle`, attributing by `map`,
    /// sampling `sample_permille` ‰ of answers, seeded by `seed`. `sample_permille`
    /// is clamped to `[0, 1000]`.
    pub fn new(
        inner: D,
        oracle: Box<dyn Dispatcher>,
        map: PlacementMap,
        sample_permille: u64,
        seed: u64,
        tol: Tolerance,
    ) -> VerifyingDispatch<D> {
        VerifyingDispatch {
            inner,
            oracle,
            map,
            tol,
            rng: SplitMix64::new(seed),
            sample_permille: sample_permille.min(1000),
            trust: TrustTally::new(),
            spot_checks: 0,
            disagreements: 0,
        }
    }

    /// The accumulated per-node trust tally.
    pub fn trust(&self) -> &TrustTally {
        &self.trust
    }

    /// Total spot-checks performed over the run.
    pub fn spot_checks(&self) -> u64 {
        self.spot_checks
    }

    /// Spot-checks that fell outside the tolerance envelope.
    pub fn disagreements(&self) -> u64 {
        self.disagreements
    }

    /// Read-only access to the wrapped dispatcher.
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Recover the wrapped dispatcher (dropping the verification state).
    pub fn into_inner(self) -> D {
        self.inner
    }

    /// Spot-check the answered experts of one dispatch batch: for each answered
    /// `(item, position)`, draw the sampling RNG in fixed order; on a hit recompute
    /// the expert from the oracle and record agree/disagree against the answered
    /// expert's round-0 primary. Answers that were not held (`None`) and experts
    /// with no placement (unattributable) are skipped.
    fn spot_check(
        &mut self,
        layer: u16,
        items: &[(&[f32], &[u16])],
        ys: &[Vec<Option<Vec<f32>>>],
    ) -> Result<()> {
        for (&(x, experts), per_expert) in items.iter().zip(ys) {
            for (pos, &e) in experts.iter().enumerate() {
                let Some(got) = per_expert.get(pos).and_then(|o| o.as_ref()) else {
                    continue; // not answered — nothing to verify
                };
                // One deterministic draw per answered expert (fixed order).
                let hit = self.rng.next_u64() % 1000 < self.sample_permille;
                if !hit {
                    continue;
                }
                let Some(&node) = self.map.replicas_of(layer, e).first() else {
                    continue; // no placement -> cannot attribute the answer to a node
                };
                // Recompute the SAME (layer, expert) from the bf16 source oracle.
                let refy = self.oracle.dispatch(layer, x, &[e])?;
                let agree = match refy.first().and_then(|o| o.as_ref()) {
                    Some(want) => self.tol.agrees(got, want),
                    // The oracle holds every expert; a not-held here is a real
                    // disagreement (the node answered where the truth cannot).
                    None => false,
                };
                self.trust.record(node, agree);
                self.spot_checks += 1;
                if !agree {
                    self.disagreements += 1;
                }
            }
        }
        Ok(())
    }
}

impl<D: Dispatcher> Dispatcher for VerifyingDispatch<D> {
    fn dispatch(
        &mut self,
        layer: u16,
        x: &[f32],
        experts: &[u16],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let ys = self.inner.dispatch(layer, x, experts)?;
        self.spot_check(layer, &[(x, experts)], std::slice::from_ref(&ys))?;
        Ok(ys)
    }

    fn dispatch_batch(
        &mut self,
        layer: u16,
        items: &[(&[f32], &[u16])],
    ) -> Result<Vec<Vec<Option<Vec<f32>>>>> {
        let ys = self.inner.dispatch_batch(layer, items)?;
        self.spot_check(layer, items, &ys)?;
        Ok(ys)
    }

    // Observability forwards to the wrapped dispatcher so BENCH accounting is
    // unchanged by the wrap (the verification state is a pure side-channel).
    fn wire_bytes(&self) -> (u64, u64) {
        self.inner.wire_bytes()
    }

    fn layer_timeouts(&self) -> u64 {
        self.inner.layer_timeouts()
    }

    fn hedges_fired(&self) -> u64 {
        self.inner.hedges_fired()
    }

    fn suspect_replicas(&self) -> Vec<(u16, u16)> {
        self.inner.suspect_replicas()
    }

    fn trust(&self) -> Option<&TrustTally> {
        Some(&self.trust)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- TrustTally math (no model) ---------------------------------------

    #[test]
    fn distrusted_flags_a_lying_node_only() {
        let mut t = TrustTally::new();
        for _ in 0..10 {
            t.record(0, true); // honest node: 0 disagreements
        }
        for _ in 0..10 {
            t.record(1, false); // lying node: 10/10 disagree
        }
        t.record(2, false); // one sample only
        assert_eq!(
            t.get(0),
            TrustCounts {
                checks: 10,
                agreements: 10,
                disagreements: 0
            }
        );
        assert_eq!(
            t.get(1),
            TrustCounts {
                checks: 10,
                agreements: 0,
                disagreements: 10
            }
        );
        // >= 50 % disagreement over >= 5 checks: only the fully-lying node 1.
        assert_eq!(t.distrusted(5, 1, 2), vec![1]);
        // A never-checked node is never distrusted, even at min_checks 0.
        assert!(!t.distrusted(0, 1, 1).contains(&9));
    }

    // --- tolerance envelope ------------------------------------------------

    #[test]
    fn tolerance_admits_close_rejects_garbage() {
        let tol = Tolerance::default();
        let want = vec![1.0f32, -2.0, 3.0, 0.5];
        // A tiny fp8-scale perturbation agrees.
        let close: Vec<f32> = want.iter().map(|&w| w * 1.001 + 0.0005).collect();
        assert!(tol.agrees(&close, &want), "an fp8-close answer agrees");
        // A direction lie disagrees (negated).
        let flipped: Vec<f32> = want.iter().map(|&w| -w).collect();
        assert!(!tol.agrees(&flipped, &want), "a negated answer disagrees");
        // A scale lie disagrees (same direction, 3× magnitude — cosine 1 but the
        // relative-error bound catches it).
        let scaled: Vec<f32> = want.iter().map(|&w| w * 3.0).collect();
        assert!(!tol.agrees(&scaled, &want), "a scaled answer disagrees");
        // Garbage disagrees.
        assert!(!tol.agrees(&[9.0, 9.0, 9.0, 9.0], &want));
        // Two zeros agree vacuously; one zero is a mismatch.
        assert!(tol.agrees(&[0.0; 4], &[0.0; 4]));
        assert!(!tol.agrees(&[0.0; 4], &want));
    }

    // --- VerifyingDispatch with mock inner + mock oracle (no model) --------

    /// A pool that answers `y_e = base_e` for honest experts and a fixed garbage
    /// vector for experts whose primary is the `lying` node — a byzantine peer
    /// with correct framing but wrong numbers.
    struct MockPool {
        hidden: usize,
        map: PlacementMap,
        lying: Option<usize>,
    }

    impl Dispatcher for MockPool {
        fn dispatch(
            &mut self,
            _layer: u16,
            _x: &[f32],
            experts: &[u16],
        ) -> Result<Vec<Option<Vec<f32>>>> {
            Ok(experts
                .iter()
                .map(|&e| {
                    let primary = self.map.replicas_of(_layer, e).first().copied();
                    if self.lying.is_some() && primary == self.lying {
                        Some(vec![42.0f32; self.hidden]) // garbage
                    } else {
                        Some(vec![e as f32 + 1.0; self.hidden]) // the "truth"
                    }
                })
                .collect())
        }
    }

    /// The oracle: the honest ground truth `y_e = e + 1` every honest node returns
    /// (so an honest node agrees exactly, a lying node's 42s disagree).
    struct MockOracle {
        hidden: usize,
    }

    impl Dispatcher for MockOracle {
        fn dispatch(
            &mut self,
            _layer: u16,
            _x: &[f32],
            experts: &[u16],
        ) -> Result<Vec<Option<Vec<f32>>>> {
            Ok(experts
                .iter()
                .map(|&e| Some(vec![e as f32 + 1.0; self.hidden]))
                .collect())
        }
    }

    /// A placement over 3 nodes: expert e -> primary node (e % 3).
    fn three_node_map() -> PlacementMap {
        use crate::placement::{HeatMap, NodeDesc, build_placement};
        let nodes: Vec<NodeDesc> = (0..3)
            .map(|j| NodeDesc {
                id: format!("n{j}"),
                failure_domain: format!("d{j}"),
                uplink_class: 1,
                ram_class: 1,
            })
            .collect();
        let mut heat = HeatMap::new();
        for e in 0..6u16 {
            heat.touch(0, e);
        }
        build_placement(&nodes, &heat, 1).unwrap()
    }

    #[test]
    fn catches_lying_node_and_passes_honest() {
        let hidden = 4;
        let map = three_node_map();
        // Lying node = node 1.
        let inner = MockPool {
            hidden,
            map: map.clone(),
            lying: Some(1),
        };
        let oracle = Box::new(MockOracle { hidden });
        let mut vd =
            VerifyingDispatch::new(inner, oracle, map.clone(), 1000, 7, Tolerance::default());

        let x = vec![0.5f32; hidden];
        let experts: Vec<u16> = (0..6).collect();
        let _ = vd.dispatch(0, &x, &experts).unwrap();

        // Every node holding an expert whose primary is node 1 must show only
        // disagreements; every other checked node only agreements.
        for e in 0..6u16 {
            let primary = map.replicas_of(0, e)[0];
            let c = vd.trust().get(primary);
            assert!(c.checks > 0, "node {primary} was checked");
        }
        assert!(
            vd.disagreements() > 0,
            "the lying node must produce disagreements"
        );
        let distrusted = vd.trust().distrusted(1, 1, 2);
        assert_eq!(distrusted, vec![1], "exactly the lying node is distrusted");
        assert_eq!(vd.trust().get(1).agreements, 0, "the liar never agrees");
    }

    #[test]
    fn transparent_output_and_deterministic_sampling() {
        let hidden = 4;
        let map = three_node_map();
        let x = vec![0.5f32; hidden];
        let experts: Vec<u16> = (0..6).collect();

        // Output equals the bare inner, wrap or not.
        let mut bare = MockPool {
            hidden,
            map: map.clone(),
            lying: None,
        };
        let want = bare.dispatch(0, &x, &experts).unwrap();
        let mut vd = VerifyingDispatch::new(
            MockPool {
                hidden,
                map: map.clone(),
                lying: None,
            },
            Box::new(MockOracle { hidden }),
            map.clone(),
            500,
            42,
            Tolerance::default(),
        );
        let got = vd.dispatch(0, &x, &experts).unwrap();
        assert_eq!(got, want, "wrapping does not change the answers");

        // Same seed + same inputs -> same spot-check set (deterministic).
        let run = || {
            let mut v = VerifyingDispatch::new(
                MockPool {
                    hidden,
                    map: map.clone(),
                    lying: Some(1),
                },
                Box::new(MockOracle { hidden }),
                map.clone(),
                500,
                42,
                Tolerance::default(),
            );
            for _ in 0..3 {
                v.dispatch(0, &x, &experts).unwrap();
            }
            (0..3)
                .map(|n| {
                    let c = v.trust().get(n);
                    (c.checks, c.agreements, c.disagreements)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "the trust tally is deterministic per seed");
        // A partial fraction really samples a subset (not every answer).
        let mut partial = VerifyingDispatch::new(
            MockPool {
                hidden,
                map: map.clone(),
                lying: None,
            },
            Box::new(MockOracle { hidden }),
            map,
            500,
            42,
            Tolerance::default(),
        );
        for _ in 0..4 {
            partial.dispatch(0, &x, &experts).unwrap();
        }
        assert!(
            partial.spot_checks() < 4 * 6,
            "a 500‰ fraction checks fewer than every answer"
        );
    }
}
