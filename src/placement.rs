//! Heat-driven, failure-domain-aware expert placement (ADR-0009).
//!
//! Placement is the scheduler, not storage assignment: every routed expert is
//! replicated `r = 2–3` across DISTINCT failure domains, hot experts land on
//! fat-uplink nodes and the cold Zipf tail on RAM-rich slow nodes, and the
//! objective is to equalize *step time* — a node's assigned dispatch volume is
//! made proportional to its uplink (ADR-0009: "RAM buys coverage, uplink buys
//! throughput — a node needs only one of the two currencies to be useful").
//!
//! This module is the pure engine: it turns a set of node descriptors plus the
//! dispatch [`HeatMap`] into a `(layer, expert) -> replica set` [`PlacementMap`].
//! It is spine-LOCAL — never on the wire, never in a manifest, never cross-node —
//! so losing or recomputing it costs only re-placement, never correctness
//! (ADR-0004 trust model). `PlacedDispatch` (`src/spine.rs`, a later M4 PR)
//! consumes the map; the fan-out composes on the existing wire with no frame or
//! `WIRE_VERSION` change (ADR-0024), exactly as batching did (ADR-0023).

use std::collections::BTreeMap;

use crate::{Error, Result};

/// A routed expert, identified by `(layer, expert)` — the same key the node
/// index and the wire use (`src/node.rs`, `src/wire.rs`).
pub type ExpertKey = (u16, u16);

/// Per-`(layer, expert)` dispatch tallies, read straight off the dispatch log.
/// `dispatches` counts every request; `failures` is the subset that came back
/// unanswered (timeout / all replicas dead) — so `failures <= dispatches`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    pub dispatches: u64,
    pub failures: u64,
}

/// The dispatch heat map (ADR-0009): `(layer, expert) -> counts`, ordered so
/// placement is deterministic. It feeds two consumers — [`build_placement`]
/// (hot vs cold steer the assignment) and the ADR-0008 dead-replica alarm
/// ([`HeatMap::suspect`]). Spine-local; never serialized onto the wire.
#[derive(Debug, Default, Clone)]
pub struct HeatMap {
    counts: BTreeMap<ExpertKey, Counts>,
}

impl HeatMap {
    pub fn new() -> HeatMap {
        HeatMap::default()
    }

    /// Record one dispatch of `(layer, expert)` (whether or not it succeeds).
    pub fn record_dispatch(&mut self, layer: u16, expert: u16) {
        self.counts.entry((layer, expert)).or_default().dispatches += 1;
    }

    /// Record that a dispatch of `(layer, expert)` came back unanswered. Also
    /// counts as a dispatch so a never-dispatched expert never looks "failing".
    pub fn record_failure(&mut self, layer: u16, expert: u16) {
        let c = self.counts.entry((layer, expert)).or_default();
        c.dispatches += 1;
        c.failures += 1;
    }

    /// Register an expert with zero counts so it enters the placement universe
    /// before it is ever dispatched — the ADR-0009 bootstrap seed (place the
    /// whole catalog cold, then let heat re-steer it as the log accumulates).
    pub fn touch(&mut self, layer: u16, expert: u16) {
        self.counts.entry((layer, expert)).or_default();
    }

    /// Counts for one expert (zero if unseen).
    pub fn get(&self, layer: u16, expert: u16) -> Counts {
        self.counts
            .get(&(layer, expert))
            .copied()
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Ordered iteration over `(key, counts)`.
    pub fn iter(&self) -> impl Iterator<Item = (ExpertKey, Counts)> + '_ {
        self.counts.iter().map(|(&k, &c)| (k, c))
    }

    /// The ADR-0008 dead/never-answering-replica alarm feed: experts dispatched
    /// at least `min_samples` times whose failure fraction is at least
    /// `num / den`. A fully dead replica set answers every dispatch with a
    /// failure, so it surfaces here for re-replication (ADR-0009). Integer
    /// comparison (`failures × den >= dispatches × num`) keeps it exact.
    pub fn suspect(&self, min_samples: u64, num: u64, den: u64) -> Vec<ExpertKey> {
        self.counts
            .iter()
            .filter(|(_, c)| {
                c.dispatches >= min_samples
                    && (c.failures as u128) * (den as u128)
                        >= (c.dispatches as u128) * (num as u128)
            })
            .map(|(&k, _)| k)
            .collect()
    }
}

/// A pool node as the placer sees it. `failure_domain` is the correlated-churn
/// unit — household / ISP / time zone (ADR-0009); two nodes in the same domain
/// can die together, so an expert's replicas must never share one.
/// `uplink_class` and `ram_class` are RELATIVE capacity weights in any single
/// consistent unit: uplink buys throughput, RAM buys coverage.
#[derive(Debug, Clone)]
pub struct NodeDesc {
    /// Stable node identity (used both for reporting and as the tie-break hash
    /// seed that spreads the cold tail across equal-RAM nodes).
    pub id: String,
    /// Correlated-failure grouping. Replicas of one expert must be in distinct
    /// domains.
    pub failure_domain: String,
    /// Relative uplink capacity. Assigned dispatch volume is made proportional
    /// to this, so step time equalizes across heterogeneous links. Must be > 0.
    pub uplink_class: u32,
    /// Relative RAM capacity. Higher RAM is preferred for the cold Zipf tail —
    /// coverage the fat-uplink nodes should not spend their throughput on.
    /// Must be > 0.
    pub ram_class: u32,
}

/// `(layer, expert) -> replica set` (node indices into the `nodes` slice given
/// to [`build_placement`], sorted ascending). Spine-local; consumed by
/// `PlacedDispatch` and by the per-node `--hold` subset ([`Self::subset_for`]).
#[derive(Debug, Default, Clone)]
pub struct PlacementMap {
    replicas: BTreeMap<ExpertKey, Vec<usize>>,
}

impl PlacementMap {
    /// The replica set (sorted node indices) for one expert, or empty if the
    /// expert is unplaced — the pre-existing not-held → renorm path (ADR-0008).
    pub fn replicas_of(&self, layer: u16, expert: u16) -> &[usize] {
        self.replicas
            .get(&(layer, expert))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Whether `node` holds `(layer, expert)` under this placement.
    pub fn holds(&self, node: usize, layer: u16, expert: u16) -> bool {
        self.replicas_of(layer, expert).contains(&node)
    }

    /// Ordered iteration over `(key, replica set)`.
    pub fn iter(&self) -> impl Iterator<Item = (ExpertKey, &[usize])> + '_ {
        self.replicas.iter().map(|(&k, v)| (k, v.as_slice()))
    }

    /// Every expert assigned to `node` — the node's `--hold` subset (a later
    /// M4 PR wires this into `kenny node --hold`).
    pub fn subset_for(&self, node: usize) -> Vec<ExpertKey> {
        self.replicas
            .iter()
            .filter(|(_, v)| v.contains(&node))
            .map(|(&k, _)| k)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.replicas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.replicas.is_empty()
    }
}

/// A `(layer, expert)` hash seeded by a node id — rendezvous-style tie-break
/// that spreads equally-preferred experts (the uniform-heat bootstrap, and the
/// cold tail across equal-RAM nodes) deterministically instead of collapsing
/// them onto the lowest-indexed node.
fn tiebreak_hash(id: &str, layer: u16, expert: u16) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(id.as_bytes());
    h.update(&layer.to_le_bytes());
    h.update(&expert.to_le_bytes());
    let digest = h.finalize();
    let mut b = [0u8; 8];
    b.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(b)
}

/// Strict "is node `a` a better replica than node `b` for this expert" order,
/// given the running per-node assigned load and this expert's dispatch weight
/// `w`. Three levels, most significant first:
///  1. **Step-time cost** `(load + w) / uplink` — the ADR-0009 objective.
///     Compared as the exact integer cross-product `(load+w)·uplink_other` to
///     avoid float non-determinism. Smaller wins → hot experts flow to fat
///     uplinks and load equalizes in *time*, not bytes.
///  2. **RAM** — higher `ram_class` wins. For the cold tail every node ties at
///     level 1 (weight ~0, zero load), so coverage lands on RAM-rich nodes.
///  3. **Rendezvous hash then index** — spreads the remaining ties and makes the
///     result fully deterministic.
fn better(nodes: &[NodeDesc], loads: &[u64], a: usize, b: usize, w: u64, key: ExpertKey) -> bool {
    let (na, nb) = (&nodes[a], &nodes[b]);
    let ca = (loads[a] as u128 + w as u128) * nb.uplink_class as u128;
    let cb = (loads[b] as u128 + w as u128) * na.uplink_class as u128;
    if ca != cb {
        return ca < cb;
    }
    if na.ram_class != nb.ram_class {
        return na.ram_class > nb.ram_class;
    }
    let (ha, hb) = (
        tiebreak_hash(&na.id, key.0, key.1),
        tiebreak_hash(&nb.id, key.0, key.1),
    );
    if ha != hb {
        return ha > hb;
    }
    a < b
}

/// Build the replica placement for every expert in `heat` over `nodes`, aiming
/// for `r` replicas each (ADR-0009).
///
/// Replicas of an expert always land in DISTINCT failure domains (the hard
/// invariant), so the achieved replica count clamps to the number of distinct
/// domains present (itself ≤ `nodes.len()`); coverage below `r` is the honest
/// signal that the pool lacks failure-domain diversity, not an error. Experts
/// are placed hot-first so the Zipf head claims the fat uplinks before the cold
/// tail fills the RAM-rich nodes.
///
/// Errors ([`Error::usage`]) only on unusable inputs: no nodes, `r == 0`, or a
/// node with a zero uplink/RAM class.
pub fn build_placement(nodes: &[NodeDesc], heat: &HeatMap, r: usize) -> Result<PlacementMap> {
    if nodes.is_empty() {
        return Err(Error::usage("placement: needs at least one node"));
    }
    if r == 0 {
        return Err(Error::usage("placement: replica count r must be >= 1"));
    }
    for n in nodes {
        if n.uplink_class == 0 || n.ram_class == 0 {
            return Err(Error::usage(format!(
                "placement: node {} has a zero uplink_class or ram_class",
                n.id
            )));
        }
    }

    // Distinct failure domains cap the achievable replica count.
    let mut domains: Vec<&str> = nodes.iter().map(|n| n.failure_domain.as_str()).collect();
    domains.sort_unstable();
    domains.dedup();
    let rr_max = r.min(domains.len());

    // Order experts hot-first (dispatches desc), key asc to break heat ties.
    let mut order: Vec<(ExpertKey, u64)> = heat.iter().map(|(k, c)| (k, c.dispatches)).collect();
    order.sort_by(|(ka, wa), (kb, wb)| wb.cmp(wa).then(ka.cmp(kb)));

    let mut loads = vec![0u64; nodes.len()];
    let mut replicas: BTreeMap<ExpertKey, Vec<usize>> = BTreeMap::new();

    for (key, w) in order {
        let mut chosen: Vec<usize> = Vec::with_capacity(rr_max);
        let mut used_domains: Vec<&str> = Vec::with_capacity(rr_max);
        for _ in 0..rr_max {
            let mut best: Option<usize> = None;
            for i in 0..nodes.len() {
                if chosen.contains(&i) {
                    continue;
                }
                let dom = nodes[i].failure_domain.as_str();
                if used_domains.contains(&dom) {
                    continue;
                }
                best = Some(match best {
                    None => i,
                    Some(b) if better(nodes, &loads, i, b, w, key) => i,
                    Some(b) => b,
                });
            }
            match best {
                Some(i) => {
                    loads[i] += w;
                    chosen.push(i);
                    used_domains.push(nodes[i].failure_domain.as_str());
                }
                // No further distinct domain available — coverage clamps here.
                None => break,
            }
        }
        chosen.sort_unstable();
        replicas.insert(key, chosen);
    }

    Ok(PlacementMap { replicas })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, domain: &str, uplink: u32, ram: u32) -> NodeDesc {
        NodeDesc {
            id: id.into(),
            failure_domain: domain.into(),
            uplink_class: uplink,
            ram_class: ram,
        }
    }

    fn hot(key: ExpertKey, n: u64) -> HeatMap {
        let mut h = HeatMap::new();
        for _ in 0..n {
            h.record_dispatch(key.0, key.1);
        }
        h
    }

    #[test]
    fn replicas_land_in_distinct_failure_domains() {
        // Two nodes share domain "A"; r=3 across a 3-domain pool must never put
        // both A-nodes in one replica set.
        let nodes = [
            node("a0", "A", 10, 10),
            node("a1", "A", 10, 10),
            node("b", "B", 10, 10),
            node("c", "C", 10, 10),
        ];
        let heat = hot((5, 2), 100);
        let map = build_placement(&nodes, &heat, 3).unwrap();
        let reps = map.replicas_of(5, 2);
        assert_eq!(reps.len(), 3, "3 distinct domains available");
        let mut doms: Vec<&str> = reps
            .iter()
            .map(|&i| nodes[i].failure_domain.as_str())
            .collect();
        doms.sort_unstable();
        doms.dedup();
        assert_eq!(doms.len(), 3, "every replica in a distinct failure domain");
    }

    #[test]
    fn hot_expert_goes_to_fat_uplink() {
        // Node 0 has 10× the uplink; a hot expert's primary replica must be it.
        let nodes = [
            node("fat", "A", 100, 10),
            node("slow1", "B", 10, 10),
            node("slow2", "C", 10, 10),
        ];
        let heat = hot((0, 0), 1000);
        let one = build_placement(&nodes, &heat, 1).unwrap();
        assert_eq!(
            one.replicas_of(0, 0),
            &[0],
            "single replica lands on fat uplink"
        );
        let two = build_placement(&nodes, &heat, 2).unwrap();
        assert!(
            two.replicas_of(0, 0).contains(&0),
            "fat uplink stays in the set"
        );
    }

    #[test]
    fn cold_expert_goes_to_ram_rich() {
        // Place a hot expert first (loads the fat node), then a stone-cold one:
        // it must avoid the busy fat node and pick the RAM-rich slow node.
        let nodes = [
            node("fat", "A", 100, 4), // fat uplink, little RAM
            node("ram", "B", 10, 64), // slow uplink, lots of RAM
            node("thin", "C", 10, 4), // slow uplink, little RAM
        ];
        let mut heat = hot((0, 0), 1000);
        heat.touch(0, 9); // cold expert, zero dispatches
        let map = build_placement(&nodes, &heat, 1).unwrap();
        assert!(map.replicas_of(0, 0).contains(&0), "hot -> fat uplink");
        assert_eq!(map.replicas_of(0, 9), &[1], "cold -> RAM-rich slow node");
    }

    #[test]
    fn replica_count_clamps_to_distinct_domains() {
        // Three nodes, only two domains: r=3 can only reach 2 replicas.
        let nodes = [
            node("a", "A", 10, 10),
            node("b0", "B", 10, 10),
            node("b1", "B", 10, 10),
        ];
        let heat = hot((1, 1), 50);
        let map = build_placement(&nodes, &heat, 3).unwrap();
        assert_eq!(map.replicas_of(1, 1).len(), 2, "clamped to the 2 domains");
    }

    #[test]
    fn replica_count_clamps_to_node_count() {
        let nodes = [node("a", "A", 10, 10), node("b", "B", 10, 10)];
        let heat = hot((2, 3), 7);
        let map = build_placement(&nodes, &heat, 3).unwrap();
        assert_eq!(map.replicas_of(2, 3).len(), 2, "only two nodes exist");
    }

    #[test]
    fn load_equalizes_in_time_across_uplinks() {
        // 4.5× uplink asymmetry, five equal-weight experts, r=1: the fat node
        // should carry ~4.5× the volume — here 4 experts vs 1 — because the
        // objective equalizes load/uplink, not raw expert count.
        let nodes = [node("fat", "A", 45, 10), node("slow", "B", 10, 10)];
        let mut heat = HeatMap::new();
        for e in 0..5u16 {
            heat.record_dispatch(0, e); // each weight 1
        }
        let map = build_placement(&nodes, &heat, 1).unwrap();
        let mut on_fat = 0;
        let mut on_slow = 0;
        for e in 0..5u16 {
            match map.replicas_of(0, e) {
                [0] => on_fat += 1,
                [1] => on_slow += 1,
                other => panic!("unexpected replica set {other:?}"),
            }
        }
        assert_eq!((on_fat, on_slow), (4, 1), "volume ~proportional to uplink");
    }

    #[test]
    fn uniform_bootstrap_spreads_across_the_pool() {
        // All-cold catalog (touch only) over an otherwise-identical pool: the
        // rendezvous tie-break must spread experts, not stack them on node 0.
        let nodes = [
            node("n0", "A", 10, 10),
            node("n1", "B", 10, 10),
            node("n2", "C", 10, 10),
        ];
        let mut heat = HeatMap::new();
        for e in 0..30u16 {
            heat.touch(7, e);
        }
        let map = build_placement(&nodes, &heat, 1).unwrap();
        let mut used = [false; 3];
        for e in 0..30u16 {
            let reps = map.replicas_of(7, e);
            assert_eq!(reps.len(), 1);
            used[reps[0]] = true;
        }
        assert!(used.iter().all(|&u| u), "bootstrap touches every node");
    }

    #[test]
    fn placement_is_deterministic() {
        let nodes = [
            node("fat", "A", 80, 8),
            node("ram", "B", 10, 64),
            node("mid", "C", 30, 16),
        ];
        let mut heat = hot((3, 1), 500);
        heat.record_dispatch(3, 2);
        heat.touch(3, 3);
        let a = build_placement(&nodes, &heat, 2).unwrap();
        let b = build_placement(&nodes, &heat, 2).unwrap();
        for key in [(3u16, 1u16), (3, 2), (3, 3)] {
            assert_eq!(a.replicas_of(key.0, key.1), b.replicas_of(key.0, key.1));
        }
    }

    #[test]
    fn subset_for_and_holds_agree() {
        let nodes = [
            node("a", "A", 10, 10),
            node("b", "B", 10, 10),
            node("c", "C", 10, 10),
        ];
        let heat = hot((0, 0), 5);
        let map = build_placement(&nodes, &heat, 2).unwrap();
        for n in 0..nodes.len() {
            for key in map.subset_for(n) {
                assert!(map.holds(n, key.0, key.1));
            }
        }
        // The union of every node's subset is the full replica set.
        let total: usize = (0..nodes.len()).map(|n| map.subset_for(n).len()).sum();
        assert_eq!(total, map.replicas_of(0, 0).len());
    }

    #[test]
    fn heatmap_records_and_suspect_flags_dead_experts() {
        let mut heat = HeatMap::new();
        for _ in 0..10 {
            heat.record_dispatch(0, 0); // healthy: 0 failures
        }
        for _ in 0..10 {
            heat.record_failure(0, 1); // dead: 10/10 fail
        }
        heat.record_failure(0, 2); // 1 sample only
        assert_eq!(
            heat.get(0, 0),
            Counts {
                dispatches: 10,
                failures: 0
            }
        );
        assert_eq!(
            heat.get(0, 1),
            Counts {
                dispatches: 10,
                failures: 10
            }
        );
        // >= 50% failures over >= 5 samples: only the fully-dead (0,1) qualifies.
        let sus = heat.suspect(5, 1, 2);
        assert_eq!(sus, vec![(0, 1)]);
    }

    #[test]
    fn invalid_inputs_error() {
        let nodes = [node("a", "A", 10, 10)];
        let heat = hot((0, 0), 1);
        assert!(build_placement(&[], &heat, 2).is_err(), "no nodes");
        assert!(build_placement(&nodes, &heat, 0).is_err(), "r == 0");
        let zero = [node("a", "A", 0, 10)];
        assert!(build_placement(&zero, &heat, 1).is_err(), "zero uplink");
        let zeror = [node("a", "A", 10, 0)];
        assert!(build_placement(&zeror, &heat, 1).is_err(), "zero ram");
    }

    #[test]
    fn empty_heat_is_an_empty_map() {
        let nodes = [node("a", "A", 10, 10)];
        let map = build_placement(&nodes, &HeatMap::new(), 2).unwrap();
        assert!(map.is_empty());
    }
}
