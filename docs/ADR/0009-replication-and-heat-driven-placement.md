# ADR-0009: Replication r=2–3; failure-domain- and heat-aware placement

- Status: accepted
- Date: 2026-07-21

## Context

Churn on a hobbyist pool is correlated, not independent: households shut down at
22:00, ISPs drop neighborhoods, one apartment unplugs three nodes at once.
Meanwhile expert routing is strongly Zipfian — a hot head of experts dominates
dispatch traffic — and node uplinks vary by orders of magnitude. Placement is
therefore not storage assignment; it IS the scheduler.

## Decision

- Every expert is replicated **r=2–3**, replicas spread across distinct
  **failure domains**: time zone, ISP, household ("same apartment" is one
  domain).
- Placement is driven by the **heat map** built from the dispatch log:
  - hot experts → fat-uplink nodes, plus the spine's L1 hot-expert cache;
  - the cold Zipf tail → slow-uplink, RAM-rich nodes.
- Placement equalizes **step time, not bytes**: a node's assigned dispatch
  volume is proportional to its uplink. RAM buys coverage, uplink buys
  throughput — a node needs only one of the two currencies to be useful.
- Later (M5+): co-activation clustering — graph partitioning on the expert
  co-occurrence matrix so co-firing experts share nodes and an activation is
  sent once per node touched.

## Consequences

- Correlated churn takes out capacity, not coverage; renorm (ADR-0008) bridges
  the gap while the orchestrator re-replicates.
- The dispatch log and heat map must exist early — they feed placement,
  alarms (ADR-0008), and later metering (ADR-0014).
- Migration machinery (move/copy blobs without disturbing the step loop) is a
  hard M5 deliverable.
- Placement quality is measurable: per-node step p99 (dashboard number 4) is its
  direct output.

## Implementation status

Implemented at M4 (issue #6):

- The placement engine — the heat map, node descriptors, and `build_placement`
  (r replicas across distinct failure domains, hot → fat-uplink, cold →
  RAM-rich, equalize step *time*) — landed in `src/placement.rs`.
- The consuming dispatch path — `PlacedDispatch` in `src/spine.rs`, which fans a
  layer's routed experts across their holders per the `PlacementMap` and
  reassembles per-expert into routed order — and the `kenny spine --node a --node
  b …` multi-node CLI hook landed next, composing on the existing wire (ADR-0024,
  no `WIRE_VERSION` change). The uniform consistent-hash bootstrap ("usable as a
  bootstrap before the first heat map exists") is the CLI default when no heat
  log is supplied; hedging (ADR-0010) is unified onto placement as a replica-set
  second-send; the dispatch/failure heat log feeds the ADR-0008 dead-replica
  alarm (`HeatMap::suspect`, `Dispatcher::suspect_replicas`).
- Still deferred to the M4 real party (#6) and M5: the per-node p99 numbers
  across *heterogeneous shaped* uplinks (the netns/netem sim), migration
  machinery, co-activation clustering, and the L1 hot-expert spine cache.

## Alternatives considered

- **Uniform random placement** — puts hot experts on DSL uplinks; tail latency
  and starvation follow.
- **Consistent hashing only** — elegant for membership, blind to heat and uplink
  asymmetry; usable as a bootstrap before the first heat map exists, not as the
  steady state.
- **r=1** — every churn event is an immediate quality dip until re-replication.
- **r>3** — RAM spent on redundant hot replicas buys less than RAM spent on
  cold-tail coverage.
