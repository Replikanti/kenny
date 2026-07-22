# ADR-0010: Tail latency — hedged dispatch and per-layer timeouts

- Status: accepted
- Date: 2026-07-21

## Context

A step synchronizes on every MoE layer: 75 barriers per token for GLM-class
models (MANIFESTO §4.4). Without countermeasures, the p99 of a single Wi-Fi node
becomes the p50 of the step — multiplied by 75. Tail latency, not bandwidth, is
what turns a WAN pool from slow into unusable.

## Decision

- **Hedging**: dispatches for an expert may be sent to more than one replica —
  immediately for hot experts, or after a short delay when the primary is late.
  First answer wins; late answers are discarded. Experts are pure functions
  (ADR-0004), so duplicate execution is always safe.
- **Per-layer timeout**: each layer has a time budget. When it expires, the
  spine renormalizes over the experts that answered (ADR-0008) and moves on —
  stragglers are dropped from the step, not waited for.
- Straggler statistics feed the heat map (ADR-0009): chronically slow nodes get
  demoted to the cold tail by placement, closing the loop.

## Consequences

- Step time becomes a controlled percentile of node response times instead of
  their maximum.
- Hedging spends duplicate bytes and compute; the hedge rate is a tunable
  budget, measured at M3 (`tc netem`) before any WAN deployment.
- The timeout budget is a quality/latency knob: too tight renorms too often
  (quality dip per ADR-0008), too loose readmits the tail. It ships as a
  measured default, not a guess.
- **Measured at M3** (`tc netem`, BENCH "M3"): at 1 % loss the per-layer timeout
  (90 ms ≈ 3·RTT) caps step p99 with a timeout rate ≤ 5.2 % of layer-steps, and
  the 2-node hedge fires on 3.56 % of layer-steps to collapse p99 4.89 s → 3.57 s
  (−27 %) with **renorm rate 0** — the redundant secondary rescues every stall, so
  the hedged path is quality-safe. This closes the "hedge rate + timeout budget
  measured at M3 before any WAN deployment" promise above; the single-node timeout
  alone gives a coarser cap (degenerate whole-layer renorm), so the hedge is what
  makes the tail control quality-safe.

## Alternatives considered

- **Wait for all dispatched experts** — tail hell; rejected by arithmetic.
- **Retry after timeout without hedging** — adds a full RTT exactly when the
  step is already late.
- **Erasure coding of activations** — wrong tool: the scarce resource is
  compute-on-time, not data durability; replicas + hedging already provide the
  redundancy.
