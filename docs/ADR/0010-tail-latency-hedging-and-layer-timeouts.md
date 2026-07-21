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

## Alternatives considered

- **Wait for all dispatched experts** — tail hell; rejected by arithmetic.
- **Retry after timeout without hedging** — adds a full RTT exactly when the
  step is already late.
- **Erasure coding of activations** — wrong tool: the scarce resource is
  compute-on-time, not data durability; replicas + hedging already provide the
  redundancy.
