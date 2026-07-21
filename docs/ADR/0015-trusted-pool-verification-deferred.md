# ADR-0015: Trusted pool now; spot-check verification designed-for, deferred

- Status: accepted
- Date: 2026-07-21

## Context

An open pool invites wrong answers: buggy nodes, bit-rotted blobs, lazy or
malicious peers returning garbage. Renormalization (ADR-0008) masks failures by
design, which makes undetected wrong results worse than missing ones. Full
verification — recomputing everything — defeats the purpose of distribution,
and verifiable-inference research is nowhere near practical for this workload.

## Decision

- **Now**: the pool runs among **trusted parties only** — your own machines,
  friends, known operators. No adversarial-safety claims are made or implied.
- **Designed-for, implemented later (M5+)**: the spine spot-checks a random
  sample of dispatches by recomputing them against a **rotating locally-held
  expert set**, weighting nodes by accumulated trust (and later, optionally,
  stake). Canonical bits (ADR-0005, ADR-0011, ADR-0012) exist partly so these
  checks can compare bytes; whether comparison is exact or tolerance-based
  follows the numeric path decision (ADR-0018).

## Consequences

- WAN milestones ship without solving byzantine compute first.
- Verification is a retrofit the architecture already accommodates: canonical
  blobs, canonical wire bytes, dispatch log with per-node attribution.
- Until spot-checks land, pool membership is a social boundary, and the docs
  must say so plainly.
- Perplexity canaries (ADR-0008) remain the only integrity signal in the
  trusted phase — another reason they are mandatory.

## Alternatives considered

- **ZK / cryptographic verifiable inference** — orders of magnitude too
  expensive for MB-scale matmuls at step cadence, for the foreseeable future.
- **Full redundant recompute (send every dispatch to 2 nodes and compare)** —
  halves effective pool throughput to catch a rare event; spot-checking buys
  nearly the same deterrence for a fraction of the cost.
- **Reputation-only without recompute** — gameable; trust must be anchored in
  occasionally-verified ground truth.
