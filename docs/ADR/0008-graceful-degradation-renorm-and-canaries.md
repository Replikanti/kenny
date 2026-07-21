# ADR-0008: Graceful degradation by top-k renormalization — with mandatory canaries

- Status: accepted
- Date: 2026-07-21

## Context

On a WAN pool, some expert will be missing, late, or dead on essentially every
step. MoE models tolerate expert dropout with soft quality loss — the router's
top-k weights can be renormalized over whatever subset answered. But a mechanism
that silently tolerates missing experts also silently tolerates dead replicas,
corrupted blobs, and creeping capacity loss.

## Decision

- A missing or late expert never blocks a step: the spine **renormalizes** the
  router weights over the available top-k subset and continues.
- **Corollary, part of this same decision, not an optional feature:** quality
  degradation must be measured from day zero —
  - **perplexity canaries**: fixed prompt sets scored continuously against known
    baselines,
  - **heat-map alarms**: per-expert dispatch/failure rates that surface dead or
    never-answering replicas.
- The day-zero dashboard (MANIFESTO §5) includes the perplexity canary as one of
  its five numbers. A pool without canaries is blind and must not serve.

## Consequences

- No step ever stalls on one node; churn shows up as a smooth quality dip, not
  an outage.
- Renormed outputs differ from the reference model — verification and diff
  tooling must account for which experts actually fired (interacts with
  ADR-0018).
- Canary + heat-map infrastructure is on the critical path of early milestones,
  not a nice-to-have for later.

## Alternatives considered

- **Block / retry until the expert answers** — hands tail latency to the slowest
  node × 75 layers (see ADR-0010); unacceptable.
- **Zero-imputation without renormalization** — silently shrinks FFN output
  magnitude; renorm preserves scale.
- **Failing the affected streams** — turns a soft quality dip into a hard error;
  strictly worse for the target workload (ADR-0006).
