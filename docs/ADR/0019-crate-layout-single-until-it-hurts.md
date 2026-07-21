# ADR-0019: Crate layout — single crate until it hurts

- Status: proposed
- Date: 2026-07-21

## Context

Premature workspace splits create boundary churn while interfaces are still
finding their shape; overgrown single crates eventually slow compiles and blur
dependency discipline (e.g. the node binary should never grow a dependency on
spine-only machinery).

## Decision (leaning)

One crate — `kenny`, lib + bin — through M0 and M1. Expected split point is M2
(first multi-machine deployments), into something like:

- `kenny-core` — formats, carve, fixtures (blob, manifest, safetensors, json)
- `kenny-wire` — codec + transport (ADR-0011, ADR-0016)
- `kenny` — the binary: node / spine / gate / carve entry points

Split when a concrete pain appears — compile times, a dependency that must not
leak into the node build, node binary size — and name that pain in the PR that
performs the split.

## Consequences

- M0/M1 development stays friction-free: one `cargo test`, one binary.
- Module boundaries inside the single crate are drawn as if they were crates
  (formats / wire / roles), so the eventual split is file moves, not surgery.
- Risk of the split arriving late: mitigated by the M2 checkpoint in the
  roadmap explicitly naming this decision.

## Accept when

M2 either performs the split (recording the trigger) or explicitly re-confirms
the single crate with reasons.

## Alternatives considered

- **Workspace from day one** — boundary churn while M0/M1 are still shaping the
  interfaces; cost now for a benefit that only materializes at M2+.
- **Never split** — node deployments (ADR-0013: smallest possible footprint)
  will eventually want a build that cannot even express spine dependencies.
