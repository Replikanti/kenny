# ADR-0019: Crate layout — single crate until it hurts

- Status: accepted
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

## M2 re-confirmation (accepted)

M2 **re-confirms the single crate** — the split trigger did not fire. Evidence
from the batching work (ADR-0023):

- Batching touched only `src/spine.rs` (batched scheduler + `generate_batch`)
  and `src/cli.rs` (`--batch`). It added ZERO node-side surface — `src/node.rs`
  is untouched — and ZERO new dependencies: the pipeline uses only
  `std::thread::scope` and `std::net::TcpStream::try_clone` from std.
- No compile-time pain and no node-binary-size pressure appeared; the
  formats / wire / spine / node module boundaries already isolate concerns as if
  they were crates (the node build pulls in no spine-only machinery today).
- The anticipated `kenny-core` / `kenny-wire` / `kenny` split therefore remains
  a *future* trigger — fired by a real dependency leak into the node build or by
  measured compile pain — neither of which M2 produced.

## Alternatives considered

- **Workspace from day one** — boundary churn while M0/M1 are still shaping the
  interfaces; cost now for a benefit that only materializes at M2+.
- **Never split** — node deployments (ADR-0013: smallest possible footprint)
  will eventually want a build that cannot even express spine dependencies.
