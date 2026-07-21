# ADR-0004: Stateless experts, stateful spine

- Status: accepted
- Date: 2026-07-21

## Context

An MoE expert is a pure function (MANIFESTO §4.1). All sequence state — KV cache,
sampler state, session bookkeeping — belongs to attention and sampling.
Distributed state is what makes distributed systems hard, and churny scrap nodes
are the worst possible place for it.

## Decision

Nodes hold **only** expert weight blobs and transient dispatch buffers. KV cache,
sessions, router state, sampling — spine only, always.

The invariant, stated once and enforced everywhere: **node death loses capacity,
never state.** Any future feature that would put per-stream or per-session state
on a node must supersede this ADR first.

## Consequences

- Churn is a capacity event, not a correctness event; recovery = re-replicating
  blobs, not reconciling state machines.
- Replication is trivial: blobs are immutable content-addressed files
  (ADR-0005), so r=2–3 (ADR-0009) is a copy, not a consensus protocol.
- Nodes are interchangeable cache entries; the orchestrator can move experts
  freely.
- The spine pays for it: the entire KV wall (MANIFESTO §5, failure mode 2) lands
  on one machine, and spine death means every stream re-prefills. Accepted — one
  well-understood SPOF beats distributed session state on scrap.

## Alternatives considered

- **KV sharded across nodes** — session state on churny untrusted hardware:
  correctness under churn, trust exposure (KV leaks more than activations), and
  per-step wire cost for KV reads. Rejected outright.
- **Expert-side micro-state (adapters, LoRA deltas, caches)** — breaks node
  interchangeability and the cache model. If ever needed, it arrives as a new
  ADR superseding this one.
