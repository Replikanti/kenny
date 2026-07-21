# ADR-0003: Star topology; four roles; orchestrator never in the data path

- Status: accepted
- Date: 2026-07-21

## Context

Expert-level sharding (ADR-0002) needs a topology and a division of labor.
Activations must flow token-synchronously with as few hops as possible; control
traffic (health, placement, migration) must never add latency to the token loop.

## Decision

- Topology is a **star**: activations flow spine ⇄ node directly, one hop each
  way per dispatch.
- Four roles, one binary:
  - `kenny carve` — offline: model → content-addressed expert blobs + manifest
  - `kenny node` — expert runner: holds N blobs, answers dispatches
  - `kenny spine` — data plane: attention, KV, router, batching, hot cache,
    sampling
  - `kenny gate` — OpenAI-compatible HTTP API in front of the spine
- The **orchestrator** (placement, health, migration) is a control plane beside
  the spine — logically separate, low-bandwidth, out of band. It must NEVER sit
  in the data path.
- Trust model: the spine sees plaintext prompts; the pool sees only anonymous
  activation vectors. Whoever owns the workload hosts the spine. Multi-spine over
  a shared pool is the designed future — experts are a stateless cache anyone may
  query; each spine is SPOF and trust anchor for its own streams only.

## Consequences

- Latency floor is 1 RTT per MoE layer — the best any topology can do for
  synchronous routed dispatch.
- The spine is a single point of failure and the bandwidth chokepoint (link math
  in MANIFESTO §4.4). Accepted: state already lives there (ADR-0004), and the
  failure blast radius is one party's streams, not the pool.
- Failure analysis stays boring: two link types (spine⇄node data, orchestrator⇄
  node control), one hop each.

## Alternatives considered

- **Mesh / p2p routing** — multi-hop latency and routing complexity, no benefit:
  traffic is inherently star-shaped (all activations originate and terminate at
  attention).
- **Per-layer node chains** — adds hops and couples node failures across layers.
- **Orchestrator in the data path** — control decisions would add latency to
  every step and its failure would stop the world; out of band it can crash
  harmlessly.
