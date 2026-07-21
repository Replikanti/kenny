# ADR-0013: CPU-only nodes are first-class citizens

- Status: accepted
- Date: 2026-07-21

## Context

The pool's workload is network-bound: for a 37 MB fp8 expert, an AVX2 matmul on
one activation costs milliseconds, while the WAN round-trip costs 30+ ms. The
scrap fleet kenny targets — old laptops, minis, NUCs, Pis — is overwhelmingly
CPU-only. A CUDA context alone costs 300–600 MB of RAM: comparable to holding a
dozen more experts.

## Decision

The `kenny node` hot path requires nothing beyond a CPU with portable SIMD
(AVX2 / NEON). GPU on a node is an optional accelerator, never a requirement,
and placement (ADR-0009) never assumes it.

## Consequences

- The fleet is "anything with RAM and a NIC" — maximum pool surface, trivial
  deployment (one static binary, no driver stack).
- Node RAM goes to experts, not runtime overhead (fixed overhead target
  ~10–15 MB, MANIFESTO §3).
- Per-node compute ceilings are irrelevant today (wire binds first) but cap
  future per-node batching if the wire ever stops being the bottleneck.
  Acceptable: that would be a good problem, and GPU nodes remain possible as an
  optimization.

## Alternatives considered

- **GPU-required nodes** — shrinks the pool to approximately nobody and misses
  the point of scrap hardware.
- **Separate CPU/GPU node tiers with distinct protocols** — complexity without a
  driving number; a GPU node is just a fast node.
