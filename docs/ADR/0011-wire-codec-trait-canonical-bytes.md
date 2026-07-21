# ADR-0011: Wire codec as a versioned trait; wire bytes are canonical

- Status: accepted
- Date: 2026-07-21

## Context

Wire cost is the binding constraint (MANIFESTO §4.3): 7.2 MB/token naive fp8 for
GLM-class, with a realistic floor around 2.3 MB/token after the known levers
(hot-expert cache, Hadamard + int4 + stochastic rounding, co-activation
placement, ANS entropy coding). The codec will therefore evolve through several
generations. At the same time, future verification (ADR-0015) wants to hash and
compare the exact bytes that crossed the wire.

## Decision

- The activation wire format lives behind a **`WireCodec` trait from day one**;
  fp8 E4M3 is the baseline implementation.
- **Wire bytes are canonical.** The codec (and its version) is part of the
  protocol consensus: for a given codec version, the bytes for a given
  activation are defined exactly. A node may never choose its own encoding,
  compression level, or "equivalent" representation.
- Codec version is carried in the protocol handshake and recorded alongside the
  manifest's codec metadata (ADR-0005).

## Consequences

- Codec upgrades (int4+SR, ANS) are coordinated version bumps, not per-link
  negotiations — deliberately boring rollouts.
- Verification can hash dispatch/gather bytes and compare across replicas
  without canonicalization gymnastics.
- No per-node opportunistic optimization (e.g. a fast node offering richer
  precision). Accepted cost: uniformity is what makes the pool auditable.
- The trait boundary keeps M3's fp8-vs-int8 benchmark (ADR-0018) a swap, not a
  rewrite.

## Alternatives considered

- **Negotiated per-link codecs** — breaks byte-level verification and doubles
  the test matrix per codec pair.
- **Single hard-coded codec** — blocks the 2–3× levers the wire math depends on.
- **Compression as a transparent transport feature** — hides bytes from the
  consensus layer; the codec must own the canonical form end to end.
