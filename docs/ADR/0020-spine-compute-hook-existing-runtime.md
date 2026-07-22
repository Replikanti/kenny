# ADR-0020: Spine compute for M1–M4 — hook an existing runtime, don't build kernels

- Status: accepted
- Date: 2026-07-21

## Context

A real spine needs attention (incl. GLM's DSA variant), KV management, and
sampling kernels — months of work that is worthless if M3's `tc netem` numbers
say the WAN math doesn't close. The only thing M1–M4 actually need from the
spine is: a correct dense forward pass whose MoE FFN calls can be intercepted.

## Decision

For M1–M4 the spine is a **spine-sim**: a dense Qwen3-30B-A3B forward whose MoE
FFN call is replaced by dispatch-to-kenny-nodes (renorm per ADR-0008; hedging,
ADR-0010, follows later). Real spine kernels are built only after M3 says WAN is
viable.

The runtime, chosen in the M1 PR and recorded here, is a **pure-Rust, in-repo
forward** — neither `candle` nor `llama.cpp`. Both named candidates are
dispositively blocked by the dependency policy (ADR-0021, enforced by
`deny.toml`): candle's tree pulls `serde`, `serde_json`, `rand`, and `thiserror`
— four banned crates — so adopting it means lifting four bans and pulling a
framework into the consensus-critical dependency tree; llama.cpp (FFI or
subprocess) cannot expose the per-layer FFN boundary + router logits this ADR
names as the selection criterion without forking/patching C++ and adding a C
toolchain to a CI that is pure-Rust and never downloads a model. The carve
already records **every** non-expert tensor by absolute byte range (`SpineEntry`,
ADR-0005) and the M0 expert kernel (`expert::forward`) is exact, so a faithful
dense forward is a small, deterministic reinvention in kenny's own house style
(ADR-0021) that owns numeric determinism in-repo (ADR-0018). It lives in
`src/spine.rs`: RMSNorm / RoPE / GQA attention + KV cache / Qwen3 router
(softmax-over-all → top-k → renorm) / greedy sampling, with the MoE FFN behind a
`Dispatcher` seam (`LocalDispatch` in-process, `NodeDispatch` over TCP).

## Consequences

- M1 arrives in days-not-months, and every M1–M3 measurement exercises the real
  protocol against a numerically correct model.
- The dispatch seam (replace-the-FFN-call) is exactly the interface the real
  spine will implement later — the spine-sim is a reference consumer, not
  throwaway.
- Constraint: the chosen runtime must expose (or be patchable to expose) the
  per-layer FFN boundary and router logits. This is the main selection
  criterion.
- GLM-specific machinery (shared expert, DSA, MTP) stays out of scope until
  post-M3 by design (ADR-0007 consequences).

## Accepted

Met by the M1 PR (issue #3): `src/spine.rs` lands the pure-Rust spine-sim with
the runtime named and justified above, and the S7 two-process run against the
real Qwen3-30B-A3B closes the milestone — the dispatched fp8 path reproduces the
in-process path bit-for-bit, with the output-sanity cosine and wire bytes in
`BENCH.md`.

## Alternatives considered

- **Build real attention/KV kernels first** — months of pre-payment on an
  unvalidated bet; rejected by the roadmap's whole philosophy (M3 is the gate).
- **Mock spine (random activations, no real model)** — protocol round-trips
  would pass while numerics rot; a real forward gives every milestone a
  perplexity-checkable output (ADR-0008 canaries need this anyway).
- **candle in-process** — dispositively blocked by `deny.toml` (ADR-0021):
  pulls `serde`/`serde_json`/`rand`/`thiserror`, four banned crates.
- **llama.cpp (FFI or subprocess)** — cannot expose the per-layer FFN boundary +
  router logits without forking C++ and adding a C toolchain to a pure-Rust CI.
