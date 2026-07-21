# ADR-0020: Spine compute for M1–M4 — hook an existing runtime, don't build kernels

- Status: proposed
- Date: 2026-07-21

## Context

A real spine needs attention (incl. GLM's DSA variant), KV management, and
sampling kernels — months of work that is worthless if M3's `tc netem` numbers
say the WAN math doesn't close. The only thing M1–M4 actually need from the
spine is: a correct dense forward pass whose MoE FFN calls can be intercepted.

## Decision (leaning)

For M1–M4, the spine is a **spine-sim**: an existing Qwen3-30B forward
implementation — `candle`, or `llama.cpp` via FFI or subprocess — with its MoE
FFN call replaced by dispatch-to-kenny-nodes (renorm and hedging logic included
per ADR-0008 / ADR-0010). Real spine kernels are built only after M3 says WAN
is viable.

The host runtime (candle vs llama.cpp, in-process vs subprocess) gets picked in
the M1 PR with a short justification; this ADR then records it.

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

## Accept when

The M1 PR lands the spine-sim with the chosen runtime named and justified.

## Alternatives considered

- **Build real attention/KV kernels first** — months of pre-payment on an
  unvalidated bet; rejected by the roadmap's whole philosophy (M3 is the gate).
- **Mock spine (random activations, no real model)** — protocol round-trips
  would pass while numerics rot; a real forward gives every milestone a
  perplexity-checkable output (ADR-0008 canaries need this anyway).
