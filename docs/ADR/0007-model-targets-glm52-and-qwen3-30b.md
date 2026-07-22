# ADR-0007: Model targets — GLM-5.2 (production), Qwen3-30B-A3B (dev testbed)

- Status: accepted
- Date: 2026-07-21

## Context

The system needs one frontier model to size every budget against (wire, KV,
spine RAM, blob counts) and one small model of the same structural family that
fits a dev machine for cheap end-to-end validation with reference outputs.

## Decision

- **GLM-5.2** is the production target (MIT license; ~743B params, 75 MoE layers
  × 256 experts, expert 75.5 MB bf16 / 37.7 MB fp8; full card in MANIFESTO
  §4.2). All capacity math, wire budgets, and dashboard thresholds are sized
  against it.
- **Qwen3-30B-A3B** is the dev testbed (Apache 2.0; 48 layers × 128 experts,
  expert 9.4 MB bf16; same `model.layers.{L}.mlp.experts.{E}.*` tensor-naming
  family, no shared expert). Everything is built and validated on Qwen3 first;
  milestones M0–M4 run on it.

## Consequences

- One carve path serves both (same naming schema family) — but "trust but
  verify": carve always starts by dumping tensor names, never assuming the
  schema.
- The dev loop is cheap: 6,144 blobs, reference forwards run locally.
- GLM-specific machinery absent from Qwen3 — shared expert (fires every token →
  spine), DSA sparse attention, MTP layer — must be carried in the design even
  though dev runs never exercise it. Checklist lives with the spine milestones.
  The shared expert's placement × renorm × failure semantics are decided in
  ADR-0025 (proposed, numerics confirmed at M5.C); DSA and MTP remain untested
  scaffolding until the real card and get their own treatment.
- Model releases move; MANIFESTO §4.2 numbers are re-verified against the actual
  checkpoint before each milestone that consumes them.

## Alternatives considered

- **Single model only** — either the dev loop costs a workstation-killing
  download per iteration (GLM) or the production math is sized against a toy
  (Qwen3 alone).
- **Synthetic models only for dev** — no reference implementation to diff
  against; synthetic fixtures are for CI, not for validation of real numerics.
