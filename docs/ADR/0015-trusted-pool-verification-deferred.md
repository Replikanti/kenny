# ADR-0015: Trusted pool now; spot-check verification designed-for, deferred

- Status: accepted
- Date: 2026-07-21

## Context

An open pool invites wrong answers: buggy nodes, bit-rotted blobs, lazy or
malicious peers returning garbage. Renormalization (ADR-0008) masks failures by
design, which makes undetected wrong results worse than missing ones. Full
verification — recomputing everything — defeats the purpose of distribution,
and verifiable-inference research is nowhere near practical for this workload.

## Decision

- **Now**: the pool runs among **trusted parties only** — your own machines,
  friends, known operators. No adversarial-safety claims are made or implied.
- **Designed-for, implemented later (M5+)**: the spine spot-checks a random
  sample of dispatches by recomputing them against a **rotating locally-held
  expert set**, weighting nodes by accumulated trust (and later, optionally,
  stake). Canonical bits (ADR-0005, ADR-0011, ADR-0012) exist partly so these
  checks can compare bytes; whether comparison is exact or tolerance-based
  follows the numeric path decision (ADR-0018).

### M5.B update (2026-07-22) — spot-checks implemented, tolerance-based

The "designed-for, implemented later" half of the Decision now EXISTS in code
(`src/verify.rs`, `kenny spine --verify-frac`): a [`VerifyingDispatch`] decorator
wraps the placed dispatcher, samples a `‰` fraction of answered dispatches, recomputes
each sampled `(layer, expert)` from the bf16 SOURCE via the canary's `SourceRefDispatch`
oracle (the `diff.rs::source_matrix` reference), and accumulates a spine-local per-node
trust tally (agree/disagree). A distrust verdict flags any node past a disagreement
threshold — the byzantine (wrong-answer) analogue of the ADR-0008 dead-replica (no-answer)
alarm. Measured on the real Qwen3-30B-A3B (BENCH "M5.B"): the honest fp8 path stays inside
the envelope (zero false distrust) and a garbage node is caught.

Two honest scopings hold this to what is real now:

- **Comparison is TOLERANCE-BASED, not byte-exact.** ADR-0018 is still `proposed`
  (only the fp8 half of the numeric table is measured), and fp8 FMA reordering means two
  correct nodes differ in bits; on top of that the node computes on the codec-rounded
  activation while the oracle sees raw f32. The comparison is therefore a cosine +
  relative-error envelope. The **exact byte-compare lane is blocked on ADR-0018's
  `Int8Codec` arm** and is a labeled follow-up, exactly as the Decision scoped ("whether
  comparison is exact or tolerance-based follows ADR-0018").
- **Trust weighting is agreement-count ONLY** — no stake, no reputation economy. Stake
  stays the deferred "(and later, optionally, stake)" clause of the Decision.

Membership therefore remains a SOCIAL boundary in the docs: spot-checks are a deterrent
and an integrity signal, not an adversarial-safety guarantee, until a real ≥20-node party
runs them (M5.C / #7). The retrofit the original Decision promised is now wired; the
byzantine-safety CLAIM is still not made.

## Consequences

- WAN milestones ship without solving byzantine compute first.
- Verification is now a WORKING retrofit (tolerance-based spot-checks + per-node trust),
  not just an accommodated one — but its comparison mode and any exact lane inherit from
  the still-proposed ADR-0018, and stake-based trust stays deferred.
- Verification is a retrofit the architecture already accommodates: canonical
  blobs, canonical wire bytes, dispatch log with per-node attribution.
- Until spot-checks land, pool membership is a social boundary, and the docs
  must say so plainly.
- Perplexity canaries (ADR-0008) remain the only integrity signal in the
  trusted phase — another reason they are mandatory.

## Alternatives considered

- **ZK / cryptographic verifiable inference** — orders of magnitude too
  expensive for MB-scale matmuls at step cadence, for the foreseeable future.
- **Full redundant recompute (send every dispatch to 2 nodes and compare)** —
  halves effective pool throughput to catch a rare event; spot-checking buys
  nearly the same deterrence for a fraction of the cost.
- **Reputation-only without recompute** — gameable; trust must be anchored in
  occasionally-verified ground truth.
