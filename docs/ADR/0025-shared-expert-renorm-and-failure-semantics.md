# ADR-0025: GLM-5.2 shared expert — spine-local placement, renorm composition, failure semantics

- Status: proposed
- Date: 2026-07-22

## Context

GLM-5.2 (ADR-0007, MANIFESTO §4.2) adds one structural element the whole system
was built around not having: a **shared expert** that fires on **every** token,
in addition to the top-8 routed experts. Qwen3-30B-A3B — the dev testbed on which
M0–M5.B were built and validated — has **no** shared expert (MANIFESTO §4.2), so
nothing in the shipped code (`src/spine.rs` router/renorm, `src/placement.rs`,
`src/canary.rs`) has ever exercised one, and there is no Qwen3 reference to diff a
shared-expert path against.

ADR-0002 already answered the *placement* question in one clause — "attention, KV
cache, router, embeddings, **shared experts**, dense FFN layers — stays
centralized on the spine" — but it did not settle the two questions that only
become live once the routed-expert pool can renorm and lose replicas around it:

1. **Composition with renorm (ADR-0008).** The spine renormalizes the router's
   top-k weights over whatever *routed* subset answered. Does the shared expert's
   output sit INSIDE that renormed routed sum (and so get rescaled / partly
   renormalized away when routed replicas are missing), or OUTSIDE it (always-on,
   full magnitude, independent of routed coverage)?
2. **Failure semantics (ADR-0008 × ADR-0009).** A routed-expert miss is a *soft*
   event — renorm bridges it. Is a shared-expert "miss" the same soft event, or a
   different, *fatal* one?

These are genuine new decisions, not restatements of ADR-0002: they fix how the
always-on remainder composes with the graceful-degradation machinery, and they
are on the critical path the first time kenny serves GLM-5.2 (M5.C). The exact
numeric contract (the model's own shared-expert weight, and whether GLM sums
`shared + routed` before or after the router normalization) is a property of the
real checkpoint and **cannot be reference-validated on Qwen3** — exactly the trap
ADR-0007 names ("synthetic fixtures are for CI, not for validation of real
numerics"). This ADR therefore fixes the *architectural* contract now and defers
the *numeric* confirmation to the real card.

This ADR is scoped to the shared expert **only**. GLM-5.2's other two
Qwen3-absent features — DSA sparse attention (spine-stateful, interacts with the
KV wall, MANIFESTO §5.2 / ADR-0022) and the MTP speculative-decoding layer — are
explicitly **out of scope** here: both are untested scaffolding without the real
card and get their own treatment when built. This ADR does not design them.

## Decision

1. **The shared expert is spine-local, never a distributed blob.** It is part of
   the always-on remainder that lives on the spine (ADR-0002, ADR-0004, MANIFESTO
   §4.2 "~14–18 GB fp8 → spine"). It is **not** carved into content-addressed
   routed blobs (ADR-0005), **not** dispatched over the wire (ADR-0011), **not**
   entered into the `PlacementMap` or `HeatMap`, and **not** subject to
   heat-driven replication or placement (ADR-0009). Consequently it can never be a
   "placement hole", can never appear as a `suspect_replica`, and its presence is
   `WIRE_VERSION`-neutral — adding it changes no wire golden and no codec.

2. **The shared expert composes OUTSIDE the renormed routed sum.** The step output
   is

   ```
   y = shared_y  +  Σ_{i ∈ answered}  ( w_i / Σ_{j ∈ answered} w_j ) · routed_y_i
   ```

   where the renormalization (`w_i / Σ w_j`, ADR-0008) is taken over the **routed**
   top-k subset that answered, and `shared_y` is added at full magnitude,
   unweighted by that sum (the shared expert carries its own model-defined scale,
   confirmed against the card per Accept-when). Renorm never rescales, divides, or
   drops `shared_y`; losing routed replicas moves weight among the *routed* terms
   only. This is the compositional invariant.

3. **A shared-expert miss is FATAL, not renormable.** Because the shared expert is
   spine-local and always fires, the only way it can be "missing" is a spine fault
   — identical in kind to attention, the router, or the LM head being unavailable.
   There is no distributed replica to lose and nothing to renorm over, so a
   shared-expert failure fails the step (it does not degrade softly). This is the
   deliberate asymmetry with routed experts: **routed miss → soft (renorm);
   shared miss → hard (spine fault).** The graceful-degradation contract (ADR-0008)
   covers routed dropout and correlated node death (M5.A); it does not — and must
   not silently pretend to — cover the shared expert.

4. **No shared-expert code lands under this ADR.** The decision is recorded ahead
   of the implementation, which is built and numerically validated against the
   real GLM-5.2 checkpoint at M5.C (ADR-0007). No speculative Rust and no
   consensus-surface change accompanies this ADR.

## Consequences

- The M5.A elasticity round-trip (join/leave → re-place → renorm; ADR-0009,
  ADR-0024) is **unaffected** by the shared expert: churn moves routed coverage
  only, and `shared_y` is a constant addend outside the renormed sum, so the
  bit-exact re-place and correlated-churn locks (`churn_domain_renorms_and_completes`,
  M5.A) carry forward unchanged when the shared term is later added.
- The perplexity canary (ADR-0008) and verification spot-checks (ADR-0015) must
  attribute `shared_y` to the **spine**, not to any node: a spot-check recomputes
  and compares the *routed* `(layer, expert)` a node returned, never the shared
  expert, and per-node trust is defined over routed dispatches only. The canary's
  full-forward reference includes `shared_y`; the divergence-from-full metric
  (M5.A) is measured on the routed subset that renorm actually varies.
- The dashboard failure taxonomy gains a hard boundary: a shared-expert fault is a
  **spine** alarm (like KV or router), categorically distinct from the routed
  heat-map "creeping capacity loss" alarm (ADR-0008). Conflating the two would
  mask a fatal condition as a soft dip.
- The GLM spine RAM budget (MANIFESTO §4.2, ~14–18 GB fp8 remainder) already
  accounts for the shared expert on the spine; this ADR adds no new blob count and
  does not touch the routed-expert totals (75 × 256 = 19,200).
- The numeric contract in Decision (2) — full-magnitude `shared_y`, added outside
  renorm — is asserted, not yet measured. If the real card weights the shared
  expert or folds it inside the router normalization, Decision (2) is amended (not
  silently reinterpreted) at M5.C, and the composition formula is corrected there.

## Accept when

The real GLM-5.2 checkpoint is carved and served at M5.C (ADR-0007) and the
shared-expert path is built against it, confirming on the actual card that:

1. `shared_y` composes **outside** the renormed routed sum (Decision 2), with the
   model's shared-expert scale reproduced (bf16-source reference, the ADR-0008 /
   `diff.rs::source_matrix` methodology), and
2. a shared-expert fault is handled as a **fatal spine fault**, not renormed
   (Decision 3), on a real party,

with the numbers recorded in BENCH "M5.C". Until then this ADR stays `proposed`:
the architectural contract is fixed, the numeric confirmation is not, and #7
tracks the M5.C exit that supplies it.

## Alternatives considered

- **Shared expert INSIDE the renormed routed sum** — folding `shared_y` into the
  top-k renormalization means missing routed replicas would rescale the shared
  contribution, so the always-on expert's magnitude would drift with WAN churn.
  That contradicts "fires every token" and couples a spine-local constant to
  pool health; rejected.
- **Shared expert as a distributed routed blob (replicated, r ≥ 2)** — treats a
  fire-every-token component like an occasionally-dispatched one: every step would
  pay a WAN round-trip for it (75 layers × every token), and a bad down-window
  would renorm it away — turning a fatal condition into a silent soft dip. It also
  contradicts ADR-0002's placement clause. Rejected on both cost and safety.
- **Design DSA and MTP in this ADR too** — they are Qwen3-absent and unvalidatable
  without the real card (ADR-0007), and DSA is spine-stateful and entangled with
  the KV wall (MANIFESTO §5.2, ADR-0022); bundling them here would be untested
  scaffolding. Deferred to their own treatment.
- **Write a synthetic shared-expert fixture now to lock the compositional
  invariant** — a fixture would only test that *our own* synthetic addend is
  placed outside the renorm; with no shared-expert compute in the tree and no
  Qwen3 reference, it validates scaffolding against itself, not the real numeric
  contract. Dropped as dishonest without the real card (this ADR stays docs-only).
</content>
</invoke>
