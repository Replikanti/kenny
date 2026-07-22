# ADR-0008: Graceful degradation by top-k renormalization — with mandatory canaries

- Status: accepted
- Date: 2026-07-21

## Context

On a WAN pool, some expert will be missing, late, or dead on essentially every
step. MoE models tolerate expert dropout with soft quality loss — the router's
top-k weights can be renormalized over whatever subset answered. But a mechanism
that silently tolerates missing experts also silently tolerates dead replicas,
corrupted blobs, and creeping capacity loss.

## Decision

- A missing or late expert never blocks a step: the spine **renormalizes** the
  router weights over the available top-k subset and continues.
- **Corollary, part of this same decision, not an optional feature:** quality
  degradation must be measured from day zero —
  - **perplexity canaries**: fixed prompt sets scored continuously against known
    baselines,
  - **heat-map alarms**: per-expert dispatch/failure rates that surface dead or
    never-answering replicas.
- The day-zero dashboard (MANIFESTO §5) includes the perplexity canary as one of
  its five numbers. A pool without canaries is blind and must not serve.

## Consequences

- No step ever stalls on one node; churn shows up as a smooth quality dip, not
  an outage.
- Renormed outputs differ from the reference model — verification and diff
  tooling must account for which experts actually fired (interacts with
  ADR-0018).
- Canary + heat-map infrastructure is on the critical path of early milestones,
  not a nice-to-have for later.

### M4 update (2026-07-22) — perplexity canary landed

The **perplexity canary** half of the corollary is now implemented (`src/canary.rs`,
`kenny canary`): teacher-forced perplexity of a carved blob+wire path scored against
the bf16-source reference (the M0/M1 `diff.rs::source_matrix` methodology, A6) over a
fixed seed-keyed prompt set, reported as `Δppl = ppl(test) − ppl(ref)`. The per-position
teacher-forced logits it needs are `Spine::logits_per_position`; the score is the stable
`logsumexp(logits) − logits[target]` mean, exponentiated. CI runs it model-free on the
fixture (deterministic); the real Qwen3-30B-A3B Δppl is the `KENNY_MODEL_DIR`-gated arm
and lands in BENCH "M4 — perplexity canary". This is also the deciding quality axis
ADR-0018 was blocked on. The **heat-map alarm** half already shipped with placement
(`PlacedDispatch::suspect_replicas`, M4 PR2). Both dashboard corollaries are now real.

### M5.A update (2026-07-22) — degradation MEASURED during correlated churn

The renorm was proven to bridge a single missing replica (M4). M5.A proves it
bridges the harder failure this ADR was written for: a whole **failure domain**
dying together. The in-process locks (`tests/dispatch.rs`, model-free, netns-free)
stand up a pool whose nodes SHARE failure domains, kill a domain mid-run (its
nodes go black hole while the placement map still points at them — the down-window
before an operator re-places), and assert the three things "no step ever stalls;
churn shows up as a smooth quality dip, not an outage" requires:

- **`churn_domain_renorms_and_completes`** — a correlated-domain kill neither
  stalls nor errors: the replica-set budget (`hedge_delay`) stalls each silent
  node so the renorm bridges the gap, the run COMPLETES, and the stranded experts
  renorm bit-for-bit like a local run that dropped exactly that set
  (`renorm_steps > 0`, degradation measured not silent). A re-place over the
  survivors then restores full coverage — the elasticity round-trip.
- **`churn_flags_dead_replicas`** — the stranded domain's experts (dispatched
  every step, answered on none) are surfaced EXACTLY by
  `PlacedDispatch::suspect_replicas` (`HeatMap::suspect`), and the survivors' are
  not: the "creeping capacity loss is never silent" half, measured off spine-local
  heat, never on the wire.
- **`renorm_quality_dip_grows_with_dropout`** — the canary corollary under
  dropout: as the fraction of experts forced not-held GROWS, the renormed output
  diverges monotonically from the full-coverage answer (scored through the canary's
  own `Spine::logits_per_position`) and returns EXACTLY to baseline when coverage
  is restored. HONEST CAVEAT (ADR-0007): on random fixture weights the canary NLL
  drifts toward the ln(vocab) floor rather than strictly worsening, so the SIGN of
  a real perplexity dip is the `KENNY_MODEL_DIR` arm (`renorm_quality_dip_real_model`,
  BENCH "M5.A — elasticity").

The shaped-uplink netns SIMULATION (`netem_churn`, `tools/netem-bench.sh --churn`)
additionally surfaces a real ADR-0010 × ADR-0008 interaction: a replica-set budget
tighter than a slow-but-healthy survivor's round-trip false-flags that survivor
suspect — the budget must clear the slowest uplink. All of the above is
spine-local: `WIRE_VERSION` and every wire golden stay frozen (ADR-0024).

## Alternatives considered

- **Block / retry until the expert answers** — hands tail latency to the slowest
  node × 75 layers (see ADR-0010); unacceptable.
- **Zero-imputation without renormalization** — silently shrinks FFN output
  magnitude; renorm preserves scale.
- **Failing the affected streams** — turns a soft quality dip into a hard error;
  strictly worse for the target workload (ADR-0006).
