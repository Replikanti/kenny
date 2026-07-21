# ADR-0006: Target workload — async agent farms, not interactive chat

- Status: accepted
- Date: 2026-07-21

## Context

Physics (MANIFESTO §4.4): decode is sequential per stream with one WAN
round-trip barrier per MoE layer, so per-stream speed is ~1 tok/s and nothing
can fix that — only batching across independent streams amortizes the barriers.
Below ~64 concurrent streams the pool is slower than a single local box.

## Decision

kenny is designed exclusively for **async agent farms**: hundreds of concurrent,
independent, latency-tolerant streams (agent colonies, batch pipelines,
overnight jobs). Aggregate throughput is the product; per-stream latency is
explicitly sacrificed.

There is no small-scale mode and no interactive mode. Critical mass is a
documented go/no-go gate for deployments, not a bug to engineer around.

Members are simultaneously suppliers (run `kenny node`) and consumers (call the
gate); fuller batches make the pool faster for everyone.

## Consequences

- Every component may assume deep batches: the spine batch scheduler, wire
  framing, hedging budgets, prefix-cache design all optimize for throughput at
  B≈64–512.
- Fairness across clients becomes a real scheduling problem (ADR-0014).
- Expectation-setting is part of the product: a party that can't feed the batch
  shouldn't launch (MANIFESTO §4.4 go/no-go numbers).
- Latency-tolerant consumers tolerate hedged/renormed steps (ADR-0008,
  ADR-0010) without UX damage.

## Alternatives considered

- **Interactive chat as target** — physics says no; anything built for it would
  be a lie at 1 tok/s.
- **Hybrid modes (interactive lane + batch lane)** — complicates the scheduler
  for a lane that can't be good; revisit only if speculative decoding (MTP,
  MANIFESTO §4.2) changes per-stream math by ~5×.
