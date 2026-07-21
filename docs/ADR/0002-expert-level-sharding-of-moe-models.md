# ADR-0002: Shard MoE models at the routed-expert level

- Status: accepted
- Date: 2026-07-21

## Context

Frontier open-weight MoE models (GLM-5.2 class, MANIFESTO §4.2) keep ~97 % of
their weights in routed experts. Each routed expert is three small matrices and a
fixed formula (MANIFESTO §4.1): a stateless pure function, typically 5–75 MB.
Running such models conventionally needs terabyte-class fast memory in one box.
Meanwhile, scrap hardware — old laptops, minis, Pis — has idle RAM, idle CPU, and
consumer WAN links with decent aggregate bandwidth and terrible latency.

## Decision

Distribute **routed experts, and only routed experts**, across pool nodes. The
unit of distribution is one expert of one layer. Everything else — attention, KV
cache, router, embeddings, shared experts, dense FFN layers — stays centralized
on the spine (ADR-0003, ADR-0004).

The model is already sharded — training did it. kenny adds no resharding math, no
tensor slicing, no weight surgery beyond cutting along boundaries the MoE
architecture already drew.

## Consequences

- Shards are small (MBs) and independent → any device with RAM and a NIC can hold
  some; heterogeneity is natural, not a special case.
- Expert calls within a layer are embarrassingly parallel; the wire protocol is
  dispatch/gather, not all-reduce.
- Per-token decode latency is dominated by RTT × layer count (sequential
  barriers): per-stream speed is ~1 tok/s. The system is only viable at batch
  scale — critical mass ~64 streams (MANIFESTO §4.4, ADR-0006).
- Zipfian routing makes placement a first-class scheduling problem (ADR-0009).

## Alternatives considered

- **Tensor parallelism over WAN** — all-reduce per layer over consumer links;
  latency × bandwidth kills it outright.
- **Pipeline / layer sharding over WAN (Petals-style)** — sequence state crosses
  untrusted nodes, chain depth multiplies tail latency, and a dead node breaks a
  whole pipeline stage instead of costing one expert out of top-8.
- **Single-box local inference** — capacity ceiling; pools no idle hardware.
- **Smaller / distilled models** — not the goal; the point is frontier weights on
  scrap.
