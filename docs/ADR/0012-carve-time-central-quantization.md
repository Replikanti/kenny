# ADR-0012: Quantize at carve time, centrally

- Status: accepted
- Date: 2026-07-21

## Context

Content addressing (ADR-0005) only means something if every replica of a CID is
bit-identical. Quantization performed independently on nodes would be a
nondeterminism zoo: floating-point rounding differs across architectures,
library versions, and vector ISAs — same input, different bytes, broken CIDs.

## Decision

All weight quantization happens **inside `kenny carve`, centrally, before
hashing**. The blob a node stores and serves is the canonical artifact; nodes
never transform weights, only load and execute them.

Carve supports explicit dtype modes (bf16 passthrough for validation; fp8 E4M3
and/or int8 with per-channel scales for deployment), each producing its own
manifest and thus its own model identity.

This keeps open the int8/int32-accumulation deterministic execution path —
bit-exact across architectures — that cheap verification would want (ADR-0018).

## Consequences

- One canonical artifact per (model, quantization config); rollouts and
  rollbacks are manifest swaps.
- Re-quantization is a new carve and a new identity — precision is part of what
  you are running, visibly.
- Nodes cannot adapt precision to local hardware. Deliberate: uniform bits are
  what make the pool a cache (ADR-0005) and auditable (ADR-0011, ADR-0015).
- Carve is a heavyweight offline step (hashing + quantizing hundreds of GB).
  Acceptable: it runs once per model revision, on the spine's hardware.

## Alternatives considered

- **Node-side quantization** — breaks content addressing; rejected outright.
- **Ship bf16 everywhere, quantize only on the wire** — doubles node RAM per
  expert for zero quality benefit at the weights level.
- **Per-node precision tiers** — multiplies placement constraints and the
  verification matrix; revisit only with evidence the quality delta matters.
