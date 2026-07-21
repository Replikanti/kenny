# ADR-0014: Per-client token budget per step (batch scheduler fairness)

- Status: accepted
- Date: 2026-07-21

## Context

The spine's batch scheduler multiplexes hundreds of streams from multiple
clients (ADR-0006). Without a policy, one verbose agent colony can occupy the
whole batch and starve everyone else. Economics and billing are out of scope for
now, but starvation is a technology problem, not an economics problem.

## Decision

The batch scheduler enforces a **per-client token budget per step**: each client
gets a bounded share of the batch, and capacity unused by one client
redistributes to the others (work-conserving). Client identity is established at
the gate (sessions → client).

## Consequences

- No client can starve another; farm operators can mix workloads on one party
  without negotiation.
- The dispatch log already records per-client consumption — future metering or
  economics reads it for free, with no new instrumentation.
- The gate must carry client identity into the spine's stream metadata from day
  one of gate work.

## Alternatives considered

- **FIFO admission** — first verbose client wins the pool; rejected.
- **Per-token strict fair queuing** — scheduler complexity disproportionate to
  the need; a per-step budget achieves fairness at step granularity, which is
  the only granularity the systolic loop has anyway.
- **Priority/pricing tiers** — economics, explicitly out of scope for now
  (MANIFESTO §2); the budget mechanism is where such tiers would later attach.
