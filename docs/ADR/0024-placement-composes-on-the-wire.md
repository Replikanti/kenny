# ADR-0024: Placement dispatch composes on the existing wire

- Status: accepted
- Date: 2026-07-22

## Context

M4 finally places distinct experts on distinct nodes: ADR-0009 replication
(`r = 2–3` across failure domains, heat-driven) stops being prose and becomes a
`PlacementMap` — `(layer, expert) -> replica set of node indices` — that a spine
`PlacedDispatch` fans a layer's routed experts across. Through M1–M3 every path
was either a single node or a fixture *mirror* pair where both nodes held every
expert (`HedgedDispatch`, ADR-0010); nothing ever partitioned a routed expert
list by holder and sent each holder its sub-list.

The question this ADR settles is the same one ADR-0023 settled for batching: how
does placement fan-out meet the wire? Two shapes are possible, and hedging
(ADR-0010) rides on whichever we pick:

1. **Compose on the existing frames.** `PlacedDispatch` partitions a layer's
   routed experts by holding node and sends each node its sub-list as ordinary
   `Dispatch` (KNYD) frames on that node's connection, then gathers `Gather`
   (KNYG) frames and reassembles per-expert `ys` into routed order. The
   KNYD/KNYG/handshake layout is UNCHANGED, `WIRE_VERSION` stays 1, every codec
   version stays 1, and every wire golden stays byte-identical. Hedging is a
   *replica-set second-send*: a stalled expert whose replica set has a second
   node is re-dispatched there, first-answer-wins — the same frames to a
   different holder.
2. **A placement-aware envelope frame** carrying the replica map or a
   multi-holder routing header in one message — a structural change to a
   consensus surface (ADR-0011): bump `WIRE_VERSION` to 2, re-baseline every wire
   golden, gate the change through `kenny-format-auditor`.

## Decision

Placement **composes on the existing wire**. No placement-envelope frame, no
`WIRE_VERSION` bump, no codec bump. Fan-out is a spine-side routing decision
(`src/placement.rs` builds the map; `PlacedDispatch` in `src/spine.rs` splits and
reassembles); `src/node.rs` stays untouched — a node already answers a stream of
`Dispatch`/`Gather` pairs and already answers `not-held` for any `(layer, expert)`
absent from its index (`src/node.rs`), so a *placement hole* (an expert no node
holds) IS the pre-existing not-held → renorm path (ADR-0008), needing no new
signalling.

Reassembly restores per-expert **routed order** before `mix_moe` renorm so a
placed step is byte-identical to a `LocalDispatch` step — the same invariant
ADR-0023 established for batching. Hedging (ADR-0010) is unified onto this: the
M3 fixture mirror pair becomes a real replica-set hedge — the stalled expert's
second replica is the redundant node, first-answer-wins, and with no loss the
primary always wins so the path reduces exactly to the unhedged placed dispatch.
The placement map, heat map, and per-`(layer, expert)` dispatch/failure counters
are all spine-LOCAL state (ADR-0004): never on the wire, never in a manifest,
never cross-node — losing them costs re-placement/recomputation, never
correctness.

## Consequences

- The consensus surface is proven untouched: the five wire goldens
  (`golden_dispatch_bytes`, `golden_gather_bytes`, `golden_handshake_bytes`,
  `golden_fp8_activation_bytes`, `golden_bf16_activation_bytes`) stay
  byte-identical and `WIRE_VERSION == 1` — the same evidence ADR-0023 gave for
  batching, and the `kenny-format-auditor` sign-off for the placement PRs.
- `WIRE_VERSION`/codec-version stability means an M1 node binary serves an M4
  placed spine with no protocol renegotiation; a node holding a *subset* of
  experts is just a node that answers `not-held` more often.
- Placement quality becomes measurable per ADR-0009: per-node step p99 across
  heterogeneous shaped uplinks (dashboard number 4) is the direct output, which
  the single-`lo` M3 harness could not produce.
- Negative: framing overhead is not shared across a fanned-out layer — each
  holder pays its own `Dispatch`/`Gather` headers. This is the same accepted
  loss as ADR-0023 (negligible against the multi-KB fp8 activation payloads that
  bound the step), and the envelope's real payoff (node-side co-activation dedup,
  MANIFESTO §4.3) stays an explicitly-deferred M5 lever, not an M4 goal.

## Accept when

Accepted in the PR that lands `PlacedDispatch` + the CLI multi-node path (the
next M4 PR): the placed spine path lands, the five wire goldens are verified
byte-identical with `WIRE_VERSION == 1`, `PlacedDispatch` reproduces the
`LocalDispatch` path bit-for-bit (`tests/dispatch.rs`), the placement-hole →
renorm path is exercised, and the replica-set hedge is bit-exact and equals the
no-hedge answer under no loss.

**Accepted (M4, issue #6):** `PlacedDispatch` (`src/spine.rs`) fans routed
experts across their holders per the `PlacementMap` and reassembles into routed
order; the `kenny spine --node … --node …` multi-node hook replaced the
single-node `nodes.len() > 1` rejection (`src/cli.rs`). The five wire goldens and
`WIRE_VERSION == 1` are frozen, and `src/node.rs` is untouched. `tests/dispatch.rs`
proves `placed_equals_local_bit_exact` (fan-out over distinct nodes), the
placement-hole renorm (`placed_renorm_on_placement_hole`), and both the
never-fired (`placed_hedge_equals_local_no_loss`) and fired
(`placed_hedge_fires_to_second_replica`) replica-set hedge — all bit-for-bit
`LocalDispatch`.

**Sim follow-up (M4, issue #6):** the netns/netem placement sim landed the three
pieces this ADR deferred to it, none touching the wire or the output. The per-node
`--hold` / `--shard` subset (`kenny node`, applied by dropping the complement from
the index via `Node::drop_expert` — the serve loop and every wire golden stay
byte-identical, a subset node just answers `not-held` more often) lets N nodes hold
DISTINCT expert sets on one host. The concurrent split-stream is realized WITHOUT
threads (which would force a `Send` bound onto `dyn WireCodec`): within a replica
round every holding node's sends are hoisted ahead of the blocking reads, so the
per-node round-trips OVERLAP and a placed step costs ≈ `max` over the nodes' RTTs,
not their `sum` — which is what makes the ADR-0009 per-node step p99 spread across
heterogeneous shaped uplinks (`tools/netem-bench.sh --nodes 3 --placement`)
observable, its "direct output". `PlacedDispatch ≡ LocalDispatch` bit-for-bit still
gates the reorder (hoisting sends reassembles nothing differently).

## Alternatives considered

- **Placement-envelope frame** (replica map or multi-holder header in one
  message). Rejected for the ADR-0023 reasons: a `WIRE_VERSION` bump and a
  re-baseline of every wire golden — a consensus-surface change — bought only
  framing bytes against multi-KB payloads, with its genuine benefit (co-activation
  dedup, §4.3) an explicitly-deferred M5 lever. Revisit if and when §4.3 dedup
  lands.
- **A distinct hedge frame** rather than a replica-set second-send. Rejected: the
  second replica is reachable with the identical `Dispatch` frame to a different
  connection (experts are pure functions, ADR-0004, so either replica's `y` is
  bit-identical), so a dedicated frame buys nothing and touches the wire.
- **Cross-node placement coordination on the wire** (nodes negotiating who holds
  what). Rejected: placement is a spine-local scheduling decision (ADR-0004
  stateful-spine / stateless-experts split); putting it on the wire would make
  the pool a consensus system, which the star topology (ADR-0003) exists to avoid.
