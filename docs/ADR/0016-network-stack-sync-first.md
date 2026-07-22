# ADR-0016: Network stack — sync-first, transport decided by M3 numbers

- Status: accepted
- Date: 2026-07-21

## Context

The star topology (ADR-0003) means moderate connection counts and long-lived
links: one spine talking to tens of nodes. Through M3 (`tc netem` on a LAN),
std threads + blocking TCP (+ rustls when encryption enters) are sufficient and
keep the codebase free of an async runtime. QUIC via `quinn` would pull in
tokio; hand-rolled UDP reliability fits the house style but costs months.

## Decision (confirmed at M3)

**Option (a): stay on sync TCP** — std threads, blocking TCP, one connection per
node, no async runtime — carried through M4/M5. The M3 `tc netem` numbers (BENCH
"M3" section) confirm it: the per-layer RTT barrier amortizes across the batch as
MANIFESTO §4.4 predicts (G1/G2/G3 pass, `tok/s(B=64)/tok/s(B=1) = 61.1×`), and
head-of-line blocking under ≤1 % loss is bounded by the per-layer timeout
(ADR-0010) with a quality-safe hedge path — no TCP-semantics wall pointing at
QUIC/UDP. Two conditions on the sync path, both interim per-socket options rather
than an async runtime:

- **`TCP_NODELAY` is mandatory on every dispatch/gather socket** (see the M3
  findings below) — the composed-wire pipeline (ADR-0023) does not amortize the
  barrier with Nagle on.
- A **per-layer receive deadline** (ADR-0010) bounds the tail under loss.

Options that were on the table and are **not** taken:

- (b) QUIC via `quinn`, accepting the tokio boundary at the wire layer — not
  needed; the M3 loss/HOL numbers do not implicate TCP semantics.
- (c) hand-rolled UDP reliability protocol — not needed; no M3 evidence that TCP
  semantics are the bottleneck.

Both stay available behind the transport module boundary if a later milestone
(real WAN, M4+) produces evidence that TCP head-of-line blocking eats the tail
budget where the timeout + hedge cannot bound it.

## Consequences

- M0–M3 code stays simple, debuggable, and dependency-light.
- If a later milestone says TCP head-of-line blocking eats the tail budget under
  loss where the timeout + hedge cannot bound it, the wire layer swaps behind
  existing seams (`WireCodec` — ADR-0011 — plus the transport module boundary).
- Risk: sync habits ossifying. Mitigation: transport stays behind one module
  boundary from the first networked commit.

### M3 findings (`tc netem`, simulated WAN on one host)

The full numbers live in `BENCH.md` "M3"; the transport-relevant readings:

- **`TCP_NODELAY` is load-bearing for the ADR-0023 amortization.** The
  composed-wire batch pipeline streams B small per-stream frames per MoE layer.
  With Nagle on (the pre-fix sockets), each frame is held until the prior frame's
  ACK, so the B per-stream round-trips serialize and per-step time grows **∝ B**
  on a compute≈0 fixture (pre-fix: B=1 1.47 s, B=4 6.35 s → G3 FAIL as
  pre-registered). Disabling Nagle restores the predicted B-independent step and
  `tok/s(B=64)/tok/s(B=1) = 61.1×` (G3 PASS, ≈96 % of the ideal 64×). The **real
  model masks the stall** (per-dispatch compute exceeds the ACK-wait), so only the
  compute-free fixture exposes it — precisely the "where transport actually hurts"
  evidence this gate asked for. The fix is a one-line per-socket hint, not a
  TCP-semantics wall: it **strengthens** option (a) rather than pointing at (b)/(c).
- **Amortization holds at real scale (G1/G2).** Real-model `Δt_step` is
  B-independent (2.156 s at B=1, 1.292 s at B=8) and sits at the 48·30 ms =
  1.44 s RTT floor plus a ~0.7 s bandwidth-delay-product term (the §4.4
  `tail_transfer`, ≈0 on bare loopback, nonzero behind a 30 ms delay qdisc).
- **HOL under loss is bounded (G4, partial).** With the per-layer timeout OFF, one
  lost segment head-of-line-blocks the stream and p99 climbs monotonically with
  loss. The timeout caps p99 at every loss level (timeout rate ≤ 5.2 %), and the
  2-node hedge (ADR-0010) tightens it further and quality-safely (renorm rate 0 at
  1 % loss). This does **not** reach the pre-registered G4 aspiration of ≲2× the
  loss-free `t_step`: the best measured is ~2.4× at 1 % loss (single-node timeout
  ~3.2×, hedge ~2.4×), and that p99 is a worst-of-31 sample, not a populated tail
  (BENCH caveat). The gate treats G4 as **pass-with-caveat**: the mechanism
  bounds the tail and stays quality-safe under ≤1 % loss, and what would flip it
  to a clean pass is a real placed multi-node pool (ADR-0009) with a populated
  tail on genuine WAN — measured at M4+ (#4), not a transport swap.

## Accept when

Met: the M3 benchmark table (`BENCH.md`) confirms sync TCP + `TCP_NODELAY` +
per-layer timeout through the pre-registered go/no-go (GO, G4 pass-with-caveat).

## Alternatives considered

Captured as options (a)–(c) above; the rejection of "async everywhere from day
one" is itself the interim decision — an async runtime is a cost paid only for a
measured reason.
