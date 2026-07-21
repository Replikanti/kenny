# ADR-0016: Network stack — sync-first, transport decided by M3 numbers

- Status: proposed
- Date: 2026-07-21

## Context

The star topology (ADR-0003) means moderate connection counts and long-lived
links: one spine talking to tens of nodes. Through M3 (`tc netem` on a LAN),
std threads + blocking TCP (+ rustls when encryption enters) are sufficient and
keep the codebase free of an async runtime. QUIC via `quinn` would pull in
tokio; hand-rolled UDP reliability fits the house style but costs months.

## Decision (interim, to be confirmed)

Start **sync**: std threads, blocking TCP, one connection per node, no async
runtime. Defer the transport decision until M3 produces data on where transport
actually hurts: head-of-line blocking under loss, connection scaling, interplay
with hedging (ADR-0010).

Options kept on the table:

- (a) stay on sync TCP + connection pool (default if M3 numbers are fine)
- (b) QUIC via `quinn`, accepting the tokio boundary at the wire layer
- (c) hand-rolled UDP reliability protocol (only with strong M3 evidence that
  TCP semantics are the bottleneck)

## Consequences

- M0–M3 code stays simple, debuggable, and dependency-light.
- If M3 says TCP head-of-line blocking eats the tail budget under loss, the wire
  layer swaps behind existing seams (`WireCodec` — ADR-0011 — plus a thin
  transport trait to be introduced with M1).
- Risk: sync habits ossifying. Mitigation: transport stays behind one module
  boundary from the first networked commit.

## Accept when

M3 benchmark table exists and either confirms sync TCP through M4/M5 or names
its replacement with numbers.

## Alternatives considered

Captured as options (a)–(c) above; the rejection of "async everywhere from day
one" is itself the interim decision — an async runtime is a cost paid only for a
measured reason.
