# ADR-0023: Batching composes on the existing wire

- Status: accepted
- Date: 2026-07-22

## Context

ADR-0006 makes batching the product: decode is sequential per stream with one
round-trip barrier per MoE layer (MANIFESTO §4.4, "the systolic pump"), so only
batching across independent streams amortizes those barriers into aggregate
throughput. M2 introduces the batched decode path.

The question this ADR settles is *how batching meets the wire*. Two shapes were
possible:

1. **Compose on the existing frames.** A batched step of `B` streams issues `B`
   independent `Dispatch` (KNYD) frames per MoE layer and reads `B` `Gather`
   (KNYG) frames, one per stream. The KNYD/KNYG/KNYW layout is UNCHANGED,
   `WIRE_VERSION` stays 1, every codec version stays 1, and every wire golden
   (`golden_dispatch_bytes` / `golden_gather_bytes` / `golden_handshake_bytes` +
   the codec goldens) stays byte-identical.
2. **A new batch-envelope frame** carrying `B` activations plus per-stream
   expert lists in one message — a structural change to a consensus surface
   (ADR-0011): bump `WIRE_VERSION` to 2, re-baseline every wire golden, gate the
   change through `kenny-format-auditor`.

## Decision

Batching **composes on the existing wire**. No batch-envelope frame, no
`WIRE_VERSION` bump, no codec bump. `B` streams are `B` independent
`Dispatch`/`Gather` pairs per MoE layer; batching lives entirely spine-side
(`src/spine.rs` + the CLI). `src/node.rs` is literally untouched — its serve
loop already answers a stream of dispatch/gather pairs (`while let Ok(dispatch)
= recv_dispatch(..)`), so a batched spine is just a faster-arriving stream of the
frames it already handles.

The single per-layer round-trip barrier is amortized across the batch by a
**concurrent split-stream pipeline** on the one TCP connection
(`NodeDispatch::dispatch_batch`): a writer thread sends the `B` encoded
`Dispatch` frames on a `TcpStream::try_clone()` handle while the main thread
concurrently drains the `B` `Gather` frames on the original handle, joined by
`std::thread::scope` (ADR-0016's std-threads, blocking-TCP posture — no async
runtime). FIFO order on the single stream preserves dispatch↔gather pairing.
The recorded fallback, if a future transport exhibits a large-`B` stall, is a
**bounded write-window** (send at most `W` dispatches ahead of the gathers
drained) rather than an envelope frame — it stays on the composed wire.

## Consequences

- The consensus surface is proven untouched: the wire goldens stay
  byte-identical, which is itself the evidence for the ADR-0019 single-crate
  re-confirmation and the `kenny-format-auditor` sign-off on the M2 PRs.
- `WIRE_VERSION`/codec-version stability means an M1 node binary serves an M2
  batched spine with no protocol renegotiation.
- Negative: the framing overhead is not shared across a batch — each stream pays
  its own `Dispatch`/`Gather` headers + expert-id list. This is `(B−1)×12`
  framing bytes per layer of pure loss versus an envelope, accepted because it is
  negligible against the 2 KB / 16 KB fp8 activation payloads that actually bound
  `B_max` (MANIFESTO §4.4). The envelope's real payoff — node-side co-activation
  dedup (MANIFESTO §4.3) — is an explicitly-deferred reduction lever, not an M2
  goal, so nothing of value is given up now.
- Negative: a naive single-threaded "send all `B`, then receive all `B`" would
  deadlock at large `B` when the socket buffers fill (the node stops reading
  dispatches because its gather writes block). The concurrent split-stream
  mechanism above is the mitigation; it is the reason `dispatch_batch` is not a
  trivial loop for `NodeDispatch`.

## Accept when

Accepted in this PR: the batched spine path lands, the wire goldens are verified
byte-identical, and the batched `NodeDispatch` path reproduces the sequential
`LocalDispatch` path bit-for-bit (`tests/dispatch.rs`).

## Alternatives considered

- **Batch-envelope frame** (`B` activations + per-stream expert lists in one
  message). Rejected: it is a `WIRE_VERSION` bump and a re-baseline of every wire
  golden — a consensus-surface change — bought only `(B−1)×12` framing bytes per
  layer against multi-KB payloads. Its genuine benefit (co-activation dedup,
  §4.3) is a deferred lever, not an M2 requirement, so the cost is paid with no
  matching return now. Revisit if and when §4.3 dedup is implemented.
- **Single-threaded send-all-then-recv-all over the composed wire.** Rejected as
  the *mechanism* (kept as the composed-wire shape): correct only for small `B`,
  deadlocks once socket buffers fill. The split-stream pipeline is the same wire
  without the head-of-line stall.
