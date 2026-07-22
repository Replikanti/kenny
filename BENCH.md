# BENCH

Measured milestone numbers. Convention: median + p99 where a metric has a
distribution, exact setup always, wire bytes counted at the socket (applies
from M1 on). No vibes.

## M0 — carve + diff (2026-07-22)

Setup: 13th Gen Intel Core i7-1355U (12 threads), 30 GiB RAM, KIOXIA
KXG8AZNV1T02 NVMe (954 GB), Fedora Linux 7.1.3, rustc 1.95.0, kenny release
build (M0 diff branch). Source model: Qwen3-30B-A3B bf16 — 61.1 GB, 16
safetensors shards, integrity-verified against upstream file sizes. Page
cache partially warm (61 GB source vs 30 GiB RAM); carve output on the same
NVMe. Carve parallelism: 12 worker threads.

### Throughput

| operation | wall | bytes out | notes |
|---|---|---|---|
| carve bf16 (cold out dir) | 67.0 s | 58.0 GB | 6,144 blobs, ~0.87 GB/s on the write side |
| re-carve bf16 (full dedup) | 31.2 s | 0 | hash-only verification pass, 6,144/6,144 skipped |
| carve fp8 e4m3 | 123.0 s | 29.1 GB | central per-channel quantization (ADR-0012) |
| carve int8 | 119.5 s | 29.1 GB | central per-channel quantization |
| diff of one layer (128 experts × batch 8) | 8.1–8.5 s | — | layers 0 and 47 measured |

Carve is a once-per-revision offline job, so throughput rows are single cold
runs; the dedup re-carve was run twice (31.2 s / 31.4 s, median reported).

### Quality — `kenny diff`, layer 0, batch 8, seed 42, worst expert of 128

| dtype | bitwise exact | max-abs | cosine |
|---|---|---|---|
| bf16 passthrough | **yes** (layer 47 sanity run identical) | 0 | 1.0 |
| fp8 e4m3 per-channel | no | 1.73e-2 | 0.998999 |
| int8 per-channel | no | 1.10e-2 | 0.999873 |

Manifest identities from one source: bf16 `e29ad154…`, fp8 `78698c9b…`,
int8 `eb489bca…` — dtype is part of the model identity (ADR-0012).

First ADR-0018 signal: at identical blob size, int8 per-channel carries ~8×
less cosine error than fp8 e4m3 per-channel on these weights (1−cos:
1.0e-3 vs 1.3e-4). The wire-path decision still waits for M3's end-to-end
numbers (perplexity canaries + throughput), as ADR-0018 specifies.

### Reproduce

```
KENNY_MODEL_DIR=<model_dir> cargo test --release --test roundtrip real_model -- --nocapture
kenny carve <model_dir> --out <dir> [--dtype bf16|fp8|int8]
kenny diff  <model_dir> <carved_dir> [--layer N] [--batch N] [--seed N]
```

## M1 — dispatch/gather round-trip (2026-07-22)

The first end-to-end run: a `kenny spine` (Qwen3-30B-A3B spine-sim, ADR-0020)
and a `kenny node` as **two separate processes on one box**, the routed MoE FFN
of every layer dispatched to the node over sync TCP (ADR-0016 interim) with the
fp8 e4m3 wire codec (ADR-0011). Same machine as M0 (i7-1355U, 30 GiB RAM, NVMe,
Fedora 7.1.3, rustc 1.95.0, release build). Model: Qwen3-30B-A3B, fp8-carved
(29,079,257,088 blob bytes, 6,144 experts); the spine reads its always-on
tensors (embeddings, attention, router, final norm, lm_head) from the source
bf16 shards by manifest range (ADR-0005), the node lazily mmaps fp8 blobs by
CID. Topology: `spine ⇄ 127.0.0.1 ⇄ node`, single stream, greedy decode, prompt
4 tokens → 8 generated (11 forward passes, 528 dispatches = 11 × 48 MoE layers,
top-k 8).

### Throughput — single stream, localhost, greedy

| metric | value | notes |
|---|---|---|
| tok/s | 0.10 | 8 tokens in 78.1 s over the node process |
| per-forward latency | median 7.0 s, p99 7.5 s | 11 forwards; ~uniform (dense pure-Rust f32) |
| spine load (always-on tensors) | 3.4 s | embeddings + 48 layers + lm_head, bf16→f32 |
| in-process reference (`--local`) | 85.1 s | same forward, no socket |

Throughput is bounded by the **un-tuned pure-Rust dense forward** (a spine-sim,
ADR-0020), not the wire: a whole 8-token run moved under 10 MB total. Forward
performance tuning is out of scope for M1 (the protocol-round-trip milestone);
the number is recorded honestly, not optimized.

### Wire — counted at the socket (per direction, exact framing)

| direction | bytes | per generated token | accounting |
|---|---|---|---|
| up (spine → node) | 1,096,172 | 137,022 | 44 handshake + 528 × (12 hdr + 2048 x + 16 ids) |
| down (node → spine) | 8,669,760 | 1,083,720 | 528 × (12 hdr + 3×8 rec hdrs + 2048 × 8 y) |
| total | 9,765,932 | 1,220,742 | down ≈ 7.9× up |

The asymmetry is structural (ADR-0011 / A5): the spine sends **one** activation
`x` per dispatch, the node returns **one `y` per answered expert** (top-k = 8),
so down carries ~8× the payload of up. fp8 makes each element 1 byte, so per
dispatch the up payload is `hidden` (2048 B) and the down payload is
`hidden × k` (16,384 B).

### Output sanity — first end-to-end ADR-0018 signal

| comparison | metric | value |
|---|---|---|
| fp8 blob + fp8 wire **vs** bf16 source weights (no quant, no codec) | cosine of final-position logits | **0.999845** |
| same, greedy next-token argmax | fp8 → 25, bf16-source → 911 | **differs** |

Reference is the original bf16 weights via the `diff.rs::source_matrix` path (no
blob quantization, no wire codec — A6), mirroring M0's fp8-vs-bf16 methodology,
teacher-forced on the same prompt. The logit-vector cosine stays at 0.99985 end
to end across 48 layers, yet the **greedy argmax over 151,936 logits still
flips** — a >0.999 cosine is not tight enough to preserve the top token when the
leaders are close. Token-level agreement is therefore explicitly **not** the
quality gate; that is the deferred perplexity canary (ADR-0008). This is the
first measured signal that fp8-on-the-wire carries real, if small, degradation.

### Protocol self-consistency — the M1 correctness gate

The CI gate (synthetic fixture, no model download) is **bit-exact
`LocalDispatch` ≡ `NodeDispatch`** under both the fp8 and bf16 codecs: the
distributed path reproduces the in-process path token-for-token (both apply the
identical codec around the identical `expert::forward`, so any drift is a real
bug — `tests/dispatch.rs`). The same equivalence holds on the **real** model
under fp8 (the S7 run above: `local == node`, bit-for-bit). Renorm over the
answered subset (ADR-0008) is exercised by dropping replicas on both paths and
asserting they still agree, with the down-byte shortfall equal to exactly the
missing experts' `y` payloads.

### Reproduce

```
# CI gate (no model): fixture local≡node bit-exact + wire-byte accounting
cargo test --test dispatch

# S7 real-model two-process run (equivalence + tok/s + wire + cosine)
KENNY_MODEL_DIR=<model_dir> cargo test --release --test dispatch \
    real_model_two_process_dispatch -- --nocapture

# Or the literal two processes by hand:
kenny node  --carved <fp8_carve>                       # prints `listening <addr>`
kenny spine --carved <fp8_carve> --model <model_dir> --node <addr> \
    --codec fp8 --prompt 40,1207,264,3405 --tokens 8
```

## M2 — localhost batching B-sweep (2026-07-22)

The first batching data: aggregate tok/s and per-**step** latency as batch size
`B` rises, with exact per-direction wire bytes. Batching composes on the M1 wire
(ADR-0023) — a batched step issues `B` independent KNYD/KNYG frame pairs per MoE
layer, no `WIRE_VERSION`/codec bump, `src/node.rs` untouched — so this measures
the spine's `generate_batch` path, not a new protocol.

**Topology is LOOPBACK, not a real LAN.** `spine ⇄ 127.0.0.1 ⇄ node`, one box:
the node runs as a serve loop over a loopback TCP socket (the S7 two-process
harness), same machine as M0–M1 (i7-1355U / 12 threads, 30 GiB RAM, KIOXIA
KXG8AZNV1T02 NVMe, Fedora 7.1.3, rustc 1.95.0, release build). Model:
Qwen3-30B-A3B, fp8-carved (6,144 blobs, 29,079,257,088 blob bytes; cold carve
69.3 s, spine always-on load 3.3 s). Greedy decode, **rectangular** batch of `B`
seed-derived independent streams, prompt 2 tokens → 2 generated each (3 forward
passes/stream), top-k 8, 48 MoE layers, fp8 blobs + fp8 wire.

### Throughput vs batch size — aggregate, localhost, greedy

| B | tok/s (aggregate) | per-step median | per-step p99 | up (B) | down (B) | total (B) |
|---|---|---|---|---|---|---|
| 1 | 0.078 | 8.62 s | 8.90 s | 298,988 | 2,364,480 | 2,663,468 |
| 2 | 0.087 | 15.20 s | 16.02 s | 597,932 | 4,728,960 | 5,326,892 |
| 4 | 0.090 | 29.51 s | 30.81 s | 1,195,820 | 9,457,920 | 10,653,740 |
| 8 | 0.092 | 58.04 s | 59.39 s | 2,391,596 | 18,915,840 | 21,307,436 |
| 16 | 0.092 | 118.13 s | 118.78 s | 4,783,148 | 37,831,680 | 42,614,828 |
| 32 | 0.092 | 236.15 s | 237.52 s | 9,566,252 | 75,663,360 | 85,229,612 |
| 64 | 0.094 | 460.77 s | 461.76 s | 19,132,460 | 151,326,720 | 170,459,180 |

The headline finding: **aggregate tok/s is ~flat (0.078 → 0.094) across a 64×
batch range** — it does NOT scale with `B`. If batching amortized a real barrier
it would trend toward `0.078 × B` (≈ 5 tok/s at B = 64); instead per-step wall
time doubles with every doubling of `B` (8.6 → 15.2 → 29.5 → 58.0 → 118 → 236 →
461 s), i.e. a batched step runs its `B` streams' dense forwards essentially
serially. On localhost **RTT ≈ 0, so the MANIFESTO §4.4 per-layer round-trip
barrier has nothing to amortize** — the only gain visible here is the modest ~20 %
rise (fixed per-step overhead spread over more streams), and throughput stays
bounded by the **un-tuned single-threaded dense pure-Rust forward** (a spine-sim,
ADR-0020; forward-perf tuning and data-parallelism are out of scope, as in M1).
This is the honest baseline the M3 `tc netem` run — and the real second-box LAN
re-run — measure the amortization win against; the win is **unobservable at
RTT ≈ 0 by construction**. Per-step median and p99 come from only 3 steps/stream
(2 prime + 1 generate), so p99 here is effectively the slowest of three steps,
not a populated tail percentile — a longer-run tail is deferred with the LAN
numbers.

> B = 128 is in the sweep code but its ~15-minute step did not complete in this
> run (the process was stopped mid-B=128 under sustained CPU load); it is
> deferred to the LAN re-run, where each step is the same CPU-bound cost. The
> completed B ∈ {1…64} already span a 64× range and settle the flat-throughput
> finding.

### Wire — counted at the socket, per direction, reconciled to framing

Batching adds **no new wire shape**: `D` independent dispatch/gather pairs, with
`D = B × forwards × moe_layers = B × 3 × 48 = 144 B` here. With `hidden = 2048`,
`elem = 1` B (fp8), `k = 8`:

```
up   = 44 + D × (12 + hidden·elem + 2k)       = 44 + 144B × 2,076
down =       D × (12 + 3k + hidden·elem·k)     =      144B × 16,420
```

Worked at B = 64: `up = 44 + 9,216 × 2,076 = 19,132,460 B`,
`down = 9,216 × 16,420 = 151,326,720 B` — matching the table exactly (the
`batch_sweep_localhost` test asserts this reconciliation on every B). Per
generated token the split is invariant in `B`: **up ≈ 149,472 B/tok, down
1,182,240 B/tok** (down = 8× up, the top-k = 8 asymmetry of ADR-0011 / A5 — one
`x` up per dispatch, one `y` down per answered expert). The `44 B` handshake is
one-per-connection, so up/tok inches down from 149,494 (B = 1) toward the
144B-term limit as `B` grows.

### Deferred — real-LAN numbers (issue #4 stays open)

The literal M2 ("node on a second physical box over LAN") is hardware-blocked:
this host is the only machine on the LAN, so the numbers above are loopback. The
§4.4 barrier-amortization win and **ADR-0006's critical-mass crossover claim
require real latency and are NOT validated here** — deferred to (a) the M3
`tc netem` injected-RTT run and (b) a genuine second-box re-run of the *same two
binaries*, only addresses changing: `kenny node … --listen 0.0.0.0:<port>` on
box B, `kenny spine … --node <boxB-ip>:<port> --batch <B>` on box A (holding the
source model dir), with the same `batch_sweep_localhost` harness pointed at the
remote address. Preconditions: identical kenny build (same `WIRE_VERSION` +
codec versions) and the same fp8 carve on both (deterministic given the source,
ADR-0012, so handshake `verify` passes), x86/LE, LAN reachability on the port.
Only then is §4.4 measurable. #4 is closed by a human once those numbers land.

### Reproduce

```
# Localhost B-sweep (gated; CI never downloads a model):
KENNY_MODEL_DIR=<model_dir> cargo test --release --test dispatch \
    batch_sweep_localhost -- --nocapture

# By hand, one B:
kenny node  --carved <fp8_carve>                       # prints `listening <addr>`
kenny spine --carved <fp8_carve> --model <model_dir> --node <addr> \
    --codec fp8 --batch <B> --tokens 2
```
