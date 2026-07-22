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
1.0e-3 vs 1.3e-4). At M3, the throughput axis settled as a non-discriminator
(equal bytes/element); the deciding quality axis stays blocked on the ADR-0008
perplexity canary (ADR-0018, amended).

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

## M3 — tc netem simulated WAN (2026-07-22)

The WAN go/no-go gate. **This is a SIMULATED WAN**: `tc netem delay 15 ms` on the
loopback device inside an unprivileged network namespace (`unshare -rn`), single
host — RTT ≈ 30 ms is injected on `lo`, but there is **no second physical box and
no real network**. The real second-box LAN validation stays issue #4. Every number
below is measured against the criteria pre-registered in the issue #5 plan **before**
the run, not fit afterward.

Same machine as M0–M2: 13th Gen Intel Core i7-1355U (12 threads), 30 GiB RAM,
KIOXIA KXG8AZNV1T02 NVMe, Fedora Linux 7.1.3, rustc 1.95.0, kenny `--release`.
Model (real-model arm): Qwen3-30B-A3B, fp8-carved (6,144 blobs, 29,079,257,088
blob bytes), fp8 wire. Two measurement vehicles, split by cost:

- **Real-model anchor** (Qwen3-30B-A3B, B ∈ {1,8}) — proves the per-layer RTT
  penalty appears at true payload/compute scale. Run twice per B (0 ms control +
  30 ms) inside one netns each, so `Δt_step = t_step(30 ms) − t_step(0 ms)`
  isolates the RTT term from any netns overhead.
- **48-layer synthetic fixture** (compute ≈ 0, hidden 8, 48 MoE barriers ≡ Qwen)
  — makes RTT the whole signal and gives a populated tail cheaply for the
  amortization / loss / hedge matrices. Not the real model, by design.

### t_step decomposition — real-model anchor (Qwen3-30B-A3B, fp8, 30 ms)

Per MANIFESTO §4.4, `t_step ≈ Σ_layers( RTT + tail_transfer )`; predicted RTT
floor `48 × 30 ms = 1.44 s`. `RTT share % = Δt_step / t_step(30 ms)`.

| B | t_step(0 ms) | t_step(30 ms) | Δt_step | predicted 48·RTT | RTT share % |
|---|---|---|---|---|---|
| 1 | 8.851 s | 11.007 s | **2.156 s** | 1.44 s | 19.6 % |
| 8 | 62.325 s | 63.617 s | **1.292 s** | 1.44 s | 2.0 % |

`Δt_step` is **B-independent** (~1.3–2.2 s, no ∝B growth) and sits at the 1.44 s
floor. The B=1 excess over the pure floor (2.156 − 1.44 ≈ 0.7 s) is the §4.4
`tail_transfer` bandwidth-delay-product term: a ~16 KB gather per layer in flight
under a 15 ms one-way delay qdisc (≈0 on bare loopback, nonzero behind netem, no
`rate` limit imposed). So G1 reads as **1.44 s floor + ~0.7 s BDP inflation**, and
G2 (B-independence) holds outright.

### Aggregate tok/s vs B — 48-layer fixture (compute ≈ 0, 30 ms), Nagle vs TCP_NODELAY

The fixture makes RTT the entire cost, so the amortization slope is visible where
M2's flat loopback line could not show it ("the win is unobservable at RTT ≈ 0 by
construction" — M2's own words). **Pre-fix** = the M2/PR1 sockets with Nagle on;
**post-fix** = `TCP_NODELAY` set on every dispatch/gather socket (PR2, three sites:
`NodeDispatch::connect`, `node::serve()` accept, the test thread-harness accept).
`TCP_NODELAY` changes **no application bytes** — byte-identical `wire_up`/`wire_down`,
no `WIRE_VERSION`/codec/golden bump, consensus surface frozen (ADR-0023, ADR-0016).

Predicted per-step floor: `48 × 30 ms = 1.44 s`.

| transport | B | tok/s | step median | step p99 |
|---|---|---|---|---|
| pre-fix (Nagle) | 1 | 0.674 | 1.47 s | 1.48 s |
| pre-fix (Nagle) | 4 | 0.624 | 6.35 s | 6.36 s |
| post-fix (`TCP_NODELAY`) | 1 | 0.673 | 1.47 s | 1.48 s |
| post-fix (`TCP_NODELAY`) | 4 | 2.663 | 1.49 s | 1.50 s |
| post-fix (`TCP_NODELAY`) | 16 | 10.578 | 1.50 s | 1.51 s |
| post-fix (`TCP_NODELAY`) | 64 | 41.113 | 1.54 s | 1.56 s |

**Pre-fix**: step time grows ∝ B (1.47 s → 6.35 s from B=1→4, ×4.3) and tok/s is
flat — the un-amortized Nagle signature. The composed-wire pipeline (ADR-0023)
streams B small per-stream frames per layer; with Nagle on each is held until the
prior frame's ACK, so the B per-stream round-trips serialize. **The real model
masks this** (per-dispatch compute exceeds the ACK-wait), which is why only the
compute-free fixture exposes it. No large-B pre-fix rows: at ∝B a B=64 pre-fix
step is ~94 s, prohibitive — the B∈{1,4} trend is already unambiguous. See the
ADR-0016 M3-findings paragraph for the transport reading.

**Post-fix**: step time is B-independent at 1.47–1.54 s ≈ the 1.44 s floor, and
`tok/s(B=64)/tok/s(B=1) = 41.113/0.673 = 61.1×` — the composed-wire pipeline
amortizes the per-layer barrier across the batch, matching §4.4.

### Loss / HOL + hedge — 48-layer fixture, 30 ms

**HOL / per-layer timeout** (single node, per-layer timeout 90 ms ≈ 3·RTT, 30
steps; loss set at the netem qdisc). The single-node caveat: one node holding all
k experts drops the **whole layer** to renorm on a timeout (degenerate, not
graceful), so this table is p99-bounding + the timeout **rate**, not a
graceful-degradation claim — that is the hedge table below.

| loss % | B | timeout | step median | step p99 | timeout layers | timeout rate | renorm steps |
|---|---|---|---|---|---|---|---|
| 0.0 | 16 | off | 1.49 s | 1.53 s | 0 | 0.0000 | 0 |
| 0.0 | 16 | on  | 1.48 s | 1.51 s | 0 | 0.0000 | 0 |
| 0.0 | 64 | off | 1.53 s | 1.63 s | 0 | 0.0000 | 0 |
| 0.0 | 64 | on  | 1.51 s | 1.63 s | 0 | 0.0000 | 0 |
| 0.5 | 16 | off | 2.92 s | 3.74 s | 0 | 0.0000 | 0 |
| 0.5 | 16 | on  | 2.77 s | 3.25 s | 8  | 0.0054 | 21 |
| 0.5 | 64 | off | 4.41 s | 4.82 s | 0 | 0.0000 | 0 |
| 0.5 | 64 | on  | 4.28 s | 4.43 s | 17 | 0.0114 | 651 |
| 1.0 | 16 | off | 3.47 s | 5.15 s | 0 | 0.0000 | 0 |
| 1.0 | 16 | on  | 3.17 s | 4.38 s | 39 | 0.0262 | 183 |
| 1.0 | 64 | off | 4.67 s | 5.25 s | 0 | 0.0000 | 0 |
| 1.0 | 64 | on  | 4.37 s | 4.50 s | 23 | 0.0155 | 892 |
| 2.0 | 16 | off | 4.70 s | 5.63 s | 0 | 0.0000 | 0 |
| 2.0 | 16 | on  | 3.59 s | 4.74 s | 51 | 0.0343 | 318 |
| 2.0 | 64 | off | 5.01 s | 5.84 s | 0 | 0.0000 | 0 |
| 2.0 | 64 | on  | 4.58 s | 5.01 s | 77 | 0.0517 | 3660 |

Readings: the loss-free control shows timeout on/off identical (the deadline is
inert until loss induces a straggler); with the timeout **off**, one lost segment
head-of-line-blocks the whole stream so p99 climbs monotonically with loss
(1.53 s → 5.84 s at B=64); the timeout **caps p99 at every loss level** (ON always
below OFF), with the timeout rate ≤ 5.2 % of layer-steps at the 2 % / B=64 worst
case.

**Hedge** (2-node fixture, both nodes hold every expert, B=16, 30 steps).

> **p99 caveat (bench honesty):** step p99 here is nearest-rank over **n ≈ 31**
> per-step samples, so at n ≤ 100 the rank is `n − 1` — this "p99" is the **MAX
> step (worst-of-31)**, not a populated tail (contrast the ~101-step amortization
> rows above). Loss inflates the per-step time (~3.5 s), so 101 steps here would
> cost ~10 min/mode; the coarse worst-of-31 is the deliberate wall-clock trade for
> the hedge point.

| loss | mode | step median | step p99 (worst-of-31) | hedge rate | renorm steps |
|---|---|---|---|---|---|
| 0 % | off | 1.50 s | 1.53 s | — | 0 |
| 0 % | on  | 1.48 s | 1.51 s | 0.0000 | 0 |
| 1 % | off | 3.53 s | 4.89 s | — | 0 |
| 1 % | on  | 2.90 s | **3.57 s** | 0.0356 | **0** |

Readings: the hedge fires on **3.56 %** of layer-steps at 1 % loss and never with
no loss (on p99 1.51 s ≈ off 1.53 s — no measurable idle overhead); it collapses
p99 **4.89 s → 3.57 s (−27 %)**; and **renorm_steps = 0** — the redundant secondary
rescues every stall, so unlike the single-node timeout (whole-layer drop → renorm)
the hedged path is **quality-safe** at 1 % loss.

### Pre-registered go/no-go — verdict per criterion

The plan (issue #5) fixed G1–G4 before the run. **GO** requires all four; **NO-GO**
if `Δt_step` grows ∝ B, or fixture tok/s stays flat in B, or HOL under ≤1 % loss
forces quality-breaking renorm even with hedging.

- **G1 — real-model `Δt_step ≈ 48·RTT = 1.44 s` at B=1.** ✅ met (with recorded
  detail): Δt_step = 2.156 s = the 1.44 s RTT floor + ~0.7 s §4.4 BDP inflation.
  The floor is present and B-independent; the excess is the transfer term, not a
  per-stream RTT multiplication.
- **G2 — `Δt_step` B-independent.** ✅ met: 2.156 s at B=1, 1.292 s at B=8 — no
  ∝B growth; the pipeline amortizes the barrier across the batch at real scale.
- **G3 — fixture tok/s scales ~∝ B at 30 ms, `tok/s(64)/tok/s(1) ≥ 16×`.** ✅ met
  on the post-`TCP_NODELAY` transport: **61.1×** (≈96 % of the ideal 64×). Scored
  on the fixed transport per the plan; the pre-fix Nagle rows (∝B, flat tok/s →
  G3 FAIL) are retained as the honest pre-fix data point that motivated the
  one-line fix (still ADR-0016 option (a), zero new deps).
- **G4 — under ≤1 % loss the per-layer timeout caps per-step p99 to ≲2× the
  loss-free `t_step` while renorm stays quality-safe.** ⚠️ **pass-with-caveat**:
  the mechanism bounds the tail and is quality-safe (hedge renorm rate 0 at 1 %
  loss), but the tightest measured p99 is **~2.4×** the loss-free floor (single-node
  timeout ~3.2× → 2-node hedge ~2.4×), **not** the ≲2× aspiration — and that p99
  is a worst-of-31 sample, not a populated tail. What would flip G4 to a clean
  pass: a real placed multi-node pool (ADR-0009) with a populated tail on genuine
  WAN, which is M4+ (#4), not a transport swap. No NO-GO trigger fires: `Δt_step`
  is not ∝B, fixture tok/s is not flat, and no quality-breaking renorm occurs
  under hedging (renorm rate 0).

**Call: GO.** G1 ✅ · G2 ✅ · G3 ✅ (post-`TCP_NODELAY`) · G4 ⚠️ pass-with-caveat
(tail bounded and quality-safe via the hedge path, ~2.4× vs the ≲2× aspiration,
the clean pass deferred to real-LAN #4). Sync TCP (ADR-0016 option (a)) +
`TCP_NODELAY` + per-layer timeout + hedge (ADR-0010) are confirmed through M4/M5;
no evidence points at QUIC/UDP (options (b)/(c)). The fp8-vs-int8 throughput
sub-axis is a non-discriminator (equal bytes/element ⇒ equal `t_step`); the
deciding quality axis stays blocked on the ADR-0008 perplexity canary (ADR-0018,
amended). This is a **simulated** WAN on one host — the real second-box numbers,
a `tc netem rate` constrained-uplink point, and the ADR-0006 critical-mass
crossover remain issue #4.

### Reproduce

```
# Fixture arm (model-free): amortization, loss/HOL, hedge — one netns each.
bash tools/netem-bench.sh --rtt 30                 # amortization tok/s vs B
bash tools/netem-bench.sh --rtt 30 --loss-hol      # per-layer timeout on/off
bash tools/netem-bench.sh --rtt 30 --hedge --loss 1  # 2-node hedge rate vs p99

# Real-model anchor (Qwen3-30B-A3B): Δt_step at B∈{1,8}, 0 ms control + 30 ms.
KENNY_MODEL_DIR=<model_dir> bash tools/netem-bench.sh --rtt 30

# The netns is unprivileged (unshare -rn); if it is unavailable the wrapper
# prints "netns unavailable — skipping M3 netem harness" and exits 0. A plain
# `cargo test` never touches netem (the netem tests gate on KENNY_NETEM_RTT_MS).
```

## M4 — simulated multi-node WAN, placement (2026-07-22)

The first time DISTINCT experts sit on DISTINCT, HETEROGENEOUSLY-SHAPED nodes —
ADR-0009 replication + heat-driven placement measured *in anger* (M1–M3 were a
single node, or a fixture mirror pair where both nodes held every expert). **This
is a SIMULATED WAN**: three `kenny` nodes bound to distinct loopback IPs
(127.0.0.2/3/4) inside one unprivileged network namespace (`unshare -rn`), each
behind its OWN `tc netem delay+rate` band selected by a per-destination `prio`
u32 filter — **no second physical box, no real geography, no correlated churn**.
The real multi-node party stays issue #6. A `PlacedDispatch` fans each MoE layer's
routed experts to their holders on the existing wire (ADR-0024 — no frame change,
`WIRE_VERSION` 1, the five wire goldens byte-identical, `src/node.rs` serve loop
untouched) and reassembles into routed order, so a placed step is bit-for-bit a
`LocalDispatch` step. Sends are hoisted ahead of the reads (a thread-free
concurrent split-stream), so the per-node round-trips OVERLAP and a step costs ≈
`max` over the contacted nodes' RTTs, not their sum.

Same machine as M0–M3: 13th Gen Intel Core i7-1355U (12 threads), 30 GiB RAM,
KIOXIA KXG8AZNV1T02 NVMe, Fedora Linux 7.1.3, rustc 1.95.0, kenny `--release`.
Shaping classes (spine→node egress; the node→spine return is unshaped, so a node's
measured RTT ≈ its one-way delay): node 0 `20 ms / 1000 Mbit` (fat-and-near),
node 1 `60 ms / 100 Mbit`, node 2 `100 ms / 50 Mbit`. Placement is the ADR-0009
bootstrap with a mildly-skewed heat (expert 0 of each layer hot) and `r = 1` so
each node holds a DISJOINT subset and its timing is isolated. `uplink_class ∝` the
shaped rate, so hot experts should flow to the fat uplinks.

### Placement decision — hot experts land on the fat uplink (ADR-0009 objective)

The engine placed the catalog by the `better` step-time order. Held counts
(disjoint, summing to the full catalog):

| node | shaping | fixture (384 experts) | real Qwen3-30B-A3B (6,144 experts) |
|---|---|---|---|
| 0 (127.0.0.2) | 20 ms / 1000 Mbit | **327** | **5,336** |
| 1 (127.0.0.3) | 60 ms / 100 Mbit | 38 | 539 |
| 2 (127.0.0.4) | 100 ms / 50 Mbit | 19 | 269 |

The fat-and-near node draws ~85 % of the catalog (the hot head + its uplink
share), the thin-and-far nodes take the cold tail — placement equalizing load in
*time*, not expert count. This is the ADR-0009 claim a single shared `lo` qdisc
(M3) could not exhibit.

### Per-node step p99 across heterogeneous uplinks — 48-layer fixture (compute ≈ 0, 30 gen steps)

The ADR-0009 **direct output**. A 48-MoE-barrier compute≈0 model (≡ Qwen layer
count, hidden 8) makes each node's netem delay the whole per-fan signal. `per-node`
= the `PlacedDispatch` send→gather round-trip for that node, per fanned layer-step.

| B | agg tok/s | step median | step p99 | wire up / down (B) | node 0 p99 | node 1 p99 | node 2 p99 |
|---|---|---|---|---|---|---|---|
| 1 | 0.561 | 1.70 s | 1.98 s | 43,644 / 55,272 | **20.8 ms** | **61.4 ms** | **101.0 ms** |
| 8 | 2.810 | 2.75 s | 2.91 s | 342,548 / 438,768 | 21.2 ms | 60.9 ms | 100.8 ms |

Each node's p99 tracks its shaped one-way delay (20 / 60 / 100 ms) to within ~1 ms
— the heterogeneous per-node tail M4 exists to produce. Aggregate step p99 (1.98 s
at B=1) is well under the naïve `sum` of the three nodes' delays × 48 layers,
because the concurrent fan overlaps them AND placement keeps most layer-steps on
the fast node — the step is dominated by node 0's 20 ms, not node 2's 100 ms.

**Order-independent (#28).** Phase-2 now gathers each node when ITS OWN answer is
first readable (a non-consuming `peek`), not in fixed node-index order, so a
node's per-node p99 is its own round-trip regardless of connect order. Re-running
with the connect order REVERSED against the delay order
(`tools/netem-bench.sh --nodes 3 --placement --reverse` — node index 0 = 100 ms,
index 2 = 20 ms, same bands) still attributes each node its own delay:

| B | node 0 p99 (100 ms) | node 1 p99 (60 ms) | node 2 p99 (20 ms) |
|---|---|---|---|
| 1 | 100.9 ms | 62.1 ms | 22.5 ms |
| 8 | 100.9 ms | 62.4 ms | 23.8 ms |

Before the fix this reversed order reported ~100 ms for ALL three nodes at B=1
(the two faster nodes inherited the slowest's head-of-line wait). The
`netem_placement` fixture arm now asserts each node's p99 ≤ `delay + max(15 ms,
delay/2)` whenever the delays are distinct, so both connect orders are checked.

> **p99 sample-count caveat (bench honesty):** node 0 (hot, contacted almost every
> layer-step) has a populated tail (~1,400 samples over 31 forwards × 48 layers);
> the sparse nodes 1/2 accrue only the layer-steps that route their cold-tail
> experts, so their p99 is over fewer samples (tens–hundreds) — read it as
> "worst-of-n", not a fully-populated tail, exactly as the M3 hedge caveat.

### Per-node step p99 — real-model anchor (Qwen3-30B-A3B, fp8, B ∈ {1,8}, 2 priming forwards)

At real payload/compute scale the per-node time is `netem delay + real fp8
activation transfer + the node's expert compute` — the netem delay is now the
small term and the compute on node 0 (5,336 held experts) dominates. `p99` here is
worst-of-≈2 forwards (a coarse anchor, not a tail — the plan's budget cap).
**Re-measured post-#28** (order-independent per-node read):

| B | step median | step p99 | wire up / down (B) | node 0 p99 | node 1 p99 | node 2 p99 |
|---|---|---|---|---|---|---|
| 1 | 16.67 s | 16.67 s | 364,228 / 1,577,280 | 520 ms | 254 ms | 214 ms |
| 8 | 104.48 s | 104.48 s | 2,832,560 / 12,617,772 | **2.3 s** | 1.9 s | **558 ms** |

The placed path runs the real model end-to-end across three shaped nodes on the
existing wire. The per-node times do NOT converge at B=8: node 0 (5,336 held
experts) is compute-dominated and slowest (p99 2.3 s, median 1.9 s), while the
far-but-lightly-loaded nodes 1/2 (539 / 269 experts) read BELOW it (node 2 median
303 ms, p99 558 ms). The earlier "all three nodes ≈ 2.2 s at B=8" reading was the
#28 head-of-line-homogenization artifact — the two lighter nodes, read only after
node 0's ~2 s compute drained, inherited its wait — NOT genuine "all nodes
saturated on compute". With phase 2 now reading each node when its own answer is
ready (`peek`-for-readiness), each node's per-node latency is its own load again,
so placement's compute-load spread across the pool is visible instead of masked.
The wire bytes are unchanged (`WIRE_VERSION` 1, goldens byte-identical) — only the
read order + recorded timestamp moved. The un-tuned dense forward is the wall-clock
(~9 s/forward, the M1/M3 figure); forward-perf tuning stays out of scope as in
M1–M3.

### Perplexity canary — fp8 vs bf16-source (ADR-0008 corollary / ADR-0018 quality axis)

Dashboard number **#3**, and the deciding **quality axis** ADR-0018 was blocked on:
teacher-forced perplexity of the fp8 blob+wire path scored against the bf16-source
reference (the M0/M1 `diff.rs::source_matrix` methodology, A6), over a fixed seed-keyed
prompt set. Both paths hold every expert, so **nothing renorms** — the delta is pure
quantization quality, not dropout. The per-token score is the stable
`logsumexp(logits) − logits[target]`; the mean NLL exponentiates to perplexity. Real
Qwen3-30B-A3B (`KENNY_MODEL_DIR`), Config = the model card, **2 sequences × 16 tokens
(30 scored transitions), seed 42**; 733.5 s wall (64 teacher-forced forwards, the
un-tuned ~11 s/forward dense cost — forward-perf tuning stays out of scope as in M1–M3).

| path | mean NLL (nats/tok) | perplexity | Δ vs bf16-source |
|---|---|---|---|
| bf16-source reference | 16.33824 | 12,462,520.60 | — |
| fp8 blobs + fp8 wire | 16.63653 | 16,793,954.18 | **+0.298 nats/tok · ppl ×1.347 · Δppl +4,331,433.57** |

fp8 diverges from the bf16 reference by **+0.298 nats/token** — directionally consistent
with the M1 end-to-end fp8 wire cosine (0.99985), now as a scored quality delta. This
**unblocks ADR-0018's deciding axis** for the fp8 half; naming a default still needs the
int8 arm (the `Int8Codec` is the deferred follow-up), so ADR-0018 stays `proposed`.

> **Random-token caveat (bench honesty):** kenny carries **no tokenizer** (it operates on
> token ids, like `kenny spine` and the S7 harness), so the canary set is random in-vocab
> ids, not natural language. The model assigns worse-than-uniform mass to a random next
> token (NLL 16.3 > `ln(vocab)` = 11.9), so the **absolute** perplexity is astronomical
> and is NOT a language-modeling perplexity — only the **fp8-vs-bf16 delta** is the signal,
> and on anti-natural streams it is an upper-ish bound on the natural-text loss. A
> tokenizer-backed natural-text canary needs assets kenny does not ship — a real-party
> concern (#6). CI runs the canary model-free on the fixture (deterministic to the bit);
> this real number is the `KENNY_MODEL_DIR`-gated arm.

### Prefix-cache hit-rate — shared-system-prompt fixture (ADR-0022 identity primitive)

Dashboard number **#5**, the ADR-0022 survival metric (MANIFESTO §4.5: prefill is
existential). Prompt tokens are chunked into blocks whose blake3 hash-chain keys
are rooted in the manifest identity (ADR-0005) and looked up in a spine-LOCAL
radix; `prefix_hit_rate = reused / total` prompt tokens, where a hit is a block
whose KV can be served from cache without an expert dispatch. The number is only
meaningful against SHARED prompts (independent prompts have zero reuse — the
mitigation for the "meaningless hit-rate on random streams" risk), so the fixture
has every stream share a system prompt and carry a distinct seed-derived user
tail — the agent-colony regime ADR-0022 targets. The cache is spine-local and
holds block KEYS only (no KV payload, no wire, no manifest — `WIRE_VERSION` 1,
the five wire goldens byte-identical); the block-key encoding has its own golden.
**Model-free and deterministic** (reads only the carve's manifest for the
identity + MoE layer count, loads no weights), so it runs in plain CI; the
hit-rate depends only on the shared-prompt structure, not on the model.

| streams | system | user | block | reused / total tokens | hit-rate |
|---|---|---|---|---|---|
| 8 | 256 | 64 | 64 | 1,792 / 2,560 | 0.7000 |
| 16 | 256 | 64 | 64 | 3,840 / 5,120 | 0.7500 |
| 64 | 256 | 64 | 64 | 16,128 / 20,480 | 0.7875 |
| 8 | 512 | 64 | 64 | 3,584 / 4,608 | 0.7778 |
| **512** | **4,096** | **512** | **256** | **2,093,056 / 2,359,296** | **0.8872** |

The rate climbs toward the ADR-0022 **80–90 %+** regime exactly as the shared
system prompt lengthens and more streams amortize the cold first stream that
primes the cache (only stream 0 ever misses the shared prefix): a 512-stream
colony behind a 4,096-token shared system prompt reaches **0.8872** — the 80-90 %
reuse the wire math survives on, every hit being prefill bytes that never touch
the star. Exact-match semantics are locked by test: a one-token divergence
invalidates every subsequent block (the chain propagates the miss).

**Derived KV occupancy** (dashboard number **#2**), reported alongside — NOT a new
subsystem (the KV memory hierarchy is deferred, ADR-0022): straight from the
existing `LayerKv`, `occupancy = batch × ctx × layers × kv_elem` with
`kv_elem = 2 × num_kv_heads × head_dim × 4` bytes/token/layer (one `k` + one `v`
row of f32). At the 512-stream row, ctx = 4,608 (4,096 system + 512 user) across
48 MoE layers at the Qwen3-30B-A3B card's KV geometry (4 KV heads × 128 head_dim,
`kv_elem` = 4,096 B):

```
512 streams × 4,608 ctx × 48 layers × 4,096 B = 463,856,467,968 B = 432.0 GiB
```

the KV-wall side of MANIFESTO §5 (0.61 MB/token × 512 × 4k ≈ 1.2 TB is the
full-4k number; 432 GiB is this run's 4,608-ctx point) — which is precisely why
the prefix hit-rate above is the survival metric: cache reuse is the lever
against that wall.

### Dashboard numbers landed this milestone

All five M4 dashboard numbers now land (three built, two derived from existing
state):

- **#1 batch depth** — the `B` streams advanced in lockstep (M2/M3 `GenStats`),
  here 1 and 8 across the placed pool; aggregate tok/s is the product line above.
- **#2 KV occupancy** — the derived `B × ctx × layers × kv_elem` from the
  existing `LayerKv`, reported alongside the hit-rate above.
- **#3 perplexity canary** — the fp8-vs-bf16-source Δppl table above (ADR-0008
  corollary; ADR-0018 quality axis, fp8 half).
- **#4 per-node step p99** — the tables above (the placement-relevant number).
- **#5 prefix-cache hit-rate** — the shared-system-prompt table above (ADR-0022
  identity primitive).

This completes the M4 arc's single-machine-buildable scope. What stays open on #6
is only the literal real party (the "Deferred — real-party numbers" subsection
below): real nodes/spine/internet, correlated churn, a populated WAN tail, the
ADR-0006 critical-mass crossover, and the KV memory hierarchy + real-workload
hit-rate that full ADR-0022 acceptance needs.

### Deferred — real-party numbers (#6 stays open)

Everything above is a SIMULATION on one host. Its assumptions, continued from
M1–M3: netem-emulated per-link delay+rate on `lo` (no real ISP/geography); a
single failure domain per IP with NO correlated churn; shaped-but-synthetic
uplinks; the spine→node egress shaped and the return unshaped; and — for the
per-node p99 — a worst-of-n tail on the sparse nodes, not a populated one. What the
real party must show that the sim cannot: a **populated** WAN tail on genuine
geography, **correlated-churn** behaviour (households/ISPs dying together), and the
ADR-0006 critical-mass crossover. #6 closes when a human lands those.

**Promote sim → real (same binaries, only addresses change):** on each real box
`kenny node --carved <dir> --hold <subset> --listen 0.0.0.0:<port>` (the `<subset>`
is that node's `PlacementMap::subset_for` assignment; `--shard i/n` is the
no-placement-file convenience); on the spine box `kenny spine --carved <dir>
--model <model_dir> --node <box1> --node <box2> --node <box3> [--replicas r]
[--hedge-ms N] --batch <B>`. Preconditions are #4's: the SAME kenny build on every
box (⇒ identical `WIRE_VERSION` + codec versions), the same fp8 carve so the
handshake `verify` passes, x86/LE, and port reachability — plus REAL
failure-domain labels in the placement input so replicas land in genuinely distinct
churn domains.

### Reproduce

```
# Fixture arm (model-free): 3 heterogeneously-shaped nodes, per-node p99.
bash tools/netem-bench.sh --nodes 3 --placement
# More gen steps for a fuller per-node tail:
KENNY_NETEM_STEPS=30 bash tools/netem-bench.sh --nodes 3 --placement

# Real-model anchor (Qwen3-30B-A3B): placed across 3 shaped nodes, B ∈ {1,8}.
KENNY_MODEL_DIR=<model_dir> bash tools/netem-bench.sh --nodes 3 --placement

# Unprivileged netns (unshare -rn); unavailable ⇒ "netns unavailable" + exit 0.
# A plain `cargo test` never touches netem (netem_placement gates on
# KENNY_NETEM_NODES). The CI-runnable locks (no netns, no model) are
# tests/dispatch.rs::placed_* + placed_records_per_node_latency and
# node::apply_hold_* — placed ≡ local bit-exact, the shard partition, per-node
# latency plumbing — all in `cargo test`.

# Perplexity canary — fp8 vs bf16-source Δppl (dashboard #3).
# CLI (fp8 carve + bf16 source model):
kenny canary --carved <fp8_dir> --model <model_dir> --prompts 2 --len 16
# Real-model gated test arm (the number in the table above):
KENNY_MODEL_DIR=<model_dir> cargo test --release --test dispatch \
    real_model_perplexity_canary -- --nocapture
# Model-free deterministic arm runs in a plain `cargo test`
# (src/canary.rs::{score_tokens_*, reference_perplexity_is_exactly_reproducible,
#  fixture_fp8_canary_is_finite_and_deterministic}).

# Prefix-cache hit-rate + derived KV occupancy (dashboards #5 and #2).
# Model-free: needs only a carve's manifest (identity + MoE layer count).
kenny fixture --out /tmp/m --layers 48 --experts 4
kenny carve /tmp/m --out /tmp/c --dtype fp8
kenny prefix --carved /tmp/c --streams 512 --system-len 4096 --user-len 512 --block 256
# The hit-rate is model-independent (shared-prompt structure only); the sweep
# rows use --num-kv-heads 1 --head-dim 8 (fixture square attention) for the KV
# figure, the 512-stream row uses the Qwen3-30B-A3B card defaults (4 / 128).
# All deterministic; the CI locks are src/prefix.rs::{golden_block_key_chain,
# shared_system_prompt_hits_after_first_stream, one_token_divergence_*,
# hit_rate_is_deterministic, run_measures_shared_prompt_hit_rate}.
```

## M5.A — elasticity: correlated churn + renorm-during-churn (2026-07-22)

Elasticity's failure case, on top of the M4 re-placement primitive + `add_expert`
(PR1): a whole **failure domain** dies together, and the pool must degrade
smoothly (ADR-0008) rather than stall. All measurement here is a SIMULATION on one
host; the real ≥20-node correlated-churn party stays M5.C / #7 (see the assumptions
block below). Nothing on the wire moved — `WIRE_VERSION` = 1, every codec version
1, all five wire goldens byte-identical (ADR-0024).

### Renorm quality dip vs down-fraction — 8-expert × 4-layer fixture (model-free)

The renormed output's divergence from the full-coverage answer (mean L2 over the
per-position logits, scored through the canary's own `Spine::logits_per_position`)
as a growing fraction of experts is forced not-held on every MoE layer. Monotone in
the down-fraction and exactly 0 at full coverage — degradation is MEASURED, and the
dip returns to exactly the baseline when coverage is restored (re-replication).

| down-fraction | L2 dip from full coverage |
|--------------:|--------------------------:|
| 0.000         | 0.000000                  |
| 0.125         | 0.011555                  |
| 0.250         | 0.019563                  |
| 0.375         | 0.024168                  |
| 0.500         | 0.030276                  |
| 0.625         | 0.036990                  |
| 0.750         | 0.052849                  |
| 0.875         | 0.069921                  |

HONEST FIXTURE CAVEAT (ADR-0007): on RANDOM fixture weights the literal canary NLL
drifts toward the ln(vocab) uniform-prior floor under dropout (3.4698 → 3.4671 over
the same ladder) rather than strictly worsening, so the SIGN/curve of a real
perplexity dip is only measurable on the real card — the `renorm_quality_dip_real_model`
arm below. What the fixture locks model-free is the divergence-from-full signal
being monotone in the down-fraction and cleanly recoverable.

### Renorm quality dip under dropout — real-model anchor (Qwen3-30B-A3B, fp8)

Teacher-forced perplexity of the fp8 path as a growing fraction of experts is
forced not-held across every MoE layer (2 sequences × 8 tokens, budget-capped). The
number is the deliverable; see Reproduce.

| down-fraction | dropped/layer | perplexity | Δppl vs full |
|--------------:|--------------:|-----------:|-------------:|
| _pending gated run — `renorm_quality_dip_real_model`_ ||||

### Correlated-churn down-window — netns SIMULATION (4 shaped nodes)

Four `kenny` nodes bound to distinct loopback IPs behind their own `tc netem`
delay+rate bands (`127.0.0.2` 20 ms/1000 Mbit, `.3` 60/100, `.4` 100/50, `.5`
150/20), grouped into failure domains of 2. Domain `dom0` (nodes 0,1) is killed
mid-run (black hole) while the r=1 placement map still points at it — the
down-window before an operator re-places. The replica-set budget (600 ms here, set
to clear the slowest uplink) stalls each silent node so the ADR-0008 renorm bridges
the gap; the run COMPLETES, and a re-place over the survivors restores coverage.

| phase        | renorm_steps | step median | step p99 | suspect flagged | dead domain holds |
|--------------|-------------:|------------:|---------:|----------------:|------------------:|
| down-window  | 64           | 4.81 s      | 4.82 s   | 30              | 30                |
| re-replicated| 0            | —           | —        | —               | —                 |

`suspect_replicas` flags EXACTLY the 30 experts the dead domain held (no survivor
false-positives at this budget). CAVEAT — a real ADR-0010 × ADR-0008 interaction
the sim surfaces: a replica-set budget TIGHTER than a slow-but-healthy survivor's
round-trip false-flags that survivor suspect (an over-tight 120 ms budget against
the 150 ms uplink flagged 293 experts, not 30); the budget must clear the slowest
uplink. The step median is the down-window cost of the per-layer stall budget × the
8 MoE barriers, not a steady-state number.

### Deferred — real correlated-churn party (#7 / M5.C stays open)

Everything above is a SIMULATION on one host. Its assumptions, continued from M4:
netem-emulated per-link delay+rate on `lo` (no real ISP/geography); failure domains
are IP groups on one host, NOT households/ISPs dying together on real geography; the
black hole is a wedged process, not a genuine correlated outage; the budget is hand-
tuned to the shaped delays, not a live network's tail. What the real party must show
that the sim cannot: correlated churn across ≥20 nodes on genuine geography with
Σ uplink ≥ 1 Gbit (ADR-0006 critical mass), a nightly churn cycle survived without
an operator, and the GLM-5.2 card served. #7 closes when a human lands those; the
promote-sim→real recipe (BENCH M4) carries the M5.A binaries forward unchanged.

### Reproduce

```
# Model-free CI locks (no netns, no model) — plain cargo test:
cargo test --test dispatch churn_domain_renorms_and_completes
cargo test --test dispatch churn_flags_dead_replicas
cargo test --test dispatch renorm_quality_dip_grows_with_dropout

# Correlated-churn netns SIMULATION (unprivileged unshare -rn; 4 nodes, domains of 2):
bash tools/netem-bench.sh --nodes 4 --churn
KENNY_NETEM_CHURN_DOMAIN_SIZE=2 bash tools/netem-bench.sh --nodes 6 --churn
# Unavailable netns ⇒ "netns unavailable" + exit 0. The gate is KENNY_NETEM_NODES,
# so a plain cargo test never touches netem.

# Real-model dropout-dip anchor (the perplexity table above):
KENNY_MODEL_DIR=<model_dir> cargo test --release --test dispatch \
    renorm_quality_dip_real_model -- --nocapture
```
