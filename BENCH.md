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
