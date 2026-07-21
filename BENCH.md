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
