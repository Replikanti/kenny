# kenny — Manifesto

> Oh my God, they killed a node! …It's fine.
> A distributed MoE expert pool where death is a non-event.

This document is the project's north star: what we are building, why it is shaped
this way, and the physics every design must obey. Design decisions derived from it
live in [docs/ADR](ADR/) — one decision per file, process in ADR-0001. Work items
live in GitHub issues; every issue references the ADRs it implements. When code and
this document disagree, flag it — don't silently pick one.

---

## 1. Thesis

kenny runs frontier open-weight MoE models on a pool of heterogeneous scrap
hardware over WAN. It exploits one structural fact: in a modern MoE model, ~97 %
of the weights live in **routed experts**, and each routed expert is a tiny,
stateless, pure function. The model is already sharded — training did it for us.
kenny just distributes the shards (ADR-0002).

- Experts (75 MB blobs for GLM-5.2, 9 MB for Qwen3-30B) live on pool nodes:
  anything from a Raspberry Pi to an old laptop. CPU-only nodes are first-class
  citizens (ADR-0013).
- Everything stateful and sequential (attention, KV cache, router, embeddings)
  lives on one strong machine: the **spine** (ADR-0004).
- The pool is a **distributed hot cache** over the spine's cold copy of the model
  (ADR-0005). Node death loses capacity, never state.

## 2. Success criteria & non-goals

kenny succeeds when a party of 10–20 fiber households (Σ uplink ≥ 1 Gbit) serves a
GLM-class model to an **async agent farm** — hundreds of concurrent, independent,
latency-tolerant streams — at aggregate throughput no single member could reach
alone, and keeps serving through nightly churn without operator attention.

Members are simultaneously suppliers (run `kenny node`) and consumers (call the
gate). Fuller batches make the pool FASTER for everyone (better RTT amortization)
— the opposite of a contended API.

Non-goals, all deliberate and physics-backed (§4):

- **Interactive chat.** Per-stream decode is ~1 tok/s; aggregate throughput is the
  product (ADR-0006).
- **Small-scale mode.** Below ~64 concurrent streams the pool is slower than a
  single local box. The pool has no small-scale mode.
- **Economics / token accounting.** Out of scope for now — technology first. The
  dispatch log gives metering for free later (ADR-0014).
- **Byzantine-resistant open pool (yet).** Trusted parties only until spot-check
  verification lands (ADR-0015).

## 3. Architecture — four roles, one binary

```
kenny carve   — offline: cut a model into content-addressed expert blobs + manifest
kenny node    — expert runner: holds N blobs, answers dispatches (runs on scrap)
kenny spine   — data plane: attention, KV, router, batching, hot cache, sampling
kenny gate    — OpenAI-compatible HTTP API in front of the spine
(orchestrator — control plane: placement, health, migration; a process beside the
 spine for now, logically separate: it must NEVER sit in the data path)
```

Topology: **star** (ADR-0003). Activations flow spine ⇄ nodes directly. Control
(heartbeats, placement, migration) flows orchestrator ⇄ nodes, low-bandwidth, out
of band.

| Component | Holds | RAM class |
|---|---|---|
| node | expert blobs + dispatch buffers | 128 MB … anything (fixed overhead ~10–15 MB + 37.7 MB/expert fp8) |
| spine | always-on weights (~14–18 GB fp8), KV cache (0.61 MB/tok), routing table, heat map, L1 hot-expert cache, cold copy of full model on NVMe (750 GB fp8) | 64 GB dev … 512 GB+ prod, high mem-BW (attention reads resident KV every step: BW ≥ KV_bytes / t_step) |
| gate | HTTP, sessions → spine | trivial |

Trust model: the spine sees plaintext prompts; the pool sees only anonymous
activation vectors. The spine is hosted by whoever owns the workload. Multi-spine
over a shared pool is the designed future (experts are a stateless cache anyone may
query); each spine = SPOF + trust anchor for its own streams only.

## 4. The physics

All design flows from these numbers. They are the quantitative source of truth;
ADRs cite them instead of duplicating them. Re-verify per model release.

### 4.1 Expert = three matrices, nothing else

For layer L, expert E (SwiGLU):

```
model.layers.{L}.mlp.experts.{E}.gate_proj.weight   [moe_intermediate, hidden]
model.layers.{L}.mlp.experts.{E}.up_proj.weight     [moe_intermediate, hidden]
model.layers.{L}.mlp.experts.{E}.down_proj.weight   [hidden, moe_intermediate]

forward:  y = down_proj( silu(gate_proj · x) ⊙ (up_proj · x) )
```

Stateless. No memory between calls. All state (KV cache) lives in attention, on
the spine.

### 4.2 Model cards

**GLM-5.2** (production target; MIT license; HF: `zai-org/GLM-5.2`, repo 1.51 TB bf16)

- ~743B total params, ~39–40B active/token
- 78 layers: first 3 dense FFN (`first_k_dense_replace: 3`), 75 MoE layers
- 256 routed experts/layer, top-8 routed + 1 shared expert (shared fires EVERY
  token → spine)
- hidden_size 6144, moe_intermediate_size 2048
- **Expert = 3 × 6144 × 2048 = 37.75M params = 75.5 MB bf16 / 37.7 MB fp8**
- Routed experts total: 75 × 256 = **19,200** (~725B params, ~725 GB fp8)
- Always-on remainder (attention + DSA indexer, shared experts, dense FFN, routers,
  embeddings/LM head, MTP layer): ~14–18B params ≈ **~14–18 GB fp8** → spine
- Attention: 64 heads / 64 KV heads, head_dim 64 (no MLA compression) + DeepSeek
  Sparse Attention (DSA) with IndexShare; native 1M context
- **KV cache ≈ 0.61 MB/token fp8** (64 × 64 × 2 × 78) — the spine's real appetite
- Built-in MTP layer → speculative decoding (later: amortizes RTT ~4–5×)

**Qwen3-30B-A3B** (dev testbed; Apache 2.0; fits on the dev machine for reference
runs)

- 48 layers, 128 experts/layer, top-8, NO shared expert
- hidden 2048, moe_intermediate 768
- **Expert = 3 × 2048 × 768 = 4.72M params = 9.4 MB bf16 / 4.7 MB fp8**
- Total routed experts: 48 × 128 = **6,144**
- Same tensor-naming schema family as GLM. Everything gets built and validated
  here first (ADR-0007).

### 4.3 Wire cost (fp8 activations, naive protocol)

Activation vector = hidden_size bytes in fp8 (GLM: 6 KB; Qwen: 2 KB).

```
per token, per MoE layer:  dispatch x to ≤8 nodes  +  gather 8 outputs
GLM  : 75 layers × 8 × 6 KB × 2 dir ≈ 7.2 MB/token total (3.6 MB each way)
Qwen : 48 layers × 8 × 2 KB × 2 dir ≈ 1.6 MB/token total
```

Reduction levers (in order of yield): spine-local hot-expert cache (every % hit =
% wire saved), Hadamard rotation + int4 + stochastic rounding (~2× vs fp8),
co-activation placement (x sent once per node touched, cluster co-firing experts →
~2× downstream), ANS entropy coding (+10–20 %). Realistic floor ≈ 2.3 MB/token
GLM-class.

### 4.4 Step time & batch envelope (the systolic pump)

Decode is sequential per stream: 75 synchronization barriers/token. Only batching
across independent streams amortizes it.

```
t_step ≈ Σ_layers ( RTT + max_over_touched_nodes(bytes_i / uplink_i) )
       ≈ 75 × (RTT + tail_transfer)              // RTT 30ms → floor ~2.3 s/step

aggregate tok/s ≈ B / t_step                      // B = batch (streams per step)
B_max ≈ t_step × Σ(node uplinks) × hit_factor / wire_per_token_up
```

Rules that follow:

- **Critical mass**: below ~64 concurrent streams the pool is slower than a single
  local box.
- **Placement equalizes TIME, not bytes**: node payload ∝ node uplink. Slow-uplink
  nodes get the cold Zipf tail (RAM buys coverage, uplink buys throughput — a node
  needs only one of the two currencies to be useful; ADR-0009).
- **Spine link** carries everything twice: ~1 GB/s at B=512 GLM fp8 → 10 Gbit
  ideal, 1 Gbit workable with B≈64–128 + aggressive hot-cache.
- **Go/no-go for a party**: Σ uplinks ≈ 1 Gbit+ (10–20 fiber households).
  DSL-only parties die under critical mass.

### 4.5 Prefill is existential

Agent workloads are input:output ≈ 10:1. A cold 20k-token prompt ≈ 72 GB through
the star each way (GLM fp8). **Prefix-cache hit rate is the survival metric** —
colonies sharing system prompts + tool defs can hit 80–90 %+. Design the KV/prefix
cache before designing anything user-facing.

## 5. Failure modes to design against (ranked by when they hit)

1. **Prefill tsunami** — cold prompts flood the star; prefix cache is the fix (§4.5).
2. **KV wall** — B=512 × 4k ctx = 1.2 TB KV. Int4 KV + NVMe paging (DSA reads
   sparsely) buy ~4×; still: long contexts × high concurrency = pick one.
3. **Critical batch mass** — <64 streams → death spiral (slow → churn → slower).
4. **Correlated churn + tail hell** — failure domains, hedging, wired-only advice
   beyond ~30 nodes (ADR-0009, ADR-0010).
5. **Silent quality rot** — renorm hides dead AND corrupted experts. Canaries or
   blind (ADR-0008).
6. **Float nondeterminism vs verification** — AVX2/AVX-512/NEON reorder FMAs.
   Exact-match checking needs the int8 deterministic kernel (ADR-0018).
7. **Spine SPOF + bus factor** — spine death = all KV gone = re-prefill the world.
   Keep the system boring enough for one maintainer.

**Day-zero dashboard (5 numbers):** prefix-cache hit rate · batch depth · KV
occupancy · per-node step p99 · perplexity canary.

## 6. Roadmap

- **M0 — carve**: cut Qwen3-30B into 6,144 blobs + manifest; reassemble one layer;
  numerically diff forward vs reference. Detail lives in the M0 issues.
- **M1 — one machine, two processes**: spine-sim ⇄ node over localhost; the
  dispatch/gather protocol exists and round-trips.
- **M2 — LAN**: node on a second physical box. Measure tok/s vs batch size.
  Split crates if needed (ADR-0019).
- **M3 — `tc netem` 30 ms**: the go/no-go gate. Tune batch, fp8 wire, hot-cache.
  Hard numbers decide whether WAN is viable. Cheap, brutal, honest.
- **M4 — WAN**: 2–3 real nodes (friends/VPS), Mac-Mini-class spine, Qwen3
  end-to-end.
- **M5 — elasticity**: join/leave, migration, failure domains; then verification
  spot-checks (ADR-0015). Then, and only then, GLM-5.2 on a ~20-node party.

Milestone hygiene: each M ends with a measured number in `BENCH.md` (median + p99,
exact setup, wire bytes counted at the socket, not estimated). No vibes.

## 7. Glossary

Use these words precisely, in code and in docs:

- **blob** — one expert's weights, canonical bytes
- **CID** — blake3 hash of a blob (content address)
- **manifest** — canonical file mapping (layer, expert) → CID + spine tensors;
  blake3 of the manifest = model identity. (Not to be confused with this
  MANIFESTO.)
- **spine** — the strong machine holding attention, KV, router, sampling
- **step** — one systolic batch iteration (all streams advance one token)
- **heat map** — per-expert dispatch frequency from the dispatch log
- **hot / cold** — Zipf head / tail of the heat map
- **carve** — offline cut: model → blobs + manifest
- **renorm** — top-k renormalization over available experts (ADR-0008)
- **hedge** — duplicate dispatch to a replica to cut tail latency (ADR-0010)
- **party** — one deployment: a spine + its pool
