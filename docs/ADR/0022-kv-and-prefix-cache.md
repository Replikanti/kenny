# ADR-0022: KV and prefix cache — content-addressed blocks, tiered, decode-first admission

- Status: proposed
- Date: 2026-07-22

## Context

Prefill is existential (MANIFESTO §4.5): agent workloads run input:output ≈
10:1, and a cold 20k-token prompt costs ≈ 72 GB through the star each way at
GLM-class fp8. The KV wall (MANIFESTO §5, failure mode 2) adds the storage
side: 0.61 MB/token means B=512 × 4k context = 1.2 TB of KV. Prefix-cache hit
rate is the survival metric — agent colonies sharing system prompts and tool
definitions can reach 80–90 %+ reuse. This design must land before anything
user-facing (the gate, M4); all state stays on the spine (ADR-0004).

## Decision

- **Prefix identity is content addressing over token blocks.** Prompt tokens
  are chunked into fixed-size blocks (block size a tunable constant, order of
  256 tokens). Each block's key is a blake3 hash chain:
  `key_n = blake3(model_identity ++ key_{n-1} ++ canonical_encoding(tokens_n))`,
  rooted in the manifest identity (ADR-0005). A prefix is identified by its
  last block key; two streams sharing a prompt prefix therefore share KV
  blocks automatically, with zero client coordination — the same dedup-by-
  construction idiom the blob store uses. Lookup is a radix structure over
  block keys on the spine.
- **Spine-side only.** KV blocks live in the spine's memory hierarchy; the
  pool never sees tokens, keys, or KV (trust model of ADR-0003/0004).
- **Tiered storage.** L0: RAM, fp8 KV — active decode. L1: RAM, int4 KV —
  warm reusable prefixes (~2× capacity for a quality cost paid only on
  reuse). L2: NVMe — cold prefixes; GLM's DSA attention reads KV sparsely,
  which is what makes page-in-on-demand viable. Eviction is LRU weighted by
  reuse count × recompute cost, so a system-prompt block with hundreds of
  streams behind it effectively never evicts.
- **Prefill admission is decode-first.** Decode steps have priority — the
  batch is the product (ADR-0006). Prefill runs chunked (order of 512 tokens)
  in the step's slack, and prefill tokens draw from the same per-client
  budget as decode (ADR-0014): one client's cold 20k prompt cannot starve
  everyone else's decode. An idle party gives prefill the whole pipe.
- **The dashboard number.** `prefix_hit_rate = reused_prompt_tokens /
  total_prompt_tokens` over a sliding window, where "reused" means the
  token's KV block was served from any tier without expert dispatch.
  Recorded per client and aggregate, derived from the dispatch log (which
  gains prefill/decode labeling — the same records ADR-0014 meters from).
- **M4 lands (rescoped single-machine slice, 2026-07-22)**: the IDENTITY
  PRIMITIVE — block hashing (the model-identity-rooted blake3 chain) + the radix
  lookup — and the `prefix_hit_rate` metric, measured on a shared-system-prompt
  fixture so the number is real and non-zero (`src/prefix.rs`, `kenny prefix`).
  **Deferred to the real party (#6)**: KV BLOCK STORAGE + tiering (the fp8 RAM
  tier, int4 KV, NVMe paging, DSA-driven sparse page-in), weighted-LRU eviction,
  and chunked decode-first prefill admission — every one is only measurable
  against a real memory-bound, concurrency-bound serving loop with genuinely
  shared prompts, which the M4 spine-sim (synthetic single-host streams) cannot
  exercise, and cross-spine prefix sharing (a multi-spine future). **KV
  occupancy** (dashboard #2) is reported now as a DERIVED number
  (`batch × ctx × layers × kv_elem` from the existing `LayerKv`), not a new
  subsystem.

## Consequences

- Colonies with disciplined shared system prompts get the 80–90 % hit rates
  the wire math survives on; every hit is prefill bytes that never touch the
  star.
- Exact-match semantics: a one-token difference invalidates every subsequent
  block. Accepted — that is precisely the pressure toward stable system
  prompts, and block size tunes the granularity/overhead trade-off.
- Every miss has a price tag in wire bytes, so retention policy becomes a
  measurable economic decision instead of a guess.
- The spine pays with RAM and bookkeeping complexity; the radix structure and
  eviction weights are new state, but all of it is cache — losing it costs
  recomputation, never correctness (consistent with ADR-0004's invariant).

## Alternatives considered

- **Paged block tables without content addressing (vLLM-style)** — solves KV
  fragmentation but not cross-stream dedup; content keys buy both for one
  hashing pass.
- **Whole-prompt exact-match keys** — all-or-nothing reuse; loses sharing
  between prompt variants that diverge only in their tail.
- **Client-declared cache-control ids** — pushes correctness onto clients and
  invites poisoning; content addressing makes reuse automatic and safe.
  Explicit hints may return later as an optimization, never as the
  correctness mechanism.
- **No prefix cache at M4** — MANIFESTO §4.5 arithmetic says the party dies
  under its own prefill; rejected by the numbers.

## Status note (M4, 2026-07-22)

This ADR stays `proposed`. What landed is the spine-LOCAL identity primitive and
the hit-rate meter only: `PrefixCache` hashes prompt blocks into the
model-identity-rooted chain, looks them up in a radix over the chained keys
(content addressing collapses the trie into membership), and reports
`prefix_hit_rate = reused / total` prompt tokens. It holds block KEYS, not KV
payload — so it touches no consensus surface (no `WIRE_VERSION`, codec, or
manifest change; the block-key encoding is a spine-local canonical encoding with
its own golden). On a shared-system-prompt fixture the meter reaches the 80–90 %
regime this ADR's arithmetic survives on (BENCH "M4 — prefix-cache hit-rate").
The KV memory hierarchy (storage, tiering, eviction, admission) is deferred to a
real serving loop; full acceptance is the real agent workload below (#6).

## Accept when

M4 lands the spine-side prefix cache and the day-zero dashboard reports a
measured prefix-cache hit rate on a real agent workload (the real party, #6 —
the identity primitive + hit-rate meter land now; the KV memory hierarchy and
the real-workload number are the remaining acceptance gate).
