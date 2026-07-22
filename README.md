# kenny

> Oh my God, they killed a node! …It's fine.

A distributed MoE expert pool where death is a non-event.

kenny runs frontier open-weight MoE models (GLM-5.2, Qwen3) on a pool of
heterogeneous scrap hardware over WAN. In a modern MoE model ~97 % of the
weights are routed experts — tiny, stateless, pure functions. Training already
sharded the model; kenny just hands the shards to whatever you've got, from a
beefy workstation down to the Raspberry Pi in your drawer.

- Pool **nodes** hold expert blobs. They die, rejoin, get replaced — nobody
  cares. CPU-only is first-class.
- One strong **spine** machine holds everything stateful: attention, KV cache,
  routing, sampling. The pool is just a very distributed hot cache.
- Built for **async agent farms** — hundreds of parallel streams, aggregate
  throughput. Explicitly not a chatbot.

**Status:** M5 (single-host scope) complete — elasticity and verification now
hold on top of the M0–M4 stack, all still on a `tc netem` simulated WAN (loopback
in an unprivileged netns — NOT a real second box yet). **M5.A elasticity:** a live
re-placement primitive + `Node::add_expert` make join/leave/migration a
between-step map swap; a whole failure domain can die mid-generation and the pool
renorms and finishes without an operator, flagging exactly the dead replicas
(ADR-0008/ADR-0009). **M5.B verification:** the spine spot-checks a `‰` sample of
node answers against a bf16-source recompute — an honest fp8 pool raises **zero**
false distrust, a byzantine node is caught (ADR-0015, tolerance-based). The GLM-5.2
machinery (shared expert, DSA, MTP) stays design, not untested scaffolding on a
card that lacks it (ADR-0025, ADR-0007). Earlier milestones still hold: **M4**
placed the pool across heterogeneous simulated uplinks with a perplexity canary +
prefix-cache hit-rate; **M3's** go/no-go was **GO** (the per-layer RTT barrier
amortizes across the batch — **61× over a 64× batch** once Nagle is off); **M2**
batched on the M1 wire with no new protocol (ADR-0023); **M1** ran a `kenny spine`
and a `kenny node` as two processes routing every MoE layer **bit-for-bit**; **M0**
carves real Qwen3-30B-A3B into 6,144 content-addressed blobs and reassembles
bit-exactly. Full numbers + the pre-registered verdicts are in [BENCH.md](BENCH.md).
**The open exit (M5.C, [#7](../../issues/7)):** GLM-5.2 on a real ≥20-node party
(Σ uplink ≥ 1 Gbit) with a nightly churn cycle green — real boxes, real geography,
real card. The real second-box LAN numbers stay [#4](../../issues/4).

- The what and why: [docs/MANIFESTO.md](docs/MANIFESTO.md)
- The decisions: [docs/ADR/](docs/ADR/)
- The work: [issues](../../issues)
