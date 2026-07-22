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

**Status:** M1 complete. A `kenny spine` (pure-Rust Qwen3-30B-A3B forward) and a
`kenny node` run as two processes over localhost, routing every MoE layer's
experts across the wire — and the dispatched fp8 path reproduces the in-process
path **bit-for-bit**. Real end-to-end numbers (tok/s, wire bytes at the socket,
fp8-vs-bf16 cosine) are in [BENCH.md](BENCH.md). M0 still holds: real
Qwen3-30B-A3B carves into 6,144 content-addressed expert blobs in ~67 s and
reassembles bit-exactly (`kenny diff`). Next: M2.

- The what and why: [docs/MANIFESTO.md](docs/MANIFESTO.md)
- The decisions: [docs/ADR/](docs/ADR/)
- The work: [issues](../../issues)
