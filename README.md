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

**Status:** paperwork phase (pre-M0). No code yet — the what and the why are
nailed down first.

- The what and why: [docs/MANIFESTO.md](docs/MANIFESTO.md)
- The decisions: [docs/ADR/](docs/ADR/)
- The work: [issues](../../issues)
