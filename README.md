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

**Status:** M3 complete — **go/no-go call is GO.** Under a `tc netem` simulated
30 ms WAN (loopback in an unprivileged netns — NOT a real second box yet), the
per-layer round-trip barrier amortizes across the batch exactly as the physics
predicts: real-model `Δt_step` sits at the 1.44 s RTT floor and is batch-size
independent (G1/G2), and the compute-free fixture scales **61× over a 64× batch**
once Nagle is off (G3). Under ≤1 % loss the per-layer timeout + hedge bound the
tail quality-safely (G4 pass-with-caveat, ~2.4× the loss-free floor). Full
numbers + the pre-registered verdicts are in [BENCH.md](BENCH.md). M2 held before
it: localhost batching composes on the M1 wire with no new protocol (ADR-0023). M1
still holds: a `kenny spine` and a `kenny node` run as two processes routing every
MoE layer over the wire, the dispatched fp8 path reproducing the in-process path
**bit-for-bit**; M0 carves real Qwen3-30B-A3B into 6,144 content-addressed blobs
and reassembles bit-exactly. Next: **M4 — real WAN** (2–3 physical nodes; the real
second-box LAN numbers stay [#4](../../issues/4)).

- The what and why: [docs/MANIFESTO.md](docs/MANIFESTO.md)
- The decisions: [docs/ADR/](docs/ADR/)
- The work: [issues](../../issues)
