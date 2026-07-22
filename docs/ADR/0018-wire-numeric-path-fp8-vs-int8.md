# ADR-0018: Wire/compute numeric path — fp8 E4M3 vs int8 with deterministic accumulation

- Status: proposed
- Date: 2026-07-21

## Context

Two candidate numeric paths for activations and expert compute:

- **fp8 E4M3** — best quality per bit, but floating-point accumulation order
  differs across architectures (AVX2 / AVX-512 / NEON reorder FMAs), so two
  correct nodes can produce different bits for the same dispatch. Verification
  (ADR-0015) must then be tolerance-based.
- **int8 with int32 accumulation in fixed order** — bit-exact across every
  architecture: same dispatch, same bytes, everywhere. Verification becomes a
  byte compare. Costs some model quality (to be measured, with Hadamard
  rotation + stochastic rounding as mitigations).

This is failure mode 6 in MANIFESTO §5: float nondeterminism versus cheap
verification.

## Decision (plan, to be settled by measurement)

Implement **both** behind the `WireCodec` trait (ADR-0011). At M3, benchmark on
Qwen3-30B (and spot-check on GLM-class dimensions):

- quality: perplexity delta vs bf16 reference, per path, on the canary sets
- throughput: bytes/token and node-side compute cost per path
- verification cost: exact byte-compare (int8) vs tolerance envelope (fp8)

Decide with the table, not with taste. The paths are not mutually exclusive
long-term (e.g. fp8 wire + int8 verification lane), but the default deployment
path gets picked at M3.

### M3 update (2026-07-22) — throughput axis settled by identity, decision deferred

M3 settles only the **throughput** sub-axis of the three above, and it turns out
to be a **non-discriminator**: fp8 E4M3 and int8 are both 1 byte/element
(`WireCodec::elem_bytes == 1` for both), so bytes/token — and therefore the
RTT-driven `t_step` measured under `tc netem` (BENCH "M3") — are **identical**
between the two paths at any RTT. Throughput cannot break the tie.

The **deciding quality axis** (perplexity delta vs bf16 on the canary sets) stays
**blocked on the deferred ADR-0008 perplexity canary**, which does not yet exist.
The standing quality signal carried forward is directional, not a decision: M0
blob cosine (int8 ~8× tighter than fp8 at equal bytes — 1−cos 1.3e-4 vs 1.0e-3)
and M1 end-to-end fp8 wire cosine 0.99985 (BENCH "M0"/"M1").

Consequently this ADR is **amended, not accepted**: the default-path pick **and**
the on-wire `Int8Codec` (a new `codec_id` + goldens + `kenny-format-auditor`
sign-off) are **deferred to the perplexity-canary milestone**. Implementing the
`Int8Codec` at M3 would buy only a non-deciding throughput proxy (identical bytes)
at real consensus-surface cost, so it is deliberately out of M3 scope.

## Consequences

- Until M3, all consensus surfaces (blob format, wire framing) carry explicit
  dtype/codec tags so both paths coexist (already required by ADR-0011).
- The int8 path requires a fixed-order accumulation kernel — deliberately
  boring scalar/SIMD code whose determinism is tested across x86 and ARM in CI.
- Verification design (ADR-0015) inherits its comparison mode from this
  decision.

## Accept when

The ADR-0008 perplexity canary exists and produces a per-path perplexity-delta
table that names the default path. (M3's throughput numbers are in — BENCH.md —
but they are a non-discriminator: equal bytes/element ⇒ equal `t_step`, so the
tie is broken only by the quality axis, which the canary supplies.)

## Alternatives considered

- **Pick fp8 now** — locks verification into tolerance-based checking before
  measuring what determinism would cost.
- **Pick int8 now** — accepts quality loss before measuring it.
- **bf16 on the wire** — 2× the bytes of fp8; dead on arrival given MANIFESTO
  §4.3, useful only as the validation reference.
