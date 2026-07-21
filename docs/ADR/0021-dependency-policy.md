# ADR-0021: Dependency policy — leaf crates only, framework crates banned by default

- Status: accepted
- Date: 2026-07-21

## Context

Dependencies are not free: they shape the architecture (an async runtime makes
everything async, a serialization framework dictates data modeling), they are
supply-chain surface on a project whose artifacts are consensus-critical
(ADR-0005: canonical bits, golden hashes), and they bloat the node binary that
is supposed to run on scrap (ADR-0013). The repo conventions already require a
justification comment per dependency; a convention without enforcement drifts
the first time something is convenient.

## Decision

- **Leaf vs framework.** A *leaf* crate does one thing behind a narrow API
  (`blake3`, `memmap2`) — acceptable with a justification comment in
  `Cargo.toml`. A *framework* crate dictates program shape or pulls a large
  tree — banned by default; introducing one requires its own ADR.
- **Hard denylist, CI-enforced.** `deny.toml` (`cargo deny check` in the
  required `audit` job) bans: async runtimes (`tokio`, `async-std`, `smol` —
  ADR-0016), the serde family (`serde`, `serde_derive`, `serde_json` —
  ADR-0017), error-macro crates (`anyhow`, `thiserror` — thin custom enums are
  the convention), CLI frameworks (`clap` — args are hand-rolled), `rand`
  (determinism wants the seeded hand-rolled RNG), `openssl-sys` (TLS, if it
  ever comes, is rustls — ADR-0016).
- **License allowlist**: MIT, Apache-2.0 (incl. the LLVM-exception variant),
  CC0-1.0. Anything else fails CI and forces a conscious decision.
- **Dev/build dependencies are exempt from the bans** (`exclude-dev`) but not
  from review or the justification comment: dev tooling may be looser
  (ADR-0017 allows dev-only serde_json), the shipped binary may not.
- **Exception path**: lifting a ban = editing `deny.toml` in the same PR as
  the ADR that authorizes it (a new ADR, or one superseding this one). A
  `deny.toml` diff is a consensus-surface change and always gets adversarial
  review.

## Consequences

- Some wheels get reinvented deliberately (a ~200-line JSON subset, 15-line
  bf16 conversion, SplitMix64) — correctness is owned in-repo, carried by
  exhaustive tests and golden hashes rather than by upstream popularity.
- Future friction is intentional: when the gate (M4+) wants HTTP, the choice
  between a minimal hand-rolled HTTP/1.1 and lifting a ban must be argued in
  an ADR with numbers, not solved by `cargo add`.
- The dependency tree stays auditable end to end; `cargo deny` output in CI is
  the proof, on every PR.
- Tools that run in CI but are not dependencies (the cargo-deny and
  cargo-audit binaries themselves) are outside this policy.

## Alternatives considered

- **Justification comments only (status quo ante)** — drifts under time
  pressure; the first tokio arrives as a transitive dependency of something
  convenient.
- **Strict allowlist (`bans.allow`)** — every transitive leaf (arrayvec,
  cfg-if, …) needs registering; high-friction bookkeeping with no additional
  safety over denylist + license gate + review.
- **Vendoring everything** — maximal audit control, unreasonable maintenance
  for a solo-maintainer project; revisit only if supply-chain requirements
  harden (the ADR-0015 verification era).
