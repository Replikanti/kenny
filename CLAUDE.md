# CLAUDE.md — kenny

kenny runs frontier open-weight MoE models on a pool of heterogeneous scrap
hardware over WAN: routed experts (~97 % of the weights) are distributed as
content-addressed stateless blobs; everything stateful (attention, KV cache,
routing, sampling) stays on one strong "spine" machine. Node death loses
capacity, never state.

## Doc map — read before working

- **`docs/MANIFESTO.md`** — north star: goals, non-goals, architecture, the
  physics (ALL load-bearing numbers live there), failure modes, roadmap,
  glossary. The quantitative source of truth.
- **`docs/ADR/`** — every design decision, one per file. Process and naming:
  `docs/ADR/0001-adr-process.md`. Status index:
  `grep -H "Status" docs/ADR/*.md`.
- **GitHub issues** — all work items; each references the ADRs it implements.
- **`tools/`** — repo lint scripts run by CI (`check-adr.sh`, `check-agents.sh`).
- **`BENCH.md`** — measured milestone numbers (exists from M1 on).

When code and docs disagree, flag it — don't silently pick one.

## Workflow

- **Docs first.** A design decision gets an ADR (`proposed`) before or with the
  PR that implements it; that PR flips it to `accepted`. Decisions never live
  only in review threads or commit messages.
- Work happens on branches/worktrees + PRs — main is branch-protected: PR
  required, `check` + `audit` green, branch up to date, linear history. Never
  merge red.
- **Every PR closes its issue.** Default: the PR body carries `Closes #N` and
  the merge closes the issue. A PR with no reasonable issue — or one that must
  not auto-close it — declares a `No-Issue: <reason>` line in the body instead.
  The `check` job enforces one or the other.
- One issue per working session.
- Every milestone ends with measured numbers in `BENCH.md`: median + p99, exact
  topology, wire bytes counted at the socket, not estimated. No vibes.

## CI & dependency policy

- CI (`.github/workflows/ci.yml`) runs two required checks on every PR:
  `check` (ADR + agent lint, PR-issue linkage; fmt/clippy/tests once Rust code
  exists) and `audit` (cargo-audit + `cargo deny check`; pre-code it verifies
  the policy file is armed). Path filters are deliberately absent — required
  checks must always report.
- Dependency policy is ADR-0021 and it is enforced, not advisory: `deny.toml`
  bans framework crates (tokio, serde, clap, rand, anyhow, …) and non-permissive
  licenses. Lifting a ban = editing `deny.toml` in the same PR as the ADR that
  authorizes it.

## Agents & skills

- `.claude/agents/kenny-format-auditor` — run on every PR touching
  consensus-critical encodings (blob/manifest/json/wire code, golden hashes,
  `deny.toml`).
- `.claude/agents/kenny-docs-auditor` — run per milestone or on MANIFESTO/
  ADR-heavy PRs; recomputes the physics arithmetic, hunts doc↔code drift.
- Model/effort tiers in agent frontmatter are **uncalibrated priors** — this
  repo has no benchmark history yet. Log every observed miss or overkill in
  the PR/issue thread and recalibrate from real task history after M1; do not
  tune tiers by feel.
- `/adr <title>` — drafts the next ADR with correct numbering + template.

## Code conventions

- Rust stable, edition 2024. Every dependency justified in a `Cargo.toml`
  comment (dependency policy: ADR-0021; JSON specifics: ADR-0017).
- Sync-first; async only via ADR-0016 with a measured reason.
- Errors: no panics outside `main`/tests; thin custom error enums, no
  error-crate zoo.
- Tests run on synthetic fixtures — CI never downloads a model. Real-model
  tests are gated behind the `KENNY_MODEL_DIR` env var.
- Determinism is a feature: same manifest + same wire bytes ⇒ same output where
  the numeric path allows (ADR-0018). Consensus-critical encodings are locked
  by golden-hash tests.
- Language: English in code, commits, PRs, issues; chat with the maintainer is
  Czech.
- Naming: use the glossary (MANIFESTO §7) precisely — blob, CID, manifest,
  spine, step, heat map, hot/cold, carve, renorm, hedge, party.
- README stays meme-forward; MANIFESTO stays factual; keep both current.
