# ADR-0001: Architecture Decision Records — process and naming

- Status: accepted
- Date: 2026-07-21

## Context

kenny is designed docs-first: the project charter lives in `docs/MANIFESTO.md`,
and every design decision derived from it must be recorded somewhere findable,
diff-able, and reviewable — not buried in chat logs, commit messages, or code
review threads. We adopt Architecture Decision Records (ADRs).

## Decision

- **Location**: `docs/ADR/`, one decision per file.
- **Filename**: `NNNN-kebab-case-slug.md`, where `NNNN` is a sequential integer
  written as a four-digit zero-padded string — decision 1 is `0001`, decision 42
  is `0042`. Numbers are allocated in the PR that adds the ADR (next free number)
  and are never reused, even for rejected ADRs.
- **Numbering is independent of GitHub issue numbers.** Issues reference the ADRs
  they implement (`ADR-NNNN` in the issue body); ADRs do not reference issue
  numbers (ADRs usually exist first).
- **Title line**: `# ADR-NNNN: <imperative or noun-phrase title>`.
- **Required sections**: Status/Date header, `Context`, `Decision`,
  `Consequences` (including the negative ones), `Alternatives considered`.
  Proposed ADRs add `Accept when` — the concrete event or measurement that will
  settle them.
- **Statuses**: `proposed` → `accepted` | `rejected`; an accepted ADR may later
  become `superseded by ADR-MMMM`.
- **Lifecycle**: a `proposed` ADR may be edited freely. An `accepted` ADR is
  immutable except for its status line; changing an accepted decision means
  writing a new ADR that supersedes it. Acceptance happens in the PR that lands
  or first exercises the decision.
- **Language**: English (as all repo content).
- **Index**: none maintained by hand. `ls docs/ADR/` is the list;
  `grep -H "Status" docs/ADR/*.md` is the status index.

## Consequences

- Every "why is it like this?" has a stable, linkable answer.
- PRs and issues gain a shared vocabulary (`ADR-0005` beats "the hashing thing").
- Small friction: deciding anything non-trivial now costs one markdown file.
  That friction is the point.
- Retroactive note: ADR-0002 … ADR-0015 record decisions made during project
  inception (pre-repo design review) and enter as `accepted` on this date;
  ADR-0016 … ADR-0020 enter as `proposed`.

## Alternatives considered

- **Decisions in MANIFESTO only** — one giant file accretes forever, no
  per-decision status, merge conflicts on every design PR.
- **Decisions in issue threads** — not diff-able, not in the repo, rots when
  issues close.
- **Heavyweight ADR tooling (adr-tools, log4brains)** — a dependency to manage a
  directory of markdown files; `ls` and `grep` do the job.
