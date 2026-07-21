---
name: adr
description: Draft a new ADR with the correct next number and the ADR-0001 template. Use when a design decision needs recording before or alongside implementation.
arguments: [title]
---

Draft a new ADR titled `$title` per the process in
`docs/ADR/0001-adr-process.md` (that file is normative; this skill is
convenience):

1. Next number = highest existing `docs/ADR/NNNN-*.md` + 1, zero-padded to
   four digits. Numbers are never reused.
2. Create `docs/ADR/NNNN-<kebab-case-slug>.md`:
   - first line `# ADR-NNNN: <title>`
   - `- Status: proposed` and `- Date: <today>` header lines
   - sections `## Context`, `## Decision`, `## Consequences`,
     `## Alternatives considered`, `## Accept when`
3. Content rules: one decision per ADR; cite MANIFESTO sections for numbers
   instead of duplicating them; link related decisions as `ADR-MMMM`;
   consequences must include the negative ones; `## Accept when` names the
   concrete event or measurement that will settle it.
4. Fill the sections from the discussion at hand. Where only the maintainer
   can decide, put an explicit open question rather than a guess.
5. Run `tools/check-adr.sh` and fix anything it flags.
6. Remind the user: the ADR flips to `accepted` in the PR that lands or
   exercises the decision — not now.
