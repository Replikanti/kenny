#!/usr/bin/env bash
# Mechanical ADR lint per ADR-0001 — filename, numbering, title, status, date,
# required sections, and cross-reference resolution. The semantic layer
# (arithmetic, doc<->code drift) belongs to the kenny-docs-auditor agent, not
# to this script.
set -euo pipefail
cd "$(dirname "$0")/.."

status=0
fail() {
  echo "FAIL: $*"
  status=1
}

[[ -f docs/MANIFESTO.md ]] || fail "docs/MANIFESTO.md missing"

shopt -s nullglob
files=(docs/ADR/*.md)
[[ ${#files[@]} -gt 0 ]] || fail "no ADRs found in docs/ADR/"

name_re='^([0-9]{4})-[a-z0-9-]+\.md$'
status_re='^- Status: (proposed|accepted|rejected|superseded by ADR-[0-9]{4})$'
date_re='^- Date: [0-9]{4}-[0-9]{2}-[0-9]{2}$'

declare -A by_num=()
for f in "${files[@]}"; do
  base=$(basename "$f")
  if [[ ! "$base" =~ $name_re ]]; then
    fail "$base: filename must be NNNN-kebab-slug.md"
    continue
  fi
  num="${BASH_REMATCH[1]}"
  [[ -n "${by_num[$num]:-}" ]] && fail "$base: number collides with ${by_num[$num]}"
  by_num[$num]="$base"

  head -1 "$f" | grep -qE "^# ADR-${num}: " \
    || fail "$base: first line must be '# ADR-${num}: <title>'"

  st=$(grep -E '^- Status: ' "$f" | head -1 || true)
  [[ "$st" =~ $status_re ]] || fail "$base: bad or missing '- Status:' line"
  grep -qE "$date_re" "$f" || fail "$base: bad or missing '- Date:' line"

  for sec in '## Context' '## Decision' '## Consequences' '## Alternatives considered'; do
    grep -qE "^${sec}" "$f" || fail "$base: missing section '${sec}'"
  done
  if [[ "$st" == "- Status: proposed" ]]; then
    grep -qE '^## Accept when' "$f" || fail "$base: proposed ADR missing '## Accept when'"
  fi
done

# Numbers are sequential from 0001 and never reused (rejected ADRs keep their
# file), so the set must be contiguous.
n=${#by_num[@]}
for ((i = 1; i <= n; i++)); do
  key=$(printf '%04d' "$i")
  [[ -n "${by_num[$key]:-}" ]] || fail "numbering gap: ADR-${key} missing"
done

# Every ADR-NNNN mention anywhere in the docs must resolve to a real file.
refs=$(grep -rhoE 'ADR-[0-9]{4}' docs CLAUDE.md README.md 2>/dev/null | sort -u || true)
for r in $refs; do
  key="${r#ADR-}"
  [[ -n "${by_num[$key]:-}" ]] || fail "dangling reference ${r} (no docs/ADR/${key}-*.md)"
done

if [[ $status -eq 0 ]]; then
  echo "OK: ${#files[@]} ADRs — numbering contiguous, structure valid, all references resolve"
fi
exit $status
