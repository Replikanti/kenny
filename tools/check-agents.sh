#!/usr/bin/env bash
# Agent-definition lint: every .claude/agents/*.md needs frontmatter with
# name/description/tools/model/effort, and the name must match the filename.
# Model tiers are deliberately NOT pinned here: kenny tiers per task and
# recalibrates from real outcomes (see CLAUDE.md, Agents & skills).
set -euo pipefail
cd "$(dirname "$0")/.."

status=0
shopt -s nullglob
files=(.claude/agents/*.md)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No agent files — OK"
  exit 0
fi

for f in "${files[@]}"; do
  base=$(basename "$f")
  if ! head -1 "$f" | grep -q '^---$'; then
    echo "FAIL: $base — missing frontmatter"
    status=1
    continue
  fi
  fm=$(sed -n '2,/^---$/p' "$f" | head -n -1)
  for field in name description tools model effort; do
    if ! grep -q "^${field}:" <<<"$fm"; then
      echo "FAIL: $base — missing '${field}' in frontmatter"
      status=1
    fi
  done
  fmname=$(grep '^name:' <<<"$fm" | head -1 | awk '{print $2}')
  if [[ "$fmname" != "${base%.md}" ]]; then
    echo "FAIL: $base — frontmatter name '${fmname}' does not match filename"
    status=1
  fi
done

if [[ $status -eq 0 ]]; then
  echo "OK: ${#files[@]} agent definitions valid"
fi
exit $status
