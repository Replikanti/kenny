# ADR-0017: JSON and serde policy — hand-rolled minimal parsing

- Status: proposed
- Date: 2026-07-21

## Context

kenny's entire JSON surface is: (1) reading safetensors headers and
`model.safetensors.index.json` — flat, well-specified structures of objects,
strings, and unsigned integers; (2) writing (and re-reading) the canonical
manifest, whose blake3 hash is the model identity (ADR-0005) and therefore
needs byte-exact control of the encoding anyway. `serde` + `serde_json` is the
convenient default but a heavyweight dependency tree for a codebase that
prizes a small, auditable footprint — and canonical output would still have to
be hand-managed on top of it.

## Decision (leaning, to be confirmed in the first carve PR)

Hand-roll one small JSON module used for both surfaces:

- **Parser**: strict recursive-descent over a documented JSON subset — objects,
  arrays, strings (full escape handling incl. `\uXXXX` surrogate pairs),
  unsigned integers; rejects floats, duplicate keys, and trailing garbage with
  clear errors. Bounded depth. On the order of ~200 lines.
- **Canonical writer**: sorted keys, no whitespace, minimal escapes, integers in
  decimal — same Value in, same bytes out, always. The manifest identity hashes
  these bytes; golden-hash tests lock the format.

`serde_json` remains acceptable as a dev/build-only dependency (tools, tests)
if a concrete need appears; it stays out of the shipped binary.

## Consequences

- Zero serde in the dependency tree; every byte of the consensus-critical
  encoding is in-repo and reviewable.
- The parser's strictness doubles as validation for safetensors headers.
- Cost: we own correctness. Mitigated by the subset's small size, exhaustive
  unit tests, and golden hashes on real headers.
- If a future surface genuinely needs general JSON (floats, arbitrary
  documents), that's a new decision, not a silent widening of this parser.

## Accept when

The first carve PR lands with the module, its tests, and golden manifest
hashes.

## Alternatives considered

- **`serde` + `serde_json` in the binary** — largest dependency in the tree for
  two flat schemas; canonical encoding must be hand-controlled regardless.
- **`safetensors` crate** — pulls serde_json transitively; parsing the header is
  the easy 20 % of what carve does with the file.
- **Binary manifest (TLV) instead of canonical JSON** — loses human
  inspectability of the model-identity artifact; revisit only if canonical JSON
  proves error-prone in practice.
