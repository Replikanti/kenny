# ADR-0005: Content-addressed expert blobs; manifest as model identity

- Status: accepted
- Date: 2026-07-21

## Context

The pool is a distributed cache over the spine's cold copy of the model
(MANIFESTO §1). A cache needs keys; nodes need an integrity check for what they
hold and serve; finetunes of a base model share most expert weights and should
share storage and placement.

## Decision

- **blob** = one expert's weights (the three matrices of MANIFESTO §4.1) in a
  fixed binary format, canonical bytes.
- **CID** = blake3 hash of the entire blob. Blobs are stored and requested by CID
  only (`blobs/<cid-prefix>/<cid>`).
- **manifest** = canonical, deterministic, sorted encoding mapping
  (layer, expert) → CID, plus records for every spine-side tensor (name, dtype,
  shape, source shard, byte range, CID) and quantization/codec metadata.
  **blake3(manifest) = the model's identity.**
- The pool is keyed by CID alone: a node neither knows nor cares which model or
  revision an expert belongs to.
- Quantization happens before hashing, centrally at carve time (ADR-0012), so
  all replicas of a CID are bit-identical everywhere.

## Consequences

- Integrity verification and dedup across model revisions come free; a finetune
  that touches 3 % of experts re-uploads 3 % of blobs.
- Cache semantics are clean: any node may hold any CID; placement (ADR-0009) is
  pure assignment, no rewriting.
- Verification (ADR-0015) gets a stable byte-level target.
- The canonical encodings (blob layout, manifest encoding) are consensus
  artifacts: they must be specified byte-exactly, versioned (`codec_version`),
  and locked by golden-hash tests. Changing them changes every CID — a
  deliberate, coordinated event.
- Re-quantization produces a new model identity. Intended: precision is part of
  what you're running.

## Alternatives considered

- **Tuple keys (model, revision, layer, expert)** — no integrity check, no dedup,
  and every finetune is a full new namespace.
- **Torrent-style piece hashing of whole checkpoints** — wrong granularity; the
  natural unit of caching, placement, and dispatch is the expert.
- **Cryptographic hash other than blake3** — blake3 is fast enough to hash a full
  carve in minutes, well-specified, and has a maintained Rust implementation.
