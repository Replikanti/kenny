//! kenny — distributed MoE expert pool. M0: offline carve tooling.
//!
//! Read `docs/MANIFESTO.md` and `docs/ADR/` before touching formats: the blob
//! (`src/blob.rs`) and manifest (`src/manifest.rs`) encodings are consensus
//! surfaces (ADR-0005) — CIDs hash their exact bytes, so byte-layout changes
//! are format-version events with an ADR, never refactors.

pub mod bf16;
pub mod blob;
pub mod carve;
pub mod cli;
pub mod diff;
pub mod error;
pub mod expert;
pub mod fixture;
pub mod fp8;
pub mod json;
pub mod manifest;
pub mod natsort;
pub mod node;
pub mod quant;
pub mod rng;
pub mod safetensors;
pub mod wire;

pub use error::{Error, Result};
