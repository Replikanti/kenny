//! Thin error enum per repo conventions — no panics outside `main`/tests, no
//! error-crate zoo (ADR-0021).

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum Error {
    /// I/O failure with the path that caused it.
    Io { path: PathBuf, source: io::Error },
    /// Structural failure parsing or validating a format (JSON, safetensors,
    /// blob, manifest). The message carries its own context.
    Parse(String),
    /// Bad command-line usage.
    Usage(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn io(path: &Path, source: io::Error) -> Self {
        Error::Io {
            path: path.to_path_buf(),
            source,
        }
    }

    pub fn parse(msg: impl Into<String>) -> Self {
        Error::Parse(msg.into())
    }

    pub fn usage(msg: impl Into<String>) -> Self {
        Error::Usage(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{}: {}", path.display(), source),
            Error::Parse(msg) | Error::Usage(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
