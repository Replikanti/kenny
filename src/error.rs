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
    /// A read exceeded its deadline (a `TcpStream::set_read_timeout` fired
    /// mid-frame). Its own variant so the per-layer timeout (ADR-0010) can tell a
    /// straggler apart from a real transport fault — a timeout drops the layer to
    /// the ADR-0008 renorm, any other error still tears the session down.
    Timeout,
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

    /// A read hit its deadline (the per-layer timeout knob, ADR-0010).
    pub fn timeout() -> Self {
        Error::Timeout
    }

    /// Whether this is a read-deadline timeout (vs a real transport fault).
    pub fn is_timeout(&self) -> bool {
        matches!(self, Error::Timeout)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io { path, source } => write!(f, "{}: {}", path.display(), source),
            Error::Parse(msg) | Error::Usage(msg) => f.write_str(msg),
            Error::Timeout => f.write_str("wire: read timed out (per-layer deadline)"),
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
