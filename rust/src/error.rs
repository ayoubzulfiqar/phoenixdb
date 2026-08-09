//! Error taxonomy for PhoenixDB.
//!
//! Every error maps to a stable, negative C status code that crosses the FFI
//! boundary. The mapping is part of the public ABI and must never change.

use std::fmt;

/// Stable status codes returned by every `phoenix_*` FFI entry point.
///
/// `0` means success; every failure is negative so callers can test `< 0`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhoenixStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// Unclassified internal failure.
    Error = -1,
    /// A pointer was null, a length exceeded the documented limit, or an
    /// argument was otherwise rejected *before* any dereference took place.
    InvalidArgument = -2,
    /// The requested key does not exist in the caller's snapshot.
    NotFound = -3,
    /// A page failed CRC32 verification, or the file header is not PhoenixDB.
    Corruption = -4,
    /// Underlying I/O failure.
    Io = -5,
    /// Write-write conflict under snapshot isolation; retry the transaction.
    Conflict = -6,
    /// A Rust panic was caught at the FFI boundary (never unwinds into Dart).
    Panic = -7,
    /// The supplied transaction id is unknown, finished, or not owned by this handle.
    TxnNotFound = -8,
    /// A structural limit was reached (page/value cannot be stored).
    Full = -9,
}

/// Internal error type. Converted to [`PhoenixStatus`] at the FFI boundary.
#[derive(Debug)]
pub enum Error {
    /// I/O failure from the filesystem layer.
    Io(std::io::Error),
    /// On-disk structure is not self-consistent (CRC / magic / bounds).
    Corruption(String),
    /// Key is absent from the visible snapshot.
    NotFound,
    /// Caller-supplied argument violates a documented precondition.
    InvalidArgument(String),
    /// Snapshot-isolation write-write conflict.
    Conflict,
    /// Unknown or already-finished transaction id.
    TxnNotFound(u64),
    /// Payload cannot be represented (exceeds structural limits).
    Full(String),
    /// bincode/serde failure while encoding or decoding a WAL record.
    Serialization(String),
    /// Handle has already been closed.
    Closed,
}

impl Error {
    /// Maps this error onto its stable C status code.
    pub fn status(&self) -> PhoenixStatus {
        match self {
            Error::Io(_) => PhoenixStatus::Io,
            Error::Corruption(_) => PhoenixStatus::Corruption,
            Error::NotFound => PhoenixStatus::NotFound,
            Error::InvalidArgument(_) => PhoenixStatus::InvalidArgument,
            Error::Conflict => PhoenixStatus::Conflict,
            Error::TxnNotFound(_) => PhoenixStatus::TxnNotFound,
            Error::Full(_) => PhoenixStatus::Full,
            Error::Serialization(_) => PhoenixStatus::Corruption,
            Error::Closed => PhoenixStatus::InvalidArgument,
        }
    }

    /// Convenience constructor for corruption reports.
    pub fn corrupt(msg: impl Into<String>) -> Self {
        Error::Corruption(msg.into())
    }

    /// Convenience constructor for argument validation failures.
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::InvalidArgument(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Corruption(m) => write!(f, "corruption detected: {m}"),
            Error::NotFound => write!(f, "key not found"),
            Error::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Error::Conflict => write!(f, "write-write conflict; retry transaction"),
            Error::TxnNotFound(id) => write!(f, "unknown or finished transaction {id}"),
            Error::Full(m) => write!(f, "capacity exceeded: {m}"),
            Error::Serialization(m) => write!(f, "serialization failure: {m}"),
            Error::Closed => write!(f, "database handle is closed"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<Box<bincode::ErrorKind>> for Error {
    fn from(e: Box<bincode::ErrorKind>) -> Self {
        Error::Serialization(e.to_string())
    }
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;
