//! Append-only audit log: who accessed what, when.
//!
//! # Why a separate file
//!
//! The audit trail deliberately lives **outside** the WAL. The WAL is truncated
//! at every checkpoint, and its purpose is redo, not history — an auditor's
//! record that vanishes on checkpoint is worthless. This log is append-only and
//! never truncated by the engine.
//!
//! # Format
//!
//! One record per line, newline-delimited, so the file stays greppable and can
//! be tailed by a log shipper without a parser:
//!
//! ```text
//! <ts_millis>\t<outcome>\t<user_id>\t<action>\t<key_hex>\t<source>\t<detail>
//! ```
//!
//! Fields are tab-separated and every field is escaped
//! ([`escape_field`]), so a key containing a tab or newline cannot forge a
//! record boundary — log injection is a real attack when keys are attacker-
//! controlled.
//!
//! Keys are written as hex rather than raw bytes: keys are arbitrary binary,
//! and hex keeps the file text-safe while remaining reversible.
//!
//! # Durability
//!
//! [`AuditLog::append`] writes and, when `sync_each` is set, `fsync`s. Syncing
//! every record is the safe default for a compliance log — an audit entry lost
//! to a crash is exactly the entry an attacker wants lost — but it costs an
//! fsync per operation, so throughput-sensitive deployments can batch and call
//! [`AuditLog::sync`] on their own schedule.

use crate::error::Result;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Longest key prefix recorded, in bytes.
///
/// Full keys can be large; the audit trail only needs enough to identify the
/// record. Truncation is marked with a trailing `~` in the hex field.
pub const MAX_LOGGED_KEY_BYTES: usize = 64;

/// What happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The operation was permitted and performed.
    Allowed,
    /// The operation was refused by access control.
    Denied,
    /// The operation was permitted but failed (I/O, corruption, conflict).
    Failed,
}

impl Outcome {
    /// Stable token written to the log.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Outcome::Allowed => "ALLOW",
            Outcome::Denied => "DENY",
            Outcome::Failed => "FAIL",
        }
    }
}

/// One audit record.
#[derive(Debug, Clone)]
pub struct AuditRecord {
    /// Milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// Whether the operation was allowed, denied, or failed.
    pub outcome: Outcome,
    /// Authenticated principal, or `"<anonymous>"` when RBAC is open.
    pub user_id: String,
    /// Operation name, e.g. `"read"` or `"write"`.
    pub action: String,
    /// Key the operation touched, if any.
    pub key: Option<Vec<u8>>,
    /// Client address in distributed mode; `"local"` for embedded use.
    pub source: String,
    /// Free-form context, e.g. the denial reason.
    pub detail: String,
}

impl AuditRecord {
    /// Creates a record stamped with the current wall-clock time.
    #[must_use]
    pub fn now(outcome: Outcome, user_id: impl Into<String>, action: impl Into<String>) -> Self {
        AuditRecord {
            timestamp_ms: current_millis(),
            outcome,
            user_id: user_id.into(),
            action: action.into(),
            key: None,
            source: "local".to_string(),
            detail: String::new(),
        }
    }

    /// Attaches the key the operation touched.
    #[must_use]
    pub fn with_key(mut self, key: &[u8]) -> Self {
        self.key = Some(key.to_vec());
        self
    }

    /// Attaches the client address.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Attaches free-form context.
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    /// Renders the record as one tab-separated line, without the newline.
    #[must_use]
    pub fn encode(&self) -> String {
        let key_field = match &self.key {
            None => "-".to_string(),
            Some(k) => {
                let shown = k.len().min(MAX_LOGGED_KEY_BYTES);
                let mut hex = String::with_capacity(shown * 2 + 1);
                for b in &k[..shown] {
                    hex.push_str(&format!("{b:02x}"));
                }
                if k.len() > shown {
                    hex.push('~'); // marks truncation
                }
                hex
            }
        };
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.timestamp_ms,
            self.outcome.name(),
            escape_field(&self.user_id),
            escape_field(&self.action),
            key_field,
            escape_field(&self.source),
            escape_field(&self.detail),
        )
    }
}

/// Milliseconds since the Unix epoch, saturating at 0 if the clock predates it.
#[must_use]
pub fn current_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Escapes tabs, newlines and backslashes so a field cannot forge a record.
///
/// Without this, a key or user id containing `\n` could inject a fabricated
/// audit line — the classic log-injection attack.
#[must_use]
pub fn escape_field(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    if out.is_empty() {
        out.push('-'); // keep the column count fixed
    }
    out
}

/// An append-only audit trail.
pub struct AuditLog {
    path: PathBuf,
    writer: BufWriter<File>,
    /// `fsync` after every record.
    sync_each: bool,
    records_written: u64,
}

impl AuditLog {
    /// Opens (creating if needed) the audit log at `path` in append mode.
    ///
    /// The file is opened with `append`, so concurrent writers interleave whole
    /// records rather than overwriting one another.
    pub fn open(path: impl AsRef<Path>, sync_each: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        Ok(AuditLog {
            path,
            writer: BufWriter::new(file),
            sync_each,
            records_written: 0,
        })
    }

    /// Path of the log file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Records written through this handle.
    #[must_use]
    pub fn records_written(&self) -> u64 {
        self.records_written
    }

    /// Appends `record`, syncing when configured to.
    pub fn append(&mut self, record: &AuditRecord) -> Result<()> {
        self.writer.write_all(record.encode().as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.records_written += 1;
        if self.sync_each {
            self.sync()?;
        }
        Ok(())
    }

    /// Convenience: log an allowed operation on `key`.
    pub fn allow(&mut self, user_id: &str, action: &str, key: &[u8]) -> Result<()> {
        self.append(&AuditRecord::now(Outcome::Allowed, user_id, action).with_key(key))
    }

    /// Convenience: log a denial, recording why.
    pub fn deny(&mut self, user_id: &str, action: &str, reason: &str) -> Result<()> {
        self.append(&AuditRecord::now(Outcome::Denied, user_id, action).with_detail(reason))
    }

    /// Flushes buffers and `fsync`s the file.
    pub fn sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Reads every record line back, for tests and offline review.
    pub fn read_lines(path: impl AsRef<Path>) -> Result<Vec<String>> {
        let content = match std::fs::read_to_string(path.as_ref()) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        Ok(content.lines().map(str::to_string).collect())
    }
}

impl Drop for AuditLog {
    fn drop(&mut self) {
        // Best-effort: an audit record buffered at shutdown must still land.
        let _ = self.sync();
    }
}

impl std::fmt::Debug for AuditLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuditLog")
            .field("path", &self.path)
            .field("records_written", &self.records_written)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_appended_one_per_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
        {
            let mut log = AuditLog::open(&path, true).unwrap();
            log.allow("alice", "read", b"key1").unwrap();
            log.allow("bob", "write", b"key2").unwrap();
            log.deny("mallory", "delete", "lacks permission").unwrap();
            assert_eq!(log.records_written(), 3);
        }
        let lines = AuditLog::read_lines(&path).unwrap();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("ALLOW"));
        assert!(lines[0].contains("alice"));
        assert!(lines[0].contains("6b657931"), "key1 in hex");
        assert!(lines[2].contains("DENY"));
        assert!(lines[2].contains("lacks permission"));
    }

    #[test]
    fn reopening_appends_rather_than_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
        {
            let mut log = AuditLog::open(&path, true).unwrap();
            log.allow("a", "read", b"k").unwrap();
        }
        {
            let mut log = AuditLog::open(&path, true).unwrap();
            log.allow("b", "read", b"k").unwrap();
        }
        let lines = AuditLog::read_lines(&path).unwrap();
        assert_eq!(lines.len(), 2, "history must survive reopen");
        assert!(lines[0].contains("\ta\t"));
        assert!(lines[1].contains("\tb\t"));
    }

    #[test]
    fn log_injection_via_newline_is_neutralised() {
        // A user id carrying a newline must not be able to forge a second line.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
        {
            let mut log = AuditLog::open(&path, true).unwrap();
            log.append(
                &AuditRecord::now(Outcome::Allowed, "evil\nALLOW\troot\tadmin", "read")
                    .with_key(b"k"),
            )
            .unwrap();
        }
        let lines = AuditLog::read_lines(&path).unwrap();
        assert_eq!(lines.len(), 1, "injected newline must not create a record");
        assert!(lines[0].contains("\\n"), "newline is escaped");
    }

    #[test]
    fn tab_injection_cannot_forge_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
        {
            let mut log = AuditLog::open(&path, true).unwrap();
            log.append(&AuditRecord::now(Outcome::Allowed, "a\tb\tc", "read"))
                .unwrap();
        }
        let lines = AuditLog::read_lines(&path).unwrap();
        // 7 fields => exactly 6 separator tabs, regardless of field content.
        assert_eq!(lines[0].matches('\t').count(), 6);
        assert!(lines[0].contains("\\t"));
    }

    #[test]
    fn long_keys_are_truncated_and_marked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.log");
        let key = vec![0xAAu8; MAX_LOGGED_KEY_BYTES * 2];
        {
            let mut log = AuditLog::open(&path, true).unwrap();
            log.allow("a", "read", &key).unwrap();
        }
        let lines = AuditLog::read_lines(&path).unwrap();
        assert!(lines[0].contains('~'), "truncation must be marked");
        let hex_field = lines[0].split('\t').nth(4).unwrap();
        assert_eq!(hex_field.len(), MAX_LOGGED_KEY_BYTES * 2 + 1);
    }

    #[test]
    fn binary_keys_are_hex_encoded() {
        let record = AuditRecord::now(Outcome::Allowed, "u", "read").with_key(&[0x00, 0xFF, 0x10]);
        let line = record.encode();
        assert!(line.contains("00ff10"));
    }

    #[test]
    fn absent_key_renders_as_a_placeholder() {
        let line = AuditRecord::now(Outcome::Failed, "u", "checkpoint").encode();
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 7);
        assert_eq!(fields[4], "-", "missing key keeps the column");
        assert_eq!(fields[1], "FAIL");
    }

    #[test]
    fn escape_handles_every_special_character() {
        assert_eq!(escape_field("plain"), "plain");
        assert_eq!(escape_field("a\tb"), "a\\tb");
        assert_eq!(escape_field("a\nb"), "a\\nb");
        assert_eq!(escape_field("a\r\nb"), "a\\r\\nb");
        assert_eq!(escape_field("a\\b"), "a\\\\b");
        assert_eq!(escape_field(""), "-", "empty field keeps the column count");
    }

    #[test]
    fn source_is_recorded_for_distributed_mode() {
        let line = AuditRecord::now(Outcome::Allowed, "u", "read")
            .with_source("10.0.0.7:5432")
            .encode();
        assert!(line.contains("10.0.0.7:5432"));
    }

    #[test]
    fn missing_file_reads_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let lines = AuditLog::read_lines(dir.path().join("nope.log")).unwrap();
        assert!(lines.is_empty());
    }

    #[test]
    fn timestamps_are_monotonic_within_a_run() {
        let a = AuditRecord::now(Outcome::Allowed, "u", "read");
        let b = AuditRecord::now(Outcome::Allowed, "u", "read");
        assert!(b.timestamp_ms >= a.timestamp_ms);
        assert!(a.timestamp_ms > 1_600_000_000_000, "clock looks unset");
    }
}
