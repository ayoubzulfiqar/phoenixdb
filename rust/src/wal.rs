//! Write-Ahead Log with per-record CRC32 framing.
//!
//! # Frame format
//!
//! ```text
//! [ len u32 ][ crc32 u32 ][ bincode(WalRecord) ... ]
//! ```
//!
//! `crc32` covers the payload bytes only. Recovery stops at the first frame
//! that is short, over-long, or fails its checksum — a torn tail write is
//! normal after a crash and must not be treated as corruption of the whole log.
//!
//! # Durability protocol
//!
//! 1. `Begin` … `Insert`/`Delete` … `Commit` are appended.
//! 2. [`Wal::commit`] appends the `Commit` record **and calls `sync_all`**, so a
//!    transaction is durable the instant the call returns.
//! 3. Only a committed transaction is replayed by [`Wal::recover`]; records
//!    belonging to a transaction with no `Commit` are discarded.
//! 4. After the tree is flushed, [`Wal::checkpoint`] truncates the log.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Refuse to allocate for a frame larger than this (guards a corrupt length).
const MAX_RECORD_BYTES: u32 = 64 * 1024 * 1024;

/// One durable log record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WalRecord {
    /// A transaction has started.
    Begin {
        /// Transaction identifier.
        txn_id: u64,
    },
    /// A key/value pair was written by `txn_id`.
    Insert {
        /// Transaction identifier.
        txn_id: u64,
        /// Key bytes.
        key: Vec<u8>,
        /// Value bytes.
        value: Vec<u8>,
    },
    /// A key was deleted by `txn_id`.
    Delete {
        /// Transaction identifier.
        txn_id: u64,
        /// Key bytes.
        key: Vec<u8>,
    },
    /// `txn_id` committed at `commit_ts`. Everything before it is durable.
    Commit {
        /// Transaction identifier.
        txn_id: u64,
        /// MVCC commit timestamp.
        commit_ts: u64,
    },
    /// `txn_id` was rolled back; its records must be ignored on replay.
    Rollback {
        /// Transaction identifier.
        txn_id: u64,
    },
    /// The tree was flushed up to `tree_ts`; earlier records are redundant.
    Checkpoint {
        /// Timestamp durably reflected in the B+Tree.
        tree_ts: u64,
    },
}

impl WalRecord {
    /// Transaction this record belongs to, if any.
    #[must_use]
    pub fn txn_id(&self) -> Option<u64> {
        match self {
            WalRecord::Begin { txn_id }
            | WalRecord::Insert { txn_id, .. }
            | WalRecord::Delete { txn_id, .. }
            | WalRecord::Commit { txn_id, .. }
            | WalRecord::Rollback { txn_id } => Some(*txn_id),
            WalRecord::Checkpoint { .. } => None,
        }
    }
}

/// The mutation applied by one committed transaction, in log order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveredOp {
    /// Write this key/value pair.
    Insert(Vec<u8>, Vec<u8>),
    /// Remove this key.
    Delete(Vec<u8>),
}

/// The result of scanning the log after a crash.
#[derive(Debug, Default)]
pub struct Recovery {
    /// Operations from committed transactions, ordered `(commit_ts, ops)`.
    pub committed: Vec<(u64, Vec<RecoveredOp>)>,
    /// Highest transaction id observed (so ids are never reused).
    pub max_txn_id: u64,
    /// Highest commit timestamp observed.
    pub max_commit_ts: u64,
    /// Frames discarded because of a torn or corrupt tail.
    pub truncated_bytes: u64,
}

/// Append-only write-ahead log.
pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
    /// Bytes appended since the last checkpoint.
    bytes_written: u64,
    lsn: u64,
}

impl Wal {
    /// Opens or creates the log at `path`, positioning the cursor at the end.
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let len = file.metadata()?.len();
        let mut writer = BufWriter::new(file);
        writer.seek(SeekFrom::End(0))?;
        Ok(Wal {
            path: path.to_path_buf(),
            writer,
            bytes_written: len,
            lsn: 0,
        })
    }

    /// Current log sequence number (records appended since open).
    #[must_use]
    pub fn lsn(&self) -> u64 {
        self.lsn
    }

    /// Bytes currently in the log.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.bytes_written
    }

    /// Path of the log file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends a record **without** syncing. Returns its LSN.
    pub fn append(&mut self, record: &WalRecord) -> Result<u64> {
        let payload = bincode::serialize(record)?;
        if payload.len() as u64 > MAX_RECORD_BYTES as u64 {
            return Err(Error::Full(format!(
                "WAL record of {} bytes exceeds the {MAX_RECORD_BYTES}-byte limit",
                payload.len()
            )));
        }
        let crc = crc32fast::hash(&payload);
        self.writer
            .write_all(&(payload.len() as u32).to_le_bytes())?;
        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.bytes_written += 8 + payload.len() as u64;
        self.lsn += 1;
        Ok(self.lsn)
    }

    /// Appends `Commit` and forces it to stable storage.
    ///
    /// This is the durability point: when it returns `Ok`, the transaction
    /// survives a power loss.
    pub fn commit(&mut self, txn_id: u64, commit_ts: u64) -> Result<u64> {
        let lsn = self.append(&WalRecord::Commit { txn_id, commit_ts })?;
        self.sync()?;
        Ok(lsn)
    }

    /// Flushes user-space buffers and calls `sync_all`.
    pub fn sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Records a checkpoint and truncates the log.
    ///
    /// Only call this once the tree is durable on disk; the log is the sole
    /// record of committed work until then.
    pub fn checkpoint(&mut self, tree_ts: u64) -> Result<()> {
        self.writer.flush()?;
        let file = self.writer.get_mut();
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        file.sync_all()?;
        self.bytes_written = 0;
        self.append(&WalRecord::Checkpoint { tree_ts })?;
        self.sync()
    }

    /// Scans the log and returns the redo set for committed transactions.
    ///
    /// A torn tail (partial frame or bad CRC at the end) is truncated rather
    /// than reported as an error: that is the expected state after a crash.
    pub fn recover(path: &Path) -> Result<Recovery> {
        let mut recovery = Recovery::default();
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(recovery),
            Err(e) => return Err(Error::Io(e)),
        };
        let total = file.metadata()?.len();
        if total == 0 {
            return Ok(recovery);
        }
        let mut bytes = Vec::with_capacity(total as usize);
        file.read_to_end(&mut bytes)?;

        let mut cursor = 0usize;
        let mut records: Vec<WalRecord> = Vec::new();
        let mut good_bytes = 0usize;

        while cursor + 8 <= bytes.len() {
            let len = u32::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ]);
            let crc = u32::from_le_bytes([
                bytes[cursor + 4],
                bytes[cursor + 5],
                bytes[cursor + 6],
                bytes[cursor + 7],
            ]);
            if len == 0 || len > MAX_RECORD_BYTES {
                break; // corrupt length: stop, treat the rest as torn
            }
            let start = cursor + 8;
            let end = match start.checked_add(len as usize) {
                Some(e) if e <= bytes.len() => e,
                _ => break, // truncated tail
            };
            let payload = &bytes[start..end];
            if crc32fast::hash(payload) != crc {
                break; // torn or corrupted frame
            }
            match bincode::deserialize::<WalRecord>(payload) {
                Ok(rec) => records.push(rec),
                Err(_) => break,
            }
            cursor = end;
            good_bytes = end;
        }
        recovery.truncated_bytes = total - good_bytes as u64;

        // Pass 1: which transactions committed, and when?
        let mut commits: HashMap<u64, u64> = HashMap::new();
        let mut rolled_back: HashSet<u64> = HashSet::new();
        for rec in &records {
            match rec {
                WalRecord::Commit { txn_id, commit_ts } => {
                    commits.insert(*txn_id, *commit_ts);
                    recovery.max_commit_ts = recovery.max_commit_ts.max(*commit_ts);
                }
                WalRecord::Rollback { txn_id } => {
                    rolled_back.insert(*txn_id);
                }
                _ => {}
            }
            if let Some(id) = rec.txn_id() {
                recovery.max_txn_id = recovery.max_txn_id.max(id);
            }
        }

        // Pass 2: collect the ops of committed transactions in log order.
        let mut per_txn: HashMap<u64, Vec<RecoveredOp>> = HashMap::new();
        for rec in &records {
            match rec {
                WalRecord::Insert { txn_id, key, value } => {
                    if commits.contains_key(txn_id) && !rolled_back.contains(txn_id) {
                        per_txn
                            .entry(*txn_id)
                            .or_default()
                            .push(RecoveredOp::Insert(key.clone(), value.clone()));
                    }
                }
                WalRecord::Delete { txn_id, key }
                    if commits.contains_key(txn_id) && !rolled_back.contains(txn_id) =>
                {
                    per_txn
                        .entry(*txn_id)
                        .or_default()
                        .push(RecoveredOp::Delete(key.clone()));
                }
                _ => {}
            }
        }

        let mut committed: Vec<(u64, Vec<RecoveredOp>)> = per_txn
            .into_iter()
            .filter_map(|(txn, ops)| commits.get(&txn).map(|ts| (*ts, ops)))
            .collect();
        // Replay in commit order so last-writer-wins matches the live engine.
        committed.sort_by_key(|(ts, _)| *ts);
        recovery.committed = committed;
        Ok(recovery)
    }
}

impl std::fmt::Debug for Wal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wal")
            .field("path", &self.path)
            .field("bytes", &self.bytes_written)
            .field("lsn", &self.lsn)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_is_replayed_uncommitted_is_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.log");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(&WalRecord::Begin { txn_id: 1 }).unwrap();
            wal.append(&WalRecord::Insert {
                txn_id: 1,
                key: b"a".to_vec(),
                value: b"1".to_vec(),
            })
            .unwrap();
            wal.commit(1, 10).unwrap();

            // Transaction 2 never commits.
            wal.append(&WalRecord::Begin { txn_id: 2 }).unwrap();
            wal.append(&WalRecord::Insert {
                txn_id: 2,
                key: b"b".to_vec(),
                value: b"2".to_vec(),
            })
            .unwrap();
            wal.sync().unwrap();
        }
        let rec = Wal::recover(&path).unwrap();
        assert_eq!(rec.committed.len(), 1);
        assert_eq!(rec.committed[0].0, 10);
        assert_eq!(
            rec.committed[0].1,
            vec![RecoveredOp::Insert(b"a".to_vec(), b"1".to_vec())]
        );
        assert_eq!(rec.max_txn_id, 2);
    }

    #[test]
    fn rollback_records_are_discarded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.log");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(&WalRecord::Begin { txn_id: 1 }).unwrap();
            wal.append(&WalRecord::Insert {
                txn_id: 1,
                key: b"x".to_vec(),
                value: b"y".to_vec(),
            })
            .unwrap();
            wal.append(&WalRecord::Rollback { txn_id: 1 }).unwrap();
            wal.sync().unwrap();
        }
        assert!(Wal::recover(&path).unwrap().committed.is_empty());
    }

    #[test]
    fn torn_tail_is_truncated_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.log");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(&WalRecord::Begin { txn_id: 1 }).unwrap();
            wal.append(&WalRecord::Insert {
                txn_id: 1,
                key: b"k".to_vec(),
                value: b"v".to_vec(),
            })
            .unwrap();
            wal.commit(1, 5).unwrap();
            // Simulate a half-written frame.
            let f = wal.writer.get_mut();
            f.write_all(&[40u8, 0, 0, 0, 1, 2, 3, 4, 9, 9]).unwrap();
            f.sync_all().unwrap();
        }
        let rec = Wal::recover(&path).unwrap();
        assert_eq!(
            rec.committed.len(),
            1,
            "committed txn must survive a torn tail"
        );
        assert_eq!(rec.truncated_bytes, 10);
    }

    #[test]
    fn bad_crc_stops_replay_at_that_frame() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.log");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(&WalRecord::Begin { txn_id: 1 }).unwrap();
            wal.commit(1, 1).unwrap();
        }
        // Corrupt the payload of the first frame.
        {
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            f.seek(SeekFrom::Start(9)).unwrap();
            f.write_all(&[0xFF]).unwrap();
            f.sync_all().unwrap();
        }
        let rec = Wal::recover(&path).unwrap();
        assert!(rec.committed.is_empty());
        assert!(rec.truncated_bytes > 0);
    }

    #[test]
    fn checkpoint_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.log");
        let mut wal = Wal::open(&path).unwrap();
        for i in 0..50u64 {
            wal.append(&WalRecord::Insert {
                txn_id: i,
                key: vec![i as u8; 32],
                value: vec![0u8; 256],
            })
            .unwrap();
        }
        wal.sync().unwrap();
        assert!(wal.size() > 10_000);
        wal.checkpoint(99).unwrap();
        assert!(wal.size() < 100, "log should be tiny after checkpoint");
    }

    #[test]
    fn missing_log_recovers_empty() {
        let dir = tempfile::tempdir().unwrap();
        let rec = Wal::recover(&dir.path().join("nope.log")).unwrap();
        assert!(rec.committed.is_empty());
    }
}
