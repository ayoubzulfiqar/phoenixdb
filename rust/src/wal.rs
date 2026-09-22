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
//! 1. A transaction's writes are staged in memory and logged **at commit**:
//!    [`Wal::log_commit`] appends every `Insert`/`Delete` followed by one
//!    `Commit`, then flushes and (optionally) calls `sync_data`, so a
//!    transaction is durable the instant the call returns. Nothing is logged
//!    for a transaction that never commits, so a checkpoint can never strand
//!    the first half of an in-flight transaction.
//! 2. Only a committed transaction is replayed by [`Wal::recover`]; records
//!    belonging to a transaction with no `Commit` are discarded (older logs
//!    that still contain `Begin`/`Rollback` records remain readable).
//! 3. After the tree is flushed, [`Wal::reset`] rewrites the log so that it
//!    holds only a `Checkpoint` marker plus every committed version the tree
//!    does not yet contain. The rewrite goes through a temporary file and an
//!    atomic rename, so a crash at any point leaves either the old or the new
//!    log — never a log missing committed work.
//! 4. [`Wal::recover`] reports where the last intact frame ends; the engine
//!    truncates a torn tail before appending, so fresh commits are never
//!    written behind garbage that would hide them from the next recovery.

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

/// Borrowed mirror of [`WalRecord`] with an identical bincode encoding.
///
/// Serde encodes `&[u8]` and `Vec<u8>` the same way, and the variants are
/// declared in the same order, so logging a commit never has to clone its
/// keys and values just to serialise them. `borrowed_encoding_matches_owned`
/// pins the equivalence.
#[derive(Serialize)]
enum WalRecordRef<'a> {
    #[allow(dead_code)]
    Begin {
        txn_id: u64,
    },
    Insert {
        txn_id: u64,
        key: &'a [u8],
        value: &'a [u8],
    },
    Delete {
        txn_id: u64,
        key: &'a [u8],
    },
    Commit {
        txn_id: u64,
        commit_ts: u64,
    },
    #[allow(dead_code)]
    Rollback {
        txn_id: u64,
    },
    Checkpoint {
        tree_ts: u64,
    },
}

/// One write of a committing transaction: `Some(value)` puts, `None` deletes.
pub type LoggedWrite<'a> = (&'a [u8], Option<&'a [u8]>);

/// A committed transaction to be re-logged by [`Wal::reset`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainedCommit {
    /// Transaction id the records are filed under.
    pub txn_id: u64,
    /// Original commit timestamp.
    pub commit_ts: u64,
    /// `(key, Some(value))` for a put, `(key, None)` for a delete.
    pub writes: Vec<(Vec<u8>, Option<Vec<u8>>)>,
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
    /// Length of the intact prefix of the log: where the next frame belongs.
    pub valid_bytes: u64,
    /// Highest `Checkpoint { tree_ts }` marker seen, if any.
    pub checkpoint_ts: Option<u64>,
}

/// Append-only write-ahead log.
pub struct Wal {
    path: PathBuf,
    /// `None` only transiently inside [`Wal::reset`] while the log file is
    /// being swapped; every public method sees `Some`.
    writer: Option<BufWriter<File>>,
    /// Bytes currently in the log (intact prefix plus everything appended).
    bytes_written: u64,
    lsn: u64,
    /// `fsync` calls issued, for metrics.
    syncs: u64,
    /// Bytes appended since open, for metrics.
    appended: u64,
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
            writer: Some(writer),
            bytes_written: len,
            lsn: 0,
            syncs: 0,
            appended: 0,
        })
    }

    /// Opens the log and discards everything past `valid_bytes`.
    ///
    /// Pass [`Recovery::valid_bytes`]: a torn tail left by a crash must be cut
    /// off before anything new is appended, otherwise the next recovery stops
    /// at the garbage and never sees the commits written after it.
    pub fn open_truncated(path: &Path, valid_bytes: u64) -> Result<Self> {
        let mut wal = Wal::open(path)?;
        if wal.bytes_written > valid_bytes {
            let writer = wal.writer()?;
            writer.flush()?;
            let file = writer.get_mut();
            file.set_len(valid_bytes)?;
            file.sync_all()?;
            writer.seek(SeekFrom::End(0))?;
            wal.bytes_written = valid_bytes;
        }
        Ok(wal)
    }

    fn writer(&mut self) -> Result<&mut BufWriter<File>> {
        self.writer.as_mut().ok_or(Error::Closed)
    }

    /// Current log sequence number (records appended, continuing from
    /// [`Wal::set_lsn`]).
    #[must_use]
    pub fn lsn(&self) -> u64 {
        self.lsn
    }

    /// Seeds the LSN counter, e.g. from the last LSN recorded in the meta page,
    /// so sequence numbers keep increasing across restarts.
    pub fn set_lsn(&mut self, lsn: u64) {
        self.lsn = self.lsn.max(lsn);
    }

    /// Bytes currently in the log.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.bytes_written
    }

    /// Number of `fsync` calls issued since open.
    #[must_use]
    pub fn sync_count(&self) -> u64 {
        self.syncs
    }

    /// Bytes appended since open.
    #[must_use]
    pub fn bytes_appended(&self) -> u64 {
        self.appended
    }

    /// Path of the log file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn append_payload(&mut self, payload: &[u8]) -> Result<u64> {
        if payload.len() as u64 > MAX_RECORD_BYTES as u64 {
            return Err(Error::Full(format!(
                "WAL record of {} bytes exceeds the {MAX_RECORD_BYTES}-byte limit",
                payload.len()
            )));
        }
        let crc = crc32fast::hash(payload);
        let writer = self.writer()?;
        writer.write_all(&(payload.len() as u32).to_le_bytes())?;
        writer.write_all(&crc.to_le_bytes())?;
        writer.write_all(payload)?;
        let framed = 8 + payload.len() as u64;
        self.bytes_written += framed;
        self.appended += framed;
        self.lsn += 1;
        Ok(self.lsn)
    }

    fn append_ref(&mut self, record: &WalRecordRef<'_>) -> Result<u64> {
        let payload = bincode::serialize(record)?;
        self.append_payload(&payload)
    }

    /// Appends a record **without** syncing. Returns its LSN.
    pub fn append(&mut self, record: &WalRecord) -> Result<u64> {
        let payload = bincode::serialize(record)?;
        self.append_payload(&payload)
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

    /// Logs a whole transaction — every write, then its `Commit` record.
    ///
    /// With `sync == true` the log is `fsync`ed before returning, so the
    /// transaction survives power loss. With `sync == false` the records are
    /// only handed to the operating system: they survive a crash of the host
    /// process but may be lost if the machine itself goes down.
    ///
    /// If an append fails midway the log holds writes without a `Commit`,
    /// which recovery ignores, so a failed call never half-commits.
    pub fn log_commit<'a>(
        &mut self,
        txn_id: u64,
        commit_ts: u64,
        writes: impl IntoIterator<Item = LoggedWrite<'a>>,
        sync: bool,
    ) -> Result<u64> {
        for (key, value) in writes {
            match value {
                Some(value) => self.append_ref(&WalRecordRef::Insert { txn_id, key, value })?,
                None => self.append_ref(&WalRecordRef::Delete { txn_id, key })?,
            };
        }
        let lsn = self.append_ref(&WalRecordRef::Commit { txn_id, commit_ts })?;
        if sync {
            self.sync()?;
        } else {
            self.writer()?.flush()?;
        }
        Ok(lsn)
    }

    /// Flushes user-space buffers and calls `sync_data`.
    pub fn sync(&mut self) -> Result<()> {
        let writer = self.writer()?;
        writer.flush()?;
        writer.get_ref().sync_data()?;
        self.syncs += 1;
        Ok(())
    }

    /// Records a checkpoint and truncates the log.
    ///
    /// Only call this once the tree is durable on disk **and** holds every
    /// committed version; when some committed versions are still only in
    /// memory (a live snapshot pinned the merge watermark) use [`Wal::reset`].
    pub fn checkpoint(&mut self, tree_ts: u64) -> Result<()> {
        self.reset(tree_ts, &[])
    }

    /// Replaces the log with a `Checkpoint { tree_ts }` marker followed by the
    /// `retained` commits — committed versions newer than `tree_ts` that the
    /// flushed tree does not contain yet.
    ///
    /// With nothing to retain, truncating in place is safe: every committed
    /// write is already in the durable tree, so a crash mid-truncate loses
    /// nothing. Otherwise the new log is built in a sibling temporary file,
    /// `fsync`ed and atomically renamed over the old one, so a crash leaves
    /// either log intact and recovery never misses a committed write.
    pub fn reset(&mut self, tree_ts: u64, retained: &[RetainedCommit]) -> Result<()> {
        self.writer()?.flush()?;
        if retained.is_empty() {
            let file = self.writer()?.get_mut();
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            file.sync_all()?;
            self.bytes_written = 0;
            self.append_ref(&WalRecordRef::Checkpoint { tree_ts })?;
            return self.sync();
        }

        let tmp_path = {
            let mut s = self.path.as_os_str().to_os_string();
            s.push(".tmp");
            PathBuf::from(s)
        };
        let tmp = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        let mut fresh = Wal {
            path: tmp_path.clone(),
            writer: Some(BufWriter::new(tmp)),
            bytes_written: 0,
            lsn: self.lsn,
            syncs: 0,
            appended: 0,
        };
        let built = (|| -> Result<()> {
            fresh.append_ref(&WalRecordRef::Checkpoint { tree_ts })?;
            for commit in retained {
                fresh.log_commit(
                    commit.txn_id,
                    commit.commit_ts,
                    commit
                        .writes
                        .iter()
                        .map(|(k, v)| (k.as_slice(), v.as_deref())),
                    false,
                )?;
            }
            fresh.sync()
        })();
        drop(fresh.writer.take());
        if let Err(e) = built {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }

        // Windows refuses to rename over a file that is still open, so the old
        // handle is released first. Until the rename lands the old log is
        // untouched and still describes the database completely.
        drop(self.writer.take());
        let renamed = std::fs::rename(&tmp_path, &self.path);
        // Reopen whatever now lives at the log path: the new log, or the old
        // one if the rename failed. Either way the handle stays usable.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.path)?;
        let len = file.metadata()?.len();
        let mut writer = BufWriter::new(file);
        writer.seek(SeekFrom::End(0))?;
        self.writer = Some(writer);
        self.bytes_written = len;
        if let Err(e) = renamed {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(Error::Io(e));
        }
        crate::fsutil::sync_parent_dir(&self.path);
        self.lsn = fresh.lsn;
        self.syncs += fresh.syncs;
        self.appended += fresh.appended;
        Ok(())
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
        recovery.valid_bytes = good_bytes as u64;

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
                WalRecord::Checkpoint { tree_ts } => {
                    recovery.checkpoint_ts =
                        Some(recovery.checkpoint_ts.map_or(*tree_ts, |t| t.max(*tree_ts)));
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
            let f = wal.writer.as_mut().unwrap().get_mut();
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

    #[test]
    fn borrowed_encoding_matches_owned() {
        let cases = [
            (
                WalRecord::Insert {
                    txn_id: 7,
                    key: b"key".to_vec(),
                    value: vec![1, 2, 3, 255],
                },
                WalRecordRef::Insert {
                    txn_id: 7,
                    key: b"key",
                    value: &[1, 2, 3, 255],
                },
            ),
            (
                WalRecord::Delete {
                    txn_id: u64::MAX,
                    key: vec![0; 40],
                },
                WalRecordRef::Delete {
                    txn_id: u64::MAX,
                    key: &[0; 40],
                },
            ),
            (
                WalRecord::Commit {
                    txn_id: 3,
                    commit_ts: 9,
                },
                WalRecordRef::Commit {
                    txn_id: 3,
                    commit_ts: 9,
                },
            ),
            (
                WalRecord::Checkpoint { tree_ts: 11 },
                WalRecordRef::Checkpoint { tree_ts: 11 },
            ),
            (
                WalRecord::Begin { txn_id: 5 },
                WalRecordRef::Begin { txn_id: 5 },
            ),
            (
                WalRecord::Rollback { txn_id: 5 },
                WalRecordRef::Rollback { txn_id: 5 },
            ),
        ];
        for (owned, borrowed) in cases {
            assert_eq!(
                bincode::serialize(&owned).unwrap(),
                bincode::serialize(&borrowed).unwrap(),
                "encoding drifted for {owned:?}"
            );
        }
    }

    #[test]
    fn log_commit_is_replayed_as_one_transaction() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.log");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_commit(
                4,
                12,
                [
                    (&b"a"[..], Some(&b"1"[..])),
                    (&b"b"[..], None),
                    (&b"c"[..], Some(&b""[..])),
                ],
                true,
            )
            .unwrap();
            assert_eq!(wal.sync_count(), 1);
        }
        let rec = Wal::recover(&path).unwrap();
        assert_eq!(rec.committed.len(), 1);
        assert_eq!(rec.committed[0].0, 12);
        assert_eq!(
            rec.committed[0].1,
            vec![
                RecoveredOp::Insert(b"a".to_vec(), b"1".to_vec()),
                RecoveredOp::Delete(b"b".to_vec()),
                RecoveredOp::Insert(b"c".to_vec(), Vec::new()),
            ]
        );
        assert_eq!(rec.valid_bytes, std::fs::metadata(&path).unwrap().len());
    }

    #[test]
    fn unsynced_commit_still_reaches_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.log");
        let mut wal = Wal::open(&path).unwrap();
        wal.log_commit(1, 1, [(&b"k"[..], Some(&b"v"[..]))], false)
            .unwrap();
        assert_eq!(wal.sync_count(), 0);
        // Not fsynced, but flushed to the OS: a reader sees it immediately.
        let rec = Wal::recover(&path).unwrap();
        assert_eq!(rec.committed.len(), 1);
    }

    #[test]
    fn reset_keeps_retained_commits_and_marks_the_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.log");
        let mut wal = Wal::open(&path).unwrap();
        for ts in 1..=20u64 {
            wal.log_commit(ts, ts, [(&b"k"[..], Some(&[ts as u8][..]))], false)
                .unwrap();
        }
        let retained = vec![
            RetainedCommit {
                txn_id: 100,
                commit_ts: 19,
                writes: vec![(b"k".to_vec(), Some(vec![19]))],
            },
            RetainedCommit {
                txn_id: 101,
                commit_ts: 20,
                writes: vec![(b"k".to_vec(), None), (b"z".to_vec(), Some(vec![1]))],
            },
        ];
        wal.reset(18, &retained).unwrap();
        // The handle keeps working after the swap.
        wal.log_commit(102, 21, [(&b"n"[..], Some(&b"new"[..]))], true)
            .unwrap();
        drop(wal);

        let rec = Wal::recover(&path).unwrap();
        assert_eq!(rec.checkpoint_ts, Some(18));
        let stamps: Vec<u64> = rec.committed.iter().map(|(ts, _)| *ts).collect();
        assert_eq!(
            stamps,
            vec![19, 20, 21],
            "only retained + new commits survive"
        );
        assert_eq!(
            rec.committed[1].1,
            vec![
                RecoveredOp::Delete(b"k".to_vec()),
                RecoveredOp::Insert(b"z".to_vec(), vec![1]),
            ]
        );
        assert!(!dir.path().join("w.log.tmp").exists());
    }

    #[test]
    fn open_truncated_drops_a_torn_tail_so_later_commits_survive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.log");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_commit(1, 1, [(&b"a"[..], Some(&b"1"[..]))], true)
                .unwrap();
            let f = wal.writer.as_mut().unwrap().get_mut();
            f.write_all(&[200u8, 0, 0, 0, 1, 2, 3]).unwrap(); // torn frame
            f.sync_all().unwrap();
        }
        let rec = Wal::recover(&path).unwrap();
        assert_eq!(rec.truncated_bytes, 7);
        {
            let mut wal = Wal::open_truncated(&path, rec.valid_bytes).unwrap();
            wal.log_commit(2, 2, [(&b"b"[..], Some(&b"2"[..]))], true)
                .unwrap();
        }
        let rec = Wal::recover(&path).unwrap();
        assert_eq!(rec.truncated_bytes, 0);
        assert_eq!(
            rec.committed.len(),
            2,
            "the commit after the tear must replay"
        );
    }
}
