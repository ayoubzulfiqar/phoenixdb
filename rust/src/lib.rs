#![deny(warnings)]
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

//! # PhoenixDB
//!
//! An ACID-compliant embedded key/value engine.
//!
//! ```text
//!   Dart (dart:ffi)  ->  ffi.rs (C ABI, validation)  ->  Database
//!                                                          |
//!                                       +------------------+------------------+
//!                                       |                  |                  |
//!                                    txn.rs             btree.rs            wal.rs
//!                                   (MVCC)            (index)            (durability)
//!                                       |                  |
//!                                       +------ pager.rs --+  (cache + CRC + mmap)
//! ```
//!
//! ## Guarantees
//!
//! * **Atomicity** — a transaction's writes are staged in memory and published
//!   at a single commit timestamp only after its WAL `Commit` record is
//!   `fsync`ed.
//! * **Consistency** — every page carries a CRC32 that is verified on read.
//! * **Isolation** — snapshot isolation with MVCC; concurrent readers never
//!   block, and a single writer is serialised by a `parking_lot::RwLock`.
//! * **Durability** — WAL-first, with `sync_all` at commit and a checkpoint
//!   that truncates the log only after the tree is flushed.
//!
//! ## Example
//!
//! ```no_run
//! use phoenixdb::{Database, Options};
//!
//! # fn main() -> phoenixdb::Result<()> {
//! let db = Database::open("data.pdb", Options::default())?;
//! let txn = db.begin(false)?;
//! db.insert(txn, b"hello", b"world")?;
//! db.commit(txn)?;
//! assert_eq!(db.get_auto(b"hello")?, b"world".to_vec());
//! # Ok(())
//! # }
//! ```

pub mod btree;
pub mod error;
pub mod ffi;
pub mod mmap;
pub mod page;
pub mod pager;
pub mod security;
pub mod txn;
pub mod wal;

pub use btree::{BTree, FillFactor};
pub use error::{Error, PhoenixStatus, Result};
pub use page::{MetaData, PAGE_SIZE};
pub use txn::{TxnState, Write};

use pager::Pager;
use parking_lot::RwLock;
use std::path::{Path, PathBuf};
use txn::VersionStore;
use wal::{RecoveredOp, Wal, WalRecord};

/// Tunable engine parameters.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Clean-page cache capacity, in pages.
    pub cache_pages: usize,
    /// B+Tree split/merge thresholds.
    pub fill_factor: FillFactor,
    /// Merge + flush + checkpoint once the WAL exceeds this many bytes.
    pub checkpoint_bytes: u64,
    /// `fsync` the WAL on every commit. Disabling trades durability for speed.
    pub sync_on_commit: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            cache_pages: pager::DEFAULT_CACHE_PAGES,
            fill_factor: FillFactor::default(),
            checkpoint_bytes: 4 * 1024 * 1024,
            sync_on_commit: true,
        }
    }
}

/// Runtime statistics, useful for tests and diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Pages allocated in the file.
    pub page_count: u32,
    /// Live transactions.
    pub active_txns: usize,
    /// Keys with unmerged in-memory versions.
    pub pending_keys: usize,
    /// Current WAL size in bytes.
    pub wal_bytes: u64,
    /// Latest commit timestamp.
    pub commit_ts: u64,
}

/// Everything guarded by the engine lock.
struct Inner {
    pager: Pager,
    wal: Wal,
    versions: VersionStore,
    tree: BTree,
}

/// The embedded database handle.
///
/// Cloning is intentionally not provided: the FFI layer owns exactly one
/// `Database` per `PhoenixDB*` and frees it in `phoenix_close`.
pub struct Database {
    inner: RwLock<Inner>,
    options: Options,
    path: PathBuf,
}

impl Database {
    /// Opens (creating if necessary) the database at `path`, replaying the WAL.
    pub fn open(path: impl AsRef<Path>, options: Options) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let wal_path = Self::wal_path(&path);

        let mut pager = Pager::open(&path, options.cache_pages)?;
        let meta = pager.meta();
        let tree = BTree::new(options.fill_factor);

        // --- crash recovery -------------------------------------------------
        let recovery = Wal::recover(&wal_path)?;
        let mut versions = VersionStore::new(meta.tree_ts + 1, meta.next_txn_id);
        versions.observe_txn_id(recovery.max_txn_id);

        let replayed = !recovery.committed.is_empty();
        for (commit_ts, ops) in &recovery.committed {
            for op in ops {
                match op {
                    RecoveredOp::Insert(k, v) => {
                        versions.apply_recovered(*commit_ts, k.clone(), Some(v.clone()));
                    }
                    RecoveredOp::Delete(k) => {
                        versions.apply_recovered(*commit_ts, k.clone(), None);
                    }
                }
            }
        }

        let mut wal = Wal::open(&wal_path)?;

        if replayed {
            // Fold the replayed versions into the tree and make them durable,
            // so a second crash does not have to replay the same records.
            let watermark = versions.current_ts();
            versions.merge_into_tree(&tree, &mut pager, watermark)?;
            let mut meta = pager.meta();
            meta.tree_ts = watermark;
            meta.next_txn_id = versions.peek_txn_id();
            pager.set_meta(meta);
            pager.flush()?;
            wal.checkpoint(watermark)?;
        }

        Ok(Database {
            inner: RwLock::new(Inner {
                pager,
                wal,
                versions,
                tree,
            }),
            options,
            path,
        })
    }

    /// Path of the WAL that accompanies a database file.
    #[must_use]
    pub fn wal_path(db_path: &Path) -> PathBuf {
        let mut s = db_path.as_os_str().to_os_string();
        s.push(".wal");
        PathBuf::from(s)
    }

    /// Path of the database file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Engine options in force.
    #[must_use]
    pub fn options(&self) -> Options {
        self.options
    }

    /// Begins a transaction and returns its id.
    ///
    /// A `read_only` transaction never takes the writer path and cannot stage
    /// writes; it is the cheapest way to get a stable snapshot.
    pub fn begin(&self, read_only: bool) -> Result<u64> {
        let mut inner = self.inner.write();
        let id = inner.versions.begin(read_only);
        if !read_only {
            inner.wal.append(&WalRecord::Begin { txn_id: id })?;
        }
        Ok(id)
    }

    /// Stages an insert (or overwrite) in transaction `txn_id`.
    pub fn insert(&self, txn_id: u64, key: &[u8], value: &[u8]) -> Result<()> {
        security::validate_key_len(key.len())?;
        security::validate_value_len(value.len())?;
        if key.len() > page::MAX_KEY_SIZE {
            return Err(Error::invalid(format!(
                "key length {} exceeds the {}-byte structural limit",
                key.len(),
                page::MAX_KEY_SIZE
            )));
        }
        let mut inner = self.inner.write();
        inner.wal.append(&WalRecord::Insert {
            txn_id,
            key: key.to_vec(),
            value: value.to_vec(),
        })?;
        inner
            .versions
            .stage(txn_id, key.to_vec(), Write::Put(value.to_vec()))
    }

    /// Stages a delete in transaction `txn_id`.
    ///
    /// Returns [`Error::NotFound`] when the key is not visible to the snapshot,
    /// so callers can distinguish "removed" from "was never there".
    pub fn delete(&self, txn_id: u64, key: &[u8]) -> Result<()> {
        security::validate_key_len(key.len())?;
        let mut inner = self.inner.write();
        // Existence check against the transaction's own view.
        let visible = match inner.versions.read(txn_id, key)? {
            Some(Some(_)) => true,
            Some(None) => false, // tombstoned in this snapshot
            None => {
                let tree = inner.tree;
                tree.contains(&mut inner.pager, key)?
            }
        };
        if !visible {
            return Err(Error::NotFound);
        }
        inner.wal.append(&WalRecord::Delete {
            txn_id,
            key: key.to_vec(),
        })?;
        inner.versions.stage(txn_id, key.to_vec(), Write::Delete)
    }

    /// Reads `key` as of transaction `txn_id`.
    pub fn get(&self, txn_id: u64, key: &[u8]) -> Result<Vec<u8>> {
        security::validate_key_len(key.len())?;
        let mut inner = self.inner.write(); // pager reads need &mut (cache)
        match inner.versions.read(txn_id, key)? {
            Some(Some(v)) => Ok(v),
            Some(None) => Err(Error::NotFound),
            None => {
                let tree = inner.tree;
                tree.get(&mut inner.pager, key)
            }
        }
    }

    /// Reads `key` in an implicit, immediately-released snapshot.
    pub fn get_auto(&self, key: &[u8]) -> Result<Vec<u8>> {
        security::validate_key_len(key.len())?;
        let mut inner = self.inner.write();
        let snapshot = inner.versions.current_ts();
        let from_versions = inner
            .versions
            .keys_with_versions(snapshot)
            .into_iter()
            .find(|(k, _)| security::ct_eq(k, key))
            .map(|(_, v)| v);
        match from_versions {
            Some(Some(v)) => Ok(v),
            Some(None) => Err(Error::NotFound),
            None => {
                let tree = inner.tree;
                tree.get(&mut inner.pager, key)
            }
        }
    }

    /// Commits transaction `txn_id`, making its writes durable.
    ///
    /// Ordering: conflict check -> WAL `Commit` + `fsync` -> publish versions.
    pub fn commit(&self, txn_id: u64) -> Result<()> {
        let mut inner = self.inner.write();
        inner.versions.detect_conflict(txn_id)?;

        let commit_ts = inner.versions.current_ts() + 1;
        if self.options.sync_on_commit {
            inner.wal.commit(txn_id, commit_ts)?;
        } else {
            inner.wal.append(&WalRecord::Commit { txn_id, commit_ts })?;
        }
        let actual = inner.versions.commit(txn_id)?;
        debug_assert_eq!(actual, commit_ts, "commit timestamp drifted");

        if inner.wal.size() >= self.options.checkpoint_bytes {
            Self::checkpoint_locked(&mut inner)?;
        }
        Ok(())
    }

    /// Rolls transaction `txn_id` back.
    pub fn rollback(&self, txn_id: u64) -> Result<()> {
        let mut inner = self.inner.write();
        inner.wal.append(&WalRecord::Rollback { txn_id })?;
        inner.versions.rollback(txn_id)
    }

    /// Convenience: single-statement insert in its own transaction.
    pub fn put_auto(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let txn = self.begin(false)?;
        match self.insert(txn, key, value) {
            Ok(()) => self.commit(txn),
            Err(e) => {
                let _ = self.rollback(txn);
                Err(e)
            }
        }
    }

    /// Convenience: single-statement delete in its own transaction.
    pub fn delete_auto(&self, key: &[u8]) -> Result<()> {
        let txn = self.begin(false)?;
        match self.delete(txn, key) {
            Ok(()) => self.commit(txn),
            Err(e) => {
                let _ = self.rollback(txn);
                Err(e)
            }
        }
    }

    /// Merges committed versions into the tree, flushes, and truncates the WAL.
    pub fn checkpoint(&self) -> Result<()> {
        let mut inner = self.inner.write();
        Self::checkpoint_locked(&mut inner)
    }

    fn checkpoint_locked(inner: &mut Inner) -> Result<()> {
        let watermark = inner.versions.merge_watermark();
        let tree = inner.tree;
        let Inner {
            pager, versions, ..
        } = inner;
        versions.merge_into_tree(&tree, pager, watermark)?;

        let mut meta = pager.meta();
        meta.tree_ts = watermark;
        meta.next_txn_id = versions.peek_txn_id();
        meta.last_lsn = inner.wal.lsn();
        pager.set_meta(meta);
        pager.flush()?; // tree is durable...
        inner.wal.checkpoint(watermark)?; // ...so the log can be discarded
        Ok(())
    }

    /// Flushes dirty pages without truncating the WAL.
    pub fn flush(&self) -> Result<()> {
        let mut inner = self.inner.write();
        inner.wal.sync()?;
        inner.pager.flush()
    }

    /// Every visible key/value pair, in ascending key order.
    pub fn scan(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut inner = self.inner.write();
        let snapshot = inner.versions.current_ts();
        let tree = inner.tree;
        let mut merged: std::collections::BTreeMap<Vec<u8>, Vec<u8>> =
            tree.scan(&mut inner.pager)?.into_iter().collect();
        for (key, value) in inner.versions.keys_with_versions(snapshot) {
            match value {
                Some(v) => {
                    merged.insert(key, v);
                }
                None => {
                    merged.remove(&key);
                }
            }
        }
        Ok(merged.into_iter().collect())
    }

    /// Number of visible keys.
    pub fn len(&self) -> Result<u64> {
        Ok(self.scan()?.len() as u64)
    }

    /// True when the database holds no visible keys.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Runtime statistics.
    pub fn stats(&self) -> Stats {
        let inner = self.inner.read();
        Stats {
            page_count: inner.pager.meta().page_count,
            active_txns: inner.versions.active_count(),
            pending_keys: inner.versions.pending_keys(),
            wal_bytes: inner.wal.size(),
            commit_ts: inner.versions.current_ts(),
        }
    }

    /// Verifies B+Tree invariants and every page checksum.
    pub fn verify(&self) -> Result<()> {
        let mut inner = self.inner.write();
        let tree = inner.tree;
        tree.verify(&mut inner.pager)
    }

    /// Flushes and checkpoints; called by `Drop` and `phoenix_close`.
    pub fn close(&self) -> Result<()> {
        self.checkpoint()
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Best-effort durability; a failing checkpoint must not panic in Drop
        // because that would unwind across the FFI boundary.
        let _ = self.checkpoint();
    }
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database")
            .field("path", &self.path)
            .field("options", &self.options)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_temp() -> (tempfile::TempDir, Database) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path().join("t.pdb"), Options::default()).unwrap();
        (dir, db)
    }

    #[test]
    fn commit_makes_writes_visible() {
        let (_d, db) = open_temp();
        let t = db.begin(false).unwrap();
        db.insert(t, b"k", b"v").unwrap();
        db.commit(t).unwrap();
        assert_eq!(db.get_auto(b"k").unwrap(), b"v");
    }

    #[test]
    fn rollback_hides_writes() {
        let (_d, db) = open_temp();
        let t = db.begin(false).unwrap();
        db.insert(t, b"k", b"v").unwrap();
        db.rollback(t).unwrap();
        assert!(matches!(db.get_auto(b"k"), Err(Error::NotFound)));
    }

    #[test]
    fn durability_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.pdb");
        {
            let db = Database::open(&path, Options::default()).unwrap();
            for i in 0..100u32 {
                db.put_auto(format!("k{i:04}").as_bytes(), format!("v{i}").as_bytes())
                    .unwrap();
            }
        }
        let db = Database::open(&path, Options::default()).unwrap();
        for i in 0..100u32 {
            assert_eq!(
                db.get_auto(format!("k{i:04}").as_bytes()).unwrap(),
                format!("v{i}").as_bytes()
            );
        }
    }

    #[test]
    fn crash_recovery_replays_committed_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.pdb");
        {
            let db = Database::open(&path, Options::default()).unwrap();
            let committed = db.begin(false).unwrap();
            db.insert(committed, b"durable", b"yes").unwrap();
            db.commit(committed).unwrap();

            let dangling = db.begin(false).unwrap();
            db.insert(dangling, b"lost", b"no").unwrap();
            db.flush().unwrap();
            // Leak the handle: no Drop, no checkpoint -> simulates a crash.
            std::mem::forget(db);
        }
        let db = Database::open(&path, Options::default()).unwrap();
        assert_eq!(db.get_auto(b"durable").unwrap(), b"yes");
        assert!(matches!(db.get_auto(b"lost"), Err(Error::NotFound)));
    }

    #[test]
    fn concurrent_readers_with_one_writer() {
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::open(dir.path().join("t.pdb"), Options::default()).unwrap());
        for i in 0..50u32 {
            db.put_auto(format!("k{i:03}").as_bytes(), b"initial").unwrap();
        }
        let mut handles = Vec::new();
        for _ in 0..4 {
            let db = Arc::clone(&db);
            handles.push(std::thread::spawn(move || {
                for i in 0..50u32 {
                    let _ = db.get_auto(format!("k{i:03}").as_bytes());
                }
            }));
        }
        let writer = {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                for i in 0..50u32 {
                    db.put_auto(format!("k{i:03}").as_bytes(), b"updated").unwrap();
                }
            })
        };
        for h in handles {
            h.join().unwrap();
        }
        writer.join().unwrap();
        assert_eq!(db.get_auto(b"k000").unwrap(), b"updated");
    }

    #[test]
    fn delete_missing_key_reports_not_found() {
        let (_d, db) = open_temp();
        assert!(matches!(db.delete_auto(b"ghost"), Err(Error::NotFound)));
    }

    #[test]
    fn checkpoint_truncates_wal_and_keeps_data() {
        let (_d, db) = open_temp();
        for i in 0..200u32 {
            db.put_auto(format!("k{i:04}").as_bytes(), &vec![7u8; 512])
                .unwrap();
        }
        db.checkpoint().unwrap();
        assert!(db.stats().wal_bytes < 1024);
        assert_eq!(db.get_auto(b"k0000").unwrap(), vec![7u8; 512]);
        assert_eq!(db.len().unwrap(), 200);
        db.verify().unwrap();
    }

    #[test]
    fn scan_is_ordered_and_reflects_deletes() {
        let (_d, db) = open_temp();
        for i in 0..30u32 {
            db.put_auto(format!("k{i:03}").as_bytes(), b"v").unwrap();
        }
        db.delete_auto(b"k005").unwrap();
        let items = db.scan().unwrap();
        assert_eq!(items.len(), 29);
        for w in items.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
        assert!(!items.iter().any(|(k, _)| k == b"k005"));
    }

    #[test]
    fn oversized_key_is_rejected_at_the_api() {
        let (_d, db) = open_temp();
        let t = db.begin(false).unwrap();
        let key = vec![b'x'; security::MAX_KEY_LEN + 1];
        assert!(matches!(
            db.insert(t, &key, b"v"),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn large_value_roundtrip() {
        let (_d, db) = open_temp();
        let value: Vec<u8> = (0..300_000u32).map(|i| (i % 253) as u8).collect();
        db.put_auto(b"big", &value).unwrap();
        db.checkpoint().unwrap();
        assert_eq!(db.get_auto(b"big").unwrap(), value);
    }
}
