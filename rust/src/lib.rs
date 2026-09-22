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
//!                                       +------ pager.rs --+  (cache + CRC + journal + mmap)
//! ```
//!
//! ## Guarantees
//!
//! * **Atomicity** — a transaction's writes are staged in memory and logged
//!   together with its `Commit` record at commit time; recovery replays only
//!   transactions whose `Commit` reached the log.
//! * **Consistency** — every page carries a CRC32 that is verified on read,
//!   and [`Database::check`] verifies the whole tree structure.
//! * **Isolation** — snapshot isolation with MVCC. Readers share a lock and
//!   never block each other; writers serialise. Write-write conflicts fail
//!   with [`Error::Conflict`].
//! * **Durability** — WAL-first. With [`Options::sync_on_commit`] a commit is
//!   `fsync`ed before it returns. Checkpoints write pages through a journal,
//!   so a crash mid-checkpoint cannot tear the file, and they keep every
//!   committed version the tree does not hold yet in the log.
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
pub mod collection;
pub mod error;
pub mod ffi;
mod fsutil;
pub mod lsm;
pub mod mmap;
pub mod observability;
pub mod page;
pub mod page_cache;
pub mod page_pool;
pub mod pager;
pub use pager::Pager;
pub mod security;
#[cfg(feature = "sql")]
pub mod sql;
pub mod txn;
pub mod vector;
pub mod wal;

pub use btree::{BTree, FillFactor, TreeReport};
pub use error::{Error, PhoenixStatus, Result};
pub use page::{MetaData, PAGE_SIZE};
pub use txn::{TxnState, Write};
pub use vector::{Metric, VectorEngine, VectorMatch, VectorOptions};

use crate::observability::metrics::{EngineMetrics, Timer};
use crate::observability::tracing::{CollectingExporter, SpanRecord, Tracer};
use parking_lot::RwLock;
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use txn::{OverlayEntry, VersionStore};
use wal::{RecoveredOp, Wal};

/// Tunable engine parameters.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Clean-page cache capacity, in pages.
    pub cache_pages: usize,
    /// B+Tree split thresholds.
    pub fill_factor: FillFactor,
    /// Merge + flush + checkpoint once the WAL exceeds this many bytes.
    pub checkpoint_bytes: u64,
    /// `fsync` the WAL on every commit. When disabled a commit is still handed
    /// to the operating system before `commit` returns — it survives the app
    /// crashing, but not the machine losing power.
    pub sync_on_commit: bool,
    /// Record spans for engine operations (see [`Database::spans`]).
    pub tracing: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            cache_pages: pager::DEFAULT_CACHE_PAGES,
            fill_factor: FillFactor::default(),
            checkpoint_bytes: 4 * 1024 * 1024,
            sync_on_commit: true,
            tracing: false,
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
    /// Every version at or below this timestamp is in the durable tree.
    pub tree_ts: u64,
    /// Page reads served from memory.
    pub cache_hits: u64,
    /// Page reads that had to decode a page from the file.
    pub cache_misses: u64,
}

/// Everything guarded by the engine lock.
struct Inner {
    pager: Pager,
    wal: Wal,
    versions: VersionStore,
    tree: BTree,
    /// WAL size at which the next automatic checkpoint runs.
    next_auto_checkpoint: u64,
    /// The configured base threshold ([`Options::checkpoint_bytes`]).
    checkpoint_bytes: u64,
}

/// Where a scan starts: `lo`/`hi` bound the keys, and the overlay supplies the
/// in-memory versions to merge over the tree.
struct ScanPlan<'a> {
    lo: Bound<&'a [u8]>,
    hi: Bound<&'a [u8]>,
    overlay: Vec<OverlayEntry>,
}

/// The embedded database handle.
///
/// Cloning is intentionally not provided: the FFI layer owns exactly one
/// `Database` per `PhoenixDB*` and frees it in `phoenix_close`. Share it across
/// threads with an `Arc`.
pub struct Database {
    inner: RwLock<Inner>,
    options: Options,
    path: PathBuf,
    exporter: Arc<CollectingExporter>,
    tracer: Tracer,
    tracing: AtomicBool,
    metrics: EngineMetrics,
    /// See [`Database::serialize_writes`].
    writer_turn: parking_lot::Mutex<()>,
    /// Set by [`Database::simulate_crash`]: skip the checkpoint in `Drop`.
    crashed: bool,
}

impl Database {
    /// Opens (creating if necessary) the database at `path`, replaying the WAL.
    ///
    /// Fails with [`Error::Busy`] when the file is already open by another
    /// `Database` or process. (The C ABI shares one engine between handles
    /// that open the same path in one process; see `phoenix_open`.)
    pub fn open(path: impl AsRef<Path>, options: Options) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let wal_path = Self::wal_path(&path);
        let exporter = Arc::new(CollectingExporter::new(1024));
        let tracer = Tracer::new(exporter.clone());
        let metrics = EngineMetrics::new();

        let pager = Pager::open(&path, options.cache_pages)?;
        let meta = pager.meta();
        let tree = BTree::new(options.fill_factor);

        // A `.wal.tmp` is a log rewrite that crashed before its rename: the
        // original log is intact and authoritative, the temporary is not.
        let mut tmp = wal_path.as_os_str().to_os_string();
        tmp.push(".tmp");
        let _ = std::fs::remove_file(PathBuf::from(tmp));

        // --- crash recovery -------------------------------------------------
        let recovery = Wal::recover(&wal_path)?;
        let mut versions = VersionStore::new(meta.tree_ts + 1, meta.next_txn_id);
        versions.observe_txn_id(recovery.max_txn_id);
        let mut replayed = 0usize;
        for (commit_ts, ops) in &recovery.committed {
            // Anything at or below tree_ts is already in the durable tree.
            if *commit_ts <= meta.tree_ts {
                continue;
            }
            replayed += 1;
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
        // Cut a torn tail off before anything is appended behind it.
        let mut wal = Wal::open_truncated(&wal_path, recovery.valid_bytes)?;
        wal.set_lsn(meta.last_lsn);

        let mut inner = Inner {
            pager,
            wal,
            versions,
            tree,
            next_auto_checkpoint: options.checkpoint_bytes,
            checkpoint_bytes: options.checkpoint_bytes,
        };

        if replayed > 0 {
            // Fold the replayed versions into the tree and make them durable,
            // so a second crash does not have to replay the same records.
            Self::checkpoint_locked(&mut inner, &metrics)?;
        }

        let db = Database {
            inner: RwLock::new(inner),
            options,
            path,
            exporter,
            tracer,
            tracing: AtomicBool::new(options.tracing),
            metrics,
            writer_turn: parking_lot::Mutex::new(()),
            crashed: false,
        };
        if db.tracing_enabled() {
            let _span = db
                .tracer
                .span("open")
                .with_attribute("replayed_txns", replayed.to_string())
                .with_attribute("torn_bytes", recovery.truncated_bytes.to_string());
        }
        Ok(db)
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

    fn tracing_enabled(&self) -> bool {
        self.tracing.load(Ordering::Relaxed)
    }

    /// Turns span recording on or off at runtime.
    pub fn set_tracing(&self, enabled: bool) {
        self.tracing.store(enabled, Ordering::Relaxed);
    }

    // -----------------------------------------------------------------------
    // Transactions
    // -----------------------------------------------------------------------

    /// Begins a transaction and returns its id.
    ///
    /// A `read_only` transaction cannot stage writes; it is the cheapest way
    /// to get a stable snapshot. Every transaction pins its snapshot until it
    /// commits or rolls back, so finish transactions promptly: a forgotten one
    /// keeps old versions in memory and in the WAL.
    pub fn begin(&self, read_only: bool) -> Result<u64> {
        let mut inner = self.inner.write();
        Ok(inner.versions.begin(read_only))
    }

    /// Stages an insert (or overwrite) in transaction `txn_id`.
    pub fn insert(&self, txn_id: u64, key: &[u8], value: &[u8]) -> Result<()> {
        Self::validate_write(key, Some(value))?;
        let _timer = Timer::start(&self.metrics.write_latency);
        let mut inner = self.inner.write();
        inner
            .versions
            .stage(txn_id, key.to_vec(), Write::Put(value.to_vec()))?;
        self.metrics.writes.increment();
        Ok(())
    }

    /// Stages a delete in transaction `txn_id`.
    ///
    /// Returns [`Error::NotFound`] when the key is not visible to the snapshot,
    /// so callers can distinguish "removed" from "was never there".
    pub fn delete(&self, txn_id: u64, key: &[u8]) -> Result<()> {
        security::validate_key_len(key.len())?;
        let _timer = Timer::start(&self.metrics.write_latency);
        let mut inner = self.inner.write();
        inner.versions.check_writable(txn_id)?;
        // Existence check against the transaction's own view.
        let visible = match inner.versions.read(txn_id, key)? {
            Some(v) => v.is_some(),
            None => key.len() <= page::MAX_KEY_SIZE && inner.tree.contains(&inner.pager, key)?,
        };
        if !visible {
            return Err(Error::NotFound);
        }
        inner.versions.stage(txn_id, key.to_vec(), Write::Delete)?;
        self.metrics.writes.increment();
        Ok(())
    }

    fn validate_write(key: &[u8], value: Option<&[u8]>) -> Result<()> {
        security::validate_key_len(key.len())?;
        if let Some(v) = value {
            security::validate_value_len(v.len())?;
        }
        if key.len() > page::MAX_KEY_SIZE {
            return Err(Error::invalid(format!(
                "key length {} exceeds the {}-byte structural limit",
                key.len(),
                page::MAX_KEY_SIZE
            )));
        }
        Ok(())
    }

    /// Reads `key` as of transaction `txn_id`, including its own writes.
    pub fn get(&self, txn_id: u64, key: &[u8]) -> Result<Vec<u8>> {
        security::validate_key_len(key.len())?;
        let _timer = Timer::start(&self.metrics.read_latency);
        self.metrics.reads.increment();
        let inner = self.inner.read();
        match inner.versions.read(txn_id, key)? {
            Some(Some(v)) => Ok(v),
            Some(None) => Err(Error::NotFound),
            None => Self::tree_get(&inner, key),
        }
    }

    /// Reads `key` in an implicit, immediately-released snapshot.
    pub fn get_auto(&self, key: &[u8]) -> Result<Vec<u8>> {
        security::validate_key_len(key.len())?;
        let _timer = Timer::start(&self.metrics.read_latency);
        self.metrics.reads.increment();
        let inner = self.inner.read();
        let snapshot = inner.versions.current_ts();
        match inner.versions.visible(key, snapshot) {
            Some(Some(v)) => Ok(v),
            Some(None) => Err(Error::NotFound),
            None => Self::tree_get(&inner, key),
        }
    }

    fn tree_get(inner: &Inner, key: &[u8]) -> Result<Vec<u8>> {
        if key.len() > page::MAX_KEY_SIZE {
            return Err(Error::NotFound); // such a key can never be stored
        }
        inner.tree.get(&inner.pager, key)
    }

    /// Commits transaction `txn_id`, making its writes durable.
    ///
    /// Ordering: conflict check -> log writes + `Commit` (+ `fsync`) ->
    /// publish versions. On [`Error::Conflict`] the transaction is aborted
    /// (a later `rollback` reports [`Error::TxnNotFound`]); begin a new one
    /// and retry.
    pub fn commit(&self, txn_id: u64) -> Result<()> {
        let _timer = Timer::start(&self.metrics.txn_commit_latency);
        let mut inner = self.inner.write();
        let mut span = self.tracing_enabled().then(|| {
            self.tracer
                .span("commit")
                .with_attribute("txn_id", txn_id.to_string())
        });

        let Inner { wal, versions, .. } = &mut *inner;
        let txn = versions.get(txn_id)?;
        if txn.read_only || txn.writes.is_empty() {
            // Nothing to make durable: just release the snapshot.
            versions.rollback(txn_id)?;
            self.metrics.txn_commits.increment();
            return Ok(());
        }
        if let Err(e) = versions.detect_conflict(txn_id) {
            // A conflicted transaction can never commit (its snapshot is
            // stale for good), so it is aborted here rather than left open:
            // a caller that forgets the rollback would otherwise pin the
            // merge watermark — and grow memory and the WAL — forever.
            versions.rollback(txn_id)?;
            self.metrics.txn_conflicts.increment();
            if let Some(s) = span.as_mut() {
                s.set_error("write-write conflict");
            }
            return Err(e);
        }
        let commit_ts = versions.next_commit_ts();
        let sync = self.options.sync_on_commit;
        let (syncs_before, bytes_before) = (wal.sync_count(), wal.bytes_appended());
        let started = std::time::Instant::now();
        let writes = txn.writes.len();
        wal.log_commit(
            txn_id,
            commit_ts,
            txn.writes.iter().map(|(k, w)| (k.as_slice(), w.as_value())),
            sync,
        )?;
        if sync {
            self.metrics.wal_fsync_latency.record(started.elapsed());
        }
        self.metrics.wal_fsyncs.add(wal.sync_count() - syncs_before);
        self.metrics
            .wal_bytes_written
            .add(wal.bytes_appended() - bytes_before);
        let actual = versions.commit(txn_id)?;
        debug_assert_eq!(actual, commit_ts, "commit timestamp drifted");
        self.metrics.txn_commits.increment();
        if let Some(s) = span.as_mut() {
            s.set_attribute("writes", writes.to_string());
            s.set_attribute("commit_ts", commit_ts.to_string());
        }

        if inner.wal.size() >= inner.next_auto_checkpoint {
            // The commit is already durable: a failed background checkpoint
            // must not turn it into an error. The next one retries.
            let _ = Self::checkpoint_locked(&mut inner, &self.metrics);
        }
        Ok(())
    }

    /// Takes this database's writer turn: a mutex that read-modify-write
    /// callers hold across `begin` .. `commit` so they never conflict with
    /// each other.
    ///
    /// Snapshot isolation is optimistic — concurrent transactions that write
    /// the same key race, and the loser gets [`Error::Conflict`]. When many
    /// writers contend for one hot key (a counter, a table's row-id sequence)
    /// retrying can starve; writes serialise inside the engine anyway, so
    /// taking turns up front costs nothing and removes the conflicts. The SQL
    /// layer holds it for autocommit statements. Readers never need it.
    pub fn serialize_writes(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.writer_turn.lock()
    }

    /// Rolls transaction `txn_id` back.
    pub fn rollback(&self, txn_id: u64) -> Result<()> {
        let mut inner = self.inner.write();
        inner.versions.rollback(txn_id)?;
        self.metrics.txn_rollbacks.increment();
        Ok(())
    }

    /// Convenience: single-statement insert in its own transaction.
    pub fn put_auto(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.write_batch(|batch| batch.put(key, value))
    }

    /// Convenience: single-statement delete in its own transaction.
    pub fn delete_auto(&self, key: &[u8]) -> Result<()> {
        self.write_batch(|batch| batch.delete(key))
    }

    /// Runs `f` against a fresh write transaction and commits it, or rolls it
    /// back if `f` (or the commit) fails. All writes land atomically.
    ///
    /// ```no_run
    /// # fn main() -> phoenixdb::Result<()> {
    /// # let db = phoenixdb::Database::open("d.pdb", Default::default())?;
    /// db.write_batch(|b| {
    ///     b.put(b"user:1", b"ada")?;
    ///     b.put(b"user:2", b"grace")?;
    ///     b.delete_if_exists(b"user:0")
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_batch<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut Batch<'_>) -> Result<()>,
    {
        let txn = self.begin(false)?;
        let mut batch = Batch { db: self, txn };
        let outcome = f(&mut batch).and_then(|()| self.commit(txn));
        if outcome.is_err() {
            let _ = self.rollback(txn);
        }
        outcome
    }

    // -----------------------------------------------------------------------
    // Checkpoints
    // -----------------------------------------------------------------------

    /// Merges committed versions into the tree, flushes, and rewrites the WAL.
    pub fn checkpoint(&self) -> Result<()> {
        let mut inner = self.inner.write();
        let _span = self
            .tracing_enabled()
            .then(|| self.tracer.span("checkpoint"));
        Self::checkpoint_locked(&mut inner, &self.metrics)
    }

    fn checkpoint_locked(inner: &mut Inner, metrics: &EngineMetrics) -> Result<()> {
        let _timer = Timer::start(&metrics.checkpoint_latency);
        let watermark = inner.versions.merge_watermark();
        let Inner {
            pager,
            versions,
            tree,
            wal,
            ..
        } = inner;
        let durable_meta = pager.meta();
        if let Err(e) = versions.merge_into_tree(tree, pager, watermark) {
            // The version store is untouched (two-phase merge); drop the
            // half-applied pages so memory matches the durable state again.
            pager.discard_dirty(durable_meta);
            return Err(e);
        }
        let tree_ts = durable_meta.tree_ts.max(watermark);
        // Versions a live snapshot still pins stay in the log.
        let retained = versions.retained_commits(tree_ts);

        let mut meta = pager.meta();
        meta.tree_ts = tree_ts;
        meta.next_txn_id = versions.peek_txn_id();
        meta.last_lsn = wal.lsn();
        pager.set_meta(meta);
        // On failure the merged pages stay staged (reads see them) and the log
        // still holds everything, so the next checkpoint simply retries.
        pager.flush()?; // tree is durable...
        wal.reset(tree_ts, &retained)?; // ...so the log can shrink
        // Versions pinned by a long-lived reader are re-logged every time, so
        // back off relative to what the log still holds instead of
        // checkpointing on every commit until the reader finishes.
        inner.next_auto_checkpoint = inner
            .checkpoint_bytes
            .max(inner.wal.size().saturating_mul(2));
        metrics.checkpoints.increment();
        Ok(())
    }

    /// Syncs the WAL and flushes any staged pages without rewriting the log.
    pub fn flush(&self) -> Result<()> {
        let mut inner = self.inner.write();
        inner.wal.sync()?;
        inner.pager.flush()
    }

    // -----------------------------------------------------------------------
    // Scans
    // -----------------------------------------------------------------------

    /// Every visible key/value pair, in ascending key order.
    pub fn scan(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut out = Vec::new();
        self.scan_iter(|item| {
            out.push(item);
            Ok(())
        })?;
        Ok(out)
    }

    /// Streams every visible key/value pair in ascending key order.
    ///
    /// The tree is streamed page by page and merged with the in-memory
    /// versions, so nothing beyond the unmerged versions is materialised. The
    /// callback runs under the engine's shared lock: it must not call back
    /// into this database.
    pub fn scan_iter<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut((Vec<u8>, Vec<u8>)) -> Result<()>,
    {
        self.scan_range(Bound::Unbounded, Bound::Unbounded, |k, v| {
            f((k, v))?;
            Ok(true)
        })
    }

    /// Streams the latest committed pairs with keys in `(lo, hi)`, ascending.
    ///
    /// `f` returns `Ok(true)` to continue or `Ok(false)` to stop early. Same
    /// re-entrancy rule as [`Database::scan_iter`].
    pub fn scan_range<F>(&self, lo: Bound<&[u8]>, hi: Bound<&[u8]>, f: F) -> Result<()>
    where
        F: FnMut(Vec<u8>, Vec<u8>) -> Result<bool>,
    {
        let _timer = Timer::start(&self.metrics.scan_latency);
        self.metrics.scans.increment();
        let inner = self.inner.read();
        let snapshot = inner.versions.current_ts();
        let plan = ScanPlan {
            lo,
            hi,
            overlay: inner.versions.overlay(snapshot, lo, hi),
        };
        Self::merge_scan(&inner, plan, f)
    }

    /// Streams the latest committed pairs whose key starts with `prefix`.
    pub fn scan_prefix<F>(&self, prefix: &[u8], f: F) -> Result<()>
    where
        F: FnMut(Vec<u8>, Vec<u8>) -> Result<bool>,
    {
        let end = prefix_successor(prefix);
        let hi = match &end {
            Some(e) => Bound::Excluded(e.as_slice()),
            None => Bound::Unbounded,
        };
        let lo = if prefix.is_empty() {
            Bound::Unbounded
        } else {
            Bound::Included(prefix)
        };
        self.scan_range(lo, hi, f)
    }

    /// Streams what transaction `txn_id` sees for keys in `(lo, hi)`: its
    /// snapshot plus its own uncommitted writes.
    pub fn scan_txn<F>(&self, txn_id: u64, lo: Bound<&[u8]>, hi: Bound<&[u8]>, f: F) -> Result<()>
    where
        F: FnMut(Vec<u8>, Vec<u8>) -> Result<bool>,
    {
        let _timer = Timer::start(&self.metrics.scan_latency);
        self.metrics.scans.increment();
        let inner = self.inner.read();
        let plan = ScanPlan {
            lo,
            hi,
            overlay: inner.versions.txn_overlay(txn_id, lo, hi)?,
        };
        Self::merge_scan(&inner, plan, f)
    }

    /// Collects up to `limit` pairs with keys in `(lo, hi)` (0 = no limit).
    pub fn range(
        &self,
        lo: Bound<&[u8]>,
        hi: Bound<&[u8]>,
        limit: usize,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut out = Vec::new();
        self.scan_range(lo, hi, |k, v| {
            out.push((k, v));
            Ok(limit == 0 || out.len() < limit)
        })?;
        Ok(out)
    }

    /// Merges the tree's pairs in `(lo, hi)` with the overlay, in key order:
    /// an overlay entry replaces the tree's value for its key, and an overlay
    /// tombstone hides it.
    fn merge_scan<F>(inner: &Inner, plan: ScanPlan<'_>, mut f: F) -> Result<()>
    where
        F: FnMut(Vec<u8>, Vec<u8>) -> Result<bool>,
    {
        let mut overlay = plan.overlay.into_iter().peekable();
        let mut stopped = false;
        inner
            .tree
            .range_iter(&inner.pager, plan.lo, plan.hi, |key, value| {
                // Overlay keys that sort before this tree key come first.
                while let Some((ok, _)) = overlay.peek() {
                    if ok.as_slice() >= key.as_slice() {
                        break;
                    }
                    let (ok, ov) = overlay.next().expect("peeked");
                    if let Some(v) = ov
                        && !f(ok, v)?
                    {
                        stopped = true;
                        return Ok(false);
                    }
                }
                let emitted = match overlay.next_if(|(ok, _)| *ok == key) {
                    Some((_, None)) => return Ok(true), // deleted in memory
                    Some((ok, Some(v))) => f(ok, v)?,   // newer in memory
                    None => f(key, value)?,
                };
                stopped = !emitted;
                Ok(emitted)
            })?;
        if stopped {
            return Ok(());
        }
        for (key, value) in overlay {
            if let Some(v) = value
                && !f(key, v)?
            {
                break;
            }
        }
        Ok(())
    }

    /// Number of visible keys.
    ///
    /// Counts the tree's keys and corrects for the in-memory versions, so it
    /// never reads a value.
    pub fn len(&self) -> Result<u64> {
        let inner = self.inner.read();
        let mut count = inner.tree.len(&inner.pager)?;
        let snapshot = inner.versions.current_ts();
        for (key, value) in inner.versions.keys_with_versions(snapshot) {
            let in_tree =
                key.len() <= page::MAX_KEY_SIZE && inner.tree.contains(&inner.pager, &key)?;
            match (value.is_some(), in_tree) {
                (true, false) => count += 1,
                (false, true) => count -= 1,
                _ => {}
            }
        }
        Ok(count)
    }

    /// True when the database holds no visible keys.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /// Runtime statistics.
    pub fn stats(&self) -> Stats {
        let inner = self.inner.read();
        let cache = inner.pager.cache_stats();
        let meta = inner.pager.meta();
        Stats {
            page_count: meta.page_count,
            active_txns: inner.versions.active_count(),
            pending_keys: inner.versions.pending_keys(),
            wal_bytes: inner.wal.size(),
            commit_ts: inner.versions.current_ts(),
            tree_ts: meta.tree_ts,
            cache_hits: cache.hits,
            cache_misses: cache.misses,
        }
    }

    /// Spans recorded while tracing was enabled, oldest first (bounded ring).
    pub fn spans(&self) -> Vec<SpanRecord> {
        self.exporter.spans()
    }

    /// Verifies B+Tree invariants and every page checksum.
    pub fn verify(&self) -> Result<()> {
        self.check().map(|_| ())
    }

    /// Full structural check; see [`BTree::check`] for what is verified.
    pub fn check(&self) -> Result<TreeReport> {
        let inner = self.inner.read();
        inner.tree.check(&inner.pager)
    }

    /// Adds an observed duration to the matching latency histogram.
    ///
    /// `op` is one of `read`/`get`, `write`/`insert`/`put`/`delete`, `scan`,
    /// `commit` or `checkpoint`; anything else is ignored. Lets a host that
    /// measures end-to-end latency (e.g. across an isolate hop) feed it into
    /// the same report.
    pub fn record_latency(&self, op: &str, micros: u64) {
        let histo = match op {
            "read" | "get" => &self.metrics.read_latency,
            "write" | "insert" | "put" | "delete" => &self.metrics.write_latency,
            "scan" => &self.metrics.scan_latency,
            "commit" => &self.metrics.txn_commit_latency,
            "checkpoint" => &self.metrics.checkpoint_latency,
            _ => return,
        };
        histo.record_micros(micros);
    }

    /// Counts an externally performed write operation (`insert`, `put`,
    /// `delete` or `write`); anything else is ignored.
    pub fn record_write(&self, op: &str) {
        if matches!(op, "insert" | "put" | "delete" | "write") {
            self.metrics.writes.increment();
        }
    }

    /// The live metrics registry.
    #[must_use]
    pub fn metrics(&self) -> &EngineMetrics {
        self.sync_cache_metrics();
        &self.metrics
    }

    fn sync_cache_metrics(&self) {
        let cache = self.inner.read().pager.cache_stats();
        self.metrics.cache_hits.raise_to(cache.hits);
        self.metrics.cache_misses.raise_to(cache.misses);
        self.metrics.cache_resident_pages.set(cache.resident);
    }

    /// Human-readable metrics snapshot for debugging.
    pub fn metrics_report(&self) -> String {
        self.sync_cache_metrics();
        self.metrics.report()
    }

    /// Metrics in the Prometheus text exposition format.
    pub fn metrics_prometheus(&self) -> String {
        self.sync_cache_metrics();
        self.metrics.prometheus()
    }

    // -----------------------------------------------------------------------
    // Backup, restore, compaction
    // -----------------------------------------------------------------------

    /// Writes a consistent, self-contained, compacted copy of the database to
    /// `backup_path`.
    ///
    /// The copy is built from one snapshot while other threads keep reading
    /// and writing, holds no WAL and no free pages, and appears at
    /// `backup_path` atomically (built next to it, then renamed). Refuses to
    /// overwrite a database that is currently open.
    pub fn backup(&self, backup_path: impl AsRef<Path>) -> Result<()> {
        let target = backup_path.as_ref();
        if same_file(target, &self.path) {
            return Err(Error::invalid("cannot back a database up onto itself"));
        }
        let _span = self.tracing_enabled().then(|| {
            self.tracer
                .span("backup")
                .with_attribute("target", target.display().to_string())
        });
        refuse_if_open(target)?;
        let tmp = sidecar(target, ".tmp");
        remove_database_files(&tmp);

        let snapshot = self.begin(true)?;
        let built = self.copy_snapshot_to(snapshot, &tmp);
        let _ = self.rollback(snapshot);
        if let Err(e) = built {
            remove_database_files(&tmp);
            return Err(e);
        }
        // A stale log next to the target would be replayed over the backup.
        remove_sidecars(target);
        std::fs::rename(&tmp, target)?;
        fsutil::sync_parent_dir(target);
        remove_database_files(&tmp);
        Ok(())
    }

    /// Copies what `snapshot` sees into a brand-new database at `dest`.
    fn copy_snapshot_to(&self, snapshot: u64, dest: &Path) -> Result<()> {
        // Chunked so the shared lock is released between batches; the
        // read-only snapshot keeps the view consistent throughout.
        copy_into_new_database(dest, self.options, |lo, emit| {
            self.scan_txn(snapshot, lo, Bound::Unbounded, emit)
        })
    }

    /// Replaces this database's contents with the database at `backup_path`.
    ///
    /// The backup is fully verified first and installed atomically: a crash
    /// midway leaves either the old contents or the backup. Requires that no
    /// transaction is open on this handle.
    pub fn restore(&self, backup_path: impl AsRef<Path>) -> Result<()> {
        let source = backup_path.as_ref();
        if same_file(source, &self.path) {
            return Err(Error::invalid("cannot restore a database onto itself"));
        }
        let _span = self.tracing_enabled().then(|| {
            self.tracer
                .span("restore")
                .with_attribute("source", source.display().to_string())
        });
        // Validate before touching anything. A backup that still has a log
        // (not one made by `backup`) is opened once so its log is folded in.
        let source_wal = Self::wal_path(source);
        if std::fs::metadata(&source_wal).is_ok_and(|m| m.len() > 64) {
            let folded = Database::open(source, Options::default())?;
            folded.verify()?;
        } else {
            if std::fs::metadata(source)?.len() == 0 {
                return Err(Error::invalid(format!(
                    "{} is empty, not a PhoenixDB backup",
                    source.display()
                )));
            }
            let pager = Pager::open(source, 64)?;
            self.inner.read().tree.check(&pager)?;
        }

        let mut inner = self.inner.write();
        if inner.versions.active_count() > 0 {
            return Err(Error::invalid(
                "cannot restore while transactions are open on this handle",
            ));
        }
        // Empty the log first, so a crash after the swap can never replay the
        // old database's commits over the restored one.
        Self::checkpoint_locked(&mut inner, &self.metrics)?;
        inner.pager.replace_contents(source)?;
        let meta = inner.pager.meta();
        inner.versions = VersionStore::new(meta.tree_ts + 1, meta.next_txn_id);
        inner.wal.reset(meta.tree_ts, &[])?;
        inner.next_auto_checkpoint = self.options.checkpoint_bytes;
        Ok(())
    }

    /// Rebuilds the file compactly: live data only, pages packed, free pages
    /// returned to the filesystem. Requires that no transaction is open.
    ///
    /// Holds the write lock for the whole rebuild, so no commit can land
    /// between the copy and the swap (and be lost with the old file).
    pub fn compact(&self) -> Result<()> {
        let mut inner = self.inner.write();
        if inner.versions.active_count() > 0 {
            return Err(Error::invalid(
                "cannot compact while transactions are open on this handle",
            ));
        }
        let _span = self.tracing_enabled().then(|| self.tracer.span("compact"));
        // With no live snapshot this merges every version: the tree is the
        // whole database.
        Self::checkpoint_locked(&mut inner, &self.metrics)?;
        let scratch = sidecar(&self.path, ".compact");
        remove_database_files(&scratch);
        let built = {
            let source: &Inner = &inner;
            copy_into_new_database(&scratch, self.options, |lo, emit| {
                source
                    .tree
                    .range_iter(&source.pager, lo, Bound::Unbounded, emit)
            })
        };
        let outcome = built.and_then(|()| {
            inner.pager.replace_contents(&scratch)?;
            let meta = inner.pager.meta();
            inner.versions = VersionStore::new(meta.tree_ts + 1, meta.next_txn_id);
            inner.wal.reset(meta.tree_ts, &[])?;
            inner.next_auto_checkpoint = inner.checkpoint_bytes;
            Ok(())
        });
        remove_database_files(&scratch);
        outcome
    }

    /// Flushes and checkpoints; called by `Drop` and `phoenix_close`.
    pub fn close(&self) -> Result<()> {
        self.checkpoint()
    }

    /// Drops the handle the way a crash would: no checkpoint, no merge, no
    /// final flush — only what already reached the WAL and the data file
    /// survives. File handles (and the file lock) are released, so the same
    /// process can reopen the database and exercise recovery.
    ///
    /// Intended for crash-recovery tests.
    #[doc(hidden)]
    pub fn simulate_crash(mut self) {
        self.crashed = true;
    }
}

/// Stages writes for [`Database::write_batch`].
pub struct Batch<'a> {
    db: &'a Database,
    txn: u64,
}

impl Batch<'_> {
    /// Stages an insert or overwrite.
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.insert(self.txn, key, value)
    }

    /// Stages a delete; fails with [`Error::NotFound`] if the key is absent.
    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.db.delete(self.txn, key)
    }

    /// Stages a delete if the key exists; returns whether it did.
    pub fn delete_if_exists(&mut self, key: &[u8]) -> Result<()> {
        match self.db.delete(self.txn, key) {
            Ok(()) | Err(Error::NotFound) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Reads through the batch: sees the batch's own staged writes.
    pub fn get(&self, key: &[u8]) -> Result<Vec<u8>> {
        self.db.get(self.txn, key)
    }

    /// The underlying transaction id.
    #[must_use]
    pub fn txn_id(&self) -> u64 {
        self.txn
    }
}

/// Builds a new database at `dest` from a key-ordered source.
///
/// `scan(lo, emit)` must call `emit(key, value)` for pairs after `lo` in
/// ascending order and stop when `emit` returns `false`; the copy pulls
/// batches of bounded size, so neither side ever holds the whole data set.
fn copy_into_new_database<S>(dest: &Path, options: Options, mut scan: S) -> Result<()>
where
    S: FnMut(Bound<&[u8]>, &mut dyn FnMut(Vec<u8>, Vec<u8>) -> Result<bool>) -> Result<()>,
{
    const BATCH_BYTES: usize = 8 * 1024 * 1024;
    const BATCH_KEYS: usize = 20_000;
    let out = Database::open(
        dest,
        Options {
            sync_on_commit: false,
            tracing: false,
            ..options
        },
    )?;
    let mut cursor: Option<Vec<u8>> = None;
    loop {
        let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut bytes = 0usize;
        let lo = match &cursor {
            Some(c) => Bound::Excluded(c.as_slice()),
            None => Bound::Unbounded,
        };
        scan(lo, &mut |k, v| {
            bytes += k.len() + v.len();
            batch.push((k, v));
            Ok(bytes < BATCH_BYTES && batch.len() < BATCH_KEYS)
        })?;
        let Some((last, _)) = batch.last() else { break };
        cursor = Some(last.clone());
        out.write_batch(|b| {
            for (k, v) in &batch {
                b.put(k, v)?;
            }
            Ok(())
        })?;
    }
    // The checkpoint's journaled flush fsyncs the file; the log it leaves is
    // a bare marker, so the result is self-contained.
    out.checkpoint()?;
    Ok(())
}

/// Smallest key greater than every key starting with `prefix`, or `None`
/// when no such key exists (empty prefix, or all `0xFF`).
#[must_use]
pub fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.pop() {
        if last < 0xFF {
            end.push(last + 1);
            return Some(end);
        }
    }
    None
}

fn sidecar(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    PathBuf::from(s)
}

/// Removes a database file and its WAL/journal sidecars, ignoring absence.
fn remove_database_files(path: &Path) {
    let _ = std::fs::remove_file(path);
    remove_sidecars(path);
}

/// Removes only the WAL/journal sidecars of a database file.
fn remove_sidecars(path: &Path) {
    for p in [
        Database::wal_path(path),
        sidecar(&Database::wal_path(path), ".tmp"),
        Pager::journal_path(path),
    ] {
        let _ = std::fs::remove_file(p);
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// Fails if `path` is a database some handle currently has open.
fn refuse_if_open(path: &Path) -> Result<()> {
    let Ok(file) = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
    else {
        return Ok(()); // does not exist (or cannot be opened): nothing to clobber
    };
    match file.try_lock() {
        Ok(()) => Ok(()),
        Err(std::fs::TryLockError::WouldBlock) => Err(Error::Busy(format!(
            "{} is an open database; close it before overwriting it",
            path.display()
        ))),
        Err(std::fs::TryLockError::Error(_)) => Ok(()),
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        if self.crashed {
            return;
        }
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
            // No checkpoint, no merge: simulates a crash.
            db.simulate_crash();
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
            db.put_auto(format!("k{i:03}").as_bytes(), b"initial")
                .unwrap();
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
                    db.put_auto(format!("k{i:03}").as_bytes(), b"updated")
                        .unwrap();
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
    fn scan_iter_streams_without_full_materialization() {
        let (_d, db) = open_temp();
        for i in 0..40u32 {
            db.put_auto(format!("k{i:03}").as_bytes(), b"v").unwrap();
        }
        db.delete_auto(b"k005").unwrap();

        let mut streamed = Vec::new();
        db.scan_iter(|(key, value)| {
            streamed.push((key, value));
            Ok(())
        })
        .unwrap();

        assert_eq!(streamed.len(), 39);
        for w in streamed.windows(2) {
            assert!(w[0].0 < w[1].0, "streamed scan is not ordered");
        }
        assert!(!streamed.iter().any(|(k, _)| k == b"k005"));
    }

    #[test]
    fn scan_iter_on_empty_database_yields_nothing() {
        let (_d, db) = open_temp();
        let mut streamed = Vec::new();
        db.scan_iter(|item| {
            streamed.push(item);
            Ok(())
        })
        .unwrap();
        assert!(streamed.is_empty());
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
