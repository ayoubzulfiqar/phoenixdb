//! MVCC transactions over the B+Tree.
//!
//! # Isolation model
//!
//! Snapshot isolation with a single writer:
//!
//! * Every transaction takes a **snapshot timestamp** at `begin`. Reads see the
//!   newest version whose `commit_ts <= snapshot` — plus the transaction's own
//!   uncommitted writes.
//! * Writers serialise on a global [`parking_lot::RwLock`], so at most one
//!   write transaction mutates state at a time while readers proceed freely.
//! * A write-write conflict (someone committed a version of a key after our
//!   snapshot) is rejected with [`Error::Conflict`] at commit time.
//!
//! # Version store
//!
//! Committed-but-not-yet-merged versions live in an in-memory version chain
//! keyed by user key. A background-free **merge** step folds versions whose
//! `commit_ts` is below the oldest live snapshot into the B+Tree, after which
//! they are dropped from memory and the WAL can be checkpointed.

use crate::btree::BTree;
use crate::error::{Error, Result};
use crate::pager::Pager;
use std::collections::{BTreeMap, HashMap, HashSet};

/// State of a transaction as seen by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    /// Accepting reads and writes.
    Active,
    /// Committed; its versions are visible to newer snapshots.
    Committed,
    /// Rolled back; its writes are discarded.
    Aborted,
}

/// One committed version of a key.
#[derive(Debug, Clone)]
pub struct Version {
    /// Timestamp at which this version became visible.
    pub commit_ts: u64,
    /// `None` represents a tombstone (deletion).
    pub value: Option<Vec<u8>>,
}

/// A pending write inside an active transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Write {
    /// Set the key to this value.
    Put(Vec<u8>),
    /// Delete the key.
    Delete,
}

/// An in-flight transaction.
#[derive(Debug)]
pub struct Transaction {
    /// Unique, monotonically increasing identifier.
    pub id: u64,
    /// Snapshot timestamp: versions with `commit_ts <= snapshot` are visible.
    pub snapshot: u64,
    /// Current state.
    pub state: TxnState,
    /// Uncommitted writes, applied over the snapshot on read.
    pub writes: BTreeMap<Vec<u8>, Write>,
    /// Keys read by this transaction (for conflict diagnostics).
    pub reads: HashSet<Vec<u8>>,
    /// True when the transaction is read-only (no writer lock needed).
    pub read_only: bool,
}

impl Transaction {
    fn new(id: u64, snapshot: u64, read_only: bool) -> Self {
        Transaction {
            id,
            snapshot,
            state: TxnState::Active,
            writes: BTreeMap::new(),
            reads: HashSet::new(),
            read_only,
        }
    }

    /// Number of buffered writes.
    #[must_use]
    pub fn write_count(&self) -> usize {
        self.writes.len()
    }
}

/// The multi-version store plus the transaction registry.
///
/// Guarded by the engine's `RwLock`; this type contains no locking of its own.
#[derive(Debug, Default)]
pub struct VersionStore {
    /// Per-key version chains, newest last.
    chains: HashMap<Vec<u8>, Vec<Version>>,
    /// Next commit timestamp to hand out.
    next_ts: u64,
    /// Next transaction id to hand out.
    next_txn_id: u64,
    /// Live transactions by id.
    active: HashMap<u64, Transaction>,
    /// Snapshots of live transactions, for computing the merge watermark.
    live_snapshots: BTreeMap<u64, usize>,
}

impl VersionStore {
    /// Creates a store seeded with the ids recovered from disk.
    #[must_use]
    pub fn new(next_ts: u64, next_txn_id: u64) -> Self {
        VersionStore {
            chains: HashMap::new(),
            next_ts: next_ts.max(1),
            next_txn_id: next_txn_id.max(1),
            active: HashMap::new(),
            live_snapshots: BTreeMap::new(),
        }
    }

    /// Latest committed timestamp handed out.
    #[must_use]
    pub fn current_ts(&self) -> u64 {
        self.next_ts.saturating_sub(1)
    }

    /// Next transaction id that will be assigned.
    #[must_use]
    pub fn peek_txn_id(&self) -> u64 {
        self.next_txn_id
    }

    /// Number of live transactions.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Starts a transaction and registers its snapshot.
    pub fn begin(&mut self, read_only: bool) -> u64 {
        let id = self.next_txn_id;
        self.next_txn_id += 1;
        let snapshot = self.current_ts();
        *self.live_snapshots.entry(snapshot).or_insert(0) += 1;
        self.active
            .insert(id, Transaction::new(id, snapshot, read_only));
        id
    }

    /// Borrows a live transaction.
    pub fn get(&self, id: u64) -> Result<&Transaction> {
        self.active
            .get(&id)
            .filter(|t| t.state == TxnState::Active)
            .ok_or(Error::TxnNotFound(id))
    }

    /// Mutably borrows a live transaction.
    pub fn get_mut(&mut self, id: u64) -> Result<&mut Transaction> {
        self.active
            .get_mut(&id)
            .filter(|t| t.state == TxnState::Active)
            .ok_or(Error::TxnNotFound(id))
    }

    /// Buffers a write in transaction `id`.
    pub fn stage(&mut self, id: u64, key: Vec<u8>, write: Write) -> Result<()> {
        let txn = self.get_mut(id)?;
        if txn.read_only {
            return Err(Error::invalid("cannot write in a read-only transaction"));
        }
        txn.writes.insert(key, write);
        Ok(())
    }

    /// Reads `key` as of transaction `id`, consulting its own writes first.
    ///
    /// `Ok(None)` means "no version visible" (never written, or tombstoned) and
    /// the caller should fall through to the B+Tree.
    pub fn read(&self, id: u64, key: &[u8]) -> Result<Option<Option<Vec<u8>>>> {
        let txn = self.get(id)?;
        if let Some(w) = txn.writes.get(key) {
            return Ok(Some(match w {
                Write::Put(v) => Some(v.clone()),
                Write::Delete => None,
            }));
        }
        Ok(self.visible_version(key, txn.snapshot))
    }

    /// Newest committed version of `key` visible at `snapshot`.
    fn visible_version(&self, key: &[u8], snapshot: u64) -> Option<Option<Vec<u8>>> {
        let chain = self.chains.get(key)?;
        chain
            .iter()
            .rev()
            .find(|v| v.commit_ts <= snapshot)
            .map(|v| v.value.clone())
    }

    /// Detects a write-write conflict for transaction `id`.
    ///
    /// A conflict exists when any key we wrote has a committed version newer
    /// than our snapshot: someone else won the race.
    pub fn detect_conflict(&self, id: u64) -> Result<()> {
        let txn = self.get(id)?;
        for key in txn.writes.keys() {
            if let Some(chain) = self.chains.get(key)
                && chain.iter().any(|v| v.commit_ts > txn.snapshot)
            {
                return Err(Error::Conflict);
            }
        }
        Ok(())
    }

    /// Commits transaction `id`, publishing its writes at a fresh timestamp.
    ///
    /// The caller must already have made the WAL `Commit` record durable.
    pub fn commit(&mut self, id: u64) -> Result<u64> {
        self.detect_conflict(id)?;
        let commit_ts = self.next_ts;
        self.next_ts += 1;

        let txn = self
            .active
            .get_mut(&id)
            .filter(|t| t.state == TxnState::Active)
            .ok_or(Error::TxnNotFound(id))?;
        let writes = std::mem::take(&mut txn.writes);
        let snapshot = txn.snapshot;
        txn.state = TxnState::Committed;

        for (key, w) in writes {
            let version = Version {
                commit_ts,
                value: match w {
                    Write::Put(v) => Some(v),
                    Write::Delete => None,
                },
            };
            self.chains.entry(key).or_default().push(version);
        }
        self.release_snapshot(snapshot);
        self.active.remove(&id);
        Ok(commit_ts)
    }

    /// Aborts transaction `id`, discarding its writes.
    pub fn rollback(&mut self, id: u64) -> Result<()> {
        let txn = self
            .active
            .get_mut(&id)
            .filter(|t| t.state == TxnState::Active)
            .ok_or(Error::TxnNotFound(id))?;
        txn.state = TxnState::Aborted;
        txn.writes.clear();
        let snapshot = txn.snapshot;
        self.release_snapshot(snapshot);
        self.active.remove(&id);
        Ok(())
    }

    fn release_snapshot(&mut self, snapshot: u64) {
        if let Some(count) = self.live_snapshots.get_mut(&snapshot) {
            *count -= 1;
            if *count == 0 {
                self.live_snapshots.remove(&snapshot);
            }
        }
    }

    /// Oldest snapshot any live transaction can still see.
    ///
    /// Versions at or below this watermark are invisible to every future reader
    /// and can be merged into the tree and dropped.
    #[must_use]
    pub fn merge_watermark(&self) -> u64 {
        self.live_snapshots
            .keys()
            .next()
            .copied()
            .unwrap_or_else(|| self.current_ts())
    }

    /// Number of keys with in-memory versions.
    #[must_use]
    pub fn pending_keys(&self) -> usize {
        self.chains.len()
    }

    /// Folds versions at or below `watermark` into the B+Tree.
    ///
    /// Returns the number of keys merged. The caller is responsible for
    /// flushing the pager and checkpointing the WAL afterwards.
    pub fn merge_into_tree(
        &mut self,
        tree: &BTree,
        pager: &mut Pager,
        watermark: u64,
    ) -> Result<usize> {
        let mut merged = 0usize;
        let mut empty_keys: Vec<Vec<u8>> = Vec::new();

        for (key, chain) in self.chains.iter_mut() {
            // Find the newest version that is safe to materialise.
            let Some(idx) = chain.iter().rposition(|v| v.commit_ts <= watermark) else {
                continue;
            };
            let version = chain[idx].clone();
            match &version.value {
                Some(v) => tree.insert(pager, key, v)?,
                None => match tree.delete(pager, key) {
                    Ok(()) | Err(Error::NotFound) => {}
                    Err(e) => return Err(e),
                },
            }
            // Drop everything we just superseded.
            chain.drain(..=idx);
            merged += 1;
            if chain.is_empty() {
                empty_keys.push(key.clone());
            }
        }
        for k in empty_keys {
            self.chains.remove(&k);
        }
        Ok(merged)
    }

    /// Applies a recovered committed write directly to the version store.
    ///
    /// Used during WAL replay before any transaction exists.
    pub fn apply_recovered(&mut self, commit_ts: u64, key: Vec<u8>, value: Option<Vec<u8>>) {
        self.chains
            .entry(key)
            .or_default()
            .push(Version { commit_ts, value });
        if commit_ts >= self.next_ts {
            self.next_ts = commit_ts + 1;
        }
    }

    /// Bumps the transaction-id counter past a recovered value.
    pub fn observe_txn_id(&mut self, id: u64) {
        if id >= self.next_txn_id {
            self.next_txn_id = id + 1;
        }
    }

    /// Every key that currently has an in-memory version, for scans.
    pub fn keys_with_versions(&self, snapshot: u64) -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
        let mut out = Vec::new();
        for (key, chain) in &self.chains {
            if let Some(v) = chain.iter().rev().find(|v| v.commit_ts <= snapshot) {
                out.push((key.clone(), v.value.clone()));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_writes_are_visible_to_the_writer() {
        let mut vs = VersionStore::new(1, 1);
        let t = vs.begin(false);
        vs.stage(t, b"k".to_vec(), Write::Put(b"v".to_vec())).unwrap();
        assert_eq!(vs.read(t, b"k").unwrap(), Some(Some(b"v".to_vec())));
    }

    #[test]
    fn snapshot_isolation_hides_later_commits() {
        let mut vs = VersionStore::new(1, 1);
        let writer = vs.begin(false);
        vs.stage(writer, b"k".to_vec(), Write::Put(b"v1".to_vec()))
            .unwrap();
        vs.commit(writer).unwrap();

        let reader = vs.begin(true); // snapshot includes v1

        let writer2 = vs.begin(false);
        vs.stage(writer2, b"k".to_vec(), Write::Put(b"v2".to_vec()))
            .unwrap();
        vs.commit(writer2).unwrap();

        // The reader still sees v1.
        assert_eq!(vs.read(reader, b"k").unwrap(), Some(Some(b"v1".to_vec())));

        // A brand-new transaction sees v2.
        let reader2 = vs.begin(true);
        assert_eq!(vs.read(reader2, b"k").unwrap(), Some(Some(b"v2".to_vec())));
    }

    #[test]
    fn write_write_conflict_is_detected() {
        let mut vs = VersionStore::new(1, 1);
        let a = vs.begin(false);
        let b = vs.begin(false);
        vs.stage(a, b"k".to_vec(), Write::Put(b"a".to_vec())).unwrap();
        vs.stage(b, b"k".to_vec(), Write::Put(b"b".to_vec())).unwrap();
        vs.commit(a).unwrap();
        assert!(matches!(vs.commit(b), Err(Error::Conflict)));
    }

    #[test]
    fn rollback_discards_writes() {
        let mut vs = VersionStore::new(1, 1);
        let t = vs.begin(false);
        vs.stage(t, b"k".to_vec(), Write::Put(b"v".to_vec())).unwrap();
        vs.rollback(t).unwrap();
        assert!(matches!(vs.get(t), Err(Error::TxnNotFound(_))));
        let t2 = vs.begin(true);
        assert_eq!(vs.read(t2, b"k").unwrap(), None);
    }

    #[test]
    fn tombstones_hide_older_versions() {
        let mut vs = VersionStore::new(1, 1);
        let a = vs.begin(false);
        vs.stage(a, b"k".to_vec(), Write::Put(b"v".to_vec())).unwrap();
        vs.commit(a).unwrap();
        let b = vs.begin(false);
        vs.stage(b, b"k".to_vec(), Write::Delete).unwrap();
        vs.commit(b).unwrap();
        let r = vs.begin(true);
        assert_eq!(vs.read(r, b"k").unwrap(), Some(None), "expected a tombstone");
    }

    #[test]
    fn read_only_txn_cannot_write() {
        let mut vs = VersionStore::new(1, 1);
        let t = vs.begin(true);
        assert!(matches!(
            vs.stage(t, b"k".to_vec(), Write::Put(b"v".to_vec())),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn unknown_txn_is_reported() {
        let vs = VersionStore::new(1, 1);
        assert!(matches!(vs.get(999), Err(Error::TxnNotFound(999))));
    }

    #[test]
    fn watermark_tracks_the_oldest_live_snapshot() {
        let mut vs = VersionStore::new(1, 1);
        let a = vs.begin(false);
        vs.stage(a, b"k".to_vec(), Write::Put(b"1".to_vec())).unwrap();
        vs.commit(a).unwrap();
        let old = vs.begin(true);
        let b = vs.begin(false);
        vs.stage(b, b"k".to_vec(), Write::Put(b"2".to_vec())).unwrap();
        vs.commit(b).unwrap();
        assert!(vs.merge_watermark() <= 1, "old reader must pin the watermark");
        vs.rollback(old).unwrap();
        assert_eq!(vs.merge_watermark(), vs.current_ts());
    }

    #[test]
    fn merge_moves_versions_into_the_tree() {
        use crate::btree::{BTree, FillFactor};
        let dir = tempfile::tempdir().unwrap();
        let mut pager = Pager::open(&dir.path().join("t.pdb"), 64).unwrap();
        let tree = BTree::new(FillFactor::default());

        let mut vs = VersionStore::new(1, 1);
        let t = vs.begin(false);
        vs.stage(t, b"a".to_vec(), Write::Put(b"1".to_vec())).unwrap();
        vs.stage(t, b"b".to_vec(), Write::Put(b"2".to_vec())).unwrap();
        vs.commit(t).unwrap();

        let wm = vs.merge_watermark();
        let merged = vs.merge_into_tree(&tree, &mut pager, wm).unwrap();
        assert_eq!(merged, 2);
        assert_eq!(vs.pending_keys(), 0);
        assert_eq!(tree.get(&mut pager, b"a").unwrap(), b"1");
        assert_eq!(tree.get(&mut pager, b"b").unwrap(), b"2");
    }
}
