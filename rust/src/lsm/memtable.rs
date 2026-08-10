//! The in-memory write buffer of the LSM tree.
//!
//! # Role in the hybrid engine
//!
//! Every write lands here first (after its WAL record is durable), making the
//! write path a single ordered-map insert instead of a B+Tree descent with
//! page splits. When a MemTable exceeds its size budget it is *frozen* —
//! becoming immutable — and handed to the flush path, which turns it into an
//! [`SSTable`](super::sstable::SSTable). A fresh MemTable takes over
//! immediately, so writers never block on a flush.
//!
//! # MVCC key encoding
//!
//! Entries are keyed by [`InternalKey`] = `(user_key, seqno)` and ordered by
//! `user_key` ascending, then `seqno` **descending**. That ordering means a
//! forward scan positioned at `(key, snapshot_seq)` sees the newest version
//! visible to that snapshot first, so a point read is one `range` call with no
//! post-filtering.
//!
//! # Tombstones
//!
//! A delete writes `None` rather than removing the entry: older versions of the
//! same key may still live in lower SSTable levels, and only a tombstone that
//! is merged *through* those levels can hide them. Tombstones are physically
//! dropped during compaction of the bottom-most level (see
//! [`super::compaction`]).

use std::collections::BTreeMap;
use std::collections::btree_map::Range;
use std::ops::Bound;

/// A versioned key: the user's bytes plus the sequence number that wrote them.
///
/// `Ord` sorts by `user_key` ascending then `seqno` **descending**, so the
/// newest version of a key always sorts first within that key's run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalKey {
    /// The key as supplied by the caller.
    pub user_key: Vec<u8>,
    /// Monotonic sequence number of the write that produced this version.
    pub seqno: u64,
}

impl InternalKey {
    /// Creates an internal key.
    #[must_use]
    pub fn new(user_key: Vec<u8>, seqno: u64) -> Self {
        InternalKey { user_key, seqno }
    }

    /// The smallest internal key for `user_key` (highest possible seqno).
    ///
    /// Used as the inclusive lower bound of a snapshot lookup.
    #[must_use]
    pub fn seek_max(user_key: Vec<u8>) -> Self {
        InternalKey {
            user_key,
            seqno: u64::MAX,
        }
    }
}

impl Ord for InternalKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Key ascending, then seqno descending (newest first).
        self.user_key
            .cmp(&other.user_key)
            .then_with(|| other.seqno.cmp(&self.seqno))
    }
}

impl PartialOrd for InternalKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// What a version says about a key: a value, or its removal.
///
/// `None` is a tombstone.
pub type ValueSlot = Option<Vec<u8>>;

/// Outcome of a MemTable point lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lookup {
    /// A live value visible to the snapshot.
    Found(Vec<u8>),
    /// A tombstone visible to the snapshot: the key is deleted, and lower
    /// levels **must not** be consulted.
    Deleted,
    /// No version of this key exists here; fall through to the next level.
    Absent,
}

/// Per-entry bookkeeping overhead charged against the size budget.
///
/// Covers the `BTreeMap` node share, the `InternalKey` struct, and the
/// `Option<Vec<u8>>` header. Deliberately conservative: over-estimating flushes
/// slightly early, under-estimating risks an OOM under write pressure.
const ENTRY_OVERHEAD: usize = 64;

/// A sorted, in-memory batch of versioned writes.
#[derive(Debug, Default)]
pub struct MemTable {
    map: BTreeMap<InternalKey, ValueSlot>,
    /// Approximate heap bytes held by keys, values and per-entry overhead.
    size_bytes: usize,
    /// Lowest sequence number present, for flush-ordering diagnostics.
    min_seqno: u64,
    /// Highest sequence number present; becomes the SSTable's upper bound.
    max_seqno: u64,
}

impl MemTable {
    /// Creates an empty MemTable.
    #[must_use]
    pub fn new() -> Self {
        MemTable {
            map: BTreeMap::new(),
            size_bytes: 0,
            min_seqno: u64::MAX,
            max_seqno: 0,
        }
    }

    /// Approximate heap usage in bytes — the trigger for a flush.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Number of versioned entries (not distinct user keys).
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True when no writes have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Lowest sequence number held, or `None` when empty.
    #[must_use]
    pub fn min_seqno(&self) -> Option<u64> {
        (!self.map.is_empty()).then_some(self.min_seqno)
    }

    /// Highest sequence number held, or `None` when empty.
    #[must_use]
    pub fn max_seqno(&self) -> Option<u64> {
        (!self.map.is_empty()).then_some(self.max_seqno)
    }

    /// Records `key = value` at `seqno`.
    pub fn insert(&mut self, key: Vec<u8>, value: Vec<u8>, seqno: u64) {
        self.put_slot(key, Some(value), seqno);
    }

    /// Records a tombstone for `key` at `seqno`.
    pub fn delete(&mut self, key: Vec<u8>, seqno: u64) {
        self.put_slot(key, None, seqno);
    }

    fn put_slot(&mut self, key: Vec<u8>, slot: ValueSlot, seqno: u64) {
        let charge = key.len() + slot.as_ref().map_or(0, Vec::len) + ENTRY_OVERHEAD;
        let ik = InternalKey::new(key, seqno);
        // Re-writing the same (key, seqno) replaces in place: refund the old charge.
        if let Some(previous) = self.map.insert(ik.clone(), slot) {
            let refund = ik.user_key.len() + previous.as_ref().map_or(0, Vec::len) + ENTRY_OVERHEAD;
            self.size_bytes = self.size_bytes.saturating_sub(refund);
        }
        self.size_bytes += charge;
        self.min_seqno = self.min_seqno.min(seqno);
        self.max_seqno = self.max_seqno.max(seqno);
    }

    /// Reads `key` as visible to `snapshot`.
    ///
    /// Returns the newest version whose `seqno <= snapshot`, distinguishing a
    /// tombstone ([`Lookup::Deleted`]) from "not here" ([`Lookup::Absent`]) so
    /// the caller knows whether to search lower levels.
    #[must_use]
    pub fn get(&self, key: &[u8], snapshot: u64) -> Lookup {
        let lower = Bound::Included(InternalKey::seek_max(key.to_vec()));
        let upper = Bound::Excluded(InternalKey::seek_max({
            // The successor of `key`: first internal key of the next user key.
            let mut next = key.to_vec();
            next.push(0);
            next
        }));
        let range: Range<'_, InternalKey, ValueSlot> = self.map.range((lower, upper));
        for (ik, slot) in range {
            if ik.user_key != key {
                break; // left this key's run
            }
            if ik.seqno <= snapshot {
                // First match in seqno-descending order is the newest visible.
                return match slot {
                    Some(v) => Lookup::Found(v.clone()),
                    None => Lookup::Deleted,
                };
            }
        }
        Lookup::Absent
    }

    /// Every entry in `(user_key asc, seqno desc)` order.
    ///
    /// This is exactly the order [`super::sstable::SSTableWriter`] requires.
    pub fn iter(&self) -> impl Iterator<Item = (&InternalKey, &ValueSlot)> {
        self.map.iter()
    }

    /// Collapses to one entry per user key as visible to `snapshot`.
    ///
    /// Tombstones are retained (as `None`) because callers merging against
    /// lower levels still need them to mask older versions.
    #[must_use]
    pub fn snapshot_view(&self, snapshot: u64) -> Vec<(Vec<u8>, ValueSlot)> {
        let mut out: Vec<(Vec<u8>, ValueSlot)> = Vec::new();
        let mut current: Option<&[u8]> = None;
        for (ik, slot) in &self.map {
            if ik.seqno > snapshot {
                continue; // invisible to this snapshot
            }
            if current == Some(ik.user_key.as_slice()) {
                continue; // already took the newest visible version
            }
            current = Some(ik.user_key.as_slice());
            out.push((ik.user_key.clone(), slot.clone()));
        }
        out
    }

    /// Smallest and largest user keys held, for SSTable range metadata.
    #[must_use]
    pub fn key_range(&self) -> Option<(Vec<u8>, Vec<u8>)> {
        let first = self.map.keys().next()?.user_key.clone();
        let last = self.map.keys().next_back()?.user_key.clone();
        Some((first, last))
    }

    /// Distinct user keys, for sizing a Bloom filter before a flush.
    #[must_use]
    pub fn distinct_key_count(&self) -> usize {
        let mut count = 0usize;
        let mut current: Option<&[u8]> = None;
        for ik in self.map.keys() {
            if current != Some(ik.user_key.as_slice()) {
                current = Some(ik.user_key.as_slice());
                count += 1;
            }
        }
        count
    }

    /// Empties the table, returning its entries in sorted order.
    ///
    /// Used by the flush path, which consumes a frozen table exactly once.
    #[must_use]
    pub fn drain(&mut self) -> Vec<(InternalKey, ValueSlot)> {
        self.size_bytes = 0;
        self.min_seqno = u64::MAX;
        self.max_seqno = 0;
        std::mem::take(&mut self.map).into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newest_version_wins_within_a_snapshot() {
        let mut m = MemTable::new();
        m.insert(b"k".to_vec(), b"v1".to_vec(), 1);
        m.insert(b"k".to_vec(), b"v2".to_vec(), 5);
        m.insert(b"k".to_vec(), b"v3".to_vec(), 9);

        assert_eq!(m.get(b"k", 9), Lookup::Found(b"v3".to_vec()));
        assert_eq!(m.get(b"k", 5), Lookup::Found(b"v2".to_vec()));
        assert_eq!(m.get(b"k", 4), Lookup::Found(b"v1".to_vec()));
        assert_eq!(
            m.get(b"k", 0),
            Lookup::Absent,
            "snapshot predates every write"
        );
    }

    #[test]
    fn tombstone_is_distinct_from_absent() {
        let mut m = MemTable::new();
        m.insert(b"k".to_vec(), b"v".to_vec(), 1);
        m.delete(b"k".to_vec(), 2);

        assert_eq!(m.get(b"k", 2), Lookup::Deleted);
        assert_eq!(m.get(b"k", 1), Lookup::Found(b"v".to_vec()));
        assert_eq!(m.get(b"missing", 99), Lookup::Absent);
    }

    #[test]
    fn ordering_is_key_asc_seqno_desc() {
        let mut m = MemTable::new();
        m.insert(b"b".to_vec(), b"1".to_vec(), 1);
        m.insert(b"a".to_vec(), b"2".to_vec(), 2);
        m.insert(b"a".to_vec(), b"3".to_vec(), 7);

        let order: Vec<(Vec<u8>, u64)> = m
            .iter()
            .map(|(ik, _)| (ik.user_key.clone(), ik.seqno))
            .collect();
        assert_eq!(
            order,
            vec![(b"a".to_vec(), 7), (b"a".to_vec(), 2), (b"b".to_vec(), 1),]
        );
    }

    #[test]
    fn prefix_keys_do_not_bleed_into_each_other() {
        // "k" must not match "kk" — the range upper bound has to be exact.
        let mut m = MemTable::new();
        m.insert(b"kk".to_vec(), b"long".to_vec(), 1);
        assert_eq!(m.get(b"k", 99), Lookup::Absent);

        m.insert(b"k".to_vec(), b"short".to_vec(), 2);
        assert_eq!(m.get(b"k", 99), Lookup::Found(b"short".to_vec()));
        assert_eq!(m.get(b"kk", 99), Lookup::Found(b"long".to_vec()));
    }

    #[test]
    fn size_accounting_tracks_payload() {
        let mut m = MemTable::new();
        assert_eq!(m.size_bytes(), 0);
        m.insert(b"key".to_vec(), vec![0u8; 100], 1);
        assert_eq!(m.size_bytes(), 3 + 100 + ENTRY_OVERHEAD);

        // Overwriting the same (key, seqno) must not double-count.
        m.insert(b"key".to_vec(), vec![0u8; 50], 1);
        assert_eq!(m.size_bytes(), 3 + 50 + ENTRY_OVERHEAD);
    }

    #[test]
    fn snapshot_view_collapses_versions() {
        let mut m = MemTable::new();
        m.insert(b"a".to_vec(), b"old".to_vec(), 1);
        m.insert(b"a".to_vec(), b"new".to_vec(), 10);
        m.insert(b"b".to_vec(), b"only".to_vec(), 2);
        m.delete(b"c".to_vec(), 3);

        let view = m.snapshot_view(10);
        assert_eq!(
            view,
            vec![
                (b"a".to_vec(), Some(b"new".to_vec())),
                (b"b".to_vec(), Some(b"only".to_vec())),
                (b"c".to_vec(), None),
            ]
        );

        // An older snapshot sees the older value and no "c" at all.
        let view = m.snapshot_view(2);
        assert_eq!(
            view,
            vec![
                (b"a".to_vec(), Some(b"old".to_vec())),
                (b"b".to_vec(), Some(b"only".to_vec())),
            ]
        );
    }

    #[test]
    fn key_range_and_distinct_count() {
        let mut m = MemTable::new();
        assert!(m.key_range().is_none());
        m.insert(b"m".to_vec(), b"1".to_vec(), 1);
        m.insert(b"a".to_vec(), b"2".to_vec(), 2);
        m.insert(b"z".to_vec(), b"3".to_vec(), 3);
        m.insert(b"a".to_vec(), b"4".to_vec(), 4);

        assert_eq!(m.key_range(), Some((b"a".to_vec(), b"z".to_vec())));
        assert_eq!(m.distinct_key_count(), 3);
        assert_eq!(m.len(), 4, "versions are counted separately");
    }

    #[test]
    fn drain_empties_and_returns_sorted() {
        let mut m = MemTable::new();
        m.insert(b"b".to_vec(), b"2".to_vec(), 2);
        m.insert(b"a".to_vec(), b"1".to_vec(), 1);
        let drained = m.drain();

        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0.user_key, b"a");
        assert_eq!(drained[1].0.user_key, b"b");
        assert!(m.is_empty());
        assert_eq!(m.size_bytes(), 0);
        assert!(m.max_seqno().is_none());
    }

    #[test]
    fn seqno_bounds_track_inserts() {
        let mut m = MemTable::new();
        assert!(m.min_seqno().is_none());
        m.insert(b"a".to_vec(), b"v".to_vec(), 42);
        m.insert(b"b".to_vec(), b"v".to_vec(), 7);
        assert_eq!(m.min_seqno(), Some(7));
        assert_eq!(m.max_seqno(), Some(42));
    }

    #[test]
    fn empty_key_and_empty_value_are_handled() {
        let mut m = MemTable::new();
        m.insert(Vec::new(), Vec::new(), 1);
        assert_eq!(m.get(b"", 1), Lookup::Found(Vec::new()));
    }
}
