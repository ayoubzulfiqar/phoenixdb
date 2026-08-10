//! Hybrid LSM storage layer.
//!
//! # Why a hybrid
//!
//! The existing B+Tree ([`crate::btree`]) is excellent for point reads on hot
//! keys: a bounded descent over cached pages, no merge cost. It is poor for
//! write bursts, where every insert can split pages and dirty the cache.
//!
//! The LSM layer inverts that trade: writes are appended to an in-memory
//! [`MemTable`] and flushed as immutable, sorted [`SSTable`] runs, giving
//! sequential I/O and no in-place page churn. Reads pay a merge cost across
//! levels, which Bloom filters and per-table key ranges make cheap for the
//! common "key is absent from this level" case.
//!
//! PhoenixDB runs both. Writes land in the LSM; the B+Tree remains the
//! authoritative store for data that has been checkpointed through it. A read
//! consults the levels newest-first and falls through to the tree only when no
//! LSM level answers definitively:
//!
//! ```text
//!   write ──▶ WAL (durable) ──▶ MemTable ──┐
//!                                          │ size trigger
//!                                          ▼
//!                                  frozen MemTable
//!                                          │ flush
//!                                          ▼
//!   read ──▶ MemTable ──▶ L0 ──▶ L1 ─▶ … ─▶ Ln ──▶ B+Tree
//!             (newest)                              (oldest)
//! ```
//!
//! # Read resolution
//!
//! Each level returns one of three answers ([`Lookup`]): `Found` and `Deleted`
//! are both **definitive** and stop the search — a tombstone must mask older
//! versions in lower levels. Only `Absent` continues to the next level. Getting
//! this wrong resurrects deleted keys, so every level's `get` is tested for the
//! tombstone case explicitly.
//!
//! # Sequence numbers
//!
//! The LSM's `seqno` is the engine's MVCC commit timestamp
//! ([`crate::txn::VersionStore`]), so a snapshot read at timestamp `T`
//! translates directly into "the newest version with `seqno <= T`" and
//! isolation semantics match the B+Tree path exactly.

pub mod bloom;
pub mod compaction;
pub mod engine;
pub mod manifest;
pub mod memtable;
pub mod sstable;

pub use bloom::BloomFilter;
pub use compaction::{CompactionJob, CompactionStats, LevelConfig, LevelManifest};
pub use engine::{LsmEngine, LsmOptions, LsmStats};
pub use memtable::{InternalKey, Lookup, MemTable, ValueSlot};
pub use sstable::{SSTable, SSTableWriter, TableMeta};
