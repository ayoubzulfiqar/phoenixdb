//! Embedded vector search: HNSW k-NN over memory-mapped `f32` vectors.
//!
//! ```text
//!   Dart  ->  ffi/vector_ffi.rs  ->  VectorEngine
//!                                        |
//!                        +---------------+---------------+
//!                        |               |               |
//!                    hnsw.rs        store.rs        distance.rs
//!                  (graph index)  (mmap + WAL)   (SIMD kernels)
//! ```
//!
//! # Design
//!
//! * **Raw vectors never live in the graph.** `store.rs` owns the bytes in an
//!   append-only, memory-mapped file; the graph holds only ids and links. That
//!   split is what lets the graph be snapshotted with `bincode` independently
//!   of the (much larger) vector payload.
//! * **One `RwLock` around the mutable state.** Searches take the read lock and
//!   run concurrently; inserts take the write lock. The lock is `parking_lot`'s,
//!   matching the rest of the engine.
//! * **Ordering distances only.** Everything internal compares squared L2 for
//!   the Euclidean metric; the square root is applied once, to the `k` results
//!   that leave the engine.
//!
//! # Example
//!
//! ```no_run
//! use phoenixdb::vector::{Metric, VectorEngine, VectorOptions};
//!
//! # fn main() -> phoenixdb::Result<()> {
//! let engine = VectorEngine::open("vectors.pvec", 3, Metric::Cosine, VectorOptions::default())?;
//! engine.insert("doc-1", &[1.0, 0.0, 0.0])?;
//! engine.insert("doc-2", &[0.0, 1.0, 0.0])?;
//! let hits = engine.search(&[0.9, 0.1, 0.0], 1, None)?;
//! assert_eq!(hits[0].id, "doc-1");
//! engine.save(None)?;
//! # Ok(())
//! # }
//! ```

pub mod distance;
pub mod hnsw;
pub mod store;

pub use distance::{MAX_DIM, Metric};
pub use hnsw::HnswParams;
pub use store::{MAX_ID_LEN, VectorRecord, VectorStore};

use crate::error::{Error, Result};
use hnsw::{DistanceSource, HnswGraph};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Magic prefix of a graph snapshot: "PHNXHNS1".
const SNAPSHOT_MAGIC: u64 = 0x5048_4E58_484E_5331;

/// Below this many live vectors, search runs exhaustively.
///
/// An HNSW graph over a handful of points is all approximation and no speedup:
/// the greedy descent has nothing to descend through, while a linear scan of
/// 500 vectors is a fraction of a millisecond. Switching automatically means a
/// small collection is *exact*, which is what a caller inserting ten documents
/// expects to see.
const BRUTE_FORCE_THRESHOLD: usize = 512;

/// One search result.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorMatch {
    /// The id supplied at insert time.
    pub id: String,
    /// Metric distance; smaller is nearer.
    pub distance: f32,
    /// Convenience "higher is better" score derived from [`Metric::score`].
    pub score: f32,
}

/// Engine tuning.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VectorOptions {
    /// Graph construction and search parameters.
    pub hnsw: HnswParams,
    /// `fsync` the vector file on every insert.
    ///
    /// Off by default: an embedded, local-first store is normally rebuilt from
    /// its source of truth, and syncing per insert costs an order of magnitude
    /// on a bulk load. [`VectorEngine::save`] and [`VectorEngine::flush`] are
    /// the durability points.
    pub sync_on_insert: bool,
    /// Expected capacity, used to pre-reserve the id map.
    pub max_elements: usize,
}

/// Runtime statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorStats {
    /// Live (non-tombstoned) vectors.
    pub live: usize,
    /// Records on disk, tombstones included.
    pub total: usize,
    /// Tombstoned records.
    pub deleted: usize,
    /// Dimensionality.
    pub dim: usize,
    /// Height of the tallest graph node.
    pub max_level: usize,
    /// Bytes written since the last flush.
    pub dirty_bytes: u64,
}

/// The serialisable half of the index.
#[derive(Debug, Serialize, Deserialize)]
struct Snapshot {
    magic: u64,
    version: u32,
    dim: usize,
    metric: Metric,
    graph: HnswGraph,
    /// `id -> ordinal`, so a reload does not have to rescan the vector file.
    ids: Vec<(String, u32)>,
}

/// Everything behind the engine lock.
struct Inner {
    store: VectorStore,
    graph: HnswGraph,
    /// External id to record ordinal.
    ids: HashMap<String, u32>,
    live: usize,
}

/// A thread-safe embedded vector index.
pub struct VectorEngine {
    inner: RwLock<Inner>,
    dim: usize,
    metric: Metric,
    options: VectorOptions,
    path: PathBuf,
    /// Set by [`VectorEngine::simulate_crash`]: skip the save in `Drop`.
    pub(crate) crashed: bool,
}

/// Adapts the store to the graph's [`DistanceSource`], holding one query.
///
/// Constructed per operation and borrows the store, so the graph reads vectors
/// straight out of the mapping with no copy.
struct StoreDistances<'a> {
    store: &'a VectorStore,
    metric: Metric,
    query: &'a [f32],
    query_norm: f32,
}

impl DistanceSource for StoreDistances<'_> {
    fn to_query(&self, id: u32) -> f32 {
        match self.store.vector_at(id) {
            Some(vector) => distance::ordering_distance(
                self.metric,
                self.query,
                self.query_norm,
                vector,
                self.store.norm_at(id).unwrap_or(0.0),
            ),
            // A missing record sorts last rather than panicking: it can only
            // happen if the graph outran the store, and refusing the whole
            // query would be a worse failure than omitting one node.
            None => f32::INFINITY,
        }
    }

    fn between(&self, a: u32, b: u32) -> f32 {
        match (self.store.vector_at(a), self.store.vector_at(b)) {
            (Some(x), Some(y)) => distance::ordering_distance(
                self.metric,
                x,
                self.store.norm_at(a).unwrap_or(0.0),
                y,
                self.store.norm_at(b).unwrap_or(0.0),
            ),
            _ => f32::INFINITY,
        }
    }

    fn is_deleted(&self, id: u32) -> bool {
        self.store.is_deleted(id)
    }
}

impl VectorEngine {
    /// Opens (creating if necessary) the index at `path`.
    ///
    /// A sibling `<path>.hnsw` snapshot is loaded when present and consistent;
    /// otherwise the graph is rebuilt from the vector file, which is always
    /// possible because the vectors are the source of truth.
    pub fn open(
        path: impl AsRef<Path>,
        dim: usize,
        metric: Metric,
        options: VectorOptions,
    ) -> Result<Self> {
        distance::validate_dim(dim)?;
        let hnsw = options.hnsw.validate()?;
        let options = VectorOptions { hnsw, ..options };

        let path = path.as_ref().to_path_buf();

        // A `.compact` file is a compaction that crashed before its atomic
        // rename: the original store is intact and authoritative.
        let _ = std::fs::remove_file(Self::compact_path(&path));

        let mut store = VectorStore::open(&path, dim, metric)?;
        let mut ids: HashMap<String, u32> =
            HashMap::with_capacity(options.max_elements.min(1 << 20).max(store.len()));
        let mut graph = HnswGraph::new(hnsw)?;
        let mut live = 0usize;

        // A snapshot describes a prefix of the store: everything up to its
        // last save. Anything appended after that (a crash skipped the final
        // save) is caught up incrementally instead of rebuilding the whole
        // graph.
        let snapshot_path = Self::snapshot_path(&path);
        let mut next = 0u32;
        if let Some(snapshot) = Self::load_snapshot(&snapshot_path, dim, metric, store.len())? {
            next = snapshot.graph.len() as u32;
            graph = snapshot.graph;
            ids.extend(snapshot.ids);
            live = (0..next)
                .filter(|ordinal| !store.is_deleted(*ordinal))
                .count();
        }

        // Replay the records the graph does not cover. Each is CRC-checked,
        // and the first torn one ends the replay: nothing after it can have
        // been synced (a sync would have made it durable too), so cutting the
        // tail there never drops a durable record.
        for ordinal in next..store.len() as u32 {
            let record = match store.record_at(ordinal) {
                Ok(r) => r,
                Err(Error::Corruption(_)) => {
                    store.truncate_to(ordinal as usize)?;
                    break;
                }
                Err(e) => return Err(e),
            };
            if !record.deleted {
                // A lost tombstone (unsynced before a crash) would leave two
                // live records for one id; the newer one wins.
                if let Some(previous) = ids.get(&record.id).copied()
                    && !store.is_deleted(previous)
                {
                    store.tombstone(previous)?;
                    live -= 1;
                }
                live += 1;
            }
            ids.insert(record.id.clone(), ordinal);
            let source = StoreDistances {
                store: &store,
                metric,
                query: &record.vector,
                query_norm: record.norm,
            };
            graph.insert(ordinal, &source)?;
        }

        Ok(VectorEngine {
            inner: RwLock::new(Inner {
                store,
                graph,
                ids,
                live,
            }),
            dim,
            metric,
            options,
            path,
            crashed: false,
        })
    }

    /// Path of the graph snapshot that accompanies a vector file.
    #[must_use]
    pub fn snapshot_path(vector_path: &Path) -> PathBuf {
        let mut s = vector_path.as_os_str().to_os_string();
        s.push(".hnsw");
        PathBuf::from(s)
    }

    /// Path of the temporary file a compaction builds.
    fn compact_path(vector_path: &Path) -> PathBuf {
        let mut s = vector_path.as_os_str().to_os_string();
        s.push(".compact");
        PathBuf::from(s)
    }

    /// Reads a snapshot, returning `None` when it is absent or unusable.
    ///
    /// A stale or mismatched snapshot is *not* an error: the vector file can
    /// always rebuild the graph, so recovering silently beats refusing to open
    /// the index. A snapshot covering fewer records than the store holds is
    /// usable (the rest is caught up); one covering more is not (the store was
    /// truncated or replaced).
    fn load_snapshot(
        path: &Path,
        dim: usize,
        metric: Metric,
        store_len: usize,
    ) -> Result<Option<Snapshot>> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(Error::Io(e)),
        };
        // Framed as [crc32 of payload][payload]; older unframed snapshots
        // fail the check and are simply rebuilt once.
        if bytes.len() < 4 {
            return Ok(None);
        }
        let (crc, payload) = bytes.split_at(4);
        if crc32fast::hash(payload) != u32::from_le_bytes([crc[0], crc[1], crc[2], crc[3]]) {
            return Ok(None);
        }
        let snapshot: Snapshot = match bincode::deserialize(payload) {
            Ok(s) => s,
            Err(_) => return Ok(None), // corrupt snapshot: rebuild instead
        };
        let usable = snapshot.magic == SNAPSHOT_MAGIC
            && snapshot.version == store::VECTOR_FORMAT_VERSION
            && snapshot.dim == dim
            && snapshot.metric == metric
            // More nodes than records means the vector file was truncated or
            // replaced: the vectors win and the graph is rebuilt.
            && snapshot.graph.len() <= store_len
            && snapshot.ids.iter().all(|(_, ordinal)| (*ordinal as usize) < snapshot.graph.len());
        Ok(usable.then_some(snapshot))
    }

    /// Dimensionality of this index.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Metric this index orders by.
    #[inline]
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Path of the vector file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Options in force.
    #[must_use]
    pub fn options(&self) -> VectorOptions {
        self.options
    }

    /// Live vector count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.read().live
    }

    /// True when no live vector remains.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `id` is present and not tombstoned.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        let inner = self.inner.read();
        inner
            .ids
            .get(id)
            .is_some_and(|ordinal| !inner.store.is_deleted(*ordinal))
    }

    /// Runtime statistics.
    #[must_use]
    pub fn stats(&self) -> VectorStats {
        let inner = self.inner.read();
        VectorStats {
            live: inner.live,
            total: inner.store.len(),
            deleted: inner.store.len() - inner.live,
            dim: self.dim,
            max_level: inner.graph.max_level(),
            dirty_bytes: inner.store.dirty_bytes(),
        }
    }

    /// Name of the SIMD kernel this CPU selected. For diagnostics.
    #[must_use]
    pub fn kernel() -> &'static str {
        distance::active_kernel()
    }
}

impl VectorEngine {
    /// Inserts or replaces `id`.
    ///
    /// Replacement tombstones the old record and appends a new one, so an
    /// overwrite never invalidates an id already embedded in the graph. The
    /// space is reclaimed by [`VectorEngine::compact`].
    pub fn insert(&self, id: &str, vector: &[f32]) -> Result<()> {
        self.insert_many(std::slice::from_ref(&(id, vector)))
    }

    /// Inserts a batch under a single lock acquisition.
    ///
    /// A bulk load through [`VectorEngine::insert`] pays lock traffic per
    /// vector; this pays it once. Validation happens for the whole batch
    /// before anything is written, so a bad vector cannot leave a partial
    /// batch on disk.
    pub fn insert_many(&self, items: &[(&str, &[f32])]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        for (id, vector) in items {
            if id.is_empty() {
                return Err(Error::invalid("vector id must not be empty"));
            }
            if id.len() > MAX_ID_LEN {
                return Err(Error::invalid(format!(
                    "vector id of {} bytes exceeds the {MAX_ID_LEN}-byte limit",
                    id.len()
                )));
            }
            distance::validate_vector(vector, self.dim)?;
        }

        let mut inner = self.inner.write();
        for (id, vector) in items {
            // Replacing: tombstone first so the graph stops returning the old
            // record the moment the new one is visible.
            if let Some(previous) = inner.ids.get(*id).copied()
                && !inner.store.is_deleted(previous)
            {
                inner.store.tombstone(previous)?;
                inner.live -= 1;
            }

            let norm = if self.metric.uses_norm() {
                distance::norm(vector)
            } else {
                0.0
            };
            let ordinal = inner
                .store
                .append(id, vector, norm, self.options.sync_on_insert)?;

            // Graph and store share one id space, so the ordinal the store
            // just handed out must be the graph's next node.
            let Inner { store, graph, .. } = &mut *inner;
            let source = StoreDistances {
                store,
                metric: self.metric,
                query: vector,
                query_norm: norm,
            };
            graph.insert(ordinal, &source)?;

            inner.ids.insert((*id).to_string(), ordinal);
            inner.live += 1;
        }
        Ok(())
    }

    /// Returns the `k` nearest live vectors to `query`.
    ///
    /// `ef` overrides the beam width for this query only: higher means better
    /// recall and more work. Passing `None` uses the configured default.
    ///
    /// Collections below [`BRUTE_FORCE_THRESHOLD`] live vectors are scanned
    /// exhaustively, so a small index returns exact results.
    pub fn search(&self, query: &[f32], k: usize, ef: Option<usize>) -> Result<Vec<VectorMatch>> {
        distance::validate_vector(query, self.dim)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let inner = self.inner.read();
        if inner.live == 0 {
            return Ok(Vec::new());
        }

        let query_norm = if self.metric.uses_norm() {
            distance::norm(query)
        } else {
            0.0
        };
        let source = StoreDistances {
            store: &inner.store,
            metric: self.metric,
            query,
            query_norm,
        };

        let raw = if inner.live <= BRUTE_FORCE_THRESHOLD {
            inner.graph.brute_force(k, &source)
        } else {
            inner.graph.search(k, ef, &source)
        };
        Ok(self.to_matches(&inner, raw))
    }

    /// Returns the `k` nearest live vectors whose id satisfies `allow`.
    ///
    /// Adaptive: the graph is searched with a widened beam, skipping rejected
    /// nodes, and if that cannot fill `k` results (a selective filter) the
    /// accepted set is scanned exactly instead — so a filter never costs
    /// recall, only time.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        allow: &dyn Fn(&str) -> bool,
    ) -> Result<Vec<VectorMatch>> {
        distance::validate_vector(query, self.dim)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let inner = self.inner.read();
        if inner.live == 0 {
            return Ok(Vec::new());
        }
        let source = self.source_for(&inner, query);
        let store = &inner.store;
        let accept = |ordinal: u32| store.id_at(ordinal).is_some_and(|id| allow(&id));
        let raw = self.search_accepting(&inner, &source, k, ef, &accept);
        Ok(self.to_matches(&inner, raw))
    }

    /// Exact `k` nearest among the given ids (unknown or deleted ids are
    /// ignored). Ideal when a metadata filter has already narrowed the
    /// candidates: cost is `O(ids.len())` and recall is perfect.
    ///
    /// Large candidate sets switch to a filtered graph search, which is
    /// cheaper once the set is a sizeable fraction of the index.
    pub fn search_ids(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        ids: &[&str],
    ) -> Result<Vec<VectorMatch>> {
        distance::validate_vector(query, self.dim)?;
        if k == 0 || ids.is_empty() {
            return Ok(Vec::new());
        }
        let inner = self.inner.read();
        let ordinals: Vec<u32> = ids
            .iter()
            .filter_map(|id| inner.ids.get(*id).copied())
            .filter(|o| !inner.store.is_deleted(*o))
            .collect();
        if ordinals.is_empty() {
            return Ok(Vec::new());
        }
        let source = self.source_for(&inner, query);
        let raw = if ordinals.len() <= BRUTE_FORCE_THRESHOLD.max(inner.live / 8) {
            inner.graph.brute_force_among(k, &ordinals, &source)
        } else {
            let set: std::collections::HashSet<u32> = ordinals.iter().copied().collect();
            self.search_accepting(&inner, &source, k, ef, &|o| set.contains(&o))
        };
        Ok(self.to_matches(&inner, raw))
    }

    /// Runs several queries under one lock acquisition.
    pub fn search_batch(
        &self,
        queries: &[&[f32]],
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<Vec<VectorMatch>>> {
        for q in queries {
            distance::validate_vector(q, self.dim)?;
        }
        let inner = self.inner.read();
        let mut out = Vec::with_capacity(queries.len());
        for query in queries {
            if k == 0 || inner.live == 0 {
                out.push(Vec::new());
                continue;
            }
            let source = self.source_for(&inner, query);
            let raw = if inner.live <= BRUTE_FORCE_THRESHOLD {
                inner.graph.brute_force(k, &source)
            } else {
                inner.graph.search(k, ef, &source)
            };
            out.push(self.to_matches(&inner, raw));
        }
        Ok(out)
    }

    /// Filtered search strategy shared by the filtered entry points.
    fn search_accepting(
        &self,
        inner: &Inner,
        source: &StoreDistances<'_>,
        k: usize,
        ef: Option<usize>,
        accept: &dyn Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        if inner.live <= BRUTE_FORCE_THRESHOLD {
            return inner.graph.brute_force_where(k, source, accept);
        }
        // Rejected nodes still occupy the beam, so widen it.
        let widened = ef
            .unwrap_or(self.options.hnsw.ef_search)
            .max(k.saturating_mul(4))
            .min(4096);
        let found = inner.graph.search_where(k, Some(widened), source, accept);
        if found.len() >= k {
            return found;
        }
        // Too selective for the graph to fill `k`: the exact answer is a scan
        // of the accepted set.
        inner.graph.brute_force_where(k, source, accept)
    }

    fn source_for<'a>(&self, inner: &'a Inner, query: &'a [f32]) -> StoreDistances<'a> {
        StoreDistances {
            store: &inner.store,
            metric: self.metric,
            query,
            query_norm: if self.metric.uses_norm() {
                distance::norm(query)
            } else {
                0.0
            },
        }
    }

    /// Converts raw `(ordinal, ordering distance)` pairs into results.
    fn to_matches(&self, inner: &Inner, raw: Vec<(u32, f32)>) -> Vec<VectorMatch> {
        let mut matches = Vec::with_capacity(raw.len());
        for (ordinal, ordering_distance) in raw {
            // A record whose id cannot be read is skipped rather than faked:
            // returning a placeholder id would be a silent data error.
            let Some(id) = inner.store.id_at(ordinal) else {
                continue;
            };
            let distance = self.metric.finalize(ordering_distance);
            matches.push(VectorMatch {
                id,
                distance,
                score: self.metric.score(distance),
            });
        }
        matches
    }

    /// Ids of every live vector (unordered).
    #[must_use]
    pub fn ids(&self) -> Vec<String> {
        let inner = self.inner.read();
        inner
            .ids
            .iter()
            .filter(|(_, o)| !inner.store.is_deleted(**o))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Fetches a stored vector by id.
    pub fn get(&self, id: &str) -> Result<Vec<f32>> {
        let inner = self.inner.read();
        let ordinal = inner.ids.get(id).copied().ok_or(Error::NotFound)?;
        if inner.store.is_deleted(ordinal) {
            return Err(Error::NotFound);
        }
        inner
            .store
            .vector_at(ordinal)
            .map(<[f32]>::to_vec)
            .ok_or(Error::NotFound)
    }

    /// Tombstones `id`. Returns [`Error::NotFound`] when it is absent.
    ///
    /// The record's bytes stay on disk so graph ids never shift; the space is
    /// reclaimed by [`VectorEngine::compact`].
    pub fn remove(&self, id: &str) -> Result<()> {
        let mut inner = self.inner.write();
        let ordinal = inner.ids.get(id).copied().ok_or(Error::NotFound)?;
        if inner.store.is_deleted(ordinal) {
            return Err(Error::NotFound);
        }
        inner.store.tombstone(ordinal)?;
        inner.live -= 1;
        Ok(())
    }

    /// Writes the graph snapshot and syncs the vector file.
    ///
    /// `path` overrides the default `<vector file>.hnsw` location. The write
    /// is atomic: a temporary file is written and `fsync`ed, then renamed, so
    /// a crash mid-save leaves the previous snapshot intact rather than a
    /// half-written one.
    pub fn save(&self, path: Option<&Path>) -> Result<()> {
        let mut inner = self.inner.write();
        self.save_locked(&mut inner, path)
    }

    fn save_locked(&self, inner: &mut Inner, path: Option<&Path>) -> Result<()> {
        inner.store.sync()?;

        let mut ids: Vec<(String, u32)> = inner
            .ids
            .iter()
            .map(|(id, ordinal)| (id.clone(), *ordinal))
            .collect();
        // Sorted so a snapshot is byte-identical for identical content, which
        // makes it diffable and cacheable.
        ids.sort_unstable();

        let snapshot = Snapshot {
            magic: SNAPSHOT_MAGIC,
            version: store::VECTOR_FORMAT_VERSION,
            dim: self.dim,
            metric: self.metric,
            graph: inner.graph.clone(),
            ids,
        };
        let payload = bincode::serialize(&snapshot)?;
        let mut bytes = Vec::with_capacity(payload.len() + 4);
        bytes.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let target = match path {
            Some(p) => p.to_path_buf(),
            None => Self::snapshot_path(&self.path),
        };
        if let Some(parent) = target.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let mut temporary = target.as_os_str().to_os_string();
        temporary.push(".tmp");
        let temporary = PathBuf::from(temporary);
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        // `rename` replaces the target atomically on every platform
        // (MoveFileExW with REPLACE_EXISTING on Windows): a crash leaves the
        // old snapshot or the new one, never neither.
        std::fs::rename(&temporary, &target)?;
        crate::fsutil::sync_parent_dir(&target);
        Ok(())
    }

    /// Syncs the vector file without writing a snapshot.
    pub fn flush(&self) -> Result<()> {
        self.inner.write().store.sync()
    }

    /// Rewrites the index without tombstoned records.
    ///
    /// Ordinals are renumbered, so the graph is rebuilt from scratch. Cost is
    /// `O(N log N)`; call it when the tombstone ratio justifies the work.
    pub fn compact(&self) -> Result<usize> {
        let mut inner = self.inner.write();
        let dead = inner.store.len() - inner.live;
        if dead == 0 {
            return Ok(0);
        }

        // Collect the survivors before touching anything on disk, so a failure
        // during the scan leaves the original index untouched.
        let mut survivors: Vec<(String, Vec<f32>, f32)> = Vec::with_capacity(inner.live);
        for ordinal in 0..inner.store.len() as u32 {
            if inner.store.is_deleted(ordinal) {
                continue;
            }
            let record = inner.store.record_at(ordinal)?;
            survivors.push((record.id, record.vector, record.norm));
        }

        let temporary = Self::compact_path(&self.path);
        let _ = std::fs::remove_file(&temporary);

        let mut rebuilt = VectorStore::open(&temporary, self.dim, self.metric)?;
        let mut graph = HnswGraph::new(self.options.hnsw)?;
        let mut ids = HashMap::with_capacity(survivors.len());
        let built = (|| -> Result<()> {
            for (id, vector, norm) in &survivors {
                let ordinal = rebuilt.append(id, vector, *norm, false)?;
                let source = StoreDistances {
                    store: &rebuilt,
                    metric: self.metric,
                    query: vector,
                    query_norm: *norm,
                };
                graph.insert(ordinal, &source)?;
                ids.insert(id.clone(), ordinal);
            }
            rebuilt.sync()
        })();
        rebuilt.close();
        if let Err(e) = built {
            let _ = std::fs::remove_file(&temporary);
            return Err(e);
        }

        // Swap: release the live store (its mapping and file lock — Windows
        // cannot replace a mapped or open file), then atomically rename the
        // rebuilt file over it. A crash before the rename leaves the old
        // store (and a stray `.compact`, removed on the next open); after it,
        // the new one. Never neither.
        inner.store.close();
        if let Err(e) = std::fs::rename(&temporary, &self.path) {
            // The original is untouched: reopen it and report.
            inner.store = VectorStore::open(&self.path, self.dim, self.metric)?;
            let _ = std::fs::remove_file(&temporary);
            return Err(Error::Io(e));
        }
        crate::fsutil::sync_parent_dir(&self.path);
        inner.store = VectorStore::open(&self.path, self.dim, self.metric)?;
        inner.graph = graph;
        inner.ids = ids;
        inner.live = survivors.len();
        // Snapshot now, so the next open does not rebuild the new graph.
        self.save_locked(&mut inner, None)?;
        Ok(dead)
    }

    /// Flushes and snapshots. Called by `Drop` and the FFI free function.
    pub fn close(&self) -> Result<()> {
        self.save(None)
    }
}

impl VectorEngine {
    /// Drops the engine the way a crash would: no final save, so only what
    /// the vector file and the last snapshot already hold survives. Releases
    /// the file lock so the same process can reopen and exercise recovery.
    #[doc(hidden)]
    pub fn simulate_crash(mut self) {
        self.crashed = true;
    }
}

impl Drop for VectorEngine {
    fn drop(&mut self) {
        if self.crashed {
            return;
        }
        // Best effort: a failing save must not panic in `Drop`, because that
        // would unwind across the FFI boundary.
        let _ = self.save(None);
    }
}

impl std::fmt::Debug for VectorEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorEngine")
            .field("path", &self.path)
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_engine(dim: usize, metric: Metric) -> (tempfile::TempDir, VectorEngine) {
        let dir = tempfile::tempdir().unwrap();
        let engine = VectorEngine::open(
            dir.path().join("v.pvec"),
            dim,
            metric,
            VectorOptions::default(),
        )
        .unwrap();
        (dir, engine)
    }

    /// Deterministic points, so a failure reproduces exactly.
    fn cloud(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        ((state >> 40) as f32 / 8_388_608.0) - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn insert_then_search_finds_the_exact_match_first() {
        let (_d, engine) = temp_engine(4, Metric::Cosine);
        engine.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        engine.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        engine.insert("c", &[0.0, 0.0, 1.0, 0.0]).unwrap();

        let hits = engine.search(&[1.0, 0.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].id, "a");
        assert!(
            hits[0].distance.abs() < 1e-6,
            "identical vector, distance 0"
        );
        assert!((hits[0].score - 1.0).abs() < 1e-6, "cosine similarity 1");
        // The other two are orthogonal, so both sit at distance 1.
        assert!((hits[1].distance - 1.0).abs() < 1e-5);
        assert!((hits[2].distance - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_ignores_magnitude() {
        let (_d, engine) = temp_engine(3, Metric::Cosine);
        engine.insert("unit", &[1.0, 0.0, 0.0]).unwrap();
        engine.insert("scaled", &[100.0, 0.0, 0.0]).unwrap();
        let hits = engine.search(&[0.001, 0.0, 0.0], 2, None).unwrap();
        // Same direction, wildly different magnitudes: both at distance 0.
        for hit in &hits {
            assert!(
                hit.distance.abs() < 1e-5,
                "{} should be at cosine distance 0, got {}",
                hit.id,
                hit.distance
            );
        }
    }

    #[test]
    fn euclidean_reports_true_distance_not_the_square() {
        let (_d, engine) = temp_engine(2, Metric::Euclidean);
        engine.insert("origin", &[0.0, 0.0]).unwrap();
        let hits = engine.search(&[3.0, 4.0], 1, None).unwrap();
        // 3-4-5 triangle: the ordering distance is 25, the reported one is 5.
        assert!(
            (hits[0].distance - 5.0).abs() < 1e-5,
            "expected 5.0, got {}",
            hits[0].distance
        );
    }

    #[test]
    fn dot_product_ranks_by_inner_product() {
        let (_d, engine) = temp_engine(2, Metric::DotProduct);
        engine.insert("small", &[1.0, 1.0]).unwrap();
        engine.insert("large", &[5.0, 5.0]).unwrap();
        let hits = engine.search(&[1.0, 1.0], 2, None).unwrap();
        // Unlike cosine, magnitude matters: 10 beats 2.
        assert_eq!(hits[0].id, "large");
        assert!((hits[0].score - 10.0).abs() < 1e-5);
        assert!((hits[1].score - 2.0).abs() < 1e-5);
    }

    #[test]
    fn dimension_mismatch_is_rejected_on_insert_and_search() {
        let (_d, engine) = temp_engine(4, Metric::Cosine);
        assert!(engine.insert("short", &[1.0, 2.0]).is_err());
        assert!(engine.insert("long", &[1.0; 9]).is_err());
        assert!(engine.insert("ok", &[1.0; 4]).is_ok());
        assert!(engine.search(&[1.0, 2.0], 1, None).is_err());
        assert!(engine.search(&[1.0; 4], 1, None).is_ok());
    }

    #[test]
    fn non_finite_components_are_rejected() {
        // A single NaN makes every comparison against it false, silently
        // corrupting the ordering of results for unrelated queries.
        let (_d, engine) = temp_engine(3, Metric::Euclidean);
        assert!(engine.insert("nan", &[1.0, f32::NAN, 3.0]).is_err());
        assert!(engine.insert("inf", &[1.0, f32::INFINITY, 3.0]).is_err());
        assert!(engine.search(&[1.0, f32::NAN, 3.0], 1, None).is_err());
        assert_eq!(engine.len(), 0, "nothing should have been stored");
    }

    #[test]
    fn empty_and_oversized_ids_are_rejected() {
        let (_d, engine) = temp_engine(2, Metric::Cosine);
        assert!(engine.insert("", &[1.0, 0.0]).is_err());
        assert!(
            engine
                .insert(&"x".repeat(MAX_ID_LEN + 1), &[1.0, 0.0])
                .is_err()
        );
        assert!(engine.insert(&"y".repeat(MAX_ID_LEN), &[1.0, 0.0]).is_ok());
    }

    #[test]
    fn reinserting_an_id_replaces_it() {
        let (_d, engine) = temp_engine(2, Metric::Euclidean);
        engine.insert("k", &[0.0, 0.0]).unwrap();
        engine.insert("k", &[10.0, 10.0]).unwrap();
        assert_eq!(engine.len(), 1, "replacement must not double-count");
        assert_eq!(engine.get("k").unwrap(), vec![10.0, 10.0]);

        // The stale record must not surface as a second hit.
        let hits = engine.search(&[0.0, 0.0], 5, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "k");
    }

    #[test]
    fn remove_tombstones_and_hides_the_vector() {
        let (_d, engine) = temp_engine(2, Metric::Euclidean);
        engine.insert("gone", &[1.0, 1.0]).unwrap();
        engine.insert("stays", &[2.0, 2.0]).unwrap();
        assert!(engine.contains("gone"));

        engine.remove("gone").unwrap();
        assert!(!engine.contains("gone"));
        assert_eq!(engine.len(), 1);
        assert!(matches!(engine.get("gone"), Err(Error::NotFound)));
        assert!(matches!(engine.remove("gone"), Err(Error::NotFound)));
        assert!(matches!(
            engine.remove("never-existed"),
            Err(Error::NotFound)
        ));

        let hits = engine.search(&[1.0, 1.0], 5, None).unwrap();
        assert!(hits.iter().all(|h| h.id != "gone"));
    }

    #[test]
    fn results_are_ordered_by_ascending_distance() {
        let (_d, engine) = temp_engine(8, Metric::Cosine);
        for (index, point) in cloud(50, 8, 17).into_iter().enumerate() {
            engine.insert(&format!("v{index}"), &point).unwrap();
        }
        let query = vec![0.5f32; 8];
        let hits = engine.search(&query, 10, None).unwrap();
        assert_eq!(hits.len(), 10);
        for window in hits.windows(2) {
            assert!(
                window[0].distance <= window[1].distance,
                "unsorted: {:?}",
                hits.iter().map(|h| h.distance).collect::<Vec<_>>()
            );
            assert!(window[0].score >= window[1].score, "score must mirror rank");
        }
    }

    #[test]
    fn small_collections_are_searched_exactly() {
        // Below the brute-force threshold the answer is exact, not
        // approximate, which is what a caller with ten documents expects.
        let points = cloud(60, 16, 31);
        let (_d, engine) = temp_engine(16, Metric::Euclidean);
        for (index, point) in points.iter().enumerate() {
            engine.insert(&format!("v{index}"), point).unwrap();
        }
        let query = &points[7];
        let hits = engine.search(query, 1, None).unwrap();
        assert_eq!(hits[0].id, "v7");
        assert!(hits[0].distance.abs() < 1e-4);
    }

    #[test]
    fn k_zero_and_empty_index_return_nothing() {
        let (_d, engine) = temp_engine(4, Metric::Cosine);
        assert!(engine.search(&[1.0; 4], 0, None).unwrap().is_empty());
        assert!(engine.search(&[1.0; 4], 10, None).unwrap().is_empty());
        assert!(engine.is_empty());

        engine.insert("a", &[1.0; 4]).unwrap();
        assert!(engine.search(&[1.0; 4], 0, None).unwrap().is_empty());
    }

    #[test]
    fn k_larger_than_the_collection_is_clamped() {
        let (_d, engine) = temp_engine(2, Metric::Cosine);
        engine.insert("a", &[1.0, 0.0]).unwrap();
        engine.insert("b", &[0.0, 1.0]).unwrap();
        assert_eq!(engine.search(&[1.0, 0.0], 100, None).unwrap().len(), 2);
    }

    #[test]
    fn batch_insert_matches_individual_inserts() {
        let points = cloud(40, 8, 55);
        let (_d1, one) = temp_engine(8, Metric::Cosine);
        let (_d2, many) = temp_engine(8, Metric::Cosine);

        let ids: Vec<String> = (0..points.len()).map(|i| format!("v{i}")).collect();
        for (id, point) in ids.iter().zip(&points) {
            one.insert(id, point).unwrap();
        }
        let batch: Vec<(&str, &[f32])> = ids
            .iter()
            .zip(&points)
            .map(|(id, p)| (id.as_str(), p.as_slice()))
            .collect();
        many.insert_many(&batch).unwrap();

        assert_eq!(one.len(), many.len());
        let query = vec![0.25f32; 8];
        assert_eq!(
            one.search(&query, 5, None).unwrap(),
            many.search(&query, 5, None).unwrap()
        );
    }

    #[test]
    fn a_rejected_batch_writes_nothing() {
        let (_d, engine) = temp_engine(4, Metric::Cosine);
        let good = vec![1.0f32; 4];
        let bad = vec![1.0f32; 2];
        let batch: Vec<(&str, &[f32])> = vec![("a", good.as_slice()), ("b", bad.as_slice())];
        assert!(engine.insert_many(&batch).is_err());
        assert_eq!(engine.len(), 0, "validation must precede any write");
    }

    #[test]
    fn index_survives_reopen_via_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        let points = cloud(80, 16, 71);
        let query = vec![0.1f32; 16];
        let before;
        {
            let engine =
                VectorEngine::open(&path, 16, Metric::Cosine, VectorOptions::default()).unwrap();
            for (index, point) in points.iter().enumerate() {
                engine.insert(&format!("v{index}"), point).unwrap();
            }
            before = engine.search(&query, 5, None).unwrap();
            engine.save(None).unwrap();
        }
        assert!(
            VectorEngine::snapshot_path(&path).exists(),
            "save must write the graph snapshot"
        );

        let engine =
            VectorEngine::open(&path, 16, Metric::Cosine, VectorOptions::default()).unwrap();
        assert_eq!(engine.len(), 80);
        assert_eq!(engine.search(&query, 5, None).unwrap(), before);
    }

    #[test]
    fn index_rebuilds_when_the_snapshot_is_missing_or_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        {
            let engine =
                VectorEngine::open(&path, 4, Metric::Euclidean, VectorOptions::default()).unwrap();
            for i in 0..30u32 {
                engine
                    .insert(&format!("v{i}"), &[i as f32, 0.0, 0.0, 0.0])
                    .unwrap();
            }
            engine.save(None).unwrap();
        }
        // A corrupt snapshot must not be fatal: the vectors are the source of
        // truth and can always rebuild the graph.
        std::fs::write(VectorEngine::snapshot_path(&path), b"not a snapshot").unwrap();
        let engine =
            VectorEngine::open(&path, 4, Metric::Euclidean, VectorOptions::default()).unwrap();
        assert_eq!(engine.len(), 30);
        assert_eq!(
            engine.search(&[7.0, 0.0, 0.0, 0.0], 1, None).unwrap()[0].id,
            "v7"
        );

        // Same again with no snapshot at all.
        drop(engine);
        std::fs::remove_file(VectorEngine::snapshot_path(&path)).unwrap();
        let engine =
            VectorEngine::open(&path, 4, Metric::Euclidean, VectorOptions::default()).unwrap();
        assert_eq!(engine.len(), 30);
        assert_eq!(
            engine.search(&[7.0, 0.0, 0.0, 0.0], 1, None).unwrap()[0].id,
            "v7"
        );
    }

    #[test]
    fn tombstones_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        {
            let engine =
                VectorEngine::open(&path, 2, Metric::Euclidean, VectorOptions::default()).unwrap();
            engine.insert("a", &[1.0, 1.0]).unwrap();
            engine.insert("b", &[2.0, 2.0]).unwrap();
            engine.remove("a").unwrap();
            engine.save(None).unwrap();
        }
        let engine =
            VectorEngine::open(&path, 2, Metric::Euclidean, VectorOptions::default()).unwrap();
        assert_eq!(engine.len(), 1);
        assert!(!engine.contains("a"));
        assert!(engine.contains("b"));
    }

    #[test]
    fn compact_reclaims_tombstoned_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        let engine =
            VectorEngine::open(&path, 4, Metric::Euclidean, VectorOptions::default()).unwrap();
        for i in 0..40u32 {
            engine
                .insert(&format!("v{i}"), &[i as f32, 1.0, 2.0, 3.0])
                .unwrap();
        }
        for i in 0..15u32 {
            engine.remove(&format!("v{i}")).unwrap();
        }
        assert_eq!(engine.stats().total, 40);
        assert_eq!(engine.stats().deleted, 15);

        let reclaimed = engine.compact().unwrap();
        assert_eq!(reclaimed, 15);
        let stats = engine.stats();
        assert_eq!(stats.total, 25);
        assert_eq!(stats.live, 25);
        assert_eq!(stats.deleted, 0);

        // Search must still be correct after renumbering.
        assert_eq!(
            engine.search(&[20.0, 1.0, 2.0, 3.0], 1, None).unwrap()[0].id,
            "v20"
        );
        assert!(!engine.contains("v3"));
        // A second compact has nothing to do.
        assert_eq!(engine.compact().unwrap(), 0);
    }

    #[test]
    fn recall_stays_high_above_the_brute_force_threshold() {
        // The point at which the graph, not the linear scan, is answering.
        let count = BRUTE_FORCE_THRESHOLD + 300;
        let points = cloud(count, 24, 913);
        let (_d, engine) = temp_engine(24, Metric::Euclidean);
        let batch: Vec<(String, &Vec<f32>)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (format!("v{i}"), p))
            .collect();
        let refs: Vec<(&str, &[f32])> = batch
            .iter()
            .map(|(id, p)| (id.as_str(), p.as_slice()))
            .collect();
        engine.insert_many(&refs).unwrap();
        assert!(engine.len() > BRUTE_FORCE_THRESHOLD);

        let mut hits = 0usize;
        for query in points.iter().take(20) {
            let found = engine.search(query, 1, Some(128)).unwrap();
            // The nearest neighbour of a stored point is itself, at distance 0.
            if found[0].distance < 1e-3 {
                hits += 1;
            }
        }
        assert!(hits >= 19, "self-retrieval succeeded only {hits}/20 times");
    }

    #[test]
    fn stats_track_the_index() {
        let (_d, engine) = temp_engine(4, Metric::Cosine);
        for i in 0..10u32 {
            engine
                .insert(&format!("v{i}"), &[i as f32, 1.0, 1.0, 1.0])
                .unwrap();
        }
        engine.remove("v0").unwrap();
        let stats = engine.stats();
        assert_eq!(stats.live, 9);
        assert_eq!(stats.total, 10);
        assert_eq!(stats.deleted, 1);
        assert_eq!(stats.dim, 4);
    }

    #[test]
    fn concurrent_searches_run_while_inserts_proceed() {
        // The lock discipline claim: many readers, one writer, no deadlock and
        // no torn read.
        use std::sync::Arc;
        let dir = tempfile::tempdir().unwrap();
        let engine = Arc::new(
            VectorEngine::open(
                dir.path().join("v.pvec"),
                8,
                Metric::Cosine,
                VectorOptions::default(),
            )
            .unwrap(),
        );
        for (index, point) in cloud(100, 8, 5).into_iter().enumerate() {
            engine.insert(&format!("seed{index}"), &point).unwrap();
        }

        let mut handles = Vec::new();
        for worker in 0..4 {
            let engine = Arc::clone(&engine);
            handles.push(std::thread::spawn(move || {
                for round in 0..25 {
                    let query = vec![(worker as f32 + round as f32) * 0.01; 8];
                    let hits = engine.search(&query, 5, None).unwrap();
                    assert!(!hits.is_empty());
                }
            }));
        }
        let writer = {
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || {
                for i in 0..25u32 {
                    engine.insert(&format!("late{i}"), &[i as f32; 8]).unwrap();
                }
            })
        };
        for handle in handles {
            handle.join().unwrap();
        }
        writer.join().unwrap();
        assert_eq!(engine.len(), 125);
    }

    #[test]
    fn metric_mismatch_on_reopen_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        {
            let engine =
                VectorEngine::open(&path, 4, Metric::Cosine, VectorOptions::default()).unwrap();
            engine.insert("a", &[1.0; 4]).unwrap();
        }
        // Reopening under a different metric would silently reorder every
        // result, so it is an error rather than a reinterpretation.
        assert!(VectorEngine::open(&path, 4, Metric::Euclidean, VectorOptions::default()).is_err());
        assert!(VectorEngine::open(&path, 8, Metric::Cosine, VectorOptions::default()).is_err());
    }

    #[test]
    fn a_kernel_is_always_reported() {
        let kernel = VectorEngine::kernel();
        assert!(
            ["avx2+fma", "neon", "portable"].contains(&kernel),
            "unexpected kernel {kernel}"
        );
    }

    // ---- recovery, locking and filtered search -------------------------

    #[test]
    fn a_second_engine_on_the_same_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        let first = VectorEngine::open(&path, 4, Metric::Cosine, VectorOptions::default()).unwrap();
        let second = VectorEngine::open(&path, 4, Metric::Cosine, VectorOptions::default());
        assert!(matches!(second, Err(Error::Busy(_))), "got {second:?}");
        drop(first);
        VectorEngine::open(&path, 4, Metric::Cosine, VectorOptions::default()).unwrap();
    }

    #[test]
    fn a_crash_after_save_catches_up_from_the_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        let points = cloud(900, 8, 11);
        {
            let engine =
                VectorEngine::open(&path, 8, Metric::Euclidean, VectorOptions::default()).unwrap();
            for (i, p) in points.iter().enumerate().take(600) {
                engine.insert(&format!("p{i}"), p).unwrap();
            }
            engine.save(None).unwrap();
            for (i, p) in points.iter().enumerate().skip(600) {
                engine.insert(&format!("p{i}"), p).unwrap();
            }
            engine.remove("p3").unwrap();
            engine.flush().unwrap();
            engine.simulate_crash(); // no final save: the snapshot covers 600
        }
        let engine =
            VectorEngine::open(&path, 8, Metric::Euclidean, VectorOptions::default()).unwrap();
        assert_eq!(engine.len(), 899);
        assert!(!engine.contains("p3"));
        for i in [0usize, 599, 600, 899] {
            if i == 899 {
                continue;
            }
            let hit = &engine.search(&points[i], 1, Some(200)).unwrap()[0];
            assert_eq!(hit.id, format!("p{i}"), "record {i} must be searchable");
        }
    }

    #[test]
    fn a_torn_tail_record_is_cut_off_instead_of_failing_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        {
            let engine =
                VectorEngine::open(&path, 4, Metric::Cosine, VectorOptions::default()).unwrap();
            for i in 0..10 {
                engine
                    .insert(&format!("v{i}"), &[1.0, i as f32, 0.5, 0.25])
                    .unwrap();
            }
            engine.flush().unwrap();
            engine.simulate_crash();
        }
        let _ = std::fs::remove_file(VectorEngine::snapshot_path(&path));
        // Scribble over the last record's payload: an unsynced, torn write.
        {
            use std::io::{Seek, SeekFrom, Write};
            let stride = VectorStore::stride_for(4) as u64;
            let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(
                store::HEADER_LEN as u64 + 9 * stride + stride - 4,
            ))
            .unwrap();
            f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        }
        let engine = VectorEngine::open(&path, 4, Metric::Cosine, VectorOptions::default())
            .expect("a torn tail must not make the index unopenable");
        assert_eq!(engine.len(), 9);
        assert!(!engine.contains("v9"));
        engine.insert("v9", &[1.0, 9.0, 0.5, 0.25]).unwrap();
        assert_eq!(engine.len(), 10);
    }

    #[test]
    fn a_lost_tombstone_does_not_leave_two_live_copies_of_an_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        {
            let engine =
                VectorEngine::open(&path, 2, Metric::Euclidean, VectorOptions::default()).unwrap();
            engine.insert("a", &[1.0, 0.0]).unwrap();
            engine.insert("a", &[0.0, 1.0]).unwrap(); // tombstones record 0
            engine.flush().unwrap();
            engine.simulate_crash();
        }
        let _ = std::fs::remove_file(VectorEngine::snapshot_path(&path));
        // Undo the tombstone flag, as if its write never reached the disk.
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(store::HEADER_LEN as u64)).unwrap();
            f.write_all(&[0]).unwrap();
        }
        let engine =
            VectorEngine::open(&path, 2, Metric::Euclidean, VectorOptions::default()).unwrap();
        assert_eq!(engine.len(), 1);
        let hits = engine.search(&[0.0, 1.0], 5, None).unwrap();
        assert_eq!(hits.len(), 1, "one id, one result: {hits:?}");
        assert_eq!(
            engine.get("a").unwrap(),
            vec![0.0, 1.0],
            "the newer write wins"
        );
    }

    #[test]
    fn a_leftover_compaction_file_is_discarded_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        {
            let engine =
                VectorEngine::open(&path, 2, Metric::Cosine, VectorOptions::default()).unwrap();
            engine.insert("a", &[1.0, 0.0]).unwrap();
        }
        // A compaction that crashed before its rename.
        std::fs::write(VectorEngine::compact_path(&path), b"half-built").unwrap();
        let engine =
            VectorEngine::open(&path, 2, Metric::Cosine, VectorOptions::default()).unwrap();
        assert!(engine.contains("a"));
        assert!(!VectorEngine::compact_path(&path).exists());
    }

    #[test]
    fn compaction_survives_reopen_and_keeps_a_fresh_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.pvec");
        let points = cloud(700, 6, 5);
        {
            let engine =
                VectorEngine::open(&path, 6, Metric::Cosine, VectorOptions::default()).unwrap();
            for (i, p) in points.iter().enumerate() {
                engine.insert(&format!("p{i}"), p).unwrap();
            }
            for i in (0..700).step_by(2) {
                engine.remove(&format!("p{i}")).unwrap();
            }
            assert_eq!(engine.compact().unwrap(), 350);
            assert_eq!(engine.stats().total, 350);
            engine.simulate_crash(); // compact already saved the snapshot
        }
        let engine =
            VectorEngine::open(&path, 6, Metric::Cosine, VectorOptions::default()).unwrap();
        assert_eq!(engine.len(), 350);
        assert_eq!(engine.search(&points[1], 1, Some(200)).unwrap()[0].id, "p1");
    }

    #[test]
    fn filtered_search_matches_an_exact_filtered_scan() {
        let (_d, engine) = temp_engine(16, Metric::Euclidean);
        let points = cloud(3000, 16, 99);
        let items: Vec<(String, &[f32])> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (format!("doc-{i}"), p.as_slice()))
            .collect();
        let refs: Vec<(&str, &[f32])> = items.iter().map(|(id, p)| (id.as_str(), *p)).collect();
        engine.insert_many(&refs).unwrap();

        let even = |id: &str| id[4..].parse::<usize>().unwrap() % 2 == 0;
        let rare = |id: &str| id[4..].parse::<usize>().unwrap() % 300 == 7; // 10 docs
        for (q, query) in points.iter().take(20).enumerate() {
            for (name, filter) in [("even", &even as &dyn Fn(&str) -> bool), ("rare", &rare)] {
                let got = engine
                    .search_filtered(query, 10, Some(128), filter)
                    .unwrap();
                assert!(got.iter().all(|m| filter(&m.id)), "{name}: filter violated");
                let mut exact: Vec<(f32, String)> = points
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| filter(&format!("doc-{i}")))
                    .map(|(i, p)| {
                        let d: f32 = p.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
                        (d.sqrt(), format!("doc-{i}"))
                    })
                    .collect();
                exact.sort_by(|a, b| a.0.total_cmp(&b.0));
                let want = exact.len().min(10);
                assert_eq!(
                    got.len(),
                    want,
                    "{name} q{q}: a filter must not cost results"
                );
                if name == "rare" {
                    // Selective filters take the exact path: identical answer.
                    let ids: Vec<&str> = got.iter().map(|m| m.id.as_str()).collect();
                    let exact_ids: Vec<&str> =
                        exact.iter().take(want).map(|e| e.1.as_str()).collect();
                    assert_eq!(ids, exact_ids);
                }
            }
        }
    }

    #[test]
    fn search_ids_is_exact_over_the_subset() {
        let (_d, engine) = temp_engine(8, Metric::Cosine);
        let points = cloud(1200, 8, 3);
        for (i, p) in points.iter().enumerate() {
            engine.insert(&format!("p{i}"), p).unwrap();
        }
        engine.remove("p5").unwrap();
        let subset = ["p1", "p5", "p9", "p700", "nope"];
        let hits = engine.search_ids(&points[9], 10, None, &subset).unwrap();
        let ids: Vec<&str> = hits.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids.len(), 3, "deleted and unknown ids are skipped: {ids:?}");
        assert_eq!(ids[0], "p9");
        assert!(
            engine
                .search_ids(&points[0], 3, None, &[])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn batch_search_matches_single_searches() {
        let (_d, engine) = temp_engine(8, Metric::DotProduct);
        let points = cloud(800, 8, 21);
        for (i, p) in points.iter().enumerate() {
            engine.insert(&format!("p{i}"), p).unwrap();
        }
        let queries: Vec<&[f32]> = points.iter().take(5).map(Vec::as_slice).collect();
        let batch = engine.search_batch(&queries, 4, Some(100)).unwrap();
        for (q, got) in queries.iter().zip(&batch) {
            assert_eq!(got, &engine.search(q, 4, Some(100)).unwrap());
        }
    }
}
