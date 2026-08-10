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
        let store = VectorStore::open(&path, dim, metric)?;

        let mut ids: HashMap<String, u32> =
            HashMap::with_capacity(options.max_elements.min(1 << 20).max(store.len()));
        let mut graph = HnswGraph::new(hnsw)?;
        let mut live = 0usize;

        let snapshot_path = Self::snapshot_path(&path);
        let restored = Self::load_snapshot(&snapshot_path, dim, metric, store.len())?;

        match restored {
            Some(snapshot) => {
                graph = snapshot.graph;
                for (id, ordinal) in snapshot.ids {
                    ids.insert(id, ordinal);
                }
                live = (0..store.len() as u32)
                    .filter(|ordinal| !store.is_deleted(*ordinal))
                    .count();
            }
            None => {
                // No usable snapshot: replay the vector file. Each record is
                // CRC-checked on the way in, so a torn record fails loudly
                // here rather than silently skewing later searches.
                for ordinal in 0..store.len() as u32 {
                    let record = store.record_at(ordinal)?;
                    ids.insert(record.id, ordinal);
                    if record.deleted {
                        continue;
                    }
                    live += 1;
                    let source = StoreDistances {
                        store: &store,
                        metric,
                        query: &record.vector,
                        query_norm: record.norm,
                    };
                    graph.insert(ordinal, &source)?;
                }
            }
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
        })
    }

    /// Path of the graph snapshot that accompanies a vector file.
    #[must_use]
    pub fn snapshot_path(vector_path: &Path) -> PathBuf {
        let mut s = vector_path.as_os_str().to_os_string();
        s.push(".hnsw");
        PathBuf::from(s)
    }

    /// Reads a snapshot, returning `None` when it is absent or unusable.
    ///
    /// A stale or mismatched snapshot is *not* an error: the vector file can
    /// always rebuild the graph, so recovering silently beats refusing to open
    /// the index.
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
        let snapshot: Snapshot = match bincode::deserialize(&bytes) {
            Ok(s) => s,
            Err(_) => return Ok(None), // corrupt snapshot: rebuild instead
        };
        let usable = snapshot.magic == SNAPSHOT_MAGIC
            && snapshot.version == store::VECTOR_FORMAT_VERSION
            && snapshot.dim == dim
            && snapshot.metric == metric
            // The graph must describe exactly the records on disk. Fewer means
            // inserts landed after the snapshot; more means the vector file was
            // truncated. Either way the vectors win.
            && snapshot.graph.len() == store_len;
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
        Ok(matches)
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
        let bytes = bincode::serialize(&snapshot)?;

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
        // Windows refuses to rename onto an existing file, so clear the way.
        // A crash between these two steps loses the snapshot but not the
        // vectors, and the next open rebuilds the graph.
        if target.exists() {
            std::fs::remove_file(&target)?;
        }
        std::fs::rename(&temporary, &target)?;
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

        let mut temporary = self.path.as_os_str().to_os_string();
        temporary.push(".compact");
        let temporary = PathBuf::from(temporary);
        if temporary.exists() {
            std::fs::remove_file(&temporary)?;
        }

        let mut rebuilt = VectorStore::open(&temporary, self.dim, self.metric)?;
        let mut graph = HnswGraph::new(self.options.hnsw)?;
        let mut ids = HashMap::with_capacity(survivors.len());
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
        rebuilt.sync()?;
        drop(rebuilt);

        // The live mapping must be torn down before the file it maps can be
        // replaced: Windows refuses to unlink or rename over a mapped file,
        // and elsewhere the mapping would silently outlive its inode. Moving
        // the store into a temporary binding and dropping it does exactly
        // that, and `inner` is left holding the rebuilt index below.
        let live = survivors.len();
        drop(std::mem::replace(
            &mut inner.store,
            VectorStore::open(&temporary, self.dim, self.metric)?,
        ));
        std::fs::remove_file(&self.path)?;
        // Same again for the rebuilt file, which must be unmapped before the
        // rename moves it into place.
        drop(std::mem::replace(
            &mut inner.store,
            VectorStore::open(&temporary, self.dim, self.metric)?,
        ));
        std::fs::rename(&temporary, &self.path)?;

        inner.store = VectorStore::open(&self.path, self.dim, self.metric)?;
        inner.graph = graph;
        inner.ids = ids;
        inner.live = live;
        Ok(dead)
    }

    /// Flushes and snapshots. Called by `Drop` and the FFI free function.
    pub fn close(&self) -> Result<()> {
        self.save(None)
    }
}

impl Drop for VectorEngine {
    fn drop(&mut self) {
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
}
