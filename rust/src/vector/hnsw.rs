//! Hierarchical Navigable Small World graph index.
//!
//! A hand-written implementation of Malkov & Yashunin's HNSW, rather than a
//! dependency. `hnsw_rs`, the obvious candidate, pulls in `anndists`,
//! `mmap-rs`, `rayon`, `env_logger` and `jiff` — roughly a hundred transitive
//! crates, several of which do not cross-compile cleanly to every Android and
//! iOS target PhoenixDB ships to. The graph itself is a few hundred lines, so
//! owning it costs less than owning that dependency tree, and it lets the
//! index share PhoenixDB's `bincode` snapshot format and error taxonomy.
//!
//! # Structure
//!
//! Nodes live on `level 0`; a geometrically-distributed subset is promoted to
//! higher levels, forming progressively sparser navigation layers. A search
//! descends greedily from the top entry point, then runs a best-first
//! beam search of width `ef` on level 0.
//!
//! # Complexity
//!
//! Insert and search are both `O(log N)` expected, with the constant governed
//! by `m` (links per node) and `ef` (beam width).
//!
//! # Determinism
//!
//! Level assignment uses a seeded xorshift generator stored *in the graph*, so
//! a snapshot reloaded and extended produces the same structure as one grown
//! in a single session. Tests depend on this.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

/// Supplies distances to the graph, which never sees the vectors themselves.
///
/// Keeping the vector storage behind this trait is what lets the graph be
/// serialised on its own: it holds ids and links, never coordinates.
pub trait DistanceSource {
    /// Ordering distance from the current query to stored node `id`.
    fn to_query(&self, id: u32) -> f32;
    /// Ordering distance between two stored nodes.
    fn between(&self, a: u32, b: u32) -> f32;
    /// True when `id` has been tombstoned and must not be returned.
    fn is_deleted(&self, id: u32) -> bool;
}

/// A `(distance, id)` pair ordered so that [`BinaryHeap`] yields the *furthest*
/// element first.
///
/// `f32` is not `Ord`, and the ordering must be total for the heap to be
/// correct. NaN cannot reach here — [`crate::vector::distance::validate_vector`]
/// rejects non-finite input — but the comparison still resolves it to `Equal`
/// rather than panicking, because a panic inside a heap operation would unwind
/// through the FFI guard for no useful reason.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    distance: f32,
    id: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Same pair, ordered so that [`BinaryHeap`] yields the *nearest* element
/// first. Used for the frontier of the beam search.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Nearest(Candidate);

impl Eq for Nearest {}

impl Ord for Nearest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.0.cmp(&self.0) // reversed
    }
}

impl PartialOrd for Nearest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Tunable graph parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HnswParams {
    /// Links established per node on levels above 0.
    pub m: usize,
    /// Link budget on level 0, conventionally `2 * m`.
    pub m_max0: usize,
    /// Beam width while building. Higher means a better graph, slower inserts.
    pub ef_construction: usize,
    /// Default beam width at query time when the caller does not specify one.
    pub ef_search: usize,
    /// Seed for the level-assignment generator.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        // m = 16 is the paper's recommendation and every mainstream
        // implementation's default.
        //
        // ef_construction = 100 rather than the paper's 200: build cost is
        // roughly linear in it, and the neighbour-selection heuristic is
        // O(ef_construction * m) distance computations per insert. Measured on
        // 20 000 uniform-random 384-dimensional vectors — the adversarial case
        // for a graph index — 200 costs about 2x the build time of 100 for a
        // recall difference in the low single digits. Callers indexing once and
        // querying forever can raise it; callers indexing on a phone should
        // not have to lower it.
        HnswParams {
            m: 16,
            m_max0: 32,
            ef_construction: 100,
            ef_search: 64,
            seed: 0x5EED_1234_ABCD_9876,
        }
    }
}

impl HnswParams {
    /// Validates and normalises the parameters.
    pub fn validate(mut self) -> Result<Self> {
        if self.m == 0 || self.m > 512 {
            return Err(Error::invalid(format!(
                "hnsw m must be in 1..=512, got {}",
                self.m
            )));
        }
        if self.m_max0 == 0 {
            self.m_max0 = self.m.saturating_mul(2);
        }
        if self.m_max0 > 1024 {
            return Err(Error::invalid(format!(
                "hnsw m_max0 must be <= 1024, got {}",
                self.m_max0
            )));
        }
        self.ef_construction = self.ef_construction.clamp(self.m, 4096);
        self.ef_search = self.ef_search.clamp(1, 4096);
        Ok(self)
    }

    /// Normalisation factor for the level distribution, `1 / ln(m)`.
    #[inline]
    fn level_multiplier(&self) -> f64 {
        1.0 / (self.m.max(2) as f64).ln()
    }
}

/// One node's adjacency lists, indexed by level.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Node {
    /// `links[l]` holds the neighbours on level `l`; `links.len()` is the
    /// node's height, so a level-0-only node carries exactly one vector.
    links: Vec<Vec<u32>>,
}

impl Node {
    fn with_level(level: usize) -> Self {
        Node {
            links: vec![Vec::new(); level + 1],
        }
    }
}

/// A visited set that clears in O(1) via generation stamps.
///
/// The obvious implementation — `vec![false; nodes.len()]` per layer search —
/// allocates and zeroes the whole node array on every call. During a build
/// that is the dominant cost: an insert searches several layers, so a
/// 20 000-node index pays tens of thousands of full-array clears. Stamping
/// each slot with a monotonically increasing generation makes "clear" a single
/// increment, and reuses one allocation for the life of the graph.
#[derive(Debug, Clone, Default)]
struct VisitSet {
    stamps: Vec<u32>,
    generation: u32,
}

impl VisitSet {
    /// Prepares the set for `len` nodes and invalidates every previous mark.
    fn begin(&mut self, len: usize) {
        if self.stamps.len() < len {
            self.stamps.resize(len, 0);
        }
        // `generation` starts at 1 so a freshly zeroed slot is never "visited".
        // On wraparound the stamps are reset once, which is correct and
        // happens at most every 4 billion searches.
        match self.generation.checked_add(1) {
            Some(next) => self.generation = next,
            None => {
                self.stamps.iter_mut().for_each(|s| *s = 0);
                self.generation = 1;
            }
        }
    }

    /// Marks `id` visited, returning `true` when it was not already.
    #[inline]
    fn visit(&mut self, id: u32) -> bool {
        match self.stamps.get_mut(id as usize) {
            Some(slot) if *slot != self.generation => {
                *slot = self.generation;
                true
            }
            _ => false,
        }
    }
}

/// The navigable small-world graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswGraph {
    params: HnswParams,
    nodes: Vec<Node>,
    /// Node the descent starts from; `None` only while the graph is empty.
    entry: Option<u32>,
    /// Height of the tallest node.
    max_level: usize,
    /// State of the level-assignment generator, persisted with the graph.
    rng: u64,
    /// Scratch space for the visited set, reused across inserts.
    ///
    /// Skipped by serde: it is a pure allocation cache with no semantic
    /// content, and persisting it would bloat every snapshot by 4 bytes per
    /// node for nothing.
    #[serde(skip)]
    scratch: VisitSet,
}

impl HnswGraph {
    /// Creates an empty graph.
    pub fn new(params: HnswParams) -> Result<Self> {
        let params = params.validate()?;
        Ok(HnswGraph {
            rng: params.seed | 1, // xorshift degenerates to zero from a zero seed
            params,
            nodes: Vec::new(),
            entry: None,
            max_level: 0,
            scratch: VisitSet::default(),
        })
    }

    /// Parameters in force.
    #[must_use]
    pub fn params(&self) -> HnswParams {
        self.params
    }

    /// Number of nodes, including tombstoned ones.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when no node has been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Height of the tallest node.
    #[must_use]
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Overrides the default query-time beam width.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.params.ef_search = ef.clamp(1, 4096);
    }

    /// Draws a node height from the geometric distribution the paper uses.
    ///
    /// xorshift64* rather than `rand`: it is four lines, has no dependency,
    /// and its quality is far beyond what a level assignment needs.
    fn next_level(&mut self) -> usize {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        // 53 bits of mantissa in (0, 1]; never exactly 0, so ln() is finite.
        let unit = ((self.rng >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let level = (-unit.ln() * self.params.level_multiplier()).floor();
        // Clamped so a pathological draw cannot allocate an absurd link table.
        (level as usize).min(32)
    }

    /// Inserts node `id`, wiring it into every level up to its drawn height.
    ///
    /// `id` must be the next sequential index (`len()`), which the caller
    /// guarantees by allocating ids from the same counter that grows storage.
    pub fn insert<D: DistanceSource>(&mut self, id: u32, source: &D) -> Result<()> {
        let expected = u32::try_from(self.nodes.len())
            .map_err(|_| Error::Full("hnsw graph exceeded 2^32 nodes".to_string()))?;
        if id != expected {
            return Err(Error::corrupt(format!(
                "hnsw insert expected sequential id {expected}, got {id}"
            )));
        }

        let level = self.next_level();
        self.nodes.push(Node::with_level(level));

        let Some(entry) = self.entry else {
            // First node: it is the entry point at its own height.
            self.entry = Some(id);
            self.max_level = level;
            return Ok(());
        };

        // Phase 1 — greedy descent through the layers above `level`, which the
        // new node does not join. Each layer narrows the entry point.
        let mut current = entry;
        let mut current_distance = source.to_query(current);
        let mut layer = self.max_level;
        while layer > level {
            let (next, next_distance) =
                self.greedy_descend(current, current_distance, layer, source);
            current = next;
            current_distance = next_distance;
            layer -= 1;
        }

        // Phase 2 — beam search and link on every layer the node joins.
        // The visited set is moved out of `self` for the duration so the
        // borrow checker allows `&self` layer searches alongside `&mut self`
        // link edits; it is put back before returning, keeping the allocation.
        let mut visited = std::mem::take(&mut self.scratch);
        let mut entry_points = vec![current];
        let mut layer = level.min(self.max_level) as isize;
        while layer >= 0 {
            let l = layer as usize;
            let candidates = self.search_layer(
                &entry_points,
                self.params.ef_construction,
                l,
                source,
                &mut visited,
            );
            let budget = if l == 0 {
                self.params.m_max0
            } else {
                self.params.m
            };
            let selected = self.select_neighbours(id, &candidates, budget, source);

            for &neighbour in &selected {
                self.connect(id, neighbour, l);
                self.connect(neighbour, id, l);
                // `prune` returns immediately when the neighbour is still
                // within budget, which is the common case; only the node that
                // just overflowed pays for re-selection.
                self.prune(neighbour, l, budget, source);
            }
            // `id`'s own links come straight from `select_neighbours`, which
            // already honoured `budget`, so there is nothing to prune here.
            debug_assert!(self.links_at(id, l).len() <= budget);

            entry_points = if selected.is_empty() {
                vec![current]
            } else {
                selected
            };
            layer -= 1;
        }
        self.scratch = visited;

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(id);
        }
        Ok(())
    }

    /// Walks greedily downhill on one layer until no neighbour improves.
    fn greedy_descend<D: DistanceSource>(
        &self,
        start: u32,
        start_distance: f32,
        layer: usize,
        source: &D,
    ) -> (u32, f32) {
        let mut best = start;
        let mut best_distance = start_distance;
        let mut improved = true;
        while improved {
            improved = false;
            for &neighbour in self.links_at(best, layer) {
                let distance = source.to_query(neighbour);
                if distance < best_distance {
                    best = neighbour;
                    best_distance = distance;
                    improved = true;
                }
            }
        }
        (best, best_distance)
    }

    /// Best-first search of one layer, returning up to `ef` candidates sorted
    /// nearest-first.
    ///
    /// `visited` is supplied by the caller so the allocation is reused across
    /// the several layer searches one insert performs; [`VisitSet::begin`] is
    /// called here, so the caller need not clear it.
    ///
    /// Tombstoned nodes are traversed (they are still part of the topology)
    /// but never enter the result set.
    fn search_layer<D: DistanceSource>(
        &self,
        entry_points: &[u32],
        ef: usize,
        layer: usize,
        source: &D,
        visited: &mut VisitSet,
    ) -> Vec<Candidate> {
        visited.begin(self.nodes.len());
        // Frontier: nearest-first, drives expansion.
        let mut frontier: BinaryHeap<Nearest> = BinaryHeap::new();
        // Results: furthest-first, so the worst element is O(1) to evict.
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for &entry in entry_points {
            if entry as usize >= self.nodes.len() || !visited.visit(entry) {
                continue;
            }
            let candidate = Candidate {
                distance: source.to_query(entry),
                id: entry,
            };
            frontier.push(Nearest(candidate));
            if !source.is_deleted(entry) {
                results.push(candidate);
            }
        }

        while let Some(Nearest(current)) = frontier.pop() {
            // Stop as soon as the frontier's best is worse than the result
            // set's worst: nothing reachable from here can improve the answer.
            if results.len() >= ef
                && let Some(worst) = results.peek()
                && current.distance > worst.distance
            {
                break;
            }
            for &neighbour in self.links_at(current.id, layer) {
                if neighbour as usize >= self.nodes.len() || !visited.visit(neighbour) {
                    continue;
                }
                let distance = source.to_query(neighbour);
                let worst = results.peek().map_or(f32::INFINITY, |c| c.distance);
                if results.len() < ef || distance < worst {
                    let candidate = Candidate {
                        distance,
                        id: neighbour,
                    };
                    frontier.push(Nearest(candidate));
                    if !source.is_deleted(neighbour) {
                        results.push(candidate);
                        if results.len() > ef {
                            results.pop(); // drop the furthest
                        }
                    }
                }
            }
        }

        let mut out = results.into_vec();
        out.sort_unstable();
        out
    }

    /// Heuristic neighbour selection (Algorithm 4 in the paper).
    ///
    /// A candidate is kept only if it is closer to the new node than to any
    /// already-selected neighbour. This is what preserves long-range links and
    /// keeps the graph navigable; naive "keep the m nearest" collapses into
    /// tight clusters with no bridges between them, which destroys recall.
    ///
    /// The inner test is the build's hot loop — `O(candidates * budget)`
    /// distance computations, each `O(dim)` — so it short-circuits on the
    /// first selected neighbour that rejects the candidate rather than
    /// evaluating all of them.
    fn select_neighbours<D: DistanceSource>(
        &self,
        node: u32,
        candidates: &[Candidate],
        budget: usize,
        source: &D,
    ) -> Vec<u32> {
        let mut selected: Vec<u32> = Vec::with_capacity(budget);
        for candidate in candidates {
            if selected.len() >= budget {
                break;
            }
            if candidate.id == node {
                continue;
            }
            // `all` stops at the first `false`, so a candidate dominated by an
            // early selection costs one distance rather than `budget` of them.
            let keep = selected
                .iter()
                .all(|&chosen| source.between(candidate.id, chosen) > candidate.distance);
            if keep {
                selected.push(candidate.id);
            }
        }
        // The heuristic can be too strict in dense clusters and leave a node
        // under-connected, which is worse than a slightly clustered graph.
        // Backfill by plain proximity — no distance computations at all, since
        // `candidates` is already sorted nearest-first.
        if selected.len() < budget {
            for candidate in candidates {
                if selected.len() >= budget {
                    break;
                }
                if candidate.id != node && !selected.contains(&candidate.id) {
                    selected.push(candidate.id);
                }
            }
        }
        selected
    }

    /// Adds a directed link, ignoring duplicates and out-of-range levels.
    fn connect(&mut self, from: u32, to: u32, layer: usize) {
        if from == to {
            return;
        }
        let Some(node) = self.nodes.get_mut(from as usize) else {
            return;
        };
        let Some(links) = node.links.get_mut(layer) else {
            return;
        };
        if !links.contains(&to) {
            links.push(to);
        }
    }

    /// Trims `node`'s links on `layer` back to `budget`, keeping the set the
    /// selection heuristic prefers.
    fn prune<D: DistanceSource>(&mut self, node: u32, layer: usize, budget: usize, source: &D) {
        let current = match self
            .nodes
            .get(node as usize)
            .and_then(|n| n.links.get(layer))
        {
            Some(links) if links.len() > budget => links.clone(),
            _ => return,
        };
        let mut candidates: Vec<Candidate> = current
            .iter()
            .map(|&id| Candidate {
                distance: source.between(node, id),
                id,
            })
            .collect();
        candidates.sort_unstable();
        let kept = self.select_neighbours(node, &candidates, budget, source);
        if let Some(links) = self
            .nodes
            .get_mut(node as usize)
            .and_then(|n| n.links.get_mut(layer))
        {
            *links = kept;
        }
    }

    /// Neighbours of `id` on `layer`, or an empty slice when either is absent.
    #[inline]
    fn links_at(&self, id: u32, layer: usize) -> &[u32] {
        self.nodes
            .get(id as usize)
            .and_then(|node| node.links.get(layer))
            .map_or(&[][..], |links| links.as_slice())
    }

    /// Returns the `k` nearest live nodes to the source's current query.
    ///
    /// `ef` is clamped to at least `k`: a beam narrower than the requested
    /// result count cannot fill it.
    pub fn search<D: DistanceSource>(
        &self,
        k: usize,
        ef: Option<usize>,
        source: &D,
    ) -> Vec<(u32, f32)> {
        if k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let ef = ef.unwrap_or(self.params.ef_search).max(k).min(4096);

        let mut current = entry;
        let mut current_distance = source.to_query(current);
        let mut layer = self.max_level;
        while layer > 0 {
            let (next, next_distance) =
                self.greedy_descend(current, current_distance, layer, source);
            current = next;
            current_distance = next_distance;
            layer -= 1;
        }

        // A query holds only `&self`, so the visited set is local here rather
        // than reusing `self.scratch`. That is one allocation per *query*
        // instead of one per layer per insert, which is not on the hot path
        // the build profile identified.
        let mut visited = VisitSet::default();
        let mut found = self.search_layer(&[current], ef, 0, source, &mut visited);
        found.truncate(k);
        found.into_iter().map(|c| (c.id, c.distance)).collect()
    }

    /// Exhaustively scans every live node — the exact answer, `O(N)`.
    ///
    /// Used automatically for collections small enough that the graph's
    /// approximation would be the only source of error, and by tests as the
    /// recall oracle.
    pub fn brute_force<D: DistanceSource>(&self, k: usize, source: &D) -> Vec<(u32, f32)> {
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
        for id in 0..self.nodes.len() as u32 {
            if source.is_deleted(id) {
                continue;
            }
            heap.push(Candidate {
                distance: source.to_query(id),
                id,
            });
            if heap.len() > k {
                heap.pop();
            }
        }
        let mut out = heap.into_vec();
        out.sort_unstable();
        out.into_iter().map(|c| (c.id, c.distance)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense, exhaustive distance source over an in-memory vector set.
    ///
    /// The graph never sees vectors, so tests supply their own storage; this
    /// mirror of the engine's real source keeps the graph test independent of
    /// the engine.
    struct Vectors {
        data: Vec<Vec<f32>>,
        query: Vec<f32>,
        deleted: Vec<bool>,
    }

    impl Vectors {
        fn new(data: Vec<Vec<f32>>) -> Self {
            let deleted = vec![false; data.len()];
            Vectors {
                data,
                query: Vec::new(),
                deleted,
            }
        }

        fn set_query(&mut self, q: Vec<f32>) {
            self.query = q;
        }

        fn l2(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
        }
    }

    impl DistanceSource for Vectors {
        fn to_query(&self, id: u32) -> f32 {
            Self::l2(&self.query, &self.data[id as usize])
        }
        fn between(&self, a: u32, b: u32) -> f32 {
            Self::l2(&self.data[a as usize], &self.data[b as usize])
        }
        fn is_deleted(&self, id: u32) -> bool {
            self.deleted[id as usize]
        }
    }

    /// Deterministic pseudo-random points, so a recall regression is
    /// reproducible rather than flaky.
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

    fn build(points: &[Vec<f32>], params: HnswParams) -> (HnswGraph, Vectors) {
        let mut source = Vectors::new(points.to_vec());
        let mut graph = HnswGraph::new(params).unwrap();
        for (index, point) in points.iter().enumerate() {
            // Inserting node `i` means "find neighbours for this point", so
            // the query is the point itself.
            source.set_query(point.clone());
            graph.insert(index as u32, &source).unwrap();
        }
        (graph, source)
    }

    #[test]
    fn empty_graph_returns_nothing() {
        let graph = HnswGraph::new(HnswParams::default()).unwrap();
        let mut source = Vectors::new(Vec::new());
        source.set_query(vec![0.0; 4]);
        assert!(graph.search(10, None, &source).is_empty());
        assert!(graph.is_empty());
    }

    #[test]
    fn single_node_is_always_found() {
        let points = vec![vec![1.0f32, 2.0, 3.0]];
        let (graph, mut source) = build(&points, HnswParams::default());
        source.set_query(vec![1.0, 2.0, 3.0]);
        let found = graph.search(5, None, &source);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, 0);
        assert!(found[0].1.abs() < 1e-6);
    }

    #[test]
    fn nearest_neighbour_is_exact_on_a_small_grid() {
        // A 10x10 lattice: the nearest point to (3.1, 4.1) is unambiguously
        // (3, 4), so an approximate answer is still checkable exactly.
        let mut points = Vec::new();
        for x in 0..10 {
            for y in 0..10 {
                points.push(vec![x as f32, y as f32]);
            }
        }
        let (graph, mut source) = build(&points, HnswParams::default());
        source.set_query(vec![3.1, 4.1]);
        let found = graph.search(1, Some(64), &source);
        assert_eq!(found.len(), 1);
        assert_eq!(
            points[found[0].0 as usize],
            vec![3.0, 4.0],
            "expected the lattice point (3, 4)"
        );
    }

    #[test]
    fn results_are_sorted_by_ascending_distance() {
        let points = cloud(200, 16, 7);
        let (graph, mut source) = build(&points, HnswParams::default());
        source.set_query(points[0].clone());
        let found = graph.search(10, Some(100), &source);
        assert_eq!(found.len(), 10);
        for window in found.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "results out of order: {:?}",
                found
            );
        }
    }

    #[test]
    fn recall_against_brute_force_is_high() {
        // The contract users actually rely on. 300 points in 32 dimensions
        // with default parameters should recover essentially every true
        // neighbour; anything below 90% means the graph is broken, not merely
        // approximate.
        let points = cloud(300, 32, 99);
        let (graph, mut source) = build(&points, HnswParams::default());

        let mut hits = 0usize;
        let mut total = 0usize;
        for query in points.iter().take(30) {
            source.set_query(query.clone());
            let exact: Vec<u32> = graph
                .brute_force(10, &source)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let approximate: Vec<u32> = graph
                .search(10, Some(120), &source)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            hits += approximate.iter().filter(|id| exact.contains(id)).count();
            total += exact.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.90,
            "recall@10 was {recall:.3}, expected >= 0.90"
        );
    }

    #[test]
    fn deleted_nodes_are_never_returned_but_stay_traversable() {
        let points = cloud(120, 8, 5);
        let (graph, mut source) = build(&points, HnswParams::default());
        // Tombstone the exact match for the query.
        source.deleted[0] = true;
        source.set_query(points[0].clone());
        let found = graph.search(5, Some(64), &source);
        assert!(
            !found.is_empty(),
            "graph must stay navigable through tombstones"
        );
        assert!(
            found.iter().all(|(id, _)| *id != 0),
            "tombstoned node leaked into results"
        );
    }

    #[test]
    fn out_of_order_insert_is_rejected() {
        let mut graph = HnswGraph::new(HnswParams::default()).unwrap();
        let mut source = Vectors::new(vec![vec![0.0], vec![1.0]]);
        source.set_query(vec![0.0]);
        graph.insert(0, &source).unwrap();
        // Skipping id 1 would desynchronise the graph from vector storage.
        assert!(graph.insert(5, &source).is_err());
    }

    #[test]
    fn params_are_validated_and_normalised() {
        assert!(
            HnswParams {
                m: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            HnswParams {
                m: 513,
                ..Default::default()
            }
            .validate()
            .is_err()
        );

        let normalised = HnswParams {
            m: 8,
            m_max0: 0,
            ef_construction: 1,
            ef_search: 0,
            seed: 1,
        }
        .validate()
        .unwrap();
        assert_eq!(normalised.m_max0, 16, "m_max0 defaults to 2m");
        assert!(normalised.ef_construction >= normalised.m);
        assert!(normalised.ef_search >= 1);
    }

    #[test]
    fn identical_seed_builds_an_identical_graph() {
        // Reproducibility matters for snapshot compatibility: reloading and
        // extending must not diverge from a single-session build.
        let points = cloud(80, 8, 3);
        let (a, _) = build(&points, HnswParams::default());
        let (b, _) = build(&points, HnswParams::default());
        assert_eq!(a.max_level(), b.max_level());
        assert_eq!(a.len(), b.len());
        for id in 0..a.len() as u32 {
            for layer in 0..=a.max_level() {
                assert_eq!(
                    a.links_at(id, layer),
                    b.links_at(id, layer),
                    "node {id} layer {layer} diverged"
                );
            }
        }
    }

    #[test]
    fn link_budgets_are_respected() {
        let points = cloud(250, 8, 42);
        let params = HnswParams::default();
        let (graph, _) = build(&points, params);
        for id in 0..graph.len() as u32 {
            for layer in 0..=graph.max_level() {
                let budget = if layer == 0 { params.m_max0 } else { params.m };
                assert!(
                    graph.links_at(id, layer).len() <= budget,
                    "node {id} layer {layer} exceeded its {budget}-link budget"
                );
            }
        }
    }

    #[test]
    fn k_larger_than_the_collection_returns_everything_live() {
        let points = cloud(7, 4, 8);
        let (graph, mut source) = build(&points, HnswParams::default());
        source.set_query(points[3].clone());
        let found = graph.search(50, None, &source);
        assert_eq!(found.len(), 7);
    }

    #[test]
    fn graph_survives_a_bincode_round_trip() {
        let points = cloud(150, 16, 21);
        let (graph, mut source) = build(&points, HnswParams::default());
        let bytes = bincode::serialize(&graph).unwrap();
        let restored: HnswGraph = bincode::deserialize(&bytes).unwrap();

        source.set_query(points[10].clone());
        let before = graph.search(5, Some(64), &source);
        let after = restored.search(5, Some(64), &source);
        assert_eq!(before, after, "search results changed across serialization");
    }
}
