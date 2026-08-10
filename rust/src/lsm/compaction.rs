//! Leveled compaction: bounding read amplification and reclaiming space.
//!
//! # Level invariants
//!
//! * **L0** holds freshly flushed tables. They may overlap each other freely,
//!   so a read must consult *every* L0 table (newest first). L0 is therefore
//!   size-limited by table *count*.
//! * **L1+** hold non-overlapping runs: within a level, at most one table can
//!   contain a given key, so a read consults at most one table per level. Each
//!   level's byte budget grows by [`LevelConfig::size_multiplier`].
//!
//! # Trigger and priority
//!
//! A level's *score* is how far it exceeds its budget (`> 1.0` means
//! over-budget). The scheduler always picks the highest-scoring level, so the
//! most write-amplifying backlog is drained first. L0 scores on file count
//! because its cost is proportional to how many tables a read must touch.
//!
//! # Merge semantics
//!
//! Compaction merges the input tables into a new sorted run, keeping only the
//! newest version of each key that is still visible to a live snapshot. Two
//! rules protect correctness:
//!
//! 1. A version at or below the **retain floor** (the oldest live snapshot) may
//!    be dropped once a newer version also sits at or below it — no reader can
//!    observe the difference.
//! 2. A tombstone may only be discarded when compacting into the **bottom-most
//!    level**, because a lower level may still hold an older version that the
//!    tombstone is responsible for masking. Dropping it early resurrects data.

use crate::error::Result;
use crate::lsm::memtable::{InternalKey, ValueSlot};
use crate::lsm::sstable::TableMeta;

/// Tunables for the level layout and compaction triggers.
#[derive(Debug, Clone, Copy)]
pub struct LevelConfig {
    /// L0 table count that triggers a compaction into L1.
    pub l0_compaction_trigger: usize,
    /// Byte budget of L1; each deeper level multiplies it.
    pub base_level_bytes: u64,
    /// Growth factor between consecutive levels.
    pub size_multiplier: u64,
    /// Number of levels, including L0.
    pub max_levels: u32,
}

impl Default for LevelConfig {
    fn default() -> Self {
        LevelConfig {
            l0_compaction_trigger: 4,
            base_level_bytes: 8 * 1024 * 1024,
            size_multiplier: 10,
            max_levels: 7,
        }
    }
}

impl LevelConfig {
    /// Byte budget for `level`. L0 is governed by file count, not bytes.
    #[must_use]
    pub fn level_budget(&self, level: u32) -> u64 {
        if level == 0 {
            return u64::MAX;
        }
        self.base_level_bytes
            .saturating_mul(self.size_multiplier.saturating_pow(level - 1))
    }
}

/// Counters describing the work a compaction performed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactionStats {
    /// Tables consumed.
    pub input_tables: usize,
    /// Entries read from the inputs.
    pub entries_in: u64,
    /// Entries written to the output.
    pub entries_out: u64,
    /// Entries dropped because a newer version supersedes them.
    pub versions_dropped: u64,
    /// Tombstones dropped at the bottom-most level.
    pub tombstones_dropped: u64,
    /// Bytes read from the inputs.
    pub bytes_in: u64,
}

impl CompactionStats {
    /// Write amplification: bytes written per byte of useful output.
    ///
    /// Returns `1.0` for an empty compaction so callers can divide safely.
    #[must_use]
    pub fn amplification(&self) -> f64 {
        if self.entries_out == 0 {
            return 1.0;
        }
        self.entries_in as f64 / self.entries_out as f64
    }
}

/// A planned unit of compaction work.
#[derive(Debug, Clone)]
pub struct CompactionJob {
    /// Level being drained.
    pub source_level: u32,
    /// Level the merged output is written to (`source_level + 1`).
    pub target_level: u32,
    /// Tables from the source level.
    pub source_tables: Vec<TableMeta>,
    /// Overlapping tables from the target level, which must be rewritten.
    pub target_tables: Vec<TableMeta>,
    /// Score that triggered this job; higher is more urgent.
    pub score: f64,
}

impl CompactionJob {
    /// Every table this job consumes.
    #[must_use]
    pub fn all_inputs(&self) -> Vec<&TableMeta> {
        self.source_tables
            .iter()
            .chain(&self.target_tables)
            .collect()
    }

    /// Total input bytes, used to estimate compaction throughput.
    #[must_use]
    pub fn input_bytes(&self) -> u64 {
        self.all_inputs().iter().map(|t| t.file_bytes).sum()
    }

    /// True when the output lands in the bottom-most level, which is the only
    /// place a tombstone may be discarded.
    #[must_use]
    pub fn is_bottom_most(&self, config: &LevelConfig) -> bool {
        self.target_level + 1 >= config.max_levels
    }
}

/// The set of live tables at each level.
#[derive(Debug, Clone)]
pub struct LevelManifest {
    levels: Vec<Vec<TableMeta>>,
    config: LevelConfig,
    next_table_id: u64,
}

impl LevelManifest {
    /// Creates an empty manifest with `config.max_levels` levels.
    #[must_use]
    pub fn new(config: LevelConfig) -> Self {
        LevelManifest {
            levels: vec![Vec::new(); config.max_levels as usize],
            config,
            next_table_id: 1,
        }
    }

    /// The level layout in force.
    #[must_use]
    pub fn config(&self) -> &LevelConfig {
        &self.config
    }

    /// Allocates a fresh, unique table id.
    pub fn allocate_table_id(&mut self) -> u64 {
        let id = self.next_table_id;
        self.next_table_id += 1;
        id
    }

    /// The id the next allocation will return, without consuming it.
    #[must_use]
    pub fn peek_table_id(&self) -> u64 {
        self.next_table_id
    }

    /// Ensures future ids are at least `id`.
    ///
    /// Used at recovery: a table that was created and then removed still burned
    /// its id, and reusing it would let a stale orphaned file be mistaken for a
    /// live table.
    pub fn reserve_table_id(&mut self, id: u64) {
        self.next_table_id = self.next_table_id.max(id);
    }

    /// Tables at `level`, or an empty slice when the level does not exist.
    #[must_use]
    pub fn level(&self, level: u32) -> &[TableMeta] {
        self.levels.get(level as usize).map_or(&[], Vec::as_slice)
    }

    /// Total number of live tables.
    #[must_use]
    pub fn table_count(&self) -> usize {
        self.levels.iter().map(Vec::len).sum()
    }

    /// Total bytes across all levels.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.levels
            .iter()
            .flat_map(|l| l.iter())
            .map(|t| t.file_bytes)
            .sum()
    }

    /// Bytes held at `level`.
    #[must_use]
    pub fn level_bytes(&self, level: u32) -> u64 {
        self.level(level).iter().map(|t| t.file_bytes).sum()
    }

    /// Registers a table, keeping L1+ sorted by key range.
    ///
    /// L0 is kept in insertion order (newest last) because its tables overlap
    /// and must be searched newest-first.
    pub fn add_table(&mut self, meta: TableMeta) {
        let level = meta.level as usize;
        if level >= self.levels.len() {
            self.levels.resize(level + 1, Vec::new());
        }
        self.next_table_id = self.next_table_id.max(meta.id + 1);
        if meta.level == 0 {
            self.levels[level].push(meta);
        } else {
            let pos = self.levels[level].partition_point(|t| t.min_key < meta.min_key);
            self.levels[level].insert(pos, meta);
        }
    }

    /// Removes a table by id, returning it when present.
    pub fn remove_table(&mut self, level: u32, id: u64) -> Option<TableMeta> {
        let tables = self.levels.get_mut(level as usize)?;
        let pos = tables.iter().position(|t| t.id == id)?;
        Some(tables.remove(pos))
    }

    /// Tables at `level` whose key range intersects `[from, to]`.
    #[must_use]
    pub fn overlapping(&self, level: u32, from: &[u8], to: &[u8]) -> Vec<TableMeta> {
        self.level(level)
            .iter()
            .filter(|t| t.overlaps(from, to))
            .cloned()
            .collect()
    }

    /// Tables that might hold `key`, ordered newest-first for a point lookup.
    ///
    /// L0 is returned in reverse insertion order (newest table first); deeper
    /// levels contribute at most one table each.
    #[must_use]
    pub fn lookup_order(&self, key: &[u8]) -> Vec<&TableMeta> {
        let mut out = Vec::new();
        for t in self.level(0).iter().rev() {
            if t.may_contain(key) {
                out.push(t);
            }
        }
        for level in 1..self.levels.len() as u32 {
            if let Some(t) = self.level(level).iter().find(|t| t.may_contain(key)) {
                out.push(t);
            }
        }
        out
    }

    /// How far `level` exceeds its budget. `>= 1.0` means compaction is due.
    #[must_use]
    pub fn score(&self, level: u32) -> f64 {
        if level == 0 {
            // L0 cost is per-table: every table must be consulted on a read.
            return self.level(0).len() as f64 / self.config.l0_compaction_trigger as f64;
        }
        let budget = self.config.level_budget(level);
        if budget == 0 {
            return 0.0;
        }
        self.level_bytes(level) as f64 / budget as f64
    }

    /// Plans the most urgent compaction, or `None` when every level is inside
    /// its budget.
    #[must_use]
    pub fn pick_compaction(&self) -> Option<CompactionJob> {
        let last_level = self.levels.len() as u32 - 1;
        let (best_level, best_score) = (0..self.levels.len() as u32)
            .filter(|&l| l < last_level) // the bottom level has nowhere to drain to
            .map(|l| (l, self.score(l)))
            .fold(
                (0u32, 0.0f64),
                |acc, cur| if cur.1 > acc.1 { cur } else { acc },
            );

        if best_score < 1.0 {
            return None;
        }

        let source_tables: Vec<TableMeta> = if best_level == 0 {
            // L0 tables overlap, so the whole level compacts as one unit.
            self.level(0).to_vec()
        } else {
            // Pick the single largest table: the biggest budget win per unit work.
            let mut candidates = self.level(best_level).to_vec();
            candidates.sort_by_key(|t| std::cmp::Reverse(t.file_bytes));
            candidates.into_iter().take(1).collect()
        };
        if source_tables.is_empty() {
            return None;
        }

        // Widen to every target-level table overlapping the source key span.
        let from = source_tables
            .iter()
            .map(|t| t.min_key.clone())
            .min()
            .unwrap_or_default();
        let to = source_tables
            .iter()
            .map(|t| t.max_key.clone())
            .max()
            .unwrap_or_default();
        let target_level = best_level + 1;
        let target_tables = self.overlapping(target_level, &from, &to);

        Some(CompactionJob {
            source_level: best_level,
            target_level,
            source_tables,
            target_tables,
            score: best_score,
        })
    }

    /// Applies a completed compaction: inputs are retired, outputs installed.
    pub fn apply(&mut self, job: &CompactionJob, outputs: Vec<TableMeta>) {
        for t in &job.source_tables {
            self.remove_table(job.source_level, t.id);
        }
        for t in &job.target_tables {
            self.remove_table(job.target_level, t.id);
        }
        for out in outputs {
            self.add_table(out);
        }
    }
}

/// Merges pre-sorted entry runs into one sorted run, dropping dead versions.
///
/// `runs` must each be sorted in `InternalKey` order (user key ascending,
/// seqno descending) — the order both [`MemTable::iter`](crate::lsm::MemTable::iter)
/// and [`SSTable::iter`](crate::lsm::SSTable::iter) produce. Earlier runs win
/// ties on identical `(key, seqno)`, so callers pass newer levels first.
///
/// * `retain_floor` — the oldest sequence number any live snapshot can still
///   read. Versions strictly older than the newest version at-or-below this
///   floor are unobservable and are dropped.
/// * `drop_tombstones` — only ever `true` when writing the bottom-most level.
///
/// Returns the merged entries and the statistics describing what was dropped.
pub fn merge_runs(
    runs: Vec<Vec<(InternalKey, ValueSlot)>>,
    retain_floor: u64,
    drop_tombstones: bool,
) -> Result<(Vec<(InternalKey, ValueSlot)>, CompactionStats)> {
    let mut stats = CompactionStats {
        input_tables: runs.len(),
        ..CompactionStats::default()
    };

    // Flatten with the run index so a stable sort keeps newer runs first on ties.
    let mut all: Vec<(usize, InternalKey, ValueSlot)> = Vec::new();
    for (run_idx, run) in runs.into_iter().enumerate() {
        for (ik, slot) in run {
            stats.entries_in += 1;
            stats.bytes_in += (ik.user_key.len() + slot.as_ref().map_or(0, Vec::len)) as u64;
            all.push((run_idx, ik, slot));
        }
    }
    // Sort by internal key, then by run index so the newer run wins a tie.
    all.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut out: Vec<(InternalKey, ValueSlot)> = Vec::with_capacity(all.len());
    let mut current_key: Option<Vec<u8>> = None;
    // Whether this key already emitted a version at or below the retain floor.
    let mut floor_covered = false;

    for (_, ik, slot) in all {
        let new_key = current_key.as_deref() != Some(ik.user_key.as_slice());
        if new_key {
            current_key = Some(ik.user_key.clone());
            floor_covered = false;
        } else if floor_covered {
            // A newer version already satisfies every live snapshot.
            stats.versions_dropped += 1;
            continue;
        }

        // Duplicate (key, seqno) from an older run: the first one already won.
        if let Some(last) = out.last()
            && last.0 == ik
        {
            stats.versions_dropped += 1;
            continue;
        }

        if ik.seqno <= retain_floor {
            // Everything older than this is invisible to all live readers.
            floor_covered = true;
        }

        if slot.is_none() && drop_tombstones && ik.seqno <= retain_floor {
            // Bottom level: no older version can survive below us, so the
            // tombstone has nothing left to mask and can be reclaimed.
            stats.tombstones_dropped += 1;
            continue;
        }

        out.push((ik, slot));
    }

    stats.entries_out = out.len() as u64;
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(id: u64, level: u32, min: &[u8], max: &[u8], bytes: u64) -> TableMeta {
        TableMeta {
            id,
            level,
            min_key: min.to_vec(),
            max_key: max.to_vec(),
            min_seqno: 1,
            max_seqno: 100,
            entry_count: 10,
            file_bytes: bytes,
        }
    }

    fn entry(key: &[u8], seqno: u64, value: Option<&[u8]>) -> (InternalKey, ValueSlot) {
        (
            InternalKey::new(key.to_vec(), seqno),
            value.map(<[u8]>::to_vec),
        )
    }

    #[test]
    fn level_budgets_grow_geometrically() {
        let c = LevelConfig::default();
        assert_eq!(c.level_budget(0), u64::MAX, "L0 is bounded by file count");
        assert_eq!(c.level_budget(1), 8 * 1024 * 1024);
        assert_eq!(c.level_budget(2), 80 * 1024 * 1024);
        assert_eq!(c.level_budget(3), 800 * 1024 * 1024);
    }

    #[test]
    fn l0_scores_on_file_count() {
        let mut m = LevelManifest::new(LevelConfig::default());
        assert_eq!(m.score(0), 0.0);
        assert!(m.pick_compaction().is_none());

        for i in 0..4 {
            m.add_table(meta(i + 1, 0, b"a", b"z", 1000));
        }
        assert_eq!(m.score(0), 1.0);
        let job = m.pick_compaction().expect("L0 is at its trigger");
        assert_eq!(job.source_level, 0);
        assert_eq!(job.target_level, 1);
        assert_eq!(job.source_tables.len(), 4, "all of L0 compacts together");
    }

    #[test]
    fn deeper_level_scores_on_bytes() {
        let config = LevelConfig {
            base_level_bytes: 1000,
            ..LevelConfig::default()
        };
        let mut m = LevelManifest::new(config);
        m.add_table(meta(1, 1, b"a", b"m", 600));
        assert!((m.score(1) - 0.6).abs() < 1e-9);
        assert!(m.pick_compaction().is_none(), "under budget");

        m.add_table(meta(2, 1, b"n", b"z", 700));
        assert!(m.score(1) > 1.0);
        let job = m.pick_compaction().expect("L1 is over budget");
        assert_eq!(job.source_level, 1);
        assert_eq!(job.source_tables.len(), 1, "one table at a time in L1+");
        assert_eq!(job.source_tables[0].id, 2, "largest table is picked first");
    }

    #[test]
    fn compaction_pulls_in_overlapping_target_tables() {
        let config = LevelConfig {
            l0_compaction_trigger: 1,
            ..LevelConfig::default()
        };
        let mut m = LevelManifest::new(config);
        m.add_table(meta(1, 0, b"d", b"g", 100));
        // Only the middle L1 table overlaps [d, g].
        m.add_table(meta(2, 1, b"a", b"c", 100));
        m.add_table(meta(3, 1, b"e", b"h", 100));
        m.add_table(meta(4, 1, b"x", b"z", 100));

        let job = m.pick_compaction().unwrap();
        let ids: Vec<u64> = job.target_tables.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![3], "only the overlapping L1 table is rewritten");
        assert_eq!(job.input_bytes(), 200);
    }

    #[test]
    fn bottom_level_never_compacts_downward() {
        let config = LevelConfig {
            max_levels: 3,
            base_level_bytes: 10,
            ..LevelConfig::default()
        };
        let mut m = LevelManifest::new(config);
        // Massively over budget, but it is the last level: nowhere to drain to.
        m.add_table(meta(1, 2, b"a", b"z", 10_000_000));
        assert!(m.score(2) > 1.0);
        assert!(m.pick_compaction().is_none());
    }

    #[test]
    fn l1_plus_tables_stay_sorted_and_lookup_order_is_newest_first() {
        let mut m = LevelManifest::new(LevelConfig::default());
        m.add_table(meta(1, 1, b"m", b"p", 10));
        m.add_table(meta(2, 1, b"a", b"c", 10));
        m.add_table(meta(3, 1, b"x", b"z", 10));
        let keys: Vec<&[u8]> = m.level(1).iter().map(|t| t.min_key.as_slice()).collect();
        assert_eq!(
            keys,
            vec![b"a".as_slice(), b"m".as_slice(), b"x".as_slice()]
        );

        // Two overlapping L0 tables plus one matching L1 table.
        m.add_table(meta(10, 0, b"a", b"z", 10));
        m.add_table(meta(11, 0, b"a", b"z", 10));
        let order: Vec<u64> = m.lookup_order(b"b").iter().map(|t| t.id).collect();
        assert_eq!(order, vec![11, 10, 2], "newest L0 first, then L1");
    }

    #[test]
    fn apply_retires_inputs_and_installs_outputs() {
        let config = LevelConfig {
            l0_compaction_trigger: 1,
            ..LevelConfig::default()
        };
        let mut m = LevelManifest::new(config);
        m.add_table(meta(1, 0, b"a", b"f", 100));
        m.add_table(meta(2, 1, b"a", b"f", 100));
        let job = m.pick_compaction().unwrap();
        assert_eq!(m.table_count(), 2);

        m.apply(&job, vec![meta(3, 1, b"a", b"f", 150)]);
        assert_eq!(m.table_count(), 1);
        assert!(m.level(0).is_empty());
        assert_eq!(m.level(1)[0].id, 3);
        assert_eq!(m.total_bytes(), 150);
    }

    #[test]
    fn table_ids_are_unique_and_monotonic() {
        let mut m = LevelManifest::new(LevelConfig::default());
        assert_eq!(m.allocate_table_id(), 1);
        assert_eq!(m.allocate_table_id(), 2);
        // Registering a high id must push the counter past it.
        m.add_table(meta(99, 0, b"a", b"b", 1));
        assert_eq!(m.allocate_table_id(), 100);
    }

    #[test]
    fn merge_keeps_newest_version_per_key() {
        let newer = vec![entry(b"k", 20, Some(b"new"))];
        let older = vec![entry(b"k", 10, Some(b"old"))];
        // retain_floor 20: no live reader predates the newest version.
        let (out, stats) = merge_runs(vec![newer, older], 20, false).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, Some(b"new".to_vec()));
        assert_eq!(stats.versions_dropped, 1);
        assert_eq!(stats.entries_in, 2);
        assert_eq!(stats.entries_out, 1);
    }

    #[test]
    fn merge_preserves_versions_a_live_snapshot_still_needs() {
        let newer = vec![entry(b"k", 20, Some(b"new"))];
        let older = vec![entry(b"k", 10, Some(b"old"))];
        // A reader is pinned at seqno 15, so v10 must survive.
        let (out, _) = merge_runs(vec![newer, older], 15, false).unwrap();
        assert_eq!(out.len(), 2, "the older version is still observable");
        assert_eq!(out[0].0.seqno, 20);
        assert_eq!(out[1].0.seqno, 10);
    }

    #[test]
    fn tombstone_is_kept_above_the_bottom_level() {
        // This is the resurrection guard: dropping this tombstone mid-tree
        // would expose the v10 value living in a lower level.
        let runs = vec![
            vec![entry(b"k", 20, None)],
            vec![entry(b"k", 10, Some(b"v"))],
        ];
        let (out, stats) = merge_runs(runs, 20, false).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, None, "tombstone must be retained");
        assert_eq!(stats.tombstones_dropped, 0);
    }

    #[test]
    fn tombstone_is_reclaimed_at_the_bottom_level() {
        let runs = vec![
            vec![entry(b"k", 20, None)],
            vec![entry(b"k", 10, Some(b"v"))],
        ];
        let (out, stats) = merge_runs(runs, 20, true).unwrap();
        assert!(out.is_empty(), "nothing survives below the bottom level");
        assert_eq!(stats.tombstones_dropped, 1);
    }

    #[test]
    fn tombstone_survives_bottom_level_if_a_reader_can_see_past_it() {
        // retain_floor 5 < tombstone seqno 20: a snapshot at 15 must still see
        // the delete, so it cannot be dropped even at the bottom level.
        let runs = vec![vec![entry(b"k", 20, None)]];
        let (out, stats) = merge_runs(runs, 5, true).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(stats.tombstones_dropped, 0);
    }

    #[test]
    fn merge_output_is_globally_sorted() {
        let run_a = vec![entry(b"a", 5, Some(b"1")), entry(b"m", 5, Some(b"2"))];
        let run_b = vec![entry(b"c", 4, Some(b"3")), entry(b"z", 4, Some(b"4"))];
        let (out, _) = merge_runs(vec![run_a, run_b], 10, false).unwrap();
        let keys: Vec<Vec<u8>> = out.iter().map(|(ik, _)| ik.user_key.clone()).collect();
        assert_eq!(
            keys,
            vec![b"a".to_vec(), b"c".to_vec(), b"m".to_vec(), b"z".to_vec()]
        );
    }

    #[test]
    fn newer_run_wins_identical_seqno() {
        // Same (key, seqno) in two runs: the earlier (newer) run must win.
        let newer = vec![entry(b"k", 7, Some(b"winner"))];
        let older = vec![entry(b"k", 7, Some(b"loser"))];
        let (out, stats) = merge_runs(vec![newer, older], 7, false).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, Some(b"winner".to_vec()));
        assert_eq!(stats.versions_dropped, 1);
    }

    #[test]
    fn merge_handles_empty_and_disjoint_runs() {
        let (out, stats) = merge_runs(vec![], 10, false).unwrap();
        assert!(out.is_empty());
        assert_eq!(stats.entries_in, 0);
        assert_eq!(stats.amplification(), 1.0);

        let (out, _) =
            merge_runs(vec![vec![], vec![entry(b"a", 1, Some(b"v"))]], 1, false).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn amplification_reflects_dropped_versions() {
        let runs = vec![
            vec![entry(b"k", 30, Some(b"c"))],
            vec![entry(b"k", 20, Some(b"b"))],
            vec![entry(b"k", 10, Some(b"a"))],
        ];
        let (out, stats) = merge_runs(runs, 30, false).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(stats.entries_in, 3);
        assert!((stats.amplification() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn is_bottom_most_matches_config() {
        let config = LevelConfig {
            max_levels: 4,
            ..LevelConfig::default()
        };
        let job = CompactionJob {
            source_level: 2,
            target_level: 3,
            source_tables: vec![],
            target_tables: vec![],
            score: 1.0,
        };
        assert!(job.is_bottom_most(&config));

        let job = CompactionJob {
            source_level: 1,
            target_level: 2,
            source_tables: vec![],
            target_tables: vec![],
            score: 1.0,
        };
        assert!(!job.is_bottom_most(&config));
    }
}
