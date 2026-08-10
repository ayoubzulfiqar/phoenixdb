//! The LSM engine: MemTable rotation, flush, and compaction execution.
//!
//! This is the layer the rest of the database talks to. It owns the active
//! MemTable, the frozen tables awaiting flush, and the [`LevelManifest`]
//! describing what is on disk.
//!
//! # Write path
//!
//! ```text
//!   put(key, value, seqno)
//!        │
//!        ├─ active MemTable insert          (caller has already WAL'd)
//!        └─ if size >= budget: rotate       (active becomes frozen)
//! ```
//!
//! Rotation is O(1) — a pointer swap — so a writer never waits for a flush.
//!
//! # Read path
//!
//! Newest-to-oldest: active MemTable, frozen MemTables (newest first), then
//! each level via [`LevelManifest::lookup_order`]. The first **definitive**
//! answer wins; [`Lookup::Deleted`] is definitive, which is what stops a
//! tombstone from being overtaken by a stale value in a lower level.
//!
//! # Durability contract
//!
//! The engine never writes the WAL itself — the caller does that first, so a
//! record is durable before it becomes visible here. A flush `fsync`s the new
//! SSTable before the manifest is updated, so a crash mid-flush leaves an
//! orphaned file (harmless, reclaimed on next open) rather than a manifest
//! entry pointing at a partial table.

use crate::error::Result;
use crate::lsm::compaction::{CompactionStats, LevelConfig, LevelManifest, merge_runs};
use crate::lsm::manifest::{
    COMPACT_AFTER_EDITS, MANIFEST_FILE, Manifest, ManifestEdit, SerializableMeta,
};
use crate::lsm::memtable::{InternalKey, Lookup, MemTable, ValueSlot};
use crate::lsm::sstable::{SSTable, SSTableWriter, TableMeta};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Tunables for the LSM engine.
#[derive(Debug, Clone, Copy)]
pub struct LsmOptions {
    /// Rotate the active MemTable once it reaches this many bytes.
    pub memtable_bytes: usize,
    /// Level layout and compaction triggers.
    pub levels: LevelConfig,
}

impl Default for LsmOptions {
    fn default() -> Self {
        LsmOptions {
            memtable_bytes: 4 * 1024 * 1024,
            levels: LevelConfig::default(),
        }
    }
}

/// Aggregate engine counters, for metrics and tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LsmStats {
    /// Bytes held by the active MemTable.
    pub active_bytes: usize,
    /// MemTables frozen and awaiting flush.
    pub frozen_count: usize,
    /// Live SSTables across all levels.
    pub table_count: usize,
    /// Total bytes across all SSTables.
    pub disk_bytes: u64,
    /// Flushes completed since open.
    pub flushes: u64,
    /// Compactions completed since open.
    pub compactions: u64,
}

/// The hybrid engine's LSM half.
pub struct LsmEngine {
    dir: PathBuf,
    options: LsmOptions,
    active: MemTable,
    /// Frozen tables, oldest first; the flush path drains from the front.
    frozen: Vec<MemTable>,
    manifest: LevelManifest,
    /// Durable record of the level set.
    log: Manifest,
    /// Highest sequence number durably reflected in the SSTables.
    checkpoint_seqno: u64,
    /// Open table handles, keyed by table id.
    open_tables: HashMap<u64, SSTable>,
    flushes: u64,
    compactions: u64,
}

impl LsmEngine {
    /// Opens the engine rooted at `dir`, replaying its manifest.
    ///
    /// Recovery does three things, in order:
    ///
    /// 1. Replay the manifest to learn which tables are live.
    /// 2. Open each one, verifying its CRC. A table the manifest references but
    ///    which fails to open is a hard error — silently dropping it would lose
    ///    committed data.
    /// 3. Delete orphaned `.sst` files: debris from a crash between writing a
    ///    table and committing its manifest edit.
    pub fn open(dir: impl AsRef<Path>, options: LsmOptions) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        let manifest_path = dir.join(MANIFEST_FILE);

        let state = Manifest::recover(&manifest_path)?;
        let mut manifest = LevelManifest::new(options.levels);
        let mut open_tables = HashMap::new();

        for meta in &state.tables {
            let path = dir.join(format!("{:010}.sst", meta.id));
            // A referenced-but-unopenable table means real data loss; report it
            // rather than starting up with a hole in the key space.
            let table = SSTable::open(&path, meta.id, meta.level)?;
            open_tables.insert(meta.id, table);
            manifest.add_table(meta.clone());
        }
        // Make sure ids never collide with a table that was removed but whose
        // id the log still remembers.
        manifest.reserve_table_id(state.next_table_id);

        // Reclaim debris from an interrupted flush or compaction.
        for orphan in Manifest::orphaned_files(&dir, &state.tables)? {
            let _ = std::fs::remove_file(orphan);
        }

        let mut log = Manifest::open(&manifest_path)?;
        // Bound future replay time.
        if state.edits_replayed > COMPACT_AFTER_EDITS {
            log = Manifest::compact(
                &manifest_path,
                &state.tables,
                manifest.peek_table_id(),
                state.checkpoint_seqno,
            )?;
        }

        Ok(LsmEngine {
            dir,
            options,
            active: MemTable::new(),
            frozen: Vec::new(),
            manifest,
            log,
            checkpoint_seqno: state.checkpoint_seqno,
            open_tables,
            flushes: 0,
            compactions: 0,
        })
    }

    /// Highest sequence number durably captured in the SSTables.
    ///
    /// WAL records at or below this are already folded into tables and need not
    /// be replayed.
    #[must_use]
    pub fn checkpoint_seqno(&self) -> u64 {
        self.checkpoint_seqno
    }

    /// Records that everything up to `seqno` is durable in the tables.
    pub fn set_checkpoint(&mut self, seqno: u64) -> Result<()> {
        if seqno <= self.checkpoint_seqno {
            return Ok(());
        }
        self.log.append(&ManifestEdit::Checkpoint { seqno })?;
        self.checkpoint_seqno = seqno;
        Ok(())
    }

    /// Directory holding this engine's SSTables.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The level manifest.
    #[must_use]
    pub fn manifest(&self) -> &LevelManifest {
        &self.manifest
    }

    /// Current counters.
    #[must_use]
    pub fn stats(&self) -> LsmStats {
        LsmStats {
            active_bytes: self.active.size_bytes(),
            frozen_count: self.frozen.len(),
            table_count: self.manifest.table_count(),
            disk_bytes: self.manifest.total_bytes(),
            flushes: self.flushes,
            compactions: self.compactions,
        }
    }

    /// File name for table `id`.
    fn table_path(&self, id: u64) -> PathBuf {
        self.dir.join(format!("{id:010}.sst"))
    }

    /// Records `key = value` at `seqno`, rotating the MemTable when full.
    ///
    /// The caller must already have made the corresponding WAL record durable.
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>, seqno: u64) {
        self.active.insert(key, value, seqno);
        self.maybe_rotate();
    }

    /// Records a tombstone for `key` at `seqno`.
    pub fn delete(&mut self, key: Vec<u8>, seqno: u64) {
        self.active.delete(key, seqno);
        self.maybe_rotate();
    }

    /// Freezes the active MemTable when it exceeds its budget.
    fn maybe_rotate(&mut self) {
        if self.active.size_bytes() >= self.options.memtable_bytes {
            self.rotate();
        }
    }

    /// Freezes the active MemTable unconditionally, if it holds anything.
    pub fn rotate(&mut self) {
        if self.active.is_empty() {
            return;
        }
        self.frozen.push(std::mem::take(&mut self.active));
    }

    /// True when a flush would do work.
    #[must_use]
    pub fn needs_flush(&self) -> bool {
        !self.frozen.is_empty()
    }

    /// Reads `key` as visible to `snapshot`.
    ///
    /// Returns `Ok(None)` when the LSM has no opinion — the caller should then
    /// consult the B+Tree. A visible tombstone returns `Ok(Some(None))`, which
    /// means "definitively deleted; do not fall through".
    pub fn get(&self, key: &[u8], snapshot: u64) -> Result<Option<Option<Vec<u8>>>> {
        // 1. Active MemTable — the newest writes.
        match self.active.get(key, snapshot) {
            Lookup::Found(v) => return Ok(Some(Some(v))),
            Lookup::Deleted => return Ok(Some(None)),
            Lookup::Absent => {}
        }
        // 2. Frozen MemTables, newest first.
        for m in self.frozen.iter().rev() {
            match m.get(key, snapshot) {
                Lookup::Found(v) => return Ok(Some(Some(v))),
                Lookup::Deleted => return Ok(Some(None)),
                Lookup::Absent => {}
            }
        }
        // 3. On-disk levels, newest first.
        for meta in self.manifest.lookup_order(key) {
            let Some(table) = self.open_tables.get(&meta.id) else {
                continue; // not resident; treated as absent at this level
            };
            match table.get(key, snapshot)? {
                Lookup::Found(v) => return Ok(Some(Some(v))),
                Lookup::Deleted => return Ok(Some(None)),
                Lookup::Absent => {}
            }
        }
        Ok(None)
    }

    /// Flushes the oldest frozen MemTable into a new L0 SSTable.
    ///
    /// Ordering: write + `fsync` the table, **then** commit the manifest edit.
    /// A crash between the two leaves an orphaned file that the next
    /// [`LsmEngine::open`] reclaims; it can never leave a manifest entry
    /// pointing at a partial table.
    pub fn flush_one(&mut self) -> Result<Option<TableMeta>> {
        if self.frozen.is_empty() {
            return Ok(None);
        }
        let mut mem = self.frozen.remove(0);
        let highest = mem.max_seqno().unwrap_or(0);
        let id = self.manifest.allocate_table_id();
        let path = self.table_path(id);

        let mut writer = SSTableWriter::create(&path, mem.distinct_key_count())?;
        for (ik, slot) in mem.drain() {
            writer.append(&ik, &slot)?;
        }
        // finish() fsyncs the table before we publish it anywhere.
        let meta = writer.finish(id, 0)?;

        // Commit point: after this returns, the table survives a crash.
        self.log.append(&ManifestEdit::AddTable {
            meta: SerializableMeta::from(&meta),
        })?;

        let table = SSTable::open(&path, id, 0)?;
        self.open_tables.insert(id, table);
        self.manifest.add_table(meta.clone());
        self.flushes += 1;

        // Everything in this table is now durable outside the WAL.
        if highest > self.checkpoint_seqno {
            self.set_checkpoint(highest)?;
        }
        Ok(Some(meta))
    }

    /// Flushes every pending MemTable.
    pub fn flush_all(&mut self) -> Result<usize> {
        let mut n = 0;
        while self.flush_one()?.is_some() {
            n += 1;
        }
        Ok(n)
    }

    /// Runs one compaction if any level is over budget.
    ///
    /// `retain_floor` is the oldest sequence number a live snapshot can read;
    /// pass the engine's merge watermark. Returns the stats of the compaction
    /// performed, or `None` when nothing was due.
    pub fn compact_once(&mut self, retain_floor: u64) -> Result<Option<CompactionStats>> {
        let Some(job) = self.manifest.pick_compaction() else {
            return Ok(None);
        };

        // Collect inputs newest-first: source level entries precede target
        // level entries so merge_runs resolves ties in favour of newer data.
        let mut runs: Vec<Vec<(InternalKey, ValueSlot)>> = Vec::new();
        for meta in job.source_tables.iter().chain(job.target_tables.iter()) {
            if let Some(t) = self.open_tables.get(&meta.id) {
                runs.push(t.entries()?);
            }
        }

        let drop_tombstones = job.is_bottom_most(self.manifest.config());
        let (merged, stats) = merge_runs(runs, retain_floor, drop_tombstones)?;

        // Write the merged run as a single new table at the target level.
        let mut outputs = Vec::new();
        if !merged.is_empty() {
            let id = self.manifest.allocate_table_id();
            let path = self.table_path(id);
            let mut writer = SSTableWriter::create(&path, merged.len())?;
            for (ik, slot) in &merged {
                writer.append(ik, slot)?;
            }
            let meta = writer.finish(id, job.target_level)?;
            self.open_tables
                .insert(id, SSTable::open(&path, id, job.target_level)?);
            outputs.push(meta);
        }

        // Commit the whole swap — additions before removals — so a crash
        // mid-sequence can drop tables but never lose their contents: replaying
        // an AddTable without its matching RemoveTable leaves the old and new
        // tables both live, which is redundant but correct.
        for out in &outputs {
            self.log.append(&ManifestEdit::AddTable {
                meta: SerializableMeta::from(out),
            })?;
        }
        for input in job.all_inputs() {
            self.log.append(&ManifestEdit::RemoveTable {
                level: input.level,
                id: input.id,
            })?;
        }

        // Retire the inputs: manifest first, then the files themselves.
        let retired: Vec<(u64, PathBuf)> = job
            .all_inputs()
            .iter()
            .map(|m| (m.id, self.table_path(m.id)))
            .collect();
        self.manifest.apply(&job, outputs);
        for (id, path) in retired {
            self.open_tables.remove(&id);
            // A failed unlink leaves a harmless orphan; never fail the compaction.
            let _ = std::fs::remove_file(path);
        }

        self.compactions += 1;
        self.maybe_compact_manifest()?;
        Ok(Some(stats))
    }

    /// Rewrites the manifest log as a snapshot once it has grown large.
    ///
    /// Without this the log grows without bound and startup replay slows down
    /// forever, which is the classic LSM manifest failure mode.
    fn maybe_compact_manifest(&mut self) -> Result<()> {
        if self.log.edits_written() < COMPACT_AFTER_EDITS {
            return Ok(());
        }
        let live: Vec<TableMeta> = (0..self.options.levels.max_levels)
            .flat_map(|l| self.manifest.level(l).iter().cloned())
            .collect();
        self.log = Manifest::compact(
            self.dir.join(MANIFEST_FILE),
            &live,
            self.manifest.peek_table_id(),
            self.checkpoint_seqno,
        )?;
        Ok(())
    }

    /// Compacts repeatedly until every level is inside its budget.
    ///
    /// `max_rounds` bounds the work so a pathological layout cannot spin.
    pub fn compact_until_stable(
        &mut self,
        retain_floor: u64,
        max_rounds: usize,
    ) -> Result<Vec<CompactionStats>> {
        let mut all = Vec::new();
        for _ in 0..max_rounds {
            match self.compact_once(retain_floor)? {
                Some(s) => all.push(s),
                None => break,
            }
        }
        Ok(all)
    }
}

impl std::fmt::Debug for LsmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LsmEngine")
            .field("dir", &self.dir)
            .field("stats", &self.stats())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine(dir: &Path, opts: LsmOptions) -> LsmEngine {
        LsmEngine::open(dir, opts).unwrap()
    }

    /// Small budgets so tests trigger real rotations and compactions cheaply.
    fn tiny_options() -> LsmOptions {
        LsmOptions {
            memtable_bytes: 512,
            levels: LevelConfig {
                l0_compaction_trigger: 2,
                base_level_bytes: 2048,
                size_multiplier: 4,
                max_levels: 4,
            },
        }
    }

    /// A large MemTable budget so only explicit `rotate()` calls freeze a
    /// table, making table boundaries deterministic, paired with an eager L0
    /// trigger so compaction still fires.
    fn manual_flush_options() -> LsmOptions {
        LsmOptions {
            memtable_bytes: 8 * 1024 * 1024,
            levels: LevelConfig {
                l0_compaction_trigger: 2,
                base_level_bytes: 2048,
                size_multiplier: 4,
                max_levels: 4,
            },
        }
    }

    #[test]
    fn put_and_get_through_the_memtable() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), LsmOptions::default());
        e.put(b"k".to_vec(), b"v".to_vec(), 1);
        assert_eq!(e.get(b"k", 1).unwrap(), Some(Some(b"v".to_vec())));
        // Unknown key: the LSM has no opinion, so the caller falls through.
        assert_eq!(e.get(b"nope", 1).unwrap(), None);
    }

    #[test]
    fn reads_survive_a_flush_to_disk() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), LsmOptions::default());
        for i in 0..100u32 {
            e.put(
                format!("k{i:04}").into_bytes(),
                format!("v{i}").into_bytes(),
                i as u64 + 1,
            );
        }
        e.rotate();
        let meta = e.flush_one().unwrap().expect("a table was written");
        assert_eq!(meta.level, 0);
        assert!(meta.file_bytes > 0);
        assert!(
            e.table_path(meta.id).exists(),
            "SSTable file must exist on disk"
        );

        for i in 0..100u32 {
            let got = e.get(format!("k{i:04}").as_bytes(), 1000).unwrap();
            assert_eq!(
                got,
                Some(Some(format!("v{i}").into_bytes())),
                "lost k{i:04}"
            );
        }
        assert_eq!(e.stats().flushes, 1);
        assert_eq!(e.stats().table_count, 1);
    }

    #[test]
    fn memtable_rotates_when_it_exceeds_its_budget() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), tiny_options());
        assert_eq!(e.stats().frozen_count, 0);
        for i in 0..50u32 {
            e.put(
                format!("key{i:04}").into_bytes(),
                vec![b'x'; 64],
                i as u64 + 1,
            );
        }
        assert!(
            e.needs_flush(),
            "writes past the budget must freeze a table"
        );
        assert!(e.stats().frozen_count >= 1);
    }

    #[test]
    fn tombstone_in_a_newer_level_masks_an_older_value() {
        // The classic resurrection bug: value in L0 table #1, delete in the
        // MemTable. The delete must win, and must stop the search.
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), LsmOptions::default());
        e.put(b"k".to_vec(), b"original".to_vec(), 1);
        e.rotate();
        e.flush_one().unwrap();

        e.delete(b"k".to_vec(), 5);
        assert_eq!(
            e.get(b"k", 5).unwrap(),
            Some(None),
            "delete must mask the flushed value"
        );
        // An older snapshot still sees the value.
        assert_eq!(e.get(b"k", 3).unwrap(), Some(Some(b"original".to_vec())));
    }

    #[test]
    fn newer_flushed_table_shadows_an_older_one() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), LsmOptions::default());
        e.put(b"k".to_vec(), b"v1".to_vec(), 1);
        e.rotate();
        e.flush_one().unwrap();

        e.put(b"k".to_vec(), b"v2".to_vec(), 10);
        e.rotate();
        e.flush_one().unwrap();

        assert_eq!(e.stats().table_count, 2);
        assert_eq!(e.get(b"k", 10).unwrap(), Some(Some(b"v2".to_vec())));
        assert_eq!(e.get(b"k", 5).unwrap(), Some(Some(b"v1".to_vec())));
    }

    #[test]
    fn compaction_merges_l0_and_preserves_all_live_data() {
        let d = tempfile::tempdir().unwrap();
        // Manual flush control: every key must land in one of the two tables.
        let mut e = engine(d.path(), manual_flush_options());

        // Two disjoint tables in L0, enough to hit the trigger.
        for i in 0..20u32 {
            e.put(
                format!("a{i:03}").into_bytes(),
                format!("v{i}").into_bytes(),
                i as u64 + 1,
            );
        }
        e.rotate();
        e.flush_one().unwrap();
        for i in 0..20u32 {
            e.put(
                format!("b{i:03}").into_bytes(),
                format!("w{i}").into_bytes(),
                i as u64 + 100,
            );
        }
        e.rotate();
        e.flush_one().unwrap();
        assert_eq!(e.manifest().level(0).len(), 2);

        let stats = e.compact_once(1000).unwrap().expect("L0 hit its trigger");
        assert_eq!(stats.entries_in, 40);
        assert_eq!(stats.entries_out, 40, "disjoint keys: nothing to drop");
        assert!(e.manifest().level(0).is_empty(), "L0 drained");
        assert_eq!(e.manifest().level(1).len(), 1, "one merged table in L1");

        // Every key must still be readable after the merge.
        for i in 0..20u32 {
            assert_eq!(
                e.get(format!("a{i:03}").as_bytes(), 1000).unwrap(),
                Some(Some(format!("v{i}").into_bytes()))
            );
            assert_eq!(
                e.get(format!("b{i:03}").as_bytes(), 1000).unwrap(),
                Some(Some(format!("w{i}").into_bytes()))
            );
        }
        assert_eq!(e.stats().compactions, 1);
    }

    #[test]
    fn compaction_collapses_overwrites_of_the_same_key() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), tiny_options());
        // Same key written in three separate flushed tables.
        for (n, seq) in [(b"v1", 1u64), (b"v2", 2), (b"v3", 3)] {
            e.put(b"hot".to_vec(), n.to_vec(), seq);
            e.rotate();
            e.flush_one().unwrap();
        }
        assert_eq!(e.manifest().level(0).len(), 3);

        // retain_floor 3: no reader needs the older versions.
        let stats = e.compact_once(3).unwrap().unwrap();
        assert_eq!(stats.entries_in, 3);
        assert_eq!(stats.entries_out, 1, "only the newest version survives");
        assert_eq!(e.get(b"hot", 3).unwrap(), Some(Some(b"v3".to_vec())));
    }

    #[test]
    fn compacted_input_files_are_removed_from_disk() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), tiny_options());
        for i in 0..10u32 {
            e.put(format!("k{i}").into_bytes(), b"v".to_vec(), i as u64 + 1);
            e.rotate();
            e.flush_one().unwrap();
        }
        let before: Vec<PathBuf> = e
            .manifest()
            .level(0)
            .iter()
            .map(|m| e.table_path(m.id))
            .collect();
        assert!(before.iter().all(|p| p.exists()));

        e.compact_until_stable(1000, 10).unwrap();
        for p in &before {
            assert!(!p.exists(), "retired input {p:?} must be unlinked");
        }
        // And the data is still there.
        for i in 0..10u32 {
            assert_eq!(
                e.get(format!("k{i}").as_bytes(), 1000).unwrap(),
                Some(Some(b"v".to_vec()))
            );
        }
    }

    #[test]
    fn compact_once_is_a_noop_when_every_level_is_in_budget() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), LsmOptions::default());
        e.put(b"k".to_vec(), b"v".to_vec(), 1);
        e.rotate();
        e.flush_one().unwrap();
        assert!(e.compact_once(10).unwrap().is_none());
    }

    #[test]
    fn compact_until_stable_terminates_and_drains_l0() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), tiny_options());
        for i in 0..12u32 {
            e.put(
                format!("k{i:03}").into_bytes(),
                vec![b'z'; 32],
                i as u64 + 1,
            );
            e.rotate();
            e.flush_one().unwrap();
        }
        let rounds = e.compact_until_stable(1000, 50).unwrap();
        assert!(!rounds.is_empty(), "a backlog of 12 L0 tables must compact");
        assert!(
            e.manifest().level(0).len() < 2,
            "L0 must end inside its trigger, got {}",
            e.manifest().level(0).len()
        );
        for i in 0..12u32 {
            assert_eq!(
                e.get(format!("k{i:03}").as_bytes(), 1000).unwrap(),
                Some(Some(vec![b'z'; 32]))
            );
        }
    }

    #[test]
    fn flush_all_drains_every_frozen_table() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), LsmOptions::default());
        for i in 0..3u32 {
            e.put(format!("k{i}").into_bytes(), b"v".to_vec(), i as u64 + 1);
            e.rotate();
        }
        assert_eq!(e.stats().frozen_count, 3);
        assert_eq!(e.flush_all().unwrap(), 3);
        assert_eq!(e.stats().frozen_count, 0);
        assert_eq!(e.stats().table_count, 3);
        assert!(!e.needs_flush());
    }

    #[test]
    fn rotating_an_empty_memtable_does_nothing() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), LsmOptions::default());
        e.rotate();
        e.rotate();
        assert_eq!(e.stats().frozen_count, 0);
        assert!(e.flush_one().unwrap().is_none());
    }

    #[test]
    fn deletes_survive_compaction_above_the_bottom_level() {
        let d = tempfile::tempdir().unwrap();
        let mut e = engine(d.path(), tiny_options());
        e.put(b"k".to_vec(), b"value".to_vec(), 1);
        e.rotate();
        e.flush_one().unwrap();
        e.delete(b"k".to_vec(), 2);
        e.rotate();
        e.flush_one().unwrap();

        // Compacting L0 -> L1 with max_levels 4: not bottom-most, so the
        // tombstone must be retained rather than dropped.
        e.compact_once(2).unwrap().unwrap();
        assert_eq!(
            e.get(b"k", 2).unwrap(),
            Some(None),
            "the key must stay deleted after compaction"
        );
    }
}
