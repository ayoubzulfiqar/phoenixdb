//! Durability tests for the LSM manifest.
//!
//! These are the tests that justify the manifest existing: each one closes the
//! engine and reopens it from disk, so a regression that loses the level layout
//! fails here rather than silently in production.

use phoenixdb::lsm::compaction::LevelConfig;
use phoenixdb::lsm::manifest::{MANIFEST_FILE, Manifest, ManifestEdit, SerializableMeta};
use phoenixdb::lsm::{LsmEngine, LsmOptions, TableMeta};
use std::path::Path;

fn tiny_options() -> LsmOptions {
    LsmOptions {
        memtable_bytes: 8 * 1024 * 1024, // manual rotation control
        levels: LevelConfig {
            l0_compaction_trigger: 2,
            base_level_bytes: 2048,
            size_multiplier: 4,
            max_levels: 4,
        },
    }
}

fn meta(id: u64, level: u32) -> TableMeta {
    TableMeta {
        id,
        level,
        min_key: format!("a{id}").into_bytes(),
        max_key: format!("z{id}").into_bytes(),
        min_seqno: 1,
        max_seqno: 100,
        entry_count: 10,
        file_bytes: 4096,
    }
}

// ---- the core promise: data survives a restart ----------------------------

#[test]
fn flushed_data_is_readable_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
        for i in 0..100u32 {
            e.put(
                format!("key{i:04}").into_bytes(),
                format!("value{i}").into_bytes(),
                i as u64 + 1,
            );
        }
        e.rotate();
        e.flush_one().unwrap();
        assert_eq!(e.stats().table_count, 1);
    } // engine dropped

    let e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
    assert_eq!(e.stats().table_count, 1, "the manifest must survive");
    for i in 0..100u32 {
        assert_eq!(
            e.get(format!("key{i:04}").as_bytes(), 1000).unwrap(),
            Some(Some(format!("value{i}").into_bytes())),
            "lost key{i:04} across restart"
        );
    }
}

#[test]
fn level_placement_survives_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
        for batch in 0..3u32 {
            for i in 0..20u32 {
                let k = format!("k{batch}{i:03}");
                e.put(k.into_bytes(), b"v".to_vec(), (batch * 100 + i) as u64 + 1);
            }
            e.rotate();
            e.flush_one().unwrap();
        }
        e.compact_until_stable(10_000, 20).unwrap();
        assert!(!e.manifest().level(1).is_empty(), "expected data in L1");
    }

    let e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
    assert!(
        !e.manifest().level(1).is_empty(),
        "a compacted table must reopen at its own level, not L0"
    );
    for batch in 0..3u32 {
        for i in 0..20u32 {
            let k = format!("k{batch}{i:03}");
            assert_eq!(
                e.get(k.as_bytes(), 10_000).unwrap(),
                Some(Some(b"v".to_vec())),
                "lost {k} across restart"
            );
        }
    }
}

#[test]
fn tombstones_survive_reopen() {
    // The nastiest regression: if a delete is lost on restart, deleted data
    // comes back from the dead.
    let dir = tempfile::tempdir().unwrap();
    {
        let mut e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
        e.put(b"gone".to_vec(), b"value".to_vec(), 1);
        e.rotate();
        e.flush_one().unwrap();
        e.delete(b"gone".to_vec(), 5);
        e.rotate();
        e.flush_one().unwrap();
    }

    let e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
    assert_eq!(
        e.get(b"gone", 10).unwrap(),
        Some(None),
        "the key must stay deleted after a restart"
    );
}

#[test]
fn multiple_reopens_are_stable() {
    let dir = tempfile::tempdir().unwrap();
    for round in 0..5u64 {
        let mut e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
        e.put(format!("r{round}").into_bytes(), b"v".to_vec(), round + 1);
        e.rotate();
        e.flush_one().unwrap();
        // Everything written in earlier rounds must still be visible.
        for earlier in 0..=round {
            assert_eq!(
                e.get(format!("r{earlier}").as_bytes(), 1000).unwrap(),
                Some(Some(b"v".to_vec())),
                "round {round} lost r{earlier}"
            );
        }
    }
    let e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
    assert_eq!(e.stats().table_count, 5);
}

#[test]
fn checkpoint_seqno_persists() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
        assert_eq!(e.checkpoint_seqno(), 0);
        e.put(b"k".to_vec(), b"v".to_vec(), 42);
        e.rotate();
        e.flush_one().unwrap();
        assert_eq!(e.checkpoint_seqno(), 42, "flush advances the checkpoint");
    }
    let e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
    assert_eq!(e.checkpoint_seqno(), 42, "checkpoint must be durable");
}

// ---- crash resilience -----------------------------------------------------

#[test]
fn an_orphaned_sstable_is_reclaimed_on_open() {
    // Simulates a crash between writing a table and committing its edit.
    let dir = tempfile::tempdir().unwrap();
    {
        let mut e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
        e.put(b"k".to_vec(), b"v".to_vec(), 1);
        e.rotate();
        e.flush_one().unwrap();
    }
    // A plausible-looking table the manifest never heard of.
    let orphan = dir.path().join("0000009999.sst");
    std::fs::write(&orphan, b"not a real sstable").unwrap();
    assert!(orphan.exists());

    let e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
    assert!(!orphan.exists(), "unreferenced .sst must be deleted");
    assert_eq!(e.get(b"k", 10).unwrap(), Some(Some(b"v".to_vec())));
}

#[test]
fn a_torn_manifest_tail_is_truncated_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
        for i in 0..3u32 {
            e.put(format!("k{i}").into_bytes(), b"v".to_vec(), i as u64 + 1);
            e.rotate();
            e.flush_one().unwrap();
        }
    }
    // Append a half-written frame, exactly what a power loss produces.
    let path = dir.path().join(MANIFEST_FILE);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.extend_from_slice(&[40u8, 0, 0, 0, 1, 2, 3, 4, 9, 9]);
    std::fs::write(&path, &bytes).unwrap();

    let state = Manifest::recover(&path).unwrap();
    assert_eq!(state.truncated_bytes, 10, "the torn frame is discarded");
    assert_eq!(state.tables.len(), 3, "complete edits must survive");

    // And the engine opens normally.
    let e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
    assert_eq!(e.stats().table_count, 3);
}

#[test]
fn a_missing_manifest_yields_an_empty_engine() {
    let dir = tempfile::tempdir().unwrap();
    let e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
    assert_eq!(e.stats().table_count, 0);
    assert_eq!(e.checkpoint_seqno(), 0);
}

#[test]
fn a_referenced_but_corrupt_table_is_a_hard_error() {
    // Losing data silently is worse than refusing to start.
    let dir = tempfile::tempdir().unwrap();
    {
        let mut e = LsmEngine::open(dir.path(), tiny_options()).unwrap();
        e.put(b"k".to_vec(), b"v".to_vec(), 1);
        e.rotate();
        e.flush_one().unwrap();
    }
    // Corrupt the one table the manifest references.
    let table = dir.path().join("0000000001.sst");
    let mut bytes = std::fs::read(&table).unwrap();
    bytes[10] ^= 0xFF;
    std::fs::write(&table, &bytes).unwrap();

    assert!(
        LsmEngine::open(dir.path(), tiny_options()).is_err(),
        "a corrupt referenced table must fail the open, not be skipped"
    );
}

// ---- manifest log mechanics ----------------------------------------------

#[test]
fn edits_replay_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("M");
    {
        let mut m = Manifest::open(&path).unwrap();
        m.append(&ManifestEdit::AddTable {
            meta: SerializableMeta::from(&meta(1, 0)),
        })
        .unwrap();
        m.append(&ManifestEdit::AddTable {
            meta: SerializableMeta::from(&meta(2, 0)),
        })
        .unwrap();
        m.append(&ManifestEdit::RemoveTable { level: 0, id: 1 })
            .unwrap();
        m.append(&ManifestEdit::Checkpoint { seqno: 77 }).unwrap();
    }
    let state = Manifest::recover(&path).unwrap();
    assert_eq!(state.tables.len(), 1);
    assert_eq!(state.tables[0].id, 2);
    assert_eq!(state.checkpoint_seqno, 77);
    assert_eq!(state.next_table_id, 3, "ids must never be reused");
    assert_eq!(state.edits_replayed, 4);
}

#[test]
fn a_removed_table_id_is_never_reused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("M");
    {
        let mut m = Manifest::open(&path).unwrap();
        m.append(&ManifestEdit::AddTable {
            meta: SerializableMeta::from(&meta(7, 0)),
        })
        .unwrap();
        m.append(&ManifestEdit::RemoveTable { level: 0, id: 7 })
            .unwrap();
    }
    let state = Manifest::recover(&path).unwrap();
    assert!(state.tables.is_empty());
    assert_eq!(
        state.next_table_id, 8,
        "reusing id 7 could resurrect a stale orphaned file"
    );
}

#[test]
fn compaction_snapshots_the_log() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("M");
    {
        let mut m = Manifest::open(&path).unwrap();
        for i in 1..=50u64 {
            m.append(&ManifestEdit::AddTable {
                meta: SerializableMeta::from(&meta(i, 0)),
            })
            .unwrap();
        }
    }
    let before = Manifest::recover(&path).unwrap();
    assert_eq!(before.edits_replayed, 50);
    let size_before = std::fs::metadata(&path).unwrap().len();

    Manifest::compact(&path, &before.tables, before.next_table_id, 99).unwrap();

    let after = Manifest::recover(&path).unwrap();
    assert_eq!(after.tables.len(), 50, "no table may be lost");
    assert_eq!(after.checkpoint_seqno, 99);
    assert_eq!(after.edits_replayed, 2, "snapshot + checkpoint");
    assert!(
        std::fs::metadata(&path).unwrap().len() < size_before,
        "compaction should shrink the log"
    );
}

#[test]
fn a_snapshot_supersedes_earlier_edits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("M");
    {
        let mut m = Manifest::open(&path).unwrap();
        m.append(&ManifestEdit::AddTable {
            meta: SerializableMeta::from(&meta(1, 0)),
        })
        .unwrap();
        m.append(&ManifestEdit::FullSnapshot {
            tables: vec![SerializableMeta::from(&meta(9, 2))],
            next_table_id: 10,
        })
        .unwrap();
    }
    let state = Manifest::recover(&path).unwrap();
    assert_eq!(state.tables.len(), 1);
    assert_eq!(state.tables[0].id, 9, "the pre-snapshot table is gone");
    assert_eq!(state.tables[0].level, 2);
}

#[test]
fn orphan_detection_ignores_referenced_and_foreign_files() {
    let dir = tempfile::tempdir().unwrap();
    let d: &Path = dir.path();
    std::fs::write(d.join("0000000001.sst"), b"x").unwrap();
    std::fs::write(d.join("0000000002.sst"), b"x").unwrap();
    std::fs::write(d.join("MANIFEST"), b"x").unwrap();
    std::fs::write(d.join("notes.txt"), b"x").unwrap();

    let live = vec![meta(1, 0)];
    let orphans = Manifest::orphaned_files(d, &live).unwrap();
    assert_eq!(orphans.len(), 1, "only the unreferenced .sst");
    assert!(orphans[0].ends_with("0000000002.sst"));
}

#[test]
fn manifest_survives_a_large_number_of_edits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("M");
    {
        let mut m = Manifest::open(&path).unwrap();
        for i in 1..=2000u64 {
            m.append(&ManifestEdit::AddTable {
                meta: SerializableMeta::from(&meta(i, 0)),
            })
            .unwrap();
        }
    }
    let state = Manifest::recover(&path).unwrap();
    assert_eq!(state.tables.len(), 2000);
    assert_eq!(state.next_table_id, 2001);
}
