//! Crash-recovery and engine-semantics tests for `Database`.
//!
//! "Crash" means [`Database::simulate_crash`]: the handle is dropped without a
//! checkpoint, so only what reached the WAL and the data file survives.

use phoenixdb::{Database, Error, Options};
use std::collections::BTreeMap;
use std::ops::Bound;
use std::path::Path;
use std::sync::Arc;

fn open(path: &Path) -> Database {
    Database::open(path, Options::default()).unwrap()
}

fn keys(db: &Database) -> Vec<Vec<u8>> {
    db.scan().unwrap().into_iter().map(|(k, _)| k).collect()
}

// ---------------------------------------------------------------------------
// Regressions: each of these lost committed data before the fix.
// ---------------------------------------------------------------------------

#[test]
fn checkpoint_under_a_live_reader_keeps_newer_commits_durable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.pdb");
    {
        let db = open(&path);
        let reader = db.begin(true).unwrap(); // pins the merge watermark
        db.put_auto(b"k", b"v").unwrap();
        db.checkpoint().unwrap(); // cannot merge k, must keep it in the log
        assert_eq!(db.stats().pending_keys, 1);
        let _ = reader;
        db.simulate_crash();
    }
    let db = open(&path);
    assert_eq!(db.get_auto(b"k").unwrap(), b"v");
    db.verify().unwrap();
}

#[test]
fn transaction_spanning_a_checkpoint_survives_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.pdb");
    {
        let db = open(&path);
        let t = db.begin(false).unwrap();
        db.insert(t, b"k", b"v").unwrap();
        db.checkpoint().unwrap();
        db.commit(t).unwrap();
        db.simulate_crash();
    }
    assert_eq!(open(&path).get_auto(b"k").unwrap(), b"v");
}

#[test]
fn commits_after_a_torn_wal_tail_are_recovered() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.pdb");
    {
        let db = open(&path);
        db.put_auto(b"a", b"1").unwrap();
        // After a clean checkpoint there is nothing to replay, so recovery
        // does not rewrite the log: the tear below survives into the next
        // session unless open explicitly cuts it off.
        db.checkpoint().unwrap();
        db.simulate_crash();
    }
    // Garbage from a half-written frame at the end of the log.
    let wal = Database::wal_path(&path);
    let mut bytes = std::fs::read(&wal).unwrap();
    bytes.extend_from_slice(&[0xC8, 0, 0, 0, 1, 2, 3]);
    std::fs::write(&wal, bytes).unwrap();
    {
        let db = open(&path); // must cut the tear off before appending
        db.put_auto(b"b", b"2").unwrap();
        db.simulate_crash();
    }
    let db = open(&path);
    assert_eq!(db.get_auto(b"a").unwrap(), b"1");
    assert_eq!(db.get_auto(b"b").unwrap(), b"2");
}

#[test]
fn scans_merge_memory_and_tree_in_order_without_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("t.pdb"));
    db.put_auto(b"b", b"old").unwrap();
    db.put_auto(b"d", b"gone").unwrap();
    db.checkpoint().unwrap(); // b, d now live in the tree
    db.put_auto(b"b", b"new").unwrap(); // newer version in memory
    db.put_auto(b"a", b"x").unwrap(); // memory only, sorts first
    db.put_auto(b"c", b"y").unwrap(); // memory only, between tree keys
    db.delete_auto(b"d").unwrap(); // tombstone over a tree key

    let expected = vec![
        (b"a".to_vec(), b"x".to_vec()),
        (b"b".to_vec(), b"new".to_vec()),
        (b"c".to_vec(), b"y".to_vec()),
    ];
    assert_eq!(db.scan().unwrap(), expected);
    let mut streamed = Vec::new();
    db.scan_iter(|kv| {
        streamed.push(kv);
        Ok(())
    })
    .unwrap();
    assert_eq!(streamed, expected);
    assert_eq!(db.len().unwrap(), 3);
}

#[test]
fn auto_checkpoint_runs_with_default_options() {
    // Previously it only ran when sync_on_commit was *off*, so the WAL and
    // the in-memory version store grew without bound by default.
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(
        dir.path().join("t.pdb"),
        Options {
            checkpoint_bytes: 64 * 1024,
            ..Options::default()
        },
    )
    .unwrap();
    for i in 0..2000u32 {
        db.put_auto(format!("k{i:05}").as_bytes(), &[5u8; 100])
            .unwrap();
    }
    let stats = db.stats();
    assert!(
        stats.wal_bytes < 200 * 1024,
        "WAL grew to {}",
        stats.wal_bytes
    );
    assert!(stats.pending_keys < 2000, "nothing was merged");
    assert!(stats.tree_ts > 0);
    assert_eq!(db.len().unwrap(), 2000);
}

#[test]
fn unsynced_commits_survive_a_process_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.pdb");
    let fast = Options {
        sync_on_commit: false,
        ..Options::default()
    };
    {
        let db = Database::open(&path, fast).unwrap();
        for i in 0..50u32 {
            db.put_auto(format!("k{i}").as_bytes(), b"v").unwrap();
        }
        assert_eq!(db.metrics().wal_fsyncs.get(), 0, "no fsync per commit");
        db.simulate_crash();
    }
    assert_eq!(open(&path).len().unwrap(), 50);
}

// ---------------------------------------------------------------------------
// Locking
// ---------------------------------------------------------------------------

#[test]
fn a_database_cannot_be_opened_twice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.pdb");
    let first = open(&path);
    assert!(matches!(
        Database::open(&path, Options::default()),
        Err(Error::Busy(_))
    ));
    drop(first);
    open(&path);
}

// ---------------------------------------------------------------------------
// Range, prefix and transactional scans
// ---------------------------------------------------------------------------

#[test]
fn range_and_prefix_scans_see_memory_and_tree() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("t.pdb"));
    for i in 0..100u32 {
        db.put_auto(format!("user:{i:03}").as_bytes(), b"t")
            .unwrap();
    }
    db.put_auto(b"order:1", b"o").unwrap();
    db.checkpoint().unwrap();
    for i in 100..120u32 {
        db.put_auto(format!("user:{i:03}").as_bytes(), b"m")
            .unwrap();
    }
    db.delete_auto(b"user:050").unwrap();
    db.put_auto(b"usera", b"not a user: key").unwrap();

    let mut users = Vec::new();
    db.scan_prefix(b"user:", |k, _| {
        users.push(k);
        Ok(true)
    })
    .unwrap();
    assert_eq!(users.len(), 119);
    assert!(users.windows(2).all(|w| w[0] < w[1]));
    assert!(!users.contains(&b"user:050".to_vec()));
    assert!(!users.contains(&b"usera".to_vec()));

    let window = db
        .range(
            Bound::Included(b"user:098"),
            Bound::Excluded(b"user:102"),
            0,
        )
        .unwrap();
    let got: Vec<&[u8]> = window.iter().map(|(k, _)| k.as_slice()).collect();
    assert_eq!(
        got,
        [&b"user:098"[..], b"user:099", b"user:100", b"user:101"]
    );
    assert_eq!(window[2].1, b"m");

    let first_three = db.range(Bound::Unbounded, Bound::Unbounded, 3).unwrap();
    assert_eq!(first_three.len(), 3);
    assert_eq!(first_three[0].0, b"order:1");
}

#[test]
fn transactional_scan_sees_its_snapshot_and_own_writes() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("t.pdb"));
    db.put_auto(b"a", b"1").unwrap();
    db.put_auto(b"b", b"1").unwrap();
    let t = db.begin(false).unwrap();
    db.put_auto(b"c", b"later").unwrap(); // committed after t's snapshot
    db.insert(t, b"b", b"mine").unwrap();
    db.delete(t, b"a").unwrap();
    db.insert(t, b"z", b"new").unwrap();

    let mut seen = Vec::new();
    db.scan_txn(t, Bound::Unbounded, Bound::Unbounded, |k, v| {
        seen.push((k, v));
        Ok(true)
    })
    .unwrap();
    assert_eq!(
        seen,
        vec![
            (b"b".to_vec(), b"mine".to_vec()),
            (b"z".to_vec(), b"new".to_vec())
        ]
    );
    db.rollback(t).unwrap();
}

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

#[test]
fn write_batch_is_all_or_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("t.pdb"));
    db.write_batch(|b| {
        b.put(b"x", b"1")?;
        b.put(b"y", b"2")?;
        assert_eq!(b.get(b"x")?, b"1", "a batch reads its own writes");
        Ok(())
    })
    .unwrap();
    let failed = db.write_batch(|b| {
        b.put(b"x", b"changed")?;
        b.delete(b"missing") // NotFound aborts the whole batch
    });
    assert!(matches!(failed, Err(Error::NotFound)));
    assert_eq!(db.get_auto(b"x").unwrap(), b"1");
    assert_eq!(
        db.stats().active_txns,
        0,
        "the failed batch was rolled back"
    );
}

// ---------------------------------------------------------------------------
// Backup, restore, compaction
// ---------------------------------------------------------------------------

#[test]
fn backup_is_self_contained_and_restores() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("live.pdb"));
    for i in 0..500u32 {
        db.put_auto(format!("k{i:04}").as_bytes(), &[1u8; 300])
            .unwrap();
    }
    db.checkpoint().unwrap();
    let reader = db.begin(true).unwrap(); // keeps later versions in memory
    db.put_auto(b"k0000", b"memory-only").unwrap();
    let backup = dir.path().join("snap.pdb");
    db.backup(&backup).unwrap();
    db.rollback(reader).unwrap();

    assert!(
        !Database::wal_path(&backup).exists() || {
            std::fs::metadata(Database::wal_path(&backup))
                .unwrap()
                .len()
                < 64
        }
    );
    {
        let copy = open(&backup);
        assert_eq!(copy.len().unwrap(), 500);
        assert_eq!(copy.get_auto(b"k0000").unwrap(), b"memory-only");
        copy.verify().unwrap();
    }

    for i in 0..500u32 {
        db.delete_auto(format!("k{i:04}").as_bytes()).unwrap();
    }
    db.put_auto(b"after", b"x").unwrap();
    db.restore(&backup).unwrap();
    assert_eq!(db.len().unwrap(), 500);
    assert!(matches!(db.get_auto(b"after"), Err(Error::NotFound)));
    db.verify().unwrap();

    // The restore itself is durable.
    let path = db.path().to_path_buf();
    db.simulate_crash();
    let reopened = open(&path);
    assert_eq!(reopened.len().unwrap(), 500);
    assert_eq!(reopened.get_auto(b"k0000").unwrap(), b"memory-only");
}

#[test]
fn backup_and_restore_refuse_unsafe_targets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live.pdb");
    let db = open(&path);
    db.put_auto(b"k", b"v").unwrap();
    assert!(matches!(db.backup(&path), Err(Error::InvalidArgument(_))));
    assert!(matches!(db.restore(&path), Err(Error::InvalidArgument(_))));

    let other_path = dir.path().join("other.pdb");
    let other = open(&other_path);
    assert!(
        db.backup(&other_path).is_err(),
        "must not overwrite an open database"
    );
    drop(other);

    let backup = dir.path().join("b.pdb");
    db.backup(&backup).unwrap();
    let t = db.begin(false).unwrap();
    assert!(matches!(
        db.restore(&backup),
        Err(Error::InvalidArgument(_))
    ));
    db.rollback(t).unwrap();

    let junk = dir.path().join("junk.pdb");
    std::fs::write(&junk, vec![3u8; 8192]).unwrap();
    assert!(db.restore(&junk).is_err());
    assert_eq!(
        db.get_auto(b"k").unwrap(),
        b"v",
        "a failed restore changes nothing"
    );
}

#[test]
fn compact_shrinks_the_file_and_keeps_data() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("t.pdb"));
    for i in 0..3000u32 {
        db.put_auto(format!("k{i:05}").as_bytes(), &[2u8; 200])
            .unwrap();
    }
    db.checkpoint().unwrap();
    for i in 0..3000u32 {
        if i % 10 != 0 {
            db.delete_auto(format!("k{i:05}").as_bytes()).unwrap();
        }
    }
    db.checkpoint().unwrap();
    let before = db.stats().page_count;
    db.compact().unwrap();
    let after = db.stats().page_count;
    assert!(after * 3 < before, "compact: {before} -> {after} pages");
    assert_eq!(db.len().unwrap(), 300);
    let report = db.check().unwrap();
    assert_eq!(report.free_pages, 0);
    assert_eq!(report.keys, 300);
    db.put_auto(b"still", b"writable").unwrap();
    assert_eq!(db.get_auto(b"still").unwrap(), b"writable");
}

// ---------------------------------------------------------------------------
// Metrics and tracing
// ---------------------------------------------------------------------------

#[test]
fn metrics_count_what_actually_happened() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("t.pdb"));
    for i in 0..10u32 {
        db.put_auto(format!("k{i}").as_bytes(), b"v").unwrap();
    }
    let _ = db.get_auto(b"k1").unwrap();
    let _ = db.get_auto(b"missing");
    let t = db.begin(false).unwrap();
    db.insert(t, b"x", b"y").unwrap();
    db.rollback(t).unwrap();
    db.scan().unwrap();
    db.checkpoint().unwrap();

    let m = db.metrics();
    assert_eq!(m.txn_commits.get(), 10);
    assert_eq!(m.txn_rollbacks.get(), 1);
    assert_eq!(m.writes.get(), 11);
    assert_eq!(m.reads.get(), 2);
    assert_eq!(m.scans.get(), 1);
    assert_eq!(m.wal_fsyncs.get(), 10, "one fsync per durable commit");
    assert!(m.wal_bytes_written.get() > 0);
    assert!(m.checkpoints.get() >= 1);
    assert_eq!(m.txn_commit_latency.count(), 10);
    let report = db.metrics_report();
    assert!(report.contains("commits=10"), "{report}");
    assert!(
        db.metrics_prometheus()
            .contains("phoenixdb_checkpoints_total")
    );
}

#[test]
fn conflicts_are_counted_and_abort_the_loser() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("t.pdb"));
    let a = db.begin(false).unwrap();
    let b = db.begin(false).unwrap();
    db.insert(a, b"k", b"a").unwrap();
    db.insert(b, b"k", b"b").unwrap();
    db.commit(a).unwrap();
    assert!(matches!(db.commit(b), Err(Error::Conflict)));
    assert_eq!(db.metrics().txn_conflicts.get(), 1);
    // The loser is gone: nothing pins old versions, nothing to leak.
    assert_eq!(db.stats().active_txns, 0);
    assert!(matches!(db.rollback(b), Err(Error::TxnNotFound(_))));
    assert_eq!(db.get_auto(b"k").unwrap(), b"a");
}

#[test]
fn tracing_records_spans_only_when_enabled() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("t.pdb"));
    db.put_auto(b"k", b"v").unwrap();
    assert!(db.spans().is_empty());
    db.set_tracing(true);
    db.put_auto(b"k2", b"v").unwrap();
    db.checkpoint().unwrap();
    let names: Vec<String> = db.spans().into_iter().map(|s| s.name).collect();
    assert!(names.contains(&"commit".to_string()), "{names:?}");
    assert!(names.contains(&"checkpoint".to_string()), "{names:?}");
    let commit = db.spans().into_iter().find(|s| s.name == "commit").unwrap();
    assert_eq!(commit.attribute("writes"), Some("1"));
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

#[test]
fn readers_and_writers_run_concurrently_and_stay_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(
        Database::open(
            dir.path().join("t.pdb"),
            Options {
                checkpoint_bytes: 32 * 1024, // force checkpoints mid-run
                sync_on_commit: false,
                ..Options::default()
            },
        )
        .unwrap(),
    );
    let mut handles = Vec::new();
    for w in 0..3u32 {
        let db = Arc::clone(&db);
        handles.push(std::thread::spawn(move || {
            for i in 0..300u32 {
                let key = format!("w{w}:{i:04}");
                db.put_auto(key.as_bytes(), key.as_bytes()).unwrap();
                if i % 3 == 0 {
                    db.delete_auto(key.as_bytes()).unwrap();
                }
            }
        }));
    }
    for _ in 0..3 {
        let db = Arc::clone(&db);
        handles.push(std::thread::spawn(move || {
            for _ in 0..40 {
                let items = db.scan().unwrap();
                assert!(items.windows(2).all(|w| w[0].0 < w[1].0));
                for (k, v) in items.iter().take(20) {
                    assert_eq!(k, v, "a key always maps to its own bytes");
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(db.len().unwrap(), 3 * 200);
    db.verify().unwrap();
}

// ---------------------------------------------------------------------------
// Model test: random operations, checkpoints, crashes and reopens
// ---------------------------------------------------------------------------

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn random_workload_with_crashes_matches_a_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.pdb");
    let options = Options {
        checkpoint_bytes: 48 * 1024,
        ..Options::default()
    };
    let mut model: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
    let mut rng = XorShift(0x5EED);
    let mut db = Database::open(&path, options).unwrap();
    for round in 0..25 {
        // A reader that sometimes spans the round pins old versions.
        let reader = (rng.below(3) == 0).then(|| db.begin(true).unwrap());
        for _ in 0..120 {
            let key = format!("key{:04}", rng.below(600)).into_bytes();
            match rng.below(10) {
                0..=5 => {
                    let len = match rng.below(3) {
                        0 => rng.below(20),
                        1 => 500 + rng.below(1500),
                        _ => 4000 + rng.below(20_000),
                    } as usize;
                    let value = vec![(rng.next() & 0xFF) as u8; len];
                    db.put_auto(&key, &value).unwrap();
                    model.insert(key, value);
                }
                6..=7 => {
                    let expected = model.remove(&key).is_some();
                    match db.delete_auto(&key) {
                        Ok(()) => assert!(expected),
                        Err(Error::NotFound) => assert!(!expected),
                        Err(e) => panic!("{e:?}"),
                    }
                }
                8 => {
                    // A multi-key transaction, sometimes rolled back.
                    let t = db.begin(false).unwrap();
                    let other = format!("key{:04}", rng.below(600)).into_bytes();
                    db.insert(t, &key, b"txn").unwrap();
                    db.insert(t, &other, b"txn2").unwrap();
                    if rng.below(2) == 0 {
                        db.commit(t).unwrap();
                        model.insert(key, b"txn".to_vec());
                        model.insert(other, b"txn2".to_vec());
                    } else {
                        db.rollback(t).unwrap();
                    }
                }
                _ => db.checkpoint().unwrap(),
            }
        }
        if let Some(r) = reader {
            db.rollback(r).unwrap();
        }
        // Every few rounds: crash (no checkpoint) or clean close, then reopen.
        if round % 4 == 3 {
            if rng.below(2) == 0 {
                db.simulate_crash();
            } else {
                drop(db);
            }
            db = Database::open(&path, options).unwrap();
        }
        let expected: Vec<(Vec<u8>, Vec<u8>)> =
            model.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        assert_eq!(db.scan().unwrap(), expected, "round {round}");
        assert_eq!(db.len().unwrap(), model.len() as u64, "round {round}");
        db.verify().unwrap();
    }
    assert!(keys(&db).len() == model.len());
}
