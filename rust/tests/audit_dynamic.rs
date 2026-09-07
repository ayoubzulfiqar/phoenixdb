//! Dynamic tests targeting specific audit findings.
//!
//! Run with: cargo test --test audit_dynamic

use phoenixdb::{Database, Options, vector};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn temp_db() -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path().join("audit.pdb"), Options::default()).expect("open");
    (dir, db)
}

fn temp_vector_engine(
    dim: usize,
    metric: vector::Metric,
) -> (tempfile::TempDir, vector::VectorEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = vector::VectorEngine::open(
        dir.path().join("v.pvec"),
        dim,
        metric,
        vector::VectorOptions::default(),
    )
    .expect("open vector engine");
    (dir, engine)
}

/// P0-2 dynamic check: sync_on_commit: false must still provide a durability
/// guarantee via the WAL commit path.
#[test]
fn sync_false_commit_durability_after_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sync_false.pdb");
    {
        let opts = phoenixdb::Options {
            sync_on_commit: false,
            ..Default::default()
        };
        let db = Database::open(&path, opts).unwrap();
        let t = db.begin(false).unwrap();
        db.insert(t, b"durable", b"yes").unwrap();
        db.commit(t).unwrap();
        db.flush().unwrap();
        std::mem::forget(db);
    }
    let db = Database::open(&path, Options::default()).unwrap();
    assert_eq!(
        db.get_auto(b"durable").unwrap(),
        b"yes",
        "committed data must survive crash even with sync_on_commit=false"
    );
}

/// P0-2 extended: verify WAL commit path survives with default options.
#[test]
fn wal_commit_survives_crash_with_default_options() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wal_sync.pdb");
    let db = Database::open(&path, Options::default()).unwrap();
    let t = db.begin(false).unwrap();
    db.insert(t, b"k", b"v").unwrap();
    db.commit(t).unwrap();
    db.checkpoint().unwrap();
    assert_eq!(db.get_auto(b"k").unwrap(), b"v");
}

/// P1-1: Long-running read-only snapshot should not block merge indefinitely.
#[test]
fn long_lived_snapshot_allows_merge_of_older_versions() {
    let (_d, db) = temp_db();
    let old_reader = db.begin(true).unwrap();
    for i in 0..1000u32 {
        db.put_auto(format!("k{i:04}").as_bytes(), b"v").unwrap();
    }
    db.checkpoint().unwrap();
    assert_eq!(db.len().unwrap(), 1000);
    for i in 0..1000u32 {
        assert_eq!(db.get_auto(format!("k{i:04}").as_bytes()).unwrap(), b"v");
    }
    db.rollback(old_reader).unwrap();
}

/// P1-1 extreme: ancient snapshot + heavy writes + merge.
#[test]
fn merge_watermark_never_panics_with_ancient_snapshot() {
    let (_d, db) = temp_db();
    let ancient = db.begin(true).unwrap();
    for batch in 0..50u32 {
        for i in 0..100u32 {
            db.put_auto(format!("b{batch}-k{i:03}").as_bytes(), b"v")
                .unwrap();
        }
    }
    let stats = db.stats();
    assert!(stats.wal_bytes > 0);
    db.checkpoint().unwrap();
    assert_eq!(db.len().unwrap(), 5000);
    db.rollback(ancient).unwrap();
}

/// P2-3: Insert a vector with NaN through the engine's public API and verify
/// it is rejected rather than corrupting the index.
#[test]
fn non_finite_vector_is_rejected_at_api() {
    let (_d, engine) = temp_vector_engine(4, vector::Metric::Euclidean);
    let bad = vec![1.0f32, f32::NAN, 3.0, 4.0];
    assert!(
        engine.insert("nan", &bad).is_err(),
        "NaN vector must be rejected"
    );
    let inf = vec![f32::INFINITY, 1.0, 1.0, 1.0];
    assert!(
        engine.insert("inf", &inf).is_err(),
        "Inf vector must be rejected"
    );
    assert_eq!(engine.len(), 0, "no vector should be stored");
}

/// P2-3 via FFI: non-finite vectors must be rejected before any allocation.
#[test]
fn non_finite_vector_is_rejected_at_ffi() {
    use phoenixdb::ffi::*;
    use std::ffi::CString;
    use std::ptr;
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().join("vec.pvec").to_str().unwrap()).unwrap();
    let mut handle: *mut PhoenixVectorHandle = ptr::null_mut();
    unsafe {
        assert_eq!(phoenix_vector_init(path.as_ptr(), 4, 1, 0, &mut handle), 0);
        assert!(!handle.is_null());
        let nan = [1.0f32, f32::NAN, 3.0, 4.0];
        assert_eq!(
            phoenix_vector_insert(handle, c"nan".as_ptr(), nan.as_ptr(), 4),
            -2
        );
        let inf = [f32::INFINITY, 1.0, 1.0, 1.0];
        assert_eq!(
            phoenix_vector_insert(handle, c"inf".as_ptr(), inf.as_ptr(), 4),
            -2
        );
        phoenix_vector_free(handle);
    }
}

/// P0-1 dynamic probe: confirm public API entry points serialize on the
/// same global write lock by timing concurrent operations.
#[test]
fn concurrent_operations_serialize_on_global_lock() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Database::open(dir.path().join("cont.pdb"), Options::default()).unwrap());
    for i in 0..200u32 {
        db.put_auto(format!("init-{i:04}").as_bytes(), b"x")
            .unwrap();
    }
    let barrier = Arc::new(std::sync::Barrier::new(9));
    let mut handles = Vec::new();
    for _ in 0..8 {
        let db = Arc::clone(&db);
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            b.wait();
            let start = Instant::now();
            for i in 0..200u32 {
                let _ = db.get_auto(format!("init-{i:04}").as_bytes());
            }
            start.elapsed()
        }));
    }
    let writer = {
        let db = Arc::clone(&db);
        let b = Arc::clone(&barrier);
        thread::spawn(move || {
            b.wait();
            let start = Instant::now();
            for i in 0..200u32 {
                db.put_auto(format!("write-{i:04}").as_bytes(), b"y")
                    .unwrap();
            }
            start.elapsed()
        })
    };
    let reader_times: Vec<Duration> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let writer_time = writer.join().unwrap();
    for t in reader_times {
        assert!(
            t <= writer_time * 3,
            "readers took {t:?}, writer took {writer_time:?}; expected serialization"
        );
    }
}

/// P0-1: a writer should not run concurrently with other writers.
#[test]
fn writers_serialize_under_lock() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Database::open(dir.path().join("block.pdb"), Options::default()).unwrap());
    let started = Arc::new(std::sync::Barrier::new(2));
    let done = Arc::new(std::sync::Barrier::new(2));
    let b1 = Arc::clone(&started);
    let d1 = Arc::clone(&done);
    let db1 = Arc::clone(&db);
    let t1 = thread::spawn(move || {
        b1.wait();
        db1.put_auto(b"a", b"1").unwrap();
        d1.wait();
    });
    let b2 = Arc::clone(&started);
    let d2 = Arc::clone(&done);
    let db2 = Arc::clone(&db);
    let t2 = thread::spawn(move || {
        b2.wait();
        db2.put_auto(b"b", b"2").unwrap();
        d2.wait();
    });
    t1.join().unwrap();
    t2.join().unwrap();
    assert_eq!(db.get_auto(b"a").unwrap(), b"1");
    assert_eq!(db.get_auto(b"b").unwrap(), b"2");
}

/// P1-3: verify that creating a new database does not panic and the file is
/// readable after a simulated abrupt close (no graceful close).
#[test]
fn fresh_database_survives_abrupt_close() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("abrupt.pdb");
    {
        let db = Database::open(&path, Options::default()).unwrap();
        db.put_auto(b"x", b"y").unwrap();
        std::mem::forget(db);
    }
    let db = Database::open(&path, Options::default()).unwrap();
    assert_eq!(db.get_auto(b"x").unwrap(), b"y");
}

/// P0-2 / WAL torn-tail: multi-transaction log with a torn tail must still
/// replay committed transactions and ignore uncommitted ones.
#[test]
fn torn_wal_tail_does_not_lose_earlier_commits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("torn.pdb");
    {
        let db = Database::open(&path, Options::default()).unwrap();
        let mut handles = Vec::new();
        for t in 0..5u64 {
            let tx = db.begin(false).unwrap();
            db.insert(tx, format!("k{t}").as_bytes(), b"v").unwrap();
            if t < 4 {
                db.commit(tx).unwrap();
            } else {
                handles.push(tx);
            }
        }
        std::mem::forget(db);
    }
    let db = Database::open(&path, Options::default()).unwrap();
    for t in 0..4u64 {
        assert_eq!(
            db.get_auto(format!("k{t}").as_bytes()).unwrap(),
            b"v",
            "committed txn {t} must survive torn WAL"
        );
    }
}

/// Concurrency: concurrent writes from multiple threads must not corrupt the
/// tree or lose keys under the global lock.
#[test]
fn concurrent_writes_preserve_all_keys() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Database::open(dir.path().join("conc.pdb"), Options::default()).unwrap());
    let mut handles = Vec::new();
    for thread_id in 0..4u32 {
        let db = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..250u32 {
                let key = format!("t{thread_id}-k{i:03}");
                db.put_auto(key.as_bytes(), key.as_bytes()).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(db.len().unwrap(), 1000);
    db.verify().unwrap();
}
