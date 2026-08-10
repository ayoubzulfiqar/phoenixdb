//! Integration tests exercising the public engine API end to end.
//!
//! Unit tests live next to the code they cover; this file focuses on
//! cross-module behaviour: durability, recovery, concurrency and corruption
//! detection through `Database`.

use phoenixdb::{Database, Error, Options};
use std::sync::Arc;

fn temp_db() -> (tempfile::TempDir, Database) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path().join("it.pdb"), Options::default()).expect("open");
    (dir, db)
}

#[test]
fn acid_atomicity_all_or_nothing() {
    let (_d, db) = temp_db();
    let txn = db.begin(false).unwrap();
    db.insert(txn, b"a", b"1").unwrap();
    db.insert(txn, b"b", b"2").unwrap();
    db.insert(txn, b"c", b"3").unwrap();
    db.rollback(txn).unwrap();

    for key in [b"a".as_ref(), b"b".as_ref(), b"c".as_ref()] {
        assert!(matches!(db.get_auto(key), Err(Error::NotFound)));
    }
    assert_eq!(db.len().unwrap(), 0);
}

#[test]
fn acid_isolation_snapshot_is_stable() {
    let (_d, db) = temp_db();
    db.put_auto(b"k", b"v1").unwrap();

    let reader = db.begin(true).unwrap();
    assert_eq!(db.get(reader, b"k").unwrap(), b"v1");

    db.put_auto(b"k", b"v2").unwrap();

    // The open snapshot still observes the old value.
    assert_eq!(db.get(reader, b"k").unwrap(), b"v1");
    assert_eq!(db.get_auto(b"k").unwrap(), b"v2");
    db.rollback(reader).unwrap();
}

#[test]
fn acid_durability_after_simulated_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crash.pdb");
    {
        let db = Database::open(&path, Options::default()).unwrap();
        for i in 0..64u32 {
            db.put_auto(format!("k{i:03}").as_bytes(), format!("v{i}").as_bytes())
                .unwrap();
        }
        // No close/checkpoint: recovery must replay the WAL.
        std::mem::forget(db);
    }
    let db = Database::open(&path, Options::default()).unwrap();
    assert_eq!(db.len().unwrap(), 64);
    for i in 0..64u32 {
        assert_eq!(
            db.get_auto(format!("k{i:03}").as_bytes()).unwrap(),
            format!("v{i}").as_bytes()
        );
    }
    db.verify().unwrap();
}

#[test]
fn write_write_conflict_is_reported() {
    let (_d, db) = temp_db();
    db.put_auto(b"k", b"base").unwrap();

    let a = db.begin(false).unwrap();
    let b = db.begin(false).unwrap();
    db.insert(a, b"k", b"from-a").unwrap();
    db.insert(b, b"k", b"from-b").unwrap();

    db.commit(a).unwrap();
    let err = db.commit(b).unwrap_err();
    assert!(matches!(err, Error::Conflict), "got {err:?}");
    assert_eq!(db.get_auto(b"k").unwrap(), b"from-a");
}

#[test]
fn many_concurrent_readers_one_writer() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(Database::open(dir.path().join("mt.pdb"), Options::default()).unwrap());
    for i in 0..100u32 {
        db.put_auto(format!("k{i:04}").as_bytes(), b"v0").unwrap();
    }

    let mut readers = Vec::new();
    for _ in 0..8 {
        let db = Arc::clone(&db);
        readers.push(std::thread::spawn(move || {
            let mut seen = 0;
            for i in 0..100u32 {
                if db.get_auto(format!("k{i:04}").as_bytes()).is_ok() {
                    seen += 1;
                }
            }
            seen
        }));
    }
    let writer = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || {
            for i in 0..100u32 {
                db.put_auto(format!("k{i:04}").as_bytes(), b"v1").unwrap();
            }
        })
    };
    for r in readers {
        assert_eq!(r.join().unwrap(), 100, "a reader lost visibility of a key");
    }
    writer.join().unwrap();
    assert_eq!(db.len().unwrap(), 100);
    db.verify().unwrap();
}

#[test]
fn checkpoint_then_reopen_preserves_everything() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ckpt.pdb");
    {
        let db = Database::open(&path, Options::default()).unwrap();
        for i in 0..500u32 {
            db.put_auto(format!("key{i:05}").as_bytes(), &vec![(i % 256) as u8; 300])
                .unwrap();
        }
        db.checkpoint().unwrap();
        assert!(db.stats().wal_bytes < 1024);
    }
    let db = Database::open(&path, Options::default()).unwrap();
    assert_eq!(db.len().unwrap(), 500);
    assert_eq!(db.get_auto(b"key00042").unwrap(), vec![42u8; 300]);
    db.verify().unwrap();
}

#[test]
fn page_corruption_is_detected_not_returned() {
    use std::io::{Read, Seek, SeekFrom, Write};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt.pdb");
    {
        let db = Database::open(&path, Options::default()).unwrap();
        for i in 0..200u32 {
            db.put_auto(format!("k{i:04}").as_bytes(), b"payload")
                .unwrap();
        }
        db.checkpoint().unwrap();
    }
    // Flip bits in a data page (page 3 is well past the meta page).
    {
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let offset = 3 * phoenixdb::PAGE_SIZE as u64 + 64;
        f.seek(SeekFrom::Start(offset)).unwrap();
        let mut buf = [0u8; 16];
        f.read_exact(&mut buf).unwrap();
        for b in buf.iter_mut() {
            *b ^= 0xA5;
        }
        f.seek(SeekFrom::Start(offset)).unwrap();
        f.write_all(&buf).unwrap();
        f.sync_all().unwrap();
    }
    // Opening or scanning must surface corruption rather than bad data.
    let outcome = Database::open(&path, Options::default()).and_then(|db| db.scan().map(|_| ()));
    match outcome {
        Err(Error::Corruption(_)) => {}
        Err(other) => panic!("expected Corruption, got {other:?}"),
        Ok(()) => panic!("corrupted page was silently accepted"),
    }
}

#[test]
fn large_dataset_round_trip() {
    let (_d, db) = temp_db();
    const N: u32 = 5_000;
    for i in 0..N {
        db.put_auto(
            format!("key-{i:07}").as_bytes(),
            format!("value-{i}").as_bytes(),
        )
        .unwrap();
    }
    db.checkpoint().unwrap();
    db.verify().unwrap();
    assert_eq!(db.len().unwrap(), N as u64);
    for i in (0..N).step_by(97) {
        assert_eq!(
            db.get_auto(format!("key-{i:07}").as_bytes()).unwrap(),
            format!("value-{i}").as_bytes()
        );
    }
    let scanned = db.scan().unwrap();
    for w in scanned.windows(2) {
        assert!(w[0].0 < w[1].0, "scan order violated");
    }
}

#[test]
fn delete_is_durable_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("del.pdb");
    {
        let db = Database::open(&path, Options::default()).unwrap();
        db.put_auto(b"keep", b"1").unwrap();
        db.put_auto(b"remove", b"2").unwrap();
        db.delete_auto(b"remove").unwrap();
        db.checkpoint().unwrap();
    }
    let db = Database::open(&path, Options::default()).unwrap();
    assert_eq!(db.get_auto(b"keep").unwrap(), b"1");
    assert!(matches!(db.get_auto(b"remove"), Err(Error::NotFound)));
    assert_eq!(db.len().unwrap(), 1);
}
