//! Reactive change notifications through the public `Database` API: what a
//! watcher sees, when it sees it, and what it must never see.

use phoenixdb::watch::{ChangeKind, WatchOptions};
use phoenixdb::{Database, Options};
use std::time::Duration;

fn temp(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(name);
    (dir, path)
}

fn open(path: &std::path::Path) -> Database {
    Database::open(
        path,
        Options {
            sync_on_commit: false,
            ..Options::default()
        },
    )
    .unwrap()
}

const SOON: Duration = Duration::from_secs(5);

#[test]
fn committed_writes_reach_a_watcher_in_order() {
    let (_dir, path) = temp("order.pdb");
    let db = open(&path);
    let watcher = db.watch(
        b"",
        WatchOptions {
            values: true,
            ..WatchOptions::default()
        },
    );

    db.put_auto(b"a", b"1").unwrap();
    db.put_auto(b"b", b"2").unwrap();
    db.delete_auto(b"a").unwrap();

    let mut seen = Vec::new();
    while seen.len() < 3 {
        let batch = watcher.poll(SOON);
        assert!(!batch.is_empty(), "poll timed out with {} seen", seen.len());
        seen.extend(batch);
    }
    assert_eq!(
        seen.iter()
            .map(|c| (c.kind, c.key.clone(), c.value.clone()))
            .collect::<Vec<_>>(),
        vec![
            (ChangeKind::Put, b"a".to_vec(), Some(b"1".to_vec())),
            (ChangeKind::Put, b"b".to_vec(), Some(b"2".to_vec())),
            (ChangeKind::Delete, b"a".to_vec(), None),
        ]
    );
    assert!(seen[0].commit_ts <= seen[1].commit_ts);
    assert_eq!(watcher.dropped(), 0);
}

#[test]
fn a_prefix_watcher_ignores_everything_else() {
    let (_dir, path) = temp("prefix.pdb");
    let db = open(&path);
    let users = db.watch(b"user:", WatchOptions::default());
    assert_eq!(db.watcher_count(), 1);

    db.write_batch(|b| {
        b.put(b"user:1", b"ada")?;
        b.put(b"post:1", b"hello")?;
        b.put(b"user:2", b"grace")
    })
    .unwrap();

    let seen = users.poll(SOON);
    assert_eq!(
        seen.iter().map(|c| c.key.clone()).collect::<Vec<_>>(),
        vec![b"user:1".to_vec(), b"user:2".to_vec()],
        "one batch, only the matching keys"
    );
    assert_eq!(seen[0].value, None, "values are opt-in");
}

#[test]
fn uncommitted_and_rolled_back_work_is_never_published() {
    let (_dir, path) = temp("uncommitted.pdb");
    let db = open(&path);
    let watcher = db.watch(b"", WatchOptions::default());

    let txn = db.begin(false).unwrap();
    db.insert(txn, b"pending", b"x").unwrap();
    assert!(
        watcher.poll(Duration::from_millis(80)).is_empty(),
        "an open transaction publishes nothing"
    );
    db.rollback(txn).unwrap();
    assert!(
        watcher.poll(Duration::from_millis(80)).is_empty(),
        "a rollback publishes nothing"
    );

    // A read-only transaction that commits is not a change either.
    let read = db.begin(true).unwrap();
    db.commit(read).unwrap();
    assert!(watcher.try_poll().is_empty());

    db.put_auto(b"real", b"y").unwrap();
    assert_eq!(watcher.poll(SOON).len(), 1, "a real commit does arrive");
}

#[test]
fn a_conflicted_commit_publishes_nothing() {
    let (_dir, path) = temp("conflict.pdb");
    let db = open(&path);
    let watcher = db.watch(b"", WatchOptions::default());

    let first = db.begin(false).unwrap();
    let second = db.begin(false).unwrap();
    db.insert(first, b"hot", b"1").unwrap();
    db.insert(second, b"hot", b"2").unwrap();
    db.commit(first).unwrap();
    let conflict = db.commit(second);
    assert!(conflict.is_err(), "the second writer must lose");

    let seen = watcher.poll(SOON);
    assert_eq!(seen.len(), 1, "only the winner was published: {seen:?}");
    assert_eq!(seen[0].key, b"hot".to_vec());
    assert_eq!(
        db.get_auto(b"hot").unwrap(),
        b"1",
        "the winner's value stands"
    );
}

#[test]
fn a_watcher_sees_writes_from_another_thread() {
    let (_dir, path) = temp("threads.pdb");
    let db = open(&path);
    let watcher = db.watch(b"k", WatchOptions::default());

    std::thread::scope(|s| {
        s.spawn(|| {
            for i in 0..50 {
                db.put_auto(format!("k{i}").as_bytes(), b"v").unwrap();
            }
        });
        let mut seen = 0;
        while seen < 50 {
            let batch = watcher.poll(SOON);
            assert!(!batch.is_empty(), "timed out after {seen}");
            seen += batch.len();
        }
    });
}

#[test]
fn a_slow_consumer_loses_the_oldest_changes_and_is_told() {
    let (_dir, path) = temp("slow.pdb");
    let db = open(&path);
    let watcher = db.watch(
        b"",
        WatchOptions {
            capacity: 8,
            values: false,
        },
    );
    for i in 0..40u32 {
        db.put_auto(format!("k{i:03}").as_bytes(), b"v").unwrap();
    }
    let seen = watcher.try_poll();
    assert_eq!(seen.len(), 8, "the queue is bounded");
    assert_eq!(seen.last().unwrap().key, b"k039".to_vec(), "newest kept");
    assert_eq!(watcher.dropped(), 32, "and the loss is reported");
}

#[test]
fn restore_tells_watchers_to_re_read() {
    let (_dir, path) = temp("restore.pdb");
    let backup = path.with_extension("backup");
    let db = open(&path);
    db.put_auto(b"before", b"1").unwrap();
    db.backup(&backup).unwrap();
    db.put_auto(b"after", b"2").unwrap();

    let watcher = db.watch(b"", WatchOptions::default());
    db.restore(&backup).unwrap();
    let seen = watcher.poll(SOON);
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].kind, ChangeKind::Reset);
    assert!(seen[0].key.is_empty());
    assert!(db.get_auto(b"after").is_err(), "the restore took effect");
}

#[test]
fn dropping_the_database_closes_a_blocked_watcher() {
    let (_dir, path) = temp("closed.pdb");
    let db = open(&path);
    let watcher = db.watch(b"", WatchOptions::default());
    assert!(!watcher.is_closed());

    let started = std::time::Instant::now();
    std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(Duration::from_millis(30));
            drop(db);
        });
        // Returns as soon as the database goes away, not after the timeout.
        assert!(watcher.poll(Duration::from_secs(30)).is_empty());
    });
    assert!(watcher.is_closed());
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
fn watching_costs_nothing_when_no_one_listens() {
    let (_dir, path) = temp("idle.pdb");
    let db = open(&path);
    assert_eq!(db.watcher_count(), 0);
    {
        let _w = db.watch(b"", WatchOptions::default());
        assert_eq!(db.watcher_count(), 1);
    }
    assert_eq!(db.watcher_count(), 0, "a dropped watcher unsubscribes");
    db.put_auto(b"k", b"v").unwrap();
}

#[test]
fn sql_and_prefs_style_writes_are_published_too() {
    let (_dir, path) = temp("layers.pdb");
    let db = open(&path);
    let watcher = db.watch(b"", WatchOptions::default());
    db.write_batch(|b| {
        b.put(b"pref:theme", b"dark")?;
        b.delete_if_exists(b"pref:stale")
    })
    .unwrap();
    let seen = watcher.poll(SOON);
    assert_eq!(seen.len(), 1, "a no-op delete is not a change: {seen:?}");
    assert_eq!(seen[0].key, b"pref:theme".to_vec());
}
