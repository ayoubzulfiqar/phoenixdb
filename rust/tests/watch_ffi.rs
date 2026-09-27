//! Change notifications through the C ABI: the wire format, blocking polls,
//! waking a blocked consumer, and handle hygiene.

use phoenixdb::ffi::collection_ffi::{
    PhoenixCollectionHandle, phoenix_collection_close, phoenix_collection_open,
    phoenix_collection_upsert,
};
use phoenixdb::ffi::watch_ffi::{
    PhoenixWatcher, phoenix_collection_watch_open, phoenix_watch_close, phoenix_watch_is_closed,
    phoenix_watch_open, phoenix_watch_poll, phoenix_watch_wake,
};
use phoenixdb::ffi::{
    PhoenixBuffer, PhoenixDbHandle, phoenix_buffer_free, phoenix_close, phoenix_delete,
    phoenix_open, phoenix_put_auto,
};
use std::ffi::CString;
use std::os::raw::c_int;
use std::time::{Duration, Instant};

/// One decoded change, mirroring the documented wire format.
#[derive(Debug, PartialEq)]
struct Wire {
    kind: u8,
    key: Vec<u8>,
    value: Option<Vec<u8>>,
}

/// Decodes a poll buffer exactly as a C caller would, then frees it.
fn take(buffer: &mut PhoenixBuffer) -> Vec<Wire> {
    let bytes = if buffer.ptr.is_null() {
        Vec::new()
    } else {
        // SAFETY: the library wrote `len` valid bytes at `ptr`.
        unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) }.to_vec()
    };
    // SAFETY: a buffer this library produced, freed exactly once.
    unsafe { phoenix_buffer_free(buffer) };

    let mut out = Vec::new();
    let mut at = 0usize;
    while at < bytes.len() {
        let kind = bytes[at];
        at += 1;
        let _commit_ts = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        at += 8;
        let key_len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        let key = bytes[at..at + key_len].to_vec();
        at += key_len;
        let value_len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        at += 4;
        let value = if value_len == u32::MAX {
            None
        } else {
            let end = at + value_len as usize;
            let value = bytes[at..end].to_vec();
            at = end;
            Some(value)
        };
        out.push(Wire { kind, key, value });
    }
    out
}

/// Polls until at least `want` changes have arrived, or fails.
fn collect(watcher: *mut PhoenixWatcher, want: usize) -> Vec<Wire> {
    let mut seen = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    while seen.len() < want && Instant::now() < deadline {
        let mut buffer = PhoenixBuffer {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        let mut dropped = 0u64;
        // SAFETY: a live watcher handle and writable out-params.
        let status = unsafe { phoenix_watch_poll(watcher, 500, &mut buffer, &mut dropped) };
        assert_eq!(status, 0, "poll failed");
        seen.extend(take(&mut buffer));
    }
    assert_eq!(seen.len(), want, "expected {want} changes, saw {seen:?}");
    seen
}

#[test]
fn key_value_changes_cross_the_abi() {
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().join("w.pdb").to_str().unwrap()).unwrap();
    let mut db: *mut PhoenixDbHandle = std::ptr::null_mut();
    // SAFETY: valid arguments throughout.
    unsafe {
        assert_eq!(phoenix_open(path.as_ptr(), 0, &mut db), 0);

        let prefix = b"user:";
        let mut watcher: *mut PhoenixWatcher = std::ptr::null_mut();
        assert_eq!(
            phoenix_watch_open(db, prefix.as_ptr(), prefix.len(), 64, 1, &mut watcher),
            0
        );
        assert!(!watcher.is_null());

        assert_eq!(
            phoenix_put_auto(db, b"user:1".as_ptr(), 6, b"ada".as_ptr(), 3),
            0
        );
        assert_eq!(
            phoenix_put_auto(db, b"post:1".as_ptr(), 6, b"x".as_ptr(), 1),
            0
        );
        assert_eq!(phoenix_delete(db, 0, b"user:1".as_ptr(), 6), 0);

        let seen = collect(watcher, 2);
        assert_eq!(
            seen,
            vec![
                Wire {
                    kind: 1,
                    key: b"user:1".to_vec(),
                    value: Some(b"ada".to_vec()),
                },
                Wire {
                    kind: 2,
                    key: b"user:1".to_vec(),
                    value: None,
                },
            ],
            "the unrelated prefix is filtered out"
        );

        let mut closed: c_int = -1;
        assert_eq!(phoenix_watch_is_closed(watcher, &mut closed), 0);
        assert_eq!(closed, 0);

        phoenix_watch_close(watcher);
        phoenix_watch_close(watcher); // poisoned tag: ignored, no double free
        assert_eq!(phoenix_close(db), 0);
    }
}

#[test]
fn a_poll_blocks_then_wake_returns_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().join("wake.pdb").to_str().unwrap()).unwrap();
    let mut db: *mut PhoenixDbHandle = std::ptr::null_mut();
    // SAFETY: valid arguments throughout.
    unsafe {
        assert_eq!(phoenix_open(path.as_ptr(), 0, &mut db), 0);
        let mut watcher: *mut PhoenixWatcher = std::ptr::null_mut();
        assert_eq!(
            phoenix_watch_open(db, std::ptr::null(), 0, 64, 0, &mut watcher),
            0
        );

        // A poll with nothing to report waits for its timeout.
        let started = Instant::now();
        let mut buffer = PhoenixBuffer {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        assert_eq!(
            phoenix_watch_poll(watcher, 150, &mut buffer, std::ptr::null_mut()),
            0
        );
        assert!(take(&mut buffer).is_empty());
        assert!(
            started.elapsed() >= Duration::from_millis(120),
            "it blocked"
        );

        // `wake` from another thread returns a blocked poll promptly.
        let address = watcher as usize;
        let waker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            phoenix_watch_wake(address as *mut PhoenixWatcher);
        });
        let started = Instant::now();
        let mut buffer = PhoenixBuffer {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        assert_eq!(
            phoenix_watch_poll(watcher, 30_000, &mut buffer, std::ptr::null_mut()),
            0
        );
        assert!(take(&mut buffer).is_empty());
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "wake returned it"
        );
        waker.join().unwrap();

        phoenix_watch_close(watcher);
        assert_eq!(phoenix_close(db), 0);
    }
}

#[test]
fn a_full_queue_reports_how_much_was_lost() {
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().join("drop.pdb").to_str().unwrap()).unwrap();
    let mut db: *mut PhoenixDbHandle = std::ptr::null_mut();
    // SAFETY: valid arguments throughout.
    unsafe {
        assert_eq!(phoenix_open(path.as_ptr(), 0, &mut db), 0);
        let mut watcher: *mut PhoenixWatcher = std::ptr::null_mut();
        assert_eq!(
            phoenix_watch_open(db, std::ptr::null(), 0, 4, 0, &mut watcher),
            0
        );
        for i in 0..20u8 {
            let key = [b'k', i];
            assert_eq!(phoenix_put_auto(db, key.as_ptr(), 2, b"v".as_ptr(), 1), 0);
        }
        let mut buffer = PhoenixBuffer {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        let mut dropped = 0u64;
        assert_eq!(
            phoenix_watch_poll(watcher, 100, &mut buffer, &mut dropped),
            0
        );
        assert_eq!(take(&mut buffer).len(), 4);
        assert_eq!(dropped, 16);

        phoenix_watch_close(watcher);
        assert_eq!(phoenix_close(db), 0);
    }
}

#[test]
fn collection_watchers_report_document_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().to_str().unwrap()).unwrap();
    let options = CString::new(r#"{"dim": 0, "sync": false}"#).unwrap();
    let mut handle: *mut PhoenixCollectionHandle = std::ptr::null_mut();
    // SAFETY: valid arguments throughout.
    unsafe {
        assert_eq!(
            phoenix_collection_open(path.as_ptr(), options.as_ptr(), &mut handle),
            0
        );
        let mut watcher: *mut PhoenixWatcher = std::ptr::null_mut();
        assert_eq!(phoenix_collection_watch_open(handle, 64, &mut watcher), 0);

        let docs = CString::new(r#"[{"id":"doc-1","text":"hello"},{"id":"doc-2"}]"#).unwrap();
        assert_eq!(
            phoenix_collection_upsert(handle, docs.as_ptr(), std::ptr::null(), 0),
            0
        );

        let seen = collect(watcher, 2);
        assert_eq!(
            seen.iter().map(|w| w.key.clone()).collect::<Vec<_>>(),
            vec![b"doc-1".to_vec(), b"doc-2".to_vec()],
            "ids, not internal keys"
        );
        assert!(seen.iter().all(|w| w.kind == 1 && w.value.is_none()));

        phoenix_watch_close(watcher);
        phoenix_collection_close(handle);
    }
}

#[test]
fn bad_arguments_are_rejected_before_anything_is_touched() {
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().join("bad.pdb").to_str().unwrap()).unwrap();
    let mut db: *mut PhoenixDbHandle = std::ptr::null_mut();
    // SAFETY: deliberately hostile arguments; none may be dereferenced.
    unsafe {
        assert_eq!(phoenix_open(path.as_ptr(), 0, &mut db), 0);
        let mut watcher: *mut PhoenixWatcher = std::ptr::null_mut();

        // A null out-param, a zero capacity, an absurd capacity, a null
        // prefix with a non-zero length, and a null handle.
        assert_ne!(
            phoenix_watch_open(db, std::ptr::null(), 0, 64, 0, std::ptr::null_mut()),
            0
        );
        assert_ne!(
            phoenix_watch_open(db, std::ptr::null(), 0, 0, 0, &mut watcher),
            0
        );
        assert_ne!(
            phoenix_watch_open(db, std::ptr::null(), 0, u64::MAX, 0, &mut watcher),
            0
        );
        assert_ne!(
            phoenix_watch_open(db, std::ptr::null(), 8, 64, 0, &mut watcher),
            0
        );
        assert_ne!(
            phoenix_watch_open(
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                64,
                0,
                &mut watcher
            ),
            0
        );
        assert!(watcher.is_null(), "the out-param stays null on failure");

        // Polling and waking a null watcher fail rather than crash.
        let mut buffer = PhoenixBuffer {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        assert_ne!(
            phoenix_watch_poll(std::ptr::null_mut(), 0, &mut buffer, std::ptr::null_mut()),
            0
        );
        assert_ne!(phoenix_watch_wake(std::ptr::null_mut()), 0);
        phoenix_watch_close(std::ptr::null_mut()); // no-op

        assert_eq!(phoenix_close(db), 0);
    }
}

#[test]
fn closing_the_database_closes_the_watcher() {
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().join("closed.pdb").to_str().unwrap()).unwrap();
    let mut db: *mut PhoenixDbHandle = std::ptr::null_mut();
    // SAFETY: valid arguments throughout.
    unsafe {
        assert_eq!(phoenix_open(path.as_ptr(), 0, &mut db), 0);
        let mut watcher: *mut PhoenixWatcher = std::ptr::null_mut();
        assert_eq!(
            phoenix_watch_open(db, std::ptr::null(), 0, 64, 0, &mut watcher),
            0
        );
        assert_eq!(phoenix_close(db), 0);

        // The watcher outlives the handle safely: it reports closed and its
        // polls return at once instead of hanging.
        let mut closed: c_int = -1;
        assert_eq!(phoenix_watch_is_closed(watcher, &mut closed), 0);
        assert_eq!(closed, 1);
        let started = Instant::now();
        let mut buffer = PhoenixBuffer {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        assert_eq!(
            phoenix_watch_poll(watcher, 30_000, &mut buffer, std::ptr::null_mut()),
            0
        );
        assert!(take(&mut buffer).is_empty());
        assert!(started.elapsed() < Duration::from_secs(5));
        phoenix_watch_close(watcher);
    }
}
