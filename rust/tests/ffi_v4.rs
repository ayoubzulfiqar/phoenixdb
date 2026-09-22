//! Tests for the ABI v4 surface: shared engines, range scans, write batches,
//! backup/restore, stats, check, metrics and tracing — plus the fixes to
//! `phoenix_scan_iter` and `phoenix_metrics_report`.

use phoenixdb::PhoenixStatus;
use phoenixdb::ffi::*;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

const OK: c_int = PhoenixStatus::Ok as c_int;

fn open_at(path: &std::path::Path) -> *mut PhoenixDbHandle {
    let c = CString::new(path.to_str().unwrap()).unwrap();
    let mut h = ptr::null_mut();
    assert_eq!(unsafe { phoenix_open(c.as_ptr(), 0, &mut h) }, OK);
    assert!(!h.is_null());
    h
}

fn put(h: *mut PhoenixDbHandle, k: &[u8], v: &[u8]) {
    let rc = unsafe { phoenix_put_auto(h, k.as_ptr(), k.len(), v.as_ptr(), v.len()) };
    assert_eq!(rc, OK);
}

fn get(h: *mut PhoenixDbHandle, k: &[u8]) -> Option<Vec<u8>> {
    let mut buf = PhoenixBuffer {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    };
    let rc = unsafe { phoenix_get(h, 0, k.as_ptr(), k.len(), &mut buf) };
    if rc == PhoenixStatus::NotFound as c_int {
        return None;
    }
    assert_eq!(rc, OK);
    let out = if buf.ptr.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(buf.ptr, buf.len) }.to_vec()
    };
    unsafe { phoenix_buffer_free(&mut buf) };
    Some(out)
}

/// Decodes a `[u32 klen][key][u32 vlen][value]...` buffer and frees it.
fn take_pairs(mut buf: PhoenixBuffer) -> Vec<(Vec<u8>, Vec<u8>)> {
    let bytes = if buf.ptr.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(buf.ptr, buf.len) }.to_vec()
    };
    unsafe { phoenix_buffer_free(&mut buf) };
    let mut out = Vec::new();
    let mut at = 0usize;
    let read = |at: &mut usize| {
        let n = u32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap()) as usize;
        *at += 4;
        let v = bytes[*at..*at + n].to_vec();
        *at += n;
        v
    };
    while at < bytes.len() {
        let k = read(&mut at);
        let v = read(&mut at);
        out.push((k, v));
    }
    out
}

fn empty_buf() -> PhoenixBuffer {
    PhoenixBuffer {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    }
}

fn last_error() -> String {
    let p = phoenix_last_error();
    if p.is_null() {
        return String::new();
    }
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { phoenix_string_free(p) };
    s
}

fn take_string(p: *mut c_char) -> String {
    assert!(!p.is_null());
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { phoenix_string_free(p) };
    s
}

#[test]
fn a_second_open_in_the_same_process_shares_the_engine() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.pdb");
    let a = open_at(&path);
    let b = open_at(&path); // used to corrupt the file; now joins the engine
    put(a, b"from-a", b"1");
    put(b, b"from-b", b"2");
    assert_eq!(get(b, b"from-a").as_deref(), Some(&b"1"[..]));
    assert_eq!(get(a, b"from-b").as_deref(), Some(&b"2"[..]));

    // Closing one handle leaves the other fully usable.
    assert_eq!(unsafe { phoenix_close(a) }, OK);
    put(b, b"after", b"3");
    let mut n = 0u64;
    assert_eq!(unsafe { phoenix_count(b, &mut n) }, OK);
    assert_eq!(n, 3);
    assert_eq!(unsafe { phoenix_verify(b) }, OK);
    assert_eq!(unsafe { phoenix_close(b) }, OK);

    // The last close released the engine: a fresh open sees everything.
    let c = open_at(&path);
    assert_eq!(get(c, b"after").as_deref(), Some(&b"3"[..]));
    assert_eq!(unsafe { phoenix_close(c) }, OK);
}

#[test]
fn open_ex_validates_and_applies_options() {
    let dir = tempfile::tempdir().unwrap();
    let path = CString::new(dir.path().join("o.pdb").to_str().unwrap()).unwrap();
    let mut h = ptr::null_mut();

    let mut bad = PhoenixOptions {
        struct_size: 4, // too small: a caller compiled against another layout
        sync_on_commit: 1,
        cache_pages: 0,
        checkpoint_bytes: 0,
        tracing: 0,
        fill_factor_max: 0.0,
    };
    let rc = unsafe { phoenix_open_ex(path.as_ptr(), &bad, &mut h) };
    assert_eq!(rc, PhoenixStatus::InvalidArgument as c_int);
    assert!(h.is_null());

    bad.struct_size = std::mem::size_of::<PhoenixOptions>() as u32;
    bad.fill_factor_max = 0.2;
    let rc = unsafe { phoenix_open_ex(path.as_ptr(), &bad, &mut h) };
    assert_eq!(rc, PhoenixStatus::InvalidArgument as c_int);

    let good = PhoenixOptions {
        struct_size: std::mem::size_of::<PhoenixOptions>() as u32,
        sync_on_commit: 0,
        cache_pages: 64,
        checkpoint_bytes: 1 << 20,
        tracing: 1,
        fill_factor_max: 0.9,
    };
    assert_eq!(unsafe { phoenix_open_ex(path.as_ptr(), &good, &mut h) }, OK);
    put(h, b"k", b"v");
    let mut json = ptr::null_mut();
    assert_eq!(unsafe { phoenix_spans_json(h, &mut json) }, OK);
    let spans = take_string(json);
    assert!(
        spans.contains("\"name\":\"commit\""),
        "tracing was requested: {spans}"
    );
    assert_eq!(unsafe { phoenix_close(h) }, OK);

    // Null options mean defaults.
    assert_eq!(
        unsafe { phoenix_open_ex(path.as_ptr(), ptr::null(), &mut h) },
        OK
    );
    assert_eq!(unsafe { phoenix_close(h) }, OK);
}

#[test]
fn scan_range_and_prefix_return_decodable_pages() {
    let dir = tempfile::tempdir().unwrap();
    let h = open_at(&dir.path().join("s.pdb"));
    for i in 0..50u32 {
        put(
            h,
            format!("user:{i:02}").as_bytes(),
            format!("{i}").as_bytes(),
        );
    }
    put(h, b"zzz", b"last");
    assert_eq!(unsafe { phoenix_checkpoint(h) }, OK);
    put(h, b"user:07", b"updated"); // in memory, over the tree

    // [user:05, user:10)
    let lo = b"user:05";
    let hi = b"user:10";
    let mut buf = empty_buf();
    let rc = unsafe {
        phoenix_scan_range(
            h,
            0,
            lo.as_ptr(),
            lo.len(),
            1,
            hi.as_ptr(),
            hi.len(),
            2,
            0,
            0,
            &mut buf,
        )
    };
    assert_eq!(rc, OK);
    let pairs = take_pairs(buf);
    let keys: Vec<String> = pairs
        .iter()
        .map(|(k, _)| String::from_utf8(k.clone()).unwrap())
        .collect();
    assert_eq!(
        keys,
        ["user:05", "user:06", "user:07", "user:08", "user:09"]
    );
    assert_eq!(pairs[2].1, b"updated");

    // Paging with a limit and an exclusive lower bound.
    let mut all = Vec::new();
    let mut cursor: Option<Vec<u8>> = None;
    loop {
        let mut buf = empty_buf();
        let (lp, ll, lm) = match &cursor {
            Some(c) => (c.as_ptr(), c.len(), 2),
            None => (ptr::null(), 0, 0),
        };
        let rc = unsafe { phoenix_scan_range(h, 0, lp, ll, lm, ptr::null(), 0, 0, 7, 0, &mut buf) };
        assert_eq!(rc, OK);
        let page = take_pairs(buf);
        if page.is_empty() {
            break;
        }
        assert!(page.len() <= 7);
        cursor = Some(page.last().unwrap().0.clone());
        all.extend(page);
    }
    assert_eq!(all.len(), 51);
    assert!(all.windows(2).all(|w| w[0].0 < w[1].0));

    // Prefix.
    let p = b"user:4";
    let mut buf = empty_buf();
    assert_eq!(
        unsafe { phoenix_scan_prefix(h, 0, p.as_ptr(), p.len(), 0, 0, &mut buf) },
        OK
    );
    assert_eq!(take_pairs(buf).len(), 10);

    // A bad bound mode is rejected.
    let mut buf = empty_buf();
    let rc =
        unsafe { phoenix_scan_range(h, 0, ptr::null(), 0, 9, ptr::null(), 0, 0, 0, 0, &mut buf) };
    assert_eq!(rc, PhoenixStatus::InvalidArgument as c_int);
    assert!(buf.ptr.is_null());
    assert_eq!(unsafe { phoenix_close(h) }, OK);
}

#[test]
fn transactional_range_scan_sees_own_writes() {
    let dir = tempfile::tempdir().unwrap();
    let h = open_at(&dir.path().join("t.pdb"));
    put(h, b"a", b"1");
    let mut txn = 0u64;
    assert_eq!(unsafe { phoenix_begin_txn(h, 0, &mut txn) }, OK);
    let (k, v) = (b"b", b"mine");
    assert_eq!(
        unsafe { phoenix_insert(h, txn, k.as_ptr(), 1, v.as_ptr(), 4) },
        OK
    );
    let mut buf = empty_buf();
    assert_eq!(
        unsafe { phoenix_scan_range(h, txn, ptr::null(), 0, 0, ptr::null(), 0, 0, 0, 0, &mut buf) },
        OK
    );
    assert_eq!(take_pairs(buf).len(), 2);
    let mut buf = empty_buf();
    assert_eq!(
        unsafe { phoenix_scan_range(h, 0, ptr::null(), 0, 0, ptr::null(), 0, 0, 0, 0, &mut buf) },
        OK
    );
    assert_eq!(
        take_pairs(buf).len(),
        1,
        "others do not see uncommitted writes"
    );
    assert_eq!(unsafe { phoenix_rollback_txn(h, txn) }, OK);
    assert_eq!(unsafe { phoenix_close(h) }, OK);
}

fn batch_put(out: &mut Vec<u8>, k: &[u8], v: &[u8]) {
    out.push(1);
    out.extend_from_slice(&(k.len() as u32).to_le_bytes());
    out.extend_from_slice(k);
    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
    out.extend_from_slice(v);
}

fn batch_delete(out: &mut Vec<u8>, k: &[u8], if_exists: bool) {
    out.push(if if_exists { 3 } else { 2 });
    out.extend_from_slice(&(k.len() as u32).to_le_bytes());
    out.extend_from_slice(k);
}

#[test]
fn write_batch_is_atomic_and_validated_up_front() {
    let dir = tempfile::tempdir().unwrap();
    let h = open_at(&dir.path().join("b.pdb"));
    put(h, b"old", b"x");

    let mut ops = Vec::new();
    batch_put(&mut ops, b"a", b"1");
    batch_put(&mut ops, b"b", b"2");
    batch_delete(&mut ops, b"old", false);
    batch_delete(&mut ops, b"never-there", true);
    assert_eq!(
        unsafe { phoenix_write_batch(h, ops.as_ptr(), ops.len()) },
        OK
    );
    assert_eq!(get(h, b"a").as_deref(), Some(&b"1"[..]));
    assert_eq!(get(h, b"old"), None);

    // A strict delete of a missing key aborts the whole batch.
    let mut ops = Vec::new();
    batch_put(&mut ops, b"c", b"3");
    batch_delete(&mut ops, b"missing", false);
    let rc = unsafe { phoenix_write_batch(h, ops.as_ptr(), ops.len()) };
    assert_eq!(rc, PhoenixStatus::NotFound as c_int);
    assert_eq!(get(h, b"c"), None, "nothing from a failed batch is visible");

    // Malformed input is rejected before anything is staged.
    let mut ops = Vec::new();
    batch_put(&mut ops, b"d", b"4");
    ops.extend_from_slice(&[1, 200, 0, 0, 0, b'x']); // key length runs past the end
    let rc = unsafe { phoenix_write_batch(h, ops.as_ptr(), ops.len()) };
    assert_eq!(rc, PhoenixStatus::InvalidArgument as c_int);
    assert!(last_error().contains("truncated"));
    assert_eq!(get(h, b"d"), None);

    let bad_op = [9u8, 1, 0, 0, 0, b'k'];
    let rc = unsafe { phoenix_write_batch(h, bad_op.as_ptr(), bad_op.len()) };
    assert_eq!(rc, PhoenixStatus::InvalidArgument as c_int);

    let mut stats = PhoenixStats::default();
    assert_eq!(unsafe { phoenix_stats(h, &mut stats) }, OK);
    assert_eq!(
        stats.active_txns, 0,
        "failed batches do not leak transactions"
    );
    assert_eq!(
        unsafe { phoenix_write_batch(h, ptr::null(), 0) },
        OK,
        "empty batch"
    );
    assert_eq!(unsafe { phoenix_close(h) }, OK);
}

#[test]
fn backup_restore_compact_over_the_abi() {
    let dir = tempfile::tempdir().unwrap();
    let h = open_at(&dir.path().join("live.pdb"));
    for i in 0..300u32 {
        put(h, format!("k{i:03}").as_bytes(), &[7u8; 500]);
    }
    let backup = CString::new(dir.path().join("snap.pdb").to_str().unwrap()).unwrap();
    assert_eq!(unsafe { phoenix_backup(h, backup.as_ptr()) }, OK);
    for i in 0..300u32 {
        let k = format!("k{i:03}");
        assert_eq!(unsafe { phoenix_delete(h, 0, k.as_ptr(), k.len()) }, OK);
    }
    assert_eq!(unsafe { phoenix_compact(h) }, OK);
    let mut n = 0u64;
    assert_eq!(unsafe { phoenix_count(h, &mut n) }, OK);
    assert_eq!(n, 0);
    assert_eq!(unsafe { phoenix_restore(h, backup.as_ptr()) }, OK);
    assert_eq!(unsafe { phoenix_count(h, &mut n) }, OK);
    assert_eq!(n, 300);

    let mut report = PhoenixTreeReport::default();
    assert_eq!(unsafe { phoenix_check(h, &mut report) }, OK);
    assert_eq!(report.keys, 300);
    assert!(report.depth >= 2);
    assert_eq!(
        unsafe { phoenix_backup(h, ptr::null()) },
        PhoenixStatus::InvalidArgument as c_int
    );
    assert_eq!(unsafe { phoenix_close(h) }, OK);
}

#[test]
fn metrics_text_and_report_are_well_behaved() {
    let dir = tempfile::tempdir().unwrap();
    let h = open_at(&dir.path().join("m.pdb"));
    put(h, b"k", b"v");
    let _ = get(h, b"k");

    let mut s = ptr::null_mut();
    assert_eq!(unsafe { phoenix_metrics_text(h, 0, &mut s) }, OK);
    let report = take_string(s);
    assert!(report.contains("commits=1"), "{report}");
    assert_eq!(unsafe { phoenix_metrics_text(h, 1, &mut s) }, OK);
    assert!(take_string(s).contains("# TYPE phoenixdb_reads_total counter"));
    assert_eq!(
        unsafe { phoenix_metrics_text(h, 7, &mut s) },
        PhoenixStatus::InvalidArgument as c_int
    );

    // Fixed-buffer variant: NUL-terminated, and "too small" is negative.
    let mut buf = vec![0xAAu8; 8192];
    let n = unsafe { phoenix_metrics_report(h, buf.as_mut_ptr(), buf.len()) };
    assert!(n > 0);
    assert_eq!(buf[n as usize], 0, "report must be NUL-terminated");
    let mut tiny = [0u8; 8];
    let rc = unsafe { phoenix_metrics_report(h, tiny.as_mut_ptr(), tiny.len()) };
    assert_eq!(
        rc,
        PhoenixStatus::Full as isize,
        "too small must be a negative status"
    );
    assert_eq!(unsafe { phoenix_close(h) }, OK);
}

unsafe extern "C" fn stop_after_three(_: *const u8, _: usize, _: *const u8, _: usize) -> c_int {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEEN: AtomicUsize = AtomicUsize::new(0);
    if SEEN.fetch_add(1, Ordering::SeqCst) >= 2 {
        1
    } else {
        0
    }
}

#[test]
fn scan_iter_rejects_a_null_callback_and_reports_aborts() {
    let dir = tempfile::tempdir().unwrap();
    let h = open_at(&dir.path().join("c.pdb"));
    for i in 0..10u32 {
        put(h, format!("k{i}").as_bytes(), b"v");
    }
    assert_eq!(
        unsafe { phoenix_scan_iter(h, None) },
        PhoenixStatus::InvalidArgument as c_int
    );
    assert_eq!(
        unsafe { phoenix_scan_iter(h, Some(stop_after_three)) },
        PhoenixStatus::Aborted as c_int
    );
    assert_eq!(unsafe { phoenix_close(h) }, OK);
}

#[test]
fn tracing_can_be_toggled_and_spans_are_valid_json() {
    let dir = tempfile::tempdir().unwrap();
    let h = open_at(&dir.path().join("tr.pdb"));
    let mut s = ptr::null_mut();
    assert_eq!(unsafe { phoenix_spans_json(h, &mut s) }, OK);
    assert_eq!(take_string(s), "[]");
    assert_eq!(unsafe { phoenix_set_tracing(h, 1) }, OK);
    put(h, b"k\"q", b"v");
    assert_eq!(unsafe { phoenix_checkpoint(h) }, OK);
    assert_eq!(unsafe { phoenix_spans_json(h, &mut s) }, OK);
    let json = take_string(s);
    assert!(json.starts_with('[') && json.ends_with(']'));
    assert!(json.contains("\"name\":\"checkpoint\""));
    assert!(json.contains("\"attributes\":{\"txn_id\":"));
    assert_eq!(unsafe { phoenix_close(h) }, OK);
}
