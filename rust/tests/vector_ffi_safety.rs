//! Vector FFI boundary tests.
//!
//! These call the exported C ABI exactly as Dart does, with the same focus as
//! `ffi_safety.rs`: null pointers, wrong lengths, misaligned buffers, stale
//! handles and double frees must all be *rejected*, never dereferenced.

use phoenixdb::ffi::vector_ffi::*;
use std::ffi::{CStr, CString};
use std::ptr;

/// Status codes, mirrored from `PhoenixStatus`.
const OK: i32 = 0;
const INVALID: i32 = -2;
const NOT_FOUND: i32 = -3;

/// Metric codes, mirrored from `Metric`.
const COSINE: u8 = 0;
const EUCLIDEAN: u8 = 1;

struct Harness {
    handle: *mut PhoenixVectorHandle,
    _dir: tempfile::TempDir,
}

impl Harness {
    fn new(dim: usize, metric: u8) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ffi.pvec");
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let mut handle: *mut PhoenixVectorHandle = ptr::null_mut();
        let status = unsafe { phoenix_vector_init(c_path.as_ptr(), dim, metric, 0, &mut handle) };
        assert_eq!(status, OK, "phoenix_vector_init failed");
        assert!(!handle.is_null());
        Harness { handle, _dir: dir }
    }

    fn insert(&self, id: &str, vector: &[f32]) -> i32 {
        let c_id = CString::new(id).unwrap();
        unsafe { phoenix_vector_insert(self.handle, c_id.as_ptr(), vector.as_ptr(), vector.len()) }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { phoenix_vector_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// Runs a search and collects the results, always freeing the id array.
fn search(
    handle: *const PhoenixVectorHandle,
    query: &[f32],
    k: usize,
    ef: usize,
) -> (i32, Vec<(String, f32)>) {
    let mut ids: Vec<*mut std::os::raw::c_char> = vec![ptr::null_mut(); k];
    let mut scores: Vec<f32> = vec![0.0; k];
    let mut count: usize = 0;
    let status = unsafe {
        phoenix_vector_search(
            handle,
            query.as_ptr(),
            query.len(),
            k,
            ef,
            ids.as_mut_ptr(),
            scores.as_mut_ptr(),
            &mut count,
        )
    };
    let mut out = Vec::new();
    if status == OK {
        assert!(count <= k, "count must never exceed k");
        for index in 0..count {
            let text = unsafe { CStr::from_ptr(ids[index]) }
                .to_string_lossy()
                .into_owned();
            out.push((text, scores[index]));
        }
        unsafe { phoenix_free_string_array(ids.as_mut_ptr(), count) };
        assert!(
            ids[..count].iter().all(|p| p.is_null()),
            "free must null every slot so a double free is a no-op"
        );
    }
    (status, out)
}

#[test]
fn init_and_free_round_trip() {
    let h = Harness::new(8, COSINE);
    assert!(!h.handle.is_null());
}

#[test]
fn null_path_is_rejected() {
    let mut handle: *mut PhoenixVectorHandle = ptr::null_mut();
    assert_eq!(
        unsafe { phoenix_vector_init(ptr::null(), 8, COSINE, 0, &mut handle) },
        INVALID
    );
    assert!(handle.is_null(), "the handle must stay null on failure");
}

#[test]
fn null_out_handle_is_rejected() {
    let c_path = CString::new("unused.pvec").unwrap();
    assert_eq!(
        unsafe { phoenix_vector_init(c_path.as_ptr(), 8, COSINE, 0, ptr::null_mut()) },
        INVALID
    );
}

#[test]
fn invalid_geometry_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.pvec");
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut handle: *mut PhoenixVectorHandle = ptr::null_mut();
    unsafe {
        // Zero dimension, absurd dimension and an unknown metric code.
        assert_eq!(
            phoenix_vector_init(c_path.as_ptr(), 0, COSINE, 0, &mut handle),
            INVALID
        );
        assert_eq!(
            phoenix_vector_init(c_path.as_ptr(), usize::MAX, COSINE, 0, &mut handle),
            INVALID
        );
        assert_eq!(
            phoenix_vector_init(c_path.as_ptr(), 8, 99, 0, &mut handle),
            INVALID
        );
    }
    assert!(handle.is_null());
}

#[test]
fn operations_on_a_null_handle_are_rejected() {
    let v = [1.0f32; 4];
    let id = CString::new("x").unwrap();
    let mut count: usize = 0;
    let mut ids: [*mut std::os::raw::c_char; 1] = [ptr::null_mut()];
    let mut scores = [0.0f32; 1];
    unsafe {
        assert_eq!(
            phoenix_vector_insert(ptr::null_mut(), id.as_ptr(), v.as_ptr(), v.len()),
            INVALID
        );
        assert_eq!(
            phoenix_vector_search(
                ptr::null(),
                v.as_ptr(),
                v.len(),
                1,
                0,
                ids.as_mut_ptr(),
                scores.as_mut_ptr(),
                &mut count
            ),
            INVALID
        );
        assert_eq!(phoenix_vector_save(ptr::null_mut(), ptr::null()), INVALID);
        assert_eq!(phoenix_vector_flush(ptr::null_mut()), INVALID);
        assert_eq!(phoenix_vector_remove(ptr::null_mut(), id.as_ptr()), INVALID);
        assert_eq!(phoenix_vector_count(ptr::null(), &mut count), INVALID);
        assert_eq!(phoenix_vector_dim(ptr::null(), &mut count), INVALID);
        // Freeing null must be a silent no-op, not a crash.
        phoenix_vector_free(ptr::null_mut());
    }
}

#[test]
fn null_vector_pointer_is_rejected_before_deref() {
    let h = Harness::new(4, COSINE);
    let id = CString::new("k").unwrap();
    unsafe {
        // Non-zero length with a null pointer must never be dereferenced.
        assert_eq!(
            phoenix_vector_insert(h.handle, id.as_ptr(), ptr::null(), 4),
            INVALID
        );
    }
    let mut count: usize = 0;
    let mut ids: [*mut std::os::raw::c_char; 1] = [ptr::null_mut()];
    let mut scores = [0.0f32; 1];
    unsafe {
        assert_eq!(
            phoenix_vector_search(
                h.handle,
                ptr::null(),
                4,
                1,
                0,
                ids.as_mut_ptr(),
                scores.as_mut_ptr(),
                &mut count
            ),
            INVALID
        );
    }
}

#[test]
fn oversized_and_zero_lengths_are_rejected_before_any_read() {
    let h = Harness::new(4, COSINE);
    let id = CString::new("k").unwrap();
    let tiny = [1.0f32];
    let max_dim = phoenix_vector_max_dim();
    unsafe {
        // A one-element buffer with a huge declared length: the length must be
        // rejected before the pointer is read, or this is a wild read.
        assert_eq!(
            phoenix_vector_insert(h.handle, id.as_ptr(), tiny.as_ptr(), max_dim + 1),
            INVALID
        );
        assert_eq!(
            phoenix_vector_insert(h.handle, id.as_ptr(), tiny.as_ptr(), 0),
            INVALID
        );
        assert_eq!(
            phoenix_vector_insert(h.handle, id.as_ptr(), tiny.as_ptr(), usize::MAX),
            INVALID
        );
    }
}

#[test]
fn a_misaligned_vector_pointer_is_rejected() {
    let h = Harness::new(2, COSINE);
    let id = CString::new("k").unwrap();
    // Deliberately offset by one byte inside a larger buffer so the pointer is
    // valid memory but not 4-byte aligned.
    let backing = [0u8; 32];
    let misaligned = unsafe { backing.as_ptr().add(1) }.cast::<f32>();
    assert_ne!(misaligned as usize % 4, 0, "test setup must be misaligned");
    unsafe {
        assert_eq!(
            phoenix_vector_insert(h.handle, id.as_ptr(), misaligned, 2),
            INVALID
        );
    }
}

#[test]
fn dimension_mismatch_is_rejected() {
    let h = Harness::new(4, COSINE);
    assert_eq!(h.insert("short", &[1.0, 2.0]), INVALID);
    assert_eq!(h.insert("long", &[1.0; 8]), INVALID);
    assert_eq!(h.insert("right", &[1.0; 4]), OK);
}

#[test]
fn null_empty_and_oversized_ids_are_rejected() {
    let h = Harness::new(2, COSINE);
    let v = [1.0f32, 0.0];
    unsafe {
        assert_eq!(
            phoenix_vector_insert(h.handle, ptr::null(), v.as_ptr(), v.len()),
            INVALID
        );
    }
    let empty = CString::new("").unwrap();
    unsafe {
        assert_eq!(
            phoenix_vector_insert(h.handle, empty.as_ptr(), v.as_ptr(), v.len()),
            INVALID
        );
    }
    let oversized = CString::new("x".repeat(phoenix_vector_max_id_len() + 1)).unwrap();
    unsafe {
        assert_eq!(
            phoenix_vector_insert(h.handle, oversized.as_ptr(), v.as_ptr(), v.len()),
            INVALID
        );
    }
}

#[test]
fn insert_and_search_via_ffi() {
    let h = Harness::new(3, COSINE);
    assert_eq!(h.insert("a", &[1.0, 0.0, 0.0]), OK);
    assert_eq!(h.insert("b", &[0.0, 1.0, 0.0]), OK);
    assert_eq!(h.insert("c", &[0.0, 0.0, 1.0]), OK);

    let (status, results) = search(h.handle, &[1.0, 0.0, 0.0], 3, 0);
    assert_eq!(status, OK);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, "a");
    assert!(results[0].1.abs() < 1e-6, "exact match, distance 0");
    // Ascending order is part of the contract.
    for window in results.windows(2) {
        assert!(window[0].1 <= window[1].1);
    }
}

#[test]
fn search_writes_fewer_results_than_k_when_the_index_is_small() {
    let h = Harness::new(2, EUCLIDEAN);
    assert_eq!(h.insert("only", &[1.0, 1.0]), OK);
    let (status, results) = search(h.handle, &[0.0, 0.0], 10, 0);
    assert_eq!(status, OK);
    assert_eq!(results.len(), 1, "count must reflect what was written");
}

#[test]
fn invalid_k_and_null_outputs_are_rejected() {
    let h = Harness::new(2, COSINE);
    assert_eq!(h.insert("a", &[1.0, 0.0]), OK);
    let query = [1.0f32, 0.0];
    let mut ids: [*mut std::os::raw::c_char; 2] = [ptr::null_mut(); 2];
    let mut scores = [0.0f32; 2];
    let mut count: usize = 7;
    unsafe {
        // k = 0
        assert_eq!(
            phoenix_vector_search(
                h.handle,
                query.as_ptr(),
                2,
                0,
                0,
                ids.as_mut_ptr(),
                scores.as_mut_ptr(),
                &mut count
            ),
            INVALID
        );
        assert_eq!(count, 0, "count must be zeroed before any failure");

        // k beyond the documented ceiling
        assert_eq!(
            phoenix_vector_search(
                h.handle,
                query.as_ptr(),
                2,
                phoenix_vector_max_k() + 1,
                0,
                ids.as_mut_ptr(),
                scores.as_mut_ptr(),
                &mut count
            ),
            INVALID
        );

        // null output arrays
        assert_eq!(
            phoenix_vector_search(
                h.handle,
                query.as_ptr(),
                2,
                1,
                0,
                ptr::null_mut(),
                scores.as_mut_ptr(),
                &mut count
            ),
            INVALID
        );
        assert_eq!(
            phoenix_vector_search(
                h.handle,
                query.as_ptr(),
                2,
                1,
                0,
                ids.as_mut_ptr(),
                ptr::null_mut(),
                &mut count
            ),
            INVALID
        );
        assert_eq!(
            phoenix_vector_search(
                h.handle,
                query.as_ptr(),
                2,
                1,
                0,
                ids.as_mut_ptr(),
                scores.as_mut_ptr(),
                ptr::null_mut()
            ),
            INVALID
        );
    }
}

#[test]
fn get_remove_and_contains_via_ffi() {
    let h = Harness::new(3, EUCLIDEAN);
    assert_eq!(h.insert("keep", &[1.0, 2.0, 3.0]), OK);

    let id = CString::new("keep").unwrap();
    let mut out = [0.0f32; 3];
    unsafe {
        assert_eq!(
            phoenix_vector_get(h.handle, id.as_ptr(), out.as_mut_ptr(), 3),
            OK
        );
        assert_eq!(out, [1.0, 2.0, 3.0]);

        // A wrong output length must be refused rather than truncating.
        assert_eq!(
            phoenix_vector_get(h.handle, id.as_ptr(), out.as_mut_ptr(), 2),
            INVALID
        );
        assert_eq!(
            phoenix_vector_get(h.handle, id.as_ptr(), ptr::null_mut(), 3),
            INVALID
        );

        let mut present: i32 = -1;
        assert_eq!(
            phoenix_vector_contains(h.handle, id.as_ptr(), &mut present),
            OK
        );
        assert_eq!(present, 1);

        assert_eq!(phoenix_vector_remove(h.handle, id.as_ptr()), OK);
        assert_eq!(
            phoenix_vector_contains(h.handle, id.as_ptr(), &mut present),
            OK
        );
        assert_eq!(present, 0);
        // A second removal reports absence rather than corrupting state.
        assert_eq!(phoenix_vector_remove(h.handle, id.as_ptr()), NOT_FOUND);
        assert_eq!(
            phoenix_vector_get(h.handle, id.as_ptr(), out.as_mut_ptr(), 3),
            NOT_FOUND
        );
    }
}

#[test]
fn save_and_reopen_via_ffi() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("persist.pvec");
    let c_path = CString::new(path.to_str().unwrap()).unwrap();

    unsafe {
        let mut handle: *mut PhoenixVectorHandle = ptr::null_mut();
        assert_eq!(
            phoenix_vector_init(c_path.as_ptr(), 4, EUCLIDEAN, 128, &mut handle),
            OK
        );
        for i in 0..20u32 {
            let id = CString::new(format!("v{i}")).unwrap();
            let v = [i as f32, 1.0, 2.0, 3.0];
            assert_eq!(
                phoenix_vector_insert(handle, id.as_ptr(), v.as_ptr(), 4),
                OK
            );
        }
        assert_eq!(phoenix_vector_flush(handle), OK);
        assert_eq!(phoenix_vector_save(handle, ptr::null()), OK);
        phoenix_vector_free(handle);

        let mut reopened: *mut PhoenixVectorHandle = ptr::null_mut();
        assert_eq!(
            phoenix_vector_init(c_path.as_ptr(), 4, EUCLIDEAN, 0, &mut reopened),
            OK
        );
        let mut count: usize = 0;
        assert_eq!(phoenix_vector_count(reopened, &mut count), OK);
        assert_eq!(count, 20);

        let (status, results) = search(reopened, &[7.0, 1.0, 2.0, 3.0], 1, 0);
        assert_eq!(status, OK);
        assert_eq!(results[0].0, "v7");
        phoenix_vector_free(reopened);
    }
}

#[test]
fn compact_and_stats_via_ffi() {
    let h = Harness::new(2, EUCLIDEAN);
    for i in 0..10u32 {
        assert_eq!(h.insert(&format!("v{i}"), &[i as f32, 0.0]), OK);
    }
    for i in 0..4u32 {
        let id = CString::new(format!("v{i}")).unwrap();
        assert_eq!(unsafe { phoenix_vector_remove(h.handle, id.as_ptr()) }, OK);
    }

    let (mut live, mut total, mut deleted) = (0usize, 0usize, 0usize);
    unsafe {
        assert_eq!(
            phoenix_vector_stats(h.handle, &mut live, &mut total, &mut deleted),
            OK
        );
    }
    assert_eq!((live, total, deleted), (6, 10, 4));

    let mut reclaimed: usize = 0;
    unsafe {
        assert_eq!(phoenix_vector_compact(h.handle, &mut reclaimed), OK);
        assert_eq!(reclaimed, 4);
        // Null outputs are tolerated everywhere they are documented as such.
        assert_eq!(
            phoenix_vector_stats(h.handle, ptr::null_mut(), &mut total, ptr::null_mut()),
            OK
        );
    }
    assert_eq!(total, 6);
}

#[test]
fn use_after_free_is_rejected_not_undefined() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("uaf.pvec");
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut handle: *mut PhoenixVectorHandle = ptr::null_mut();
    unsafe {
        assert_eq!(
            phoenix_vector_init(c_path.as_ptr(), 4, COSINE, 0, &mut handle),
            OK
        );
        phoenix_vector_free(handle);
        // The allocation is gone, but the poisoned tag means a second free is
        // ignored rather than double-freeing, and a later call is rejected.
        phoenix_vector_free(handle);
        assert_eq!(phoenix_vector_flush(handle), INVALID);
    }
}

#[test]
fn free_string_array_is_null_safe_and_idempotent() {
    let mut empty: [*mut std::os::raw::c_char; 2] = [ptr::null_mut(); 2];
    unsafe {
        phoenix_free_string_array(ptr::null_mut(), 4);
        phoenix_free_string_array(empty.as_mut_ptr(), 0);
        phoenix_free_string_array(empty.as_mut_ptr(), 2);
        phoenix_free_string_array(empty.as_mut_ptr(), 2);
        // A length beyond anything this library produces must be refused
        // rather than walked.
        phoenix_free_string_array(empty.as_mut_ptr(), usize::MAX);
    }
}

#[test]
fn last_error_is_populated_and_freeable() {
    let h = Harness::new(4, COSINE);
    // Provoke a validation failure.
    assert_eq!(h.insert("bad", &[1.0, 2.0]), INVALID);
    unsafe {
        let msg = phoenix_vector_last_error();
        assert!(!msg.is_null(), "expected an error message");
        let text = CStr::from_ptr(msg).to_string_lossy().into_owned();
        assert!(text.contains("invalid"), "unexpected message: {text}");
        phoenixdb::ffi::phoenix_string_free(msg);
    }
}

#[test]
fn limits_and_capability_flags_are_reported() {
    assert_eq!(phoenix_vector_max_dim(), 65_536);
    assert_eq!(phoenix_vector_max_k(), 4096);
    assert_eq!(phoenix_vector_max_id_len(), 128);
    assert_eq!(phoenixdb::ffi::phoenix_has_vector(), 1);
    // The vector surface arrived in ABI v3 and every later ABI keeps it; the
    // exact current version is pinned by ffi_safety.rs.
    assert!(phoenixdb::ffi::phoenix_abi_version() >= 3);

    let kernel = unsafe { CStr::from_ptr(phoenix_vector_kernel()) }
        .to_string_lossy()
        .into_owned();
    assert!(
        ["avx2+fma", "neon", "portable"].contains(&kernel.as_str()),
        "unexpected kernel {kernel}"
    );
}

#[test]
fn replacing_an_id_through_the_ffi_does_not_duplicate_it() {
    let h = Harness::new(2, EUCLIDEAN);
    assert_eq!(h.insert("k", &[0.0, 0.0]), OK);
    assert_eq!(h.insert("k", &[9.0, 9.0]), OK);

    let mut count: usize = 0;
    unsafe { assert_eq!(phoenix_vector_count(h.handle, &mut count), OK) };
    assert_eq!(count, 1);

    let (status, results) = search(h.handle, &[9.0, 9.0], 5, 0);
    assert_eq!(status, OK);
    assert_eq!(results.len(), 1);
    assert!(results[0].1.abs() < 1e-5, "the replacement must be current");
}

// ---------------------------------------------------------------------------
// ABI v4 additions: shared engines, subset search, batch search
// ---------------------------------------------------------------------------

fn init_at(path: &std::path::Path, dim: usize, metric: u8) -> (i32, *mut PhoenixVectorHandle) {
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut handle: *mut PhoenixVectorHandle = ptr::null_mut();
    let status = unsafe { phoenix_vector_init(c_path.as_ptr(), dim, metric, 0, &mut handle) };
    (status, handle)
}

fn contains(h: *mut PhoenixVectorHandle, id: &CString) -> i32 {
    let mut out = -1i32;
    assert_eq!(
        unsafe { phoenix_vector_contains(h, id.as_ptr(), &mut out) },
        OK
    );
    out
}

#[test]
fn a_second_init_of_the_same_file_shares_the_engine() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.pvec");
    let (s1, a) = init_at(&path, 3, COSINE);
    let (s2, b) = init_at(&path, 3, COSINE);
    assert_eq!((s1, s2), (OK, OK));
    let id = CString::new("x").unwrap();
    let v = [1.0f32, 0.0, 0.0];
    assert_eq!(
        unsafe { phoenix_vector_insert(a, id.as_ptr(), v.as_ptr(), 3) },
        OK
    );
    assert_eq!(contains(b, &id), 1);

    // A mismatched geometry is refused rather than silently reinterpreted.
    let (s3, c) = init_at(&path, 4, COSINE);
    assert_eq!(s3, INVALID);
    assert!(c.is_null());
    let (s4, _) = init_at(&path, 3, EUCLIDEAN);
    assert_eq!(s4, INVALID);

    unsafe { phoenix_vector_free(a) };
    assert_eq!(contains(b, &id), 1, "b still works");
    unsafe { phoenix_vector_free(b) };
    let (s5, d) = init_at(&path, 3, COSINE);
    assert_eq!(s5, OK);
    assert_eq!(contains(d, &id), 1, "saved on last free");
    unsafe { phoenix_vector_free(d) };
}

#[test]
fn search_ids_restricts_results_to_the_subset() {
    let h = Harness::new(2, EUCLIDEAN);
    for i in 0..50 {
        assert_eq!(h.insert(&format!("p{i}"), &[i as f32, 0.0]), OK);
    }
    let names: Vec<CString> = ["p10", "p40", "p41", "missing"]
        .iter()
        .map(|s| CString::new(*s).unwrap())
        .collect();
    let ptrs: Vec<*const std::os::raw::c_char> = names.iter().map(|c| c.as_ptr()).collect();
    let mut ids = vec![ptr::null_mut(); 5];
    let mut scores = vec![0f32; 5];
    let mut count = 0usize;
    let q = [39.0f32, 0.0];
    let rc = unsafe {
        phoenix_vector_search_ids(
            h.handle,
            q.as_ptr(),
            2,
            5,
            0,
            ptrs.as_ptr(),
            ptrs.len(),
            ids.as_mut_ptr(),
            scores.as_mut_ptr(),
            &mut count,
        )
    };
    assert_eq!(rc, OK);
    assert_eq!(count, 3);
    let got: Vec<String> = ids[..count]
        .iter()
        .map(|p| unsafe { CStr::from_ptr(*p) }.to_string_lossy().into_owned())
        .collect();
    assert_eq!(got, ["p40", "p41", "p10"]);
    unsafe { phoenix_free_string_array(ids.as_mut_ptr(), count) };
}

#[test]
fn batch_search_writes_each_query_to_its_own_slot() {
    let h = Harness::new(2, EUCLIDEAN);
    for i in 0..20 {
        assert_eq!(h.insert(&format!("p{i}"), &[i as f32, 0.0]), OK);
    }
    let queries = [0.0f32, 0.0, 19.0, 0.0, 10.2, 0.0];
    let k = 2;
    let mut ids = vec![ptr::null_mut(); 3 * k];
    let mut scores = vec![0f32; 3 * k];
    let mut counts = vec![0usize; 3];
    let rc = unsafe {
        phoenix_vector_search_batch(
            h.handle,
            queries.as_ptr(),
            3,
            2,
            k,
            0,
            ids.as_mut_ptr(),
            scores.as_mut_ptr(),
            counts.as_mut_ptr(),
        )
    };
    assert_eq!(rc, OK);
    assert_eq!(counts, [2, 2, 2]);
    let first = |slot: usize| {
        unsafe { CStr::from_ptr(ids[slot * k]) }
            .to_string_lossy()
            .into_owned()
    };
    assert_eq!(first(0), "p0");
    assert_eq!(first(1), "p19");
    assert_eq!(first(2), "p10");
    for (slot, count) in counts.iter().enumerate() {
        unsafe { phoenix_free_string_array(ids.as_mut_ptr().add(slot * k), *count) };
    }
    // Wrong width is rejected.
    let rc = unsafe {
        phoenix_vector_search_batch(
            h.handle,
            queries.as_ptr(),
            2,
            3,
            k,
            0,
            ids.as_mut_ptr(),
            scores.as_mut_ptr(),
            counts.as_mut_ptr(),
        )
    };
    assert_eq!(rc, INVALID);
}

#[test]
fn insert_batch_validates_everything_before_writing() {
    let h = Harness::new(2, EUCLIDEAN);
    let names: Vec<CString> = (0..4)
        .map(|i| CString::new(format!("b{i}")).unwrap())
        .collect();
    let ptrs: Vec<*const std::os::raw::c_char> = names.iter().map(|c| c.as_ptr()).collect();
    let flat = [0.0f32, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
    assert_eq!(
        unsafe { phoenix_vector_insert_batch(h.handle, ptrs.as_ptr(), 4, flat.as_ptr(), 2) },
        OK
    );
    let mut count = 0usize;
    assert_eq!(unsafe { phoenix_vector_count(h.handle, &mut count) }, OK);
    assert_eq!(count, 4);

    // One non-finite component rejects the whole batch.
    let bad = [5.0f32, 0.0, f32::NAN, 0.0];
    let more: Vec<CString> = ["c0", "c1"]
        .iter()
        .map(|s| CString::new(*s).unwrap())
        .collect();
    let more_ptrs: Vec<*const std::os::raw::c_char> = more.iter().map(|c| c.as_ptr()).collect();
    assert_eq!(
        unsafe { phoenix_vector_insert_batch(h.handle, more_ptrs.as_ptr(), 2, bad.as_ptr(), 2) },
        INVALID
    );
    assert_eq!(unsafe { phoenix_vector_count(h.handle, &mut count) }, OK);
    assert_eq!(count, 4, "nothing from a rejected batch is written");
    assert_eq!(
        unsafe { phoenix_vector_insert_batch(h.handle, ptrs.as_ptr(), 4, flat.as_ptr(), 3) },
        INVALID,
        "wrong dimension"
    );
}
