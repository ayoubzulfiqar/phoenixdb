//! FFI boundary tests.
//!
//! These call the exported C ABI exactly as Dart does, with a strong focus on
//! the hostile inputs the guardrails exist for: null pointers, oversized
//! lengths, stale handles and double frees.

use phoenixdb::ffi::*;
use std::ffi::CString;
use std::ptr;

/// Status codes, mirrored from `PhoenixStatus`.
const OK: i32 = 0;
const INVALID: i32 = -2;
const NOT_FOUND: i32 = -3;

struct Harness {
    handle: *mut PhoenixDbHandle,
    _dir: tempfile::TempDir,
}

impl Harness {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ffi.pdb");
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let mut handle: *mut PhoenixDbHandle = ptr::null_mut();
        let status = unsafe { phoenix_open(c_path.as_ptr(), 64, &mut handle) };
        assert_eq!(status, OK, "phoenix_open failed");
        assert!(!handle.is_null());
        Harness { handle, _dir: dir }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { phoenix_close(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[test]
fn open_and_close_round_trip() {
    let h = Harness::new();
    assert!(!h.handle.is_null());
}

#[test]
fn null_path_is_rejected() {
    let mut handle: *mut PhoenixDbHandle = ptr::null_mut();
    let status = unsafe { phoenix_open(ptr::null(), 0, &mut handle) };
    assert_eq!(status, INVALID);
    assert!(handle.is_null());
}

#[test]
fn null_out_handle_is_rejected() {
    let c_path = CString::new("unused.pdb").unwrap();
    let status = unsafe { phoenix_open(c_path.as_ptr(), 0, ptr::null_mut()) };
    assert_eq!(status, INVALID);
}

#[test]
fn operations_on_null_handle_are_rejected() {
    let key = b"k";
    let mut txn: u64 = 0;
    unsafe {
        assert_eq!(phoenix_begin_txn(ptr::null_mut(), 0, &mut txn), INVALID);
        assert_eq!(phoenix_commit_txn(ptr::null_mut(), 1), INVALID);
        assert_eq!(phoenix_rollback_txn(ptr::null_mut(), 1), INVALID);
        assert_eq!(
            phoenix_insert(ptr::null_mut(), 1, key.as_ptr(), 1, key.as_ptr(), 1),
            INVALID
        );
        assert_eq!(phoenix_delete(ptr::null_mut(), 1, key.as_ptr(), 1), INVALID);
        assert_eq!(phoenix_checkpoint(ptr::null_mut()), INVALID);
        assert_eq!(phoenix_flush(ptr::null_mut()), INVALID);
        assert_eq!(phoenix_verify(ptr::null_mut()), INVALID);
    }
}

#[test]
fn null_key_pointer_is_rejected_before_deref() {
    let h = Harness::new();
    let value = b"v";
    unsafe {
        // Non-zero length with a null pointer must never be dereferenced.
        assert_eq!(
            phoenix_insert(h.handle, 0, ptr::null(), 8, value.as_ptr(), 1),
            INVALID
        );
        assert_eq!(phoenix_delete(h.handle, 0, ptr::null(), 8), INVALID);
    }
}

#[test]
fn empty_key_is_rejected() {
    let h = Harness::new();
    let key = b"k";
    let value = b"v";
    unsafe {
        assert_eq!(
            phoenix_insert(h.handle, 0, key.as_ptr(), 0, value.as_ptr(), 1),
            INVALID
        );
    }
}

#[test]
fn oversized_lengths_are_rejected() {
    let h = Harness::new();
    let key = b"k";
    let value = b"v";
    let max_key = phoenix_max_key_len();
    let max_value = phoenix_max_value_len();
    assert_eq!(max_key, 1024 * 1024);
    assert_eq!(max_value, 10 * 1024 * 1024);
    unsafe {
        // Lengths are validated before the pointer is read, so passing a tiny
        // buffer with a huge length must be safe.
        assert_eq!(
            phoenix_insert(h.handle, 0, key.as_ptr(), max_key + 1, value.as_ptr(), 1),
            INVALID
        );
        assert_eq!(
            phoenix_insert(h.handle, 0, key.as_ptr(), 1, value.as_ptr(), max_value + 1),
            INVALID
        );
    }
}

#[test]
fn insert_get_delete_via_ffi() {
    let h = Harness::new();
    let key = b"hello";
    let value = b"world";
    let mut buf = PhoenixBuffer {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    };
    unsafe {
        assert_eq!(
            phoenix_put_auto(
                h.handle,
                key.as_ptr(),
                key.len(),
                value.as_ptr(),
                value.len()
            ),
            OK
        );
        assert_eq!(
            phoenix_get(h.handle, 0, key.as_ptr(), key.len(), &mut buf),
            OK
        );
        assert_eq!(buf.len, value.len());
        let got = std::slice::from_raw_parts(buf.ptr, buf.len);
        assert_eq!(got, value);
        phoenix_buffer_free(&mut buf);
        assert!(buf.ptr.is_null(), "free must null the pointer");

        assert_eq!(phoenix_delete(h.handle, 0, key.as_ptr(), key.len()), OK);
        assert_eq!(
            phoenix_get(h.handle, 0, key.as_ptr(), key.len(), &mut buf),
            NOT_FOUND
        );
    }
}

#[test]
fn transaction_lifecycle_via_ffi() {
    let h = Harness::new();
    let key = b"txn-key";
    let value = b"txn-value";
    let mut txn: u64 = 0;
    unsafe {
        assert_eq!(phoenix_begin_txn(h.handle, 0, &mut txn), OK);
        assert!(txn > 0);
        assert_eq!(
            phoenix_insert(
                h.handle,
                txn,
                key.as_ptr(),
                key.len(),
                value.as_ptr(),
                value.len()
            ),
            OK
        );
        assert_eq!(phoenix_commit_txn(h.handle, txn), OK);

        let mut buf = PhoenixBuffer {
            ptr: ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        assert_eq!(
            phoenix_get(h.handle, 0, key.as_ptr(), key.len(), &mut buf),
            OK
        );
        assert_eq!(std::slice::from_raw_parts(buf.ptr, buf.len), value);
        phoenix_buffer_free(&mut buf);
    }
}

#[test]
fn rollback_via_ffi_discards_writes() {
    let h = Harness::new();
    let key = b"rb";
    let value = b"v";
    let mut txn: u64 = 0;
    let mut buf = PhoenixBuffer {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    };
    unsafe {
        assert_eq!(phoenix_begin_txn(h.handle, 0, &mut txn), OK);
        assert_eq!(
            phoenix_insert(
                h.handle,
                txn,
                key.as_ptr(),
                key.len(),
                value.as_ptr(),
                value.len()
            ),
            OK
        );
        assert_eq!(phoenix_rollback_txn(h.handle, txn), OK);
        assert_eq!(
            phoenix_get(h.handle, 0, key.as_ptr(), key.len(), &mut buf),
            NOT_FOUND
        );
    }
}

#[test]
fn use_after_close_is_rejected_not_undefined() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("uaf.pdb");
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut handle: *mut PhoenixDbHandle = ptr::null_mut();
    unsafe {
        assert_eq!(phoenix_open(c_path.as_ptr(), 0, &mut handle), OK);
        assert_eq!(phoenix_close(handle), OK);
        // The handle allocation is freed, but the poisoned tag means a second
        // close is reported as invalid instead of double-freeing.
        assert_eq!(phoenix_close(handle), INVALID);
    }
}

#[test]
fn buffer_free_is_null_safe_and_idempotent() {
    let mut buf = PhoenixBuffer {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    };
    unsafe {
        phoenix_buffer_free(ptr::null_mut());
        phoenix_buffer_free(&mut buf);
        phoenix_buffer_free(&mut buf);
    }
}

#[test]
fn count_and_maintenance_via_ffi() {
    let h = Harness::new();
    let mut count: u64 = 0;
    unsafe {
        for i in 0..10u32 {
            let key = format!("k{i}");
            let value = format!("v{i}");
            assert_eq!(
                phoenix_put_auto(
                    h.handle,
                    key.as_ptr(),
                    key.len(),
                    value.as_ptr(),
                    value.len()
                ),
                OK
            );
        }
        assert_eq!(phoenix_count(h.handle, &mut count), OK);
        assert_eq!(count, 10);
        assert_eq!(phoenix_checkpoint(h.handle), OK);
        assert_eq!(phoenix_flush(h.handle), OK);
        assert_eq!(phoenix_verify(h.handle), OK);
        assert_eq!(phoenix_count(h.handle, ptr::null_mut()), INVALID);
    }
}

#[test]
fn last_error_is_populated_and_freeable() {
    let h = Harness::new();
    unsafe {
        // Provoke a validation failure.
        assert_eq!(phoenix_delete(h.handle, 0, ptr::null(), 4), INVALID);
        let msg = phoenix_last_error();
        assert!(!msg.is_null(), "expected an error message");
        let text = std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned();
        assert!(text.contains("invalid"), "unexpected message: {text}");
        phoenix_string_free(msg);
        phoenix_string_free(ptr::null_mut()); // null-safe
    }
}

#[test]
fn abi_version_matches_expectation() {
    // Bumped 1 -> 2 in PhoenixDB 2.0, which adds `phoenix_sql_query` and
    // `phoenix_has_sql`. Must stay in lockstep with `kExpectedAbiVersion` in
    // lib/src/bindings.dart, or Dart refuses to load the library.
    assert_eq!(phoenix_abi_version(), 2);
}

#[test]
fn the_sql_capability_flag_matches_the_build() {
    // Dart branches on this to degrade gracefully on a lean embedded build,
    // so it must reflect the actual compiled feature set rather than a
    // hardcoded answer.
    let expected = i32::from(cfg!(feature = "sql"));
    assert_eq!(phoenix_has_sql(), expected);
}
