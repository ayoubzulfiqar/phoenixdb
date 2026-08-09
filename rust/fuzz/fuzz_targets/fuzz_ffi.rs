//! Fuzzes the C ABI with hostile pointer/length combinations.
//!
//! The guardrails under test: every entry point must return a status code
//! (never crash, never dereference an unvalidated pointer) no matter what the
//! caller passes.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use phoenixdb::ffi::*;
use std::ffi::CString;
use std::ptr;

/// A fuzzer-chosen FFI call.
#[derive(Arbitrary, Debug)]
enum Call {
    Insert {
        key: Vec<u8>,
        value: Vec<u8>,
        null_key: bool,
        lie_about_len: bool,
    },
    Get {
        key: Vec<u8>,
        null_out: bool,
    },
    Delete {
        key: Vec<u8>,
        null_key: bool,
    },
    Begin,
    Commit {
        txn: u64,
    },
    Rollback {
        txn: u64,
    },
    Checkpoint,
    Verify,
    Count {
        null_out: bool,
    },
}

fuzz_target!(|calls: Vec<Call>| {
    if calls.len() > 128 {
        return;
    }
    let Ok(dir) = tempfile::tempdir() else {
        return;
    };
    let path = dir.path().join("fuzz.pdb");
    let Some(path_str) = path.to_str() else {
        return;
    };
    let Ok(c_path) = CString::new(path_str) else {
        return;
    };

    let mut handle: *mut PhoenixDbHandle = ptr::null_mut();
    if unsafe { phoenix_open(c_path.as_ptr(), 32, &mut handle) } != 0 {
        return;
    }
    let mut last_txn: u64 = 0;

    for call in calls {
        match call {
            Call::Insert {
                key,
                value,
                null_key,
                lie_about_len,
            } => {
                let key_ptr = if null_key { ptr::null() } else { key.as_ptr() };
                // Deliberately claim a length far larger than the allocation:
                // validation must reject it before any read occurs.
                let key_len = if lie_about_len { usize::MAX / 2 } else { key.len() };
                let status = unsafe {
                    phoenix_insert(
                        handle,
                        last_txn,
                        key_ptr,
                        key_len,
                        value.as_ptr(),
                        value.len(),
                    )
                };
                assert!(status <= 0, "status must be 0 or negative, got {status}");
            }
            Call::Get { key, null_out } => {
                let mut buf = PhoenixBuffer {
                    ptr: ptr::null_mut(),
                    len: 0,
                    cap: 0,
                };
                let out = if null_out { ptr::null_mut() } else { &mut buf };
                let status =
                    unsafe { phoenix_get(handle, last_txn, key.as_ptr(), key.len(), out) };
                assert!(status <= 0);
                if !null_out {
                    unsafe { phoenix_buffer_free(&mut buf) };
                    // Freeing twice must be harmless.
                    unsafe { phoenix_buffer_free(&mut buf) };
                }
            }
            Call::Delete { key, null_key } => {
                let key_ptr = if null_key { ptr::null() } else { key.as_ptr() };
                let status =
                    unsafe { phoenix_delete(handle, last_txn, key_ptr, key.len()) };
                assert!(status <= 0);
            }
            Call::Begin => {
                let mut txn: u64 = 0;
                if unsafe { phoenix_begin_txn(handle, 0, &mut txn) } == 0 {
                    last_txn = txn;
                }
            }
            Call::Commit { txn } => {
                let status = unsafe { phoenix_commit_txn(handle, txn) };
                assert!(status <= 0);
                if txn == last_txn {
                    last_txn = 0;
                }
            }
            Call::Rollback { txn } => {
                let status = unsafe { phoenix_rollback_txn(handle, txn) };
                assert!(status <= 0);
                if txn == last_txn {
                    last_txn = 0;
                }
            }
            Call::Checkpoint => {
                assert!(unsafe { phoenix_checkpoint(handle) } <= 0);
            }
            Call::Verify => {
                let status = unsafe { phoenix_verify(handle) };
                assert!(status <= 0, "verify reported corruption: {status}");
            }
            Call::Count { null_out } => {
                let mut count: u64 = 0;
                let out = if null_out { ptr::null_mut() } else { &mut count };
                assert!(unsafe { phoenix_count(handle, out) } <= 0);
            }
        }
        // Drain any recorded error so the thread-local does not grow unbounded.
        let msg = phoenix_last_error();
        if !msg.is_null() {
            unsafe { phoenix_string_free(msg) };
        }
    }
    unsafe { phoenix_close(handle) };
});
