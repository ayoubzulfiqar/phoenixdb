//! C ABI surface consumed by `dart:ffi`.
//!
//! # Boundary contract
//!
//! * Every function returns an `int32_t` status from [`PhoenixStatus`]; `0` is
//!   success and every failure is negative. Validation failures are `-2`.
//! * **No function dereferences a caller pointer before validating it.** Null
//!   checks, length limits and handle-tag verification all happen first.
//! * No Rust panic may unwind into Dart: every body is wrapped in
//!   [`std::panic::catch_unwind`] and a caught panic becomes `-7`.
//! * Memory allocated by this library is owned by this library. The caller must
//!   release it with `phoenix_buffer_free` (values) or `phoenix_string_free`
//!   (error strings) — never with the host `free`.
//!
//! The vector-search surface (`phoenix_vector_*`) follows exactly the same
//! contract and lives in [`vector_ffi`]; it is re-exported here so the whole C
//! ABI is reachable from one module.

pub mod vector_ffi;

pub use vector_ffi::{
    MAX_SEARCH_K, PhoenixVectorHandle, phoenix_free_string_array, phoenix_vector_compact,
    phoenix_vector_contains, phoenix_vector_count, phoenix_vector_dim, phoenix_vector_flush,
    phoenix_vector_free, phoenix_vector_get, phoenix_vector_init, phoenix_vector_insert,
    phoenix_vector_kernel, phoenix_vector_last_error, phoenix_vector_max_dim,
    phoenix_vector_max_id_len, phoenix_vector_max_k, phoenix_vector_remove, phoenix_vector_save,
    phoenix_vector_search, phoenix_vector_stats,
};

use crate::error::{Error, PhoenixStatus};
use crate::security::{self, HandleTag, MAX_KEY_LEN, MAX_VALUE_LEN, ct_eq_u64, slice_from_parts};
use crate::{Database, Options};
use std::os::raw::{c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

/// Opaque database handle handed to C.
///
/// The `tag` is the first field so a stale or foreign pointer is caught by the
/// constant-time check in [`DbHandle::validate`] before `db` is touched.
#[repr(C)]
pub struct PhoenixDbHandle {
    tag: HandleTag,
    db: *mut Database,
}

impl PhoenixDbHandle {
    /// Validates a raw handle pointer and borrows the database.
    ///
    /// # Safety
    /// `handle` must be a pointer previously returned by `phoenix_open` and not
    /// yet passed to `phoenix_close`.
    unsafe fn validate<'a>(handle: *mut PhoenixDbHandle) -> Result<&'a Database, Error> {
        if handle.is_null() {
            return Err(Error::invalid("null database handle"));
        }
        // SAFETY: non-null; the tag is read first and rejects freed memory with
        // overwhelming probability before `db` is dereferenced.
        let h = unsafe { &*handle };
        if !h.tag.is_valid() {
            return Err(Error::invalid("invalid or already-closed database handle"));
        }
        if h.db.is_null() {
            return Err(Error::Closed);
        }
        // SAFETY: `db` was created by `Box::into_raw` in `phoenix_open` and is
        // only freed in `phoenix_close`, which poisons the tag first.
        Ok(unsafe { &*h.db })
    }
}

/// An owned byte buffer returned to the caller.
///
/// Release it with [`phoenix_buffer_free`]. `ptr` is null when `len` is zero.
#[repr(C)]
pub struct PhoenixBuffer {
    /// Pointer to `len` bytes owned by PhoenixDB.
    pub ptr: *mut u8,
    /// Number of valid bytes.
    pub len: usize,
    /// Allocated capacity; required to reconstruct the `Vec` on free.
    pub cap: usize,
}

impl PhoenixBuffer {
    /// An empty buffer.
    fn empty() -> Self {
        PhoenixBuffer {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }
    }

    /// Transfers ownership of `v` to the caller.
    fn from_vec(mut v: Vec<u8>) -> Self {
        if v.is_empty() {
            return PhoenixBuffer::empty();
        }
        v.shrink_to_fit();
        let ptr = v.as_mut_ptr();
        let len = v.len();
        let cap = v.capacity();
        std::mem::forget(v); // ownership moves to C
        PhoenixBuffer { ptr, len, cap }
    }
}

/// Runs `f`, converting panics and errors into a stable status code.
fn guard(f: impl FnOnce() -> Result<(), Error>) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => PhoenixStatus::Ok as c_int,
        Ok(Err(e)) => {
            set_last_error(&e);
            e.status() as c_int
        }
        Err(_) => {
            set_last_error(&Error::corrupt("panic caught at the FFI boundary"));
            PhoenixStatus::Panic as c_int
        }
    }
}

thread_local! {
    /// Human-readable description of the most recent failure on this thread.
    static LAST_ERROR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

fn set_last_error(e: &Error) {
    let msg = e.to_string();
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(msg);
    });
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Opens (or creates) a database.
///
/// `path` must be a NUL-terminated UTF-8 string. On success `*out_handle`
/// receives a handle that must be released with [`phoenix_close`].
///
/// # Safety
/// `path` must point to a valid NUL-terminated string and `out_handle` to a
/// writable pointer-sized location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_open(
    path: *const c_char,
    cache_pages: usize,
    out_handle: *mut *mut PhoenixDbHandle,
) -> c_int {
    guard(|| {
        if out_handle.is_null() {
            return Err(Error::invalid("out_handle is null"));
        }
        // SAFETY: checked non-null immediately above.
        unsafe { *out_handle = std::ptr::null_mut() };
        if path.is_null() {
            return Err(Error::invalid("path is null"));
        }
        // SAFETY: caller guarantees a NUL-terminated string.
        let c_str = unsafe { std::ffi::CStr::from_ptr(path) };
        let bytes = c_str.to_bytes();
        if bytes.is_empty() {
            return Err(Error::invalid("path is empty"));
        }
        if bytes.len() > 4096 {
            return Err(Error::invalid("path exceeds 4096 bytes"));
        }
        let path_str = c_str
            .to_str()
            .map_err(|_| Error::invalid("path is not valid UTF-8"))?;

        let mut options = Options::default();
        if cache_pages > 0 {
            options.cache_pages = cache_pages.min(1 << 20);
        }
        let db = Database::open(PathBuf::from(path_str), options)?;
        let handle = Box::new(PhoenixDbHandle {
            tag: HandleTag::new(),
            db: Box::into_raw(Box::new(db)),
        });
        // SAFETY: `out_handle` was validated as non-null above.
        unsafe { *out_handle = Box::into_raw(handle) };
        Ok(())
    })
}

/// Checkpoints and closes a database, freeing the handle.
///
/// Passing the same handle twice is detected by the poisoned tag and reported
/// as `-2` rather than causing a double free.
///
/// # Safety
/// `handle` must come from [`phoenix_open`] and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_close(handle: *mut PhoenixDbHandle) -> c_int {
    guard(|| {
        if handle.is_null() {
            return Err(Error::invalid("null database handle"));
        }
        // SAFETY: non-null; tag verified before any other field is read.
        let h = unsafe { &mut *handle };
        if !h.tag.is_valid() {
            return Err(Error::invalid("invalid or already-closed database handle"));
        }
        h.tag.poison(); // reject any concurrent/subsequent use
        let db_ptr = std::mem::replace(&mut h.db, std::ptr::null_mut());
        if !db_ptr.is_null() {
            // SAFETY: created by `Box::into_raw` in `phoenix_open`; freed once
            // because the tag is poisoned before this point.
            let db = unsafe { Box::from_raw(db_ptr) };
            let _ = db.close();
            drop(db);
        }
        // SAFETY: the handle box itself is freed exactly once, here.
        drop(unsafe { Box::from_raw(handle) });
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Transactions
// ---------------------------------------------------------------------------

/// Begins a transaction. `read_only != 0` requests a read-only snapshot.
///
/// # Safety
/// `handle` must be live and `out_txn` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_begin_txn(
    handle: *mut PhoenixDbHandle,
    read_only: c_int,
    out_txn: *mut u64,
) -> c_int {
    guard(|| {
        if out_txn.is_null() {
            return Err(Error::invalid("out_txn is null"));
        }
        // SAFETY: validated non-null above.
        unsafe { *out_txn = 0 };
        // SAFETY: handle validity is the caller's documented obligation.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        let id = db.begin(read_only != 0)?;
        // SAFETY: validated non-null above.
        unsafe { *out_txn = id };
        Ok(())
    })
}

/// Commits a transaction, making its writes durable before returning.
///
/// # Safety
/// `handle` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_commit_txn(handle: *mut PhoenixDbHandle, txn_id: u64) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        if ct_eq_u64(txn_id, 0) {
            return Err(Error::invalid("transaction id 0 is never valid"));
        }
        db.commit(txn_id)
    })
}

/// Rolls a transaction back, discarding its writes.
///
/// # Safety
/// `handle` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_rollback_txn(handle: *mut PhoenixDbHandle, txn_id: u64) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        if ct_eq_u64(txn_id, 0) {
            return Err(Error::invalid("transaction id 0 is never valid"));
        }
        db.rollback(txn_id)
    })
}

// ---------------------------------------------------------------------------
// Data plane
// ---------------------------------------------------------------------------

/// Inserts or replaces a key within `txn_id`.
///
/// Rejects a null pointer, an empty key, a key over 1 MiB or a value over
/// 10 MiB with `-2` *before* dereferencing anything.
///
/// # Safety
/// `key`/`value` must each point to at least the stated number of readable
/// bytes for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_insert(
    handle: *mut PhoenixDbHandle,
    txn_id: u64,
    key: *const u8,
    key_len: usize,
    value: *const u8,
    value_len: usize,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        security::validate_key_len(key_len)?;
        security::validate_value_len(value_len)?;
        // SAFETY: lengths are bounded above; `slice_from_parts` rejects null.
        let k = unsafe { slice_from_parts(key, key_len, MAX_KEY_LEN) }?;
        // SAFETY: as above for the value buffer.
        let v = unsafe { slice_from_parts(value, value_len, MAX_VALUE_LEN) }?;
        db.insert(txn_id, k, v)
    })
}

/// Reads a key within `txn_id` into a freshly allocated buffer.
///
/// On success `*out` owns the value and must be released with
/// [`phoenix_buffer_free`]. Returns `-3` when the key is not visible.
///
/// # Safety
/// `key` must be readable for `key_len` bytes; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_get(
    handle: *mut PhoenixDbHandle,
    txn_id: u64,
    key: *const u8,
    key_len: usize,
    out: *mut PhoenixBuffer,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err(Error::invalid("out buffer pointer is null"));
        }
        // SAFETY: validated non-null; always leave `out` in a defined state.
        unsafe { std::ptr::write(out, PhoenixBuffer::empty()) };
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        security::validate_key_len(key_len)?;
        // SAFETY: length bounded; null rejected inside.
        let k = unsafe { slice_from_parts(key, key_len, MAX_KEY_LEN) }?;
        let value = if ct_eq_u64(txn_id, 0) {
            db.get_auto(k)?
        } else {
            db.get(txn_id, k)?
        };
        // SAFETY: `out` validated non-null above.
        unsafe { std::ptr::write(out, PhoenixBuffer::from_vec(value)) };
        Ok(())
    })
}

/// Deletes a key within `txn_id`. Returns `-3` when the key does not exist.
///
/// # Safety
/// `key` must be readable for `key_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_delete(
    handle: *mut PhoenixDbHandle,
    txn_id: u64,
    key: *const u8,
    key_len: usize,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        security::validate_key_len(key_len)?;
        // SAFETY: length bounded; null rejected inside.
        let k = unsafe { slice_from_parts(key, key_len, MAX_KEY_LEN) }?;
        if ct_eq_u64(txn_id, 0) {
            db.delete_auto(k)
        } else {
            db.delete(txn_id, k)
        }
    })
}

/// Single-call insert in an implicit transaction (begin + insert + commit).
///
/// # Safety
/// Same buffer requirements as [`phoenix_insert`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_put_auto(
    handle: *mut PhoenixDbHandle,
    key: *const u8,
    key_len: usize,
    value: *const u8,
    value_len: usize,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        security::validate_key_len(key_len)?;
        security::validate_value_len(value_len)?;
        // SAFETY: lengths bounded; null rejected inside.
        let k = unsafe { slice_from_parts(key, key_len, MAX_KEY_LEN) }?;
        // SAFETY: as above.
        let v = unsafe { slice_from_parts(value, value_len, MAX_VALUE_LEN) }?;
        db.put_auto(k, v)
    })
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

/// Releases a buffer produced by [`phoenix_get`].
///
/// Idempotent for a zeroed buffer and safe with a null argument. This is the
/// **only** legal way to release PhoenixDB memory; the host allocator's `free`
/// must never be used.
///
/// # Safety
/// `buf`, if non-null, must point to a `PhoenixBuffer` this library produced
/// and that has not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_buffer_free(buf: *mut PhoenixBuffer) {
    if buf.is_null() {
        return;
    }
    // SAFETY: non-null and produced by us; reading the three POD fields is
    // valid, and we immediately null the pointer to make a double free a no-op.
    unsafe {
        let b = &mut *buf;
        if !b.ptr.is_null() && b.cap > 0 {
            let v = Vec::from_raw_parts(b.ptr, b.len, b.cap);
            drop(v);
        }
        b.ptr = std::ptr::null_mut();
        b.len = 0;
        b.cap = 0;
    }
}

/// Frees a string returned by [`phoenix_last_error`].
///
/// # Safety
/// `s` must have come from [`phoenix_last_error`] and not been freed already.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_string_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: produced by `CString::into_raw` in `phoenix_last_error`.
    drop(unsafe { std::ffi::CString::from_raw(s) });
}

// ---------------------------------------------------------------------------
// Diagnostics and maintenance
// ---------------------------------------------------------------------------

/// Returns a NUL-terminated description of this thread's last failure.
///
/// The caller owns the string and must release it with
/// [`phoenix_string_free`]. Returns null when no error has been recorded.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_last_error() -> *mut c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(msg) => match std::ffi::CString::new(msg.as_str()) {
            Ok(c) => c.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    })
}

/// Merges pending versions into the tree, flushes and truncates the WAL.
///
/// # Safety
/// `handle` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_checkpoint(handle: *mut PhoenixDbHandle) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        db.checkpoint()
    })
}

/// Flushes dirty pages and syncs the WAL without truncating it.
///
/// # Safety
/// `handle` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_flush(handle: *mut PhoenixDbHandle) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        db.flush()
    })
}

/// Verifies every page checksum and the B+Tree ordering invariants.
///
/// # Safety
/// `handle` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_verify(handle: *mut PhoenixDbHandle) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        db.verify()
    })
}

/// Writes the number of visible keys to `*out_len`.
///
/// # Safety
/// `handle` must be live and `out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_count(handle: *mut PhoenixDbHandle, out_len: *mut u64) -> c_int {
    guard(|| {
        if out_len.is_null() {
            return Err(Error::invalid("out_len is null"));
        }
        // SAFETY: validated non-null above.
        unsafe { *out_len = 0 };
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        let n = db.len()?;
        // SAFETY: validated non-null above.
        unsafe { *out_len = n };
        Ok(())
    })
}

/// ABI version of this build. Dart refuses to load a mismatched library.
///
/// Bumped to 3 in PhoenixDB 2.1: the `phoenix_vector_*` surface was added.
/// Every earlier entry point keeps its signature, so the change is purely
/// additive, but the version is what tells Dart the vector symbols are
/// present — the loader would otherwise fail with a missing symbol at first
/// use rather than at load time.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_abi_version() -> u32 {
    3
}

/// Whether this build includes the vector search engine.
///
/// Always true for the current build: the vector engine has no optional
/// dependencies and is compiled unconditionally. The flag exists so a Dart
/// caller can branch on capability rather than on version arithmetic, exactly
/// as it does for [`phoenix_has_sql`].
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_has_vector() -> c_int {
    1
}

/// Whether this build was compiled with the `sql` feature.
///
/// Lets a Dart caller degrade gracefully instead of getting an error from a
/// lean embedded build that has no query layer.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_has_sql() -> c_int {
    c_int::from(cfg!(feature = "sql"))
}

/// Executes one SQL statement, returning the result as a JSON document.
///
/// JSON is deliberate: a result set is a ragged, dynamically-typed table, and
/// modelling it as a C struct would mean a second allocation protocol and a
/// matching free function for every shape. One UTF-8 buffer with one owner is
/// far harder to leak.
///
/// The document is one of:
///
/// ```json
/// {"type":"rows","columns":["a","b"],"rows":[[1,"x"]]}
/// {"type":"affected","count":3}
/// {"type":"schema","detail":"table `t` created with 2 column(s)"}
/// ```
///
/// On success `*out_json` receives a NUL-terminated string that the caller
/// must release with [`phoenix_string_free`]. On failure it is set to null and
/// a negative status is returned; the message is available from
/// `phoenix_last_error`.
///
/// # Safety
/// `handle` must be live, `sql` a NUL-terminated UTF-8 string, and `out_json` a
/// writable pointer-sized location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_sql_query(
    handle: *mut PhoenixDbHandle,
    sql: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    guard(|| {
        if out_json.is_null() {
            return Err(Error::invalid("out_json is null"));
        }
        // SAFETY: checked non-null immediately above.
        unsafe { *out_json = std::ptr::null_mut() };
        if sql.is_null() {
            return Err(Error::invalid("sql is null"));
        }
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string.
        let text = unsafe { std::ffi::CStr::from_ptr(sql) }
            .to_str()
            .map_err(|_| Error::invalid("sql is not valid UTF-8"))?;

        #[cfg(feature = "sql")]
        {
            let result = crate::sql::Executor::new(db).run(text)?;
            let json = crate::sql::executor::result_to_json(&result);
            let c = std::ffi::CString::new(json)
                .map_err(|_| Error::invalid("result contained an interior NUL"))?;
            // SAFETY: validated non-null above; ownership moves to the caller.
            unsafe { *out_json = c.into_raw() };
            Ok(())
        }
        #[cfg(not(feature = "sql"))]
        {
            let _ = (db, text);
            Err(Error::invalid(
                "this build was compiled without the `sql` feature",
            ))
        }
    })
}

/// Maximum key length accepted by the FFI layer, in bytes.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_max_key_len() -> usize {
    MAX_KEY_LEN
}

/// Maximum value length accepted by the FFI layer, in bytes.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_max_value_len() -> usize {
    MAX_VALUE_LEN
}
