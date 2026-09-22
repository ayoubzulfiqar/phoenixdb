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
//! The vector-search surface (`phoenix_vector_*`) and document collections
//! (`phoenix_collection_*`) follow exactly the same contract and live in
//! [`vector_ffi`] and [`collection_ffi`]; they are re-exported here so the
//! whole C ABI is reachable from one module.

pub mod collection_ffi;
pub mod vector_ffi;

pub use collection_ffi::{
    PhoenixCollectionHandle, phoenix_collection_close, phoenix_collection_count,
    phoenix_collection_delete, phoenix_collection_flush, phoenix_collection_get,
    phoenix_collection_list, phoenix_collection_open, phoenix_collection_open_count,
    phoenix_collection_search, phoenix_collection_stats, phoenix_collection_upsert,
};

pub use vector_ffi::{
    MAX_SEARCH_K, PhoenixVectorHandle, phoenix_free_string_array, phoenix_vector_compact,
    phoenix_vector_contains, phoenix_vector_count, phoenix_vector_dim, phoenix_vector_flush,
    phoenix_vector_free, phoenix_vector_get, phoenix_vector_init, phoenix_vector_insert,
    phoenix_vector_insert_batch, phoenix_vector_kernel, phoenix_vector_last_error,
    phoenix_vector_max_dim, phoenix_vector_max_id_len, phoenix_vector_max_k, phoenix_vector_remove,
    phoenix_vector_save, phoenix_vector_search, phoenix_vector_search_batch,
    phoenix_vector_search_ids, phoenix_vector_stats,
};

use crate::error::{Error, PhoenixStatus};
use crate::security::{self, HandleTag, MAX_KEY_LEN, MAX_VALUE_LEN, ct_eq_u64, slice_from_parts};
use crate::{Database, FillFactor, Options};
use parking_lot::Mutex;
use std::ops::Bound;
use std::os::raw::{c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

/// Opaque database handle handed to C.
///
/// The `tag` is the first field so a stale or foreign pointer is caught by the
/// constant-time check in [`PhoenixDbHandle::validate`] before `db` is
/// touched. `db` is one strong reference (`Arc::into_raw`) to an engine that
/// may be shared with other handles for the same file.
#[repr(C)]
pub struct PhoenixDbHandle {
    tag: HandleTag,
    db: *const Database,
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
        // SAFETY: `db` came from `Arc::into_raw` in `phoenix_open` and this
        // handle's strong reference is only released in `phoenix_close`,
        // which poisons the tag first.
        Ok(unsafe { &*h.db })
    }
}

/// Engines open in this process, by canonical path.
///
/// A second `phoenix_open` of a file this process already has open shares
/// the running engine instead of failing on the file lock. That is what makes
/// several isolates (a UI isolate plus a worker, a preferences store and an
/// app store on one file) safe — they all go through one engine and one lock —
/// and what keeps Flutter hot restart working: the restarted isolate's handles
/// from before the restart are never closed, so their engine is simply
/// reused. Only another *process* sees `PHOENIX_STATUS_BUSY`.
///
/// The last handle's engine is dropped while this lock is held, so a
/// concurrent open never races a closing engine for the file lock.
static REGISTRY: Mutex<Vec<(PathBuf, Weak<Database>)>> = parking_lot::const_mutex(Vec::new());

fn registry_key(path: &Path) -> Option<PathBuf> {
    std::fs::canonicalize(path).ok()
}

/// Opens `path` or joins the engine this process already has open for it.
fn open_shared(path: &Path, options: Options) -> Result<Arc<Database>, Error> {
    let mut registry = REGISTRY.lock();
    registry.retain(|(_, weak)| weak.strong_count() > 0);
    if let Some(key) = registry_key(path)
        && let Some(db) = registry
            .iter()
            .find(|(p, _)| *p == key)
            .and_then(|(_, weak)| weak.upgrade())
    {
        return Ok(db);
    }
    let db = Arc::new(Database::open(path, options)?);
    if let Some(key) = registry_key(path) {
        registry.push((key, Arc::downgrade(&db)));
    }
    Ok(db)
}

/// Number of distinct engines currently open through the C ABI (diagnostics).
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_open_engine_count() -> usize {
    let registry = REGISTRY.lock();
    registry
        .iter()
        .filter(|(_, weak)| weak.strong_count() > 0)
        .count()
}

/// Engine options accepted by [`phoenix_open_ex`].
///
/// Zero means "engine default" for every numeric field, so a zeroed struct
/// with only `struct_size` set is valid.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PhoenixOptions {
    /// `sizeof(PhoenixOptions)` as the caller was compiled with. Lets later
    /// versions append fields without breaking older callers.
    pub struct_size: u32,
    /// `1` fsyncs the WAL on every commit (the default); `0` hands commits to
    /// the OS only (survives an app crash, not power loss). `-1` = default.
    pub sync_on_commit: i32,
    /// Clean-page cache capacity in pages; `0` = default.
    pub cache_pages: u64,
    /// WAL size that triggers an automatic checkpoint; `0` = default.
    pub checkpoint_bytes: u64,
    /// Non-zero records engine spans (see `phoenix_spans_json`).
    pub tracing: i32,
    /// Leaf fill factor before a split, in `(0.5, 1.0]`; `0` = default.
    pub fill_factor_max: f32,
}

impl PhoenixOptions {
    fn to_options(self) -> Result<Options, Error> {
        let mut options = Options::default();
        if self.cache_pages > 0 {
            options.cache_pages = (self.cache_pages as usize).min(1 << 20);
        }
        if self.checkpoint_bytes > 0 {
            options.checkpoint_bytes = self.checkpoint_bytes.max(4096);
        }
        match self.sync_on_commit {
            -1 | 1 => options.sync_on_commit = true,
            0 => options.sync_on_commit = false,
            other => {
                return Err(Error::invalid(format!(
                    "sync_on_commit must be -1, 0 or 1, got {other}"
                )));
            }
        }
        options.tracing = self.tracing != 0;
        if self.fill_factor_max != 0.0 {
            if !(self.fill_factor_max > 0.5 && self.fill_factor_max <= 1.0) {
                return Err(Error::invalid(format!(
                    "fill_factor_max must be in (0.5, 1.0], got {}",
                    self.fill_factor_max
                )));
            }
            options.fill_factor = FillFactor::new(0.5, self.fill_factor_max);
        }
        Ok(options)
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

/// Reads a NUL-terminated UTF-8 path argument.
///
/// # Safety
/// `path`, when non-null, must point to a NUL-terminated string.
unsafe fn path_arg(path: *const c_char, what: &str) -> Result<PathBuf, Error> {
    if path.is_null() {
        return Err(Error::invalid(format!("{what} is null")));
    }
    // SAFETY: caller guarantees a NUL-terminated string.
    let c_str = unsafe { std::ffi::CStr::from_ptr(path) };
    let bytes = c_str.to_bytes();
    if bytes.is_empty() {
        return Err(Error::invalid(format!("{what} is empty")));
    }
    if bytes.len() > 4096 {
        return Err(Error::invalid(format!("{what} exceeds 4096 bytes")));
    }
    let s = c_str
        .to_str()
        .map_err(|_| Error::invalid(format!("{what} is not valid UTF-8")))?;
    Ok(PathBuf::from(s))
}

/// Hands `s` to the caller as a NUL-terminated string (free with
/// [`phoenix_string_free`]).
///
/// # Safety
/// `out` must be valid for a pointer-sized write.
unsafe fn give_string(out: *mut *mut c_char, s: String) -> Result<(), Error> {
    let c = std::ffi::CString::new(s)
        .map_err(|_| Error::invalid("string contained an interior NUL"))?;
    // SAFETY: caller guarantees `out` is writable.
    unsafe { *out = c.into_raw() };
    Ok(())
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Opens (or creates) a database.
///
/// `path` must be a NUL-terminated UTF-8 string. On success `*out_handle`
/// receives a handle that must be released with [`phoenix_close`].
///
/// Opening a path this process already has open returns a new handle to the
/// same engine (options of the later open are ignored); every handle must be
/// closed. A file locked by another process fails with `PHOENIX_STATUS_BUSY`.
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
    let options = PhoenixOptions {
        struct_size: std::mem::size_of::<PhoenixOptions>() as u32,
        sync_on_commit: -1,
        cache_pages: cache_pages as u64,
        checkpoint_bytes: 0,
        tracing: 0,
        fill_factor_max: 0.0,
    };
    // SAFETY: forwarded unchanged; the same contract applies.
    unsafe { phoenix_open_ex(path, &options, out_handle) }
}

/// Opens (or creates) a database with explicit engine options.
///
/// `options` may be null for all defaults. See [`phoenix_open`] for sharing
/// and ownership rules.
///
/// # Safety
/// `path` must point to a valid NUL-terminated string, `options` (if non-null)
/// to a `PhoenixOptions` of at least `options->struct_size` bytes, and
/// `out_handle` to a writable pointer-sized location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_open_ex(
    path: *const c_char,
    options: *const PhoenixOptions,
    out_handle: *mut *mut PhoenixDbHandle,
) -> c_int {
    guard(|| {
        if out_handle.is_null() {
            return Err(Error::invalid("out_handle is null"));
        }
        // SAFETY: checked non-null immediately above.
        unsafe { *out_handle = std::ptr::null_mut() };
        // SAFETY: caller guarantees a NUL-terminated string when non-null.
        let path = unsafe { path_arg(path, "path") }?;
        let options = if options.is_null() {
            Options::default()
        } else {
            // SAFETY: non-null; `struct_size` is read before anything else.
            let size = unsafe { (*options).struct_size } as usize;
            if size < std::mem::size_of::<PhoenixOptions>() {
                return Err(Error::invalid(format!(
                    "PhoenixOptions.struct_size is {size}, expected at least {}",
                    std::mem::size_of::<PhoenixOptions>()
                )));
            }
            // SAFETY: the caller's struct is at least as large as ours.
            unsafe { *options }.to_options()?
        };
        let db = open_shared(&path, options)?;
        let handle = Box::new(PhoenixDbHandle {
            tag: HandleTag::new(),
            db: Arc::into_raw(db),
        });
        // SAFETY: `out_handle` was validated as non-null above.
        unsafe { *out_handle = Box::into_raw(handle) };
        Ok(())
    })
}

/// Closes a database handle, freeing it.
///
/// When this is the last handle to its engine, the engine checkpoints and
/// releases the file. Passing the same handle twice is detected by the
/// poisoned tag and reported as `-2` rather than causing a double free.
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
        let db_ptr = std::mem::replace(&mut h.db, std::ptr::null());
        if !db_ptr.is_null() {
            // SAFETY: created by `Arc::into_raw` in `phoenix_open_ex`; this
            // handle's reference is released exactly once because the tag is
            // poisoned before this point.
            let db = unsafe { Arc::from_raw(db_ptr) };
            // Drop under the registry lock: if this was the last reference the
            // engine checkpoints and unlocks the file before any concurrent
            // open can look for it.
            let _registry = REGISTRY.lock();
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
/// * 3 (PhoenixDB 2.1): the `phoenix_vector_*` surface was added.
/// * 4 (PhoenixDB 4.0): `phoenix_open_ex`, range/prefix scans, write batches,
///   backup/restore/compact, stats, structural check, metrics text, tracing,
///   and the `ABORTED`/`BUSY` status codes. Every earlier entry point keeps
///   its signature.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_abi_version() -> u32 {
    4
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

/// Executes one SQL statement with bound parameters, optionally inside the
/// caller's transaction.
///
/// * `txn_id == 0` runs the statement in its own transaction (retried
///   transparently on a write-write conflict). Otherwise it runs inside
///   `txn_id`, atomically (a failure stages nothing); committing and retrying
///   on `PHOENIX_STATUS_CONFLICT` are then the caller's.
/// * `params_json` is null or a JSON array of scalars bound to `?` / `?N` in
///   order: `null`, booleans (as 0/1), numbers and strings. Bind user data
///   this way rather than splicing it into the SQL text.
///
/// The result document and its ownership are as for [`phoenix_sql_query`].
///
/// # Safety
/// `handle` must be live, `sql` (and `params_json` when non-null) NUL-terminated
/// UTF-8 strings, and `out_json` a writable pointer-sized location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_sql_query_params(
    handle: *mut PhoenixDbHandle,
    txn_id: u64,
    sql: *const c_char,
    params_json: *const c_char,
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
        let sql_bytes = unsafe { std::ffi::CStr::from_ptr(sql) }.to_bytes();
        if sql_bytes.len() > 16 << 20 {
            return Err(Error::invalid("SQL text exceeds 16 MiB"));
        }
        let text =
            std::str::from_utf8(sql_bytes).map_err(|_| Error::invalid("sql is not valid UTF-8"))?;
        let params_text = if params_json.is_null() {
            None
        } else {
            // SAFETY: caller guarantees a NUL-terminated string when non-null.
            Some(
                unsafe { std::ffi::CStr::from_ptr(params_json) }
                    .to_str()
                    .map_err(|_| Error::invalid("params_json is not valid UTF-8"))?,
            )
        };

        #[cfg(feature = "sql")]
        {
            let params = match params_text {
                Some(json) => crate::sql::params_from_json(json)?,
                None => Vec::new(),
            };
            let executor = crate::sql::Executor::new(db);
            let result = if ct_eq_u64(txn_id, 0) {
                executor.run_with(text, &params)?
            } else {
                executor.run_in(txn_id, text, &params)?
            };
            let json = crate::sql::executor::result_to_json(&result);
            // SAFETY: validated non-null above; ownership moves to the caller.
            unsafe { give_string(out_json, json) }
        }
        #[cfg(not(feature = "sql"))]
        {
            let _ = (db, text, params_text, txn_id);
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

/// Receives one key/value pair; return `0` to continue, non-zero to stop.
pub type PhoenixScanCallback =
    Option<unsafe extern "C" fn(*const u8, usize, *const u8, usize) -> c_int>;

/// Streams every visible key/value pair to `callback`, in key order.
///
/// The callback receives pointers into Rust-owned memory that are valid only
/// for the duration of that call; it must copy what it keeps and must not
/// free them. It runs while the engine's shared lock is held, so it must not
/// call back into this database. Returning non-zero stops the scan and makes
/// this function return `PHOENIX_STATUS_ABORTED`.
///
/// # Safety
/// `callback` must be null or valid for the duration of the scan.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_scan_iter(
    handle: *mut PhoenixDbHandle,
    callback: PhoenixScanCallback,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        let Some(callback) = callback else {
            return Err(Error::invalid("scan callback is null"));
        };
        db.scan_iter(|(key, value)| {
            // SAFETY: the callback's validity is the caller's obligation; the
            // pointers stay valid for the duration of the call.
            let rc = unsafe { callback(key.as_ptr(), key.len(), value.as_ptr(), value.len()) };
            if rc != 0 {
                return Err(Error::Aborted);
            }
            Ok(())
        })
    })
}

/// Bound kinds for [`phoenix_scan_range`].
const BOUND_UNBOUNDED: c_int = 0;
const BOUND_INCLUDED: c_int = 1;
const BOUND_EXCLUDED: c_int = 2;

/// Decodes one scan bound.
///
/// # Safety
/// `ptr` must be readable for `len` bytes when `mode` is inclusive/exclusive.
unsafe fn bound_arg<'a>(ptr: *const u8, len: usize, mode: c_int) -> Result<Bound<&'a [u8]>, Error> {
    match mode {
        BOUND_UNBOUNDED => Ok(Bound::Unbounded),
        BOUND_INCLUDED | BOUND_EXCLUDED => {
            // SAFETY: length bounded; null rejected inside for len > 0.
            let key = unsafe { slice_from_parts(ptr, len, MAX_KEY_LEN) }?;
            Ok(if mode == BOUND_INCLUDED {
                Bound::Included(key)
            } else {
                Bound::Excluded(key)
            })
        }
        other => Err(Error::invalid(format!(
            "bound mode must be 0 (unbounded), 1 (inclusive) or 2 (exclusive), got {other}"
        ))),
    }
}

/// Appends one `[u32 key_len][key][u32 value_len][value]` record.
fn push_pair(out: &mut Vec<u8>, key: &[u8], value: &[u8]) {
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value);
}

/// Collects key/value pairs with keys between two bounds into one buffer.
///
/// * `txn_id == 0` reads the latest committed state; otherwise the scan sees
///   that transaction's snapshot and its own uncommitted writes.
/// * `lo_mode` / `hi_mode`: `0` unbounded (pointer ignored), `1` inclusive,
///   `2` exclusive.
/// * Stops after `limit` pairs, or once the buffer holds at least `max_bytes`
///   bytes (each `0` = no limit). To page, call again with the last key as an
///   exclusive lower bound.
///
/// `*out` receives repeated `[u32 LE key_len][key][u32 LE value_len][value]`
/// records in ascending key order; release it with [`phoenix_buffer_free`].
/// No callback is involved, so this is safe to call from any thread.
///
/// # Safety
/// Bound pointers must be readable for their lengths; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_scan_range(
    handle: *mut PhoenixDbHandle,
    txn_id: u64,
    lo: *const u8,
    lo_len: usize,
    lo_mode: c_int,
    hi: *const u8,
    hi_len: usize,
    hi_mode: c_int,
    limit: u64,
    max_bytes: u64,
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
        // SAFETY: the caller's obligation, checked inside.
        let lo = unsafe { bound_arg(lo, lo_len, lo_mode) }?;
        // SAFETY: as above.
        let hi = unsafe { bound_arg(hi, hi_len, hi_mode) }?;
        let mut buf = Vec::new();
        let mut count = 0u64;
        let mut emit = |k: Vec<u8>, v: Vec<u8>| {
            push_pair(&mut buf, &k, &v);
            count += 1;
            Ok((limit == 0 || count < limit) && (max_bytes == 0 || (buf.len() as u64) < max_bytes))
        };
        if ct_eq_u64(txn_id, 0) {
            db.scan_range(lo, hi, &mut emit)?;
        } else {
            db.scan_txn(txn_id, lo, hi, &mut emit)?;
        }
        // SAFETY: `out` validated non-null above.
        unsafe { std::ptr::write(out, PhoenixBuffer::from_vec(buf)) };
        Ok(())
    })
}

/// Collects pairs whose key starts with `prefix`; otherwise identical to
/// [`phoenix_scan_range`]. An empty prefix scans everything.
///
/// # Safety
/// `prefix` must be readable for `prefix_len` bytes; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_scan_prefix(
    handle: *mut PhoenixDbHandle,
    txn_id: u64,
    prefix: *const u8,
    prefix_len: usize,
    limit: u64,
    max_bytes: u64,
    out: *mut PhoenixBuffer,
) -> c_int {
    let prefix_bytes = if prefix_len == 0 {
        &[][..]
    } else {
        // SAFETY: the caller's obligation; bounded, null rejected.
        match unsafe { slice_from_parts(prefix, prefix_len, MAX_KEY_LEN) } {
            Ok(p) => p,
            Err(e) => {
                set_last_error(&e);
                if !out.is_null() {
                    // SAFETY: non-null; leave it defined.
                    unsafe { std::ptr::write(out, PhoenixBuffer::empty()) };
                }
                return e.status() as c_int;
            }
        }
    };
    let end = crate::prefix_successor(prefix_bytes);
    let (lo_mode, hi_mode) = (
        if prefix_bytes.is_empty() {
            BOUND_UNBOUNDED
        } else {
            BOUND_INCLUDED
        },
        if end.is_some() {
            BOUND_EXCLUDED
        } else {
            BOUND_UNBOUNDED
        },
    );
    let end = end.unwrap_or_default();
    // SAFETY: both bounds point into live local buffers.
    unsafe {
        phoenix_scan_range(
            handle,
            txn_id,
            prefix_bytes.as_ptr(),
            prefix_bytes.len(),
            lo_mode,
            end.as_ptr(),
            end.len(),
            hi_mode,
            limit,
            max_bytes,
            out,
        )
    }
}

/// Batch operation codes for [`phoenix_write_batch`].
const BATCH_PUT: u8 = 1;
const BATCH_DELETE: u8 = 2;
const BATCH_DELETE_IF_EXISTS: u8 = 3;

enum BatchOp<'a> {
    Put(&'a [u8], &'a [u8]),
    Delete(&'a [u8], bool),
}

fn parse_batch(mut bytes: &[u8]) -> Result<Vec<BatchOp<'_>>, Error> {
    fn take<'a>(bytes: &mut &'a [u8], n: usize, what: &str) -> Result<&'a [u8], Error> {
        if bytes.len() < n {
            return Err(Error::invalid(format!("write batch truncated in {what}")));
        }
        let (head, tail) = bytes.split_at(n);
        *bytes = tail;
        Ok(head)
    }
    fn take_len(bytes: &mut &[u8], what: &str) -> Result<usize, Error> {
        let raw = take(bytes, 4, what)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize)
    }
    let mut ops = Vec::new();
    while !bytes.is_empty() {
        let code = take(&mut bytes, 1, "an op code")?[0];
        let key_len = take_len(&mut bytes, "a key length")?;
        security::validate_key_len(key_len)?;
        let key = take(&mut bytes, key_len, "a key")?;
        ops.push(match code {
            BATCH_PUT => {
                let value_len = take_len(&mut bytes, "a value length")?;
                security::validate_value_len(value_len)?;
                BatchOp::Put(key, take(&mut bytes, value_len, "a value")?)
            }
            BATCH_DELETE => BatchOp::Delete(key, false),
            BATCH_DELETE_IF_EXISTS => BatchOp::Delete(key, true),
            other => return Err(Error::invalid(format!("unknown write batch op {other}"))),
        });
    }
    Ok(ops)
}

/// Applies a batch of writes atomically in one transaction.
///
/// `ops` holds repeated records: `[u8 op][u32 LE key_len][key]` followed, for
/// a put, by `[u32 LE value_len][value]`. Ops: `1` put, `2` delete (fails the
/// whole batch with `NOT_FOUND` if the key is absent), `3` delete if present.
/// The batch is validated before anything is staged; any failure rolls the
/// whole batch back. Maximum batch size: 1 GiB.
///
/// # Safety
/// `ops` must be readable for `ops_len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_write_batch(
    handle: *mut PhoenixDbHandle,
    ops: *const u8,
    ops_len: usize,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        // SAFETY: bounded; null rejected for a non-empty batch.
        let bytes = unsafe { slice_from_parts(ops, ops_len, 1 << 30) }?;
        let parsed = parse_batch(bytes)?;
        if parsed.is_empty() {
            return Ok(());
        }
        db.write_batch(|batch| {
            for op in &parsed {
                match *op {
                    BatchOp::Put(k, v) => batch.put(k, v)?,
                    BatchOp::Delete(k, false) => batch.delete(k)?,
                    BatchOp::Delete(k, true) => batch.delete_if_exists(k)?,
                }
            }
            Ok(())
        })
    })
}

// ---------------------------------------------------------------------------
// Backup, restore, compaction
// ---------------------------------------------------------------------------

/// Writes a consistent, compacted, self-contained copy of the database to
/// `path` while other callers keep reading and writing.
///
/// # Safety
/// `handle` must be live and `path` a NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_backup(
    handle: *mut PhoenixDbHandle,
    path: *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string.
        let path = unsafe { path_arg(path, "backup path") }?;
        db.backup(path)
    })
}

/// Replaces the database's contents with the backup at `path`, atomically.
/// Fails with `INVALID_ARGUMENT` while any transaction is open.
///
/// # Safety
/// `handle` must be live and `path` a NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_restore(
    handle: *mut PhoenixDbHandle,
    path: *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string.
        let path = unsafe { path_arg(path, "restore path") }?;
        db.restore(path)
    })
}

/// Rebuilds the file with live data only, returning free pages to the
/// filesystem. Blocks other callers for the duration.
///
/// # Safety
/// `handle` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_compact(handle: *mut PhoenixDbHandle) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        db.compact()
    })
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Runtime statistics written by [`phoenix_stats`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PhoenixStats {
    /// Pages allocated in the file.
    pub page_count: u32,
    /// Reserved; always zero.
    pub reserved: u32,
    /// Live transactions.
    pub active_txns: u64,
    /// Keys with versions not yet merged into the tree.
    pub pending_keys: u64,
    /// Current WAL size in bytes.
    pub wal_bytes: u64,
    /// Latest commit timestamp.
    pub commit_ts: u64,
    /// Every version at or below this timestamp is in the durable tree.
    pub tree_ts: u64,
    /// Page reads served from memory.
    pub cache_hits: u64,
    /// Page reads that decoded a page from the file.
    pub cache_misses: u64,
}

/// Writes runtime statistics to `*out`.
///
/// # Safety
/// `handle` must be live and `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_stats(
    handle: *mut PhoenixDbHandle,
    out: *mut PhoenixStats,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err(Error::invalid("out is null"));
        }
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        let s = db.stats();
        let stats = PhoenixStats {
            page_count: s.page_count,
            reserved: 0,
            active_txns: s.active_txns as u64,
            pending_keys: s.pending_keys as u64,
            wal_bytes: s.wal_bytes,
            commit_ts: s.commit_ts,
            tree_ts: s.tree_ts,
            cache_hits: s.cache_hits,
            cache_misses: s.cache_misses,
        };
        // SAFETY: validated non-null above.
        unsafe { std::ptr::write(out, stats) };
        Ok(())
    })
}

/// Result of [`phoenix_check`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PhoenixTreeReport {
    /// Levels from the root to the leaves.
    pub depth: u32,
    /// Leaf pages reachable from the root.
    pub leaf_pages: u32,
    /// Internal pages reachable from the root.
    pub internal_pages: u32,
    /// Overflow pages reachable from leaf cells.
    pub overflow_pages: u32,
    /// Pages on the free list.
    pub free_pages: u32,
    /// Allocated pages neither reachable nor free (leaked by a crash).
    pub unreachable_pages: u32,
    /// Non-root leaves below the minimum fill factor.
    pub underfull_leaves: u32,
    /// Reserved; always zero.
    pub reserved: u32,
    /// Keys stored in the tree (excluding unmerged in-memory versions).
    pub keys: u64,
}

/// Runs the full structural check and writes its report to `*out`.
///
/// Returns `PHOENIX_STATUS_CORRUPTION` (with details from
/// `phoenix_last_error`) when any invariant is violated.
///
/// # Safety
/// `handle` must be live and `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_check(
    handle: *mut PhoenixDbHandle,
    out: *mut PhoenixTreeReport,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err(Error::invalid("out is null"));
        }
        // SAFETY: validated non-null; always leave `out` defined.
        unsafe { std::ptr::write(out, PhoenixTreeReport::default()) };
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        let r = db.check()?;
        let report = PhoenixTreeReport {
            depth: r.depth,
            leaf_pages: r.leaf_pages,
            internal_pages: r.internal_pages,
            overflow_pages: r.overflow_pages,
            free_pages: r.free_pages,
            unreachable_pages: r.unreachable_pages,
            underfull_leaves: r.underfull_leaves,
            reserved: 0,
            keys: r.keys,
        };
        // SAFETY: validated non-null above.
        unsafe { std::ptr::write(out, report) };
        Ok(())
    })
}

/// Renders the metrics registry: `format` `0` = human-readable report, `1` =
/// Prometheus text exposition. `*out` receives a string to release with
/// [`phoenix_string_free`].
///
/// # Safety
/// `handle` must be live and `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_metrics_text(
    handle: *mut PhoenixDbHandle,
    format: c_int,
    out: *mut *mut c_char,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err(Error::invalid("out is null"));
        }
        // SAFETY: checked non-null.
        unsafe { *out = std::ptr::null_mut() };
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        let text = match format {
            0 => db.metrics_report(),
            1 => db.metrics_prometheus(),
            other => return Err(Error::invalid(format!("unknown metrics format {other}"))),
        };
        // SAFETY: `out` validated above.
        unsafe { give_string(out, text) }
    })
}

/// Writes the engine's metrics report into `out_buf` as a NUL-terminated UTF-8
/// string.
///
/// Returns the number of bytes written (excluding the NUL terminator), or a
/// negative [`PhoenixStatus`]: `PHOENIX_STATUS_FULL` when `out_len` is too
/// small (nothing is written). Prefer [`phoenix_metrics_text`], which has no
/// size negotiation.
///
/// # Safety
/// `handle` must be live, `out_buf` must point to `out_len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_metrics_report(
    handle: *mut PhoenixDbHandle,
    out_buf: *mut u8,
    out_len: usize,
) -> isize {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if out_buf.is_null() && out_len != 0 {
            return Err(Error::invalid("null buffer with non-zero length"));
        }
        // SAFETY: handle validity is the caller's documented obligation.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        let report = db.metrics_report();
        let bytes = report.as_bytes();
        if out_len < bytes.len() + 1 {
            return Err(Error::Full(format!(
                "metrics report needs {} bytes, buffer holds {out_len}",
                bytes.len() + 1
            )));
        }
        // SAFETY: out_buf is valid for out_len bytes, out_len >= bytes.len()+1.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf, bytes.len());
            *out_buf.add(bytes.len()) = 0;
        }
        Ok(bytes.len())
    }));

    match result {
        Ok(Ok(n)) => n as isize,
        Ok(Err(e)) => {
            set_last_error(&e);
            // Status codes are already negative, so they can never be
            // mistaken for a byte count.
            e.status() as isize
        }
        Err(_) => {
            set_last_error(&Error::corrupt("panic caught at the FFI boundary"));
            PhoenixStatus::Panic as isize
        }
    }
}

/// Turns span recording on (`enabled != 0`) or off.
///
/// # Safety
/// `handle` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_set_tracing(
    handle: *mut PhoenixDbHandle,
    enabled: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        db.set_tracing(enabled != 0);
        Ok(())
    })
}

/// Writes the recorded spans (oldest first, at most the last 1024) as a JSON
/// array to `*out`; release it with [`phoenix_string_free`]. Each element is
/// `{"name","trace_id","span_id","parent_id","duration_us","error","attributes"}`.
///
/// # Safety
/// `handle` must be live and `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_spans_json(
    handle: *mut PhoenixDbHandle,
    out: *mut *mut c_char,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err(Error::invalid("out is null"));
        }
        // SAFETY: checked non-null.
        unsafe { *out = std::ptr::null_mut() };
        // SAFETY: see `phoenix_begin_txn`.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        let mut json = String::from("[");
        for (i, span) in db.spans().iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str("{\"name\":");
            json_string(&mut json, &span.name);
            json.push_str(&format!(
                ",\"trace_id\":{},\"span_id\":{},\"parent_id\":{},\"duration_us\":{},\"error\":{},\"attributes\":{{",
                span.trace_id,
                span.span_id,
                span.parent_id.map_or_else(|| "null".to_string(), |p| p.to_string()),
                span.duration_micros(),
                span.error
            ));
            for (j, (k, v)) in span.attributes.iter().enumerate() {
                if j > 0 {
                    json.push(',');
                }
                json_string(&mut json, k);
                json.push(':');
                json_string(&mut json, v);
            }
            json.push_str("}}");
        }
        json.push(']');
        // SAFETY: `out` validated above.
        unsafe { give_string(out, json) }
    })
}

/// Appends `s` as a JSON string literal.
fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}
