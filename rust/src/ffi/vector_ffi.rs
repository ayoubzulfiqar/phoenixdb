//! C ABI for the vector search engine.
//!
//! # Boundary contract
//!
//! Identical to `ffi.rs`, and deliberately so — a Dart caller should not have
//! to learn a second set of rules:
//!
//! * Every function returns an `int32_t` [`PhoenixStatus`]; `0` is success and
//!   every failure is negative.
//! * **No caller pointer is dereferenced before it is validated.** Null checks,
//!   length limits and handle-tag verification all happen first.
//! * No Rust panic may unwind into Dart: every body is wrapped in
//!   [`std::panic::catch_unwind`], and a caught panic becomes `-7`.
//! * Memory allocated here is owned here. Release id arrays with
//!   [`phoenix_free_string_array`] and single strings with
//!   `phoenix_string_free` — never with the host `free`.
//!
//! # Zero-copy
//!
//! Query and insert vectors are read straight from the caller's
//! `Float32List` through a borrowed `&[f32]`: nothing is copied on the way in.
//! Results are written directly into caller-owned output arrays, so the only
//! allocation per search is the `k` id strings, which the caller frees in one
//! call.
//!
//! # Alignment
//!
//! `*const f32` from Dart is 4-byte aligned by construction (a `Float32List`
//! is), but a hostile or buggy caller could pass anything, so
//! [`slice_from_raw_f32`] verifies alignment before building the slice.

use crate::error::{Error, PhoenixStatus};
use crate::security::HandleTag;
use crate::vector::{MAX_DIM, MAX_ID_LEN, Metric, VectorEngine, VectorOptions};
use std::os::raw::{c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

/// Largest `k` a single search may request.
///
/// Bounds the caller-visible output arrays: `out_ids` and `out_scores` must
/// each hold `k` elements, and a nonsensical `k` would otherwise be an
/// invitation to write past them.
pub const MAX_SEARCH_K: usize = 4096;

/// Opaque vector-engine handle handed to C.
///
/// The `tag` is the first field so a stale or foreign pointer is caught by the
/// constant-time check in [`PhoenixVectorHandle::validate`] before `engine` is
/// touched.
#[repr(C)]
pub struct PhoenixVectorHandle {
    tag: HandleTag,
    engine: *mut VectorEngine,
}

impl PhoenixVectorHandle {
    /// Validates a raw handle pointer and borrows the engine.
    ///
    /// # Safety
    /// `handle` must be a pointer previously returned by
    /// `phoenix_vector_init` and not yet passed to `phoenix_vector_free`.
    unsafe fn validate<'a>(handle: *const PhoenixVectorHandle) -> Result<&'a VectorEngine, Error> {
        if handle.is_null() {
            return Err(Error::invalid("null vector engine handle"));
        }
        // SAFETY: non-null; the tag is read first and rejects freed memory
        // with overwhelming probability before `engine` is dereferenced.
        let h = unsafe { &*handle };
        if !h.tag.is_valid() {
            return Err(Error::invalid(
                "invalid or already-freed vector engine handle",
            ));
        }
        if h.engine.is_null() {
            return Err(Error::Closed);
        }
        // SAFETY: `engine` was created by `Box::into_raw` in
        // `phoenix_vector_init` and is only freed in `phoenix_vector_free`,
        // which poisons the tag first.
        Ok(unsafe { &*h.engine })
    }
}

thread_local! {
    /// Human-readable description of the most recent vector failure.
    static LAST_VECTOR_ERROR: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

fn set_last_error(e: &Error) {
    let msg = e.to_string();
    LAST_VECTOR_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(msg);
    });
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
            set_last_error(&Error::corrupt("panic caught at the vector FFI boundary"));
            PhoenixStatus::Panic as c_int
        }
    }
}

/// Rebuilds a `&[f32]` from a C pointer/length pair after full validation.
///
/// Rejects a null pointer, a zero or over-long length, a misaligned pointer,
/// and a pointer/length pair that would wrap the address space. The pointer is
/// **never** dereferenced before every check passes.
///
/// # Safety
/// When `ptr` is non-null and the checks pass, the caller guarantees that
/// `ptr..ptr + len` is one allocated object, immutable and live for `'a`.
unsafe fn slice_from_raw_f32<'a>(ptr: *const f32, len: usize) -> Result<&'a [f32], Error> {
    if len == 0 {
        return Err(Error::invalid("vector length must be greater than zero"));
    }
    if len > MAX_DIM {
        return Err(Error::invalid(format!(
            "vector length {len} exceeds the limit of {MAX_DIM}"
        )));
    }
    if ptr.is_null() {
        return Err(Error::invalid("null vector pointer"));
    }
    let address = ptr as usize;
    if address % std::mem::align_of::<f32>() != 0 {
        return Err(Error::invalid(
            "vector pointer is not 4-byte aligned; pass a Float32List buffer",
        ));
    }
    let bytes = len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::invalid("vector length overflows a byte count"))?;
    if address.checked_add(bytes).is_none() {
        return Err(Error::invalid(
            "vector pointer + length overflows the address space",
        ));
    }
    // SAFETY: non-null, aligned, non-wrapping and length-checked; validity for
    // the stated extent is the caller's documented obligation.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Reads a NUL-terminated UTF-8 id, enforcing the length limit.
///
/// # Safety
/// `ptr` must be a valid NUL-terminated string.
unsafe fn id_from_raw<'a>(ptr: *const c_char) -> Result<&'a str, Error> {
    if ptr.is_null() {
        return Err(Error::invalid("null vector id"));
    }
    // SAFETY: caller guarantees a NUL-terminated string.
    let c_str = unsafe { std::ffi::CStr::from_ptr(ptr) };
    let bytes = c_str.to_bytes();
    if bytes.is_empty() {
        return Err(Error::invalid("vector id must not be empty"));
    }
    if bytes.len() > MAX_ID_LEN {
        return Err(Error::invalid(format!(
            "vector id of {} bytes exceeds the {MAX_ID_LEN}-byte limit",
            bytes.len()
        )));
    }
    c_str
        .to_str()
        .map_err(|_| Error::invalid("vector id is not valid UTF-8"))
}

/// Reads a NUL-terminated UTF-8 filesystem path.
///
/// # Safety
/// `ptr` must be a valid NUL-terminated string.
unsafe fn path_from_raw(ptr: *const c_char) -> Result<PathBuf, Error> {
    if ptr.is_null() {
        return Err(Error::invalid("path is null"));
    }
    // SAFETY: caller guarantees a NUL-terminated string.
    let c_str = unsafe { std::ffi::CStr::from_ptr(ptr) };
    let bytes = c_str.to_bytes();
    if bytes.is_empty() {
        return Err(Error::invalid("path is empty"));
    }
    if bytes.len() > 4096 {
        return Err(Error::invalid("path exceeds 4096 bytes"));
    }
    let text = c_str
        .to_str()
        .map_err(|_| Error::invalid("path is not valid UTF-8"))?;
    Ok(PathBuf::from(text))
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Creates a vector index at `path`, or opens an existing one.
///
/// * `path` — NUL-terminated UTF-8 file path. The graph snapshot lives beside
///   it at `<path>.hnsw`.
/// * `dim` — dimensionality, `1..=65536`. Must match an existing file.
/// * `metric` — `0` cosine, `1` euclidean, `2` dot product.
/// * `max_elements` — capacity hint used to pre-reserve the id map; `0` means
///   "unknown", and the index still grows without bound.
///
/// On success `*out_handle` receives a handle that must be released with
/// [`phoenix_vector_free`]. On failure it is set to null.
///
/// # Safety
/// `path` must be a valid NUL-terminated string and `out_handle` a writable
/// pointer-sized location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_init(
    path: *const c_char,
    dim: usize,
    metric: u8,
    max_elements: usize,
    out_handle: *mut *mut PhoenixVectorHandle,
) -> c_int {
    guard(|| {
        if out_handle.is_null() {
            return Err(Error::invalid("out_handle is null"));
        }
        // SAFETY: checked non-null immediately above; always leave the output
        // in a defined state before anything can fail.
        unsafe { *out_handle = std::ptr::null_mut() };

        // SAFETY: caller guarantees a NUL-terminated string.
        let path = unsafe { path_from_raw(path) }?;
        let metric = Metric::from_u8(metric)?;
        let options = VectorOptions {
            max_elements,
            ..VectorOptions::default()
        };
        let engine = VectorEngine::open(path, dim, metric, options)?;

        let handle = Box::new(PhoenixVectorHandle {
            tag: HandleTag::new(),
            engine: Box::into_raw(Box::new(engine)),
        });
        // SAFETY: `out_handle` was validated as non-null above.
        unsafe { *out_handle = Box::into_raw(handle) };
        Ok(())
    })
}

/// Saves and destroys a vector index, freeing the handle.
///
/// Passing the same handle twice is caught by the poisoned tag and ignored,
/// rather than causing a double free. Null is a no-op.
///
/// # Safety
/// `handle` must come from [`phoenix_vector_init`] and must not be used
/// afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_free(handle: *mut PhoenixVectorHandle) {
    if handle.is_null() {
        return;
    }
    // A free path must never unwind into Dart, so the whole body is guarded
    // and the status is deliberately discarded: there is nothing a caller
    // could do about a failure here.
    let _ = guard(|| {
        // SAFETY: non-null; the tag is verified before any other field is read.
        let h = unsafe { &mut *handle };
        if !h.tag.is_valid() {
            return Err(Error::invalid(
                "invalid or already-freed vector engine handle",
            ));
        }
        h.tag.poison(); // reject any concurrent or subsequent use
        let engine_ptr = std::mem::replace(&mut h.engine, std::ptr::null_mut());
        if !engine_ptr.is_null() {
            // SAFETY: created by `Box::into_raw` in `phoenix_vector_init`, and
            // freed exactly once because the tag is poisoned above. `Drop`
            // saves the snapshot.
            drop(unsafe { Box::from_raw(engine_ptr) });
        }
        // SAFETY: the handle box itself is freed exactly once, here.
        drop(unsafe { Box::from_raw(handle) });
        Ok(())
    });
}

// ---------------------------------------------------------------------------
// Data plane
// ---------------------------------------------------------------------------

/// Inserts or replaces a vector.
///
/// `vec_ptr` must point to `vec_len` contiguous `f32`s, and `vec_len` must
/// equal the index's dimensionality. The buffer is read, not retained: it may
/// be freed the moment this returns.
///
/// Re-inserting an existing id replaces it.
///
/// # Safety
/// `id` must be a valid NUL-terminated string and `vec_ptr` must be readable
/// for `vec_len` floats for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_insert(
    handle: *mut PhoenixVectorHandle,
    id: *const c_char,
    vec_ptr: *const f32,
    vec_len: usize,
) -> c_int {
    guard(|| {
        // SAFETY: handle validity is the caller's documented obligation.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string.
        let id = unsafe { id_from_raw(id) }?;
        // SAFETY: length and alignment are checked inside before any read.
        let vector = unsafe { slice_from_raw_f32(vec_ptr, vec_len) }?;
        engine.insert(id, vector)
    })
}

/// Searches for the `k` nearest neighbours of `query_ptr`.
///
/// The caller owns both output arrays and must size each to hold at least `k`
/// elements:
///
/// * `out_ids` receives `*out_count` NUL-terminated strings allocated by this
///   library. Release the whole array with [`phoenix_free_string_array`],
///   passing the same count.
/// * `out_scores` receives the matching distances, ascending (nearest first).
///
/// `*out_count` is set to the number of results actually written, which is at
/// most `k` and may be fewer when the index holds fewer live vectors. It is
/// written before anything else can fail, so it is always meaningful.
///
/// `ef` overrides the search beam width; `0` selects the configured default.
/// Higher values trade latency for recall.
///
/// # Safety
/// `query_ptr` must be readable for `query_len` floats; `out_ids` and
/// `out_scores` must each be writable for `k` elements; `out_count` must be
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_search(
    handle: *const PhoenixVectorHandle,
    query_ptr: *const f32,
    query_len: usize,
    k: usize,
    ef: usize,
    out_ids: *mut *mut c_char,
    out_scores: *mut f32,
    out_count: *mut usize,
) -> c_int {
    guard(|| {
        if out_count.is_null() {
            return Err(Error::invalid("out_count is null"));
        }
        // SAFETY: validated non-null above; the count is always defined even
        // when a later check fails.
        unsafe { *out_count = 0 };

        if k == 0 {
            return Err(Error::invalid("k must be greater than zero"));
        }
        if k > MAX_SEARCH_K {
            return Err(Error::invalid(format!(
                "k of {k} exceeds the limit of {MAX_SEARCH_K}"
            )));
        }
        if out_ids.is_null() {
            return Err(Error::invalid("out_ids is null"));
        }
        if out_scores.is_null() {
            return Err(Error::invalid("out_scores is null"));
        }

        // SAFETY: see `phoenix_vector_insert`.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        // SAFETY: length and alignment are checked inside before any read.
        let query = unsafe { slice_from_raw_f32(query_ptr, query_len) }?;

        let matches = engine.search(query, k, (ef > 0).then_some(ef))?;
        debug_assert!(matches.len() <= k, "engine returned more than k results");

        // Every id is converted to a C string *before* any is published, so a
        // failure part-way through frees its own allocations instead of
        // leaving the caller a half-filled array it cannot safely release.
        let mut owned: Vec<*mut c_char> = Vec::with_capacity(matches.len());
        for entry in &matches {
            match std::ffi::CString::new(entry.id.as_str()) {
                Ok(c) => owned.push(c.into_raw()),
                Err(_) => {
                    for ptr in owned {
                        // SAFETY: each pointer came from `CString::into_raw`
                        // in this loop and has not been handed out.
                        drop(unsafe { std::ffi::CString::from_raw(ptr) });
                    }
                    return Err(Error::corrupt("stored id contains an interior NUL"));
                }
            }
        }

        for (index, (ptr, entry)) in owned.iter().zip(&matches).enumerate() {
            // SAFETY: `out_ids` and `out_scores` are non-null and the caller
            // guarantees room for `k` elements; `index < matches.len() <= k`.
            unsafe {
                out_ids.add(index).write(*ptr);
                out_scores.add(index).write(entry.distance);
            }
        }
        // SAFETY: validated non-null above.
        unsafe { *out_count = matches.len() };
        Ok(())
    })
}

/// Fetches a stored vector by id, copying it into `out_vec`.
///
/// `out_vec` must be writable for `out_len` floats, and `out_len` must equal
/// the index's dimensionality. Returns `-3` when the id is unknown or has been
/// removed.
///
/// # Safety
/// `id` must be a valid NUL-terminated string; `out_vec` must be writable for
/// `out_len` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_get(
    handle: *const PhoenixVectorHandle,
    id: *const c_char,
    out_vec: *mut f32,
    out_len: usize,
) -> c_int {
    guard(|| {
        if out_vec.is_null() {
            return Err(Error::invalid("out_vec is null"));
        }
        // SAFETY: see `phoenix_vector_insert`.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        if out_len != engine.dim() {
            return Err(Error::invalid(format!(
                "out_len is {out_len}, index dimensionality is {}",
                engine.dim()
            )));
        }
        // SAFETY: `out_vec` is non-null and the caller guarantees `out_len`
        // writable floats; alignment matches because `f32` is the element type.
        let destination = unsafe { std::slice::from_raw_parts_mut(out_vec, out_len) };
        // SAFETY: caller guarantees a NUL-terminated string.
        let id = unsafe { id_from_raw(id) }?;
        let vector = engine.get(id)?;
        destination.copy_from_slice(&vector);
        Ok(())
    })
}

/// Removes a vector by id. Returns `-3` when it is absent.
///
/// # Safety
/// `id` must be a valid NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_remove(
    handle: *mut PhoenixVectorHandle,
    id: *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_vector_insert`.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string.
        let id = unsafe { id_from_raw(id) }?;
        engine.remove(id)
    })
}

/// Writes `1` to `*out_present` when `id` is stored and live, `0` otherwise.
///
/// # Safety
/// `id` must be a valid NUL-terminated string; `out_present` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_contains(
    handle: *const PhoenixVectorHandle,
    id: *const c_char,
    out_present: *mut c_int,
) -> c_int {
    guard(|| {
        if out_present.is_null() {
            return Err(Error::invalid("out_present is null"));
        }
        // SAFETY: validated non-null above.
        unsafe { *out_present = 0 };
        // SAFETY: see `phoenix_vector_insert`.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string.
        let id = unsafe { id_from_raw(id) }?;
        // SAFETY: validated non-null above.
        unsafe { *out_present = c_int::from(engine.contains(id)) };
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

/// Syncs the vector file and writes the HNSW snapshot.
///
/// `path` overrides the snapshot location; pass null for the default
/// `<vector file>.hnsw`. The write is atomic — a temporary file is `fsync`ed
/// and then renamed — so a crash mid-save leaves the previous snapshot intact.
///
/// # Safety
/// `handle` must be live; `path`, when non-null, must be a valid
/// NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_save(
    handle: *mut PhoenixVectorHandle,
    path: *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_vector_insert`.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        if path.is_null() {
            return engine.save(None);
        }
        // SAFETY: caller guarantees a NUL-terminated string when non-null.
        let path = unsafe { path_from_raw(path) }?;
        engine.save(Some(&path))
    })
}

/// Syncs the vector file without writing a snapshot.
///
/// # Safety
/// `handle` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_flush(handle: *mut PhoenixVectorHandle) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_vector_insert`.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        engine.flush()
    })
}

/// Rewrites the index without tombstoned records, writing the number of
/// reclaimed slots to `*out_reclaimed`.
///
/// # Safety
/// `handle` must be live; `out_reclaimed`, when non-null, must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_compact(
    handle: *mut PhoenixVectorHandle,
    out_reclaimed: *mut usize,
) -> c_int {
    guard(|| {
        if !out_reclaimed.is_null() {
            // SAFETY: checked non-null immediately above.
            unsafe { *out_reclaimed = 0 };
        }
        // SAFETY: see `phoenix_vector_insert`.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        let reclaimed = engine.compact()?;
        if !out_reclaimed.is_null() {
            // SAFETY: checked non-null above.
            unsafe { *out_reclaimed = reclaimed };
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

/// Writes the number of live vectors to `*out_len`.
///
/// # Safety
/// `handle` must be live and `out_len` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_count(
    handle: *const PhoenixVectorHandle,
    out_len: *mut usize,
) -> c_int {
    guard(|| {
        if out_len.is_null() {
            return Err(Error::invalid("out_len is null"));
        }
        // SAFETY: validated non-null above.
        unsafe { *out_len = 0 };
        // SAFETY: see `phoenix_vector_insert`.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        // SAFETY: validated non-null above.
        unsafe { *out_len = engine.len() };
        Ok(())
    })
}

/// Writes the index's dimensionality to `*out_dim`.
///
/// # Safety
/// `handle` must be live and `out_dim` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_dim(
    handle: *const PhoenixVectorHandle,
    out_dim: *mut usize,
) -> c_int {
    guard(|| {
        if out_dim.is_null() {
            return Err(Error::invalid("out_dim is null"));
        }
        // SAFETY: validated non-null above.
        unsafe { *out_dim = 0 };
        // SAFETY: see `phoenix_vector_insert`.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        // SAFETY: validated non-null above.
        unsafe { *out_dim = engine.dim() };
        Ok(())
    })
}

/// Writes live, total and deleted record counts to the three outputs.
///
/// Any output may be null, in which case that statistic is skipped.
///
/// # Safety
/// `handle` must be live; each non-null output must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_vector_stats(
    handle: *const PhoenixVectorHandle,
    out_live: *mut usize,
    out_total: *mut usize,
    out_deleted: *mut usize,
) -> c_int {
    guard(|| {
        // SAFETY: see `phoenix_vector_insert`.
        let engine = unsafe { PhoenixVectorHandle::validate(handle) }?;
        let stats = engine.stats();
        // SAFETY: each pointer is checked non-null immediately before its
        // single write.
        unsafe {
            if !out_live.is_null() {
                *out_live = stats.live;
            }
            if !out_total.is_null() {
                *out_total = stats.total;
            }
            if !out_deleted.is_null() {
                *out_deleted = stats.deleted;
            }
        }
        Ok(())
    })
}

/// Returns a NUL-terminated description of this thread's last vector failure.
///
/// The caller owns the string and must release it with `phoenix_string_free`.
/// Returns null when no error has been recorded.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_vector_last_error() -> *mut c_char {
    LAST_VECTOR_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(msg) => match std::ffi::CString::new(msg.as_str()) {
            Ok(c) => c.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    })
}

/// Name of the SIMD kernel this CPU selected: `avx2+fma`, `neon` or
/// `portable`.
///
/// The returned pointer is a `'static` string owned by the library and must
/// **not** be freed.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_vector_kernel() -> *const c_char {
    // NUL-terminated at the source so no allocation and no free are needed.
    match VectorEngine::kernel() {
        "avx2+fma" => c"avx2+fma".as_ptr(),
        "neon" => c"neon".as_ptr(),
        _ => c"portable".as_ptr(),
    }
}

/// Largest dimensionality this build accepts.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_vector_max_dim() -> usize {
    MAX_DIM
}

/// Largest `k` a single search may request.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_vector_max_k() -> usize {
    MAX_SEARCH_K
}

/// Largest vector id this build accepts, in bytes.
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_vector_max_id_len() -> usize {
    MAX_ID_LEN
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

/// Releases an array of `len` strings produced by [`phoenix_vector_search`].
///
/// Frees each string and then nulls its slot, so a double free is a no-op
/// rather than undefined behaviour. The array itself belongs to the caller and
/// is **not** freed here — only the strings inside it. Null is a no-op.
///
/// # Safety
/// `ptrs` must point to `len` pointers that this library produced and that have
/// not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_free_string_array(ptrs: *mut *mut c_char, len: usize) {
    if ptrs.is_null() || len == 0 {
        return;
    }
    if len > MAX_SEARCH_K {
        // A length beyond any array this library ever produces means the
        // caller is confused; walking it would be a wild read.
        return;
    }
    for index in 0..len {
        // SAFETY: `ptrs` is non-null with `len` valid slots (the caller's
        // obligation); each slot is read once, freed, then nulled so a second
        // call over the same array does nothing.
        unsafe {
            let slot = ptrs.add(index);
            let ptr = slot.read();
            if !ptr.is_null() {
                drop(std::ffi::CString::from_raw(ptr));
                slot.write(std::ptr::null_mut());
            }
        }
    }
}
