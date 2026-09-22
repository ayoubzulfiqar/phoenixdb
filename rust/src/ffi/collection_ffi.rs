//! C ABI for document collections ([`crate::collection`]).
//!
//! Same boundary contract as the rest of the ABI: `int32_t` status returns,
//! validation before any dereference, no unwinding, library-owned memory
//! released with `phoenix_string_free`. Failures are described by
//! `phoenix_last_error`.
//!
//! # Wire format
//!
//! Structured data crosses as JSON; embeddings cross as raw `float` arrays,
//! because a JSON round trip of every float would dominate the cost of an
//! insert.
//!
//! * Upsert takes a JSON array of documents
//!   `{"id": str, "text"?: str, "metadata"?: object, "has_vector"?: bool}`
//!   plus one buffer holding, back to back and in document order, `dim`
//!   floats for every document with `"has_vector": true`.
//! * Search takes a request object (see
//!   [`SearchRequest::from_json`](crate::collection::SearchRequest::from_json))
//!   plus an optional query vector, and returns a JSON array of hits.
//! * Documents come back as `{"id", "text"?, "metadata", "vector"?}`.

use super::{give_string, guard, path_arg};
use crate::collection::{Collection, CollectionOptions, Document, Filter, SearchRequest};
use crate::error::Error;
use crate::security::HandleTag;
use crate::vector::{HnswParams, Metric};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::os::raw::{c_char, c_int};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

/// Largest JSON argument accepted, in bytes.
const MAX_JSON_BYTES: usize = 256 << 20;
/// Most floats accepted in one upsert (1 GiB).
const MAX_UPSERT_FLOATS: usize = 1 << 28;

/// Opaque collection handle handed to C.
#[repr(C)]
pub struct PhoenixCollectionHandle {
    tag: HandleTag,
    /// One strong reference (`Arc::into_raw`) to a possibly shared collection.
    collection: *const Collection,
}

/// Collections open in this process, by canonical directory; see the
/// key/value registry in `ffi/mod.rs` for why opens are shared.
static COLLECTION_REGISTRY: Mutex<Vec<(PathBuf, Weak<Collection>)>> =
    parking_lot::const_mutex(Vec::new());

fn open_shared(dir: &Path, options: CollectionOptions) -> Result<Arc<Collection>, Error> {
    let mut registry = COLLECTION_REGISTRY.lock();
    registry.retain(|(_, weak)| weak.strong_count() > 0);
    if let Ok(key) = std::fs::canonicalize(dir)
        && let Some(existing) = registry
            .iter()
            .find(|(p, _)| *p == key)
            .and_then(|(_, weak)| weak.upgrade())
    {
        if options.dim != 0 && options.dim != existing.dim() {
            return Err(Error::invalid(format!(
                "{} is already open with {}-dimensional vectors",
                dir.display(),
                existing.dim()
            )));
        }
        if options.dim != 0 && existing.dim() != 0 && options.metric != existing.metric() {
            return Err(Error::invalid(format!(
                "{} is already open with the {} metric",
                dir.display(),
                existing.metric().name()
            )));
        }
        return Ok(existing);
    }
    let collection = Arc::new(Collection::open(dir, options)?);
    if let Ok(key) = std::fs::canonicalize(dir) {
        registry.push((key, Arc::downgrade(&collection)));
    }
    Ok(collection)
}

impl PhoenixCollectionHandle {
    /// Validates a raw handle pointer and borrows the collection.
    ///
    /// # Safety
    /// `handle` must come from [`phoenix_collection_open`] and not yet have
    /// been passed to [`phoenix_collection_close`].
    unsafe fn validate<'a>(
        handle: *const PhoenixCollectionHandle,
    ) -> Result<&'a Collection, Error> {
        if handle.is_null() {
            return Err(Error::invalid("null collection handle"));
        }
        // SAFETY: non-null; the tag is checked before `collection` is read.
        let h = unsafe { &*handle };
        if !h.tag.is_valid() {
            return Err(Error::invalid(
                "invalid or already-closed collection handle",
            ));
        }
        if h.collection.is_null() {
            return Err(Error::Closed);
        }
        // SAFETY: from `Arc::into_raw`, released only by close, which poisons
        // the tag first.
        Ok(unsafe { &*h.collection })
    }
}

/// Reads an optional NUL-terminated JSON argument (null means absent).
///
/// # Safety
/// `ptr`, when non-null, must point to a NUL-terminated string.
unsafe fn json_arg(ptr: *const c_char, what: &str) -> Result<Option<Value>, Error> {
    if ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: caller guarantees a NUL-terminated string.
    let bytes = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_bytes();
    if bytes.len() > MAX_JSON_BYTES {
        return Err(Error::invalid(format!(
            "{what} exceeds {MAX_JSON_BYTES} bytes"
        )));
    }
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }
    serde_json::from_slice(bytes)
        .map(Some)
        .map_err(|e| Error::invalid(format!("{what} is not valid JSON: {e}")))
}

/// Reads an optional NUL-terminated UTF-8 string argument.
///
/// # Safety
/// `ptr`, when non-null, must point to a NUL-terminated string.
unsafe fn str_arg<'a>(ptr: *const c_char, what: &str) -> Result<Option<&'a str>, Error> {
    if ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: caller guarantees a NUL-terminated string.
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_str()
        .map(Some)
        .map_err(|_| Error::invalid(format!("{what} is not valid UTF-8")))
}

/// Borrows `len` floats (`len == 0` needs no pointer).
///
/// # Safety
/// When `len > 0`, `ptr..ptr+len` must be readable floats for the call.
unsafe fn floats_arg<'a>(ptr: *const f32, len: usize, max: usize) -> Result<&'a [f32], Error> {
    if len == 0 {
        return Ok(&[]);
    }
    if len > max {
        return Err(Error::invalid(format!(
            "{len} floats exceed the limit of {max}"
        )));
    }
    if ptr.is_null() {
        return Err(Error::invalid("null float pointer"));
    }
    if !(ptr as usize).is_multiple_of(std::mem::align_of::<f32>()) {
        return Err(Error::invalid("float pointer is not 4-byte aligned"));
    }
    if (ptr as usize).checked_add(len * 4).is_none() {
        return Err(Error::invalid("float pointer + length overflows"));
    }
    // SAFETY: non-null, aligned, bounded; validity is the caller's obligation.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

fn filter_from(value: Option<Value>) -> Result<Option<Filter>, Error> {
    value.as_ref().map(Filter::parse).transpose()
}

fn document_json(doc: &Document) -> Value {
    let mut out = json!({"id": doc.id, "metadata": doc.metadata});
    if let Some(t) = &doc.text {
        out["text"] = json!(t);
    }
    if let Some(v) = &doc.vector {
        out["vector"] = json!(v);
    }
    out
}

fn options_from(value: Option<&Value>) -> Result<CollectionOptions, Error> {
    let mut options = CollectionOptions::default();
    let Some(value) = value else {
        return Ok(options);
    };
    let Value::Object(map) = value else {
        return Err(Error::invalid("collection options must be a JSON object"));
    };
    let size = |v: &Value, what: &str| -> Result<usize, Error> {
        v.as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::invalid(format!("`{what}` must be a non-negative integer")))
    };
    let flag = |v: &Value, what: &str| -> Result<bool, Error> {
        v.as_bool()
            .ok_or_else(|| Error::invalid(format!("`{what}` must be a boolean")))
    };
    let mut hnsw = HnswParams::default();
    for (key, v) in map {
        match key.as_str() {
            "dim" => options.dim = size(v, "dim")?,
            "metric" => {
                options.metric = match v {
                    Value::String(s) => match s.as_str() {
                        "cosine" => Metric::Cosine,
                        "euclidean" | "l2" => Metric::Euclidean,
                        "dot" | "dot_product" => Metric::DotProduct,
                        other => return Err(Error::invalid(format!("unknown metric `{other}`"))),
                    },
                    other => Metric::from_u8(
                        u8::try_from(size(other, "metric")?)
                            .map_err(|_| Error::invalid("unknown metric"))?,
                    )?,
                }
            }
            "text_index" => options.text_index = flag(v, "text_index")?,
            "sync" => options.sync_on_write = flag(v, "sync")?,
            "m" => {
                hnsw.m = size(v, "m")?;
                hnsw.m_max0 = hnsw.m * 2;
            }
            "ef_construction" => hnsw.ef_construction = size(v, "ef_construction")?,
            "ef_search" => hnsw.ef_search = size(v, "ef_search")?,
            other => {
                return Err(Error::invalid(format!(
                    "unknown collection option `{other}`"
                )));
            }
        }
    }
    options.hnsw = hnsw;
    Ok(options)
}

/// Opens (creating if needed) the collection in directory `path`.
///
/// `options_json` may be null or an object with any of `dim` (0 = no vectors
/// / adopt the existing layout), `metric` (`"cosine"`, `"euclidean"`,
/// `"dot_product"` or 0/1/2), `text_index`, `sync`, `m`, `ef_construction`,
/// `ef_search`. A directory already open in this process is shared.
///
/// # Safety
/// `path` must be a NUL-terminated string, `options_json` null or one, and
/// `out_handle` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_open(
    path: *const c_char,
    options_json: *const c_char,
    out_handle: *mut *mut PhoenixCollectionHandle,
) -> c_int {
    guard(|| {
        if out_handle.is_null() {
            return Err(Error::invalid("out_handle is null"));
        }
        // SAFETY: checked non-null above.
        unsafe { *out_handle = std::ptr::null_mut() };
        // SAFETY: caller guarantees NUL-terminated strings.
        let dir = unsafe { path_arg(path, "path") }?;
        // SAFETY: as above.
        let options = options_from(unsafe { json_arg(options_json, "options_json") }?.as_ref())?;
        let collection = open_shared(&dir, options)?;
        let handle = Box::new(PhoenixCollectionHandle {
            tag: HandleTag::new(),
            collection: Arc::into_raw(collection),
        });
        // SAFETY: validated non-null above.
        unsafe { *out_handle = Box::into_raw(handle) };
        Ok(())
    })
}

/// Releases a handle; the last handle for a directory closes it cleanly.
/// Null and already-closed handles are ignored.
///
/// # Safety
/// `handle` must come from [`phoenix_collection_open`] and not be used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_close(handle: *mut PhoenixCollectionHandle) {
    if handle.is_null() {
        return;
    }
    let _ = guard(|| {
        // SAFETY: non-null; the tag is verified before any other field.
        let h = unsafe { &mut *handle };
        if !h.tag.is_valid() {
            return Err(Error::invalid(
                "invalid or already-closed collection handle",
            ));
        }
        h.tag.poison();
        let ptr = std::mem::replace(&mut h.collection, std::ptr::null());
        if !ptr.is_null() {
            // SAFETY: from `Arc::into_raw`, released exactly once (tag
            // poisoned above). Dropped under the registry lock so a
            // concurrent open never races the final close for the file lock.
            let collection = unsafe { Arc::from_raw(ptr) };
            let _registry = COLLECTION_REGISTRY.lock();
            drop(collection);
        }
        // SAFETY: the box is freed exactly once, here.
        drop(unsafe { Box::from_raw(handle) });
        Ok(())
    });
}

/// Inserts or replaces documents atomically; see the module docs for the
/// wire format. `vectors` may be null when `vectors_len` (a float count) is 0.
///
/// # Safety
/// `docs_json` must be a NUL-terminated string and `vectors` readable for
/// `vectors_len` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_upsert(
    handle: *mut PhoenixCollectionHandle,
    docs_json: *const c_char,
    vectors: *const f32,
    vectors_len: usize,
) -> c_int {
    guard(|| {
        // SAFETY: handle validity is the caller's obligation.
        let c = unsafe { PhoenixCollectionHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string.
        let Some(Value::Array(items)) = (unsafe { json_arg(docs_json, "docs_json") })? else {
            return Err(Error::invalid(
                "docs_json must be a JSON array of documents",
            ));
        };
        // SAFETY: length-checked inside before any read.
        let floats = unsafe { floats_arg(vectors, vectors_len, MAX_UPSERT_FLOATS) }?;
        let dim = c.dim();
        let mut offset = 0usize;
        let mut docs = Vec::with_capacity(items.len());
        for item in items {
            let Value::Object(mut map) = item else {
                return Err(Error::invalid("each document must be a JSON object"));
            };
            let id = match map.remove("id") {
                Some(Value::String(s)) => s,
                _ => return Err(Error::invalid("each document needs a string `id`")),
            };
            let text = match map.remove("text") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s),
                Some(_) => {
                    return Err(Error::invalid(format!(
                        "document `{id}`: `text` must be a string"
                    )));
                }
            };
            let metadata = map.remove("metadata").unwrap_or(Value::Null);
            let has_vector = match map.remove("has_vector") {
                None | Some(Value::Bool(false)) => false,
                Some(Value::Bool(true)) => true,
                Some(_) => {
                    return Err(Error::invalid(format!(
                        "document `{id}`: `has_vector` must be a boolean"
                    )));
                }
            };
            if let Some(extra) = map.keys().next() {
                return Err(Error::invalid(format!(
                    "document `{id}`: unknown field `{extra}`"
                )));
            }
            let vector = if has_vector {
                if dim == 0 {
                    return Err(Error::invalid("this collection has no vectors"));
                }
                let end = offset + dim;
                let Some(slice) = floats.get(offset..end) else {
                    return Err(Error::invalid(format!(
                        "the vector buffer holds {} floats, too few for the documents given",
                        floats.len()
                    )));
                };
                offset = end;
                Some(slice.to_vec())
            } else {
                None
            };
            docs.push(Document {
                id,
                text,
                metadata,
                vector,
            });
        }
        if offset != floats.len() {
            return Err(Error::invalid(format!(
                "the vector buffer holds {} floats but the documents use {offset}",
                floats.len()
            )));
        }
        c.upsert(&docs)
    })
}

/// Deletes documents; `ids_json` is a JSON array of ids. `*out_deleted`
/// (optional) receives how many existed.
///
/// # Safety
/// `ids_json` must be a NUL-terminated string; `out_deleted` null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_delete(
    handle: *mut PhoenixCollectionHandle,
    ids_json: *const c_char,
    out_deleted: *mut u64,
) -> c_int {
    guard(|| {
        if !out_deleted.is_null() {
            // SAFETY: non-null, caller guarantees writable.
            unsafe { *out_deleted = 0 };
        }
        // SAFETY: handle validity is the caller's obligation.
        let c = unsafe { PhoenixCollectionHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string.
        let Some(Value::Array(items)) = (unsafe { json_arg(ids_json, "ids_json") })? else {
            return Err(Error::invalid("ids_json must be a JSON array of strings"));
        };
        let ids = items
            .iter()
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| Error::invalid("ids must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let n = c.delete(&ids)?;
        if !out_deleted.is_null() {
            // SAFETY: as above.
            unsafe { *out_deleted = n };
        }
        Ok(())
    })
}

/// Fetches one document as JSON; `PHOENIX_STATUS_NOT_FOUND` when absent.
///
/// # Safety
/// `id` must be a NUL-terminated string and `out_json` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_get(
    handle: *mut PhoenixCollectionHandle,
    id: *const c_char,
    with_vector: c_int,
    out_json: *mut *mut c_char,
) -> c_int {
    guard(|| {
        if out_json.is_null() {
            return Err(Error::invalid("out_json is null"));
        }
        // SAFETY: checked non-null above.
        unsafe { *out_json = std::ptr::null_mut() };
        // SAFETY: handle validity is the caller's obligation.
        let c = unsafe { PhoenixCollectionHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string.
        let id = unsafe { str_arg(id, "id") }?.ok_or_else(|| Error::invalid("id is null"))?;
        let doc = c.get(id, with_vector != 0)?.ok_or(Error::NotFound)?;
        // SAFETY: `out_json` validated above.
        unsafe { give_string(out_json, document_json(&doc).to_string()) }
    })
}

/// Runs a search. `request_json` is a request object (null = `{}`); the
/// query vector is optional (`query_len == 0`). `*out_json` receives a JSON
/// array of hits `{"id", "score", "vector_score"?, "distance"?,
/// "text_score"?, "text"?, "metadata"?, "vector"?}`.
///
/// # Safety
/// `request_json` null or NUL-terminated; `query` readable for `query_len`
/// floats; `out_json` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_search(
    handle: *mut PhoenixCollectionHandle,
    request_json: *const c_char,
    query: *const f32,
    query_len: usize,
    out_json: *mut *mut c_char,
) -> c_int {
    guard(|| {
        if out_json.is_null() {
            return Err(Error::invalid("out_json is null"));
        }
        // SAFETY: checked non-null above.
        unsafe { *out_json = std::ptr::null_mut() };
        // SAFETY: handle validity is the caller's obligation.
        let c = unsafe { PhoenixCollectionHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string or null.
        let request =
            unsafe { json_arg(request_json, "request_json") }?.unwrap_or_else(|| json!({}));
        let mut req = SearchRequest::from_json(&request)?;
        // SAFETY: length-checked inside before any read.
        let q = unsafe { floats_arg(query, query_len, crate::vector::MAX_DIM) }?;
        if !q.is_empty() {
            req.vector = Some(q.to_vec());
        }
        let hits = c.search(&req)?;
        let text = serde_json::to_string(&hits)
            .map_err(|e| Error::invalid(format!("serialising hits: {e}")))?;
        // SAFETY: `out_json` validated above.
        unsafe { give_string(out_json, text) }
    })
}

/// Counts documents matching `filter_json` (null = all).
///
/// # Safety
/// `filter_json` null or NUL-terminated; `out_count` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_count(
    handle: *mut PhoenixCollectionHandle,
    filter_json: *const c_char,
    out_count: *mut u64,
) -> c_int {
    guard(|| {
        if out_count.is_null() {
            return Err(Error::invalid("out_count is null"));
        }
        // SAFETY: checked non-null above.
        unsafe { *out_count = 0 };
        // SAFETY: handle validity is the caller's obligation.
        let c = unsafe { PhoenixCollectionHandle::validate(handle) }?;
        // SAFETY: caller guarantees a NUL-terminated string or null.
        let filter = filter_from(unsafe { json_arg(filter_json, "filter_json") }?)?;
        let n = c.count(filter.as_ref())?;
        // SAFETY: validated above.
        unsafe { *out_count = n };
        Ok(())
    })
}

/// Lists documents matching `filter_json` (null = all) in id order, after
/// `after` (null = from the start), at most `limit` (0 = all), as a JSON
/// array. Vectors are not included.
///
/// # Safety
/// `filter_json` and `after` null or NUL-terminated; `out_json` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_list(
    handle: *mut PhoenixCollectionHandle,
    filter_json: *const c_char,
    limit: u64,
    after: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    guard(|| {
        if out_json.is_null() {
            return Err(Error::invalid("out_json is null"));
        }
        // SAFETY: checked non-null above.
        unsafe { *out_json = std::ptr::null_mut() };
        // SAFETY: handle validity is the caller's obligation.
        let c = unsafe { PhoenixCollectionHandle::validate(handle) }?;
        // SAFETY: caller guarantees NUL-terminated strings or null.
        let filter = filter_from(unsafe { json_arg(filter_json, "filter_json") }?)?;
        // SAFETY: as above.
        let after = unsafe { str_arg(after, "after") }?;
        let limit = usize::try_from(limit).unwrap_or(usize::MAX);
        let docs = c.list(filter.as_ref(), limit, after)?;
        let out: Vec<Value> = docs.iter().map(document_json).collect();
        // SAFETY: `out_json` validated above.
        unsafe { give_string(out_json, Value::Array(out).to_string()) }
    })
}

/// Writes collection statistics as a JSON object
/// `{"documents", "text_documents", "vectors", "dim", "repaired", "metric"}`.
///
/// # Safety
/// `out_json` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_stats(
    handle: *mut PhoenixCollectionHandle,
    out_json: *mut *mut c_char,
) -> c_int {
    guard(|| {
        if out_json.is_null() {
            return Err(Error::invalid("out_json is null"));
        }
        // SAFETY: checked non-null above.
        unsafe { *out_json = std::ptr::null_mut() };
        // SAFETY: handle validity is the caller's obligation.
        let c = unsafe { PhoenixCollectionHandle::validate(handle) }?;
        let mut stats = serde_json::to_value(c.stats()?)
            .map_err(|e| Error::invalid(format!("serialising stats: {e}")))?;
        stats["metric"] = json!(c.metric().name());
        // SAFETY: `out_json` validated above.
        unsafe { give_string(out_json, stats.to_string()) }
    })
}

/// Syncs vectors and checkpoints documents.
///
/// # Safety
/// `handle` must be a live collection handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_flush(handle: *mut PhoenixCollectionHandle) -> c_int {
    guard(|| {
        // SAFETY: handle validity is the caller's obligation.
        let c = unsafe { PhoenixCollectionHandle::validate(handle) }?;
        c.flush()
    })
}

/// Number of distinct collections open in this process (diagnostics).
#[unsafe(no_mangle)]
pub extern "C" fn phoenix_collection_open_count() -> u32 {
    let mut registry = COLLECTION_REGISTRY.lock();
    registry.retain(|(_, weak)| weak.strong_count() > 0);
    u32::try_from(registry.len()).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};

    fn take(p: *mut c_char) -> String {
        assert!(!p.is_null());
        // SAFETY: a string this library allocated.
        let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string();
        // SAFETY: allocated by `give_string`.
        unsafe { super::super::phoenix_string_free(p) };
        s
    }

    #[test]
    fn round_trip_through_the_c_abi() {
        let dir = tempfile::tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let opts = CString::new(r#"{"dim": 2, "metric": "cosine", "sync": false}"#).unwrap();
        let mut h = std::ptr::null_mut();
        // SAFETY: valid arguments throughout this test.
        unsafe {
            assert_eq!(
                phoenix_collection_open(path.as_ptr(), opts.as_ptr(), &mut h),
                0
            );
            let mut h2 = std::ptr::null_mut();
            assert_eq!(
                phoenix_collection_open(path.as_ptr(), std::ptr::null(), &mut h2),
                0
            );
            assert_eq!(
                phoenix_collection_open_count(),
                1,
                "second open shares the first"
            );
            phoenix_collection_close(h2);

            let docs = CString::new(
                r#"[{"id":"a","text":"rust vector database","metadata":{"k":1},"has_vector":true},
                    {"id":"b","text":"dart bindings","has_vector":true},
                    {"id":"c","text":"no vector here"}]"#,
            )
            .unwrap();
            let floats = [1.0f32, 0.0, 0.0, 1.0];
            assert_eq!(
                phoenix_collection_upsert(h, docs.as_ptr(), floats.as_ptr(), 4),
                0
            );
            assert_ne!(
                phoenix_collection_upsert(h, docs.as_ptr(), floats.as_ptr(), 3),
                0,
                "buffer too short"
            );

            let mut out = std::ptr::null_mut();
            let req =
                CString::new(r#"{"k": 2, "text": "vector", "include": ["metadata"]}"#).unwrap();
            let q = [0.9f32, 0.1];
            assert_eq!(
                phoenix_collection_search(h, req.as_ptr(), q.as_ptr(), 2, &mut out),
                0
            );
            let hits: Value = serde_json::from_str(&take(out)).unwrap();
            assert_eq!(hits[0]["id"], "a");
            assert_eq!(hits[0]["metadata"]["k"], 1);
            assert!(hits[0].get("text").is_none());

            let id = CString::new("a").unwrap();
            assert_eq!(phoenix_collection_get(h, id.as_ptr(), 1, &mut out), 0);
            let doc: Value = serde_json::from_str(&take(out)).unwrap();
            assert_eq!(doc["vector"], json!([1.0, 0.0]));
            let missing = CString::new("zz").unwrap();
            assert_eq!(
                phoenix_collection_get(h, missing.as_ptr(), 0, &mut out),
                crate::PhoenixStatus::NotFound as c_int
            );

            let mut n = 0u64;
            let filter = CString::new(r#"{"k": {"$exists": true}}"#).unwrap();
            assert_eq!(phoenix_collection_count(h, filter.as_ptr(), &mut n), 0);
            assert_eq!(n, 1);
            assert_eq!(phoenix_collection_count(h, std::ptr::null(), &mut n), 0);
            assert_eq!(n, 3);

            let after = CString::new("a").unwrap();
            assert_eq!(
                phoenix_collection_list(h, std::ptr::null(), 1, after.as_ptr(), &mut out),
                0
            );
            let page: Value = serde_json::from_str(&take(out)).unwrap();
            assert_eq!(
                page,
                json!([{"id": "b", "text": "dart bindings", "metadata": null}])
            );

            let ids = CString::new(r#"["a", "nope"]"#).unwrap();
            assert_eq!(phoenix_collection_delete(h, ids.as_ptr(), &mut n), 0);
            assert_eq!(n, 1);
            assert_eq!(phoenix_collection_stats(h, &mut out), 0);
            let stats: Value = serde_json::from_str(&take(out)).unwrap();
            assert_eq!(stats["documents"], 2);
            assert_eq!(stats["vectors"], 1);
            assert_eq!(stats["metric"], "cosine");
            assert_eq!(phoenix_collection_flush(h), 0);

            let bad = CString::new(r#"[{"id":"x","bogus":1}]"#).unwrap();
            assert_ne!(
                phoenix_collection_upsert(h, bad.as_ptr(), std::ptr::null(), 0),
                0
            );
            phoenix_collection_close(h);
            phoenix_collection_close(h); // poisoned tag: ignored, no double free
            assert_eq!(phoenix_collection_open_count(), 0);
            assert_ne!(phoenix_collection_flush(std::ptr::null_mut()), 0);
        }
    }

    #[test]
    fn options_are_validated() {
        assert!(options_from(Some(&json!({"dim": 3, "metric": "l2"}))).is_ok());
        assert!(options_from(Some(&json!({"metric": "manhattan"}))).is_err());
        assert!(options_from(Some(&json!({"dim": -1}))).is_err());
        assert!(options_from(Some(&json!({"surprise": true}))).is_err());
        assert!(options_from(Some(&json!([]))).is_err());
    }
}
