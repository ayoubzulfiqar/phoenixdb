//! C ABI for change notifications ([`crate::watch`]).
//!
//! Same boundary contract as the rest of the ABI: `int32_t` status returns,
//! validation before any dereference, no unwinding, library-owned memory
//! released with `phoenix_buffer_free`. Failures are described by
//! `phoenix_last_error`.
//!
//! # How a caller streams changes
//!
//! [`phoenix_watch_poll`] **blocks** for up to `timeout_ms` waiting for the
//! first change, then returns every buffered change in one buffer. A consumer
//! therefore runs it on its own thread (in Dart: its own isolate) and loops.
//! To stop, another thread calls [`phoenix_watch_wake`], which returns the
//! blocked poll immediately, and then [`phoenix_watch_close`].
//!
//! # Wire format
//!
//! The buffer holds changes back to back, each:
//!
//! ```text
//! [u8 kind][u64 commit_ts][u32 key_len][key bytes][u32 value_len][value bytes]
//! ```
//!
//! `kind` is 1 put, 2 delete, 3 reset. `value_len` is `0xFFFF_FFFF` when there
//! is no value (a delete, a reset, or a subscription that did not ask for
//! values), which is distinct from a present but empty value. All integers are
//! little-endian.

use super::{PhoenixBuffer, PhoenixCollectionHandle, PhoenixDbHandle, guard};
use crate::collection::DocumentWatcher;
use crate::error::Error;
use crate::security::HandleTag;
use crate::watch::{Change, WatchOptions, Watcher};
use std::os::raw::c_int;
use std::time::Duration;

/// Largest buffered-change queue a caller may ask for.
const MAX_WATCH_CAPACITY: u64 = 1 << 20;
/// Longest a single poll may block, so a stuck consumer cannot wedge forever.
const MAX_POLL_MS: u64 = 60_000;
/// Marks "no value" in the wire format.
const NO_VALUE: u32 = u32::MAX;
/// Longest key prefix a subscription may filter on.
const MAX_PREFIX: usize = 1024;

/// A subscription over either raw keys or a collection's documents. Both
/// encode into the same wire format, so one handle type serves both.
enum Sub {
    /// Keys of a key/value database.
    Keys(Watcher),
    /// Document ids of a collection.
    Documents(DocumentWatcher),
}

impl Sub {
    fn poll(&self, timeout: Duration) -> Vec<Change> {
        match self {
            Sub::Keys(watcher) => watcher.poll(timeout),
            // Document ids travel in the key field.
            Sub::Documents(watcher) => watcher
                .poll(timeout)
                .into_iter()
                .map(|change| Change {
                    kind: change.kind,
                    key: change.id.into_bytes(),
                    value: None,
                    commit_ts: change.commit_ts,
                })
                .collect(),
        }
    }

    fn dropped(&self) -> u64 {
        match self {
            Sub::Keys(w) => w.dropped(),
            Sub::Documents(w) => w.dropped(),
        }
    }

    fn wake(&self) {
        match self {
            Sub::Keys(w) => w.wake(),
            Sub::Documents(w) => w.wake(),
        }
    }

    fn is_closed(&self) -> bool {
        match self {
            Sub::Keys(w) => w.is_closed(),
            Sub::Documents(w) => w.is_closed(),
        }
    }
}

/// Opaque watcher handle handed to C.
#[repr(C)]
pub struct PhoenixWatcher {
    tag: HandleTag,
    /// Owned subscription; boxed so the handle is pointer-stable.
    sub: *mut Sub,
}

impl PhoenixWatcher {
    fn new(sub: Sub) -> *mut PhoenixWatcher {
        Box::into_raw(Box::new(PhoenixWatcher {
            tag: HandleTag::new(),
            sub: Box::into_raw(Box::new(sub)),
        }))
    }

    /// Validates a raw handle pointer and borrows the subscription.
    ///
    /// # Safety
    /// `handle` must come from a `phoenix_*_watch_open` and not yet have been
    /// passed to [`phoenix_watch_close`].
    unsafe fn validate<'a>(handle: *const PhoenixWatcher) -> Result<&'a Sub, Error> {
        if handle.is_null() {
            return Err(Error::invalid("null watcher handle"));
        }
        // SAFETY: non-null; the tag is read before `watcher` is touched.
        let h = unsafe { &*handle };
        if !h.tag.is_valid() {
            return Err(Error::invalid("invalid or already-closed watcher handle"));
        }
        if h.sub.is_null() {
            return Err(Error::Closed);
        }
        // SAFETY: from `Box::into_raw`, freed only by `phoenix_watch_close`,
        // which poisons the tag first.
        Ok(unsafe { &*h.sub })
    }
}

fn options(capacity: u64, with_values: c_int) -> Result<WatchOptions, Error> {
    if capacity == 0 || capacity > MAX_WATCH_CAPACITY {
        return Err(Error::invalid(format!(
            "watch capacity must be 1..={MAX_WATCH_CAPACITY}"
        )));
    }
    Ok(WatchOptions {
        capacity: usize::try_from(capacity).unwrap_or(usize::MAX),
        values: with_values != 0,
    })
}

/// Encodes a batch of changes; see the module docs for the layout.
fn encode(changes: &[Change]) -> Vec<u8> {
    let mut out = Vec::with_capacity(changes.len() * 32);
    for change in changes {
        out.push(change.kind as u8);
        out.extend_from_slice(&change.commit_ts.to_le_bytes());
        out.extend_from_slice(&(change.key.len() as u32).to_le_bytes());
        out.extend_from_slice(&change.key);
        match &change.value {
            Some(value) => {
                out.extend_from_slice(&(value.len() as u32).to_le_bytes());
                out.extend_from_slice(value);
            }
            None => out.extend_from_slice(&NO_VALUE.to_le_bytes()),
        }
    }
    out
}

/// Subscribes to committed changes to keys starting with `prefix`.
///
/// * `prefix` — may be null with `prefix_len` 0 to watch everything.
/// * `capacity` — changes buffered before the oldest are dropped.
/// * `with_values` — non-zero to receive written values as well as keys.
///
/// On success `*out_watcher` receives a handle to release with
/// [`phoenix_watch_close`].
///
/// # Safety
/// `prefix` must be readable for `prefix_len` bytes and `out_watcher` must be
/// a writable pointer-sized location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_watch_open(
    handle: *mut PhoenixDbHandle,
    prefix: *const u8,
    prefix_len: usize,
    capacity: u64,
    with_values: c_int,
    out_watcher: *mut *mut PhoenixWatcher,
) -> c_int {
    guard(|| {
        if out_watcher.is_null() {
            return Err(Error::invalid("out_watcher is null"));
        }
        // SAFETY: checked non-null above; always leave the output defined.
        unsafe { *out_watcher = std::ptr::null_mut() };
        // SAFETY: handle validity is the caller's documented obligation.
        let db = unsafe { PhoenixDbHandle::validate(handle) }?;
        let options = options(capacity, with_values)?;
        // SAFETY: validated against the length below before any read.
        let prefix = unsafe { crate::security::slice_from_parts(prefix, prefix_len, MAX_PREFIX) }?;
        let watcher = db.watch(prefix, options);
        // SAFETY: `out_watcher` was validated as non-null above.
        unsafe { *out_watcher = PhoenixWatcher::new(Sub::Keys(watcher)) };
        Ok(())
    })
}

/// Subscribes to a collection's document changes.
///
/// The keys delivered are document ids (UTF-8), and `kind` tells an upsert
/// (put) from a removal (delete). Values are never delivered; read the
/// document if you need its contents.
///
/// # Safety
/// `out_watcher` must be a writable pointer-sized location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_collection_watch_open(
    handle: *mut PhoenixCollectionHandle,
    capacity: u64,
    out_watcher: *mut *mut PhoenixWatcher,
) -> c_int {
    guard(|| {
        if out_watcher.is_null() {
            return Err(Error::invalid("out_watcher is null"));
        }
        // SAFETY: checked non-null above.
        unsafe { *out_watcher = std::ptr::null_mut() };
        // SAFETY: handle validity is the caller's documented obligation.
        let collection = unsafe { PhoenixCollectionHandle::validate(handle) }?;
        let options = options(capacity, 0)?;
        let watcher = collection.watch(options);
        // SAFETY: validated non-null above.
        unsafe { *out_watcher = PhoenixWatcher::new(Sub::Documents(watcher)) };
        Ok(())
    })
}

/// Waits up to `timeout_ms` for changes, then writes every buffered change
/// into `*out` (see the module docs for the encoding).
///
/// An empty buffer means nothing arrived in time, [`phoenix_watch_wake`] was
/// called, or the database closed — check [`phoenix_watch_is_closed`].
/// `*out_dropped` (optional) receives how many changes were lost to a full
/// queue since the last poll, and reading it resets the count.
///
/// Release the buffer with `phoenix_buffer_free`.
///
/// # Safety
/// `out` must be writable for a [`PhoenixBuffer`]; `out_dropped` must be null
/// or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_watch_poll(
    handle: *mut PhoenixWatcher,
    timeout_ms: u64,
    out: *mut PhoenixBuffer,
    out_dropped: *mut u64,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err(Error::invalid("out is null"));
        }
        // SAFETY: checked non-null above; the buffer is always defined, so a
        // caller can free it unconditionally.
        unsafe { *out = PhoenixBuffer::empty() };
        if !out_dropped.is_null() {
            // SAFETY: non-null; the caller guarantees it is writable.
            unsafe { *out_dropped = 0 };
        }
        // SAFETY: handle validity is the caller's documented obligation.
        let watcher = unsafe { PhoenixWatcher::validate(handle) }?;
        let changes = watcher.poll(Duration::from_millis(timeout_ms.min(MAX_POLL_MS)));
        if !out_dropped.is_null() {
            // SAFETY: as above.
            unsafe { *out_dropped = watcher.dropped() };
        }
        if !changes.is_empty() {
            // SAFETY: `out` was validated as non-null above.
            unsafe { *out = PhoenixBuffer::from_vec(encode(&changes)) };
        }
        Ok(())
    })
}

/// Returns a blocked [`phoenix_watch_poll`] immediately, so a consumer thread
/// can shut down without waiting out its timeout. Safe to call from another
/// thread.
///
/// # Safety
/// `handle` must be a live watcher handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_watch_wake(handle: *mut PhoenixWatcher) -> c_int {
    guard(|| {
        // SAFETY: handle validity is the caller's documented obligation.
        let watcher = unsafe { PhoenixWatcher::validate(handle) }?;
        watcher.wake();
        Ok(())
    })
}

/// Writes 1 into `*out_closed` when the watched database has closed, else 0.
///
/// # Safety
/// `out_closed` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_watch_is_closed(
    handle: *mut PhoenixWatcher,
    out_closed: *mut c_int,
) -> c_int {
    guard(|| {
        if out_closed.is_null() {
            return Err(Error::invalid("out_closed is null"));
        }
        // SAFETY: checked non-null above.
        unsafe { *out_closed = 0 };
        // SAFETY: handle validity is the caller's documented obligation.
        let watcher = unsafe { PhoenixWatcher::validate(handle) }?;
        // SAFETY: as above.
        unsafe { *out_closed = c_int::from(watcher.is_closed()) };
        Ok(())
    })
}

/// Unsubscribes and frees the handle. Null and already-closed handles are
/// ignored rather than double-freed.
///
/// A poll in progress on another thread must be returned first with
/// [`phoenix_watch_wake`]; closing under a blocked poll is undefined, exactly
/// as freeing any handle in use is.
///
/// # Safety
/// `handle` must come from a `phoenix_*_watch_open` and must not be used
/// afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phoenix_watch_close(handle: *mut PhoenixWatcher) {
    if handle.is_null() {
        return;
    }
    // A free path must never unwind into the caller.
    let _ = guard(|| {
        // SAFETY: non-null; the tag is verified before any other field.
        let h = unsafe { &mut *handle };
        if !h.tag.is_valid() {
            return Err(Error::invalid("invalid or already-closed watcher handle"));
        }
        h.tag.poison();
        let sub = std::mem::replace(&mut h.sub, std::ptr::null_mut());
        if !sub.is_null() {
            // SAFETY: from `Box::into_raw` in `PhoenixWatcher::new`, released
            // exactly once because the tag is poisoned above. Dropping the
            // subscription unsubscribes it.
            drop(unsafe { Box::from_raw(sub) });
        }
        // The handle struct itself is deliberately *not* freed: it stays
        // allocated as a poisoned tombstone. Freeing it would let the
        // allocator hand the same address to the next open, and a caller's
        // second close would then read a live handle's tag and release it —
        // a double free of someone else's handle. A tombstone costs a few
        // bytes per handle ever opened and makes a double close safe.
        Ok(())
    });
}

/// Decodes the wire format back into changes; the mirror of [`encode`], for
/// tests and for C consumers that want a reference implementation.
#[cfg(test)]
fn decode(bytes: &[u8]) -> Vec<Change> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < bytes.len() {
        let kind = match bytes[at] {
            1 => crate::watch::ChangeKind::Put,
            2 => crate::watch::ChangeKind::Delete,
            _ => crate::watch::ChangeKind::Reset,
        };
        at += 1;
        let commit_ts = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        at += 8;
        let key_len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        let key = bytes[at..at + key_len].to_vec();
        at += key_len;
        let value_len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        at += 4;
        let value = if value_len == NO_VALUE {
            None
        } else {
            let end = at + value_len as usize;
            let value = bytes[at..end].to_vec();
            at = end;
            Some(value)
        };
        out.push(Change {
            kind,
            key,
            value,
            commit_ts,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wire_format_round_trips() {
        let changes = vec![
            Change {
                kind: crate::watch::ChangeKind::Put,
                key: b"user:1".to_vec(),
                value: Some(b"ada".to_vec()),
                commit_ts: 9,
            },
            Change {
                kind: crate::watch::ChangeKind::Put,
                key: b"empty".to_vec(),
                value: Some(Vec::new()),
                commit_ts: 10,
            },
            Change {
                kind: crate::watch::ChangeKind::Delete,
                key: b"gone".to_vec(),
                value: None,
                commit_ts: 11,
            },
            Change {
                kind: crate::watch::ChangeKind::Reset,
                key: Vec::new(),
                value: None,
                commit_ts: 12,
            },
        ];
        let decoded = decode(&encode(&changes));
        assert_eq!(decoded, changes);
        assert_eq!(
            decoded[1].value.as_deref(),
            Some(&[][..]),
            "an empty value is not the same as no value"
        );
    }

    #[test]
    fn capacity_is_validated() {
        assert!(options(0, 0).is_err());
        assert!(options(MAX_WATCH_CAPACITY + 1, 0).is_err());
        let ok = options(16, 1).unwrap();
        assert_eq!(ok.capacity, 16);
        assert!(ok.values);
    }
}
