//! Simple page-buffer recycler to cut allocation pressure.
//!
//! `Page::new` currently allocates `Box::new([0u8; PAGE_SIZE])` on every call;
//! this pool keeps previously-used buffers in a `Vec` and reissues them on
//! demand. The caller must zero or fully reinitialise the page before reuse,
//! which `Page::new` already does.

use std::sync::Mutex;

const DEFAULT_POOL_CAP: usize = 2048;

/// Recycler for `[u8; PAGE_SIZE]` buffers.
pub struct PagePool {
    pool: Mutex<Vec<Box<[u8; crate::page::PAGE_SIZE]>>>,
}

impl PagePool {
    /// Creates a pool with a small default capacity.
    #[must_use]
    pub fn new() -> Self {
        PagePool {
            pool: Mutex::new(Vec::with_capacity(DEFAULT_POOL_CAP)),
        }
    }

    /// Takes a buffer from the pool or allocates a fresh one.
    pub fn acquire(&self) -> Box<[u8; crate::page::PAGE_SIZE]> {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        pool.pop().unwrap_or_else(|| Box::new([0u8; crate::page::PAGE_SIZE]))
    }

    /// Returns a buffer to the pool for later reuse.
    pub fn release(&self, buf: Box<[u8; crate::page::PAGE_SIZE]>) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < 16_384 {
            pool.push(buf);
        }
    }
}

impl Default for PagePool {
    fn default() -> Self {
        Self::new()
    }
}
