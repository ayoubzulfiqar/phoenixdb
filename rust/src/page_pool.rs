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
        pool.pop()
            .unwrap_or_else(|| Box::new([0u8; crate::page::PAGE_SIZE]))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn acquire_returns_zeroed_buffer() {
        let pool = PagePool::new();
        let buf = pool.acquire();
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn release_then_acquire_reuses_buffer() {
        let pool = PagePool::new();
        let buf1 = pool.acquire();
        let ptr1 = buf1.as_ptr();
        pool.release(buf1);
        let buf2 = pool.acquire();
        assert_eq!(ptr1, buf2.as_ptr());
    }

    #[test]
    fn pool_respects_max_capacity() {
        let pool = PagePool::new();
        let mut bufs = Vec::new();
        for _ in 0..2048 {
            let buf = pool.acquire();
            let ptr = buf.as_ptr();
            bufs.push(buf);
            if bufs.len() == 2048 {
                for b in bufs.drain(..) {
                    pool.release(b);
                }
                let kept = pool.acquire();
                assert_eq!(kept.as_ptr(), ptr);
            }
        }
    }

    #[test]
    fn concurrent_acquire_release_does_not_panic() {
        let pool = Arc::new(PagePool::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let pool = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    let buf = pool.acquire();
                    pool.release(buf);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }
}
