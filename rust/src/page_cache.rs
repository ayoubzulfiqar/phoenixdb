//! Bounded `Arc<Page>` cache with safe eviction.
//!
//! This layer lets `Pager` reuse already-decoded pages without copying
//! bytes around, and it caps memory by evicting the least-recently-used
//! entry when the cache is full. Writes invalidate the cached entry so
//! readers never see a stale snapshot.

use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;

use crate::page::Page;

/// Thread-safe page cache keyed by page id.
pub struct PageCache {
    cache: Mutex<LruCache<u32, Arc<Page>>>,
}

impl PageCache {
    /// Creates a cache that holds up to `capacity` pages.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        PageCache {
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(cap).unwrap())),
        }
    }

    /// Returns a cached `Arc<Page>` when present.
    pub fn get(&self, page_id: u32) -> Option<Arc<Page>> {
        self.cache.lock().get(&page_id).cloned()
    }

    /// Caches a page, evicting the LRU entry when the cache is full.
    pub fn put(&self, page_id: u32, page: Arc<Page>) {
        self.cache.lock().put(page_id, page);
    }

    /// Removes a cached entry, if any.
    pub fn pop(&self, page_id: u32) {
        self.cache.lock().pop(&page_id);
    }

    /// Clears every cached entry.
    pub fn clear(&self) {
        self.cache.lock().clear();
    }

    /// Number of cached pages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.lock().len()
    }

    /// True when the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.lock().len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_after_put_returns_arc() {
        let cache = PageCache::new(4);
        let page = Arc::new(Page::new(1, crate::page::PageType::Leaf, None));
        cache.put(1, Arc::clone(&page));
        let got = cache.get(1).unwrap();
        assert!(Arc::ptr_eq(&page, &got));
    }

    #[test]
    fn pop_removes_entry() {
        let cache = PageCache::new(4);
        cache.put(7, Arc::new(Page::new(7, crate::page::PageType::Leaf, None)));
        assert!(cache.get(7).is_some());
        cache.pop(7);
        assert!(cache.get(7).is_none());
    }

    #[test]
    fn capacity_is_respected() {
        let cache = PageCache::new(2);
        cache.put(1, Arc::new(Page::new(1, crate::page::PageType::Leaf, None)));
        cache.put(2, Arc::new(Page::new(2, crate::page::PageType::Leaf, None)));
        cache.put(3, Arc::new(Page::new(3, crate::page::PageType::Leaf, None)));
        // LRU: 1 should be evicted.
        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_some());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn concurrent_puts_and_gets_do_not_panic() {
        use std::thread;

        let cache = Arc::new(PageCache::new(32));
        let mut handles = Vec::new();
        for i in 0..8 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..1000 {
                    let id = (i * 1000 + j) as u32;
                    let page = Arc::new(Page::new(id, crate::page::PageType::Leaf, None));
                    cache.put(id, page);
                    let _ = cache.get(id);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }
}
