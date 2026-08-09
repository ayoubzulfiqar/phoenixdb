//! Page cache and durable page store.
//!
//! Reads take three tiers, cheapest first:
//!   1. the dirty set (uncommitted in-memory modifications),
//!   2. the [`lru`] clean-page cache,
//!   3. the `mmap` view — a zero-copy borrow verified by CRC before use.
//!
//! Writes are buffered in the dirty set and flushed by [`Pager::flush`], which
//! stamps each page's CRC, writes it positionally, and calls `sync_all` so the
//! data is durable before the WAL is allowed to truncate.

use crate::error::{Error, Result};
use crate::mmap::Mmap;
use crate::page::{MetaData, Page, PageType, PAGE_SIZE, SENTINEL};
use lru::LruCache;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

/// Default clean-page cache capacity (4 MiB worth of pages).
pub const DEFAULT_CACHE_PAGES: usize = 1024;

/// Grow the file by this many pages at a time to amortise `set_len` + remap.
const GROW_CHUNK: u32 = 32;

/// Owns the database file, its mapping, the page cache and the free list.
pub struct Pager {
    path: PathBuf,
    file: File,
    map: Mmap,
    /// Bytes of the file currently covered by `map`.
    mapped_len: usize,
    /// Physical pages the file has been extended to hold.
    file_pages: u32,
    cache: LruCache<u32, Page>,
    dirty: HashMap<u32, Page>,
    meta: MetaData,
}

impl Pager {
    /// Opens (or creates and initialises) the database file at `path`.
    pub fn open(path: &Path, cache_pages: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let len = file.metadata()?.len();
        let capacity = NonZeroUsize::new(cache_pages.max(16)).unwrap_or(
            NonZeroUsize::new(16).expect("16 is non-zero"),
        );

        let mut pager = Pager {
            path: path.to_path_buf(),
            file,
            map: Mmap::empty(),
            mapped_len: 0,
            file_pages: (len / PAGE_SIZE as u64) as u32,
            cache: LruCache::new(capacity),
            dirty: HashMap::new(),
            meta: MetaData::default(),
        };

        if len == 0 {
            pager.initialize()?;
        } else {
            if len % PAGE_SIZE as u64 != 0 {
                return Err(Error::corrupt(format!(
                    "file length {len} is not a multiple of the {PAGE_SIZE}-byte page size"
                )));
            }
            pager.refresh_map()?;
            let meta_page = pager.read_page(0)?;
            pager.meta = meta_page.read_meta()?;
        }
        Ok(pager)
    }

    /// Lays out a brand-new database: meta page 0 plus an empty leaf root.
    fn initialize(&mut self) -> Result<()> {
        self.ensure_file_pages(2)?;
        let meta = MetaData::default();

        let mut meta_page = Page::new(0, PageType::Meta);
        meta_page.write_meta(&meta);
        self.write_page_raw(&mut meta_page)?;

        let mut root = Page::new(meta.root, PageType::Leaf);
        root.set_extra(SENTINEL); // no sibling yet
        self.write_page_raw(&mut root)?;

        self.file.sync_all()?;
        self.meta = meta;
        self.refresh_map()?;
        Ok(())
    }

    /// Current metadata snapshot.
    #[must_use]
    pub fn meta(&self) -> MetaData {
        self.meta
    }

    /// Replaces the metadata snapshot; persisted by the next [`Pager::flush`].
    pub fn set_meta(&mut self, meta: MetaData) {
        self.meta = meta;
    }

    /// Path of the database file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Extends the file so it holds at least `pages` pages, then remaps.
    fn ensure_file_pages(&mut self, pages: u32) -> Result<()> {
        if pages <= self.file_pages {
            return Ok(());
        }
        let target = pages.max(self.file_pages.saturating_add(GROW_CHUNK));
        self.file.set_len(target as u64 * PAGE_SIZE as u64)?;
        self.file_pages = target;
        self.refresh_map()
    }

    /// Re-establishes the mapping over the whole file.
    fn refresh_map(&mut self) -> Result<()> {
        let len = self.file.metadata()?.len() as usize;
        self.map.remap(&self.file, len)?;
        self.mapped_len = len;
        self.file_pages = (len / PAGE_SIZE) as u32;
        Ok(())
    }

    /// Reads a page, verifying its CRC32.
    ///
    /// Returns [`Error::Corruption`] when the checksum does not match, so
    /// corrupt bytes never reach the B+Tree code.
    pub fn read_page(&mut self, page_id: u32) -> Result<Page> {
        if let Some(p) = self.dirty.get(&page_id) {
            return Ok(p.clone());
        }
        if let Some(p) = self.cache.get(&page_id) {
            return Ok(p.clone());
        }
        let offset = page_id as usize * PAGE_SIZE;

        // Tier 3a: zero-copy borrow from the mapping.
        if let Some(bytes) = self.map.slice(offset, PAGE_SIZE) {
            let page = Page::from_bytes(bytes)?;
            self.cache.put(page_id, page.clone());
            return Ok(page);
        }

        // Tier 3b: the page is beyond the mapping (freshly grown file).
        if offset + PAGE_SIZE > self.file.metadata()?.len() as usize {
            return Err(Error::corrupt(format!(
                "page {page_id} is past the end of the database file"
            )));
        }
        let mut buf = vec![0u8; PAGE_SIZE];
        self.file.seek(SeekFrom::Start(offset as u64))?;
        self.file.read_exact(&mut buf)?;
        let page = Page::from_bytes(&buf)?;
        self.cache.put(page_id, page.clone());
        Ok(page)
    }

    /// Stages a page for the next flush.
    pub fn write_page(&mut self, page: Page) {
        let id = page.page_id();
        self.cache.pop(&id);
        self.dirty.insert(id, page);
    }

    /// Immediately writes one page (checksummed) without touching the dirty set.
    fn write_page_raw(&mut self, page: &mut Page) -> Result<()> {
        page.finalize();
        let offset = page.page_id() as u64 * PAGE_SIZE as u64;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(page.as_bytes())?;
        Ok(())
    }

    /// Allocates a page, reusing the free list when possible.
    pub fn allocate_page(&mut self, page_type: PageType) -> Result<Page> {
        if self.meta.free_list != SENTINEL {
            let id = self.meta.free_list;
            let recycled = self.read_page(id)?;
            self.meta.free_list = recycled.extra();
            let page = Page::new(id, page_type);
            self.write_page(page.clone());
            return Ok(page);
        }
        let id = self.meta.page_count;
        if id == SENTINEL {
            return Err(Error::Full("page id space exhausted".into()));
        }
        self.meta.page_count += 1;
        self.ensure_file_pages(self.meta.page_count)?;
        let page = Page::new(id, page_type);
        self.write_page(page.clone());
        Ok(page)
    }

    /// Returns a page to the free list.
    pub fn free_page(&mut self, page_id: u32) -> Result<()> {
        if page_id == 0 {
            return Err(Error::invalid("refusing to free the meta page"));
        }
        let mut page = Page::new(page_id, PageType::Free);
        page.set_extra(self.meta.free_list);
        self.meta.free_list = page_id;
        self.write_page(page);
        Ok(())
    }

    /// Number of pages staged for writing.
    #[must_use]
    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    /// Writes every dirty page plus the meta page and `fsync`s the file.
    ///
    /// On return the tree is durable, which is the precondition for
    /// [`crate::wal::Wal::checkpoint`] to discard log records.
    pub fn flush(&mut self) -> Result<()> {
        if self.dirty.is_empty() {
            return self.flush_meta();
        }
        let max_id = self.dirty.keys().copied().max().unwrap_or(0);
        self.ensure_file_pages(max_id + 1)?;

        let mut ids: Vec<u32> = self.dirty.keys().copied().collect();
        ids.sort_unstable(); // sequential writes beat random ones
        for id in ids {
            if let Some(mut page) = self.dirty.remove(&id) {
                self.write_page_raw(&mut page)?;
                self.cache.put(id, page);
            }
        }
        self.flush_meta()?;
        Ok(())
    }

    /// Writes page 0 and `fsync`s.
    fn flush_meta(&mut self) -> Result<()> {
        let mut meta_page = Page::new(0, PageType::Meta);
        let meta = self.meta;
        meta_page.write_meta(&meta);
        self.write_page_raw(&mut meta_page)?;
        self.file.flush()?;
        self.file.sync_all()?;
        self.cache.put(0, meta_page);
        self.refresh_map()?;
        Ok(())
    }

    /// Drops staged writes (transaction rollback before any flush).
    pub fn discard_dirty(&mut self) {
        self.dirty.clear();
    }

    /// Writes a value across a chain of overflow pages, returning the head id.
    pub fn write_overflow_chain(&mut self, value: &[u8]) -> Result<u32> {
        use crate::page::OVERFLOW_PAYLOAD;
        let chunks: Vec<&[u8]> = value.chunks(OVERFLOW_PAYLOAD).collect();
        let mut next = SENTINEL;
        // Build the chain backwards so each page knows its successor.
        for chunk in chunks.iter().rev() {
            let mut page = self.allocate_page(PageType::Overflow)?;
            page.write_overflow(chunk, next)?;
            next = page.page_id();
            self.write_page(page);
        }
        Ok(next)
    }

    /// Reassembles a value from its overflow chain, guarding against cycles.
    pub fn read_overflow_chain(&mut self, head: u32, total_len: u32) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(total_len as usize);
        let mut cursor = head;
        let mut guard = 0u32;
        let max_links = total_len / crate::page::OVERFLOW_PAYLOAD as u32 + 2;
        while cursor != SENTINEL {
            guard += 1;
            if guard > max_links {
                return Err(Error::corrupt("overflow chain is longer than declared"));
            }
            let page = self.read_page(cursor)?;
            out.extend_from_slice(page.read_overflow()?);
            cursor = page.extra();
        }
        if out.len() != total_len as usize {
            return Err(Error::corrupt(format!(
                "overflow chain yielded {} bytes, expected {total_len}",
                out.len()
            )));
        }
        Ok(out)
    }

    /// Frees every page of an overflow chain.
    pub fn free_overflow_chain(&mut self, head: u32) -> Result<()> {
        let mut cursor = head;
        let mut guard = 0u32;
        while cursor != SENTINEL {
            guard += 1;
            if guard > 1_000_000 {
                return Err(Error::corrupt("overflow chain appears to be cyclic"));
            }
            let next = self.read_page(cursor)?.extra();
            self.free_page(cursor)?;
            cursor = next;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Pager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pager")
            .field("path", &self.path)
            .field("file_pages", &self.file_pages)
            .field("dirty", &self.dirty.len())
            .field("meta", &self.meta)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_pager() -> (tempfile::TempDir, Pager) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.pdb");
        let pager = Pager::open(&path, 64).unwrap();
        (dir, pager)
    }

    #[test]
    fn fresh_db_has_meta_and_root() {
        let (_d, pager) = temp_pager();
        let meta = pager.meta();
        assert_eq!(meta.version, MetaData::VERSION);
        assert_eq!(meta.root, 1);
        assert_eq!(meta.page_count, 2);
    }

    #[test]
    fn write_read_roundtrip_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.pdb");
        {
            let mut pager = Pager::open(&path, 64).unwrap();
            let mut page = pager.allocate_page(PageType::Leaf).unwrap();
            let cell = Page::encode_leaf_cell(b"key", b"value", 5, None);
            page.insert_cell_at(0, &cell).unwrap();
            let id = page.page_id();
            pager.write_page(page);
            pager.flush().unwrap();
            assert_eq!(id, 2);
        }
        let mut pager = Pager::open(&path, 64).unwrap();
        let page = pager.read_page(2).unwrap();
        assert_eq!(page.leaf_cell(0).unwrap().value, b"value");
    }

    #[test]
    fn corrupted_page_is_rejected_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.pdb");
        {
            let mut pager = Pager::open(&path, 64).unwrap();
            let page = pager.allocate_page(PageType::Leaf).unwrap();
            pager.write_page(page);
            pager.flush().unwrap();
        }
        // Flip a bit inside page 2's body.
        {
            let mut f = OpenOptions::new().read(true).write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(2 * PAGE_SIZE as u64 + 100)).unwrap();
            let mut b = [0u8; 1];
            f.read_exact(&mut b).unwrap();
            b[0] ^= 0xFF;
            f.seek(SeekFrom::Start(2 * PAGE_SIZE as u64 + 100)).unwrap();
            f.write_all(&b).unwrap();
            f.sync_all().unwrap();
        }
        let mut pager = Pager::open(&path, 64).unwrap();
        let err = pager.read_page(2).unwrap_err();
        assert!(matches!(err, Error::Corruption(_)), "got {err:?}");
    }

    #[test]
    fn free_list_recycles_pages() {
        let (_d, mut pager) = temp_pager();
        let a = pager.allocate_page(PageType::Leaf).unwrap().page_id();
        pager.flush().unwrap();
        pager.free_page(a).unwrap();
        pager.flush().unwrap();
        let b = pager.allocate_page(PageType::Leaf).unwrap().page_id();
        assert_eq!(a, b, "freed page should be recycled");
    }

    #[test]
    fn overflow_chain_roundtrip() {
        let (_d, mut pager) = temp_pager();
        let value: Vec<u8> = (0..50_000u32).map(|i| (i % 251) as u8).collect();
        let head = pager.write_overflow_chain(&value).unwrap();
        pager.flush().unwrap();
        let read = pager.read_overflow_chain(head, value.len() as u32).unwrap();
        assert_eq!(read, value);
        pager.free_overflow_chain(head).unwrap();
        pager.flush().unwrap();
    }

    #[test]
    fn growth_beyond_initial_map_works() {
        let (_d, mut pager) = temp_pager();
        let mut ids = Vec::new();
        for _ in 0..200 {
            let p = pager.allocate_page(PageType::Leaf).unwrap();
            ids.push(p.page_id());
            pager.write_page(p);
        }
        pager.flush().unwrap();
        for id in ids {
            assert_eq!(pager.read_page(id).unwrap().page_id(), id);
        }
    }
}
