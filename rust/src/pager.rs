//! Page cache and durable page store.
//!
//! Reads take three tiers, cheapest first:
//!   1. the dirty set (modifications staged for the next flush),
//!   2. the [`lru`] clean-page cache,
//!   3. the `mmap` view — a zero-copy borrow verified by CRC before use.
//!
//! Reading needs only `&self`, so any number of readers can share a pager
//! behind a read lock; the cache has its own short-lived mutex.
//!
//! # Atomic flush
//!
//! Writes are buffered in the dirty set and made durable by [`Pager::flush`].
//! Overwriting pages in place is not atomic — a crash can leave some pages
//! new and some old, or tear a single page (including the meta page) — so a
//! flush is journaled:
//!
//! 1. every page image of the flush, the meta page included, is written to
//!    `<db>.journal` with a checksum, and the journal is `fsync`ed;
//! 2. the pages are written in place and the database file is `fsync`ed;
//! 3. the journal is truncated.
//!
//! [`Pager::open`] replays a complete journal (the crash hit step 2) and
//! discards an incomplete one (the crash hit step 1, before the database was
//! touched). Either way the file holds exactly the state of one flush.

use crate::error::{Error, Result};
use crate::fsutil::{lock_exclusive, read_at, sync_parent_dir, write_at};
use crate::mmap::Mmap;
use crate::page::{MetaData, PAGE_SIZE, Page, PageType, SENTINEL};
use lru::LruCache;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Default clean-page cache capacity (4 MiB worth of pages).
pub const DEFAULT_CACHE_PAGES: usize = 1024;

/// Grow the file by this many pages at a time to amortise `set_len` + remap.
const GROW_CHUNK: u32 = 32;

/// Journal header magic: `PHXJRNL1`.
const JOURNAL_MAGIC: u64 = 0x5048_584A_524E_4C31;
/// Journal trailer magic, written last: a journal without it is incomplete.
const JOURNAL_END: u32 = 0x4A45_4E44; // "JEND"
const JOURNAL_VERSION: u32 = 1;
const JOURNAL_HEADER: usize = 32;
const JOURNAL_ENTRY: usize = 8 + PAGE_SIZE;
const JOURNAL_TRAILER: usize = 8;

/// Page-cache counters, for metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Reads served from the dirty set or the clean-page cache.
    pub hits: u64,
    /// Reads that had to decode a page from the mapping or the file.
    pub misses: u64,
    /// Pages currently held by the clean-page cache.
    pub resident: u64,
    /// Pages staged for the next flush.
    pub dirty: u64,
}

/// Owns the database file, its mapping, the page cache and the free list.
pub struct Pager {
    path: PathBuf,
    journal_path: PathBuf,
    file: File,
    map: Mmap,
    file_pages: u32,
    cache: Mutex<LruCache<u32, Page>>,
    dirty: HashMap<u32, Page>,
    meta: MetaData,
    /// Meta changed since the last flush.
    meta_dirty: bool,
    pool: std::sync::Arc<crate::page_pool::PagePool>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl Pager {
    /// Opens (or creates and initialises) the database file at `path`.
    ///
    /// A journal left by a flush that was interrupted by a crash is replayed
    /// first, so the file always reflects exactly one complete flush.
    pub fn open(path: &Path, cache_pages: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        lock_exclusive(&file, path)?;
        let journal_path = Self::journal_path(path);
        recover_journal(&file, &journal_path)?;

        let len = file.metadata()?.len();
        let capacity = NonZeroUsize::new(cache_pages.max(16))
            .unwrap_or(NonZeroUsize::new(16).expect("16 is non-zero"));

        let mut pager = Pager {
            path: path.to_path_buf(),
            journal_path,
            file,
            map: Mmap::empty(),
            file_pages: (len / PAGE_SIZE as u64) as u32,
            cache: Mutex::new(LruCache::new(capacity)),
            dirty: HashMap::new(),
            meta: MetaData::default(),
            meta_dirty: false,
            pool: std::sync::Arc::new(crate::page_pool::PagePool::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
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
            if pager.meta.page_count > pager.file_pages {
                return Err(Error::corrupt(format!(
                    "meta page claims {} pages but the file holds {}",
                    pager.meta.page_count, pager.file_pages
                )));
            }
        }
        Ok(pager)
    }

    /// Path of the flush journal that accompanies a database file.
    #[must_use]
    pub fn journal_path(db_path: &Path) -> PathBuf {
        let mut s = db_path.as_os_str().to_os_string();
        s.push(".journal");
        PathBuf::from(s)
    }

    /// Lays out a brand-new database: meta page 0 plus an empty leaf root.
    fn initialize(&mut self) -> Result<()> {
        self.ensure_file_pages(2)?;
        let meta = MetaData::default();

        let mut meta_page = Page::new(0, PageType::Meta, Some(&self.pool));
        meta_page.write_meta(&meta);
        meta_page.finalize();
        write_at(&self.file, meta_page.as_bytes(), 0)?;

        let mut root = Page::new(meta.root, PageType::Leaf, Some(&self.pool));
        root.set_extra(SENTINEL); // no sibling yet
        root.finalize();
        write_at(
            &self.file,
            root.as_bytes(),
            u64::from(meta.root) * PAGE_SIZE as u64,
        )?;

        self.file.sync_all()?;
        sync_parent_dir(&self.path);
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
        if meta != self.meta {
            self.meta = meta;
            self.meta_dirty = true;
        }
    }

    /// Path of the database file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Pages the file currently holds (allocated or not).
    #[must_use]
    pub fn file_pages(&self) -> u32 {
        self.file_pages
    }

    /// Extends the file so it holds at least `pages` pages, then remaps.
    fn ensure_file_pages(&mut self, pages: u32) -> Result<()> {
        if pages <= self.file_pages {
            return Ok(());
        }
        let target = pages.max(self.file_pages.saturating_add(GROW_CHUNK));
        self.file.set_len(u64::from(target) * PAGE_SIZE as u64)?;
        self.file_pages = target;
        self.refresh_map()
    }

    /// Re-establishes the mapping over the whole file.
    fn refresh_map(&mut self) -> Result<()> {
        let len = self.file.metadata()?.len() as usize;
        self.map.remap(&self.file, len)?;
        self.file_pages = (len / PAGE_SIZE) as u32;
        Ok(())
    }

    /// Reads a page, verifying its CRC32 and that it is the page asked for.
    ///
    /// Returns [`Error::Corruption`] when the checksum does not match, so
    /// corrupt bytes never reach the B+Tree code. Takes `&self`: concurrent
    /// readers only contend on the cache mutex, briefly.
    pub fn read_page(&self, page_id: u32) -> Result<Page> {
        if let Some(p) = self.dirty.get(&page_id) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(p.clone());
        }
        if let Some(p) = self.cache.lock().get(&page_id) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(p.clone());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let offset = page_id as usize * PAGE_SIZE;

        let page = if let Some(bytes) = self.map.slice(offset, PAGE_SIZE) {
            // Zero-copy borrow from the mapping.
            Page::from_bytes(bytes)?
        } else {
            // The page is beyond the mapping.
            if (offset + PAGE_SIZE) as u64 > self.file.metadata()?.len() {
                return Err(Error::corrupt(format!(
                    "page {page_id} is past the end of the database file"
                )));
            }
            let mut buf = vec![0u8; PAGE_SIZE];
            read_at(&self.file, &mut buf, offset as u64)?;
            Page::from_bytes(&buf)?
        };
        if page.page_id() != page_id {
            return Err(Error::corrupt(format!(
                "page {page_id} holds the image of page {} (misdirected write)",
                page.page_id()
            )));
        }
        self.cache.lock().put(page_id, page.clone());
        Ok(page)
    }

    /// Stages a page for the next flush.
    pub fn write_page(&mut self, page: Page) {
        let id = page.page_id();
        self.cache.lock().pop(&id);
        self.dirty.insert(id, page);
    }

    /// Allocates a page, reusing the free list when possible.
    pub fn allocate_page(&mut self, page_type: PageType) -> Result<Page> {
        if self.meta.free_list != SENTINEL {
            let id = self.meta.free_list;
            let recycled = self.read_page(id)?;
            if recycled.page_type()? != PageType::Free {
                return Err(Error::corrupt(format!(
                    "free list points at page {id}, which is a {:?} page",
                    recycled.page_type()?
                )));
            }
            self.meta.free_list = recycled.extra();
            self.meta_dirty = true;
            let page = Page::new(id, page_type, Some(&self.pool));
            self.write_page(page.clone());
            return Ok(page);
        }
        let id = self.meta.page_count;
        if id == SENTINEL {
            return Err(Error::Full("page id space exhausted".into()));
        }
        self.meta.page_count += 1;
        self.meta_dirty = true;
        self.ensure_file_pages(self.meta.page_count)?;
        let page = Page::new(id, page_type, Some(&self.pool));
        self.write_page(page.clone());
        Ok(page)
    }

    /// Returns a page to the free list.
    pub fn free_page(&mut self, page_id: u32) -> Result<()> {
        if page_id == 0 {
            return Err(Error::invalid("refusing to free the meta page"));
        }
        if page_id >= self.meta.page_count {
            return Err(Error::corrupt(format!(
                "cannot free page {page_id}: only {} pages are allocated",
                self.meta.page_count
            )));
        }
        let mut page = Page::new(page_id, PageType::Free, Some(&self.pool));
        page.set_extra(self.meta.free_list);
        self.meta.free_list = page_id;
        self.meta_dirty = true;
        self.write_page(page);
        Ok(())
    }

    /// Number of pages staged for writing.
    #[must_use]
    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    /// Cache and dirty-set counters.
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            resident: self.cache.lock().len() as u64,
            dirty: self.dirty.len() as u64,
        }
    }

    /// Makes every staged page and the metadata durable, atomically.
    ///
    /// On return the tree is durable, which is the precondition for the WAL
    /// to discard log records. A flush with nothing staged is free.
    ///
    /// On error the staged pages stay staged, so a retry (or
    /// [`Pager::discard_dirty`]) is possible; the file itself holds either the
    /// previous flush or — once the journal reached the disk — this one.
    pub fn flush(&mut self) -> Result<()> {
        if self.dirty.is_empty() && !self.meta_dirty {
            return Ok(());
        }
        let mut pages: Vec<Page> = self.dirty.values().cloned().collect();
        let mut meta_page = Page::new(0, PageType::Meta, Some(&self.pool));
        meta_page.write_meta(&self.meta);
        pages.push(meta_page);
        pages.sort_unstable_by_key(Page::page_id); // sequential writes beat random ones
        for page in &mut pages {
            page.finalize();
        }
        let max_id = pages.last().map_or(0, Page::page_id);
        let target_pages = self.file_pages.max(max_id + 1);

        self.write_journal(&pages, target_pages)?;
        self.ensure_file_pages(target_pages)?;
        for page in &pages {
            write_at(
                &self.file,
                page.as_bytes(),
                u64::from(page.page_id()) * PAGE_SIZE as u64,
            )?;
        }
        self.file.sync_all()?;
        self.clear_journal()?;

        self.dirty.clear();
        self.meta_dirty = false;
        let mut cache = self.cache.lock();
        for page in pages {
            cache.put(page.page_id(), page);
        }
        Ok(())
    }

    /// Writes and `fsync`s the journal for one flush.
    fn write_journal(&mut self, pages: &[Page], target_pages: u32) -> Result<()> {
        let existed = self.journal_path.exists();
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.journal_path)?;
        let mut crc = crc32fast::Hasher::new();
        let mut out = BufWriter::with_capacity(1 << 16, &file);
        let mut header = [0u8; JOURNAL_HEADER];
        header[0..8].copy_from_slice(&JOURNAL_MAGIC.to_le_bytes());
        header[8..12].copy_from_slice(&JOURNAL_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&(pages.len() as u32).to_le_bytes());
        header[16..20].copy_from_slice(&target_pages.to_le_bytes());
        crc.update(&header);
        out.write_all(&header)?;
        for page in pages {
            let mut entry_head = [0u8; 8];
            entry_head[0..4].copy_from_slice(&page.page_id().to_le_bytes());
            crc.update(&entry_head);
            crc.update(page.as_bytes());
            out.write_all(&entry_head)?;
            out.write_all(page.as_bytes())?;
        }
        let mut trailer = [0u8; JOURNAL_TRAILER];
        trailer[0..4].copy_from_slice(&crc.finalize().to_le_bytes());
        trailer[4..8].copy_from_slice(&JOURNAL_END.to_le_bytes());
        out.write_all(&trailer)?;
        out.flush()?;
        drop(out);
        file.sync_all()?;
        if !existed {
            // A journal whose directory entry is lost cannot be replayed.
            sync_parent_dir(&self.journal_path);
        }
        Ok(())
    }

    /// Marks the journal as consumed.
    fn clear_journal(&mut self) -> Result<()> {
        let file = OpenOptions::new().write(true).open(&self.journal_path)?;
        file.set_len(0)?;
        file.sync_all()?;
        Ok(())
    }

    /// Drops staged writes and restores `meta`, e.g. after a failed merge, so
    /// the pager describes the last durable state again.
    pub fn discard_dirty(&mut self, meta: MetaData) {
        self.dirty.clear();
        self.meta_dirty = meta != self.meta || self.meta_dirty;
        self.meta = meta;
    }

    /// Atomically replaces the whole database file with the database at `src`.
    ///
    /// `src` must be a PhoenixDB file (its meta page and every allocated page
    /// are CRC-checked first). The copy goes through the flush journal, so a
    /// crash midway leaves either the old database or the new one. Staged
    /// pages are discarded; the caller must reset anything derived from the
    /// old contents (the version store, the WAL).
    pub fn replace_contents(&mut self, src: &Path) -> Result<()> {
        if src == self.path {
            return Err(Error::invalid("cannot restore a database onto itself"));
        }
        let src_file = File::open(src)?;
        let len = src_file.metadata()?.len();
        if len == 0 || len % PAGE_SIZE as u64 != 0 {
            return Err(Error::corrupt(format!(
                "{} is not a PhoenixDB file (length {len})",
                src.display()
            )));
        }
        let pages = u32::try_from(len / PAGE_SIZE as u64)
            .map_err(|_| Error::Full("backup has too many pages".into()))?;
        let mut buf = vec![0u8; PAGE_SIZE];
        read_at(&src_file, &mut buf, 0)?;
        let meta = Page::from_bytes(&buf)?.read_meta()?;
        if meta.page_count > pages || meta.root >= meta.page_count {
            return Err(Error::corrupt("backup meta page is inconsistent"));
        }
        for id in 1..meta.page_count {
            read_at(&src_file, &mut buf, u64::from(id) * PAGE_SIZE as u64)?;
            Page::from_bytes(&buf)?;
        }

        // Journal: every page of the backup, with the backup's exact length.
        let journal = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.journal_path)?;
        {
            let mut crc = crc32fast::Hasher::new();
            let mut out = BufWriter::with_capacity(1 << 16, &journal);
            let mut header = [0u8; JOURNAL_HEADER];
            header[0..8].copy_from_slice(&JOURNAL_MAGIC.to_le_bytes());
            header[8..12].copy_from_slice(&JOURNAL_VERSION.to_le_bytes());
            header[12..16].copy_from_slice(&pages.to_le_bytes());
            header[16..20].copy_from_slice(&pages.to_le_bytes());
            crc.update(&header);
            out.write_all(&header)?;
            for id in 0..pages {
                read_at(&src_file, &mut buf, u64::from(id) * PAGE_SIZE as u64)?;
                let mut entry_head = [0u8; 8];
                entry_head[0..4].copy_from_slice(&id.to_le_bytes());
                crc.update(&entry_head);
                crc.update(&buf);
                out.write_all(&entry_head)?;
                out.write_all(&buf)?;
            }
            let mut trailer = [0u8; JOURNAL_TRAILER];
            trailer[0..4].copy_from_slice(&crc.finalize().to_le_bytes());
            trailer[4..8].copy_from_slice(&JOURNAL_END.to_le_bytes());
            out.write_all(&trailer)?;
            out.flush()?;
        }
        journal.sync_all()?;
        drop(journal);
        sync_parent_dir(&self.journal_path);

        // Unmap before a possible shrink: touching a mapping past EOF faults.
        self.map = Mmap::empty();
        self.dirty.clear();
        self.cache.lock().clear();
        let applied = recover_journal(&self.file, &self.journal_path);
        self.refresh_map()?;
        applied?;
        self.meta = self.read_page(0)?.read_meta()?;
        self.meta_dirty = false;
        Ok(())
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
    pub fn read_overflow_chain(&self, head: u32, total_len: u32) -> Result<Vec<u8>> {
        let max_links = total_len / crate::page::OVERFLOW_PAYLOAD as u32 + 2;
        if max_links > self.meta.page_count {
            return Err(Error::corrupt(format!(
                "overflow value of {total_len} bytes needs more pages than the file has"
            )));
        }
        let mut out = Vec::with_capacity(total_len as usize);
        let mut cursor = head;
        let mut guard = 0u32;
        while cursor != SENTINEL {
            guard += 1;
            if guard > max_links {
                return Err(Error::corrupt("overflow chain is longer than declared"));
            }
            let page = self.read_page(cursor)?;
            out.extend_from_slice(page.read_overflow()?);
            if out.len() > total_len as usize {
                return Err(Error::corrupt(
                    "overflow chain holds more bytes than declared",
                ));
            }
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
            if guard > self.meta.page_count {
                return Err(Error::corrupt("overflow chain appears to be cyclic"));
            }
            let page = self.read_page(cursor)?;
            if page.page_type()? != PageType::Overflow {
                return Err(Error::corrupt(format!(
                    "overflow chain reaches page {cursor}, which is not an overflow page"
                )));
            }
            let next = page.extra();
            self.free_page(cursor)?;
            cursor = next;
        }
        Ok(())
    }
}

/// Replays a complete journal into `file`, then marks it consumed.
///
/// An incomplete or corrupt journal means the crash happened while it was
/// being written — before the database file was touched — so it is simply
/// discarded.
fn recover_journal(file: &File, journal_path: &Path) -> Result<()> {
    let mut journal = match OpenOptions::new().read(true).write(true).open(journal_path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(Error::Io(e)),
    };
    let len = journal.metadata()?.len();
    if len == 0 {
        return Ok(());
    }
    if let Some((target_pages, count)) = validate_journal(&mut journal, len)? {
        let mut entry = vec![0u8; JOURNAL_ENTRY];
        let target_len = u64::from(target_pages) * PAGE_SIZE as u64;
        if file.metadata()?.len() != target_len {
            file.set_len(target_len)?;
        }
        for i in 0..count as u64 {
            read_at(
                &journal,
                &mut entry,
                JOURNAL_HEADER as u64 + i * JOURNAL_ENTRY as u64,
            )?;
            let id = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
            if id >= target_pages {
                return Err(Error::corrupt(format!(
                    "journal entry for page {id} lies past its declared file size"
                )));
            }
            write_at(file, &entry[8..], u64::from(id) * PAGE_SIZE as u64)?;
        }
        file.sync_all()?;
    }
    journal.set_len(0)?;
    journal.sync_all()?;
    Ok(())
}

/// Checks a journal's framing and checksum. Returns `(target_pages, entries)`
/// for a complete journal, `None` for an incomplete or corrupt one.
fn validate_journal(journal: &mut File, len: u64) -> Result<Option<(u32, u32)>> {
    if len < (JOURNAL_HEADER + JOURNAL_TRAILER) as u64 {
        return Ok(None);
    }
    let mut header = [0u8; JOURNAL_HEADER];
    read_at(journal, &mut header, 0)?;
    let magic = u64::from_le_bytes(header[0..8].try_into().expect("8 bytes"));
    let version = u32::from_le_bytes(header[8..12].try_into().expect("4 bytes"));
    let count = u32::from_le_bytes(header[12..16].try_into().expect("4 bytes"));
    let target_pages = u32::from_le_bytes(header[16..20].try_into().expect("4 bytes"));
    if magic != JOURNAL_MAGIC || version != JOURNAL_VERSION {
        return Ok(None);
    }
    let expected =
        (JOURNAL_HEADER + JOURNAL_TRAILER) as u64 + u64::from(count) * JOURNAL_ENTRY as u64;
    if len != expected {
        return Ok(None);
    }
    let mut crc = crc32fast::Hasher::new();
    crc.update(&header);
    let mut remaining = len - (JOURNAL_HEADER + JOURNAL_TRAILER) as u64;
    let mut offset = JOURNAL_HEADER as u64;
    let mut chunk = vec![0u8; 1 << 16];
    while remaining > 0 {
        let n = remaining.min(chunk.len() as u64) as usize;
        read_at(journal, &mut chunk[..n], offset)?;
        crc.update(&chunk[..n]);
        offset += n as u64;
        remaining -= n as u64;
    }
    let mut trailer = [0u8; JOURNAL_TRAILER];
    read_at(journal, &mut trailer, offset)?;
    let stored_crc = u32::from_le_bytes(trailer[0..4].try_into().expect("4 bytes"));
    let end = u32::from_le_bytes(trailer[4..8].try_into().expect("4 bytes"));
    if end != JOURNAL_END || stored_crc != crc.finalize() {
        return Ok(None);
    }
    Ok(Some((target_pages, count)))
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
    use std::io::{Read, Seek, SeekFrom};

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
        let pager = Pager::open(&path, 64).unwrap();
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
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            f.seek(SeekFrom::Start(2 * PAGE_SIZE as u64 + 100)).unwrap();
            let mut b = [0u8; 1];
            f.read_exact(&mut b).unwrap();
            b[0] ^= 0xFF;
            f.seek(SeekFrom::Start(2 * PAGE_SIZE as u64 + 100)).unwrap();
            f.write_all(&b).unwrap();
            f.sync_all().unwrap();
        }
        let pager = Pager::open(&path, 64).unwrap();
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
    fn free_list_pointing_at_a_live_page_is_corruption() {
        let (_d, mut pager) = temp_pager();
        let mut meta = pager.meta();
        meta.free_list = meta.root; // a leaf, not a free page
        pager.set_meta(meta);
        assert!(matches!(
            pager.allocate_page(PageType::Leaf),
            Err(Error::Corruption(_))
        ));
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
    fn absurd_overflow_length_is_rejected_before_allocating() {
        let (_d, mut pager) = temp_pager();
        let head = pager.write_overflow_chain(b"tiny").unwrap();
        assert!(matches!(
            pager.read_overflow_chain(head, u32::MAX),
            Err(Error::Corruption(_))
        ));
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

    #[test]
    fn flush_with_nothing_staged_is_a_no_op() {
        let (dir, mut pager) = temp_pager();
        pager.flush().unwrap();
        let journal = Pager::journal_path(&dir.path().join("t.pdb"));
        assert!(
            !journal.exists(),
            "an idle flush must not even create a journal"
        );
    }

    #[test]
    fn flush_leaves_an_empty_journal() {
        let (dir, mut pager) = temp_pager();
        let p = pager.allocate_page(PageType::Leaf).unwrap();
        pager.write_page(p);
        pager.flush().unwrap();
        let journal = Pager::journal_path(&dir.path().join("t.pdb"));
        assert_eq!(std::fs::metadata(journal).unwrap().len(), 0);
    }

    /// Builds the journal a flush of `pages` would write, without applying it.
    fn journal_bytes(pages: &[Page], target_pages: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&JOURNAL_MAGIC.to_le_bytes());
        out.extend_from_slice(&JOURNAL_VERSION.to_le_bytes());
        out.extend_from_slice(&(pages.len() as u32).to_le_bytes());
        out.extend_from_slice(&target_pages.to_le_bytes());
        out.extend_from_slice(&[0u8; 12]);
        for p in pages {
            out.extend_from_slice(&p.page_id().to_le_bytes());
            out.extend_from_slice(&[0u8; 4]);
            out.extend_from_slice(p.as_bytes());
        }
        let crc = crc32fast::hash(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&JOURNAL_END.to_le_bytes());
        out
    }

    #[test]
    fn complete_journal_is_replayed_on_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.pdb");
        drop(Pager::open(&path, 64).unwrap());

        // A crash after the journal was synced but before page 1 was rewritten.
        let mut leaf = Page::new(1, PageType::Leaf, None);
        leaf.insert_cell_at(0, &Page::encode_leaf_cell(b"k", b"v", 1, None))
            .unwrap();
        leaf.finalize();
        let mut meta_page = Page::new(0, PageType::Meta, None);
        meta_page.write_meta(&MetaData::default());
        meta_page.finalize();
        std::fs::write(
            Pager::journal_path(&path),
            journal_bytes(&[meta_page, leaf], 32),
        )
        .unwrap();

        let pager = Pager::open(&path, 64).unwrap();
        assert_eq!(pager.read_page(1).unwrap().leaf_cell(0).unwrap().key, b"k");
        assert_eq!(
            std::fs::metadata(Pager::journal_path(&path)).unwrap().len(),
            0
        );
    }

    #[test]
    fn torn_journal_is_discarded_and_the_file_is_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.pdb");
        drop(Pager::open(&path, 64).unwrap());
        let before = std::fs::read(&path).unwrap();

        let mut leaf = Page::new(1, PageType::Leaf, None);
        leaf.insert_cell_at(0, &Page::encode_leaf_cell(b"k", b"v", 1, None))
            .unwrap();
        leaf.finalize();
        let mut bytes = journal_bytes(&[leaf], 32);
        bytes.truncate(bytes.len() - 3); // crash while writing the journal
        std::fs::write(Pager::journal_path(&path), bytes).unwrap();

        let pager = Pager::open(&path, 64).unwrap();
        drop(pager);
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn replace_contents_swaps_the_whole_file() {
        let dir = tempfile::tempdir().unwrap();
        let src_path = dir.path().join("src.pdb");
        {
            let mut src = Pager::open(&src_path, 64).unwrap();
            for _ in 0..40 {
                let p = src.allocate_page(PageType::Leaf).unwrap();
                src.write_page(p);
            }
            src.flush().unwrap();
        }
        let mut dst = Pager::open(&dir.path().join("dst.pdb"), 64).unwrap();
        dst.replace_contents(&src_path).unwrap();
        assert_eq!(dst.meta().page_count, 42);
        assert_eq!(dst.read_page(41).unwrap().page_id(), 41);
        assert!(matches!(
            dst.replace_contents(&dir.path().join("dst.pdb")),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn replace_contents_rejects_a_foreign_file() {
        let dir = tempfile::tempdir().unwrap();
        let junk = dir.path().join("junk.bin");
        std::fs::write(&junk, vec![7u8; PAGE_SIZE * 2]).unwrap();
        let mut dst = Pager::open(&dir.path().join("dst.pdb"), 64).unwrap();
        assert!(dst.replace_contents(&junk).is_err());
        assert_eq!(dst.meta().page_count, 2, "a failed restore changes nothing");
    }

    #[test]
    fn second_open_of_the_same_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.pdb");
        let first = Pager::open(&path, 64).unwrap();
        let err = Pager::open(&path, 64).unwrap_err();
        assert!(matches!(err, Error::Busy(_)), "got {err:?}");
        drop(first);
        Pager::open(&path, 64).expect("lock is released when the handle closes");
    }
}
