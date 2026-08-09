//! Fixed-size 4 KiB slotted pages with CRC32 integrity protection.
//!
//! # On-disk layout
//!
//! Every page is exactly [`PAGE_SIZE`] bytes and begins with a 32-byte header:
//!
//! ```text
//!  offset  size  field
//!   0       4    crc32        CRC32 of bytes [4..4096]
//!   4       4    page_id
//!   8       4    parent_id    SENTINEL when root / not applicable
//!  12       4    extra        leaf: next_leaf | internal: leftmost_child | overflow: next
//!  16       2    num_keys
//!  18       2    cell_start   offset where the cell heap begins (grows downward)
//!  20       1    page_type    0 free, 1 internal, 2 leaf, 3 overflow, 4 meta
//!  21       1    is_leaf      1 for leaf pages, 0 otherwise
//!  22       2    flags
//!  24       8    lsn          WAL sequence number that last modified this page
//! ```
//!
//! The remainder is a classic slotted page: a slot directory of 4-byte
//! `(offset, len)` pairs grows upward from byte 32, while variable-length cells
//! are appended downward from byte 4096. Free space is the gap between them.
//!
//! # Integrity
//!
//! [`Page::finalize`] recomputes the CRC before a page is written, and
//! [`Page::verify`] recomputes it on every read. A mismatch is reported as
//! [`Error::Corruption`] — corrupt data is never handed to the caller.

use crate::error::{Error, Result};

/// Size of every page in bytes.
pub const PAGE_SIZE: usize = 4096;

/// Size of the fixed page header in bytes.
pub const PAGE_HEADER_SIZE: usize = 32;

/// Bytes consumed by one slot-directory entry.
pub const SLOT_SIZE: usize = 4;

/// Sentinel for "no such page".
pub const SENTINEL: u32 = u32::MAX;

/// Largest key the engine can store in a page (structural limit).
///
/// Two keys plus their overhead must always fit in one page so that a split can
/// make progress; 1 KiB keeps the tree fan-out sane.
pub const MAX_KEY_SIZE: usize = 1024;

/// Largest value stored inline in a leaf cell. Anything larger spills into an
/// overflow-page chain.
pub const MAX_INLINE_VALUE: usize = 1024;

/// Payload capacity of a single overflow page.
pub const OVERFLOW_PAYLOAD: usize = PAGE_SIZE - PAGE_HEADER_SIZE - 8;

/// Usable space for slots + cells on a normal page.
pub const USABLE_SPACE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;

// ---- header field offsets -------------------------------------------------
const OFF_CRC: usize = 0;
const OFF_PAGE_ID: usize = 4;
const OFF_PARENT: usize = 8;
const OFF_EXTRA: usize = 12;
const OFF_NUM_KEYS: usize = 16;
const OFF_CELL_START: usize = 18;
const OFF_PAGE_TYPE: usize = 20;
const OFF_IS_LEAF: usize = 21;
const OFF_FLAGS: usize = 22;
const OFF_LSN: usize = 24;

/// Discriminates the four page kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageType {
    /// Page is on the free list.
    Free = 0,
    /// Interior B+Tree node: separator keys and child pointers.
    Internal = 1,
    /// Leaf B+Tree node: keys and values.
    Leaf = 2,
    /// Continuation page of a large value.
    Overflow = 3,
    /// Page 0: database metadata.
    Meta = 4,
}

impl PageType {
    fn from_u8(v: u8) -> Result<Self> {
        Ok(match v {
            0 => PageType::Free,
            1 => PageType::Internal,
            2 => PageType::Leaf,
            3 => PageType::Overflow,
            4 => PageType::Meta,
            other => return Err(Error::corrupt(format!("unknown page type {other}"))),
        })
    }
}

/// One decoded leaf cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafCell {
    /// Full key bytes.
    pub key: Vec<u8>,
    /// Inline value bytes (empty when [`LeafCell::overflow`] is set).
    pub value: Vec<u8>,
    /// Total value length including overflow bytes.
    pub total_len: u32,
    /// First page of the overflow chain, or `None` when the value is inline.
    pub overflow: Option<u32>,
}

/// A single 4 KiB page held in memory.
///
/// The buffer is boxed so pages stay off the stack and can be handed to the
/// page cache without copying.
#[derive(Clone)]
pub struct Page {
    buf: Box<[u8; PAGE_SIZE]>,
}

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("id", &self.page_id())
            .field("type", &self.page_type().ok())
            .field("num_keys", &self.num_keys())
            .field("free", &self.free_space())
            .finish()
    }
}

impl Page {
    /// Allocates a zeroed page and initialises its header.
    #[must_use]
    pub fn new(page_id: u32, page_type: PageType) -> Self {
        let mut p = Page {
            buf: Box::new([0u8; PAGE_SIZE]),
        };
        p.set_page_id(page_id);
        p.set_parent(SENTINEL);
        p.set_extra(SENTINEL);
        p.set_num_keys(0);
        p.set_cell_start(PAGE_SIZE as u16);
        p.set_page_type(page_type);
        p.set_lsn(0);
        p
    }

    /// Wraps an existing 4 KiB buffer **and verifies its CRC**.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != PAGE_SIZE {
            return Err(Error::corrupt(format!(
                "page must be {PAGE_SIZE} bytes, got {}",
                bytes.len()
            )));
        }
        let mut buf = Box::new([0u8; PAGE_SIZE]);
        buf.copy_from_slice(bytes);
        let page = Page { buf };
        page.verify()?;
        page.validate_structure()?;
        Ok(page)
    }

    /// Raw bytes, for writing to disk. Call [`Page::finalize`] first.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.buf
    }

    // ---- header accessors -------------------------------------------------

    fn read_u16(&self, off: usize) -> u16 {
        u16::from_le_bytes([self.buf[off], self.buf[off + 1]])
    }

    fn write_u16(&mut self, off: usize, v: u16) {
        self.buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
    }

    fn read_u32(&self, off: usize) -> u32 {
        u32::from_le_bytes([
            self.buf[off],
            self.buf[off + 1],
            self.buf[off + 2],
            self.buf[off + 3],
        ])
    }

    fn write_u32(&mut self, off: usize, v: u32) {
        self.buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// This page's identifier.
    #[must_use]
    pub fn page_id(&self) -> u32 {
        self.read_u32(OFF_PAGE_ID)
    }

    /// Sets the page identifier.
    pub fn set_page_id(&mut self, v: u32) {
        self.write_u32(OFF_PAGE_ID, v);
    }

    /// Parent page id, or [`SENTINEL`] for the root.
    #[must_use]
    pub fn parent(&self) -> u32 {
        self.read_u32(OFF_PARENT)
    }

    /// Sets the parent page id.
    pub fn set_parent(&mut self, v: u32) {
        self.write_u32(OFF_PARENT, v);
    }

    /// Overloaded pointer: next leaf / leftmost child / next overflow page.
    #[must_use]
    pub fn extra(&self) -> u32 {
        self.read_u32(OFF_EXTRA)
    }

    /// Sets the overloaded pointer field.
    pub fn set_extra(&mut self, v: u32) {
        self.write_u32(OFF_EXTRA, v);
    }

    /// Number of cells stored on this page.
    #[must_use]
    pub fn num_keys(&self) -> u16 {
        self.read_u16(OFF_NUM_KEYS)
    }

    /// Sets the cell count.
    pub fn set_num_keys(&mut self, v: u16) {
        self.write_u16(OFF_NUM_KEYS, v);
    }

    fn cell_start(&self) -> u16 {
        self.read_u16(OFF_CELL_START)
    }

    fn set_cell_start(&mut self, v: u16) {
        self.write_u16(OFF_CELL_START, v);
    }

    /// Page kind, or [`Error::Corruption`] for an unknown discriminant.
    pub fn page_type(&self) -> Result<PageType> {
        PageType::from_u8(self.buf[OFF_PAGE_TYPE])
    }

    /// Sets the page kind and keeps the `is_leaf` byte in sync.
    pub fn set_page_type(&mut self, t: PageType) {
        self.buf[OFF_PAGE_TYPE] = t as u8;
        self.buf[OFF_IS_LEAF] = u8::from(t == PageType::Leaf);
    }

    /// True when this page is a B+Tree leaf.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.buf[OFF_IS_LEAF] == 1
    }

    /// Implementation-defined flag bits.
    #[must_use]
    pub fn flags(&self) -> u16 {
        self.read_u16(OFF_FLAGS)
    }

    /// Sets the flag bits.
    pub fn set_flags(&mut self, v: u16) {
        self.write_u16(OFF_FLAGS, v);
    }

    /// Log sequence number of the last modification.
    #[must_use]
    pub fn lsn(&self) -> u64 {
        u64::from_le_bytes(self.buf[OFF_LSN..OFF_LSN + 8].try_into().unwrap_or([0; 8]))
    }

    /// Records the log sequence number of the last modification.
    pub fn set_lsn(&mut self, v: u64) {
        self.buf[OFF_LSN..OFF_LSN + 8].copy_from_slice(&v.to_le_bytes());
    }

    // ---- integrity --------------------------------------------------------

    /// Computes the CRC32 of everything after the checksum field.
    #[must_use]
    pub fn compute_crc(&self) -> u32 {
        crc32fast::hash(&self.buf[4..])
    }

    /// Stores the current CRC32. Must be called immediately before writing.
    pub fn finalize(&mut self) {
        let crc = self.compute_crc();
        self.write_u32(OFF_CRC, crc);
    }

    /// Checksum recorded in the header.
    #[must_use]
    pub fn stored_crc(&self) -> u32 {
        self.read_u32(OFF_CRC)
    }

    /// Verifies the CRC32, returning [`Error::Corruption`] on mismatch.
    pub fn verify(&self) -> Result<()> {
        let stored = self.stored_crc();
        let actual = self.compute_crc();
        if stored != actual {
            return Err(Error::corrupt(format!(
                "page {} CRC mismatch: stored {stored:#010x}, computed {actual:#010x}",
                self.page_id()
            )));
        }
        Ok(())
    }

    /// Bounds-checks the slot directory so later accessors cannot read outside
    /// the page even if the CRC happens to match attacker-chosen bytes.
    pub fn validate_structure(&self) -> Result<()> {
        let ty = self.page_type()?;
        if matches!(ty, PageType::Overflow | PageType::Meta | PageType::Free) {
            return Ok(());
        }
        let n = self.num_keys() as usize;
        let cell_start = self.cell_start() as usize;
        if !(PAGE_HEADER_SIZE..=PAGE_SIZE).contains(&cell_start) {
            return Err(Error::corrupt(format!(
                "page {}: cell_start {cell_start} out of range",
                self.page_id()
            )));
        }
        if PAGE_HEADER_SIZE + n * SLOT_SIZE > cell_start {
            return Err(Error::corrupt(format!(
                "page {}: slot directory ({n} slots) overlaps cell heap at {cell_start}",
                self.page_id()
            )));
        }
        for i in 0..n {
            let (off, len) = self.slot(i)?;
            if off < cell_start || off + len > PAGE_SIZE {
                return Err(Error::corrupt(format!(
                    "page {}: slot {i} ({off}..{}) escapes the cell heap",
                    self.page_id(),
                    off + len
                )));
            }
        }
        Ok(())
    }

    // ---- slot directory ---------------------------------------------------

    fn slot(&self, index: usize) -> Result<(usize, usize)> {
        if index >= self.num_keys() as usize {
            return Err(Error::corrupt(format!(
                "slot index {index} >= num_keys {}",
                self.num_keys()
            )));
        }
        let base = PAGE_HEADER_SIZE + index * SLOT_SIZE;
        let off = u16::from_le_bytes([self.buf[base], self.buf[base + 1]]) as usize;
        let len = u16::from_le_bytes([self.buf[base + 2], self.buf[base + 3]]) as usize;
        if off < PAGE_HEADER_SIZE || off.saturating_add(len) > PAGE_SIZE {
            return Err(Error::corrupt(format!(
                "page {}: slot {index} points to {off}..{}",
                self.page_id(),
                off + len
            )));
        }
        Ok((off, len))
    }

    fn set_slot(&mut self, index: usize, off: usize, len: usize) {
        let base = PAGE_HEADER_SIZE + index * SLOT_SIZE;
        self.buf[base..base + 2].copy_from_slice(&(off as u16).to_le_bytes());
        self.buf[base + 2..base + 4].copy_from_slice(&(len as u16).to_le_bytes());
    }

    /// Raw bytes of cell `index`.
    pub fn cell(&self, index: usize) -> Result<&[u8]> {
        let (off, len) = self.slot(index)?;
        Ok(&self.buf[off..off + len])
    }

    /// Bytes currently available for a new cell **including** its slot entry.
    #[must_use]
    pub fn free_space(&self) -> usize {
        let dir_end = PAGE_HEADER_SIZE + self.num_keys() as usize * SLOT_SIZE;
        (self.cell_start() as usize).saturating_sub(dir_end)
    }

    /// Bytes occupied by live cells and slots, as a fraction of usable space.
    ///
    /// Used to evaluate the configured fill factors.
    #[must_use]
    pub fn fill_ratio(&self) -> f32 {
        let used = USABLE_SPACE - self.free_space();
        used as f32 / USABLE_SPACE as f32
    }

    /// Inserts raw cell bytes at slot position `index`, shifting later slots.
    ///
    /// Returns [`Error::Full`] when the page cannot accommodate the cell; the
    /// caller is expected to split.
    pub fn insert_cell_at(&mut self, index: usize, cell: &[u8]) -> Result<()> {
        let n = self.num_keys() as usize;
        if index > n {
            return Err(Error::invalid(format!("slot index {index} > num_keys {n}")));
        }
        if cell.len() + SLOT_SIZE > self.free_space() {
            self.compact()?;
            if cell.len() + SLOT_SIZE > self.free_space() {
                return Err(Error::Full(format!(
                    "page {} cannot fit a {}-byte cell",
                    self.page_id(),
                    cell.len()
                )));
            }
        }
        let new_start = self.cell_start() as usize - cell.len();
        self.buf[new_start..new_start + cell.len()].copy_from_slice(cell);
        self.set_cell_start(new_start as u16);

        // Shift slot entries right by one to open a hole at `index`.
        let dir = PAGE_HEADER_SIZE;
        let from = dir + index * SLOT_SIZE;
        let to = dir + n * SLOT_SIZE;
        self.buf.copy_within(from..to, from + SLOT_SIZE);
        self.set_slot(index, new_start, cell.len());
        self.set_num_keys((n + 1) as u16);
        Ok(())
    }

    /// Removes the cell at `index`. Space is reclaimed lazily by [`Page::compact`].
    pub fn remove_cell_at(&mut self, index: usize) -> Result<()> {
        let n = self.num_keys() as usize;
        if index >= n {
            return Err(Error::invalid(format!("slot index {index} >= num_keys {n}")));
        }
        let dir = PAGE_HEADER_SIZE;
        let from = dir + (index + 1) * SLOT_SIZE;
        let to = dir + n * SLOT_SIZE;
        self.buf.copy_within(from..to, dir + index * SLOT_SIZE);
        self.set_num_keys((n - 1) as u16);
        Ok(())
    }

    /// Replaces the cell at `index`, compacting if the new cell is larger.
    pub fn replace_cell_at(&mut self, index: usize, cell: &[u8]) -> Result<()> {
        let (_, len) = self.slot(index)?;
        if cell.len() == len {
            let (off, _) = self.slot(index)?;
            self.buf[off..off + len].copy_from_slice(cell);
            return Ok(());
        }
        self.remove_cell_at(index)?;
        self.insert_cell_at(index, cell)
    }

    /// Rebuilds the cell heap, discarding space left by removed cells.
    pub fn compact(&mut self) -> Result<()> {
        let n = self.num_keys() as usize;
        let mut cells: Vec<Vec<u8>> = Vec::with_capacity(n);
        for i in 0..n {
            cells.push(self.cell(i)?.to_vec());
        }
        let mut cursor = PAGE_SIZE;
        for (i, cell) in cells.iter().enumerate() {
            cursor -= cell.len();
            self.buf[cursor..cursor + cell.len()].copy_from_slice(cell);
            self.set_slot(i, cursor, cell.len());
        }
        self.set_cell_start(cursor as u16);
        Ok(())
    }

    /// Drops every cell, resetting the heap.
    pub fn clear_cells(&mut self) {
        self.set_num_keys(0);
        self.set_cell_start(PAGE_SIZE as u16);
    }

    // ---- leaf cells -------------------------------------------------------
    //
    // Encoding: [key_len u16][flags u8][pad u8][total_len u32][key][payload]
    //   flags bit 0 set  -> payload is a 4-byte overflow page id
    //   flags bit 0 clear -> payload is `total_len` inline value bytes

    /// Serialises a leaf cell.
    #[must_use]
    pub fn encode_leaf_cell(
        key: &[u8],
        value: &[u8],
        total_len: u32,
        overflow: Option<u32>,
    ) -> Vec<u8> {
        let payload_len = if overflow.is_some() { 4 } else { value.len() };
        let mut out = Vec::with_capacity(8 + key.len() + payload_len);
        out.extend_from_slice(&(key.len() as u16).to_le_bytes());
        out.push(u8::from(overflow.is_some()));
        out.push(0);
        out.extend_from_slice(&total_len.to_le_bytes());
        out.extend_from_slice(key);
        match overflow {
            Some(pid) => out.extend_from_slice(&pid.to_le_bytes()),
            None => out.extend_from_slice(value),
        }
        out
    }

    /// Deserialises the leaf cell at `index`.
    pub fn leaf_cell(&self, index: usize) -> Result<LeafCell> {
        let cell = self.cell(index)?;
        if cell.len() < 8 {
            return Err(Error::corrupt("leaf cell shorter than its header"));
        }
        let key_len = u16::from_le_bytes([cell[0], cell[1]]) as usize;
        let has_overflow = cell[2] & 1 == 1;
        let total_len = u32::from_le_bytes([cell[4], cell[5], cell[6], cell[7]]);
        if 8 + key_len > cell.len() {
            return Err(Error::corrupt("leaf cell key length escapes the cell"));
        }
        let key = cell[8..8 + key_len].to_vec();
        let payload = &cell[8 + key_len..];
        if has_overflow {
            if payload.len() < 4 {
                return Err(Error::corrupt("overflow leaf cell missing chain pointer"));
            }
            let pid = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
            Ok(LeafCell {
                key,
                value: Vec::new(),
                total_len,
                overflow: Some(pid),
            })
        } else {
            if payload.len() != total_len as usize {
                return Err(Error::corrupt(format!(
                    "inline value length {} != declared {total_len}",
                    payload.len()
                )));
            }
            Ok(LeafCell {
                key,
                value: payload.to_vec(),
                total_len,
                overflow: None,
            })
        }
    }

    /// Borrows just the key of cell `index` without copying.
    pub fn cell_key(&self, index: usize) -> Result<&[u8]> {
        let (off, len) = self.slot(index)?;
        let cell = &self.buf[off..off + len];
        let (hdr, key_len) = if self.is_leaf() {
            if cell.len() < 8 {
                return Err(Error::corrupt("leaf cell too short"));
            }
            (8usize, u16::from_le_bytes([cell[0], cell[1]]) as usize)
        } else {
            if cell.len() < 6 {
                return Err(Error::corrupt("internal cell too short"));
            }
            (6usize, u16::from_le_bytes([cell[0], cell[1]]) as usize)
        };
        if hdr + key_len > cell.len() {
            return Err(Error::corrupt("cell key escapes the cell"));
        }
        Ok(&cell[hdr..hdr + key_len])
    }

    // ---- internal cells ---------------------------------------------------
    //
    // Encoding: [key_len u16][child u32][key]
    // The leftmost child lives in the header `extra` field.

    /// Serialises a separator key plus its right child pointer.
    #[must_use]
    pub fn encode_internal_cell(key: &[u8], child: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(6 + key.len());
        out.extend_from_slice(&(key.len() as u16).to_le_bytes());
        out.extend_from_slice(&child.to_le_bytes());
        out.extend_from_slice(key);
        out
    }

    /// Child pointer stored in internal cell `index`.
    pub fn internal_child(&self, index: usize) -> Result<u32> {
        let cell = self.cell(index)?;
        if cell.len() < 6 {
            return Err(Error::corrupt("internal cell too short"));
        }
        Ok(u32::from_le_bytes([cell[2], cell[3], cell[4], cell[5]]))
    }

    /// Rewrites the child pointer of internal cell `index` in place.
    pub fn set_internal_child(&mut self, index: usize, child: u32) -> Result<()> {
        let (off, len) = self.slot(index)?;
        if len < 6 {
            return Err(Error::corrupt("internal cell too short"));
        }
        self.buf[off + 2..off + 6].copy_from_slice(&child.to_le_bytes());
        Ok(())
    }

    /// Binary-searches the page for `key`.
    ///
    /// `Ok(i)` means an exact match at slot `i`; `Err(i)` is the insertion point.
    pub fn search(&self, key: &[u8]) -> Result<std::result::Result<usize, usize>> {
        let mut lo = 0usize;
        let mut hi = self.num_keys() as usize;
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.cell_key(mid)?.cmp(key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Ok(Ok(mid)),
            }
        }
        Ok(Err(lo))
    }

    // ---- overflow pages ---------------------------------------------------
    //
    // Body: [payload_len u32][reserved u32][payload bytes...]

    /// Writes one link of an overflow chain.
    pub fn write_overflow(&mut self, payload: &[u8], next: u32) -> Result<()> {
        if payload.len() > OVERFLOW_PAYLOAD {
            return Err(Error::Full(format!(
                "overflow payload {} exceeds {OVERFLOW_PAYLOAD}",
                payload.len()
            )));
        }
        self.set_page_type(PageType::Overflow);
        self.set_extra(next);
        let base = PAGE_HEADER_SIZE;
        self.buf[base..base + 4].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        self.buf[base + 4..base + 8].copy_from_slice(&0u32.to_le_bytes());
        self.buf[base + 8..base + 8 + payload.len()].copy_from_slice(payload);
        Ok(())
    }

    /// Reads the payload of an overflow page.
    pub fn read_overflow(&self) -> Result<&[u8]> {
        if self.page_type()? != PageType::Overflow {
            return Err(Error::corrupt("expected an overflow page"));
        }
        let base = PAGE_HEADER_SIZE;
        let len = u32::from_le_bytes([
            self.buf[base],
            self.buf[base + 1],
            self.buf[base + 2],
            self.buf[base + 3],
        ]) as usize;
        if len > OVERFLOW_PAYLOAD {
            return Err(Error::corrupt(format!(
                "overflow payload length {len} exceeds page capacity"
            )));
        }
        Ok(&self.buf[base + 8..base + 8 + len])
    }

    // ---- meta page --------------------------------------------------------

    /// Writes the page-0 metadata block.
    pub fn write_meta(&mut self, meta: &MetaData) {
        self.set_page_type(PageType::Meta);
        let b = &mut self.buf[PAGE_HEADER_SIZE..];
        b[0..8].copy_from_slice(&MetaData::MAGIC.to_le_bytes());
        b[8..12].copy_from_slice(&meta.version.to_le_bytes());
        b[12..16].copy_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
        b[16..20].copy_from_slice(&meta.root.to_le_bytes());
        b[20..24].copy_from_slice(&meta.free_list.to_le_bytes());
        b[24..28].copy_from_slice(&meta.page_count.to_le_bytes());
        b[28..36].copy_from_slice(&meta.tree_ts.to_le_bytes());
        b[36..44].copy_from_slice(&meta.next_txn_id.to_le_bytes());
        b[44..52].copy_from_slice(&meta.last_lsn.to_le_bytes());
    }

    /// Reads and validates the page-0 metadata block.
    pub fn read_meta(&self) -> Result<MetaData> {
        if self.page_type()? != PageType::Meta {
            return Err(Error::corrupt("page 0 is not a meta page"));
        }
        let b = &self.buf[PAGE_HEADER_SIZE..];
        let magic = u64::from_le_bytes(b[0..8].try_into().map_err(|_| Error::corrupt("meta"))?);
        if !crate::security::ct_eq_u64(magic, MetaData::MAGIC) {
            return Err(Error::corrupt("bad magic: not a PhoenixDB file"));
        }
        let page_size = u32::from_le_bytes(b[12..16].try_into().map_err(|_| Error::corrupt("meta"))?);
        if page_size as usize != PAGE_SIZE {
            return Err(Error::corrupt(format!(
                "file was created with page size {page_size}, this build uses {PAGE_SIZE}"
            )));
        }
        Ok(MetaData {
            version: u32::from_le_bytes(b[8..12].try_into().map_err(|_| Error::corrupt("meta"))?),
            root: u32::from_le_bytes(b[16..20].try_into().map_err(|_| Error::corrupt("meta"))?),
            free_list: u32::from_le_bytes(b[20..24].try_into().map_err(|_| Error::corrupt("meta"))?),
            page_count: u32::from_le_bytes(b[24..28].try_into().map_err(|_| Error::corrupt("meta"))?),
            tree_ts: u64::from_le_bytes(b[28..36].try_into().map_err(|_| Error::corrupt("meta"))?),
            next_txn_id: u64::from_le_bytes(b[36..44].try_into().map_err(|_| Error::corrupt("meta"))?),
            last_lsn: u64::from_le_bytes(b[44..52].try_into().map_err(|_| Error::corrupt("meta"))?),
        })
    }
}

/// Contents of the page-0 metadata block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetaData {
    /// On-disk format version.
    pub version: u32,
    /// Page id of the B+Tree root.
    pub root: u32,
    /// Head of the free-page list, or [`SENTINEL`].
    pub free_list: u32,
    /// Total pages allocated in the file.
    pub page_count: u32,
    /// Every version with `commit_ts <= tree_ts` is merged into the tree.
    pub tree_ts: u64,
    /// Next transaction id to hand out after recovery.
    pub next_txn_id: u64,
    /// Highest LSN durably reflected in the tree.
    pub last_lsn: u64,
}

impl MetaData {
    /// File magic: `PHOENIXDB` truncated to 8 bytes.
    pub const MAGIC: u64 = 0x5048_4F45_4E49_5844;
    /// Current on-disk format version.
    pub const VERSION: u32 = 1;
}

impl Default for MetaData {
    fn default() -> Self {
        MetaData {
            version: Self::VERSION,
            root: 1,
            free_list: SENTINEL,
            page_count: 2,
            tree_ts: 0,
            next_txn_id: 1,
            last_lsn: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let mut p = Page::new(7, PageType::Leaf);
        p.set_parent(3);
        p.set_extra(9);
        p.set_lsn(42);
        p.finalize();
        assert_eq!(p.page_id(), 7);
        assert_eq!(p.parent(), 3);
        assert_eq!(p.extra(), 9);
        assert_eq!(p.lsn(), 42);
        assert!(p.is_leaf());
        assert_eq!(p.page_type().unwrap(), PageType::Leaf);
        p.verify().unwrap();
    }

    #[test]
    fn crc_detects_single_bit_flip() {
        let mut p = Page::new(1, PageType::Leaf);
        let cell = Page::encode_leaf_cell(b"alpha", b"beta", 4, None);
        p.insert_cell_at(0, &cell).unwrap();
        p.finalize();
        let mut bytes = *p.as_bytes();
        bytes[3000] ^= 0x01;
        let err = Page::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, Error::Corruption(_)), "got {err:?}");
    }

    #[test]
    fn crc_detects_header_tamper() {
        let mut p = Page::new(1, PageType::Leaf);
        p.finalize();
        let mut bytes = *p.as_bytes();
        bytes[OFF_PAGE_ID] = 0xFF;
        assert!(Page::from_bytes(&bytes).is_err());
    }

    #[test]
    fn slotted_insert_keeps_order() {
        let mut p = Page::new(1, PageType::Leaf);
        for (i, k) in [b"bbb", b"aaa", b"ccc"].iter().enumerate() {
            let cell = Page::encode_leaf_cell(*k, b"v", 1, None);
            let pos = match p.search(*k).unwrap() {
                Ok(x) | Err(x) => x,
            };
            p.insert_cell_at(pos, &cell).unwrap();
            assert_eq!(p.num_keys() as usize, i + 1);
        }
        assert_eq!(p.cell_key(0).unwrap(), b"aaa");
        assert_eq!(p.cell_key(1).unwrap(), b"bbb");
        assert_eq!(p.cell_key(2).unwrap(), b"ccc");
        assert_eq!(p.search(b"bbb").unwrap(), Ok(1));
        assert_eq!(p.search(b"abc").unwrap(), Err(1));
    }

    #[test]
    fn leaf_cell_roundtrip_inline_and_overflow() {
        let mut p = Page::new(1, PageType::Leaf);
        let c1 = Page::encode_leaf_cell(b"k1", b"hello", 5, None);
        let c2 = Page::encode_leaf_cell(b"k2", &[], 100_000, Some(77));
        p.insert_cell_at(0, &c1).unwrap();
        p.insert_cell_at(1, &c2).unwrap();

        let a = p.leaf_cell(0).unwrap();
        assert_eq!(a.key, b"k1");
        assert_eq!(a.value, b"hello");
        assert_eq!(a.overflow, None);

        let b = p.leaf_cell(1).unwrap();
        assert_eq!(b.key, b"k2");
        assert_eq!(b.total_len, 100_000);
        assert_eq!(b.overflow, Some(77));
    }

    #[test]
    fn remove_then_compact_reclaims_space() {
        let mut p = Page::new(1, PageType::Leaf);
        let payload = vec![0xABu8; 512];
        for i in 0..6u8 {
            let key = [i];
            let cell = Page::encode_leaf_cell(&key, &payload, payload.len() as u32, None);
            p.insert_cell_at(i as usize, &cell).unwrap();
        }
        let before = p.free_space();
        for _ in 0..3 {
            p.remove_cell_at(0).unwrap();
        }
        p.compact().unwrap();
        assert!(p.free_space() > before, "compaction did not reclaim space");
        assert_eq!(p.num_keys(), 3);
        assert_eq!(p.leaf_cell(0).unwrap().key, vec![3]);
    }

    #[test]
    fn full_page_reports_full_not_panic() {
        let mut p = Page::new(1, PageType::Leaf);
        let big = vec![0u8; 1000];
        let mut inserted = 0;
        loop {
            let key = (inserted as u32).to_be_bytes();
            let cell = Page::encode_leaf_cell(&key, &big, big.len() as u32, None);
            match p.insert_cell_at(inserted, &cell) {
                Ok(()) => inserted += 1,
                Err(Error::Full(_)) => break,
                Err(e) => panic!("unexpected {e:?}"),
            }
        }
        assert!(inserted >= 3, "expected at least 3 x 1KiB cells, got {inserted}");
    }

    #[test]
    fn structure_validation_rejects_wild_slots() {
        let mut p = Page::new(1, PageType::Leaf);
        let cell = Page::encode_leaf_cell(b"k", b"v", 1, None);
        p.insert_cell_at(0, &cell).unwrap();
        p.set_num_keys(500); // slot directory would overlap the heap
        p.finalize();
        let bytes = *p.as_bytes();
        assert!(Page::from_bytes(&bytes).is_err());
    }

    #[test]
    fn meta_roundtrip() {
        let meta = MetaData {
            version: 1,
            root: 5,
            free_list: 9,
            page_count: 12,
            tree_ts: 34,
            next_txn_id: 56,
            last_lsn: 78,
        };
        let mut p = Page::new(0, PageType::Meta);
        p.write_meta(&meta);
        p.finalize();
        let bytes = *p.as_bytes();
        let parsed = Page::from_bytes(&bytes).unwrap().read_meta().unwrap();
        assert_eq!(parsed, meta);
    }

    #[test]
    fn overflow_roundtrip() {
        let mut p = Page::new(3, PageType::Overflow);
        let payload = vec![7u8; OVERFLOW_PAYLOAD];
        p.write_overflow(&payload, 4).unwrap();
        p.finalize();
        assert_eq!(p.read_overflow().unwrap(), &payload[..]);
        assert_eq!(p.extra(), 4);
    }
}
