//! Immutable, sorted on-disk runs produced by flushing a MemTable.
//!
//! # File layout
//!
//! ```text
//!  +-------------------+  offset 0
//!  |  data section     |   entries, ascending (user_key, seqno desc)
//!  +-------------------+
//!  |  bloom filter     |   encoded BloomFilter over distinct user keys
//!  +-------------------+
//!  |  index section    |   sparse: one (key, offset) per RESTART_INTERVAL
//!  +-------------------+
//!  |  footer (68 B)    |   section offsets, counts, CRC32, magic
//!  +-------------------+  EOF
//! ```
//!
//! # Entry encoding
//!
//! ```text
//!  [ key_len u32 ][ val_len u32 ][ seqno u64 ][ kind u8 ][ key ][ value ]
//! ```
//!
//! `kind` is `0` for a value and `1` for a tombstone; a tombstone stores
//! `val_len == 0`. Fixed-width headers keep decoding branch-free, which matters
//! for the vectorised batch path.
//!
//! # Integrity
//!
//! The footer carries a CRC32 over every byte that precedes it. [`SSTable::open`]
//! verifies it once, up front, so later reads can trust the index and bloom
//! offsets without re-checking. Every length read out of the file is bounds-checked
//! against the actual file size *before* it is used to slice or allocate — a
//! truncated or hostile table yields [`Error::Corruption`], never a panic.

use crate::error::{Error, Result};
use crate::lsm::bloom::{BloomFilter, DEFAULT_BITS_PER_KEY};
use crate::lsm::memtable::{InternalKey, Lookup, ValueSlot};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Magic trailer identifying a PhoenixDB SSTable: "PDBSSTBL".
pub const SSTABLE_MAGIC: u64 = 0x5044_4253_5354_424C;

/// Bytes in the fixed-size footer.
///
/// Layout: eight `u64` fields (64 B) + CRC32 (4 B) + magic (8 B).
pub const FOOTER_LEN: usize = 76;

/// Fixed-width prefix of one encoded entry.
const ENTRY_HEADER_LEN: usize = 4 + 4 + 8 + 1;

/// One index entry is emitted every this many data entries.
///
/// A sparse index keeps the whole index resident in memory even for large
/// tables; a lookup binary-searches it, then scans at most this many entries.
pub const RESTART_INTERVAL: usize = 16;

/// Refuse to decode a key or value larger than this (guards a corrupt length).
const MAX_ENTRY_FIELD: u32 = 64 * 1024 * 1024;

/// Discriminates a live value from a tombstone in the on-disk entry header.
const KIND_VALUE: u8 = 0;
/// Marks a deletion; the entry stores no value bytes.
const KIND_TOMBSTONE: u8 = 1;

/// Summary of a table, cached in memory by the level manifest.
///
/// Holding the key range lets a lookup skip an entire table without opening it,
/// and holding the seqno range lets compaction reason about version overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableMeta {
    /// Monotonic table identifier, unique within a database.
    pub id: u64,
    /// LSM level this table belongs to.
    pub level: u32,
    /// Smallest user key present.
    pub min_key: Vec<u8>,
    /// Largest user key present.
    pub max_key: Vec<u8>,
    /// Lowest sequence number present.
    pub min_seqno: u64,
    /// Highest sequence number present.
    pub max_seqno: u64,
    /// Number of versioned entries.
    pub entry_count: u64,
    /// Total file size in bytes.
    pub file_bytes: u64,
}

impl TableMeta {
    /// True when this table's key range intersects `[from, to]`.
    #[must_use]
    pub fn overlaps(&self, from: &[u8], to: &[u8]) -> bool {
        self.min_key.as_slice() <= to && from <= self.max_key.as_slice()
    }

    /// True when `key` falls inside this table's key range.
    #[must_use]
    pub fn may_contain(&self, key: &[u8]) -> bool {
        self.min_key.as_slice() <= key && key <= self.max_key.as_slice()
    }
}

/// Serialises one entry into `out`.
fn encode_entry(out: &mut Vec<u8>, key: &InternalKey, slot: &ValueSlot) {
    let (kind, value): (u8, &[u8]) = match slot {
        Some(v) => (KIND_VALUE, v.as_slice()),
        None => (KIND_TOMBSTONE, &[]),
    };
    out.extend_from_slice(&(key.user_key.len() as u32).to_le_bytes());
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(&key.seqno.to_le_bytes());
    out.push(kind);
    out.extend_from_slice(&key.user_key);
    out.extend_from_slice(value);
}

/// Decodes the entry starting at `pos` in `buf`.
///
/// Returns the entry and the offset just past it. Every length is validated
/// against the remaining buffer before it is used to slice.
fn decode_entry(buf: &[u8], pos: usize) -> Result<(InternalKey, ValueSlot, usize)> {
    let header_end = pos
        .checked_add(ENTRY_HEADER_LEN)
        .ok_or_else(|| Error::corrupt("sstable entry offset overflows"))?;
    if header_end > buf.len() {
        return Err(Error::corrupt(format!(
            "sstable entry at {pos} is truncated: need {ENTRY_HEADER_LEN} header bytes, \
             {} remain",
            buf.len().saturating_sub(pos)
        )));
    }
    let key_len = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]);
    let val_len = u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]);
    if key_len > MAX_ENTRY_FIELD || val_len > MAX_ENTRY_FIELD {
        return Err(Error::corrupt(format!(
            "sstable entry at {pos} declares key_len {key_len} / val_len {val_len}"
        )));
    }
    let seqno = u64::from_le_bytes(
        buf[pos + 8..pos + 16]
            .try_into()
            .map_err(|_| Error::corrupt("sstable seqno truncated"))?,
    );
    let kind = buf[pos + 16];

    let key_start = header_end;
    let key_end = key_start
        .checked_add(key_len as usize)
        .ok_or_else(|| Error::corrupt("sstable key length overflows"))?;
    let val_end = key_end
        .checked_add(val_len as usize)
        .ok_or_else(|| Error::corrupt("sstable value length overflows"))?;
    if val_end > buf.len() {
        return Err(Error::corrupt(format!(
            "sstable entry at {pos} runs past the data section ({val_end} > {})",
            buf.len()
        )));
    }

    let user_key = buf[key_start..key_end].to_vec();
    let slot: ValueSlot = match kind {
        KIND_VALUE => Some(buf[key_end..val_end].to_vec()),
        KIND_TOMBSTONE => None,
        other => {
            return Err(Error::corrupt(format!(
                "sstable entry at {pos} has unknown kind {other}"
            )));
        }
    };
    Ok((InternalKey::new(user_key, seqno), slot, val_end))
}

/// Builds an SSTable file from entries supplied in sorted order.
///
/// The writer streams: entries are serialised straight into a buffered file
/// handle, so flushing a large MemTable never needs a second copy of the data
/// in memory. Only the sparse index and the Bloom filter are accumulated.
pub struct SSTableWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    /// Running CRC over everything written so far.
    hasher: crc32fast::Hasher,
    /// Sparse index: `(user_key, absolute data offset)`.
    index: Vec<(Vec<u8>, u64)>,
    bloom: BloomFilter,
    data_bytes: u64,
    entry_count: u64,
    /// Distinct user keys seen, used to place index restart points.
    distinct_keys: u64,
    last_user_key: Option<Vec<u8>>,
    min_key: Option<Vec<u8>>,
    max_key: Vec<u8>,
    min_seqno: u64,
    max_seqno: u64,
}

impl SSTableWriter {
    /// Creates a writer for a table expected to hold `expected_keys` distinct keys.
    pub fn create(path: impl AsRef<Path>, expected_keys: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(&path)?;
        Ok(SSTableWriter {
            path,
            writer: BufWriter::new(file),
            hasher: crc32fast::Hasher::new(),
            index: Vec::new(),
            bloom: BloomFilter::with_capacity(expected_keys, DEFAULT_BITS_PER_KEY),
            data_bytes: 0,
            entry_count: 0,
            distinct_keys: 0,
            last_user_key: None,
            min_key: None,
            max_key: Vec::new(),
            min_seqno: u64::MAX,
            max_seqno: 0,
        })
    }

    /// Writes `bytes` to the file and folds them into the running CRC.
    fn emit(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes)?;
        self.hasher.update(bytes);
        Ok(())
    }

    /// Appends one entry. Entries **must** arrive in `InternalKey` order.
    ///
    /// Returns [`Error::InvalidArgument`] on an out-of-order key rather than
    /// silently producing a table whose index lies about its contents.
    pub fn append(&mut self, key: &InternalKey, slot: &ValueSlot) -> Result<()> {
        if let Some(last) = &self.last_user_key
            && key.user_key < *last
        {
            return Err(Error::invalid(format!(
                "sstable entries must be sorted: {:?} follows {:?}",
                key.user_key, last
            )));
        }
        let is_new_key = self.last_user_key.as_deref() != Some(key.user_key.as_slice());
        if is_new_key {
            // One index restart point every RESTART_INTERVAL distinct keys.
            if self.distinct_keys % RESTART_INTERVAL as u64 == 0 {
                self.index.push((key.user_key.clone(), self.data_bytes));
            }
            self.bloom.insert(&key.user_key);
            self.distinct_keys += 1;
            self.last_user_key = Some(key.user_key.clone());
        }
        if self.min_key.is_none() {
            self.min_key = Some(key.user_key.clone());
        }
        self.max_key = key.user_key.clone();
        self.min_seqno = self.min_seqno.min(key.seqno);
        self.max_seqno = self.max_seqno.max(key.seqno);

        let mut buf = Vec::with_capacity(ENTRY_HEADER_LEN + key.user_key.len() + 32);
        encode_entry(&mut buf, key, slot);
        self.emit(&buf)?;
        self.data_bytes += buf.len() as u64;
        self.entry_count += 1;
        Ok(())
    }

    /// Finishes the file — bloom, index, footer — and `fsync`s it.
    ///
    /// The `fsync` is what makes a flushed table durable, and must complete
    /// before the WAL segment covering these writes may be discarded.
    pub fn finish(mut self, id: u64, level: u32) -> Result<TableMeta> {
        let bloom_offset = self.data_bytes;
        let bloom_bytes = self.bloom.encode();
        self.emit(&bloom_bytes)?;

        let index_offset = bloom_offset + bloom_bytes.len() as u64;
        let mut index_buf = Vec::new();
        for (key, offset) in &self.index {
            index_buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
            index_buf.extend_from_slice(&offset.to_le_bytes());
            index_buf.extend_from_slice(key);
        }
        self.emit(&index_buf)?;

        // Footer: everything a reader needs to locate the sections, plus the
        // CRC of all preceding bytes and the magic.
        let mut footer = Vec::with_capacity(FOOTER_LEN);
        footer.extend_from_slice(&bloom_offset.to_le_bytes());
        footer.extend_from_slice(&(bloom_bytes.len() as u64).to_le_bytes());
        footer.extend_from_slice(&index_offset.to_le_bytes());
        footer.extend_from_slice(&(index_buf.len() as u64).to_le_bytes());
        footer.extend_from_slice(&(self.index.len() as u64).to_le_bytes());
        footer.extend_from_slice(&self.entry_count.to_le_bytes());
        footer.extend_from_slice(&self.min_seqno.to_le_bytes());
        footer.extend_from_slice(&self.max_seqno.to_le_bytes());
        // CRC covers data + bloom + index (everything before the footer).
        let crc = self.hasher.clone().finalize();
        footer.extend_from_slice(&crc.to_le_bytes());
        footer.extend_from_slice(&SSTABLE_MAGIC.to_le_bytes());
        debug_assert_eq!(footer.len(), FOOTER_LEN, "footer layout drifted");
        self.writer.write_all(&footer)?;

        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;

        let file_bytes = index_offset + index_buf.len() as u64 + FOOTER_LEN as u64;
        Ok(TableMeta {
            id,
            level,
            min_key: self.min_key.unwrap_or_default(),
            max_key: self.max_key,
            min_seqno: if self.entry_count == 0 {
                0
            } else {
                self.min_seqno
            },
            max_seqno: self.max_seqno,
            entry_count: self.entry_count,
            file_bytes,
        })
    }

    /// Path of the file being written.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes of entry data written so far.
    #[must_use]
    pub fn data_bytes(&self) -> u64 {
        self.data_bytes
    }
}

/// Sections located by the footer, after validation.
#[derive(Debug, Clone, Copy)]
struct Footer {
    bloom_offset: u64,
    bloom_len: u64,
    index_offset: u64,
    index_len: u64,
    index_count: u64,
    entry_count: u64,
    min_seqno: u64,
    max_seqno: u64,
    crc: u32,
}

impl Footer {
    /// Parses and sanity-checks the footer against the real file size.
    fn parse(bytes: &[u8], file_len: u64) -> Result<Self> {
        if bytes.len() != FOOTER_LEN {
            return Err(Error::corrupt(format!(
                "sstable footer is {} bytes, expected {FOOTER_LEN}",
                bytes.len()
            )));
        }
        let rd = |off: usize| -> u64 {
            u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap_or([0; 8]))
        };
        // Offsets: 0..64 are the eight u64 fields, 64..68 the CRC, 68..76 the magic.
        let magic = rd(68);
        if magic != SSTABLE_MAGIC {
            return Err(Error::corrupt(format!(
                "not a PhoenixDB SSTable: magic {magic:#018x}"
            )));
        }
        let f = Footer {
            bloom_offset: rd(0),
            bloom_len: rd(8),
            index_offset: rd(16),
            index_len: rd(24),
            index_count: rd(32),
            entry_count: rd(40),
            min_seqno: rd(48),
            max_seqno: rd(56),
            crc: u32::from_le_bytes(bytes[64..68].try_into().unwrap_or([0; 4])),
        };
        // Sections must tile the file exactly, in order, with no overlap and no
        // slice extending past the footer.
        let body_len = file_len
            .checked_sub(FOOTER_LEN as u64)
            .ok_or_else(|| Error::corrupt("sstable shorter than its footer"))?;
        let bloom_end = f
            .bloom_offset
            .checked_add(f.bloom_len)
            .ok_or_else(|| Error::corrupt("sstable bloom section overflows"))?;
        let index_end = f
            .index_offset
            .checked_add(f.index_len)
            .ok_or_else(|| Error::corrupt("sstable index section overflows"))?;
        if bloom_end != f.index_offset || index_end != body_len {
            return Err(Error::corrupt(format!(
                "sstable sections do not tile the file: data..{}, bloom..{bloom_end}, \
                 index..{index_end}, body {body_len}",
                f.bloom_offset
            )));
        }
        Ok(f)
    }
}

/// A read-only handle to an SSTable file.
///
/// The data section is read into memory once at open. Bloom filter and index
/// are decoded eagerly so a point lookup is pure computation — no syscalls.
#[derive(Debug)]
pub struct SSTable {
    meta: TableMeta,
    path: PathBuf,
    /// Entry bytes: `[0, bloom_offset)` of the file.
    data: Vec<u8>,
    bloom: BloomFilter,
    /// Sparse index, ascending by key.
    index: Vec<(Vec<u8>, u64)>,
}

impl SSTable {
    /// Opens and fully validates the table at `path`.
    ///
    /// The CRC over data + bloom + index is verified here, once; a mismatch is
    /// reported as [`Error::Corruption`] and the table is never used.
    pub fn open(path: impl AsRef<Path>, id: u64, level: u32) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        if file_len < FOOTER_LEN as u64 {
            return Err(Error::corrupt(format!(
                "sstable {} is {file_len} bytes, smaller than a footer",
                path.display()
            )));
        }

        let mut footer_buf = [0u8; FOOTER_LEN];
        file.seek(SeekFrom::Start(file_len - FOOTER_LEN as u64))?;
        file.read_exact(&mut footer_buf)?;
        let footer = Footer::parse(&footer_buf, file_len)?;

        // Read the whole body (data + bloom + index) and verify the CRC before
        // trusting any offset inside it.
        let body_len = (file_len - FOOTER_LEN as u64) as usize;
        let mut body = vec![0u8; body_len];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut body)?;
        let actual_crc = crc32fast::hash(&body);
        if actual_crc != footer.crc {
            return Err(Error::corrupt(format!(
                "sstable {} CRC mismatch: stored {:#010x}, computed {actual_crc:#010x}",
                path.display(),
                footer.crc
            )));
        }

        let bloom_start = footer.bloom_offset as usize;
        let index_start = footer.index_offset as usize;
        let bloom = BloomFilter::decode(&body[bloom_start..index_start])?;
        let index = Self::decode_index(&body[index_start..], footer.index_count)?;

        let data = body[..bloom_start].to_vec();
        let (min_key, max_key) = Self::scan_key_bounds(&data)?;

        Ok(SSTable {
            meta: TableMeta {
                id,
                level,
                min_key,
                max_key,
                min_seqno: footer.min_seqno,
                max_seqno: footer.max_seqno,
                entry_count: footer.entry_count,
                file_bytes: file_len,
            },
            path,
            data,
            bloom,
            index,
        })
    }

    /// Parses the sparse index, bounds-checking every entry.
    fn decode_index(buf: &[u8], count: u64) -> Result<Vec<(Vec<u8>, u64)>> {
        let mut index = Vec::with_capacity(count.min(1 << 20) as usize);
        let mut pos = 0usize;
        while pos < buf.len() {
            if pos + 12 > buf.len() {
                return Err(Error::corrupt("sstable index entry header truncated"));
            }
            let key_len =
                u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
            let offset = u64::from_le_bytes(
                buf[pos + 4..pos + 12]
                    .try_into()
                    .map_err(|_| Error::corrupt("sstable index offset truncated"))?,
            );
            let key_start = pos + 12;
            let key_end = key_start
                .checked_add(key_len)
                .ok_or_else(|| Error::corrupt("sstable index key length overflows"))?;
            if key_end > buf.len() {
                return Err(Error::corrupt("sstable index key runs past the section"));
            }
            index.push((buf[key_start..key_end].to_vec(), offset));
            pos = key_end;
        }
        if index.len() as u64 != count {
            return Err(Error::corrupt(format!(
                "sstable index holds {} entries, footer declares {count}",
                index.len()
            )));
        }
        Ok(index)
    }

    /// Walks the data section once to recover the true key bounds.
    fn scan_key_bounds(data: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        if data.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let (first, _, mut pos) = decode_entry(data, 0)?;
        let min = first.user_key;
        let mut max = min.clone();
        while pos < data.len() {
            let (ik, _, next) = decode_entry(data, pos)?;
            max = ik.user_key;
            pos = next;
        }
        Ok((min, max))
    }

    /// Cached metadata for this table.
    #[must_use]
    pub fn meta(&self) -> &TableMeta {
        &self.meta
    }

    /// Path of the backing file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The table's Bloom filter.
    #[must_use]
    pub fn bloom(&self) -> &BloomFilter {
        &self.bloom
    }

    /// Number of entries held.
    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.meta.entry_count
    }

    /// Reads `key` as visible to `snapshot`.
    ///
    /// Fast paths, cheapest first: key range check, then Bloom filter, then a
    /// binary search of the sparse index followed by a bounded linear scan.
    pub fn get(&self, key: &[u8], snapshot: u64) -> Result<Lookup> {
        if self.data.is_empty() || !self.meta.may_contain(key) {
            return Ok(Lookup::Absent);
        }
        if !self.bloom.contains(key) {
            // Definitive: the filter has no false negatives.
            return Ok(Lookup::Absent);
        }
        let mut pos = self.seek_offset(key);
        while pos < self.data.len() {
            let (ik, slot, next) = decode_entry(&self.data, pos)?;
            match ik.user_key.as_slice().cmp(key) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Equal => {
                    // Versions are seqno-descending: first visible one wins.
                    if ik.seqno <= snapshot {
                        return Ok(match slot {
                            Some(v) => Lookup::Found(v),
                            None => Lookup::Deleted,
                        });
                    }
                }
                std::cmp::Ordering::Greater => break, // passed it: absent
            }
            pos = next;
        }
        Ok(Lookup::Absent)
    }

    /// Largest index offset whose key is `<= key`, i.e. where to start scanning.
    fn seek_offset(&self, key: &[u8]) -> usize {
        match self.index.binary_search_by(|(k, _)| k.as_slice().cmp(key)) {
            Ok(i) => self.index[i].1 as usize,
            // `Err(0)` means `key` precedes the first restart point; start at 0.
            Err(0) => 0,
            Err(i) => self.index[i - 1].1 as usize,
        }
    }

    /// Every entry in the table, in stored order.
    pub fn iter(&self) -> SSTableIter<'_> {
        SSTableIter {
            data: &self.data,
            pos: 0,
        }
    }

    /// Collects all entries — used by compaction, which merges whole tables.
    pub fn entries(&self) -> Result<Vec<(InternalKey, ValueSlot)>> {
        self.iter().collect()
    }
}

/// Forward iterator over a table's entries.
pub struct SSTableIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Iterator for SSTableIter<'_> {
    type Item = Result<(InternalKey, ValueSlot)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        match decode_entry(self.data, self.pos) {
            Ok((ik, slot, next)) => {
                self.pos = next;
                Some(Ok((ik, slot)))
            }
            Err(e) => {
                self.pos = self.data.len(); // stop after a decode failure
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test row: `(key, value_or_tombstone, seqno)`.
    type Row<'a> = (&'a [u8], Option<&'a [u8]>, u64);

    /// Builds a table from `(key, value_or_tombstone, seqno)` triples.
    fn build(dir: &Path, name: &str, rows: &[Row<'_>]) -> (PathBuf, TableMeta) {
        let path = dir.join(name);
        let mut w = SSTableWriter::create(&path, rows.len()).unwrap();
        for (k, v, seq) in rows {
            let ik = InternalKey::new(k.to_vec(), *seq);
            let slot: ValueSlot = v.map(<[u8]>::to_vec);
            w.append(&ik, &slot).unwrap();
        }
        let meta = w.finish(1, 0).unwrap();
        (path, meta)
    }

    #[test]
    fn write_then_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let rows: Vec<(Vec<u8>, Option<Vec<u8>>, u64)> = (0..200u32)
            .map(|i| {
                (
                    format!("key{i:05}").into_bytes(),
                    Some(format!("value{i}").into_bytes()),
                    i as u64 + 1,
                )
            })
            .collect();
        let path = dir.path().join("t.sst");
        let mut w = SSTableWriter::create(&path, rows.len()).unwrap();
        for (k, v, s) in &rows {
            w.append(&InternalKey::new(k.clone(), *s), v).unwrap();
        }
        let meta = w.finish(7, 2).unwrap();
        assert_eq!(meta.entry_count, 200);
        assert_eq!(meta.id, 7);
        assert_eq!(meta.level, 2);

        let t = SSTable::open(&path, 7, 2).unwrap();
        assert_eq!(t.entry_count(), 200);
        assert_eq!(t.meta().min_key, b"key00000");
        assert_eq!(t.meta().max_key, b"key00199");
        for (k, v, _) in &rows {
            assert_eq!(
                t.get(k, u64::MAX).unwrap(),
                Lookup::Found(v.clone().unwrap()),
                "lost key {k:?}"
            );
        }
    }

    #[test]
    fn bloom_filter_short_circuits_absent_keys() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = build(
            dir.path(),
            "b.sst",
            &[(b"alpha", Some(b"1"), 1), (b"beta", Some(b"2"), 2)],
        );
        let t = SSTable::open(&path, 1, 0).unwrap();
        // Inside the key range but absent -> the bloom filter must answer.
        assert_eq!(t.get(b"alphb", u64::MAX).unwrap(), Lookup::Absent);
        // Outside the key range -> the range check answers.
        assert_eq!(t.get(b"zzz", u64::MAX).unwrap(), Lookup::Absent);
        assert!(t.bloom().contains(b"alpha"));
    }

    #[test]
    fn snapshot_visibility_and_tombstones() {
        let dir = tempfile::tempdir().unwrap();
        // Versions must be appended newest-first within a key.
        let (path, _) = build(
            dir.path(),
            "v.sst",
            &[
                (b"k", None, 30),
                (b"k", Some(b"v20"), 20),
                (b"k", Some(b"v10"), 10),
            ],
        );
        let t = SSTable::open(&path, 1, 0).unwrap();
        assert_eq!(t.get(b"k", 30).unwrap(), Lookup::Deleted);
        assert_eq!(t.get(b"k", 25).unwrap(), Lookup::Found(b"v20".to_vec()));
        assert_eq!(t.get(b"k", 15).unwrap(), Lookup::Found(b"v10".to_vec()));
        assert_eq!(t.get(b"k", 5).unwrap(), Lookup::Absent);
    }

    #[test]
    fn sparse_index_seeks_correctly_across_restarts() {
        let dir = tempfile::tempdir().unwrap();
        // Far more keys than RESTART_INTERVAL, so the index has many entries.
        let n = RESTART_INTERVAL * 20;
        let path = dir.path().join("i.sst");
        let mut w = SSTableWriter::create(&path, n).unwrap();
        for i in 0..n {
            let k = format!("k{i:06}").into_bytes();
            w.append(&InternalKey::new(k, i as u64 + 1), &Some(vec![b'x'; 8]))
                .unwrap();
        }
        w.finish(1, 0).unwrap();

        let t = SSTable::open(&path, 1, 0).unwrap();
        // Probe every key, including the first and last of each restart block.
        for i in 0..n {
            let k = format!("k{i:06}").into_bytes();
            assert!(
                matches!(t.get(&k, u64::MAX).unwrap(), Lookup::Found(_)),
                "index seek missed {i}"
            );
        }
    }

    #[test]
    fn out_of_order_append_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = SSTableWriter::create(dir.path().join("o.sst"), 4).unwrap();
        w.append(&InternalKey::new(b"b".to_vec(), 1), &Some(b"1".to_vec()))
            .unwrap();
        let err = w.append(&InternalKey::new(b"a".to_vec(), 2), &Some(b"2".to_vec()));
        assert!(matches!(err, Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn corrupted_data_byte_is_caught_by_crc() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = build(
            dir.path(),
            "c.sst",
            &[(b"key", Some(b"value"), 1), (b"key2", Some(b"value2"), 2)],
        );
        SSTable::open(&path, 1, 0).unwrap(); // sane before tampering

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[20] ^= 0xFF; // flip a bit in the data section
        std::fs::write(&path, &bytes).unwrap();

        let err = SSTable::open(&path, 1, 0);
        assert!(
            matches!(err, Err(Error::Corruption(_))),
            "CRC must reject a tampered data section, got {err:?}"
        );
    }

    #[test]
    fn truncated_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = build(dir.path(), "t.sst", &[(b"key", Some(b"value"), 1)]);
        let bytes = std::fs::read(&path).unwrap();

        // Chop the footer in half.
        std::fs::write(&path, &bytes[..bytes.len() - FOOTER_LEN / 2]).unwrap();
        assert!(matches!(
            SSTable::open(&path, 1, 0),
            Err(Error::Corruption(_))
        ));

        // A file shorter than a footer entirely.
        std::fs::write(&path, [0u8; 8]).unwrap();
        assert!(matches!(
            SSTable::open(&path, 1, 0),
            Err(Error::Corruption(_))
        ));
    }

    #[test]
    fn hostile_footer_offsets_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = build(dir.path(), "h.sst", &[(b"key", Some(b"value"), 1)]);
        let mut bytes = std::fs::read(&path).unwrap();
        let footer_at = bytes.len() - FOOTER_LEN;

        // Point the bloom section far beyond EOF.
        bytes[footer_at..footer_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let err = SSTable::open(&path, 1, 0);
        assert!(
            matches!(err, Err(Error::Corruption(_))),
            "a wild bloom offset must not slice out of bounds, got {err:?}"
        );
    }

    #[test]
    fn bad_magic_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = build(dir.path(), "m.sst", &[(b"key", Some(b"v"), 1)]);
        let mut bytes = std::fs::read(&path).unwrap();
        let len = bytes.len();
        bytes[len - 8..].copy_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            SSTable::open(&path, 1, 0),
            Err(Error::Corruption(_))
        ));
    }

    #[test]
    fn empty_table_is_valid_and_answers_absent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("e.sst");
        let w = SSTableWriter::create(&path, 0).unwrap();
        let meta = w.finish(1, 0).unwrap();
        assert_eq!(meta.entry_count, 0);

        let t = SSTable::open(&path, 1, 0).unwrap();
        assert_eq!(t.get(b"anything", u64::MAX).unwrap(), Lookup::Absent);
        assert_eq!(t.entries().unwrap().len(), 0);
    }

    #[test]
    fn iteration_yields_every_entry_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = build(
            dir.path(),
            "it.sst",
            &[
                (b"a", Some(b"1"), 3),
                (b"b", None, 2),
                (b"c", Some(b"3"), 1),
            ],
        );
        let t = SSTable::open(&path, 1, 0).unwrap();
        let entries = t.entries().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].0.user_key, b"a");
        assert_eq!(entries[1].1, None, "tombstone must survive a roundtrip");
        assert_eq!(entries[2].1, Some(b"3".to_vec()));
    }

    #[test]
    fn table_meta_overlap_logic() {
        let m = TableMeta {
            id: 1,
            level: 0,
            min_key: b"d".to_vec(),
            max_key: b"m".to_vec(),
            min_seqno: 1,
            max_seqno: 9,
            entry_count: 5,
            file_bytes: 100,
        };
        assert!(m.overlaps(b"a", b"e"));
        assert!(m.overlaps(b"e", b"f"));
        assert!(m.overlaps(b"l", b"z"));
        assert!(m.overlaps(b"a", b"z"));
        assert!(!m.overlaps(b"a", b"c"));
        assert!(!m.overlaps(b"n", b"z"));
        assert!(m.may_contain(b"d"));
        assert!(m.may_contain(b"m"));
        assert!(!m.may_contain(b"c"));
    }

    #[test]
    fn large_values_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let big = vec![0xABu8; 300_000];
        let path = dir.path().join("big.sst");
        let mut w = SSTableWriter::create(&path, 1).unwrap();
        w.append(&InternalKey::new(b"big".to_vec(), 1), &Some(big.clone()))
            .unwrap();
        w.finish(1, 0).unwrap();

        let t = SSTable::open(&path, 1, 0).unwrap();
        assert_eq!(t.get(b"big", 1).unwrap(), Lookup::Found(big));
    }
}
