//! Durable storage for raw vector bytes.
//!
//! # Layout
//!
//! One append-only file holds a fixed header followed by fixed-stride records:
//!
//! ```text
//! [ header: 64 bytes ]
//! [ record 0 ][ record 1 ] ...
//!
//! header  = magic u64 | version u32 | dim u32 | metric u8 | pad 3 |
//!           count u64 | reserved 40
//! record  = flags u8 | pad 3 | id_len u32 | crc32 u32 | norm f32 |
//!           id bytes (padded to `MAX_ID_LEN`) | dim * f32 little-endian
//! ```
//!
//! # Why a fixed stride
//!
//! Every record is the same size, so record `n` lives at
//! `HEADER_LEN + n * stride` with no index and no scan. Combined with the
//! read-only mmap, a search reads vectors straight out of the page cache —
//! that is the "zero-copy" claim in the FFI docs, and it is only true because
//! the stride is constant.
//!
//! # Durability
//!
//! Appends are WAL-style: the record is written positionally at the end of the
//! file and the header's `count` is only bumped *after* the record bytes are
//! durable. A crash between the two leaves a complete record the header does
//! not yet claim — invisible, and overwritten by the next append. The record's
//! own CRC32 catches a torn write within a record.
//!
//! Deletion is a tombstone flag written in place; the record's bytes stay, so
//! ids in the graph never shift.

use crate::error::{Error, Result};
use crate::mmap::Mmap;
use crate::vector::distance::{Metric, validate_dim};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// "PHNXVEC1" — identifies a vector store and its layout generation.
pub const VECTOR_MAGIC: u64 = 0x5048_4E58_5645_4331;

/// Layout version. Bumped only for an incompatible record change.
pub const VECTOR_FORMAT_VERSION: u32 = 1;

/// Bytes reserved for the header.
pub const HEADER_LEN: usize = 64;

/// Longest external id, in bytes. Ids are padded to this width so the record
/// stride stays constant.
pub const MAX_ID_LEN: usize = 128;

/// Fixed prefix of every record, before the id and the payload.
const RECORD_PREFIX_LEN: usize = 16;

/// Record flag: this entry has been tombstoned.
const FLAG_DELETED: u8 = 0b0000_0001;

/// One decoded record.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorRecord {
    /// Caller-supplied identifier.
    pub id: String,
    /// The vector itself, exactly `dim` components.
    pub vector: Vec<f32>,
    /// Cached L2 norm, so cosine never recomputes it.
    pub norm: f32,
    /// Whether this record is tombstoned.
    pub deleted: bool,
}

/// Append-only, memory-mapped vector file.
pub struct VectorStore {
    path: PathBuf,
    file: File,
    map: Mmap,
    dim: usize,
    metric: Metric,
    stride: usize,
    /// Records the header claims are durable.
    count: usize,
    /// Bytes appended since the last `sync`.
    dirty_bytes: u64,
}

impl VectorStore {
    /// Byte width of one record for `dim` dimensions.
    #[inline]
    #[must_use]
    pub fn stride_for(dim: usize) -> usize {
        RECORD_PREFIX_LEN + MAX_ID_LEN + dim * std::mem::size_of::<f32>()
    }

    /// Opens the store at `path`, creating it with this geometry if absent.
    ///
    /// An existing file whose `dim` or `metric` disagrees with the arguments is
    /// rejected rather than reinterpreted: silently reading 768-float records
    /// as 1536-float ones would produce plausible-looking garbage.
    pub fn open(path: impl AsRef<Path>, dim: usize, metric: Metric) -> Result<Self> {
        validate_dim(dim)?;
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let stride = Self::stride_for(dim);
        let len = file.metadata()?.len();

        let count = if len == 0 {
            Self::write_header(&mut file, dim, metric, 0)?;
            file.sync_all()?;
            0
        } else {
            let (stored_dim, stored_metric, stored_count) = Self::read_header(&mut file)?;
            if stored_dim != dim {
                return Err(Error::invalid(format!(
                    "vector store at {} holds {stored_dim}-dimensional vectors, not {dim}",
                    path.display()
                )));
            }
            if stored_metric != metric {
                return Err(Error::invalid(format!(
                    "vector store at {} was built for the {} metric, not {}",
                    path.display(),
                    stored_metric.name(),
                    metric.name()
                )));
            }
            // Trust the file's length over the header when the header claims
            // more than the file can hold: that is the crash-between-write-
            // and-header case, and the shorter count is the safe reading.
            let capacity = (len as usize).saturating_sub(HEADER_LEN) / stride;
            stored_count.min(capacity)
        };

        let mapped_len = HEADER_LEN + count * stride;
        let map = Mmap::map(&file, mapped_len)?;

        Ok(VectorStore {
            path,
            file,
            map,
            dim,
            metric,
            stride,
            count,
            dirty_bytes: 0,
        })
    }

    /// Path of the backing file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Dimensionality of every stored vector.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Metric this store was created for.
    #[inline]
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Number of durable records, tombstones included.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.count
    }

    /// True when nothing has been appended.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Bytes appended since the last [`VectorStore::sync`].
    #[inline]
    #[must_use]
    pub fn dirty_bytes(&self) -> u64 {
        self.dirty_bytes
    }

    fn write_header(file: &mut File, dim: usize, metric: Metric, count: u64) -> Result<()> {
        let mut header = [0u8; HEADER_LEN];
        header[0..8].copy_from_slice(&VECTOR_MAGIC.to_le_bytes());
        header[8..12].copy_from_slice(&VECTOR_FORMAT_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&(dim as u32).to_le_bytes());
        header[16] = metric.as_u8();
        header[20..28].copy_from_slice(&count.to_le_bytes());
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        Ok(())
    }

    fn read_header(file: &mut File) -> Result<(usize, Metric, usize)> {
        let mut header = [0u8; HEADER_LEN];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut header).map_err(|e| {
            Error::corrupt(format!(
                "vector store header is shorter than {HEADER_LEN} bytes: {e}"
            ))
        })?;

        let magic = u64::from_le_bytes(header[0..8].try_into().unwrap_or_default());
        if magic != VECTOR_MAGIC {
            return Err(Error::corrupt(
                "file is not a PhoenixDB vector store (bad magic)",
            ));
        }
        let version = u32::from_le_bytes(header[8..12].try_into().unwrap_or_default());
        if version != VECTOR_FORMAT_VERSION {
            return Err(Error::corrupt(format!(
                "vector store format version {version} is not supported (expected {VECTOR_FORMAT_VERSION})"
            )));
        }
        let dim = u32::from_le_bytes(header[12..16].try_into().unwrap_or_default()) as usize;
        validate_dim(dim)?;
        let metric = Metric::from_u8(header[16])?;
        let count = u64::from_le_bytes(header[20..28].try_into().unwrap_or_default()) as usize;
        Ok((dim, metric, count))
    }

    /// Appends a record and returns its ordinal, which is also its graph id.
    ///
    /// The bytes are durable when this returns only if `sync_now` is set;
    /// otherwise durability is deferred to [`VectorStore::sync`], which the
    /// engine calls at flush time.
    pub fn append(&mut self, id: &str, vector: &[f32], norm: f32, sync_now: bool) -> Result<u32> {
        if vector.len() != self.dim {
            return Err(Error::invalid(format!(
                "vector has {} dimension(s), store expects {}",
                vector.len(),
                self.dim
            )));
        }
        let id_bytes = id.as_bytes();
        if id_bytes.is_empty() {
            return Err(Error::invalid("vector id must not be empty"));
        }
        if id_bytes.len() > MAX_ID_LEN {
            return Err(Error::invalid(format!(
                "vector id of {} bytes exceeds the {MAX_ID_LEN}-byte limit",
                id_bytes.len()
            )));
        }
        let ordinal = u32::try_from(self.count)
            .map_err(|_| Error::Full("vector store exceeded 2^32 records".to_string()))?;

        let mut record = vec![0u8; self.stride];
        record[4..8].copy_from_slice(&(id_bytes.len() as u32).to_le_bytes());
        record[12..16].copy_from_slice(&norm.to_le_bytes());
        record[RECORD_PREFIX_LEN..RECORD_PREFIX_LEN + id_bytes.len()].copy_from_slice(id_bytes);
        let payload_start = RECORD_PREFIX_LEN + MAX_ID_LEN;
        for (index, value) in vector.iter().enumerate() {
            let at = payload_start + index * 4;
            record[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        // CRC covers everything after the checksum field, so a torn write in
        // either the id or the payload is caught on read.
        let crc = crc32fast::hash(&record[12..]);
        record[8..12].copy_from_slice(&crc.to_le_bytes());

        let offset = (HEADER_LEN + self.count * self.stride) as u64;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(&record)?;
        self.dirty_bytes += self.stride as u64;

        // Record bytes first, then the header: a crash in between leaves an
        // orphan record rather than a header pointing at nothing.
        if sync_now {
            self.file.sync_data()?;
        }
        self.count += 1;
        Self::write_header(&mut self.file, self.dim, self.metric, self.count as u64)?;
        if sync_now {
            self.file.sync_all()?;
            self.dirty_bytes = 0;
        }

        self.remap()?;
        Ok(ordinal)
    }

    /// Marks record `ordinal` deleted, in place.
    pub fn tombstone(&mut self, ordinal: u32) -> Result<()> {
        let index = ordinal as usize;
        if index >= self.count {
            return Err(Error::NotFound);
        }
        let offset = (HEADER_LEN + index * self.stride) as u64;
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(&[FLAG_DELETED])?;
        self.dirty_bytes += 1;
        self.remap()?;
        Ok(())
    }

    /// Re-establishes the mapping after the file grew or changed.
    fn remap(&mut self) -> Result<()> {
        let mapped_len = HEADER_LEN + self.count * self.stride;
        self.file.flush()?;
        self.map.remap(&self.file, mapped_len)
    }

    /// Borrows record `ordinal`'s vector directly out of the mapping.
    ///
    /// Zero-copy: no allocation, no `read(2)`. Returns `None` for an unknown
    /// ordinal or a range the mapping does not cover.
    #[must_use]
    pub fn vector_at(&self, ordinal: u32) -> Option<&[f32]> {
        let index = ordinal as usize;
        if index >= self.count {
            return None;
        }
        let start = HEADER_LEN + index * self.stride + RECORD_PREFIX_LEN + MAX_ID_LEN;
        let bytes = self.map.slice(start, self.dim * 4)?;
        // f32 has no invalid bit patterns and the mapping outlives `&self`,
        // but the byte offset is only 4-aligned by construction, so the values
        // are read one at a time rather than transmuted wholesale.
        //
        // SAFETY: `bytes` is exactly `dim * 4` bytes inside the mapping, and
        // `f32` is `Copy` with no niche; `align_to` reports any misaligned
        // prefix, which the caller-visible `None` below rejects.
        let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
        if prefix.is_empty() && suffix.is_empty() && values.len() == self.dim {
            Some(values)
        } else {
            None
        }
    }

    /// Cached L2 norm of record `ordinal`.
    #[must_use]
    pub fn norm_at(&self, ordinal: u32) -> Option<f32> {
        let index = ordinal as usize;
        if index >= self.count {
            return None;
        }
        let start = HEADER_LEN + index * self.stride + 12;
        let bytes = self.map.slice(start, 4)?;
        Some(f32::from_le_bytes(bytes.try_into().ok()?))
    }

    /// Whether record `ordinal` is tombstoned. Unknown ordinals read as
    /// deleted, so a stale graph id can never surface a phantom result.
    #[must_use]
    pub fn is_deleted(&self, ordinal: u32) -> bool {
        let index = ordinal as usize;
        if index >= self.count {
            return true;
        }
        let start = HEADER_LEN + index * self.stride;
        match self.map.slice(start, 1) {
            Some(bytes) => bytes[0] & FLAG_DELETED != 0,
            None => true,
        }
    }

    /// External id of record `ordinal`.
    #[must_use]
    pub fn id_at(&self, ordinal: u32) -> Option<String> {
        let index = ordinal as usize;
        if index >= self.count {
            return None;
        }
        let base = HEADER_LEN + index * self.stride;
        let id_len = u32::from_le_bytes(self.map.slice(base + 4, 4)?.try_into().ok()?) as usize;
        if id_len == 0 || id_len > MAX_ID_LEN {
            return None;
        }
        let bytes = self.map.slice(base + RECORD_PREFIX_LEN, id_len)?;
        String::from_utf8(bytes.to_vec()).ok()
    }

    /// Reads and verifies record `ordinal`, checking its CRC32.
    ///
    /// Slower than the borrowing accessors; used by recovery, which must not
    /// trust the bytes, rather than by the search path, which reads records
    /// that recovery already validated.
    pub fn record_at(&self, ordinal: u32) -> Result<VectorRecord> {
        let index = ordinal as usize;
        if index >= self.count {
            return Err(Error::NotFound);
        }
        let base = HEADER_LEN + index * self.stride;
        let raw = self
            .map
            .slice(base, self.stride)
            .ok_or_else(|| Error::corrupt(format!("record {ordinal} is outside the mapping")))?;

        let stored_crc = u32::from_le_bytes(
            raw[8..12]
                .try_into()
                .map_err(|_| Error::corrupt("record checksum field is truncated"))?,
        );
        if crc32fast::hash(&raw[12..]) != stored_crc {
            return Err(Error::corrupt(format!(
                "vector record {ordinal} failed its CRC32 check"
            )));
        }

        let id_len = u32::from_le_bytes(
            raw[4..8]
                .try_into()
                .map_err(|_| Error::corrupt("record id length is truncated"))?,
        ) as usize;
        if id_len == 0 || id_len > MAX_ID_LEN {
            return Err(Error::corrupt(format!(
                "vector record {ordinal} declares an id length of {id_len}"
            )));
        }
        let id = String::from_utf8(raw[RECORD_PREFIX_LEN..RECORD_PREFIX_LEN + id_len].to_vec())
            .map_err(|_| Error::corrupt(format!("vector record {ordinal} has a non-UTF-8 id")))?;

        let payload_start = RECORD_PREFIX_LEN + MAX_ID_LEN;
        let mut vector = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let at = payload_start + i * 4;
            vector.push(f32::from_le_bytes(
                raw[at..at + 4]
                    .try_into()
                    .map_err(|_| Error::corrupt("record payload is truncated"))?,
            ));
        }

        Ok(VectorRecord {
            id,
            vector,
            norm: f32::from_le_bytes(
                raw[12..16]
                    .try_into()
                    .map_err(|_| Error::corrupt("record norm field is truncated"))?,
            ),
            deleted: raw[0] & FLAG_DELETED != 0,
        })
    }

    /// Forces every buffered write to stable storage.
    pub fn sync(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.sync_all()?;
        self.dirty_bytes = 0;
        Ok(())
    }
}

impl std::fmt::Debug for VectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStore")
            .field("path", &self.path)
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("count", &self.count)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store(dim: usize) -> (tempfile::TempDir, VectorStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open(dir.path().join("v.vec"), dim, Metric::Cosine).unwrap();
        (dir, store)
    }

    #[test]
    fn append_then_read_back_zero_copy() {
        let (_d, mut store) = temp_store(4);
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let ordinal = store.append("first", &v, 5.4772253, false).unwrap();
        assert_eq!(ordinal, 0);
        assert_eq!(store.len(), 1);
        assert_eq!(store.vector_at(0).unwrap(), v.as_slice());
        assert_eq!(store.id_at(0).unwrap(), "first");
        assert!((store.norm_at(0).unwrap() - 5.4772253).abs() < 1e-6);
        assert!(!store.is_deleted(0));
    }

    #[test]
    fn records_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.vec");
        {
            let mut store = VectorStore::open(&path, 3, Metric::Euclidean).unwrap();
            for i in 0..25u32 {
                let v = vec![i as f32, i as f32 + 1.0, i as f32 + 2.0];
                store.append(&format!("id-{i}"), &v, 0.0, false).unwrap();
            }
            store.sync().unwrap();
        }
        let store = VectorStore::open(&path, 3, Metric::Euclidean).unwrap();
        assert_eq!(store.len(), 25);
        for i in 0..25u32 {
            assert_eq!(store.id_at(i).unwrap(), format!("id-{i}"));
            assert_eq!(store.vector_at(i).unwrap()[0], i as f32);
        }
    }

    #[test]
    fn geometry_mismatch_is_refused_not_reinterpreted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.vec");
        {
            let mut store = VectorStore::open(&path, 8, Metric::Cosine).unwrap();
            store.append("a", &[1.0; 8], 1.0, true).unwrap();
        }
        // Wrong dimension and wrong metric must both be hard errors: reading
        // the file with either mistake yields plausible nonsense.
        assert!(VectorStore::open(&path, 16, Metric::Cosine).is_err());
        assert!(VectorStore::open(&path, 8, Metric::Euclidean).is_err());
        assert!(VectorStore::open(&path, 8, Metric::Cosine).is_ok());
    }

    #[test]
    fn tombstone_is_persistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.vec");
        {
            let mut store = VectorStore::open(&path, 2, Metric::Cosine).unwrap();
            store.append("keep", &[1.0, 0.0], 1.0, false).unwrap();
            store.append("drop", &[0.0, 1.0], 1.0, false).unwrap();
            store.tombstone(1).unwrap();
            store.sync().unwrap();
        }
        let store = VectorStore::open(&path, 2, Metric::Cosine).unwrap();
        assert!(!store.is_deleted(0));
        assert!(store.is_deleted(1));
        // The record itself survives, so graph ids never shift.
        assert_eq!(store.id_at(1).unwrap(), "drop");
        assert!(store.record_at(1).unwrap().deleted);
    }

    #[test]
    fn out_of_range_reads_are_safe() {
        let (_d, mut store) = temp_store(4);
        store.append("only", &[1.0; 4], 2.0, false).unwrap();
        assert!(store.vector_at(1).is_none());
        assert!(store.id_at(99).is_none());
        assert!(store.norm_at(99).is_none());
        // An unknown ordinal reads as deleted so a stale graph id cannot
        // surface a phantom result.
        assert!(store.is_deleted(99));
        assert!(matches!(store.record_at(99), Err(Error::NotFound)));
        assert!(matches!(store.tombstone(99), Err(Error::NotFound)));
    }

    #[test]
    fn dimension_and_id_limits_are_enforced() {
        let (_d, mut store) = temp_store(4);
        assert!(store.append("short", &[1.0; 3], 1.0, false).is_err());
        assert!(store.append("long", &[1.0; 5], 1.0, false).is_err());
        assert!(store.append("", &[1.0; 4], 1.0, false).is_err());
        let oversized = "x".repeat(MAX_ID_LEN + 1);
        assert!(store.append(&oversized, &[1.0; 4], 1.0, false).is_err());
        // Exactly at the limit must be accepted.
        let exact = "y".repeat(MAX_ID_LEN);
        assert!(store.append(&exact, &[1.0; 4], 1.0, false).is_ok());
        assert_eq!(store.id_at(0).unwrap(), exact);
    }

    #[test]
    fn crc_detects_a_flipped_payload_byte() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.vec");
        {
            let mut store = VectorStore::open(&path, 4, Metric::Cosine).unwrap();
            store
                .append("victim", &[1.0, 2.0, 3.0, 4.0], 5.0, true)
                .unwrap();
        }
        {
            // Corrupt one byte inside the payload.
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let at = (HEADER_LEN + RECORD_PREFIX_LEN + MAX_ID_LEN + 1) as u64;
            f.seek(SeekFrom::Start(at)).unwrap();
            f.write_all(&[0xFF]).unwrap();
            f.sync_all().unwrap();
        }
        let store = VectorStore::open(&path, 4, Metric::Cosine).unwrap();
        assert!(
            matches!(store.record_at(0), Err(Error::Corruption(_))),
            "a flipped payload byte must fail the checksum"
        );
    }

    #[test]
    fn a_non_phoenix_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-store.bin");
        std::fs::write(&path, vec![7u8; 512]).unwrap();
        assert!(matches!(
            VectorStore::open(&path, 4, Metric::Cosine),
            Err(Error::Corruption(_))
        ));
    }

    #[test]
    fn a_torn_tail_record_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.vec");
        {
            let mut store = VectorStore::open(&path, 4, Metric::Cosine).unwrap();
            store.append("a", &[1.0; 4], 2.0, true).unwrap();
            store.append("b", &[2.0; 4], 4.0, true).unwrap();
        }
        // Truncate mid-record: the header claims two, the file holds one and
        // a fragment. Recovery must fall back to the count the file supports.
        let stride = VectorStore::stride_for(4);
        let truncated = (HEADER_LEN + stride + stride / 2) as u64;
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(truncated)
            .unwrap();

        let store = VectorStore::open(&path, 4, Metric::Cosine).unwrap();
        assert_eq!(store.len(), 1, "the partial record must not be visible");
        assert_eq!(store.id_at(0).unwrap(), "a");
    }

    #[test]
    fn stride_is_stable_for_a_given_dimension() {
        // The whole zero-copy design depends on this being a pure function of
        // `dim`; a change here is an on-disk format break.
        assert_eq!(VectorStore::stride_for(0), RECORD_PREFIX_LEN + MAX_ID_LEN);
        assert_eq!(
            VectorStore::stride_for(768),
            RECORD_PREFIX_LEN + MAX_ID_LEN + 768 * 4
        );
    }

    #[test]
    fn empty_store_reports_empty() {
        let (_d, store) = temp_store(16);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.vector_at(0).is_none());
    }
}
