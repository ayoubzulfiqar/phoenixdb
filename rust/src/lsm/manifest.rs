//! Durable level manifest: the LSM's record of which SSTables are live.
//!
//! # Why this exists
//!
//! Without it the level layout lives only in memory, so a restart forgets which
//! files belong to which level — the tables on disk become unreadable garbage.
//! Persisting the manifest is what makes the LSM a real storage path rather
//! than a cache.
//!
//! # Format
//!
//! An append-only log of *edits*, framed exactly like the WAL
//! ([`crate::wal`]) so the recovery reasoning is identical:
//!
//! ```text
//!   [ len u32 ][ crc32 u32 ][ bincode(ManifestEdit) ... ]
//! ```
//!
//! Recovery replays every intact frame and stops at the first short or
//! bad-CRC frame. A torn tail is the expected state after a crash, not
//! corruption: the edit that was mid-write simply never happened.
//!
//! # Why edits rather than snapshots
//!
//! Appending one small record per flush/compaction costs a single `write` +
//! `fsync`, where rewriting the whole table list would cost O(tables) on every
//! change. The log is compacted ([`Manifest::compact`]) into a single
//! `FullSnapshot` edit whenever it grows past a threshold, so replay stays
//! bounded.
//!
//! # Crash-consistency contract
//!
//! The ordering that makes this safe:
//!
//! 1. Write the new SSTable and `fsync` it.
//! 2. Append the manifest edit and `fsync` it.  ← the commit point
//! 3. Only then unlink the inputs the edit obsoletes.
//!
//! A crash before step 2 leaves an orphaned file that no edit references, which
//! [`Manifest::orphaned_files`] finds and deletes. A crash after step 2 but
//! before step 3 leaves a file that is no longer referenced — same cleanup. At
//! no point can the manifest reference a file that does not exist.

use crate::error::{Error, Result};
use crate::lsm::sstable::TableMeta;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// File name of the manifest inside the LSM directory.
pub const MANIFEST_FILE: &str = "MANIFEST";

/// Refuse to allocate for a frame larger than this (guards a corrupt length).
const MAX_RECORD_BYTES: u32 = 16 * 1024 * 1024;

/// Compact the log once it exceeds this many edits.
pub const COMPACT_AFTER_EDITS: u64 = 1024;

/// A single change to the live table set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestEdit {
    /// A new table became live at `meta.level`.
    AddTable {
        /// The table's metadata.
        meta: SerializableMeta,
    },
    /// A table was retired and its file may be deleted.
    RemoveTable {
        /// Level the table belonged to.
        level: u32,
        /// Table identifier.
        id: u64,
    },
    /// The complete live set, written when the log is compacted.
    ///
    /// Replay treats this as a reset: everything before it is superseded.
    FullSnapshot {
        /// Every live table.
        tables: Vec<SerializableMeta>,
        /// Next unused table id.
        next_table_id: u64,
    },
    /// Records the highest sequence number durably reflected in the tables.
    ///
    /// Lets recovery know which WAL records are already folded into SSTables
    /// and need not be replayed.
    Checkpoint {
        /// Durable sequence number.
        seqno: u64,
    },
}

/// [`TableMeta`] in a form `serde` can round-trip.
///
/// `TableMeta` deliberately has no derive: it is a hot in-memory struct and
/// adding `Serialize` to it would leak the on-disk format into the read path.
/// Converting here keeps the persistence format explicit and versionable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableMeta {
    /// Table identifier.
    pub id: u64,
    /// LSM level.
    pub level: u32,
    /// Smallest user key.
    pub min_key: Vec<u8>,
    /// Largest user key.
    pub max_key: Vec<u8>,
    /// Lowest sequence number.
    pub min_seqno: u64,
    /// Highest sequence number.
    pub max_seqno: u64,
    /// Entry count.
    pub entry_count: u64,
    /// File size in bytes.
    pub file_bytes: u64,
}

impl From<&TableMeta> for SerializableMeta {
    fn from(m: &TableMeta) -> Self {
        SerializableMeta {
            id: m.id,
            level: m.level,
            min_key: m.min_key.clone(),
            max_key: m.max_key.clone(),
            min_seqno: m.min_seqno,
            max_seqno: m.max_seqno,
            entry_count: m.entry_count,
            file_bytes: m.file_bytes,
        }
    }
}

impl From<SerializableMeta> for TableMeta {
    fn from(m: SerializableMeta) -> Self {
        TableMeta {
            id: m.id,
            level: m.level,
            min_key: m.min_key,
            max_key: m.max_key,
            min_seqno: m.min_seqno,
            max_seqno: m.max_seqno,
            entry_count: m.entry_count,
            file_bytes: m.file_bytes,
        }
    }
}

/// The result of replaying a manifest log.
#[derive(Debug, Default)]
pub struct ManifestState {
    /// Live tables, in no particular order.
    pub tables: Vec<TableMeta>,
    /// Next table id to hand out.
    pub next_table_id: u64,
    /// Highest checkpointed sequence number.
    pub checkpoint_seqno: u64,
    /// Edits successfully replayed.
    pub edits_replayed: u64,
    /// Bytes discarded from a torn tail.
    pub truncated_bytes: u64,
}

/// Append-only manifest log.
pub struct Manifest {
    path: PathBuf,
    writer: BufWriter<File>,
    edits_written: u64,
}

impl Manifest {
    /// Opens (creating if needed) the manifest at `path`, positioned to append.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        let mut writer = BufWriter::new(file);
        writer.seek(SeekFrom::End(0))?;
        Ok(Manifest {
            path,
            writer,
            edits_written: 0,
        })
    }

    /// Path of the manifest file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Edits appended through this handle.
    #[must_use]
    pub fn edits_written(&self) -> u64 {
        self.edits_written
    }

    /// Appends `edit` and forces it to stable storage.
    ///
    /// This is the commit point for a flush or compaction: when it returns
    /// `Ok`, the change survives a power loss.
    pub fn append(&mut self, edit: &ManifestEdit) -> Result<()> {
        let payload = bincode::serialize(edit)?;
        if payload.len() as u64 > MAX_RECORD_BYTES as u64 {
            return Err(Error::Full(format!(
                "manifest edit of {} bytes exceeds the {MAX_RECORD_BYTES}-byte limit",
                payload.len()
            )));
        }
        let crc = crc32fast::hash(&payload);
        self.writer
            .write_all(&(payload.len() as u32).to_le_bytes())?;
        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.sync()?;
        self.edits_written += 1;
        Ok(())
    }

    /// Flushes buffers and `fsync`s.
    pub fn sync(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Current size of the manifest file in bytes.
    pub fn size(&self) -> Result<u64> {
        Ok(self.writer.get_ref().metadata()?.len())
    }

    /// Replays the manifest at `path`, returning the live table set.
    ///
    /// A torn tail is truncated rather than reported as an error — that is the
    /// expected state after a crash mid-append.
    pub fn recover(path: impl AsRef<Path>) -> Result<ManifestState> {
        let mut state = ManifestState {
            next_table_id: 1,
            ..ManifestState::default()
        };
        let mut file = match File::open(path.as_ref()) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(state),
            Err(e) => return Err(Error::Io(e)),
        };
        let total = file.metadata()?.len();
        if total == 0 {
            return Ok(state);
        }
        let mut bytes = Vec::with_capacity(total as usize);
        file.read_to_end(&mut bytes)?;

        // `live` is keyed by (level, id): a compaction moves a table between
        // levels, and the same id must not linger at the old level.
        let mut live: Vec<TableMeta> = Vec::new();
        let mut cursor = 0usize;
        let mut good_bytes = 0usize;

        while cursor + 8 <= bytes.len() {
            let len = u32::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ]);
            let crc = u32::from_le_bytes([
                bytes[cursor + 4],
                bytes[cursor + 5],
                bytes[cursor + 6],
                bytes[cursor + 7],
            ]);
            if len == 0 || len > MAX_RECORD_BYTES {
                break; // corrupt length: treat the rest as torn
            }
            let start = cursor + 8;
            let end = match start.checked_add(len as usize) {
                Some(e) if e <= bytes.len() => e,
                _ => break, // truncated tail
            };
            let payload = &bytes[start..end];
            if crc32fast::hash(payload) != crc {
                break; // torn or corrupted frame
            }
            let Ok(edit) = bincode::deserialize::<ManifestEdit>(payload) else {
                break;
            };

            match edit {
                ManifestEdit::AddTable { meta } => {
                    let meta: TableMeta = meta.into();
                    state.next_table_id = state.next_table_id.max(meta.id + 1);
                    live.retain(|t| t.id != meta.id);
                    live.push(meta);
                }
                ManifestEdit::RemoveTable { level, id } => {
                    live.retain(|t| !(t.id == id && t.level == level));
                }
                ManifestEdit::FullSnapshot {
                    tables,
                    next_table_id,
                } => {
                    // A snapshot supersedes everything before it.
                    live = tables.into_iter().map(TableMeta::from).collect();
                    state.next_table_id = next_table_id.max(1);
                    for t in &live {
                        state.next_table_id = state.next_table_id.max(t.id + 1);
                    }
                }
                ManifestEdit::Checkpoint { seqno } => {
                    state.checkpoint_seqno = state.checkpoint_seqno.max(seqno);
                }
            }
            state.edits_replayed += 1;
            cursor = end;
            good_bytes = end;
        }

        state.truncated_bytes = total - good_bytes as u64;
        state.tables = live;
        Ok(state)
    }

    /// Rewrites the log as one `FullSnapshot`, bounding replay time.
    ///
    /// Written to a temporary file and renamed over the original, so a crash
    /// mid-compaction leaves the old manifest intact: `rename` is atomic on
    /// every platform PhoenixDB targets.
    pub fn compact(
        path: impl AsRef<Path>,
        tables: &[TableMeta],
        next_table_id: u64,
        checkpoint_seqno: u64,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let tmp = path.with_extension("tmp");

        {
            let file = File::create(&tmp)?;
            let mut fresh = Manifest {
                path: tmp.clone(),
                writer: BufWriter::new(file),
                edits_written: 0,
            };
            fresh.append(&ManifestEdit::FullSnapshot {
                tables: tables.iter().map(SerializableMeta::from).collect(),
                next_table_id,
            })?;
            if checkpoint_seqno > 0 {
                fresh.append(&ManifestEdit::Checkpoint {
                    seqno: checkpoint_seqno,
                })?;
            }
            fresh.sync()?;
        }

        // Atomic swap. On Windows `rename` fails if the target exists, so the
        // original is removed first; the temp file is already durable.
        #[cfg(windows)]
        {
            if path.exists() {
                std::fs::remove_file(&path)?;
            }
        }
        std::fs::rename(&tmp, &path)?;

        // fsync the directory so the rename itself is durable, not just the
        // file contents. Without this a crash can resurrect the old manifest.
        if let Some(dir) = path.parent()
            && let Ok(handle) = File::open(dir)
        {
            let _ = handle.sync_all(); // best-effort: not supported everywhere
        }

        Manifest::open(&path)
    }

    /// SSTable files in `dir` that the live set does not reference.
    ///
    /// These are the debris of a crash between writing a table and committing
    /// its manifest edit. Deleting them is always safe: no manifest edit
    /// mentions them, so no reader can reach them.
    pub fn orphaned_files(dir: impl AsRef<Path>, live: &[TableMeta]) -> Result<Vec<PathBuf>> {
        let referenced: HashSet<u64> = live.iter().map(|t| t.id).collect();
        let mut orphans = Vec::new();
        let entries = match std::fs::read_dir(dir.as_ref()) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(orphans),
            Err(e) => return Err(Error::Io(e)),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("sst") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            // File names are zero-padded ids: `0000000042.sst`.
            let Ok(id) = stem.parse::<u64>() else {
                continue; // not ours; leave it alone
            };
            if !referenced.contains(&id) {
                orphans.push(path);
            }
        }
        orphans.sort();
        Ok(orphans)
    }
}

impl std::fmt::Debug for Manifest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Manifest")
            .field("path", &self.path)
            .field("edits_written", &self.edits_written)
            .finish()
    }
}
