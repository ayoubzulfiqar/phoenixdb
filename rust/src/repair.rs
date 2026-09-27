//! Salvage: recover what is still readable from a damaged database.
//!
//! [`Database::check`](crate::Database::check) and the `phoenixdb_verify`
//! binary *report* damage; this recovers from it. The tree may be unusable —
//! a corrupt root, a broken link, a page the checksum rejects — and a
//! structural walk cannot get past that. Salvage ignores the structure
//! entirely: it scans every page, keeps the leaf cells whose checksum passes,
//! and writes them into a fresh database.
//!
//! What survives is every key/value pair on an intact leaf page. What is lost
//! is whatever lived on a damaged page. The report says exactly how much of
//! each, so a caller can decide whether to trust the result.
//!
//! # Newer copies win
//!
//! A key can appear on several pages: a split leaves the old copy behind, and
//! a freed page keeps its bytes until it is reused. Pages are therefore
//! replayed in order of their log sequence number, so a later version of a key
//! overwrites an earlier one — the same order the engine wrote them in.
//!
//! # Usage
//!
//! ```no_run
//! use phoenixdb::repair::salvage;
//!
//! # fn main() -> phoenixdb::Result<()> {
//! let report = salvage("damaged.pdb", "recovered.pdb")?;
//! println!("{} of {} keys recovered", report.keys_recovered, report.keys_seen());
//! # Ok(())
//! # }
//! ```
//!
//! Close the database first: salvage reads the file directly, and a half-
//! written flush would look like damage.

use crate::error::{Error, Result};
use crate::page::{PAGE_SIZE, Page, PageType};
use crate::{Database, Options};
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

/// Sentinel used by the pager for "no next page".
const SENTINEL: u32 = u32::MAX;

/// What [`salvage`] found and recovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub struct SalvageReport {
    /// Pages read from the source file.
    pub pages_scanned: u32,
    /// Leaf pages whose checksum passed and whose cells were readable.
    pub leaf_pages: u32,
    /// Pages whose checksum or structure was damaged, and whose contents are
    /// therefore lost.
    pub pages_damaged: u32,
    /// Key/value pairs written to the destination (a key recovered twice is
    /// counted once, the newer copy winning).
    pub keys_recovered: u64,
    /// Cells that could not be recovered: a damaged overflow chain, or a key
    /// or value the engine would refuse.
    pub keys_unreadable: u64,
    /// Value bytes recovered.
    pub bytes_recovered: u64,
}

impl SalvageReport {
    /// Cells encountered, recoverable or not.
    #[must_use]
    pub fn keys_seen(&self) -> u64 {
        self.keys_recovered + self.keys_unreadable
    }

    /// Whether the source was intact: nothing damaged, nothing unreadable.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.pages_damaged == 0 && self.keys_unreadable == 0
    }
}

/// Recovers every readable key/value pair from `source` into a new database at
/// `destination`.
///
/// `destination` must not exist. The source is only read, so it can be tried
/// again with different tooling if the result is unsatisfying.
pub fn salvage(source: impl AsRef<Path>, destination: impl AsRef<Path>) -> Result<SalvageReport> {
    let source = source.as_ref();
    let destination = destination.as_ref();
    if source == destination {
        return Err(Error::invalid("cannot salvage a database onto itself"));
    }
    if destination.exists() {
        return Err(Error::invalid(format!(
            "{} already exists; salvage writes a new database",
            destination.display()
        )));
    }
    let file = File::open(source)?;
    let len = file.metadata()?.len();
    if len < PAGE_SIZE as u64 {
        return Err(Error::invalid(format!(
            "{} is too small to be a PhoenixDB file",
            source.display()
        )));
    }
    let page_count = u32::try_from(len / PAGE_SIZE as u64).unwrap_or(u32::MAX);

    // Probe before scanning. A database that is still open holds an exclusive
    // lock on its file, and on Windows that lock is mandatory: every read
    // through this handle fails. Without the probe each failure would look
    // like a damaged page and salvage would cheerfully report that nothing
    // could be recovered.
    let mut probe = vec![0u8; PAGE_SIZE];
    if let Err(e) = crate::fsutil::read_at(&file, &mut probe, 0) {
        return Err(Error::Busy(format!(
            "cannot read {}: {e}. Close the database first — salvage reads the \
             file directly",
            source.display()
        )));
    }

    let mut report = SalvageReport::default();

    // Pass one: find the intact leaf pages and their log sequence numbers, so
    // they can be replayed oldest-first.
    let mut leaves: Vec<(u64, u32)> = Vec::new();
    for id in 0..page_count {
        report.pages_scanned += 1;
        let page = match read_page(&file, id) {
            PageState::Ok(page) => page,
            // A page the engine has never written holds nothing and is not
            // damage: a file grows in chunks, and a freed page keeps its old
            // bytes only until it is reused.
            PageState::Blank => continue,
            PageState::Damaged => {
                report.pages_damaged += 1;
                continue;
            }
        };
        match page.page_type() {
            Ok(PageType::Leaf) => {
                if page.validate_structure().is_ok() {
                    leaves.push((page.lsn(), id));
                } else {
                    report.pages_damaged += 1;
                }
            }
            // An internal, meta, overflow or free page carries no pairs of its
            // own; overflow pages are read through the cells that point at
            // them.
            Ok(_) => {}
            Err(_) => report.pages_damaged += 1,
        }
    }
    leaves.sort_unstable();

    // Pass two: replay the cells into a fresh database.
    let recovered = Database::open(destination, Options::default())?;
    let mut written = 0u64;
    for (_, id) in &leaves {
        let PageState::Ok(page) = read_page(&file, *id) else {
            // It verified in the first pass, so this cannot normally happen.
            report.pages_damaged += 1;
            continue;
        };
        report.leaf_pages += 1;
        let keys = page.num_keys() as usize;
        let mut batch: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(keys);
        for index in 0..keys {
            let Ok(cell) = page.leaf_cell(index) else {
                report.keys_unreadable += 1;
                continue;
            };
            let value = match cell.overflow {
                None => cell.value,
                Some(head) => match read_chain(&file, page_count, head, cell.total_len) {
                    Some(bytes) => bytes,
                    None => {
                        report.keys_unreadable += 1;
                        continue;
                    }
                },
            };
            report.bytes_recovered += value.len() as u64;
            batch.push((cell.key, value));
        }
        // One transaction per page keeps memory flat and means a failure part
        // way through still leaves everything before it recovered.
        let outcome = recovered.write_batch(|b| {
            for (key, value) in &batch {
                b.put(key, value)?;
            }
            Ok(())
        });
        match outcome {
            Ok(()) => written += batch.len() as u64,
            Err(_) => {
                // A key or value the engine refuses (over a limit, say). Retry
                // one at a time so one bad cell cannot cost a whole page.
                for (key, value) in &batch {
                    match recovered.put_auto(key, value) {
                        Ok(()) => written += 1,
                        Err(_) => report.keys_unreadable += 1,
                    }
                }
            }
        }
    }
    recovered.checkpoint()?;
    // Distinct keys, since a key recovered from several pages was written more
    // than once and only the newest copy survives.
    report.keys_recovered = recovered.len()?.min(written);
    drop(recovered);
    Ok(report)
}

/// What one page turned out to be.
enum PageState {
    /// Readable and checksummed.
    Ok(Page),
    /// Never written: all zeroes.
    Blank,
    /// Written, but the checksum, structure or page id disagrees.
    Damaged,
}

/// Reads and verifies one page.
fn read_page(file: &File, id: u32) -> PageState {
    let mut bytes = vec![0u8; PAGE_SIZE];
    let offset = u64::from(id) * PAGE_SIZE as u64;
    if crate::fsutil::read_at(file, &mut bytes, offset).is_err() {
        return PageState::Damaged;
    }
    if bytes.iter().all(|&b| b == 0) {
        return PageState::Blank;
    }
    let Ok(page) = Page::from_bytes(&bytes) else {
        return PageState::Damaged;
    };
    if page.verify().is_err() || page.page_id() != id {
        // A stale page from an earlier layout, a torn write, or a scribble.
        return PageState::Damaged;
    }
    PageState::Ok(page)
}

/// Follows an overflow chain, or `None` when any link is damaged.
fn read_chain(file: &File, page_count: u32, head: u32, total_len: u32) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(total_len as usize);
    let mut cursor = head;
    let mut seen = HashSet::new();
    while cursor != SENTINEL {
        if cursor >= page_count || !seen.insert(cursor) {
            return None; // outside the file, or a cycle
        }
        let PageState::Ok(page) = read_page(file, cursor) else {
            return None;
        };
        if page.page_type().ok()? != PageType::Overflow {
            return None;
        }
        out.extend_from_slice(page.read_overflow().ok()?);
        if out.len() > total_len as usize {
            return None;
        }
        cursor = page.extra();
    }
    (out.len() == total_len as usize).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsutil::write_at;
    use std::fs::OpenOptions;

    fn populate(path: &Path, keys: usize) {
        let db = Database::open(path, Options::default()).unwrap();
        db.write_batch(|b| {
            for i in 0..keys {
                b.put(
                    format!("key{i:04}").as_bytes(),
                    format!("value for {i}").as_bytes(),
                )?;
            }
            Ok(())
        })
        .unwrap();
        // A value large enough to spill onto overflow pages.
        db.put_auto(b"big", &vec![b'x'; 40_000]).unwrap();
        db.checkpoint().unwrap();
    }

    #[test]
    fn an_intact_database_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("intact.pdb");
        let dest = dir.path().join("out.pdb");
        populate(&source, 200);

        let report = salvage(&source, &dest).unwrap();
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(report.keys_recovered, 201);
        assert!(report.leaf_pages > 0);

        let recovered = Database::open(&dest, Options::default()).unwrap();
        recovered.verify().unwrap();
        assert_eq!(recovered.get_auto(b"key0007").unwrap(), b"value for 7");
        assert_eq!(recovered.get_auto(b"big").unwrap().len(), 40_000);
        assert_eq!(recovered.len().unwrap(), 201);
    }

    #[test]
    fn a_torn_page_costs_only_its_own_keys() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("torn.pdb");
        let dest = dir.path().join("out.pdb");
        populate(&source, 400);

        // Scribble over a page that actually holds data — what a failing
        // disk or a truncated copy leaves behind.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&source)
            .unwrap();
        let pages = (file.metadata().unwrap().len() / PAGE_SIZE as u64) as u32;
        let victim = (0..pages)
            .filter(|id| match read_page(&file, *id) {
                PageState::Ok(page) => matches!(page.page_type(), Ok(PageType::Leaf)),
                _ => false,
            })
            .nth(1)
            .expect("expected several leaf pages");
        write_at(
            &file,
            &[0xAA; PAGE_SIZE],
            u64::from(victim) * PAGE_SIZE as u64,
        )
        .unwrap();
        drop(file);

        let report = salvage(&source, &dest).unwrap();
        assert!(!report.is_clean());
        assert_eq!(report.pages_damaged, 1, "{report:?}");
        // One lost leaf page costs the keys that lived on it and nothing
        // more: the rest of the file is still readable.
        assert!(
            (200..401).contains(&report.keys_recovered),
            "one bad page should cost one page's worth of keys: {report:?}"
        );

        let recovered = Database::open(&dest, Options::default()).unwrap();
        recovered.verify().unwrap();
        assert_eq!(recovered.len().unwrap(), report.keys_recovered);
    }

    #[test]
    fn newer_copies_of_a_key_win() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("versions.pdb");
        let dest = dir.path().join("out.pdb");
        {
            let db = Database::open(&source, Options::default()).unwrap();
            for round in 0..6 {
                for i in 0..40 {
                    db.put_auto(
                        format!("k{i:03}").as_bytes(),
                        format!("round {round}").as_bytes(),
                    )
                    .unwrap();
                }
                db.checkpoint().unwrap();
            }
        }
        let report = salvage(&source, &dest).unwrap();
        assert_eq!(report.keys_recovered, 40, "{report:?}");
        let recovered = Database::open(&dest, Options::default()).unwrap();
        assert_eq!(recovered.get_auto(b"k007").unwrap(), b"round 5");
    }

    #[test]
    fn refuses_a_destination_it_would_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("a.pdb");
        populate(&source, 5);
        assert!(salvage(&source, &source).is_err(), "onto itself");
        let existing = dir.path().join("taken.pdb");
        std::fs::write(&existing, b"do not clobber me").unwrap();
        assert!(salvage(&source, &existing).is_err());
        assert_eq!(std::fs::read(&existing).unwrap(), b"do not clobber me");
        assert!(
            salvage(dir.path().join("missing.pdb"), dir.path().join("b.pdb")).is_err(),
            "a missing source"
        );
    }
}
