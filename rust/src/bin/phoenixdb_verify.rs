//! Offline integrity checker: `phoenixdb_verify <database_path>`.
//!
//! Checks every allocated page's CRC and header, then runs the full B+Tree
//! structural check (key order and ranges, uniform leaf depth, leaf chain,
//! overflow chains, free list). Exit codes: 0 healthy, 1 corruption found,
//! 2 the file could not be checked at all (missing, locked, not PhoenixDB).
//!
//! The database must not be open elsewhere: the checker takes the same
//! exclusive lock as the engine.

use std::path::PathBuf;

use phoenixdb::btree::{BTree, FillFactor};
use phoenixdb::page::PageType;
use phoenixdb::{Error, Options, Pager};

const EXIT_OK: i32 = 0;
const EXIT_CORRUPTION: i32 = 1;
const EXIT_ERROR: i32 = 2;

fn usage() -> ! {
    eprintln!("Usage: phoenixdb_verify <database_path>");
    std::process::exit(EXIT_ERROR);
}

fn verify(path: PathBuf) -> Result<i32, Error> {
    if !path.exists() {
        return Err(Error::invalid(format!("{} does not exist", path.display())));
    }
    let pager = Pager::open(&path, Options::default().cache_pages)?;
    let meta = pager.meta();
    let page_count = meta.page_count;

    let mut corrupted = 0u32;
    for id in 0..page_count {
        match pager.read_page(id) {
            Ok(page) => {
                if let Err(e) = page.page_type() {
                    eprintln!("page {id}: {e}");
                    corrupted += 1;
                } else if id == 0 && page.page_type().ok() != Some(PageType::Meta) {
                    eprintln!("page 0: not a meta page");
                    corrupted += 1;
                }
            }
            Err(Error::Corruption(msg)) => {
                eprintln!("page {id}: {msg}");
                corrupted += 1;
            }
            Err(other) => {
                eprintln!("page {id}: unexpected error — {other}");
                corrupted += 1;
            }
        }
    }

    if corrupted > 0 {
        eprintln!("{corrupted} of {page_count} pages failed verification");
        return Ok(EXIT_CORRUPTION);
    }
    match BTree::new(FillFactor::default()).check(&pager) {
        Ok(report) => {
            println!(
                "all {page_count} pages passed verification: {} keys, depth {}, \
                 {} leaf / {} internal / {} overflow / {} free pages, {} unreachable",
                report.keys,
                report.depth,
                report.leaf_pages,
                report.internal_pages,
                report.overflow_pages,
                report.free_pages,
                report.unreachable_pages
            );
            Ok(EXIT_OK)
        }
        Err(e) => {
            eprintln!("tree structure check failed: {e}");
            Ok(EXIT_CORRUPTION)
        }
    }
}

fn main() {
    let mut args = std::env::args_os();
    let _ = args.next();
    let path = match args.next() {
        Some(p) => PathBuf::from(p),
        None => usage(),
    };

    let code = match verify(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("fatal: {e}");
            EXIT_ERROR
        }
    };
    std::process::exit(code);
}
