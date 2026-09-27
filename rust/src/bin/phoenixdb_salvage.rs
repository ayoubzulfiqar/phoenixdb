//! Offline recovery: `phoenixdb_salvage <damaged_database> <destination>`.
//!
//! Scans the damaged file page by page, keeps every key/value pair on a leaf
//! page whose checksum passes, and writes them into a new database. Use it
//! when `phoenixdb_verify` reports corruption the engine cannot open past.
//!
//! The destination must not exist and the source is only read, so an
//! unsatisfying result can be retried with other tooling.
//!
//! Exit codes: 0 everything was recovered intact, 1 recovered with losses
//! (the report says what), 2 nothing could be recovered.

use std::path::PathBuf;

use phoenixdb::Error;
use phoenixdb::repair::salvage;

const EXIT_OK: i32 = 0;
const EXIT_PARTIAL: i32 = 1;
const EXIT_ERROR: i32 = 2;

fn usage() -> ! {
    eprintln!("Usage: phoenixdb_salvage <damaged_database> <destination>");
    std::process::exit(EXIT_ERROR);
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    let (Some(source), Some(destination), None) = (args.next(), args.next(), args.next()) else {
        usage();
    };
    let source = PathBuf::from(source);
    let destination = PathBuf::from(destination);

    match salvage(&source, &destination) {
        Ok(report) => {
            println!("scanned      {} pages", report.pages_scanned);
            println!("leaf pages   {}", report.leaf_pages);
            println!("damaged      {} pages", report.pages_damaged);
            println!(
                "recovered    {} keys ({} bytes of values)",
                report.keys_recovered, report.bytes_recovered
            );
            if report.keys_unreadable > 0 {
                println!("unreadable   {} cells", report.keys_unreadable);
            }
            println!("written to   {}", destination.display());
            if report.is_clean() {
                println!("\nthe source was intact: everything was recovered");
                std::process::exit(EXIT_OK);
            }
            println!(
                "\nrecovered with losses: {} damaged page(s) and {} unreadable cell(s).",
                report.pages_damaged, report.keys_unreadable
            );
            println!("Whatever lived on those pages is gone; the rest is in the new file.");
            std::process::exit(EXIT_PARTIAL);
        }
        Err(e) => {
            eprintln!("salvage failed: {e}");
            if matches!(e, Error::Busy(_)) {
                eprintln!("close the database first: salvage reads the file directly");
            }
            std::process::exit(EXIT_ERROR);
        }
    }
}
