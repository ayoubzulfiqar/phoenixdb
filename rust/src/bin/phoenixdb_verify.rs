use std::path::PathBuf;

use phoenixdb::{Error, Options, Pager};

const EXIT_OK: i32 = 0;
const EXIT_CORRUPTION: i32 = 1;
const EXIT_ERROR: i32 = 2;

fn usage() -> ! {
    eprintln!("Usage: phoenixdb_verify <database_path>");
    std::process::exit(EXIT_ERROR);
}

fn verify(path: PathBuf) -> Result<i32, Error> {
    let mut pager = Pager::open(&path, Options::default().cache_pages)?;
    let meta = pager.meta();
    let page_count = meta.page_count;

    let mut corrupted = 0u32;
    for id in 0..page_count {
        match pager.read_page(id) {
            Ok(page) => {
                if page.page_id() != id {
                    eprintln!("page {id}: page_id mismatch (got {})", page.page_id());
                    corrupted += 1;
                }
                match page.page_type() {
                    Ok(phoenixdb::page::PageType::Meta)
                    | Ok(phoenixdb::page::PageType::Leaf)
                    | Ok(phoenixdb::page::PageType::Internal)
                    | Ok(phoenixdb::page::PageType::Free)
                    | Ok(phoenixdb::page::PageType::Overflow) => {}
                    Err(_) => corrupted += 1,
                }
            }
            Err(Error::Corruption(msg)) => {
                eprintln!("page {id}: CRC/checksum failure — {msg}");
                corrupted += 1;
            }
            Err(other) => {
                eprintln!("page {id}: unexpected error — {other:?}");
                corrupted += 1;
            }
        }
    }

    if corrupted > 0 {
        eprintln!("{corrupted} of {page_count} pages failed verification");
        Ok(EXIT_CORRUPTION)
    } else {
        println!("all {page_count} pages passed verification");
        Ok(EXIT_OK)
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
            eprintln!("fatal: {e:?}");
            EXIT_ERROR
        }
    };
    std::process::exit(code);
}
