//! Fuzzes page parsing with arbitrary bytes.
//!
//! The contract under test: `Page::from_bytes` must **never** panic and must
//! never accept a buffer whose CRC does not match. Anything that survives
//! parsing must also survive every accessor without panicking.

#![no_main]

use libfuzzer_sys::fuzz_target;
use phoenixdb::page::{Page, PAGE_SIZE};

fuzz_target!(|data: &[u8]| {
    // Arbitrary-length input: parsing must reject anything that is not exactly
    // one page without panicking.
    let _ = Page::from_bytes(data);

    if data.len() < PAGE_SIZE {
        return;
    }
    let page_bytes = &data[..PAGE_SIZE];

    if let Ok(page) = Page::from_bytes(page_bytes) {
        // A page that parsed must have a valid checksum.
        page.verify().expect("parsed page failed CRC verification");

        // Every accessor must be panic-free on a validated page.
        let n = page.num_keys() as usize;
        let _ = page.page_id();
        let _ = page.parent();
        let _ = page.extra();
        let _ = page.lsn();
        let _ = page.flags();
        let _ = page.free_space();
        let _ = page.fill_ratio();
        let _ = page.page_type();

        for i in 0..n {
            let _ = page.cell(i);
            let _ = page.cell_key(i);
            if page.is_leaf() {
                let _ = page.leaf_cell(i);
            } else {
                let _ = page.internal_child(i);
            }
        }
        let _ = page.search(b"probe");
        let _ = page.search(&[]);
        let _ = page.read_overflow();
        let _ = page.read_meta();
    }

    // Round-trip: a re-checksummed page must always parse back.
    let mut rebuilt = Page::new(7, phoenixdb::page::PageType::Leaf);
    rebuilt.set_parent(u32::from_le_bytes([data[0], data[1], data[2], data[3]]));
    rebuilt.finalize();
    let bytes = *rebuilt.as_bytes();
    Page::from_bytes(&bytes).expect("freshly finalized page must parse");
});
