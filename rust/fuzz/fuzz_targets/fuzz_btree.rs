//! Fuzzes B+Tree insert/delete/get against a `BTreeMap` oracle.
//!
//! Run with:
//! ```sh
//! cargo +nightly fuzz run fuzz_btree -- -max_total_time=60
//! ```
//!
//! Any divergence from the oracle, any panic, or any violated tree invariant
//! is a crash the fuzzer will minimise for you.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use phoenixdb::btree::{BTree, FillFactor};
use phoenixdb::pager::Pager;
use std::collections::BTreeMap;

/// One fuzzer-chosen operation.
#[derive(Arbitrary, Debug)]
enum Op {
    Insert { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    Get { key: Vec<u8> },
    Scan,
    Verify,
}

/// Keys are capped well below the structural limit so most operations succeed
/// and the fuzzer spends its budget on tree shape rather than rejections.
const MAX_KEY: usize = 200;
const MAX_VALUE: usize = 2048;

fuzz_target!(|ops: Vec<Op>| {
    if ops.len() > 512 {
        return;
    }
    let Ok(dir) = tempfile::tempdir() else {
        return;
    };
    let Ok(mut pager) = Pager::open(&dir.path().join("fuzz.pdb"), 128) else {
        return;
    };
    let tree = BTree::new(FillFactor::default());
    let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();

    for op in ops {
        match op {
            Op::Insert { mut key, mut value } => {
                key.truncate(MAX_KEY);
                value.truncate(MAX_VALUE);
                if key.is_empty() {
                    continue;
                }
                match tree.insert(&mut pager, &key, &value) {
                    Ok(()) => {
                        oracle.insert(key, value);
                    }
                    Err(phoenixdb::Error::Full(_)) => {} // acceptable back-pressure
                    Err(e) => panic!("insert failed unexpectedly: {e:?}"),
                }
            }
            Op::Delete { mut key } => {
                key.truncate(MAX_KEY);
                if key.is_empty() {
                    continue;
                }
                let expected = oracle.remove(&key).is_some();
                match tree.delete(&mut pager, &key) {
                    Ok(()) => assert!(expected, "deleted a key the oracle does not have"),
                    Err(phoenixdb::Error::NotFound) => {
                        assert!(!expected, "failed to delete a key the oracle has")
                    }
                    Err(e) => panic!("delete failed unexpectedly: {e:?}"),
                }
            }
            Op::Get { mut key } => {
                key.truncate(MAX_KEY);
                if key.is_empty() {
                    continue;
                }
                match tree.get(&mut pager, &key) {
                    Ok(v) => assert_eq!(
                        Some(&v),
                        oracle.get(&key),
                        "value mismatch for key {key:?}"
                    ),
                    Err(phoenixdb::Error::NotFound) => {
                        assert!(!oracle.contains_key(&key), "lost key {key:?}")
                    }
                    Err(e) => panic!("get failed unexpectedly: {e:?}"),
                }
            }
            Op::Scan => {
                if let Ok(items) = tree.scan(&mut pager) {
                    assert_eq!(items.len(), oracle.len(), "scan length diverged");
                    for w in items.windows(2) {
                        assert!(w[0].0 < w[1].0, "scan is not sorted");
                    }
                }
            }
            Op::Verify => {
                tree.verify(&mut pager).expect("tree invariants violated");
            }
        }
    }
    tree.verify(&mut pager).expect("final verify failed");
});
