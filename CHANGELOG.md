# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - 2026-08-09

Initial release.

### Added

**Storage engine (Rust)**
- B+Tree index with configurable fill factors (minimum 50%, maximum 100%).
- Fixed 4096-byte slotted pages with a 32-byte header carrying `page_id`,
  `is_leaf`, `num_keys`, `parent_id` and a CRC32 checksum.
- CRC32 written before every page write and verified on every read; corruption
  is reported as an error and never returned as data.
- Structural validation of the slot directory in addition to the checksum, so a
  page cannot direct an accessor outside its own buffer.
- Overflow-page chains for values larger than 1 KiB, with cycle guards.
- Free-list recycling of released pages.
- Zero-copy reads through `mmap` (Unix) and `CreateFileMappingW` (Windows),
  paired with positional writes and `fsync`/`sync_all` for durability.
- LRU page cache in front of the mapping.

**Transactions**
- MVCC snapshot isolation: many concurrent readers, one writer serialised by
  `parking_lot::RwLock`.
- Write-ahead log with `Begin`, `Insert`, `Delete`, `Commit` and `Rollback`
  records, each framed with its own CRC32.
- `fsync` on commit, so a transaction is durable when `commit` returns.
- Crash recovery that replays only committed transactions and tolerates a torn
  log tail.
- Write-write conflict detection with a dedicated status code for retry.
- Checkpointing that merges versions into the tree, flushes, then truncates the
  log.

**FFI layer**
- Pure C ABI (`extern "C"`) covering open, close, insert, get, delete,
  begin/commit/rollback, checkpoint, flush, verify, count and free.
- `phoenixdb.h` generated automatically by `cbindgen` during the build.
- Pointer non-nullness and length limits (key 1 MiB, value 10 MiB) validated
  before any dereference, returning `-2`.
- Constant-time handle-tag verification, poisoned on close, so use-after-free
  and double close are rejected rather than followed.
- Panics contained with `catch_unwind` and mapped to `-7`; nothing unwinds into
  Dart.

**Dart package**
- Type-safe API using `Uint8List` for keys and values.
- `NativeFinalizer` attached to the native handle for automatic cleanup.
- `AsyncPhoenixDB`, an isolate-backed client that keeps blocking disk I/O off
  the UI thread.
- Automatic native-library discovery with an ABI-version check on load.

**Tooling**
- `build.sh` and `build.ps1` with cross-compilation support via `CC`/`AR`.
- `libfuzzer` harnesses for the B+Tree, page parsing and the FFI surface.
- 86 Rust tests and 33 Dart tests; `dart analyze --fatal-infos` clean.
