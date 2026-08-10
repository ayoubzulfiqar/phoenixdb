# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 2.0.0 - 2026-08-09

Multi-modal storage: a hybrid LSM layer, a SQL front end, encryption at rest,
RBAC, audit logging, and metrics — all behind feature flags so the embedded
core stays small.

### Breaking

- **Native ABI raised from 1 to 2.** The change is additive (every v1 entry
  point keeps its signature) but the Dart loader enforces an *exact* match, so
  a v1 native library and this package will not load together. Both are
  rebuilt and shipped in this release; anyone building the library themselves
  must rebuild it.

### Added

**LSM storage layer (`lsm`) — library only, see Known limitations**

- MemTable, SSTable (block-based, with a Bloom filter per table), leveled
  compaction, and a compaction scheduler prioritised by write amplification.
- A crash-safe, CRC-framed **durable manifest** recording which SSTables are
  live at which level. A torn tail from a power loss is truncated rather than
  treated as corruption; the layout, level placement, tombstones, and
  checkpoint sequence number all survive a restart. Unreferenced `.sst` files
  are reclaimed on open; a *referenced* table that fails its checksum is a hard
  error rather than silent data loss.
- The manifest log is snapshotted once it grows past a threshold, so startup
  replay does not slow down without bound.

**SQL front end (`sql`, opt-in)**

- Hand-written lexer, recursive-descent parser, and executor supporting
  `CREATE TABLE`, `DROP TABLE`, `INSERT`, `SELECT` (projection, `WHERE`,
  `AND`/`OR`, `ORDER BY`, `LIMIT`), `UPDATE`, and `DELETE`, plus
  `IF NOT EXISTS` / `IF EXISTS`.
- SQL semantics where they matter: `NULL` is never equal to anything under an
  ordinary comparison, `ORDER BY` is applied before `LIMIT`, integers and
  floats compare numerically, and mismatched types yield no rows rather than an
  arbitrary ordering.
- Mutations are transactional — a multi-row `INSERT` either lands completely or
  not at all.
- Reachable from Dart as `db.query(...)` (synchronous) and
  `AsyncPhoenixDB.query(...)` (on the worker isolate, so a slow query cannot
  block a Flutter frame). Results arrive as a typed `SqlResult` with
  `scalar`, `firstOrNull`, `cell()`, and `asMaps` helpers.

**Security (`encryption`, opt-in) — library only, see Known limitations**

- Transparent AES-256-GCM encryption at rest. Pages are encrypted before
  checksumming and decrypted after verification, so a tampered or swapped page
  is rejected rather than decrypted into garbage.
- Role-based access control with constant-time credential comparison.
- An append-only audit log kept separate from the WAL, resistant to log
  injection.

**Observability (`metrics`, opt-in) — library only, see Known limitations**

- Latency histograms with p50/p99/p999, covering WAL fsync, compaction
  throughput, and cache hit/miss ratios.
- Structured tracing spans with trace ids.

**Dart API**

- `PhoenixPrefs`, a `shared_preferences`-style typed facade (`getString`,
  `setInt`, `getBool`, …) over the key/value store.
- `phoenix_has_sql()` reports whether the loaded library includes the SQL
  layer, so an app can degrade gracefully on a lean embedded build.

### Fixed

- **The package could not be used as an ordinary dependency.** Every native
  library search path was relative to the *consumer's* working directory, but
  the binaries ship inside the installed package (in the pub cache, or at a
  `path:` dependency's location). A plain `dart pub get` followed by
  `dart run` failed with "Could not load phoenixdb.dll". The loader now
  resolves its own package root first — via `Isolate.resolvePackageUriSync`,
  falling back to reading `.dart_tool/package_config.json`, which is what the
  `flutter test` runner needs.
- **Windows lookups missed MSVC builds.** Only `x86_64-pc-windows-gnu` was
  searched, so the MSVC library that CI ships would not be found. Both triples
  are now searched, and `windows_arm64` was added.
- **Flutter and script builds produced an unloadable library.** The Linux and
  Windows CMake files, the Apple build script, `build.sh`, and `build.ps1` all
  ran `cargo build` without `--features sql`, yielding an ABI v1 library that
  the v2 loader rejects. All build paths now enable the feature.
- **Platform manifests still declared 0.1.0.** The iOS and macOS podspecs and
  `android/build.gradle` were never bumped, so CocoaPods and Gradle advertised
  a version that no longer matched the package.
- **`dart pub get` failed for plain Dart consumers.** `pubspec.yaml` declared a
  `flutter:` constraint under `environment:`, which makes the whole package
  require the Flutter SDK:

  ```
  Because phoenixdb requires the Flutter SDK, version solving failed.
  ```

  Nothing under `lib/` imports `package:flutter` — the only package imports are
  `ffi` and `phoenixdb` — so the constraint was never warranted. Flutter
  support comes from the `flutter: plugin:` section, which plain Dart ignores.
  The constraint is removed and CI now guards against its return.
- **A missing Rust toolchain failed opaquely.** A desktop Flutter build without
  `cargo` on PATH stopped at `Error 1` from the custom build command, with no
  indication that Rust was the cause. The Linux and Windows CMake files now
  fail configuration with an explicit message pointing at rustup.

### Changed

- Feature flags (`encryption`, `json`, `sql`, `metrics`, `async-runtime`,
  `full`) keep the default build lean for Flutter. The default build adds no
  heavy dependencies.
- `serde_json` is now optional, behind the `json` feature.

### Notes on dependencies

Three crates named in the original design could not be used, because they
require a C toolchain that is unavailable on the Flutter/Android
cross-compilation path. Substitutions with the same guarantees were used
instead:

| Planned | Shipped | Reason |
| --- | --- | --- |
| `ring` | `aes-gcm` | Pure Rust, same AES-256-GCM construction |
| `sqlparser-rs` | hand-written parser | Its `stacker` dependency needs a C compiler |
| OpenTelemetry OTLP | internal tracing | `tonic` pulls in `cc`-based dependencies |

### Known limitations

- **The `lsm`, `security::encryption`, `security::rbac`, `security::audit` and
  `observability` modules are libraries, not yet engine behaviour.** They are
  implemented and tested, but `Database` still writes through the B+Tree only,
  the pager does not encrypt, no permission check runs at the FFI boundary, and
  the engine emits no metrics. Key and value length validation from `security`
  *is* enforced on every FFI call.
- `SELECT` scans the visible key space per query rather than using a
  prefix-bounded iterator: appropriate for embedded workloads, O(database) on
  large tables.
- The SQL layer has no planner, joins, subqueries, or aggregate functions.
- Raft replication, gossip anti-entropy, full-text search, and the REPL are
  not implemented.
- Web is not supported: the engine is native code reached over `dart:ffi`.

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

**Packaging**
- Installable with `dart pub add phoenixdb` and `flutter pub add phoenixdb`.
- Declared as a Flutter FFI plugin (`ffiPlugin: true`) for Android, iOS, macOS,
  Linux and Windows, so no method-channel registrant code is generated.
- Android: prebuilt `.so` for `arm64-v8a`, `armeabi-v7a`, `x86_64` and `x86`
  shipped in `android/src/main/jniLibs/`, packaged into the host APK/AAB with
  no NDK required by the consumer (minSdk 21).
- iOS and macOS: CocoaPods podspecs that compile a static archive during the
  Xcode build via `rust/build-apple.sh`; the archive is `-force_load`ed so the
  `phoenix_*` symbols survive dead-stripping.
- Linux and Windows: CMake integration that runs `cargo build` and bundles the
  resulting shared library with the app.
- Platform-aware library loading: bare-name `dlopen` on Android,
  `DynamicLibrary.process()` on iOS (statically linked), and a per-target-triple
  filesystem search on desktop.

**Tooling**
- `build.sh` and `build.ps1` with cross-compilation support via `CC`/`AR`.
- `libfuzzer` harnesses for the B+Tree, page parsing and the FFI surface.
- 86 Rust tests and 33 Dart tests; `dart analyze --fatal-infos` clean.
- Scores 160/160 on pub.dev's package analysis (`pana`).

### License

Released under the BSD 3-Clause License.
