# Contributing to PhoenixDB

Thanks for helping. PhoenixDB is a Rust storage engine behind a C ABI, with
Dart/Flutter bindings on top, so most changes touch more than one layer. This
guide covers setting up, the checks every change must pass, and the few rules
that keep the layers in step.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security problems privately as described in [SECURITY.md](SECURITY.md), not in
a public issue.

## Where things live

| Path | What it is |
| --- | --- |
| `rust/src/` | The engine: B+Tree, pager, WAL, MVCC (`lib.rs`, `txn.rs`), SQL (`sql/`), vector index (`vector/`), document collections (`collection/`) |
| `rust/src/ffi/` | The C ABI consumed by Dart; `native/include/phoenixdb.h` is generated from it |
| `rust/tests/`, `rust/fuzz/` | Integration, durability and FFI-safety tests; fuzz targets |
| `lib/` | The Dart package (`phoenixdb.dart`) and the AI toolkit (`ai.dart`, `lib/src/ai/`) |
| `test/` | Dart tests, run against the real native library |
| `tool/`, `.github/workflows/` | ABI checks, artifact staging and CI/release pipelines |

## Setting up

You need a stable Rust toolchain (1.89 or newer) and the Flutter SDK. Flutter
is required even for pure-Dart work: the package declares Flutter plugin
platforms, so plain `dart pub get` refuses it.

```bash
git clone https://github.com/ayoubzulfiqar/phoenixdb.git
cd phoenixdb
./build.sh          # release build of the native library into native/
flutter pub get
dart test
```

`./build.sh` builds with the `sql` feature, which shipped binaries always
include; the Dart tests expect it. On Windows use `.\build.ps1`.

## Checks

CI runs all of these on Linux, macOS and Windows. Run them before opening a
pull request:

```bash
cd rust
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features full -- -D warnings
cargo test
cargo test --features full
(cd fuzz && cargo check --bins)
cd ..
./build.sh            # rebuild the library the Dart tests load
dart analyze
dart test
```

Both feature configurations matter: the default build is what ships inside
Flutter apps, `full` is what servers use, and a feature-gate mistake shows up
in only one of them.

## Rules that keep the layers in step

- **Commit the regenerated C header.** Every build rewrites
  `native/include/phoenixdb.h`; CI fails if the committed copy is stale.
  Public Rust constants leak into the header as macros unless excluded in
  `rust/cbindgen.toml` — exclude anything that is not part of the C ABI.
- **ABI changes are versioned.** The Dart loader requires an exact ABI match.
  Adding or changing an entry point means bumping `phoenix_abi_version` in
  `rust/src/ffi/mod.rs` and `kExpectedAbiVersion` in `lib/src/bindings.dart`,
  and adding representative symbols to `tool/check_abi.sh` so stale binaries
  cannot be released.
- **FFI entry points follow the boundary contract** documented at the top of
  `rust/src/ffi/mod.rs`: validate every pointer and length before touching it,
  run the body inside `catch_unwind`, return a status code, and free
  library-owned memory only through the library's own free functions.
- **Durability claims need a crash test.** Changes to the WAL, pager,
  checkpoints or recovery should come with a test in `rust/tests/durability.rs`
  (or the relevant module) that simulates the crash it guards against.
- **Keep the Dart library pure Dart.** Nothing under `lib/` may import
  `package:flutter`, and the AI toolkit uses `dart:io` only — no new
  dependencies without discussion.

## Pull requests

- Keep each pull request to one logical change, with tests.
- Write commit messages in the [Conventional Commits](https://www.conventionalcommits.org/)
  style used throughout the history: `fix(wal): ...`, `feat(ai): ...`,
  `test(sql): ...`, `docs: ...`. Explain *why* in the body when it is not
  obvious.
- Add user-visible changes to `CHANGELOG.md` under an unreleased heading.
- Update `README.md` when behavior or the public API changes.

## Reporting bugs and requesting features

Use the issue templates. For bugs, include the PhoenixDB version, platform, a
minimal reproduction and the full error — a failing test case is ideal. If a
database file is involved, `cd rust && cargo run --bin phoenixdb_verify -- <file>`
checks its structure, and its output is very helpful.

## License

By contributing you agree that your contributions are licensed under the
[BSD 3-Clause License](LICENSE) that covers the project.
