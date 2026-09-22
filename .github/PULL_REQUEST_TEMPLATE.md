## What and why

<!-- What does this change, and why is it needed? Link the issue it fixes. -->

Fixes #

## How it was tested

<!-- New or updated tests, and anything verified by hand. -->

## Checklist

- [ ] `cargo fmt --all -- --check` and `cargo clippy --all-targets [--features full] -- -D warnings` pass
- [ ] `cargo test` and `cargo test --features full` pass
- [ ] `dart analyze` and `dart test` pass against a freshly built library (`./build.sh`)
- [ ] The regenerated `native/include/phoenixdb.h` is committed, if the C ABI changed
- [ ] ABI changes bump `phoenix_abi_version` / `kExpectedAbiVersion` and update `tool/check_abi.sh`
- [ ] Durability or recovery changes include a crash test
- [ ] User-visible changes are in `CHANGELOG.md`, and `README.md` is updated if the API changed
