#!/usr/bin/env bash
#
# Builds PhoenixDB as a static archive for Apple platforms.
#
# Invoked automatically by the iOS/macOS podspecs during `flutter build`, and
# usable directly:
#
#   ./build-apple.sh ios     # device + simulator, merged per-arch
#   ./build-apple.sh macos   # x86_64 + arm64 universal
#
# Apple targets must be built on macOS with Xcode installed, because linking
# requires the platform SDK and (for device builds) code signing.

set -euo pipefail

PLATFORM="${1:-ios}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "!! Apple targets can only be built on macOS (found $(uname -s))" >&2
  exit 1
fi

command -v cargo >/dev/null 2>&1 || {
  echo "!! cargo not found. Install Rust: https://rustup.rs" >&2
  exit 1
}

# Xcode invokes script phases with a sanitised environment.
export PATH="$HOME/.cargo/bin:$PATH"

build_target() {
  local triple="$1"
  rustup target add "$triple" >/dev/null 2>&1 || true
  echo ">> building $triple (features: sql)"
  # `sql` is required, not optional: `phoenix_sql_query` and `phoenix_has_sql`
  # are part of ABI v2, and the Dart loader rejects a library reporting a
  # different version.
  ( cd "$SCRIPT_DIR" && cargo build --lib --release --features sql --target "$triple" )
  echo "$SCRIPT_DIR/target/$triple/release/libphoenixdb.a"
}

case "$PLATFORM" in
  ios)
    # Device is arm64 only; the simulator needs both so it runs on Intel and
    # Apple silicon Macs. They cannot be lipo'd into one archive because
    # device-arm64 and sim-arm64 share a slice name, so the correct archive is
    # selected from the SDK Xcode is currently building for.
    if [[ "${PLATFORM_NAME:-iphoneos}" == "iphonesimulator" ]]; then
      libs=()
      libs+=("$(build_target aarch64-apple-ios-sim)")
      libs+=("$(build_target x86_64-apple-ios)")
      lipo -create "${libs[@]}" -output "$REPO_ROOT/ios/libphoenixdb.a"
    else
      lib="$(build_target aarch64-apple-ios)"
      cp -f "$lib" "$REPO_ROOT/ios/libphoenixdb.a"
    fi
    echo ">> installed -> $REPO_ROOT/ios/libphoenixdb.a"
    ;;
  macos)
    a="$(build_target aarch64-apple-darwin)"
    b="$(build_target x86_64-apple-darwin)"
    lipo -create "$a" "$b" -output "$REPO_ROOT/macos/libphoenixdb.a"
    echo ">> installed -> $REPO_ROOT/macos/libphoenixdb.a (universal)"
    ;;
  *)
    echo "usage: $0 [ios|macos]" >&2
    exit 2
    ;;
esac
