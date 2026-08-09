#!/usr/bin/env bash
#
# Builds the PhoenixDB native library and installs it into the Dart package's
# native/ folder.
#
# Usage:
#   ./build.sh                      # release build for the host
#   ./build.sh --debug              # debug build
#   ./build.sh --target <triple>    # cross-compile
#   ./build.sh --all                # every supported target
#   ./build.sh --zig                # force the zig cross-linker
#
# Cross-compilation:
#   Cross builds need a linker for the target platform. The most reliable
#   option from any host is `cargo-zigbuild`, which uses Zig's bundled clang
#   and works for Linux and macOS without an Xcode installation:
#
#     cargo install cargo-zigbuild && winget install zig.zig   # or brew/apt
#     ./build.sh --all
#
#   The script auto-detects cargo-zigbuild and uses it for non-host targets.
#   Otherwise it falls back to plain cargo, honouring the standard CC / AR
#   variables:
#
#     CC=aarch64-linux-gnu-gcc AR=aarch64-linux-gnu-ar \
#       ./build.sh --target aarch64-unknown-linux-gnu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/rust"
NATIVE_DIR="$SCRIPT_DIR/native"

PROFILE="release"
PROFILE_FLAG="--release"
TARGETS=()
BUILD_ALL=0
FORCE_ZIG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      PROFILE="debug"
      PROFILE_FLAG=""
      shift
      ;;
    --target)
      TARGETS+=("$2")
      shift 2
      ;;
    --all)
      BUILD_ALL=1
      shift
      ;;
    --zig)
      FORCE_ZIG=1
      shift
      ;;
    -h|--help)
      sed -n '2,30p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ $BUILD_ALL -eq 1 ]]; then
  TARGETS=(
    x86_64-unknown-linux-gnu
    aarch64-unknown-linux-gnu
    x86_64-apple-darwin
    aarch64-apple-darwin
  )
fi

# Host triple, so we know which targets actually need cross-linking.
HOST_TRIPLE="$(rustc -vV | awk '/^host:/ {print $2}')"

# cargo-zigbuild handles Linux and macOS cross-linking from any host.
HAVE_ZIGBUILD=0
if command -v cargo-zigbuild >/dev/null 2>&1; then
  HAVE_ZIGBUILD=1
fi

# Maps a Rust target triple to the produced shared-library file name.
lib_name_for() {
  case "$1" in
    *windows*) echo "phoenixdb.dll" ;;
    *apple*)   echo "libphoenixdb.dylib" ;;
    *)         echo "libphoenixdb.so" ;;
  esac
}

host_lib_name() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) echo "phoenixdb.dll" ;;
    Darwin)               echo "libphoenixdb.dylib" ;;
    *)                    echo "libphoenixdb.so" ;;
  esac
}

mkdir -p "$NATIVE_DIR"

build_one() {
  local target="${1:-}"
  local subcmd="build"
  local args=(--lib)
  [[ -n "$PROFILE_FLAG" ]] && args+=("$PROFILE_FLAG")

  local out_dir lib_name
  if [[ -n "$target" ]]; then
    args+=(--target "$target")
    out_dir="$RUST_DIR/target/$target/$PROFILE"
    lib_name="$(lib_name_for "$target")"

    # Use zigbuild when cross-compiling (or when explicitly requested), since
    # the host linker cannot produce ELF or Mach-O output.
    if [[ $FORCE_ZIG -eq 1 ]] || { [[ "$target" != "$HOST_TRIPLE" ]] && [[ $HAVE_ZIGBUILD -eq 1 ]]; }; then
      subcmd="zigbuild"
    fi
    echo ">> building for $target ($PROFILE, cargo $subcmd)"
  else
    out_dir="$RUST_DIR/target/$PROFILE"
    lib_name="$(host_lib_name)"
    echo ">> building for the host ($PROFILE)"
  fi

  (cd "$RUST_DIR" && cargo "$subcmd" "${args[@]}")

  local src="$out_dir/$lib_name"
  if [[ ! -f "$src" ]]; then
    echo "!! expected library not found: $src" >&2
    return 1
  fi

  # Host builds land in native/; cross builds go in native/<triple>/ so several
  # architectures can be shipped side by side.
  local dest_dir="$NATIVE_DIR"
  [[ -n "$target" ]] && dest_dir="$NATIVE_DIR/$target" && mkdir -p "$dest_dir"
  cp -f "$src" "$dest_dir/$lib_name"
  echo "   installed -> $dest_dir/$lib_name"
}

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  build_one ""
else
  for t in "${TARGETS[@]}"; do
    build_one "$t" || echo "!! target $t failed (is the toolchain installed?)" >&2
  done
fi

# The header is emitted by build.rs into native/include/phoenixdb.h.
if [[ -f "$NATIVE_DIR/include/phoenixdb.h" ]]; then
  echo ">> header: $NATIVE_DIR/include/phoenixdb.h"
fi

echo ">> done"
