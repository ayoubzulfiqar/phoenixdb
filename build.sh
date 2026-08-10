#!/usr/bin/env bash
#
# Builds the PhoenixDB native library and installs it into the Dart package's
# native/ folder.
#
# Usage:
#   ./build.sh                      # release build for the host
#   ./build.sh --debug              # debug build
#   ./build.sh --target <triple>    # cross-compile
#   ./build.sh --all                # every supported desktop target
#   ./build.sh --android            # all four Android ABIs -> jniLibs/
#   ./build.sh --zig                # force the zig cross-linker
#
# Android:
#   Needs the NDK. The script looks at ANDROID_NDK_HOME, then NDK_HOME, then
#   the newest NDK under $ANDROID_HOME/ndk. Output goes straight into
#   android/src/main/jniLibs/<abi>/ so Gradle packages it into the host APK.
#   Override the API level with ANDROID_API (default 21).
#
# Apple (iOS/macOS):
#   Use rust/build-apple.sh, which must run on macOS with Xcode installed.
#   The podspecs invoke it automatically during `flutter build`.
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
BUILD_ANDROID=0
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
    --android)
      BUILD_ANDROID=1
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
  # `sql` is required, not optional: `phoenix_sql_query` and `phoenix_has_sql`
  # are part of ABI v2, and the Dart loader rejects a library reporting a
  # different version.
  local args=(--lib --features sql)
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

# Builds every Android ABI with the NDK's clang wrappers and installs the
# results into android/src/main/jniLibs/<abi>/, the layout Gradle packages
# into the host APK.
build_android() {
  local ndk="${ANDROID_NDK_HOME:-${NDK_HOME:-}}"
  if [[ -z "$ndk" ]]; then
    # Fall back to the newest NDK under the default SDK location.
    local sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/AppData/Local/Android/Sdk}}"
    [[ -d "$sdk/ndk" ]] && ndk="$(find "$sdk/ndk" -maxdepth 1 -mindepth 1 -type d | sort -V | tail -1)"
  fi
  if [[ -z "$ndk" || ! -d "$ndk" ]]; then
    echo "!! Android NDK not found. Set ANDROID_NDK_HOME." >&2
    return 1
  fi
  echo ">> using NDK: $ndk"

  # Pick the prebuilt toolchain matching this host.
  local hosttag
  case "$(uname -s)" in
    Darwin)               hosttag="darwin-x86_64" ;;
    MINGW*|MSYS*|CYGWIN*) hosttag="windows-x86_64" ;;
    *)                    hosttag="linux-x86_64" ;;
  esac
  local ndkbin="$ndk/toolchains/llvm/prebuilt/$hosttag/bin"
  [[ -d "$ndkbin" ]] || { echo "!! NDK toolchain missing: $ndkbin" >&2; return 1; }

  # Rust needs a native path for the linker; MSYS-style paths are mangled.
  local ndkbin_native="$ndkbin"
  local ext=""
  if [[ "$hosttag" == "windows-x86_64" ]]; then
    ndkbin_native="$(cygpath -w "$ndkbin" 2>/dev/null || echo "$ndkbin")"
    ext=".cmd"
  fi

  # triple : jniLibs ABI dir : NDK clang wrapper prefix
  local specs=(
    "aarch64-linux-android:arm64-v8a:aarch64-linux-android${ANDROID_API:-21}"
    "armv7-linux-androideabi:armeabi-v7a:armv7a-linux-androideabi${ANDROID_API:-21}"
    "x86_64-linux-android:x86_64:x86_64-linux-android${ANDROID_API:-21}"
    "i686-linux-android:x86:i686-linux-android${ANDROID_API:-21}"
  )

  local spec triple abi prefix var
  for spec in "${specs[@]}"; do
    IFS=: read -r triple abi prefix <<<"$spec"
    rustup target add "$triple" >/dev/null 2>&1 || true

    # cargo reads CARGO_TARGET_<TRIPLE>_LINKER, upper-cased with dashes as _.
    var="CARGO_TARGET_$(tr '[:lower:]-' '[:upper:]_' <<<"$triple")_LINKER"
    export "$var=$ndkbin_native/$prefix-clang$ext"

    echo ">> building for $triple ($PROFILE, features: sql) -> $abi"
    local args=(build --lib --features sql --target "$triple")
    [[ -n "$PROFILE_FLAG" ]] && args+=("$PROFILE_FLAG")
    ( cd "$RUST_DIR" && cargo "${args[@]}" )

    local src="$RUST_DIR/target/$triple/$PROFILE/libphoenixdb.so"
    [[ -f "$src" ]] || { echo "!! missing $src" >&2; return 1; }
    mkdir -p "$SCRIPT_DIR/android/src/main/jniLibs/$abi"
    cp -f "$src" "$SCRIPT_DIR/android/src/main/jniLibs/$abi/"
    echo "   installed -> android/src/main/jniLibs/$abi/libphoenixdb.so"
  done
}

if [[ $BUILD_ANDROID -eq 1 ]]; then
  build_android
  echo ">> done"
  exit 0
fi

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
