#!/usr/bin/env bash
# Copies downloaded CI artifacts into the layout the published package expects.
#
# Shared by the CI and release workflows: both download the same per-target
# artifacts and must arrange them identically, so the mapping lives here rather
# than being restated in each workflow.
#
# Two artifact naming schemes are accepted, because the workflows differ:
#
#   native-android              — one artifact holding the whole jniLibs/ tree
#                                 (ci.yml, release.yml: one `cargo ndk` call
#                                 emits all four ABIs at once)
#   native-android-<abi>        — one artifact per ABI, holding that ABI's
#                                 directory contents (build_native.yml, which
#                                 splits the ABIs so arm64-v8a can be built
#                                 with NEON and armeabi-v7a without it)
#   native-<triple>             — a desktop or Apple library
#
# Getting the android cases wrong is not cosmetic: an `android-arm64-v8a`
# artifact treated as a triple lands a .so in native/, where the Dart loader
# would never look, and the real jniLibs slot stays empty.
#
# Usage: tool/stage_artifacts.sh <artifacts-dir>
set -euo pipefail

src="${1:?usage: $0 <artifacts-dir>}"

staged=0
for dir in "$src"/native-*; do
  [[ -d "$dir" ]] || continue
  name="${dir##*/native-}"

  case "$name" in
    android)
      # cargo-ndk already emitted the jniLibs/<abi>/ tree.
      mkdir -p android/src/main/jniLibs
      cp -r "$dir"/* android/src/main/jniLibs/
      echo "staged android jniLibs (all ABIs)"
      ;;
    android-*)
      # Per-ABI upload: the artifact holds that one ABI's directory contents.
      abi="${name#android-}"
      mkdir -p "android/src/main/jniLibs/$abi"
      find "$dir" -type f -exec cp {} "android/src/main/jniLibs/$abi/" \;
      echo "staged android jniLibs/$abi"
      ;;
    *)
      mkdir -p "native/$name"
      find "$dir" -type f -exec cp {} "native/$name/" \;
      echo "staged native/$name"
      ;;
  esac
  staged=$((staged + 1))
done

if [[ $staged -eq 0 ]]; then
  echo "::error::no native-* artifacts found under $src" >&2
  exit 1
fi
echo "staged $staged artifact(s)"
