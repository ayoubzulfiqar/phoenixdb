#!/usr/bin/env bash
# Copies downloaded CI artifacts into the layout the published package expects.
#
# Shared by the CI and release workflows: both download the same per-target
# artifacts and must arrange them identically, so the mapping lives here rather
# than being restated in each workflow.
#
# Usage: tool/stage_artifacts.sh <artifacts-dir>
set -euo pipefail

src="${1:?usage: $0 <artifacts-dir>}"

for dir in "$src"/native-*; do
  [[ -d "$dir" ]] || continue
  triple="${dir##*/native-}"

  if [[ "$triple" == android ]]; then
    # cargo-ndk already emits the jniLibs/<abi>/ tree.
    mkdir -p android/src/main/jniLibs
    cp -r "$dir"/* android/src/main/jniLibs/
    echo "staged android jniLibs"
  else
    mkdir -p "native/$triple"
    find "$dir" -type f -exec cp {} "native/$triple/" \;
    echo "staged native/$triple"
  fi
done
