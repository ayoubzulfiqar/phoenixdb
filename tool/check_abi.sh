#!/usr/bin/env bash
# Verifies that prebuilt native libraries carry the current ABI surface.
#
# Why this exists: PhoenixDB commits prebuilt binaries into the published
# package, and the Dart loader enforces an *exact* ABI match. One library built
# before an ABI bump does not degrade a feature — it makes the package unable
# to open a database on that platform at all. This is the gate that stops such
# a release, and it is shared by the CI and release workflows so the rule is
# defined once.
#
# Usage:
#   tool/check_abi.sh <file>...     check specific libraries
#   tool/check_abi.sh --shipped     check everything in the package layout
set -euo pipefail

# Symbols that must be present in an ABI v2 library.
REQUIRED=(phoenix_open phoenix_sql_query phoenix_has_sql)

if [[ "${1:-}" == "--shipped" ]]; then
  mapfile -t files < <(find native android -type f \
    \( -name '*.so' -o -name '*.dylib' -o -name '*.dll' \) 2>/dev/null | sort)
  [[ ${#files[@]} -gt 0 ]] || { echo "no native libraries found" >&2; exit 1; }
else
  files=("$@")
  [[ ${#files[@]} -gt 0 ]] || { echo "usage: $0 <file>... | --shipped" >&2; exit 2; }
fi

fail=0
for lib in "${files[@]}"; do
  if [[ ! -f "$lib" ]]; then
    echo "::error::missing $lib"
    fail=1
    continue
  fi
  missing=()
  for sym in "${REQUIRED[@]}"; do
    # -a: treat the binary as text. Symbol names live in the export table as
    # plain ASCII on every format we ship (ELF, Mach-O, PE), so this is a
    # portable check that needs no nm/objdump on the runner.
    grep -qa "$sym" "$lib" || missing+=("$sym")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "::error::$lib is stale — missing: ${missing[*]}"
    fail=1
  else
    echo "ok: $lib"
  fi
done

if [[ $fail -ne 0 ]]; then
  echo
  echo "A stale library cannot be published: the Dart loader requires an exact"
  echo "ABI match and will refuse to open a database against it."
fi
exit "$fail"
