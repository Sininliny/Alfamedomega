#!/usr/bin/env bash
# Format first-party C++ sources with clang-format.
#   scripts/format.sh          rewrite files in place
#   scripts/format.sh --check  report unformatted files and exit non-zero (used by CI)
set -euo pipefail
cd "$(dirname "$0")/.."

files=$(git ls-files 'src/*.h' 'src/*.hpp' 'src/*.cpp' 'tests/*.h' 'tests/*.cpp')
[ -z "$files" ] && exit 0

if [ "${1:-}" = "--check" ]; then
    # shellcheck disable=SC2086
    clang-format --dry-run --Werror $files
else
    # shellcheck disable=SC2086
    clang-format -i $files
fi
