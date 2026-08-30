#!/usr/bin/env bash
# Runs every tests/test_*.py file (each has its own standalone runner, no
# pytest required) and reports an aggregate pass/fail. Exits non-zero if
# any file failed, so this is CI-friendly as-is (see .github/workflows/ci.yml).
set -uo pipefail

cd "$(dirname "$0")"

total_pass=0
total_fail=0
failed_files=()

for f in tests/test_*.py; do
    echo "=== $f ==="
    output="$(python3 "$f" 2>&1)"
    status=$?
    echo "$output" | tail -3
    echo

    if [ $status -ne 0 ]; then
        failed_files+=("$f")
    fi

    counts="$(echo "$output" | grep -Eo '[0-9]+ passed, [0-9]+ failed' | tail -1)"
    if [ -n "$counts" ]; then
        p="$(echo "$counts" | grep -Eo '^[0-9]+')"
        fcount="$(echo "$counts" | grep -Eo '[0-9]+ failed' | grep -Eo '^[0-9]+')"
        total_pass=$((total_pass + p))
        total_fail=$((total_fail + fcount))
    fi
done

echo "============================================================"
echo "TOTAL: $total_pass passed, $total_fail failed, across $(ls tests/test_*.py | wc -l | tr -d ' ') files"
echo "============================================================"

if [ ${#failed_files[@]} -gt 0 ]; then
    echo "Files with failures or errors:"
    printf '  %s\n' "${failed_files[@]}"
    exit 1
fi

exit 0
