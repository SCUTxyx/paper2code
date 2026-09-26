#!/usr/bin/env bash
# Full test run: the calibration set (5 exams) + every repro under examples/.
# Usage: bash scripts/run_all_tests.sh
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# honor $PYTHON; else prefer python, fall back to python3 (stock macOS has no `python`)
PY="${PYTHON:-}"
[ -z "$PY" ] && PY="$(command -v python || command -v python3)"
[ -z "$PY" ] && { echo "no python/python3 on PATH"; exit 2; }
rc=0

echo "=== calibration set ==="
bash scripts/run_calibration.sh || rc=1

echo ""
echo "=== examples (real-paper repro artifacts) ==="
for d in examples/*/; do
  [ -d "$d/tests" ] || continue
  echo "--- $d"
  "$PY" -m pytest "$d/tests" -q || rc=1
done

exit $rc
