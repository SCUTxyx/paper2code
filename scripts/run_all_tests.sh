#!/usr/bin/env bash
# Full test run: the calibration set (5 exams) + every repro under examples/.
# Usage: bash scripts/run_all_tests.sh
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

PY="${PYTHON:-python}"
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
