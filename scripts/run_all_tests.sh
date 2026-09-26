#!/usr/bin/env bash
# Full test run: the calibration set (5 exams) + every repro under examples/.
# Usage: bash scripts/run_all_tests.sh
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# Interpreter selection: honor $PYTHON; else pick the first candidate that can
# actually import numpy+pytest (the repo's deps) — `python` on PATH may be an
# environment without the deps (e.g. unactivated conda base).
pick_python() {
  local c
  for c in "$PYTHON" "./venv/bin/python" "./.venv/bin/python" "python" "python3"; do
    [ -n "$c" ] || continue
    if "$c" -c "import numpy, pytest" >/dev/null 2>&1; then
      printf '%s' "$c"
      return 0
    fi
  done
  return 1
}
if ! PY="$(pick_python)"; then
  echo "no interpreter with numpy+pytest found." \
       "Run 'pip install -e .' in your environment first (and activate it),"
  echo "or point PYTHON=<python> at it explicitly."
  exit 2
fi
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
