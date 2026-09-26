#!/usr/bin/env bash
# Skill-level regression test: run each of the 5 exams under pytest, append results
# to CALIBRATION_LOG.md. Pass criterion: 5/5 green (PLAN §4.3). Mandatory after any
# change to SKILL.md or shared scripts.
#
# Verdicts are gated on pytest EXIT CODES (0=pass, 1=failures, 5=no tests collected —
# also a failure), never on parsing output text: the output format may change; the
# regression gate must not silently break with it.
#
# Usage: conda activate test && bash scripts/run_calibration.sh
#   PYTHON=/path/to/python bash scripts/run_calibration.sh   # or point at an interpreter
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# honor $PYTHON; else prefer python, fall back to python3 (stock macOS has no `python`)
PY="${PYTHON:-}"
[ -z "$PY" ] && PY="$(command -v python || command -v python3)"
[ -z "$PY" ] && { echo "no python/python3 on PATH"; exit 2; }
LOG="CALIBRATION_LOG.md"
passed=0; failed=0; detail_lines=""

for d in calibration/*/; do
  [ -d "$d/tests" ] || continue
  name=$(basename "$d")
  # no pipeline inside the command substitution, so rc is pytest's own exit code
  out_full="$("$PY" -m pytest "$d/tests" -q 2>&1)"
  rc=$?
  out="$(printf '%s\n' "$out_full" | tail -1)"
  if [ "$rc" -eq 0 ]; then
    passed=$((passed+1)); detail_lines="${detail_lines}| $name | ✅ $out |"$'\n'
  else
    failed=$((failed+1))
    reason="exit=$rc"
    [ "$rc" -eq 5 ] && reason="no tests collected"
    detail_lines="${detail_lines}| $name | ❌ $out ($reason) |"$'\n'
  fi
done

total=$((passed+failed))
if [ "$failed" -eq 0 ]; then
  verdict="✅ ${passed}/${total} all green"
else
  verdict="❌ ${failed}/${total} exam(s) failed"
fi

{
  echo ""
  echo "## $(date '+%Y-%m-%d %H:%M:%S') — $verdict"
  echo ""
  echo "| Exam | pytest result |"
  echo "|---|---|"
  printf '%s' "$detail_lines"
} >> "$LOG"

tail -n $((total + 5)) "$LOG"
echo ""
if [ "$failed" -gt 0 ]; then exit 1; fi
