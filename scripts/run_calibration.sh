#!/usr/bin/env bash
# skill 级回归测试:5 篇考卷逐一 pytest,结果追加到 CALIBRATION_LOG.md。
# 通过标准:5/5 全绿(PLAN §4.3)。改 SKILL.md 或共享脚本后必须重跑。
#
# 用法: bash scripts/run_calibration.sh
#   PYTHON=/path/to/python bash scripts/run_calibration.sh   # 指定解释器
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

PY="${PYTHON:-python}"
LOG="CALIBRATION_LOG.md"
passed=0; failed=0; detail_lines=""

for d in calibration/*/; do
  [ -d "$d/tests" ] || continue
  name=$(basename "$d")
  out="$("$PY" -m pytest "$d/tests" -q 2>&1 | tail -1)"
  if echo "$out" | grep -q " failed\| error"; then
    failed=$((failed+1)); detail_lines="${detail_lines}| $name | ❌ $out |"$'\n'
  else
    passed=$((passed+1)); detail_lines="${detail_lines}| $name | ✅ $out |"$'\n'
  fi
done

total=$((passed+failed))
verdict="✅ 5/5 全绿"
[ "$failed" -gt 0 ] && verdict="❌ $failed/$total 个考卷未通过"

{
  echo ""
  echo "## $(date '+%Y-%m-%d %H:%M:%S') — $verdict"
  echo ""
  echo "| 考卷 | pytest 结果 |"
  echo "|---|---|"
  printf '%s' "$detail_lines"
} >> "$LOG"

tail -n $((total + 5)) "$LOG"
echo ""
if [ "$failed" -gt 0 ]; then exit 1; fi
