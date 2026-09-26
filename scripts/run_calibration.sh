#!/usr/bin/env bash
# skill 级回归测试:5 篇考卷逐一 pytest,结果追加到 CALIBRATION_LOG.md。
# 通过标准:5/5 全绿(PLAN §4.3)。改 SKILL.md 或共享脚本后必须重跑。
#
# 判定基于 pytest 退出码(0=通过,1=有失败,5=没收集到测试——也算失败),
# 不解析输出文本:输出格式变了不能让回归判定跟着失效。
#
# 用法: conda activate test && bash scripts/run_calibration.sh
#   PYTHON=/path/to/python bash scripts/run_calibration.sh   # 或显式指定解释器
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

PY="${PYTHON:-python}"
LOG="CALIBRATION_LOG.md"
passed=0; failed=0; detail_lines=""

for d in calibration/*/; do
  [ -d "$d/tests" ] || continue
  name=$(basename "$d")
  # 命令替换内不带管道,这样 rc 就是 pytest 本身的退出码
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
  verdict="✅ ${passed}/${total} 全绿"
else
  verdict="❌ ${failed}/${total} 个考卷未通过"
fi

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
