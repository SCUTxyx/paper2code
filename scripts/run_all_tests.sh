#!/usr/bin/env bash
# 全量测试:校准集(5 篇考卷)+ examples/ 全部 repro 产物。
# 用法: bash scripts/run_all_tests.sh
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

PY="${PYTHON:-python}"
rc=0

echo "=== 校准集 ==="
bash scripts/run_calibration.sh || rc=1

echo ""
echo "=== examples(真实论文 repro 产物)==="
for d in examples/*/; do
  [ -d "$d/tests" ] || continue
  echo "--- $d"
  "$PY" -m pytest "$d/tests" -q || rc=1
done

exit $rc
