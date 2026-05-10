#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

STAGE="${1:-stage1}"
BASELINE="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$STAGE" in
  stage1|stage2|stage3|stage4) ;;
  *)
    printf 'Usage: bash scripts/eval.sh <stage1|stage2|stage3|stage4> [baseline_experiment]\n'
    exit 1
    ;;
esac

export STAGE
export BASELINE
export PYTHON_BIN

"$PYTHON_BIN" - <<'PY'
import os
import subprocess

from configs.experiments import build_stage_experiments

stage = os.environ['STAGE']
baseline = os.environ['BASELINE'] or None
python_bin = os.environ['PYTHON_BIN']

names = [config.name for config in build_stage_experiments(stage, baseline=baseline)]
if not names:
    raise SystemExit(f'No experiments generated for {stage}')

subprocess.run(
    [python_bin, 'scripts/analysis.py', 'eval', '--model', *names, '--force'],
    check=True,
)
subprocess.run(
    [python_bin, 'scripts/analysis.py', 'tsne', '--model', *names],
    check=True,
)
subprocess.run(
    [python_bin, 'scripts/summarize_stage.py', '--stage', stage, '--baseline', baseline or ''],
    check=True,
)
PY
