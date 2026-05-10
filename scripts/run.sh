#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_STAGE="stage1"
VALIDATION_DIR="STL10/val"
PYTHON_BIN="${PYTHON_BIN:-python}"

print_usage() {
  printf 'Usage: bash scripts/run.sh <stage1|stage2|stage3|stage4> [baseline_experiment]\n'
}

validate_stage() {
  case "$1" in
    stage1|stage2|stage3|stage4) ;;
    *)
      print_usage
      exit 1
      ;;
  esac
}

ensure_validation_split() {
  if [ -d "$VALIDATION_DIR" ]; then
    return
  fi

  printf 'Validation split not found at %s\n' "$VALIDATION_DIR"
  printf 'Create it first with: python utils/split.py\n'
  exit 1
}

print_header() {
  printf '============================================\n'
  printf 'Running %s experiments\n' "$1"
  if [ -n "$2" ]; then
    printf 'Using baseline: %s\n' "$2"
  fi
  printf '============================================\n'
}

main() {
  local stage="${1:-$DEFAULT_STAGE}"
  local baseline="${2:-}"

  cd "$PROJECT_DIR"

  validate_stage "$stage"
  ensure_validation_split
  print_header "$stage" "$baseline"

  export STAGE="$stage"
  export BASELINE="$baseline"
  export PYTHON_BIN

  "$PYTHON_BIN" - <<'PY'
import os
import subprocess

from configs.experiments import build_stage_experiments

stage = os.environ['STAGE']
baseline = os.environ['BASELINE'] or None
python_bin = os.environ['PYTHON_BIN']

experiments = build_stage_experiments(stage, baseline=baseline)
if not experiments:
    raise SystemExit(f'No experiments generated for {stage}')

for config in experiments:
    print('=' * 60)
    print(f'Training: {config.name}')
    print('=' * 60)
    subprocess.run(
        [python_bin, 'scripts/train.py', '--experiment', config.name],
        check=True,
    )
PY
}

main "$@"
