#!/bin/bash
set -e

echo "============================================"
echo "  STL-10 Image Classification Experiments"
echo "============================================"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -d "STL10/val" ]; then
    echo "Splitting dataset (80/20 train/val)..."
    python scripts/split.py
else
    echo "Validation set already exists, skipping split."
fi

echo "Running all experiments..."
mkdir -p outputs/models outputs/logs outputs/figures

CONFIGS=(
    "04_avgpool"
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Training: config_${cfg}"
    echo "========================================"
    python scripts/train.py --config "configs.config_${cfg}"
    echo "  Finished: config_${cfg}"
done

echo ""
echo "Running inference on all models..."
echo "========================================"
python scripts/infer.py --all

echo ""
echo "============================================"
echo "  All experiments completed!"
echo "  Models:   outputs/models/"
echo "  Logs:     outputs/logs/"
echo "  Figures:  outputs/figures/"
echo "============================================"
