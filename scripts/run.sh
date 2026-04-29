#!/bin/bash
set -e

echo "============================================"
echo "  STL-10 Image Classification Experiments"
echo "============================================"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
fi

echo "[1/5] Activating virtual environment..."
source venv/bin/activate

echo "[2/5] Installing dependencies..."
pip install -r requirements.txt -q

if [ ! -d "STL10/val" ]; then
    echo "[3/5] Splitting dataset (80/20 train/val)..."
    python scripts/split.py
else
    echo "[3/5] Validation set already exists, skipping split."
fi

echo "[4/5] Running all experiments..."
mkdir -p outputs/models outputs/logs outputs/figures

CONFIGS=(
    "01_baseline"
    "02_augment"
    "03_mixup"
    "04_residual"
    "05_deep"
    "06_sigmoid"
    "07_avgpool"
    "08_batchnorm"
    "09_adam"
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
echo "[5/5] Running inference on all models..."
echo "========================================"
python scripts/infer.py --all

echo ""
echo "============================================"
echo "  All experiments completed!"
echo "  Models:   outputs/models/"
echo "  Logs:     outputs/logs/"
echo "  Figures:  outputs/figures/"
echo "============================================"
