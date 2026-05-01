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

AUGMENT_CONFIGS=(
    "01_baseline"
    "02_color_jitter"
    "02_flip_h"
    "02_random_crop"
)

NORM_CONFIGS=(
    "01_baseline"
    "02_flip_h"
    "03_adamw"
    "03_batchnorm"
    "03_dropout"
)

MODEL_CONFIGS=(
    "01_baseline"
    "03_adamw"
    "04_avgpool"
    "04_deep"
    "04_sigmoid"
)

GROUPS=("AUGMENT" "NORM" "MODEL")

for group in "${GROUPS[@]}"; do
    declare -n configs="${group}_CONFIGS"

    for cfg in "${configs[@]}"; do
        echo ""
        echo "========================================"
        echo "  Training: config_${cfg}"
        echo "========================================"
        python scripts/train.py --config "configs.config_${cfg}"
        echo "  Finished: config_${cfg}"
    done

    echo ""
    echo "============================================"
    echo "Running inference on ${group} group models..."
    echo "============================================"
    python scripts/infer.py --model "${configs[@]}" --output-dir "outputs/figures/${group}"
done

echo ""
echo "============================================"
echo "  All experiments completed!"
echo "============================================"
