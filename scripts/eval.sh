#!/bin/bash
set -e

echo "============================================"
echo "  Generating LaTeX Evaluation Tables"
echo "============================================"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="/root/miniconda3/envs/DL/bin/python"
METRICS="accuracy precision_macro recall_macro f1_macro auc_ovr"

OUTDIR="$PROJECT_DIR/outputs/tables"
mkdir -p "$OUTDIR"

AUGMENT_MODELS=("01_baseline" "02_color_jitter" "02_flip_h" "02_random_crop")
NORM_MODELS=("01_baseline" "02_flip_h" "03_adamw" "03_batchnorm" "03_dropout")
MODEL_MODELS=("01_baseline" "03_adamw" "04_avgpool" "04_deep" "04_sigmoid")

declare -A GROUP_MAP
GROUP_MAP["AUGMENT"]="${AUGMENT_MODELS[*]}"
GROUP_MAP["NORM"]="${NORM_MODELS[*]}"
GROUP_MAP["MODEL"]="${MODEL_MODELS[*]}"

for group in AUGMENT NORM MODEL; do
    models=(${GROUP_MAP[$group]})
    echo ""
    echo "----------------------------------------"
    echo "  Group: $group (${#models[@]} models)"
    echo "----------------------------------------"

    for src in test_metrics val_metrics; do
        echo "  -> $src"
        $PYTHON scripts/analysis.py table \
            --model "${models[@]}" \
            --metrics $METRICS \
            --source "$src" \
            --table-output "$OUTDIR/${group}_${src}.tex" || true
    done
done

echo ""
echo "============================================"
echo "  Tables saved to $OUTDIR"
echo "============================================"
ls -la "$OUTDIR"
