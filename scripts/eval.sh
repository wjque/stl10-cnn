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

ALL_MODELS=("01_baseline" "02_color_jitter" "02_flip_h" "02_random_crop"
            "03_adamw" "03_batchnorm" "03_dropout"
            "04_avgpool" "04_deep" "04_sigmoid")

AUGMENT_MODELS=("01_baseline" "02_color_jitter" "02_flip_h" "02_random_crop")
NORM_MODELS=("01_baseline" "02_flip_h" "03_adamw" "03_batchnorm" "03_dropout")
MODEL_MODELS=("01_baseline" "03_adamw" "04_avgpool" "04_deep" "04_sigmoid")

echo "Step 1: Running inference on val+test sets (best model)..."
$PYTHON scripts/analysis.py eval --model "${ALL_MODELS[@]}"
echo ""

echo "Step 2: Generating LaTeX tables..."
echo ""

_generate() {
    local title="$1"
    local group="$2"
    shift 2
    local models=("$@")

    for src in test_metrics val_metrics; do
        local set_label
        if [ "$src" = "test_metrics" ]; then
            set_label="测试集指标"
        else
            set_label="验证集指标"
        fi
        echo "  -> ${group} (${src})"
        $PYTHON scripts/analysis.py table \
            --model "${models[@]}" \
            --metrics $METRICS \
            --source "$src" \
            --log-dir outputs/eval \
            --title "${title}-${set_label}" \
            --table-output "$OUTDIR/${group}_${src}.tex" || true
    done
}

_generate "数据增强方式" "AUGMENT" "${AUGMENT_MODELS[@]}"
_generate "正则化方式" "NORM" "${NORM_MODELS[@]}"
_generate "模型结构" "MODEL" "${MODEL_MODELS[@]}"

echo ""
echo "============================================"
echo "  Tables saved to $OUTDIR"
echo "============================================"
ls -la "$OUTDIR"
