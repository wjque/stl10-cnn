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
FIGURE_DIR="$PROJECT_DIR/outputs/figures"
mkdir -p "$OUTDIR"
mkdir -p "$FIGURE_DIR/comparison"

ALL_MODELS=("01_baseline" "02_color_jitter" "02_flip_h" "02_random_crop"
            "03_adamw" "03_batchnorm" "03_dropout"
            "04_avgpool" "04_deep" "04_sigmoid")

AUGMENT_MODELS=("01_baseline" "02_color_jitter" "02_flip_h" "02_random_crop")
NORM_MODELS=("01_baseline" "02_flip_h" "03_adamw" "03_batchnorm" "03_dropout")
MODEL_MODELS=("01_baseline" "03_adamw" "04_avgpool" "04_deep" "04_sigmoid")

_run_table() {
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

_run_multi_mode() {
    local mode="$1"
    local group="$2"
    shift 2
    local models=("$@")

    echo "  -> ${mode} (${group})"
    $PYTHON scripts/analysis.py "$mode" --model "${models[@]}" || true
}

_run_single_mode() {
    local mode="$1"
    local group="$2"
    shift 2
    local models=("$@")

    for model in "${models[@]}"; do
        echo "  -> ${mode} (${group}/${model})"
        $PYTHON scripts/analysis.py "$mode" --model "$model" || true
    done
}

_run_compare() {
    local group="$1"
    shift
    local models=("$@")

    echo "  -> compare (${group})"
    $PYTHON scripts/analysis.py compare \
        --model "${models[@]}" \
        --output "$FIGURE_DIR/comparison/${group,,}" || true
}

echo "Step 1: Running inference on val+test sets (best model)..."
$PYTHON scripts/analysis.py eval --model "${ALL_MODELS[@]}"
echo ""

echo "Step 2: Generating LaTeX tables..."
echo ""

_run_table "数据增强方式" "AUGMENT" "${AUGMENT_MODELS[@]}"
_run_table "正则化方式" "NORM" "${NORM_MODELS[@]}"
_run_table "模型结构" "MODEL" "${MODEL_MODELS[@]}"

echo ""
echo "Step 3: Running PCA analysis..."
echo ""

_run_multi_mode "pca" "AUGMENT" "${AUGMENT_MODELS[@]}"
_run_multi_mode "pca" "NORM" "${NORM_MODELS[@]}"
_run_multi_mode "pca" "MODEL" "${MODEL_MODELS[@]}"

echo ""
echo "Step 4: Running confusion matrix analysis..."
echo ""
_run_multi_mode "cm" "AUGMENT" "${AUGMENT_MODELS[@]}"
_run_multi_mode "cm" "NORM" "${NORM_MODELS[@]}"
_run_multi_mode "cm" "MODEL" "${MODEL_MODELS[@]}"

echo ""
echo "Step 5: Running comparison analysis..."
echo ""
_run_compare "AUGMENT" "${AUGMENT_MODELS[@]}"
_run_compare "NORM" "${NORM_MODELS[@]}"
_run_compare "MODEL" "${MODEL_MODELS[@]}"

echo ""
echo "Step 6: Running training curve analysis..."
echo ""
_run_single_mode "train" "ALL" "${ALL_MODELS[@]}"

echo ""
echo "Step 7: Running gradient norm analysis..."
echo ""
_run_single_mode "grad" "ALL" "${ALL_MODELS[@]}"

echo ""
echo "============================================"
echo "  Tables saved to $OUTDIR"
echo "============================================"
ls -la "$OUTDIR"
