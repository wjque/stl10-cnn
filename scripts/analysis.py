"""
Unified analysis script

Usage (CLI):
    python scripts/analysis.py cm   --model 01_baseline
    python scripts/analysis.py compare --model 01_baseline 03_adamw 04_deep
    python scripts/analysis.py grad   --model 01_baseline
    python scripts/analysis.py pca --model 01_baseline 02_flip_h
    python scripts/analysis.py train --model 01_baseline

Usage (programmatic):
    from scripts.analysis import parse_args, run_pca, run_cm, ...
    args = parse_args(['pca', '--model', '01_baseline'])
    run_pca(args)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.visualization import (
    load_model, generate_pca_visualization,
    plot_confusion_matrix, plot_training_curves,
    plot_comparison, plot_grad_norms,
)

DEFAULT_MODELS = ['01_baseline', '02_flip_h', '03_adamw', '04_avgpool']
OUPUTS_DIR = './outputs'


# ====================
#  Argument parser
# ====================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Visualization and analysis utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'mode',
        choices=['pca', 'cm', 'train', 'compare', 'grad'],
        help='Analysis mode to run',
    )
    parser.add_argument(
        '-m', '--model', type=str, nargs='+',
        help='Model name(s). '
             'For "train" / "grad" pass one name; '
             'for "pca" / "cm" / "compare" pass one or more.',
    )
    parser.add_argument(
        '--window', type=int, default=5,
        help='Smoothing window size.',
    )

    # extra args for pca mode
    parser.add_argument(
        '--data-dir', type=str, default='STL10/test',
        help='Data directory for pca mode.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for pca mode.',
    )

    # extra args for compare mode
    parser.add_argument(
        '--output', type=str, default='outputs/figures/comparison',
        help='Output path prefix for compare mode.',
    )

    # extra args for grad mode
    parser.add_argument(
        '--top-k', type=int, default=3,
        help='Number of top layers to show when use grad mode.',
    )

    return parser.parse_args(argv)


# ============================
#  Mode implementations
# ============================

def run_pca(args):
    """PCA feature visualization for one or more models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = _resolve_models(args, multi=True, default=DEFAULT_MODELS)
    print(f'Using device: {device}')

    for model_name in models:
        log_p = _log_path(model_name)
        model_p = f'outputs/models/{model_name}.pth'
        if not os.path.exists(log_p) or not os.path.exists(model_p):
            print(f'Skipping {model_name}: log or model not found')
            continue
        model, _ = load_model(log_p, model_p, device)
        save_dir = f'outputs/figures/pca/{model_name}'
        generate_pca_visualization(
            model_name, model, args.data_dir, save_dir, device,
            n_samples_per_class=1, seed=args.seed,
        )


def run_cm(args):
    """Confusion matrix visualization for one or more models."""
    models = _resolve_models(args, multi=True, default=DEFAULT_MODELS)

    for model_name in models:
        log_p = _log_path(model_name)
        if not os.path.exists(log_p):
            print(f'Skipping {model_name}: log not found')
            continue
        with open(log_p) as f:
            data = json.load(f)
        cm = data.get('test_metrics', {}).get('confusion_matrix', [])
        if len(cm) == 0:
            print(f'Skipping {model_name}: no confusion matrix')
            continue
        save_path = _build_fig_path(model_name, 'confusion')
        plot_confusion_matrix(np.array(cm), model_name, save_path, normalize=True)
        print(f'Confusion matrix for {model_name} saved.')
    print('Done.')


def run_compare(args):
    """Compare multiple models side-by-side."""
    models = _resolve_models(args, multi=True)
    if not models:
        return

    all_logs = {}
    for model_name in models:
        log_p = _log_path(model_name)
        if not os.path.exists(log_p):
            print(f'Skipping {model_name}: log not found')
            continue
        with open(log_p) as f:
            all_logs[model_name] = json.load(f)

    if not all_logs:
        print('No valid logs found.')
        return

    plot_comparison(all_logs, args.output, window=args.window)
    print(f'Comparison plots saved to {args.output}*')


def run_train(args):
    """Plot training curves for a single model."""
    model_name = _resolve_models(args, multi=False)
    if model_name is None:
        return

    log_p = _log_path(model_name)
    if not os.path.exists(log_p):
        return _warn_missing(log_p)
    with open(log_p) as f:
        log_data = json.load(f)

    save_path = _build_fig_path(model_name, 'training')
    plot_training_curves(log_data, save_path, window=args.window)
    print(f'Training curves saved to {save_path}')


def run_grad(args):
    """Plot gradient norms for a single model."""
    model_name = _resolve_models(args, multi=False)
    if model_name is None:
        return

    log_p = _log_path(model_name)
    if not os.path.exists(log_p):
        return _warn_missing(log_p)
    with open(log_p) as f:
        log_data = json.load(f)

    save_path = _build_fig_path(model_name, 'grad_norms')
    plot_grad_norms(log_data, save_path, top_k=args.top_k, window=args.window)
    print(f'Gradient norm plots saved to {save_path}')


# ======================
#  Argument helpers
# ======================

def _resolve_models(args, multi=False, default=None):
    """Return a list of model names from args, or a single name for single mode.

    Args:
        args:     argparse namespace.
        multi:    if True, return a list (never None).
        default:  fallback list when args.model is None and multi=True.
    """
    names = args.model

    if multi:
        if names is None and default is not None:
            return default
        if names is None:
            print('Error: --model is required.')
            return []
        return names

    # single-mode
    if names is None:
        print('Error: --model is required.')
        return None
    if len(names) > 1:
        print(f'Warning: only one model expected, using first ("{names[0]}").')
    return names[0]


def _warn_missing(log_p):
    print(f'Log not found: {log_p}')

    
def _build_fig_path(model_name, suffix):
    return f"{OUPUTS_DIR}/figures/{model_name}_{suffix}.png"


def _log_path(model_name):
    return f"{OUPUTS_DIR}/logs/{model_name}.json"


# ======================
#  Main dispatcher
# ======================

DISPATCH = {
    'pca':     run_pca,
    'cm':      run_cm,
    'train':   run_train,
    'compare': run_compare,
    'grad':    run_grad,
}


def main(argv=None):
    args = parse_args(argv)
    fn = DISPATCH.get(args.mode)
    if fn is None:
        print(f'Unknown mode: {args.mode}')
        return
    fn(args)


if __name__ == '__main__':
    main()
