"""
Unified analysis script

Usage (CLI):
    python scripts/analysis.py cm   --model 01_baseline
    python scripts/analysis.py compare --model 01_baseline 03_adamw 04_deep
    python scripts/analysis.py grad   --model 01_baseline
    python scripts/analysis.py pca --model 01_baseline 02_flip_h
    python scripts/analysis.py train --model 01_baseline
    python scripts/analysis.py table --model 01_baseline 03_adamw \\
        --metrics accuracy f1_macro auc_ovr --source test_metrics

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
from scripts.infer import load_model_from_log, infer
from utils.dataloader import create_dataloaders
from configs import Config

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
        choices=['pca', 'cm', 'train', 'compare', 'grad', 'eval', 'table'],
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
    parser.add_argument(
        '--force', action='store_true',
        help='Force rerun for modes that support skipping existing outputs.',
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

    # extra args for table mode
    parser.add_argument(
        '--metrics', type=str, nargs='+',
        default=['accuracy', 'f1_macro'],
        help='Metric names for table mode (e.g. accuracy f1_macro auc_ovr).',
    )
    parser.add_argument(
        '--source', type=str, default='test_metrics',
        choices=['test_metrics', 'val_metrics'],
        help='Which metrics dict to read from logs.',
    )
    parser.add_argument(
        '--table-output', type=str, default=None,
        help='Output .tex path for table mode '
             '(default: outputs/tables/comparison_{source}.tex).',
    )
    parser.add_argument(
        '--title', type=str, default=None,
        help='Custom caption title for the LaTeX table.',
    )
    parser.add_argument(
        '--log-dir', type=str, default='outputs/logs',
        help='Directory containing JSON log/eval files.',
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


def run_table(args):
    """Generate a LaTeX-formatted comparison table for one or more models."""
    models = _resolve_models(args, multi=True)
    if not models:
        return

    # Gather metrics from logs/eval
    table_data = {}
    for model_name in models:
        json_p = f'{args.log_dir}/{model_name}.json'
        if not os.path.exists(json_p):
            print(f'Skipping {model_name}: not found')
            continue
        with open(json_p) as f:
            log_data = json.load(f)
        source_dict = log_data.get(args.source, {})
        table_data[model_name] = {
            metric: source_dict.get(metric, None)
            for metric in args.metrics
        }

    if not table_data:
        print('No valid logs found.')
        return

    # Check which models have all requested metrics
    valid_models = {
        m: d for m, d in table_data.items()
        if all(v is not None for v in d.values())
    }
    missing = set(models) - set(valid_models.keys())
    if missing:
        print(f'WARNING: skipping models (missing metrics): {", ".join(sorted(missing))}')

    if not valid_models:
        print('No models with all requested metrics.')
        return

    # Determine best value per metric (higher is better for all common metrics)
    best = {}
    for metric in args.metrics:
        values = [(m, d[metric]) for m, d in valid_models.items()]
        best[metric] = max(values, key=lambda x: x[1])

    # Build LaTeX table
    n_cols = len(args.metrics)
    col_spec = 'l' + 'c' * n_cols
    header_metrics = ' & '.join(
        _fmt_metric_name(m) for m in args.metrics
    )
    if args.title:
        caption = args.title
    else:
        caption = (
            f'Model comparison on {args.source.replace("_", " ")} '
            f'({", ".join(_fmt_metric_name(m) for m in args.metrics)})'
        )

    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'  \centering')
    lines.append(f'  \\caption{{{caption}}}')
    lines.append(r'  \label{tab:model_comparison}')
    lines.append(r'  \begin{tabular}{' + col_spec + '}')
    lines.append(r'    \toprule')
    lines.append(f'    Model & {header_metrics} \\\\')
    lines.append(r'    \midrule')

    for model_name in models:
        if model_name not in valid_models:
            continue
        row_d = valid_models[model_name]
        cells = []
        for metric in args.metrics:
            val = row_d[metric]
            best_model, best_val = best[metric]
            cell = f'{val:.4f}'
            if model_name == best_model:
                cell = f'\\textbf{{{cell}}}'
            cells.append(cell)
        tex_name = model_name.replace('_', r'\_')
        lines.append(f'    {tex_name} & {" & ".join(cells)} \\\\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    tex_content = '\n'.join(lines)

    os.makedirs(f'{OUPUTS_DIR}/tables', exist_ok=True)
    if args.table_output:
        output_path = args.table_output
    else:
        output_path = f'{OUPUTS_DIR}/tables/comparison_{args.source}.tex'
    with open(output_path, 'w') as f:
        f.write(tex_content + '\n')
    print(f'LaTeX table saved to {output_path}')


def run_eval(args):
    """Run inference on val+test using best model, save to outputs/eval/."""
    models = _resolve_models(args, multi=True, default=DEFAULT_MODELS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f'{OUPUTS_DIR}/eval', exist_ok=True)

    for model_name in models:
        log_p = _log_path(model_name)
        model_p = f'{OUPUTS_DIR}/models/{model_name}.pth'
        eval_path = f'{OUPUTS_DIR}/eval/{model_name}.json'
        if not os.path.exists(log_p) or not os.path.exists(model_p):
            print(f'Skipping {model_name}: log or model not found')
            continue
        if os.path.exists(eval_path) and not args.force:
            print(f'Skipping {model_name}: eval results already exist')
            continue

        model, log_data = load_model_from_log(log_p, device)
        config = Config(**log_data['config'])
        _, val_loader, test_loader, _ = create_dataloaders(config)

        val_metrics = infer(model, val_loader, device)
        test_metrics = infer(model, test_loader, device)

        print(f'{model_name}  Val Accuracy: {val_metrics["accuracy"]:.4f}  Test Accuracy: {test_metrics["accuracy"]:.4f}')

        eval_data = {
            'model': model_name,
            'config': log_data['config'],
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
        }
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2,
                      default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    print(f'Eval results saved to {OUPUTS_DIR}/eval/')


def _fmt_metric_name(metric):
    return metric.replace('_', ' ').title()


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
    subdirs = {
        'confusion': 'coonfusion',
        'training': 'training',
    }
    subdir = subdirs.get(suffix)
    if subdir is not None:
        output_dir = f"{OUPUTS_DIR}/figures/{subdir}"
        os.makedirs(output_dir, exist_ok=True)
        return f"{output_dir}/{model_name}_{suffix}.png"
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
    'eval':    run_eval,
    'table':   run_table,
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
