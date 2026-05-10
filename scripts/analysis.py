import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs import Config
from configs.experiments import build_stage_experiments
from scripts.infer import evaluate_experiment, load_model_from_log
from utils.dataloader import create_dataloaders
from utils.visualization import (
    generate_pca_visualization,
    plot_comparison,
    plot_confusion_matrix,
    plot_training_curves,
)


OUTPUTS_DIR = Path('outputs')
DEFAULT_EXPERIMENT = 's1_optsgd_lr1e-2_seed42'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Analysis utilities for the staged experiment pipeline.')
    parser.add_argument('mode', choices=['pca', 'cm', 'train', 'compare', 'eval', 'table'])
    parser.add_argument('--model', nargs='+', help='Experiment name(s).')
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'stage3', 'stage4'])
    parser.add_argument('--baseline', default='', help='Optional baseline experiment name for stage expansion.')
    parser.add_argument('--window', type=int, default=5, help='Smoothing window size for curve plots.')
    parser.add_argument('--force', action='store_true', help='Overwrite cached eval results if they exist.')
    parser.add_argument('--data-dir', default='STL10/test', help='Dataset directory used by PCA mode.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for PCA sampling.')
    parser.add_argument('--output', default='outputs/figures/comparison', help='Output path prefix for compare mode.')
    parser.add_argument('--metrics', nargs='+', default=['accuracy', 'f1_macro'], help='Metric names for table mode.')
    parser.add_argument('--source', default='test_metrics', choices=['test_metrics', 'val_metrics'], help='Metric source for experiment table mode.')
    parser.add_argument('--table-output', default=None, help='Output path for table mode.')
    parser.add_argument('--title', default=None, help='Custom caption title for the LaTeX table.')
    parser.add_argument('--log-dir', default='outputs/logs', help='Directory containing per-experiment logs.')
    parser.add_argument('--report-dir', default='outputs/reports', help='Directory containing stage summary reports.')
    parser.add_argument('--table-kind', default='experiments', choices=['experiments', 'stage-summary'], help='Table source kind.')
    return parser.parse_args(argv)


def resolve_models(args, require_models=True):
    if args.model:
        return args.model
    if args.stage:
        baseline = args.baseline or None
        return [config.name for config in build_stage_experiments(args.stage, baseline=baseline)]
    if require_models:
        return [DEFAULT_EXPERIMENT]
    return []


def log_path(model_name, log_dir='outputs/logs'):
    return Path(log_dir) / f'{model_name}.json'


def build_fig_path(model_name, suffix):
    subdirs = {
        'confusion': 'confusion',
        'training': 'training',
    }
    output_dir = OUTPUTS_DIR / 'figures' / subdirs.get(suffix, '')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{model_name}_{suffix}.png'


def load_logs(model_names, log_dir='outputs/logs'):
    logs = {}
    for model_name in model_names:
        path = log_path(model_name, log_dir)
        if not path.exists():
            print(f'Skipping {model_name}: log not found')
            continue
        with path.open('r') as f:
            logs[model_name] = json.load(f)
    return logs


def run_pca(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = resolve_models(args)

    for model_name in models:
        current_log_path = log_path(model_name)
        model_path = OUTPUTS_DIR / 'models' / f'{model_name}.pth'
        if not current_log_path.exists() or not model_path.exists():
            print(f'Skipping {model_name}: log or model not found')
            continue

        model, _ = load_model_from_log(str(current_log_path), device)
        save_dir = OUTPUTS_DIR / 'figures' / 'pca' / model_name
        generate_pca_visualization(
            model_name,
            model,
            args.data_dir,
            str(save_dir),
            device,
            n_samples_per_class=1,
            seed=args.seed,
        )


def run_cm(args):
    models = resolve_models(args)
    logs = load_logs(models)

    for model_name, data in logs.items():
        cm = data.get('test_metrics', {}).get('confusion_matrix', [])
        if not cm:
            print(f'Skipping {model_name}: no confusion matrix in log')
            continue
        save_path = build_fig_path(model_name, 'confusion')
        plot_confusion_matrix(np.array(cm), model_name, str(save_path), normalize=True)
        print(f'Confusion matrix for {model_name} saved to {save_path}')


def run_train(args):
    model_name = resolve_models(args)[0]
    logs = load_logs([model_name])
    if model_name not in logs:
        return
    save_path = build_fig_path(model_name, 'training')
    plot_training_curves(logs[model_name], str(save_path), window=args.window)
    print(f'Training curves saved to {save_path}')


def run_compare(args):
    models = resolve_models(args)
    logs = load_logs(models)
    if not logs:
        print('No valid logs found.')
        return
    plot_comparison(logs, args.output, window=args.window)
    print(f'Comparison plots saved to {args.output}*')


def run_eval(args):
    models = resolve_models(args)
    eval_dir = OUTPUTS_DIR / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        current_log_path = log_path(model_name, args.log_dir)
        if not current_log_path.exists():
            print(f'Skipping {model_name}: log not found')
            continue

        eval_path = eval_dir / f'{model_name}.json'
        if eval_path.exists() and not args.force:
            print(f'Skipping {model_name}: eval results already exist')
            continue

        with current_log_path.open('r') as f:
            log_data = json.load(f)

        val_metrics = evaluate_experiment(model_name, split='val')
        test_metrics = evaluate_experiment(model_name, split='test')
        print(f'{model_name}  Val Accuracy: {val_metrics["accuracy"]:.4f}  Test Accuracy: {test_metrics["accuracy"]:.4f}')

        log_data['val_metrics'] = val_metrics
        log_data['test_metrics'] = test_metrics
        with current_log_path.open('w') as f:
            json.dump(log_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

        eval_data = {
            'model': model_name,
            'config': log_data['config'],
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
        }
        with eval_path.open('w') as f:
            json.dump(eval_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    print(f'Eval results saved to {eval_dir}')


def build_experiment_table(args):
    models = resolve_models(args)
    rows = []
    for model_name in models:
        current_log_path = log_path(model_name, args.log_dir)
        if not current_log_path.exists():
            print(f'Skipping {model_name}: log not found')
            continue
        with current_log_path.open('r') as f:
            log_data = json.load(f)
        source_dict = log_data.get(args.source, {})
        values = [source_dict.get(metric) for metric in args.metrics]
        if any(value is None for value in values):
            print(f'Skipping {model_name}: missing metrics in {args.source}')
            continue
        rows.append((model_name, values))
    return rows


def build_stage_summary_table(args):
    if not args.stage:
        raise ValueError('--stage is required for table-kind=stage-summary')
    report_path = Path(args.report_dir) / f'{args.stage}_summary.json'
    if not report_path.exists():
        raise FileNotFoundError(f'Stage summary not found: {report_path}')

    with report_path.open('r') as f:
        report = json.load(f)

    rows = []
    for item in report.get('summary', []):
        rows.append((
            item['group'],
            [
                item['runs'],
                f"{item['best_val_acc_mean']:.4f} $\\pm$ {item['best_val_acc_std']:.4f}",
                f"{item['test_acc_mean']:.4f} $\\pm$ {item['test_acc_std']:.4f}",
            ],
        ))
    return rows


def render_latex_table(headers, rows, output_path, caption):
    OUTPUTS_DIR.joinpath('tables').mkdir(parents=True, exist_ok=True)
    col_spec = 'l' + 'c' * (len(headers) - 1)
    lines = [
        r'\begin{table}[H]',
        r'  \centering',
        f'  \\caption{{{caption}}}',
        r'  \label{tab:experiment_summary}',
        f'  \\begin{{tabular}}{{{col_spec}}}',
        r'    \toprule',
        f'    {" & ".join(headers)} \\\\',
        r'    \midrule',
    ]

    for name, values in rows:
        rendered_values = [value if isinstance(value, str) else f'{value:.4f}' for value in values]
        escaped_name = name.replace('_', r'\_')
        lines.append(f'    {escaped_name} & {" & ".join(rendered_values)} \\\\')

    lines.extend([
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'LaTeX table saved to {output_path}')


def run_table(args):
    if args.table_kind == 'stage-summary':
        rows = build_stage_summary_table(args)
        headers = ['Group', 'Runs', 'Best Val Acc', 'Test Acc']
        output_path = args.table_output or str(OUTPUTS_DIR / 'tables' / f'{args.stage}_summary.tex')
        caption = args.title or f'{args.stage} summary'
        render_latex_table(headers, rows, output_path, caption)
        return

    rows = build_experiment_table(args)
    headers = ['Experiment'] + [metric.replace('_', ' ').title() for metric in args.metrics]
    output_path = args.table_output or str(OUTPUTS_DIR / 'tables' / f'experiments_{args.source}.tex')
    caption = args.title or f'Experiment comparison on {args.source.replace("_", " ")}'
    render_latex_table(headers, rows, output_path, caption)


DISPATCH = {
    'pca': run_pca,
    'cm': run_cm,
    'train': run_train,
    'compare': run_compare,
    'eval': run_eval,
    'table': run_table,
}


def main(argv=None):
    args = parse_args(argv)
    DISPATCH[args.mode](args)


if __name__ == '__main__':
    main()
