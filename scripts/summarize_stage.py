import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs.experiments import build_stage_experiments


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize stage experiments across seeds.')
    parser.add_argument('--stage', required=True, choices=['stage1', 'stage2', 'stage3', 'stage4'])
    parser.add_argument('--baseline', default='', help='Optional baseline experiment name.')
    parser.add_argument('--log-dir', default='outputs/logs')
    parser.add_argument('--output-dir', default='outputs/reports')
    return parser.parse_args()


def make_group_key(config_dict):
    fields_by_stage = {
        'stage1': ['optimizer_name', 'learning_rate'],
        'stage2': ['augmentations'],
        'stage3': ['use_bn', 'dropout', 'weight_decay'],
        'stage4': ['depth', 'pooling'],
    }
    fields = fields_by_stage[config_dict['stage']]
    values = []
    for field in fields:
        value = config_dict[field]
        if isinstance(value, list):
            value = tuple(value)
        values.append((field, value))
    return tuple(values)


def key_to_name(group_key):
    parts = []
    for field, value in group_key:
        if isinstance(value, tuple):
            value = 'none' if not value else '+'.join(value)
        parts.append(f'{field}={value}')
    return ', '.join(parts)


def summarize_group(entries):
    val_scores = np.array([entry['best_val_acc'] for entry in entries], dtype=float)
    test_acc = np.array([
        entry.get('test_metrics', {}).get('accuracy', np.nan)
        for entry in entries
    ], dtype=float)
    test_precision = np.array([
        entry.get('test_metrics', {}).get('precision_macro', np.nan)
        for entry in entries
    ], dtype=float)
    test_f1 = np.array([
        entry.get('test_metrics', {}).get('f1_macro', np.nan)
        for entry in entries
    ], dtype=float)
    test_auc = np.array([
        entry.get('test_metrics', {}).get('auc_ovr', np.nan)
        for entry in entries
    ], dtype=float)
    return {
        'runs': len(entries),
        'best_val_acc_mean': float(np.nanmean(val_scores)),
        'best_val_acc_std': float(np.nanstd(val_scores)),
        'test_acc_mean': float(np.nanmean(test_acc)),
        'test_acc_std': float(np.nanstd(test_acc)),
        'test_precision_mean': float(np.nanmean(test_precision)),
        'test_precision_std': float(np.nanstd(test_precision)),
        'test_f1_mean': float(np.nanmean(test_f1)),
        'test_f1_std': float(np.nanstd(test_f1)),
        'test_auc_mean': float(np.nanmean(test_auc)),
        'test_auc_std': float(np.nanstd(test_auc)),
        'experiments': [entry['name'] for entry in entries],
    }


def main():
    args = parse_args()
    baseline = args.baseline or None
    experiments = build_stage_experiments(args.stage, baseline=baseline)
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    missing = []

    for config in experiments:
        log_path = log_dir / f'{config.name}.json'
        if not log_path.exists():
            missing.append(config.name)
            continue
        with log_path.open('r') as f:
            log_data = json.load(f)
        config_dict = log_data['config']
        config_dict.setdefault('stage', log_data.get('stage', config.stage))
        grouped[make_group_key(config_dict)].append({
            'name': config.name,
            'best_val_acc': log_data.get('best_val_acc', np.nan),
            'test_metrics': log_data.get('test_metrics', {}),
        })

    summary = []
    for group_key, entries in grouped.items():
        metrics = summarize_group(entries)
        summary.append({
            'group': key_to_name(group_key),
            **metrics,
        })

    summary.sort(key=lambda item: item['best_val_acc_mean'], reverse=True)

    payload = {
        'stage': args.stage,
        'baseline': baseline,
        'summary': summary,
        'missing_logs': missing,
    }

    output_path = output_dir / f'{args.stage}_summary.json'
    with output_path.open('w') as f:
        json.dump(payload, f, indent=2)

    print(f'Saved summary to {output_path}')
    for item in summary:
        print(
            f"{item['group']}: "
            f"val={item['best_val_acc_mean']:.4f}±{item['best_val_acc_std']:.4f}, "
            f"test_acc={item['test_acc_mean']:.4f}±{item['test_acc_std']:.4f}, "
            f"test_precision={item['test_precision_mean']:.4f}±{item['test_precision_std']:.4f}, "
            f"test_f1={item['test_f1_mean']:.4f}±{item['test_f1_std']:.4f}, "
            f"test_auc={item['test_auc_mean']:.4f}±{item['test_auc_std']:.4f}"
        )
    if missing:
        print('Missing logs:')
        for name in missing:
            print(f'  - {name}')


if __name__ == '__main__':
    main()
