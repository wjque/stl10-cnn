import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.cnn import CNNFactory
from utils.dataloader import create_dataloaders
from utils.metrics import compute_metrics
from utils.visualization import plot_comparison
from configs import Config


def load_model_from_log(log_path, device):
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    config_dict = log_data['config']
    model_path = config_dict.get('model_save_path', log_path.replace('logs', 'models').replace('.json', '.pth'))

    model = CNNFactory(
        num_classes=10,
        use_residual=config_dict.get('use_residual', False),
        depth=config_dict.get('depth', 'shallow'),
        activation=config_dict.get('activation', 'relu'),
        pooling=config_dict.get('pooling', 'max'),
        use_bn=config_dict.get('use_bn', False),
        dropout=config_dict.get('dropout', 0.5),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, log_data


@torch.no_grad()
def infer(model, test_loader, device):
    model.eval()
    all_scores = []
    all_targets = []

    for inputs, targets in tqdm(test_loader, desc='Inference'):
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_scores.append(outputs.softmax(dim=1).cpu().numpy())
        all_targets.append(targets.numpy())

    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    all_preds = all_scores.argmax(axis=1)

    metrics = compute_metrics(all_targets, all_preds, all_scores)
    return metrics


def infer_single(config_module_path):
    print(f'\n{"="*60}')
    print(f'Inferring: {config_module_path}')

    config_name = config_module_path.split('.')[-1]
    log_path = f'outputs/logs/{config_name.replace("config_", "")}.json'

    if not os.path.exists(log_path):
        print(f'Log not found: {log_path}, skipping...')
        return config_name, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, log_data = load_model_from_log(log_path, device)
    config = Config(**log_data['config'])
    _, _, test_loader, _ = create_dataloaders(config)

    metrics = infer(model, test_loader, device)

    print(f'Test Accuracy:  {metrics["accuracy"]:.4f}')
    print(f'Test Precision: {metrics["precision_macro"]:.4f}')
    print(f'Test Recall:    {metrics["recall_macro"]:.4f}')
    print(f'Test F1 (macro):{metrics["f1_macro"]:.4f}')
    print(f'Test AUC (ovr): {metrics.get("auc_ovr", 0):.4f}')
    print(f'Confusion Matrix:\n{np.array(metrics["confusion_matrix"])}')

    log_data['test_metrics'] = metrics
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    return config_name, log_data


def infer_all():
    config_names = [
        '01_baseline', '02_augment', '03_mixup',
        '04_residual', '05_deep', '06_sigmoid',
        '07_avgpool', '08_batchnorm', '09_adamw',
    ]

    all_logs = {}
    for name in config_names:
        config_name, log_data = infer_single(f'configs.config_{name}')
        if log_data is not None:
            all_logs[config_name] = log_data

    if all_logs:
        comparison_path = 'outputs/figures/comparison.png'
        plot_comparison(all_logs, comparison_path)
        print(f'\nComparison figure saved to {comparison_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('--all', action='store_true', help='Run inference on all trained models')
    parser.add_argument('--model', type=str, default=None, help='Config module path for a single model')
    args = parser.parse_args()

    if args.all:
        infer_all()
    elif args.model:
        infer_single(args.model)
    else:
        parser.print_help()
