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
    """
    Args:
        log_path: Path to the JSON log file (e.g., outputs/logs/01_baseline.json).
        device: 'cuda' or 'cpu'.

    Returns:
        Tuple of (model, log_data_dict).
    """
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    config_dict = log_data['config']  # 训练时保存的配置字典
    # 模型权重文件路径
    model_path = config_dict.get('model_save_path', log_path.replace('logs', 'models').replace('.json', '.pth'))

    # 使用与训练时相同的架构参数重建 CNN
    model = CNNFactory(
        num_classes=10,
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
    """Run inference on the test set and compute evaluation metrics.

    Args:
        model: Trained CNN model in eval mode.
        test_loader: Test DataLoader.
        device: 'cuda' or 'cpu'.

    Returns:
        Dict of classification metrics (accuracy, precision, recall, f1, auc, confusion_matrix).
    """
    model.eval()
    all_scores = []   # softmax 概率输出
    all_targets = []  # 真实标签

    for inputs, targets in tqdm(test_loader, desc='Inference'):
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_scores.append(outputs.softmax(dim=1).cpu().numpy())
        all_targets.append(targets.numpy())

    # 拼接所有 batch 的结果
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    all_preds = all_scores.argmax(axis=1)

    metrics = compute_metrics(all_targets, all_preds, all_scores)
    return metrics


def infer_single(config_module_path):
    """Run inference for a single experiment configuration.

    Args:
        config_module_path: Dotted path to config module (e.g., 'configs.config_01_baseline').

    Returns:
        Tuple of (config_name, log_data_dict_or_None).
        Returns (config_name, None) if no log file is found.
    """
    print(f'\n{"="*60}')
    print(f'Inferring: {config_module_path}')

    # 从模块路径提取配置名（如 'config_01_baseline' -> '01_baseline'）
    config_name = config_module_path.split('.')[-1]
    log_path = f'outputs/logs/{config_name.replace("config_", "")}.json'

    if not os.path.exists(log_path):
        print(f'Log not found: {log_path}, skipping...')
        return config_name, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, log_data = load_model_from_log(log_path, device)
    # 用日志中的配置重建 Config 对象以创建 DataLoader
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
    # 所有实验配置名（不含 'config_' 前缀）
    config_names = [
        '01_baseline', '03_adamw', '04_deep', '04_sigmoid', '04_avgpool'
    ]

    all_logs = {}
    for name in config_names:
        config_name, log_data = infer_single(f'configs.config_{name}')
        if log_data is not None:
            all_logs[config_name] = log_data

    # 生成所有模型的多维度对比图
    if all_logs:
        comparison_path = 'outputs/figures/comparison.png'
        plot_comparison(all_logs, comparison_path)
        print(f'\nComparison figures saved to outputs/figures/')


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
