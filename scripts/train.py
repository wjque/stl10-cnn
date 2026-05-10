import os
import sys
import json
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.cnn import CNNFactory
from configs.experiments import get_experiment_config
from utils.dataloader import create_dataloaders
from utils.metrics import compute_metrics
from utils.visualization import plot_training_curves


def build_mode_fig_path(mode, model_name, ext='png'):
    output_dir = os.path.join('outputs', 'figures', mode)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f'{model_name}.{ext}')


def set_seed(seed):
    """Fix random seeds for reproducibility across numpy, torch, and CUDA."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, config):
    """Run a single training epoch.

    Args:
        model: The CNN model in training mode.
        loader: Training DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: torch optimizer (SGD or AdamW).
        device: 'cuda' or 'cpu'.
        config: Experiment configuration dataclass.
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # 标准训练步骤
        # 清零梯度 -> 前向 -> 计算损失 -> 反向传播 -> 梯度裁剪 -> 更新参数
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

        # 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 累加损失和正确预测数
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model on validation or test set.

    Args:
        model: The CNN model in eval mode.
        loader: Validation / test DataLoader.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        Tuple of (average_loss, accuracy, metrics_dict).
        metrics_dict includes accuracy, precision, recall, f1, confusion_matrix, etc.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_scores = []   # 存储每个 batch 的 softmax 输出
    all_targets = []  # 存储真实标签

    for inputs, targets in tqdm(loader, desc='Evaluating', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_scores.append(outputs.softmax(dim=1).cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    loss = running_loss / total
    acc = correct / total
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    all_preds = all_scores.argmax(axis=1)

    # 计算多分类指标：准确率、精确率、召回率、F1、AUC、混淆矩阵等
    metrics = compute_metrics(all_targets, all_preds, all_scores)

    return loss, acc, metrics


def train(config_module):
    """
    Args:
        config_module: Imported Python module whose .config attribute holds
                       a Config dataclass instance.
    Returns:
        Tuple of (trained_model, log_data_dict).
    """
    config = config_module.config
    set_seed(config.seed)  # 固定随机种子保证可复现

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Config: {config.name}')

    # 创建训练/验证/测试 DataLoader
    train_loader, val_loader, test_loader, classes = create_dataloaders(config)
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')

    # 根据配置构建 CNN 模型
    model = CNNFactory(
        num_classes=10,
        depth=config.depth,
        pooling=config.pooling,
        use_bn=config.use_bn,
        dropout=config.dropout,
    ).to(device)

    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()

    # 选择优化器：SGD (带动量) 或 AdamW (动量 + 自适应梯度调节 + 权重衰减)
    if config.optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                     momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                       betas=(0.9, 0.999), weight_decay=config.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {config.optimizer_name}')

    scheduler = None
    if config.scheduler_name == 'cosine':
        t_max = config.scheduler_t_max or config.num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif config.scheduler_name != 'none':
        raise ValueError(f'Unknown scheduler: {config.scheduler_name}')

    # 初始化日志数据结构
    log_data = {
        'config': {k: v for k, v in config.__dict__.items()},
        'stage': config.stage,
        'seed': config.seed,
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }

    # Early stopping 相关变量
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    # Training loop!
    for epoch in range(1, config.num_epochs + 1):
        print(f'\n--- Epoch {epoch}/{config.num_epochs} ---')

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, config)
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()  # 每个 epoch 结束后更新学习率

        # 记录本轮指标
        log_data['train_loss'].append(train_loss)
        log_data['train_acc'].append(train_acc)
        log_data['val_loss'].append(val_loss)
        log_data['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}')
        print(f'Val   F1: {val_metrics["f1_macro"]:.4f} | AUC: {val_metrics.get("auc_ovr", 0):.4f}')

        # Early stopping 判断
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f'  -> New best model (val_acc={best_val_acc:.4f})')
        else:
            patience_counter += 1
            print(f'  -> No improvement ({patience_counter}/{config.patience})')

        if config.use_early_stopping and patience_counter >= config.patience:
            print(f'\nEarly stopping at epoch {epoch} (best val_acc={best_val_acc:.4f} at epoch {best_epoch})')
            break

    # 保存最佳模型的相关信息
    log_data['best_val_acc'] = best_val_acc
    log_data['best_epoch'] = best_epoch
    log_data['stopped_epoch'] = epoch
    log_data['val_metrics'] = val_metrics

    # 恢复最佳模型权重
    model.load_state_dict(best_model_state)

    # 保存模型权重
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), config.model_save_path)
    print(f'\nModel saved to {config.model_save_path}')

    # 保存训练日志（JSON 格式）
    os.makedirs(os.path.dirname(config.log_save_path), exist_ok=True)
    with open(config.log_save_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f'Log saved to {config.log_save_path}')

    # 绘制并保存训练曲线（loss / accuracy）
    fig_save_path = build_mode_fig_path('training', config.name)
    plot_training_curves(log_data, fig_save_path)
    print(f'Training curves saved to {fig_save_path}')

    return model, log_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a generated experiment configuration')
    parser.add_argument('--experiment', type=str, required=True, help='Generated experiment name (e.g., s1_optsgd_lr1e-2_seed42)')
    args = parser.parse_args()

    config_module = type('ConfigModule', (), {'config': get_experiment_config(args.experiment)})

    train(config_module)
