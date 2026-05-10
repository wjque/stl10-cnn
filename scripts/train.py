import os
import sys
import json
import copy
import argparse
import warnings
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.cnn import CNNFactory
from configs.experiments import get_experiment_config
from utils.dataloader import create_dataloaders
from utils.grad_utils import (
    compute_grad_norms,
    accumulate_grad_norms,
    average_grad_norms,
    get_top_grad_norms,
    find_activation_layers,
    register_act_grad_hooks,
    remove_act_grad_hooks,
    get_act_grad_stats,
    accumulate_act_grad_stats,
    average_act_grad_stats,
)
from utils.metrics import compute_metrics
from utils.visualization import plot_training_curves, plot_tsne, plot_grad_norms, plot_act_grad_curves


def set_seed(seed):
    """Fix random seeds for reproducibility across numpy, torch, and CUDA."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, config, log_grad_norms=False, act_layers=None):
    """Run a single training epoch.

    Args:
        model: The CNN model in training mode.
        loader: Training DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: torch optimizer (SGD or AdamW).
        device: 'cuda' or 'cpu'.
        config: Experiment configuration dataclass.
        log_grad_norms: If True, accumulate per-layer gradient L2 norms.
        act_layers: Optional list of (name, module) activation-layer tuples
                    to record output/gradient norm and zero-ratio stats.

    Returns:
        Tuple of (average_loss, accuracy, gradient_norms_dict_or_None,
                  act_grad_stats_dict_or_None).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    grad_sum = defaultdict(float)  # 累积各层梯度 L2 范数
    grad_steps = 0

    act_grad_accum = {}
    act_grad_steps = 0
    handles = None
    hook_data = None
    if act_layers:
        handles, hook_data = register_act_grad_hooks(act_layers)

    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # 标准训练步骤
        # 清零梯度 -> 前向 -> 计算损失 -> 反向传播 -> 梯度裁剪 -> 更新参数
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

        if log_grad_norms:
            batch_grad_norms = compute_grad_norms(model)
            accumulate_grad_norms(grad_sum, batch_grad_norms)
            grad_steps += 1

        if act_layers:
            batch_act_stats = get_act_grad_stats(hook_data)
            accumulate_act_grad_stats(act_grad_accum, batch_act_stats)
            act_grad_steps += 1

        # 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 累加损失和正确预测数
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    if handles is not None:
        remove_act_grad_hooks(handles)

    epoch_grad_norms = average_grad_norms(grad_sum, grad_steps) if log_grad_norms else None
    epoch_act_grad = average_act_grad_stats(act_grad_accum, act_grad_steps) if act_layers else None

    return running_loss / total, correct / total, epoch_grad_norms, epoch_act_grad


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


def train(config_module, log_grad_norms=False, grad_top_k=3, log_act_grad=False, top_k_layer=None):
    """
    Args:
        config_module: Imported Python module whose .config attribute holds
                       a Config dataclass instance.
        log_grad_norms: Whether to record per-layer gradient norms each epoch.
        grad_top_k: Print the top-k largest gradient norms per epoch.
        log_act_grad: Whether to record activation-layer output/gradient norms
                      and zero-ratio each epoch.
        top_k_layer: When log_act_grad is True, record only the first
                     ``top_k_layer`` activation layers from the input side.
                     None means record all layers.

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
        activation=config.activation,
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
    if log_grad_norms:
        log_data['train_grad_norms'] = []

    act_layers = None
    if log_act_grad:
        act_layers = find_activation_layers(model, top_k=top_k_layer)
        print(f'Tracking {len(act_layers)} activation layers for act-grad logging.')
        log_data['train_act_grad'] = []

    # Early stopping 相关变量
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    # Training loop!
    for epoch in range(1, config.num_epochs + 1):
        print(f'\n--- Epoch {epoch}/{config.num_epochs} ---')

        train_loss, train_acc, epoch_grad_norms, epoch_act_grad = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config,
            log_grad_norms=log_grad_norms, act_layers=act_layers
        )
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()  # 每个 epoch 结束后更新学习率

        # 记录本轮指标
        log_data['train_loss'].append(train_loss)
        log_data['train_acc'].append(train_acc)
        log_data['val_loss'].append(val_loss)
        log_data['val_acc'].append(val_acc)
        if log_grad_norms:
            log_data['train_grad_norms'].append(epoch_grad_norms)
        if log_act_grad:
            log_data['train_act_grad'].append(epoch_act_grad)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}')
        print(f'Val   F1: {val_metrics["f1_macro"]:.4f} | AUC: {val_metrics.get("auc_ovr", 0):.4f}')
        if log_grad_norms and epoch_grad_norms is not None:
            print(f'Grad L2: {epoch_grad_norms.get("global_l2", 0.0):.4e}')
            top_grad_norms = get_top_grad_norms(epoch_grad_norms, top_k=grad_top_k)
            if top_grad_norms:
                top_text = ', '.join([f'{name}:{value:.4e}' for name, value in top_grad_norms])
                print(f'Top-{grad_top_k} grad norms: {top_text}')

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
    fig_save_path = f'outputs/figures/{config.name}_curves.png'
    plot_training_curves(log_data, fig_save_path)
    print(f'Training curves saved to {fig_save_path}')

    # 绘制并保存 t-SNE 特征可视化，定性判断分类器性能
    tsne_save_path = f'outputs/figures/{config.name}_tsne.png'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            plot_tsne(model, val_loader, tsne_save_path, device=device)
        print(f't-SNE visualization saved to {tsne_save_path}')
    except Exception as e:
        print(f't-SNE visualization skipped (error: {e})')

    # Optional: 绘制梯度范数变化图，观察梯度衰减情况
    if log_grad_norms:
        grad_fig_save_path = f'outputs/grad/{config.name}_grad_norms.png'
        plot_grad_norms(log_data, grad_fig_save_path, top_k=grad_top_k)
        print(f'Gradient norm visualization saved to {grad_fig_save_path}')

    if log_act_grad:
        act_grad_save_dir = 'outputs/act_grad'
        plot_act_grad_curves(log_data, act_grad_save_dir, config.name)
        print(f'Activation gradient curves saved to {act_grad_save_dir}/')

    return model, log_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a generated experiment configuration')
    parser.add_argument('--experiment', type=str, required=True, help='Generated experiment name (e.g., s1_optsgd_lr1e-2_seed42)')
    parser.add_argument('--log-grad-norms', action='store_true', help='Enable per-epoch gradient norm logging')
    parser.add_argument('--grad-top-k', type=int, default=3, help='Number of largest per-layer gradient norms to print')
    parser.add_argument('--log-act-grad', action='store_true', help='Enable per-epoch activation-layer output/gradient norm and zero-ratio logging')
    parser.add_argument('--top-k-layer', type=int, default=None, help='When --log-act-grad is set, record only the first K activation layers from input side')
    args = parser.parse_args()

    if args.top_k_layer is not None and not args.log_act_grad:
        print('Warning: --top-k-layer is set but --log-act-grad is not enabled. It will have no effect.')

    config_module = type('ConfigModule', (), {'config': get_experiment_config(args.experiment)})

    train(config_module, log_grad_norms=args.log_grad_norms, grad_top_k=args.grad_top_k,
          log_act_grad=args.log_act_grad, top_k_layer=args.top_k_layer)
