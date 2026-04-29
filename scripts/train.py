import os
import sys
import json
import copy
import argparse
import importlib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cnn import CNNFactory
from utils.split_validation import create_dataloaders
from utils.metrics import compute_metrics
from utils.visualization import plot_training_curves, plot_tsne


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        if config.augmentation == 'mixup':
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, config.mixup_alpha)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.augmentation == 'mixup':
            correct += (lam * predicted.eq(targets_a).float() +
                       (1 - lam) * predicted.eq(targets_b).float()).sum().item()
        else:
            correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_scores = []
    all_targets = []

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

    metrics = compute_metrics(all_targets, all_preds, all_scores)

    return loss, acc, metrics


def train(config_module):
    config = config_module.config
    set_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Config: {config.name}')

    train_loader, val_loader, test_loader, classes = create_dataloaders(config)
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')

    model = CNNFactory(
        num_classes=10,
        use_residual=config.use_residual,
        depth=config.depth,
        activation=config.activation,
        pooling=config.pooling,
        use_bn=config.use_bn,
        dropout=config.dropout,
    ).to(device)

    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()

    if config.optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                     momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {config.optimizer_name}')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    log_data = {
        'config': {k: v for k, v in config.__dict__.items()},
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, config.num_epochs + 1):
        print(f'\n--- Epoch {epoch}/{config.num_epochs} ---')

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, config)
        val_loss, val_acc, val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        log_data['train_loss'].append(train_loss)
        log_data['train_acc'].append(train_acc)
        log_data['val_loss'].append(val_loss)
        log_data['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}')
        print(f'Val   F1: {val_metrics["f1_macro"]:.4f} | AUC: {val_metrics.get("auc_ovr", 0):.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f'  -> New best model (val_acc={best_val_acc:.4f})')

    log_data['best_val_acc'] = best_val_acc
    log_data['val_metrics'] = val_metrics

    model.load_state_dict(best_model_state)

    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), config.model_save_path)
    print(f'\nModel saved to {config.model_save_path}')

    os.makedirs(os.path.dirname(config.log_save_path), exist_ok=True)
    with open(config.log_save_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f'Log saved to {config.log_save_path}')

    fig_save_path = f'outputs/figures/{config.name}_curves.png'
    plot_training_curves(log_data, fig_save_path)
    print(f'Training curves saved to {fig_save_path}')

    tsne_save_path = f'outputs/figures/{config.name}_tsne.png'
    plot_tsne(model, val_loader, tsne_save_path, device=device)
    print(f't-SNE visualization saved to {tsne_save_path}')

    return model, log_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model with given config')
    parser.add_argument('--config', type=str, required=True, help='Config module path (e.g., configs.config_01_baseline)')
    args = parser.parse_args()

    config_module = importlib.import_module(args.config)
    train(config_module)
