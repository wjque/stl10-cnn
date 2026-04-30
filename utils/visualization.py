import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


def moving_average(data, window=3):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='valid')
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(smoothed, (pad_left, pad_right), mode='edge')
    if len(padded) < len(data):
        padded = np.pad(padded, (0, len(data) - len(padded)), mode='edge')
    elif len(padded) > len(data):
        padded = padded[:len(data)]
    return padded


def plot_training_curves(log_data, save_path, window=3):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = np.arange(1, len(log_data['train_loss']) + 1)

    train_loss_raw = np.array(log_data['train_loss'])
    val_loss_raw = np.array(log_data['val_loss'])
    train_acc_raw = np.array(log_data['train_acc'])
    val_acc_raw = np.array(log_data['val_acc'])

    train_loss_smoothed = moving_average(train_loss_raw, window)
    val_loss_smoothed = moving_average(val_loss_raw, window)
    train_acc_smoothed = moving_average(train_acc_raw, window)
    val_acc_smoothed = moving_average(val_acc_raw, window)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss_raw, 'b-', alpha=0.2, linewidth=0.8)
    ax1.plot(epochs, train_loss_smoothed, 'b-', linewidth=1.8, label='Train Loss')
    ax1.plot(epochs, val_loss_raw, 'r-', alpha=0.2, linewidth=0.8)
    ax1.plot(epochs, val_loss_smoothed, 'r-', linewidth=1.8, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Validation Loss (smoothing window={window})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc_raw, 'b-', alpha=0.2, linewidth=0.8)
    ax2.plot(epochs, train_acc_smoothed, 'b-', linewidth=1.8, label='Train Acc')
    ax2.plot(epochs, val_acc_raw, 'r-', alpha=0.2, linewidth=0.8)
    ax2.plot(epochs, val_acc_smoothed, 'r-', linewidth=1.8, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Training and Validation Accuracy (smoothing window={window})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    best_epoch = np.argmax(val_acc_raw)
    best_acc = val_acc_raw[best_epoch]
    y_min, y_max = ax2.get_ylim()
    y_range = max(y_max - y_min, 1e-8)
    offset = y_range * 0.3
    text_y = best_acc - offset
    text_y = max(text_y, y_min + y_range * 0.02)
    ax2.annotate(f'Best: {best_acc:.4f} @ epoch {best_epoch + 1}',
                 xy=(best_epoch + 1, best_acc),
                 xytext=(best_epoch + 1, text_y),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(all_logs, save_path, window=3):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_logs)))

    for (name, log_data), color in zip(all_logs.items(), colors):
        epochs = np.arange(1, len(log_data['val_loss']) + 1)
        val_loss_raw = np.array(log_data['val_loss'])
        val_loss_smoothed = moving_average(val_loss_raw, window)
        val_acc_raw = np.array(log_data['val_acc'])
        val_acc_smoothed = moving_average(val_acc_raw, window)

        axes[0, 0].plot(epochs, val_loss_raw, color=color, alpha=0.15, linewidth=0.8)
        axes[0, 0].plot(epochs, val_loss_smoothed, color=color, label=name, linewidth=1.5)
        axes[0, 1].plot(epochs, val_acc_raw, color=color, alpha=0.15, linewidth=0.8)
        axes[0, 1].plot(epochs, val_acc_smoothed, color=color, label=name, linewidth=1.5)

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Val Loss')
    axes[0, 0].set_title(f'Validation Loss Comparison (smoothing window={window})')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Val Accuracy')
    axes[0, 1].set_title(f'Validation Accuracy Comparison (smoothing window={window})')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    names = list(all_logs.keys())
    final_accs = [log_data['val_acc'][-1] for log_data in all_logs.values()]
    best_accs = [max(log_data['val_acc']) for log_data in all_logs.values()]
    x = np.arange(len(names))
    width = 0.35
    axes[1, 0].bar(x - width / 2, final_accs, width, label='Final Acc', alpha=0.8)
    axes[1, 0].bar(x + width / 2, best_accs, width, label='Best Acc', alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Final vs Best Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    ax_bar = axes[1, 1]
    test_accs = [log_data.get('test_acc', 0) for log_data in all_logs.values()]
    bars = ax_bar.bar(x, test_accs, alpha=0.8, color=colors)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax_bar.set_ylabel('Test Accuracy')
    ax_bar.set_title('Test Accuracy Comparison')
    ax_bar.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, test_accs):
        if acc > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                       f'{acc:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_tsne(model, dataloader, save_path, device='cpu', max_samples=1000):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            feats = model.extract_features(inputs).cpu().numpy()
            features.append(feats)
            labels.append(targets.numpy())
            if len(np.concatenate(labels)) >= max_samples:
                break

    features = np.concatenate(features)[:max_samples]
    labels = np.concatenate(labels)[:max_samples]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=15)
    plt.colorbar(scatter, ticks=range(10))
    plt.title('t-SNE Feature Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
