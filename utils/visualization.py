import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


STYLE_CONTEXT = 'seaborn-v0_8-whitegrid'

COLORS = {
    'train': '#2b83ba',
    'val': '#d7191c',
    'raw_alpha': 0.12,
    'raw_lw': 0.6,
    'line_lw': 2.0,
}


def _style_ax(ax, xlabel='', ylabel='', title=''):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle='--')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight='bold')


def _derive_paths(save_path):
    base, ext = os.path.splitext(save_path)
    return {
        'val_loss': f'{base}_val_loss{ext or ".png"}',
        'val_acc': f'{base}_val_acc{ext or ".png"}',
        'final_vs_best': f'{base}_final_vs_best{ext or ".png"}',
        'test_acc': f'{base}_test_acc{ext or ".png"}',
    }


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

    with plt.style.context(STYLE_CONTEXT):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(epochs, train_loss_raw, color=COLORS['train'],
                 alpha=COLORS['raw_alpha'], linewidth=COLORS['raw_lw'])
        ax1.plot(epochs, train_loss_smoothed, color=COLORS['train'],
                 linewidth=COLORS['line_lw'], label='Train Loss')
        ax1.plot(epochs, val_loss_raw, color=COLORS['val'],
                 alpha=COLORS['raw_alpha'], linewidth=COLORS['raw_lw'])
        ax1.plot(epochs, val_loss_smoothed, color=COLORS['val'],
                 linewidth=COLORS['line_lw'], label='Val Loss')
        _style_ax(ax1, 'Epoch', 'Loss', f'Training & Validation Loss')
        ax1.legend(frameon=True, fancybox=True, shadow=True)

        ax2.plot(epochs, train_acc_raw, color=COLORS['train'],
                 alpha=COLORS['raw_alpha'], linewidth=COLORS['raw_lw'])
        ax2.plot(epochs, train_acc_smoothed, color=COLORS['train'],
                 linewidth=COLORS['line_lw'], label='Train Acc')
        ax2.plot(epochs, val_acc_raw, color=COLORS['val'],
                 alpha=COLORS['raw_alpha'], linewidth=COLORS['raw_lw'])
        ax2.plot(epochs, val_acc_smoothed, color=COLORS['val'],
                 linewidth=COLORS['line_lw'], label='Val Acc')
        _style_ax(ax2, 'Epoch', 'Accuracy', f'Training & Validation Accuracy')

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
                     arrowprops=dict(arrowstyle='->', color='#2ca02c',
                                     lw=1.5),
                     fontsize=9, color='#2ca02c', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.8, edgecolor='#2ca02c'))
        ax2.legend(frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()


def plot_comparison(all_logs, save_path, window=3):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    paths = _derive_paths(save_path)

    n_models = len(all_logs)
    cmap = plt.cm.Set2 if n_models <= 8 else plt.cm.tab10
    colors = cmap(np.linspace(0, 1, n_models))

    names = list(all_logs.keys())

    with plt.style.context(STYLE_CONTEXT):
        # --- Figure 1: Validation Loss ---
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        for (name, log_data), color in zip(all_logs.items(), colors):
            epochs = np.arange(1, len(log_data['val_loss']) + 1)
            val_loss_raw = np.array(log_data['val_loss'])
            val_loss_smoothed = moving_average(val_loss_raw, window)
            ax1.plot(epochs, val_loss_raw, color=color,
                     alpha=COLORS['raw_alpha'], linewidth=COLORS['raw_lw'])
            ax1.plot(epochs, val_loss_smoothed, color=color,
                     linewidth=COLORS['line_lw'], label=name)
        _style_ax(ax1, 'Epoch', 'Val Loss', f'Validation Loss Comparison')
        ax1.legend(loc='upper right', frameon=True, fancybox=True,
                   shadow=True, fontsize=9)
        fig1.tight_layout()
        fig1.savefig(paths['val_loss'], dpi=200, bbox_inches='tight')
        plt.close(fig1)

        # --- Figure 2: Validation Accuracy ---
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for (name, log_data), color in zip(all_logs.items(), colors):
            epochs = np.arange(1, len(log_data['val_acc']) + 1)
            val_acc_raw = np.array(log_data['val_acc'])
            val_acc_smoothed = moving_average(val_acc_raw, window)
            ax2.plot(epochs, val_acc_raw, color=color,
                     alpha=COLORS['raw_alpha'], linewidth=COLORS['raw_lw'])
            ax2.plot(epochs, val_acc_smoothed, color=color,
                     linewidth=COLORS['line_lw'], label=name)
        _style_ax(ax2, 'Epoch', 'Val Accuracy',
                   f'Validation Accuracy Comparison')
        ax2.legend(loc='lower right', frameon=True, fancybox=True,
                   shadow=True, fontsize=9)
        fig2.tight_layout()
        fig2.savefig(paths['val_acc'], dpi=200, bbox_inches='tight')
        plt.close(fig2)

        # --- Figure 3: Final vs Best Accuracy ---
        fig3, ax3 = plt.subplots(figsize=(max(8, n_models * 1.6), 6))
        final_accs = [log_data['val_acc'][-1] for log_data in all_logs.values()]
        best_accs = [max(log_data['val_acc']) for log_data in all_logs.values()]
        x = np.arange(len(names))
        width = 0.32
        bars1 = ax3.bar(x - width / 2, final_accs, width, label='Final Acc',
                         color='#66c2a5', edgecolor='white', linewidth=0.5)
        bars2 = ax3.bar(x + width / 2, best_accs, width, label='Best Acc',
                         color='#fc8d62', edgecolor='white', linewidth=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        _style_ax(ax3, '', 'Accuracy', 'Final vs Best Validation Accuracy')
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        for bar in bars1:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                     f'{h:.3f}', ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                     f'{h:.3f}', ha='center', va='bottom', fontsize=7)
        ax3.set_ylim(bottom=0)
        fig3.tight_layout()
        fig3.savefig(paths['final_vs_best'], dpi=200, bbox_inches='tight')
        plt.close(fig3)

        # --- Figure 4: Test Accuracy ---
        fig4, ax4 = plt.subplots(figsize=(max(8, n_models * 1.6), 6))
        test_accs = [log_data.get('test_metrics', {}).get('accuracy', 0)
                      for log_data in all_logs.values()]
        bars = ax4.bar(x, test_accs, color=colors,
                       edgecolor='white', linewidth=0.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        _style_ax(ax4, '', 'Test Accuracy', 'Test Accuracy Comparison')
        for bar, acc in zip(bars, test_accs):
            if acc > 0:
                ax4.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.003,
                         f'{acc:.3f}', ha='center', va='bottom', fontsize=8,
                         fontweight='bold')
        ax4.set_ylim(bottom=0)
        fig4.tight_layout()
        fig4.savefig(paths['test_acc'], dpi=200, bbox_inches='tight')
        plt.close(fig4)


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

    with plt.style.context(STYLE_CONTEXT):
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                             c=labels, cmap='tab10', alpha=0.55, s=18,
                             edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(10),
                            shrink=0.85, pad=0.02)
        cbar.set_label('Class', fontsize=10)
        _style_ax(ax, 't-SNE Component 1', 't-SNE Component 2',
                   't-SNE Feature Visualization')
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)


def plot_grad_norms(log_data, save_path, top_k=5, window=3):
    grad_logs = log_data.get('train_grad_norms', [])
    if not grad_logs:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = np.arange(1, len(grad_logs) + 1)
    global_norms = np.array([entry.get('global_l2', 0.0) for entry in grad_logs], dtype=float)
    smoothed_global = moving_average(global_norms, window)

    layer_keys = [k for k in grad_logs[0].keys() if k != 'global_l2']
    if not layer_keys:
        layer_keys = sorted(set().union(*[set(entry.keys()) for entry in grad_logs]))
        layer_keys = [k for k in layer_keys if k != 'global_l2']

    layer_norms = {
        name: np.array([entry.get(name, 0.0) for entry in grad_logs], dtype=float)
        for name in layer_keys
    }

    top_names = [
        name for _, name in sorted(
            [(values.mean(), name) for name, values in layer_norms.items() if values.size > 0],
            reverse=True,
        )
    ][:top_k]

    with plt.style.context(STYLE_CONTEXT):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(epochs, global_norms, color='#1f77b4',
                     alpha=COLORS['raw_alpha'], linewidth=COLORS['raw_lw'])
        axes[0].plot(epochs, smoothed_global, color='#1f77b4',
                     linewidth=COLORS['line_lw'], label='Global L2 (epoch mean)')
        axes[0].set_yscale('log')
        _style_ax(axes[0], 'Epoch', 'L2 Gradient Norm',
                   'Gradient Global L2 Norm')
        axes[0].legend(frameon=True, fancybox=True, shadow=True)

        if top_names:
            layer_cmap = plt.cm.Set2(np.linspace(0, 1, min(len(top_names), 8)))
            for name, color in zip(top_names, layer_cmap):
                values = layer_norms[name]
                smoothed_values = moving_average(values, window)
                axes[1].plot(epochs, smoothed_values, color=color,
                             linewidth=1.5, label=name)
            axes[1].set_yscale('log')
            _style_ax(axes[1], 'Epoch', 'L2 Gradient Norm',
                       f'Top {len(top_names)} Layer Gradient Norms')
            axes[1].legend(loc='best', frameon=True, fancybox=True,
                           shadow=True, fontsize=7)
        else:
            _style_ax(axes[1], '', '', '')
            axes[1].text(0.5, 0.5, 'No per-layer gradient norms found',
                         ha='center', va='center', transform=axes[1].transAxes,
                         fontsize=12, color='gray')

        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
