import os
import sys
import json
import numpy as np
import torch
from torchvision import transforms, datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.cnn import CNNFactory


MEAN = np.array([0.447, 0.440, 0.407])
STD = np.array([0.260, 0.257, 0.276])
CLASS_NAMES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

STYLE_CONTEXT = 'seaborn-v0_8-whitegrid'
FIG_DPI = 200
LEGEND_KWARGS = dict(frameon=True, fancybox=True, shadow=True)

COLORS = {
    'train': '#2b83ba',
    'val': '#d7191c',
    'group': ['#E9B44C', '#904C77', '#52B788', '#2A4D69', '#E76F51'],
    'raw_alpha': 0.12,
    'raw_lw': 0.6,
    'line_lw': 2.0,
}


# =============================
#  Shared helpers
# =============================

def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)


def _add_legend(ax, loc='best', fontsize=9, **kwargs):
    ax.legend(loc=loc, fontsize=fontsize, **LEGEND_KWARGS, **kwargs)


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


def moving_average(data, window=3):
    arr = np.asarray(data)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode='valid')
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(smoothed, (pad_left, pad_right), mode='edge')
    if len(padded) < len(arr):
        padded = np.pad(padded, (0, len(arr) - len(padded)), mode='edge')
    elif len(padded) > len(arr):
        padded = padded[:len(arr)]
    return padded


def _plot_smoothed(ax, epochs, values, color, label, window=3):
    raw = np.asarray(values, dtype=float)
    smoothed = moving_average(raw, window)
    ax.plot(epochs, raw, color=color, alpha=COLORS['raw_alpha'], lw=COLORS['raw_lw'])
    ax.plot(epochs, smoothed, color=color, lw=COLORS['line_lw'], label=label)


def _get_color_palette(n):
    group = COLORS['group']
    if n <= len(group):
        return group[:n]
    cmap = plt.cm.Set2 if n <= 8 else plt.cm.tab10
    extra = cmap(np.linspace(0, 1, n - len(group)))
    return group + list(extra)


def _add_bar_labels(ax, bars, fmt='{:.3f}'):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    fmt.format(h), ha='center', va='bottom', fontsize=7)


def _annotate_best(ax, best_epoch, best_val):
    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1e-8)
    text_y = max(best_val - y_range * 0.3, y_min + y_range * 0.02)
    ax.annotate(f'Best: {best_val:.4f} @ epoch {best_epoch + 1}',
                xy=(best_epoch + 1, best_val),
                xytext=(best_epoch + 1, text_y),
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
                fontsize=9, color='#2ca02c', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#2ca02c'))


def _comparison_paths(save_path):
    base, ext = os.path.splitext(save_path)
    ext = ext or '.png'
    return {
        'val_loss': f'{base}_val_loss{ext}',
        'val_acc': f'{base}_val_acc{ext}',
        'final_vs_best': f'{base}_final_vs_best{ext}',
        'test_acc': f'{base}_test_acc{ext}',
    }


# ============================
#  Data / model loading
# ============================

def denormalize(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def load_model(log_path, model_path, device):
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    config_dict = log_data['config']

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
    return model, config_dict


# ===========================
#  Training curves
# ===========================

def plot_training_curves(log_data, save_path, window=3):
    _ensure_dir(save_path)
    epochs = np.arange(1, len(log_data['train_loss']) + 1)

    with plt.style.context(STYLE_CONTEXT):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        _plot_smoothed(ax1, epochs, log_data['train_loss'], COLORS['train'], 'Train Loss', window)
        _plot_smoothed(ax1, epochs, log_data['val_loss'], COLORS['val'], 'Val Loss', window)
        _style_ax(ax1, 'Epoch', 'Loss', 'Training & Validation Loss')
        _add_legend(ax1)

        _plot_smoothed(ax2, epochs, log_data['train_acc'], COLORS['train'], 'Train Acc', window)
        _plot_smoothed(ax2, epochs, log_data['val_acc'], COLORS['val'], 'Val Acc', window)
        _style_ax(ax2, 'Epoch', 'Accuracy', 'Training & Validation Accuracy')

        best_epoch = np.argmax(log_data['val_acc'])
        _annotate_best(ax2, best_epoch, log_data['val_acc'][best_epoch])
        _add_legend(ax2)

        _save_fig(fig, save_path)


# ==========================
#  Model comparison
# ==========================

def _plot_comparison_metric(ax, all_logs, key, ylabel, title, colors, window):
    """Plot a single metric across all models (raw + smoothed)."""
    for (name, log_data), color in zip(all_logs.items(), colors):
        epochs = np.arange(1, len(log_data[key]) + 1)
        _plot_smoothed(ax, epochs, log_data[key], color, name, window)
    _style_ax(ax, 'Epoch', ylabel, title)
    _add_legend(ax, 'upper right', fontsize=9)


def plot_comparison(all_logs, save_path, window=3):
    _ensure_dir(save_path)
    paths = _comparison_paths(save_path)

    n_models = len(all_logs)
    colors = _get_color_palette(n_models)
    names = list(all_logs.keys())

    with plt.style.context(STYLE_CONTEXT):
        # --- val loss comparison ---
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        _plot_comparison_metric(ax1, all_logs, 'val_loss',
                                'Val Loss', 'Validation Loss Comparison',
                                colors, window)
        _save_fig(fig1, paths['val_loss'])

        # --- val accuracy comparison ---
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        _plot_comparison_metric(ax2, all_logs, 'val_acc',
                                'Val Accuracy', 'Validation Accuracy Comparison',
                                colors, window)
        _save_fig(fig2, paths['val_acc'])

        # --- final vs best bar chart ---
        fig3, ax3 = plt.subplots(figsize=(max(8, n_models * 1.6), 6))
        final_accs = [log['val_acc'][-1] for log in all_logs.values()]
        best_accs = [max(log['val_acc']) for log in all_logs.values()]
        x = np.arange(len(names))
        width = 0.32
        bars1 = ax3.bar(x - width / 2, final_accs, width, label='Final Acc',
                        color='#66c2a5', edgecolor='white', linewidth=0.5)
        bars2 = ax3.bar(x + width / 2, best_accs, width, label='Best Acc',
                        color='#fc8d62', edgecolor='white', linewidth=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        _style_ax(ax3, '', 'Accuracy', 'Final vs Best Validation Accuracy')
        _add_legend(ax3)
        _add_bar_labels(ax3, bars1)
        _add_bar_labels(ax3, bars2)
        ax3.set_ylim(bottom=0)
        _save_fig(fig3, paths['final_vs_best'])

        # --- test accuracy bar chart ---
        fig4, ax4 = plt.subplots(figsize=(max(8, n_models * 1.6), 6))
        test_accs = [log.get('test_metrics', {}).get('accuracy', 0)
                     for log in all_logs.values()]
        bars = ax4.bar(x, test_accs, color=colors, edgecolor='white', linewidth=0.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        _style_ax(ax4, '', 'Test Accuracy', 'Test Accuracy Comparison')
        _add_bar_labels(ax4, bars, fmt='{:.3f}')
        ax4.set_ylim(bottom=0)
        _save_fig(fig4, paths['test_acc'])


# ========================
#  Confusion matrix
# ========================

def plot_confusion_matrix(cm, model_name, save_path, normalize=True):
    if normalize:
        cm_norm = cm.astype(float)
        for i in range(cm_norm.shape[0]):
            row_sum = cm_norm[i].sum()
            if row_sum > 0:
                cm_norm[i] /= row_sum
        fmt, vmax = '.2f', 1.0
    else:
        cm_norm, fmt, vmax = cm, 'd', cm.max()

    with plt.style.context(STYLE_CONTEXT):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=vmax)

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                text = f'{cm_norm[i, j]:{fmt}}'
                text_color = 'white' if cm_norm[i, j] > vmax * 0.6 else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color=text_color)

        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(model_name, fontweight='bold', fontsize=11)

        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label('Fraction' if normalize else 'Count', fontsize=9)

        _ensure_dir(save_path)
        _save_fig(fig, save_path)


# =========================
#  t-SNE visualization
# =========================

def plot_tsne(model, dataloader, save_path, device='cpu', max_samples=1000):
    _ensure_dir(save_path)

    model.eval()
    features, labels = [], []

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
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(10), shrink=0.85, pad=0.02)
        cbar.set_label('Class', fontsize=10)
        _style_ax(ax, 't-SNE Component 1', 't-SNE Component 2',
                  't-SNE Feature Visualization')
        _save_fig(fig, save_path)
        
        
# ========================
#  PCA visualization
# ========================

def pca_heatmap(features, seed=42):
    C, H, W = features.shape
    feat_2d = features.reshape(C, -1).permute(1, 0).cpu().numpy()
    
    # 对所有通道特征做 PCA，提取最重要的前 3 个特征绘制热力图
    pca = PCA(n_components=3, random_state=seed)
    pca.fit(feat_2d)
    projected = pca.transform(feat_2d)

    for c in range(3):
        ch = projected[:, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            projected[:, c] = (ch - ch_min) / (ch_max - ch_min)
        else:
            projected[:, c] = 0.5

    heatmap = projected.reshape(H, W, 3)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    return heatmap_uint8, pca.explained_variance_ratio_


def _overlay_heatmap(orig, heatmap_pil, alpha=0.55):
    """Overlay heatmap on original image with semi-transparency.

    Optionally draws grid lines whose spacing reflects the scale ratio
    between heatmap resolution and original image size.
    """
    orig_pil = Image.fromarray(orig)
    orig_w, orig_h = orig_pil.size

    heatmap_resized = heatmap_pil.resize((orig_w, orig_h), Image.BILINEAR)

    orig_arr = np.array(orig_pil, dtype=float)
    heat_arr = np.array(heatmap_resized, dtype=float)
    blended = (orig_arr * (1 - alpha) + heat_arr * alpha).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(blended)

    return result


def _save_pca_sample(model_name, dataset, idx, cls_name, model, device, save_dir, seed):
    img, _ = dataset[idx]
    inp = img.unsqueeze(0).to(device)
    stage_features = model.extract_stage_features(inp)

    orig = denormalize(img)
    orig_pil = Image.fromarray(orig)

    n_stages = len(stage_features)
    heatmap_pils = []
    for feat in stage_features:
        feat_cpu = feat[0].cpu()
        heatmap_data, _ = pca_heatmap(feat_cpu, seed=seed)
        heatmap_pils.append(Image.fromarray(heatmap_data))

    # Row 1: original + raw heatmaps (resized to 96×96, no overlay)
    row1 = [orig_pil]
    for hp in heatmap_pils:
        row1.append(hp.resize((96, 96), Image.BILINEAR))

    # Row 2: original + heatmaps overlaid on original
    row2 = [orig_pil]
    for hp in heatmap_pils:
        row2.append(_overlay_heatmap(orig, hp))

    # Assemble 2-row combined strip
    col_count = n_stages + 1
    gap = 4
    total_w = 96 * col_count + gap * col_count
    total_h = 96 * 2 + gap
    combined = Image.new('RGB', (total_w, total_h))

    x = 0
    for pil_img in row1:
        combined.paste(pil_img, (x, 0))
        x += 96 + gap

    x = 0
    for pil_img in row2:
        combined.paste(pil_img, (x, 96 + gap))
        x += 96 + gap

    cls_save_dir = os.path.join(save_dir, cls_name)
    os.makedirs(cls_save_dir, exist_ok=True)
    combined.save(os.path.join(cls_save_dir, f'{model_name}_stages.png'))


def generate_pca_visualization(model_name, model, data_dir, save_dir, device,
                               n_samples_per_class=1, seed=42):
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
    classes = dataset.classes
    n_classes = len(classes)
    rng = np.random.RandomState(seed)

    with torch.no_grad():
        for cls_idx in range(n_classes):
            cls_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == cls_idx]
            chosen = rng.choice(cls_indices, size=n_samples_per_class, replace=False)
            for idx in chosen:
                _save_pca_sample(
                    model_name, dataset, idx, classes[cls_idx],
                    model, device, save_dir, seed
                )

    print(f'PCA visualization for {model_name} saved to {save_dir}')


# ========================
#  Gradient norms
# ========================

def plot_grad_norms(log_data, save_path, top_k=5, window=3):
    grad_logs = log_data.get('train_grad_norms', [])
    if not grad_logs:
        return

    _ensure_dir(save_path)

    epochs = np.arange(1, len(grad_logs) + 1)
    global_norms = np.array([e.get('global_l2', 0.0) for e in grad_logs], dtype=float)

    layer_keys = [k for k in grad_logs[0].keys() if k != 'global_l2']
    if not layer_keys:
        layer_keys = sorted(set().union(*[set(e.keys()) for e in grad_logs]))
        layer_keys = [k for k in layer_keys if k != 'global_l2']

    layer_norms = {
        name: np.array([e.get(name, 0.0) for e in grad_logs], dtype=float)
        for name in layer_keys
    }

    top_names = [
        name for _, name in sorted(
            [(v.mean(), name) for name, v in layer_norms.items() if v.size > 0],
            reverse=True,
        )
    ][:top_k]

    with plt.style.context(STYLE_CONTEXT):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        _plot_smoothed(axes[0], epochs, global_norms, '#1f77b4', 'Global L2 (epoch mean)', window)
        axes[0].set_yscale('log')
        _style_ax(axes[0], 'Epoch', 'L2 Gradient Norm', 'Gradient Global L2 Norm')
        _add_legend(axes[0])

        if top_names:
            layer_cmap = plt.cm.Set2(np.linspace(0, 1, min(len(top_names), 8)))
            for name, color in zip(top_names, layer_cmap):
                values = layer_norms[name]
                _plot_smoothed(axes[1], epochs, values, color, name, window)
            axes[1].set_yscale('log')
            _style_ax(axes[1], 'Epoch', 'L2 Gradient Norm',
                      f'Top {len(top_names)} Layer Gradient Norms')
            _add_legend(axes[1], fontsize=7)
        else:
            _style_ax(axes[1], '', '', '')
            axes[1].text(0.5, 0.5, 'No per-layer gradient norms found',
                         ha='center', va='center', transform=axes[1].transAxes,
                         fontsize=12, color='gray')

        _save_fig(fig, save_path)


# ================================
#  Activation-layer gradient data
# ================================

def plot_act_grad_curves(log_data, save_dir, model_name, window=3):
    """Plot activation-layer gradient statistics as three separate figures.

    Produces:
        {save_dir}/{model_name}_act_norm.png       -- activation output L2 norms
        {save_dir}/{model_name}_grad_norm.png      -- gradient L2 norms at activations
        {save_dir}/{model_name}_grad_zero_ratio.png -- gradient zero-element ratio

    Each figure contains one curve per tracked activation layer, ordered from
    input to output, with consistent colours across all three figures.
    """
    act_logs = log_data.get('train_act_grad', [])
    if not act_logs:
        return

    _ensure_dir(os.path.join(save_dir, '_placeholder'))

    epochs = np.arange(1, len(act_logs) + 1)

    layer_names = list(act_logs[0].keys())
    if not layer_names:
        return

    act_norms_by_layer = {}
    grad_norms_by_layer = {}
    zero_ratios_by_layer = {}
    for name in layer_names:
        act_norms_by_layer[name] = np.array(
            [e[name]['act_norm'] for e in act_logs], dtype=float
        )
        grad_norms_by_layer[name] = np.array(
            [e[name]['grad_norm'] for e in act_logs], dtype=float
        )
        zero_ratios_by_layer[name] = np.array(
            [e[name]['grad_zero_ratio'] for e in act_logs], dtype=float
        )

    colors = _get_color_palette(len(layer_names))
    layer_color_map = dict(zip(layer_names, colors))

    def _layer_label(name):
        idx = layer_names.index(name) + 1
        return f'L{idx}'

    # ------ Figure 1: activation output L2 norms ------
    with plt.style.context(STYLE_CONTEXT):
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        for name in layer_names:
            _plot_smoothed(ax1, epochs, act_norms_by_layer[name],
                           layer_color_map[name], _layer_label(name), window)
        ax1.set_yscale('log')
        _style_ax(ax1, 'Epoch', 'L2 Activation Norm',
                  'Activation Output Norms per Layer')
        _add_legend(ax1, fontsize=7)
        _save_fig(fig1, os.path.join(save_dir, f'{model_name}_act_norm.png'))

    # ------ Figure 2: gradient L2 norms at activations ------
    with plt.style.context(STYLE_CONTEXT):
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for name in layer_names:
            _plot_smoothed(ax2, epochs, grad_norms_by_layer[name],
                           layer_color_map[name], _layer_label(name), window)
        ax2.set_yscale('log')
        _style_ax(ax2, 'Epoch', 'L2 Gradient Norm',
                  'Gradient Norms at Activation Layers')
        _add_legend(ax2, fontsize=7)
        _save_fig(fig2, os.path.join(save_dir, f'{model_name}_grad_norm.png'))

    # ------ Figure 3: gradient zero-element ratio ------
    with plt.style.context(STYLE_CONTEXT):
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        for name in layer_names:
            _plot_smoothed(ax3, epochs, zero_ratios_by_layer[name],
                           layer_color_map[name], _layer_label(name), window)
        ax3.set_ylim(0, 1)
        _style_ax(ax3, 'Epoch', 'Zero Ratio',
                  'Gradient Zero-Element Ratio at Activation Layers')
        _add_legend(ax3, fontsize=7)
        _save_fig(fig3, os.path.join(save_dir, f'{model_name}_grad_zero_ratio.png'))
