import torch
import torch.nn as nn


def compute_grad_norms(model):
    """Return L2 norms of gradients for all parameters and a global norm."""
    norms = {}
    total_sq = 0.0

    for name, param in model.named_parameters():
        if param.grad is None:
            grad_norm = 0.0
        else:
            grad_norm = param.grad.detach().data.norm(2).item()

        norms[name] = grad_norm
        total_sq += grad_norm ** 2

    norms['global_l2'] = total_sq ** 0.5

    return norms


def accumulate_grad_norms(grad_sum, grad_norms):
    """Accumulate gradient norms over multiple iterations."""
    for name, value in grad_norms.items():
        grad_sum[name] = grad_sum.get(name, 0.0) + value
    return grad_sum


def average_grad_norms(grad_sum, step_count):
    """Average accumulated gradient norms by number of steps."""
    if step_count <= 0:
        return {}

    return {name: value / float(step_count) for name, value in grad_sum.items()}


def get_top_grad_norms(grad_norms, top_k=5):
    """Return top-k gradient norms excluding the global norm field."""
    filtered = [(name, value) for name, value in grad_norms.items() if name != 'global_l2']
    return sorted(filtered, key=lambda item: item[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
#  Activation-layer gradient & output-norm hooks
# ---------------------------------------------------------------------------

def find_activation_layers(model, top_k=None):
    """Find all activation modules (ReLU, Sigmoid) in input→output order.

    Args:
        model: nn.Module.
        top_k: if given, keep only the first ``top_k`` layers from the input
               side.  When it exceeds the actual count a warning is printed and
               all layers are kept.

    Returns:
        List of (name, module) tuples.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Sigmoid)):
            layers.append((name, module))

    if top_k is not None:
        if top_k > len(layers):
            print(f'Warning: --top-k-layer={top_k} but model has only '
                  f'{len(layers)} activation layers. Using all {len(layers)}.')
        elif top_k < len(layers):
            layers = layers[:top_k]

    return layers


def register_act_grad_hooks(act_layers):
    """Register forward hooks on activation modules that also capture gradients.

    Forward  → stores output L2 norm and attaches a tensor gradient hook.
    The gradient hook captures gradient L2 norm and zero-element ratio
    when backprop passes through the activation output tensor.

    Args:
        act_layers: list of (name, module) from ``find_activation_layers``.

    Returns:
        (handles, hook_data) where *handles* is a flat list of forward-hook
        handles and *hook_data* is a dict keyed by layer name.
    """
    handles = []
    hook_data = {}

    for name, module in act_layers:
        store = {
            'act_norm': 0.0,
            'grad_norm': 0.0,
            'grad_zero_ratio': 0.0,
        }
        hook_data[name] = store

        def _make_fwd(s):
            def hook(mod, inp, out):
                s['act_norm'] = out.detach().norm(2).item()

                def _grad_hook(grad):
                    g = grad.detach()
                    total = g.numel()
                    s['grad_norm'] = g.norm(2).item()
                    s['grad_zero_ratio'] = (g == 0).sum().item() / max(total, 1)

                out.register_hook(_grad_hook)
            return hook

        handles.append(module.register_forward_hook(_make_fwd(store)))

    return handles, hook_data


def remove_act_grad_hooks(handles):
    """Remove all activation-gradient hook handles."""
    for h in handles:
        h.remove()


def get_act_grad_stats(hook_data):
    """Snapshot the current per-layer stats accumulated by hooks in this batch.

    Returns a dict of the form::

        {layer_name: {'act_norm': ..., 'grad_norm': ..., 'grad_zero_ratio': ...}}
    """
    stats = {}
    for name, data in hook_data.items():
        stats[name] = {
            'act_norm': data['act_norm'],
            'grad_norm': data['grad_norm'],
            'grad_zero_ratio': data['grad_zero_ratio'],
        }
    return stats


def accumulate_act_grad_stats(accum, stats):
    """Sum per-layer activation-gradient stats across batches."""
    for name, layer_stats in stats.items():
        if name not in accum:
            accum[name] = {}
        for key, value in layer_stats.items():
            accum[name][key] = accum[name].get(key, 0.0) + value
    return accum


def average_act_grad_stats(accum, batch_count):
    """Average accumulated activation-gradient stats over *batch_count* steps."""
    if batch_count <= 0:
        return {}
    result = {}
    for name, layer_stats in accum.items():
        result[name] = {key: value / float(batch_count) for key, value in layer_stats.items()}
    return result
