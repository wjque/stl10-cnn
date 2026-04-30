import torch


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
