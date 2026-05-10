from pathlib import Path

from configs import Config
from configs.base import get_base_config
from configs.experiments import registry


DEFAULT_SEEDS = (42, 52, 62)


def resolve_baseline(stage_name, baseline=None):
    if baseline is None:
        return get_base_config(name=f'{stage_name}_base')
    if isinstance(baseline, Config):
        return baseline
    try:
        return registry.load_baseline_from_log(baseline)
    except FileNotFoundError:
        return registry.get_experiment_config(baseline)


def format_value(value):
    if isinstance(value, float):
        if value == 0.0:
            return '0'
        if value >= 1:
            return str(value).replace('.', 'p')
        scientific = f'{value:.0e}'.replace('+', '')
        mantissa, exponent = scientific.split('e')
        return f'{mantissa}e{int(exponent)}'
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, (list, tuple)):
        return '_'.join(format_value(item) for item in value) if value else 'none'
    return str(value)


def build_experiment_name(stage_prefix, seed, **parts):
    tokens = [stage_prefix]
    for key, value in parts.items():
        tokens.append(f'{key}{format_value(value)}')
    tokens.append(f'seed{seed}')
    return '_'.join(tokens)


def config_from_baseline(base_config, *, name, stage, seed, notes='', **updates):
    payload = base_config.__dict__.copy()
    payload.update(updates)
    payload.update({
        'name': name,
        'stage': stage,
        'seed': seed,
        'notes': notes,
        'model_save_path': f'outputs/models/{name}.pth',
        'log_save_path': f'outputs/logs/{name}.json',
    })
    return Config(**payload)


def ensure_output_dirs():
    Path('outputs/models').mkdir(parents=True, exist_ok=True)
    Path('outputs/logs').mkdir(parents=True, exist_ok=True)
