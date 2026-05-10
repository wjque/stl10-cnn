import importlib
import json
from pathlib import Path

from configs import Config


# Stage modules that can be imported through the registry.
AVAILABLE_STAGES = ('stage1', 'stage2', 'stage3', 'stage4')


def get_stage_module(stage):
    # Resolve a stage name to its experiment-definition module.
    if stage not in AVAILABLE_STAGES:
        raise ValueError(f'Unknown stage: {stage}')
    return importlib.import_module(f'configs.experiments.{stage}')


def list_stage_names(stage, baseline=None, seeds=None):
    # Convenience helper for callers that only need experiment names.
    return [config.name for config in build_stage_experiments(stage, baseline=baseline, seeds=seeds)]


def build_stage_experiments(stage, baseline=None, seeds=None):
    # Delegate experiment generation to the selected stage module.
    module = get_stage_module(stage)
    if seeds is None:
        seeds = module.DEFAULT_SEEDS
    return module.build_experiments(baseline=baseline, seeds=seeds)


def get_experiment_config(name, baseline=None, seeds=None):
    # Map a concrete experiment name back to its generated Config.
    stage_prefix_map = {
        's1_': 'stage1',
        's2_': 'stage2',
        's3_': 'stage3',
        's4_': 'stage4',
    }
    stage = next((value for prefix, value in stage_prefix_map.items() if name.startswith(prefix)), None)
    if stage is None:
        raise ValueError(f'Unknown experiment name: {name}')

    for config in build_stage_experiments(stage, baseline=baseline, seeds=seeds):
        if config.name == name:
            return config
    raise ValueError(f'Unknown experiment name: {name}')


def load_baseline_from_log(experiment_name, log_dir='outputs/logs'):
    # Rebuild a baseline Config from a saved training log.
    log_path = Path(log_dir) / f'{experiment_name}.json'
    if not log_path.exists():
        raise FileNotFoundError(f'Baseline log not found: {log_path}')

    with log_path.open('r') as f:
        log_data = json.load(f)

    config_dict = log_data['config']
    config_dict['name'] = experiment_name
    return Config(**config_dict)
