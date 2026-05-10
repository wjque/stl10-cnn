from configs.experiments.common import (
    DEFAULT_SEEDS,
    build_experiment_name,
    config_from_baseline,
    resolve_baseline,
)


DEFAULT_BASELINE = 's3_bn0_do0_wd0_seed42'


def build_experiments(baseline=None, seeds=DEFAULT_SEEDS):
    base = resolve_baseline('stage4', baseline or DEFAULT_BASELINE)
    variants = [
        {'depth': 'shallow', 'pooling': 'max'},
        {'depth': 'shallow', 'pooling': 'avg'},
        {'depth': 'deep', 'pooling': 'max'},
        {'depth': 'deep', 'pooling': 'avg'},
    ]

    experiments = []
    for seed in seeds:
        for variant in variants:
            name = build_experiment_name('s4', seed, depth=variant['depth'], pool=variant['pooling'])
            experiments.append(
                config_from_baseline(
                    base,
                    name=name,
                    stage='stage4',
                    seed=seed,
                    scheduler_name='none',
                    use_early_stopping=False,
                    notes='Stage 4 architecture sweep.',
                    **variant,
                )
            )
    return experiments
