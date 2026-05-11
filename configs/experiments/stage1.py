from configs.experiments.common import (
    DEFAULT_SEEDS,
    build_experiment_name,
    config_from_baseline,
    resolve_baseline,
)
def build_experiments(baseline=None, seeds=DEFAULT_SEEDS):
    base = resolve_baseline('stage1', baseline)
    variants = [
        {'optimizer_name': 'sgd', 'learning_rate': 1e-2},
        {'optimizer_name': 'sgd', 'learning_rate': 1e-3},
        {'optimizer_name': 'adamw', 'learning_rate': 1e-3},
        {'optimizer_name': 'adamw', 'learning_rate': 1e-4},
    ]

    experiments = []
    for seed in seeds:
        for variant in variants:
            name = build_experiment_name('s1', seed, opt=variant['optimizer_name'], lr=variant['learning_rate'])
            experiments.append(
                config_from_baseline(
                    base,
                    name=name,
                    stage='stage1',
                    seed=seed,
                    augmentations=[],
                    depth='shallow',
                    pooling='max',
                    use_bn=False,
                    dropout=0.0,
                    weight_decay=0.0,
                    scheduler_name='none',
                    use_early_stopping=False,
                    notes='Stage 1 optimizer and learning-rate sweep.',
                    **variant,
                )
            )
    return experiments
