from configs.experiments.common import (
    DEFAULT_SEEDS,
    build_experiment_name,
    config_from_baseline,
    resolve_baseline,
)
def build_experiments(baseline=None, seeds=DEFAULT_SEEDS):
    base = resolve_baseline('stage3', baseline)
    experiments = []
    for seed in seeds:
        for use_bn in (False, True):
            for dropout in (0.0, 0.5):
                for weight_decay in (0.0, 1e-4):
                    name = build_experiment_name('s3', seed, bn=use_bn, do=dropout, wd=weight_decay)
                    experiments.append(
                        config_from_baseline(
                            base,
                            name=name,
                            stage='stage3',
                            seed=seed,
                            use_bn=use_bn,
                            dropout=dropout,
                            weight_decay=weight_decay,
                            scheduler_name='none',
                            use_early_stopping=False,
                            notes='Stage 3 regularization full-factorial sweep.',
                        )
                    )
    return experiments
