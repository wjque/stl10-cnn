from configs.experiments.common import (
    DEFAULT_SEEDS,
    build_experiment_name,
    config_from_baseline,
    resolve_baseline,
)
def build_experiments(baseline=None, seeds=DEFAULT_SEEDS):
    base = resolve_baseline('stage2', baseline)
    variants = [
        {'augmentations': [], 'tag': 'none'},
        {'augmentations': ['random_crop'], 'tag': 'crop'},
        {'augmentations': ['flip_h'], 'tag': 'flip'},
        {'augmentations': ['random_crop', 'flip_h'], 'tag': 'crop_flip'},
    ]

    experiments = []
    for seed in seeds:
        for variant in variants:
            name = build_experiment_name('s2', seed, aug=variant['tag'])
            experiments.append(
                config_from_baseline(
                    base,
                    name=name,
                    stage='stage2',
                    seed=seed,
                    augmentations=variant['augmentations'],
                    use_bn=False,
                    dropout=0.0,
                    weight_decay=0.0,
                    use_early_stopping=False,
                    notes='Stage 2 augmentation sweep.',
                )
            )
    return experiments
