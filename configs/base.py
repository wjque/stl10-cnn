from configs import Config


def get_base_config(name='base', seed=42):
    return Config(
        name=name,
        stage='base',
        seed=seed,
        num_epochs=120,
        batch_size=64,
        num_workers=4,
        input_size=96,
        augmentations=[],
        depth='shallow',
        pooling='max',
        use_bn=False,
        dropout=0.0,
        optimizer_name='sgd',
        learning_rate=1e-2,
        weight_decay=0.0,
        momentum=0.9,
        scheduler_name='cosine',
        scheduler_t_max=None,
        use_early_stopping=False,
        patience=20,
        notes='Base configuration for stage sweeps.',
    )
