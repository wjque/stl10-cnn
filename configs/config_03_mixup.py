from configs import Config

config = Config(
    name='03_mixup',
    augmentation='mixup',
    use_residual=False,
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=False,
    optimizer_name='sgd',
    learning_rate=0.01,
    mixup_alpha=1.0,
)
