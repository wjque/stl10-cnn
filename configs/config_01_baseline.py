from configs import Config

config = Config(
    name='01_baseline',
    augmentation='none',
    use_residual=False,
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=False,
    optimizer_name='sgd',
    learning_rate=0.01,
)
