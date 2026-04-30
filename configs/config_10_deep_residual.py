from configs import Config

config = Config(
    name='10_deep_residual',
    augmentation='random_crop',
    use_residual=True,
    depth='deep',
    activation='relu',
    pooling='max',
    use_bn=False,
    optimizer_name='sgd',
    learning_rate=0.01,
)
