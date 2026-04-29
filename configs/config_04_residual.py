from configs import Config

config = Config(
    name='04_residual',
    augmentation='random_crop',
    use_residual=True,
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=False,
    optimizer_name='sgd',
    learning_rate=0.01,
)
