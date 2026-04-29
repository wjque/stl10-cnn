from configs import Config

config = Config(
    name='06_sigmoid',
    augmentation='random_crop',
    use_residual=False,
    depth='shallow',
    activation='sigmoid',
    pooling='max',
    use_bn=False,
    optimizer_name='sgd',
    learning_rate=0.01,
)
