from configs import Config

config = Config(
    name='08_batchnorm',
    augmentation='random_crop',
    use_residual=False,
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=True,
    optimizer_name='sgd',
    learning_rate=0.01,
)
