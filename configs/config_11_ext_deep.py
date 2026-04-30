from configs import Config

config = Config(
    name='11_ext_deep',
    augmentation='random_crop',
    use_residual=False,
    depth='extrem_deep',
    activation='relu',
    pooling='max',
    use_bn=False,
    optimizer_name='sgd',
    learning_rate=0.01,
)
