from configs import Config

config = Config(
    name='04_deep',
    augmentation='random_crop',
    depth='deep',
    activation='relu',
    pooling='max',
    use_bn=False,
    dropout=0.0,
    optimizer_name='sgd',
    learning_rate=0.01,
)
