from configs import Config

config = Config(
    name='07_batchnorm',
    augmentation='random_crop',
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=True,
    optimizer_name='sgd',
    learning_rate=0.01,
)
