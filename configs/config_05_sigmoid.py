from configs import Config

config = Config(
    name='05_sigmoid',
    augmentation='random_crop',
    depth='shallow',
    activation='sigmoid',
    pooling='max',
    use_bn=True,
    optimizer_name='sgd',
    learning_rate=0.01,
)
