from configs import Config

config = Config(
    name='02_flip_h',
    augmentation='flip_h',
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=False,
    dropout=0.0,
    optimizer_name='sgd',
    learning_rate=0.01,
)
