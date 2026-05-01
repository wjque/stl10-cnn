from configs import Config

config = Config(
    name='03_dropout',
    augmentation='flip_h',
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=True,
    dropout=0.5,
    optimizer_name='sgd',
    learning_rate=0.01,
)
