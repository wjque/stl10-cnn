from configs import Config

config = Config(
    name='04_deep',
    augmentation='flip_h',
    depth='deep',
    activation='relu',
    pooling='max',
    use_bn=True,
    dropout=0.0,
    optimizer_name='adamw',
    learning_rate=0.01,
)
