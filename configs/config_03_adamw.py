from configs import Config

config = Config(
    name='03_adamw',
    augmentation='flip_h',
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=True,
    dropout=0.0,
    optimizer_name='adamw',
    learning_rate=0.01,
)
