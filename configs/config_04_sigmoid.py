from configs import Config

config = Config(
    name='04_sigmoid',
    augmentation='flip_h',
    depth='shallow',
    activation='sigmoid',
    pooling='max',
    use_bn=True,
    dropout=0.0,
    optimizer_name='adamw',
    learning_rate=0.01,
)
