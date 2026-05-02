from configs import Config

config = Config(
    name='05_deep_no_bn',
    augmentation='flip_h',
    depth='deep',
    activation='relu',
    pooling='max',
    use_bn=False,
    dropout=0.0,
    optimizer_name='adamw',
    learning_rate=0.01,
)
