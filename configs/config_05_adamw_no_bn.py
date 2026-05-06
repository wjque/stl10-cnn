from configs import Config

config = Config(
    name='05_adamw_no_bn',
    augmentation='flip_h',
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=False,
    dropout=0.0,
    optimizer_name='adamw',
    learning_rate=0.01,
)
