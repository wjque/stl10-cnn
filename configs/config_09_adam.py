from configs import Config

config = Config(
    name='09_adamw',
    augmentation='random_crop',
    use_residual=False,
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=False,
    optimizer_name='adamw',
    learning_rate=0.001,
)
