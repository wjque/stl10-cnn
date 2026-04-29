from configs import Config

config = Config(
    name='09_adam',
    augmentation='random_crop',
    use_residual=False,
    depth='shallow',
    activation='relu',
    pooling='max',
    use_bn=False,
    optimizer_name='adam',
    learning_rate=0.001,
)
