from configs import Config

config = Config(
    name='09_sigmoid_adamw',
    augmentation='random_crop',
    depth='shallow',
    activation='sigmoid',
    pooling='max',
    use_bn=True,
    optimizer_name='adamw',
    learning_rate=0.01,
)
