from configs import Config

config = Config(
    name='05_sigmoid_no_bn',
    augmentation='flip_h',
    depth='shallow',
    activation='sigmoid',
    pooling='max',
    use_bn=False,
    dropout=0.0,
    optimizer_name='sgd',
    learning_rate=0.01,
)
