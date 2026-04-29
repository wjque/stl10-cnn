from configs import Config

config = Config(
    name='07_avgpool',
    augmentation='random_crop',
    use_residual=False,
    depth='shallow',
    activation='relu',
    pooling='avg',
    use_bn=False,
    optimizer_name='sgd',
    learning_rate=0.01,
)
