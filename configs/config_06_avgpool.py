from configs import Config

config = Config(
    name='06_avgpool',
    augmentation='random_crop',
    depth='shallow',
    activation='relu',
    pooling='avg',
    use_bn=False,
    optimizer_name='sgd',
    learning_rate=0.01,
)
