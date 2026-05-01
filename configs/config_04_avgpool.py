from configs import Config

config = Config(
    name='04_avgpool',
    augmentation='flip_h',
    depth='shallow',
    activation='relu',
    pooling='avg',
    use_bn=True,
    dropout=0.0,
    optimizer_name='adamw',
    learning_rate=0.01,
)
