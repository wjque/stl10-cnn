import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', use_bn=False, stride=1):
        super().__init__()
        self.use_bn = use_bn
        act_fn = nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act1 = act_fn

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act2 = nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not use_bn),
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class CNNFactory(nn.Module):
    def __init__(self, num_classes=10, use_residual=False, depth='shallow',
                 activation='relu', pooling='max', use_bn=False, dropout=0.5):
        super().__init__()

        if depth == 'shallow':
            channels = [32, 64, 128]
        else:
            channels = [32, 64, 128, 256, 512]

        pool_fn = nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2)
        act_fn = nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid

        layers = []
        in_ch = 3

        for i, out_ch in enumerate(channels):
            if use_residual:
                stride = 2 if i == 0 else 1
                layers.append(ResidualBlock(in_ch, out_ch, activation, use_bn, stride=stride))
                layers.append(ResidualBlock(out_ch, out_ch, activation, use_bn, stride=1))
                layers.append(pool_fn)
            else:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
                if use_bn:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(act_fn() if isinstance(act_fn, type) else act_fn)
                layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
                if use_bn:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(act_fn() if isinstance(act_fn, type) else act_fn)
                layers.append(pool_fn)

            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        final_channels = channels[-1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)
