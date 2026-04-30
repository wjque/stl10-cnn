import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', use_bn=False, stride=1):
        super().__init__()
        self.use_bn = use_bn
        self.activation = activation

        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_bn)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_bn)

        self._init_conv(conv1)
        self._init_conv(conv2)
        self.conv1 = conv1
        self.conv2 = conv2

        self.act1 = nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid()
        self.act2 = nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid()

        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not use_bn)
            self._init_conv(shortcut_conv)
            self.shortcut = nn.Sequential(
                shortcut_conv,
                nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def _init_conv(self, m):
        if self.activation == 'relu':
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

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
        self.activation = activation

        if depth == 'shallow':
            channels = [32, 64, 128]
        else:
            channels = [32, 64, 128, 256, 512]

        pool_fn = nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2)

        layers = []
        in_ch = 3

        for i, out_ch in enumerate(channels):
            if use_residual:
                stride = 2 if i == 0 else 1
                layers.append(ResidualBlock(in_ch, out_ch, activation, use_bn, stride=stride))
                layers.append(ResidualBlock(out_ch, out_ch, activation, use_bn, stride=1))
                layers.append(pool_fn)
            else:
                conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)
                conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)
                self._init_conv(conv1)
                self._init_conv(conv2)

                act1 = nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid()
                act2 = nn.ReLU(inplace=True) if activation == 'relu' else nn.Sigmoid()

                layers.append(conv1)
                if use_bn:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(act1)
                layers.append(conv2)
                if use_bn:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(act2)
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

        self._init_classifier()

    def _init_conv(self, m):
        if self.activation == 'relu':
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def _init_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)
