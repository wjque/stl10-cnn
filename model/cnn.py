import torch
import torch.nn as nn


class CNNFactory(nn.Module):
    def __init__(self, num_classes=10, depth='shallow',
                 activation='relu', pooling='max', use_bn=False, dropout=0.5):
        super().__init__()
        self.activation = activation

        if depth == 'shallow':
            channels = [32, 64, 128]
        elif depth == 'deep':
            channels = [32, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown depth: {depth}")

        blocks_per_stage = 1

        pool_fn = nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2)

        layers = []
        in_ch = 3

        for i, out_ch in enumerate(channels):
            for j in range(blocks_per_stage):
                c_in = in_ch if j == 0 else out_ch
                conv1 = nn.Conv2d(c_in, out_ch, kernel_size=3, padding=1, bias=not use_bn)
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
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def _init_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
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
