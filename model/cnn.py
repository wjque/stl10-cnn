import torch
import torch.nn as nn


class CNNFactory(nn.Module):
    """Build a CNN with configurable depth, activation, pooling, and normalization.

    The network consists of: 
        a feature extractor (sequential conv blocks + pooling stages)
        a classifier (global average pool → FC 256 → FC 10)

    Args:
        num_classes: Number of output classes (10 for STL-10).
        depth: 'shallow' (3 stages) or 'deep' (5 stages).
        activation: 'relu' or 'sigmoid'.
        pooling: 'max' for MaxPool2d, 'avg' for AvgPool2d.
        use_bn: Whether to insert BatchNorm after each conv.
        dropout: Dropout rate applied before the final linear layer.
    """

    def __init__(self, num_classes=10, depth='shallow',
                 activation='relu', pooling='max', use_bn=False, dropout=0.0):
        super().__init__()
        self.activation = activation

        # 根据 depth 参数选择卷积通道配置：浅层网络 3 个 stage，深层网络 5 个 stage
        if depth == 'shallow':
            channels = [32, 64, 128]
        elif depth == 'deep':
            channels = [32, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown depth: {depth}")

        blocks_per_stage = 1  # 每个 stage 内包含 1 个双卷积块

        # 选择池化方式：最大池化 or 平均池化
        pool_fn = nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2)

        layers = []
        in_ch = 3  # STL-10 图像为 RGB 三通道

        # 逐 stage 构建卷积块
        # each stage = [Conv -> (BN) -> Act -> Conv -> (BN) -> Act] -> Pool
        for i, out_ch in enumerate(channels):
            for j in range(blocks_per_stage):
                c_in = in_ch if j == 0 else out_ch
                conv1 = nn.Conv2d(c_in, out_ch, kernel_size=3, padding=1, bias=not use_bn)
                conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)
                self._init_conv(conv1)
                self._init_conv(conv2)

                # 根据 activation 参数选择激活函数
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
            layers.append(pool_fn)  # 每个 stage 末尾做下采样

            in_ch = out_ch  # 下一 stage 的输入通道数等于上一 stage 的输出通道数

        # 特征提取器
        self.features = nn.Sequential(*layers)

        # 全局平均池化，将空间维度压缩为 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头，将通道特征映射为类别原始分数 (softmax 交给损失函数做)
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
        """Initialize convolution layer weights.

        Uses Kaiming normal for ReLU networks (preserves variance through ReLU)
        and Xavier normal for Sigmoid networks.
        """
        if self.activation == 'relu':
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def _init_classifier(self):
        """Initialize the classifier's fully-connected layers.

        Linear weights use Kaiming normal initialization; biases are set to zero.
        """
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass: features → global avg pool → classifier → logits.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """Extract the feature vector before the final classification layer.

        Useful for t-SNE visualization or downstream analysis.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            Feature tensor of shape (N, final_channels).
        """
        x = self.features(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def extract_stage_features(self, x):
        """Extract intermediate feature maps after each pooling stage.

        Args:
            x: Input tensor of shape (N, 3, H, W).

        Returns:
            list of tensors, each of shape (N, C_i, H_i, W_i), one per stage.
        """
        stage_outputs = []
        current = x
        for module in self.features:
            current = module(current)
            if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                stage_outputs.append(current)  # 记录每个下采样 stage 的输出
        return stage_outputs
